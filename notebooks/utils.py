from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from presto.utils import device
from torch.utils.data import DataLoader


def pick_croptypes(df: pd.DataFrame, samples_threshold: int = 100):
    import ipywidgets as widgets

    potential_classes = df["croptype_name"].value_counts().reset_index()
    potential_classes = potential_classes[
        potential_classes["count"] > samples_threshold
    ]

    checkbox_widgets = [
        widgets.Checkbox(
            value=False, description=f"{row['croptype_name']} ({row['count']} samples)"
        )
        for ii, row in potential_classes.iterrows()
    ]
    vbox = widgets.VBox(
        checkbox_widgets,
        layout=widgets.Layout(width="50%", display="inline-flex", flex_flow="row wrap"),
    )

    return vbox, checkbox_widgets


def get_inputs_outputs(
    df: pd.DataFrame, batch_size: int = 256, task_type: str = "croptype"
) -> Tuple[np.ndarray, np.ndarray]:
    from presto.dataset import WorldCerealLabelledDataset
    from presto.presto import Presto

    if task_type == "croptype":
        presto_model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct_long-parquet_30D_CROPTYPE0_split%3Drandom_time-token%3Dmonth_balance%3DTrue_augment%3DTrue.pt"
    if task_type == "cropland":
        presto_model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ft-cl_30D_cropland_random.pt"
    logger.info(f"Presto URL: {presto_model_url}")
    presto_model = Presto.load_pretrained(presto_model_url, from_url=True, strict=False)

    tds = WorldCerealLabelledDataset(df, task_type=task_type)
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=False)

    encoding_list, targets = [], []

    logger.info("Computing Presto embeddings ...")
    for x, y, dw, latlons, month, valid_month, variable_mask in tdl:
        x_f, dw_f, latlons_f, month_f, valid_month_f, variable_mask_f = [
            t.to(device) for t in (x, dw, latlons, month, valid_month, variable_mask)
        ]
        input_d = {
            "x": x_f,
            "dynamic_world": dw_f.long(),
            "latlons": latlons_f,
            "mask": variable_mask_f,
            "month": month_f,
            "valid_month": valid_month_f,
        }

        presto_model.eval()
        encodings = presto_model.encoder(**input_d).detach().numpy()

        encoding_list.append(encodings)
        targets.append(y)

    encodings_np = np.concatenate(encoding_list)
    targets = np.concatenate(targets)

    logger.info("Done.")

    return encodings_np, targets


def get_custom_labels(df, checkbox_widgets):
    selected_crops = [
        checkbox.description.split(" ")[0]
        for checkbox in checkbox_widgets
        if checkbox.value
    ]
    df["downstream_class"] = "other"
    df.loc[df["croptype_name"].isin(selected_crops), "downstream_class"] = df[
        "croptype_name"
    ]

    return df


def train_classifier(inputs, targets):
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from presto.utils import DEFAULT_SEED
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    logger.info("Split train/test ...")
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs,
        targets,
        stratify=targets,
        test_size=0.3,
        random_state=DEFAULT_SEED,
    )

    if np.unique(targets_train).shape[0] > 2:
        eval_metric = "MultiClass"
        loss_function = "MultiClass"
    else:
        eval_metric = "F1"
        loss_function = "Logloss"

    logger.info("Computing class weights ...")
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(targets_train), y=targets_train
    )
    class_weights = {k: v for k, v in zip(np.unique(targets_train), class_weights)}
    logger.info("Class weights:", class_weights)

    sample_weights = np.ones((len(targets_train),))
    sample_weights_val = np.ones((len(targets_val),))
    for k, v in class_weights.items():
        sample_weights[targets_train == k] = v
        sample_weights_val[targets_val == k] = v

    custom_downstream_model = CatBoostClassifier(
        iterations=8000,
        depth=8,
        # learning_rate=0.05,
        early_stopping_rounds=50,
        # l2_leaf_reg=30,
        # colsample_bylevel=0.9,
        # l2_leaf_reg=3,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_state=DEFAULT_SEED,
        verbose=25,
        class_names=np.unique(targets_train),
    )

    logger.info("Training CatBoost classifier ...")
    custom_downstream_model.fit(
        inputs_train,
        targets_train,
        sample_weight=sample_weights,
        eval_set=Pool(inputs_val, targets_val, weight=sample_weights_val),
    )

    pred = custom_downstream_model.predict(inputs_val).flatten()

    report = classification_report(targets_val, pred)
    confuson_matrix = confusion_matrix(targets_val, pred)

    return custom_downstream_model, report, confuson_matrix
