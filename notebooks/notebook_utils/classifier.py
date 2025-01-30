from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from presto.utils import DEFAULT_SEED
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from worldcereal.parameters import CropLandParameters, CropTypeParameters


def get_input(label):
    while True:
        modelname = input(f"Enter a short name for your {label} (don't use spaces): ")
        if " " not in modelname:
            return modelname
        print("Invalid input. Please enter a name without spaces.")


LANDCOVER_LUT = {
    10: "Unspecified cropland",
    11: "Temporary crops",
    12: "Perennial crops",
    13: "Grassland",
    20: "Herbaceous vegetation",
    30: "Shrubland",
    40: "Deciduous forest",
    41: "Evergreen forest",
    42: "Mixed forest",
    50: "Bare or sparse vegetation",
    60: "Built-up",
    70: "Water",
    80: "Snow and ice",
    98: "No temporary crops nor perennial crops",
    99: "No temporary crops",
}


def select_landcover(df: pd.DataFrame):

    import ipywidgets as widgets

    df["LANDCOVER_LABEL"] = df["LANDCOVER_LABEL"].astype(int)
    df = df.loc[df["LANDCOVER_LABEL"] != 0]
    potential_classes = df["LANDCOVER_LABEL"].value_counts().reset_index()

    checkbox_widgets = [
        widgets.Checkbox(
            value=False,
            description=f"{LANDCOVER_LUT[row['LANDCOVER_LABEL']]} ({row['count']} samples)",
        )
        for ii, row in potential_classes.iterrows()
    ]
    vbox = widgets.VBox(
        checkbox_widgets,
        layout=widgets.Layout(width="50%", display="inline-flex", flex_flow="row wrap"),
    )

    return vbox, checkbox_widgets


def pick_croptypes(df: pd.DataFrame, samples_threshold: int = 100):
    import ipywidgets as widgets
    from IPython.display import Markdown

    nodata_helper_message = """
    ### What to do?
    1. **Increase the buffer size**: Try increasing the buffer size by passing the `buffer` parameter to the `query_public_extractions` function (to a reasonable extent).
    2. **Consult the WorldCereal Reference Data Module portal**: Assess data density in the selected region by visiting the [WorldCereal Reference Data Module portal](https://rdm.esa-worldcereal.org/map).
    3. **Pick another area**: Consult RDM portal (see above) to find areas with more data density.
    4. **Contribute data**: Collect some data and contribute to our global database! 🌍🌾 [Learn how to contribute here.](https://worldcereal.github.io/worldcereal-documentation/rdm/upload.html)
    """

    # CREATING A HIERARCHICAL LAYOUT OF OPTIONS
    # ==========================================
    _class_map = dict(df[["ewoc_code", "label_level3"]].value_counts().index.to_list())
    class_counts = pd.DataFrame(
        df[["label_level1", "label_level2", "label_level3"]].value_counts().sort_index()
    ).reset_index()
    class_counts["original_label_level3"] = class_counts["label_level3"]
    level2_class_counts = class_counts.groupby("label_level2")["label_level3"].nunique()

    for index, row in class_counts.iterrows():
        if (row["label_level3"] != row["label_level2"]) & (
            row["count"] < samples_threshold
        ):
            class_counts.loc[index, "label_level3"] = f"other_{row['label_level1']}"
            class_counts.loc[index, "label_level2"] = f"other_{row['label_level1']}"
            ewoc_codes_to_change = df[df["label_level3"] == row["label_level3"]][
                "ewoc_code"
            ].to_list()
            for ewoc_code in ewoc_codes_to_change:
                _class_map[ewoc_code] = f"other_{row['label_level1']}"
        elif (
            (row["label_level3"] == row["label_level2"])
            & (row["count"] < samples_threshold)
            & (level2_class_counts[row["label_level2"]] > 1)
        ):
            class_counts.loc[index, "label_level3"] = "ambiguous_class"
            ewoc_codes_to_change = df[df["label_level3"] == row["label_level3"]][
                "ewoc_code"
            ].to_list()
            for ewoc_code in ewoc_codes_to_change:
                _class_map[ewoc_code] = "ambiguous_class"
    class_counts = class_counts[class_counts["label_level3"] != "ambiguous_class"]
    class_counts = (
        class_counts.groupby(["label_level1", "label_level2", "label_level3"])
        .sum()
        .sort_index()
        .reset_index()
    )
    level2_class_counts = class_counts.groupby("label_level2")["label_level3"].nunique()

    for index, row in class_counts.iterrows():
        level_2_label = row["label_level2"]
        if (level2_class_counts[level_2_label] == 1) & (
            row["count"] < samples_threshold
        ):
            class_counts.loc[index, "label_level3"] = f"other_{row['label_level1']}"
            class_counts.loc[index, "label_level2"] = f"other_{row['label_level1']}"
            ewoc_codes_to_change = df[
                df["label_level3"] == row["original_label_level3"]
            ]["ewoc_code"].to_list()
            ewoc_codes_to_change.extend(
                df[df["label_level3"] == row["label_level3"]]["ewoc_code"].to_list()
            )
            for ewoc_code in np.unique(ewoc_codes_to_change):
                _class_map[ewoc_code] = f"other_{row['label_level1']}"
    class_counts = (
        class_counts.groupby(["label_level1", "label_level2", "label_level3"])
        .sum()
        .sort_index()
        .reset_index()
    )

    if len(class_counts) == 1:
        logger.error(
            f"Only one class remained after aggregation with threshold {samples_threshold}. Consider lowering the threshold."
        )
        Markdown(nodata_helper_message)
        raise ValueError(
            "Only one class remained after aggregation. Please lower the threshold."
        )

    # Convert to a hierarchically arranged dictionary
    class_counts = class_counts[class_counts["label_level3"] != "other_temporary_crops"]
    level2_class_counts = class_counts.groupby("label_level2")["label_level3"].nunique()
    hierarchical_dict = {}  # type: ignore
    for _, row in class_counts.iterrows():
        label_level1 = row["label_level2"]
        label_level2 = row["label_level3"]

        if label_level1 not in hierarchical_dict:
            hierarchical_dict[label_level1] = []
        if (label_level2 not in hierarchical_dict[label_level1]) & (
            level2_class_counts[label_level1] > 1
        ):
            hierarchical_dict[label_level1].append(label_level2)
    # ==========================================

    # CONSTRUCTING A WIDGET FOR SELECTING CROPTYPES
    # ==========================================
    options = hierarchical_dict

    # Create a description widget
    description_widget = widgets.HTML(
        value="""
        <div style='text-align: left;'>
            <div style='font-size: 16px; font-weight: bold;'>
                Croptype Picker
            </div>
            <div style='font-size: 14px; margin-top: 10px;'>
                Use the checkboxes below to select croptypes for analysis.<br>
                All classes that are not selected will be merged into <i>other_temporary_crops</i> class.<br>
                Note that depending on the <i>samples_threshold</i>, you will see different options.
            </div>
        </div>
        """,
        layout=widgets.Layout(margin="10px 0px 20px 0px"),
    )

    # Create checkboxes for categories and sub-options
    checkboxes = {}
    for category, sub_options in options.items():
        category_count = class_counts.loc[
            class_counts["label_level2"] == category, "count"
        ].sum()
        if sub_options:  # Only create sub-option checkboxes if sub_options is not empty
            category_checkbox = widgets.Checkbox(
                value=False,
                description=f"{category} ({category_count}) samples",
                layout=widgets.Layout(margin="0 0 0 0px", width="auto"),
            )
            sub_option_checkboxes = [
                widgets.Checkbox(
                    value=True,
                    description=f"{sub_option} ({class_counts.loc[class_counts['label_level3'] == sub_option, 'count'].iloc[0]}) samples",
                    layout=widgets.Layout(margin="0 0 0 30px", width="auto"),
                )
                for sub_option in sub_options
            ]
            checkboxes[category] = {
                "category_checkbox": category_checkbox,
                "sub_option_checkboxes": sub_option_checkboxes,
            }

            # Define the event handler for the category checkbox
            def on_category_checkbox_change(
                change, sub_option_checkboxes=sub_option_checkboxes
            ):
                for sub_option_checkbox in sub_option_checkboxes:
                    sub_option_checkbox.disabled = change["new"]
                    if change["new"]:
                        sub_option_checkbox.value = False

            # Attach the event handler to the category checkbox
            category_checkbox.observe(on_category_checkbox_change, names="value")
        else:
            category_checkbox = widgets.Checkbox(
                value=True,
                description=f"{category} ({category_count}) samples",
                layout=widgets.Layout(margin="0 0 0 0px", width="auto"),
            )
            checkboxes[category] = {
                "category_checkbox": category_checkbox,
                "sub_option_checkboxes": [],
            }

    # Function to get all selected values
    def get_selected_values(button=None):
        global selected_values
        selected_values = {}
        for category, widgets_dict in checkboxes.items():
            selected_values[category] = [
                sub_option_checkbox.description
                for sub_option_checkbox in widgets_dict["sub_option_checkboxes"]
                if sub_option_checkbox.value
            ]
        return selected_values

    # Create a VBox to group the checkboxes and the description widget
    vbox_items = [description_widget]
    for category, widgets_dict in checkboxes.items():
        vbox_items.append(widgets_dict["category_checkbox"])
        for sub_option_checkbox in widgets_dict["sub_option_checkboxes"]:
            vbox_items.append(sub_option_checkbox)
    vbox = widgets.VBox(vbox_items, layout=widgets.Layout(align_items="flex-start"))

    return vbox, vbox_items, _class_map


def prepare_training_dataframe(
    df: pd.DataFrame,
    batch_size: int = 256,
    task_type: str = "croptype",
    augment: bool = True,
    mask_ratio: float = 0.30,
    repeats: int = 1,
) -> pd.DataFrame:
    """Method to generate a training dataframe with Presto embeddings for downstream Catboost training.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe with required input features for Presto
    batch_size : int, optional
        by default 256
    task_type : str, optional
        cropland or croptype task, by default "croptype"
    augment : bool, optional
        if True, temporal jittering is enabled, by default True
    mask_ratio : float, optional
        if > 0, inputs are randomly masked before computing Presto embeddings, by default 0.30
    repeats: int, optional
        number of times to repeat each, by default 1

    Returns
    -------
    pd.DataFrame
        output training dataframe for downstream training

    Raises
    ------
    ValueError
        if an unknown tasktype is specified
    ValueError
        if repeats > 1 and augment=False and mask_ratio=0
    """
    from presto.presto import Presto

    from worldcereal.train.data import WorldCerealTrainingDataset, get_training_df

    if task_type == "croptype":
        presto_model_url = CropTypeParameters().feature_parameters.presto_model_url
        use_valid_date_token = (
            CropTypeParameters().feature_parameters.use_valid_date_token
        )
    elif task_type == "cropland":
        presto_model_url = CropLandParameters().feature_parameters.presto_model_url
        use_valid_date_token = (
            CropLandParameters().feature_parameters.use_valid_date_token
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    if repeats > 1 and not augment and mask_ratio == 0:
        raise ValueError("Repeats > 1 requires augment=True or mask_ratio > 0.")

    # Load pretrained Presto model
    logger.info(f"Presto URL: {presto_model_url}")
    presto_model = Presto.load_pretrained(
        presto_model_url,
        from_url=True,
        strict=False,
        valid_month_as_token=use_valid_date_token,
    )

    # Initialize dataset
    df = df.reset_index()
    ds = WorldCerealTrainingDataset(
        df,
        task_type=task_type,
        augment=True,
        mask_ratio=mask_ratio,
        repeats=repeats,
    )
    logger.info("Computing Presto embeddings ...")
    df = get_training_df(
        ds,
        presto_model,
        batch_size=batch_size,
        valid_date_as_token=use_valid_date_token,
    )

    logger.info("Done.")

    return df


def get_custom_croptype_labels(df, checkbox_widgets, class_map):

    selected_crops = [
        checkbox.description.split(" ")[0]
        for checkbox in checkbox_widgets
        if checkbox.value
    ]

    df["downstream_class"] = df["ewoc_code"].map(class_map)

    df = df[df["downstream_class"] != "ambiguous_class"]

    level2_classes = [xx for xx in selected_crops if xx in df["label_level2"].unique()]
    for crop in level2_classes:
        df.loc[df["label_level2"] == crop, "downstream_class"] = crop
    df.loc[~df["downstream_class"].isin(selected_crops), "downstream_class"] = (
        "other_temporary_crops"
    )

    return df


def get_custom_cropland_labels(df, checkbox_widgets, new_label="cropland"):

    # read selected classes from widget
    selected_lc = [
        checkbox.description.split(" (")[0]
        for checkbox in checkbox_widgets
        if checkbox.value
    ]
    # convert to landcover labels matching those in df
    selected_lc = [k for k, v in LANDCOVER_LUT.items() if v in selected_lc]
    # assign new labels
    df["downstream_class"] = "other"
    df.loc[df["LANDCOVER_LABEL"].isin(selected_lc), "downstream_class"] = new_label

    return df


def train_classifier(
    training_dataframe: pd.DataFrame,
    class_names: Optional[List[str]] = None,
    balance_classes: bool = False,
) -> Tuple[CatBoostClassifier, Union[str | dict], np.ndarray]:
    """Method to train a custom CatBoostClassifier on a training dataframe.

    Parameters
    ----------
    training_dataframe : pd.DataFrame
        training dataframe containing inputs and targets
    class_names : Optional[List[str]], optional
        class names to use, by default None
    balance_classes : bool, optional
        if True, class weights are used during training to balance the classes, by default False

    Returns
    -------
    Tuple[CatBoostClassifier, Union[str | dict], np.ndarray]
        The trained CatBoost model, the classification report, and the confusion matrix

    Raises
    ------
    ValueError
        When not enough classes are present in the training dataframe to train a model
    """

    logger.info("Split train/test ...")
    samples_train, samples_test = train_test_split(
        training_dataframe,
        test_size=0.2,
        random_state=DEFAULT_SEED,
        stratify=training_dataframe["downstream_class"],
    )

    # Define loss function and eval metric
    if np.unique(samples_train["downstream_class"]).shape[0] < 2:
        raise ValueError("Not enough classes to train a classifier.")
    elif np.unique(samples_train["downstream_class"]).shape[0] > 2:
        eval_metric = "MultiClass"
        loss_function = "MultiClass"
    else:
        eval_metric = "Logloss"
        loss_function = "Logloss"

    # Compute sample weights
    if balance_classes:
        logger.info("Computing class weights ...")
        class_weights = np.round(
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(samples_train["downstream_class"]),
                y=samples_train["downstream_class"],
            ),
            3,
        )
        class_weights = {
            k: v
            for k, v in zip(np.unique(samples_train["downstream_class"]), class_weights)
        }
        logger.info(f"Class weights: {class_weights}")

        sample_weights = np.ones((len(samples_train["downstream_class"]),))
        sample_weights_val = np.ones((len(samples_test["downstream_class"]),))
        for k, v in class_weights.items():
            sample_weights[samples_train["downstream_class"] == k] = v
            sample_weights_val[samples_test["downstream_class"] == k] = v
        samples_train["weight"] = sample_weights
        samples_test["weight"] = sample_weights_val
    else:
        samples_train["weight"] = 1
        samples_test["weight"] = 1

    # Define classifier
    custom_downstream_model = CatBoostClassifier(
        iterations=2000,  # Not too high to avoid too large model size
        depth=8,
        early_stopping_rounds=20,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_state=DEFAULT_SEED,
        verbose=25,
        class_names=(
            class_names
            if class_names is not None
            else np.unique(samples_train["downstream_class"])
        ),
    )

    # Setup dataset Pool
    bands = [f"presto_ft_{i}" for i in range(128)]
    calibration_data = Pool(
        data=samples_train[bands],
        label=samples_train["downstream_class"],
        weight=samples_train["weight"],
    )
    eval_data = Pool(
        data=samples_test[bands],
        label=samples_test["downstream_class"],
        weight=samples_test["weight"],
    )

    # Train classifier
    logger.info("Training CatBoost classifier ...")
    custom_downstream_model.fit(
        calibration_data,
        eval_set=eval_data,
    )

    # Make predictions
    pred = custom_downstream_model.predict(samples_test[bands]).flatten()

    report = classification_report(samples_test["downstream_class"], pred)
    confuson_matrix = confusion_matrix(samples_test["downstream_class"], pred)

    return custom_downstream_model, report, confuson_matrix


def train_cropland_classifier(training_dataframe: pd.DataFrame):
    return train_classifier(training_dataframe, class_names=["other", "cropland"])
