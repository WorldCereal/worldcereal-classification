from calendar import monthrange
from datetime import datetime, timedelta
from typing import List, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.patches import Rectangle
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from presto.utils import device
from torch.utils.data import DataLoader

from worldcereal.parameters import CropLandParameters, CropTypeParameters
from worldcereal.seasons import get_season_dates_for_extent


class date_slider:
    """Class that provides a slider for selecting a processing period.
    The processing period is fixed in length, amounting to one year.
    The processing period will always start the first day of a month and end the last day of a month.
    """

    def __init__(self, start_date=datetime(2018, 1, 1), end_date=datetime(2024, 1, 1)):

        dates = pd.date_range(start_date, end_date, freq="MS")
        options = [(date.strftime("%b %Y"), date) for date in dates]
        self.interval_slider = widgets.SelectionRangeSlider(
            options=options,
            index=(0, 12),  # Default to a 12-month interval
            orientation="horizontal",
            continuous_update=False,
            readout=True,
            behaviour="drag",
            layout={"width": "75%", "height": "100px", "margin": "auto"},
            style={
                "handle_color": "dodgerblue",
            },
        )
        self.selected_range = [
            pd.to_datetime(start_date),
            pd.to_datetime(start_date) + pd.DateOffset(months=12) - timedelta(days=1),
        ]

    def on_slider_change(self, change):
        start, end = change["new"]
        # keep the interval fixed
        expected_end = start + pd.DateOffset(months=12)
        if end != expected_end:
            end = start + pd.DateOffset(months=12)
            self.interval_slider.value = (start, end)
        self.selected_range = (start, end - timedelta(days=1))

    def get_slider(self):
        self.interval_slider.observe(self.on_slider_change, names="value")
        return self.interval_slider

    def get_processing_period(self):

        start = self.selected_range[0].strftime("%Y-%m-%d")
        end = self.selected_range[1].strftime("%Y-%m-%d")
        print(f"Selected processing period: {start} to {end}")

        return TemporalContext(start, end)


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


def get_month_decimal(date):

    return date.timetuple().tm_mon + (
        date.timetuple().tm_mday / monthrange(2021, date.timetuple().tm_mon)[1]
    )


def retrieve_worldcereal_seasons(
    extent: BoundingBoxExtent, seasons: List[str] = ["s1", "s2"]
):
    """Method to retrieve default WorldCereal seasons from global crop calendars.
    These will be logged to the screen for informative purposes.

    Parameters
    ----------
    extent : BoundingBoxExtent
        extent for which to load seasonality
    seasons : List[str], optional
        seasons to load, by default s1 and s2
    """
    results = {}

    # prepare figure
    fig, ax = plt.subplots()
    plt.title("WorldCereal seasons")
    ax.set_ylim((0.4, len(seasons) + 0.5))
    ax.set_xlim((0, 13))
    ax.set_yticks(range(1, len(seasons) + 1))
    ax.set_yticklabels(seasons)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    facecolor = "darkgoldenrod"

    # Get the start and end date for each season
    for idx, season in enumerate(seasons):
        seasonal_extent = get_season_dates_for_extent(extent, 2021, f"tc-{season}")
        sos = pd.to_datetime(seasonal_extent.start_date)
        eos = pd.to_datetime(seasonal_extent.end_date)
        results[season] = (sos, eos)

        # get start and end month (decimals) for plotting
        start = get_month_decimal(sos)
        end = get_month_decimal(eos)

        # add rectangle to plot
        if start < end:
            ax.add_patch(
                Rectangle((start, idx + 0.75), end - start, 0.5, color=facecolor)
            )
        else:
            ax.add_patch(
                Rectangle((start, idx + 0.75), 12 - start, 0.5, color=facecolor)
            )
            ax.add_patch(Rectangle((1, idx + 0.75), end - 1, 0.5, color=facecolor))

        # add labels to plot
        label_start = sos.strftime("%B %d")
        label_end = eos.strftime("%B %d")
        plt.text(
            start - 0.2,
            idx + 0.65,
            label_start,
            fontsize=8,
            color="darkgreen",
            ha="left",
            va="center",
        )
        plt.text(
            end + 0.2,
            idx + 0.65,
            label_end,
            fontsize=8,
            color="darkred",
            ha="right",
            va="center",
        )

    # display plot
    plt.show()

    return results


def suggest_seasons(extent: BoundingBoxExtent, season: str = "tc-annual"):
    """Method to probe WorldCereal seasonality and suggest start, end and focus time.
    These will be logged to the screen for informative purposes

    Parameters
    ----------
    extent : BoundingBoxExtent
        extent for which to load seasonality
    season : str, optional
        season to load, by default "tc-annual"
    """
    seasonal_extent = get_season_dates_for_extent(extent, 2021, season)
    sos = pd.to_datetime(seasonal_extent.start_date)
    eos = pd.to_datetime(seasonal_extent.end_date)

    peak = sos + (eos - sos) / 2

    print(f"Start of `{season}` season: {sos.strftime('%B %d')}")
    print(f"End of `{season}` season: {eos.strftime('%B %d')}")
    print(f"Suggested focus time of `{season}` season: {peak.strftime('%B %d')}")


def get_inputs_outputs(
    df: pd.DataFrame, batch_size: int = 256, task_type: str = "croptype"
) -> Tuple[np.ndarray, np.ndarray]:
    from presto.dataset import WorldCerealLabelledDataset
    from presto.presto import Presto

    if task_type == "croptype":
        presto_model_url = CropTypeParameters().feature_parameters.presto_model_url
    if task_type == "cropland":
        presto_model_url = CropLandParameters().feature_parameters.presto_model_url
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
