import ast
import copy
import logging
import random
from calendar import monthrange
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

import ipywidgets as widgets
import leafmap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from catboost import CatBoostClassifier, Pool
from IPython.display import display
from loguru import logger
from matplotlib.patches import Rectangle
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from presto.utils import DEFAULT_SEED
from pyproj import Transformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from worldcereal.parameters import CropLandParameters, CropTypeParameters
from worldcereal.seasons import get_season_dates_for_extent

logging.getLogger("rasterio").setLevel(logging.ERROR)


class date_slider:
    """Class that provides a slider for selecting a processing period.
    The processing period is fixed in length, amounting to one year.
    The processing period will always start the first day of a month and end the last day of a month.
    """

    def __init__(self, start_date=datetime(2018, 1, 1), end_date=datetime(2023, 12, 1)):

        self.start_date = start_date
        self.end_date = end_date

        dates = pd.date_range(start_date, end_date, freq="MS")
        options = [(date.strftime("%b %Y"), date) for date in dates]
        self.interval_slider = widgets.SelectionRangeSlider(
            options=options,
            index=(0, 11),  # Default to a 11-month interval
            orientation="horizontal",
            continuous_update=False,
            readout=True,
            behaviour="drag",
            layout={"width": "80%", "height": "100px", "margin": "0 auto 0 auto"},
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
        expected_end = start + pd.DateOffset(months=11)
        if end != expected_end:
            end = start + pd.DateOffset(months=11)
            self.interval_slider.value = (start, end)
        self.selected_range = (start, end + pd.DateOffset(months=1) - timedelta(days=1))

    def show_slider(self):
        self.interval_slider.observe(self.on_slider_change, names="value")

        # Add description widget
        descr_widget = widgets.HTML(
            value="""
            <div style='text-align: center;'>
                <div style='font-size: 20px; font-weight: bold;'>
                    Position the slider to select your processing period:
                </div>
            </div>
            """
        )

        # Generate a list of dates for the ticks every 3 months
        tick_dates = pd.date_range(
            self.start_date, self.end_date + pd.DateOffset(months=3), freq="5ME"
        )

        # Create a list of tick labels in the format "Aug 2023"
        tick_labels = [date.strftime("%b<br>%Y") for date in tick_dates]

        # Calculate the positions of the ticks to align them with the slider
        total_days = (self.end_date - self.start_date).days
        tick_positions = [
            ((date - self.start_date).days / total_days * 100) for date in tick_dates
        ]

        # Create a text widget to display the tick labels with calculated positions
        tick_widget_html = "<div style='text-align: center; font-size: 12px; position: relative; width: 86%; height: 20px; margin-top: -20px;'>"
        for label, position in zip(tick_labels, tick_positions):
            tick_widget_html += f"<div style='position: absolute; left: {position}%; transform: translateX(-50%);'>{label}</div>"
        tick_widget_html += "</div>"

        tick_widget = widgets.HTML(
            value=tick_widget_html, layout={"width": "80%", "margin": "0 auto 0 auto"}
        )

        # Arrange the text widget, interval slider, and tick widget using VBox
        vbox_with_ticks = widgets.VBox(
            [descr_widget, self.interval_slider, tick_widget],
            layout={"height": "200px"},
        )

        display(vbox_with_ticks)

    def get_processing_period(self):

        start = self.selected_range[0].strftime("%Y-%m-%d")
        end = self.selected_range[1].strftime("%Y-%m-%d")
        logger.info(f"Selected processing period: {start} to {end}")

        return TemporalContext(start, end)


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
    2. **Consult the WorldCereal Reference Data Module portal**: Assess data density in the selected region by visiting the [WorldCereal Reference Data Module portal](https://ewoc-rdm-ui.iiasa.ac.at/map).
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

    # get lat, lon centroid of extent
    transformer = Transformer.from_crs(
        f"EPSG:{extent.epsg}", "EPSG:4326", always_xy=True
    )
    minx, miny = transformer.transform(extent.west, extent.south)
    maxx, maxy = transformer.transform(extent.east, extent.north)
    lat = (maxy + miny) / 2
    lon = (maxx + minx) / 2
    location = f"lat={lat:.2f}, lon={lon:.2f}"

    # prepare figure
    fig, ax = plt.subplots()
    plt.title(f"WorldCereal seasons ({location})")
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
        iterations=8000,
        depth=8,
        early_stopping_rounds=50,
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


############# PRODUCT POSTPROCESSING #############


def get_probability_cmap():
    colormap = plt.get_cmap("RdYlGn")
    cmap = {}
    for i in range(101):
        cmap[i] = tuple((np.array(colormap(int(2.55 * i))) * 255).astype(int))
    return cmap


NODATAVALUE = {
    "cropland": 255,
    "croptype": 255,
    "probability": 255,
}


COLORMAP = {
    "cropland": {
        0: (186, 186, 186, 0),  # no cropland
        1: (224, 24, 28, 200),  # cropland
    },
    "probability": get_probability_cmap(),
}


def _get_nodata(product_type):
    return NODATAVALUE[product_type]


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(1, 255))


def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def generate_distinct_colors(n, min_distance=100):
    colors = [(186, 186, 186)]  # grey is reserved for no-cropland
    while len(colors) < n:
        new_color = generate_random_color()
        if all(color_distance(new_color, c) > min_distance for c in colors):
            colors.append(new_color)
    return colors[1:]


def _get_colormap(product, lut=None):

    if product in COLORMAP.keys():
        colormap = copy.deepcopy(COLORMAP[product])
    else:
        if lut is not None:
            logger.info((f"Assigning random color map for product {product}. "))
            colormap = {}
            distinct_colors = generate_distinct_colors(len(lut))
            for i, color in enumerate(distinct_colors):
                colormap[i] = (*color, 255)
            if product == "croptype":
                colormap[254] = (186, 186, 186, 255)  # add no cropland color
        else:
            colormap = None

    return colormap


def prepare_visualization(results):

    final_paths = {}
    colormaps = {}

    for product, product_params in results.products.items():

        paths = {}

        # Get product parameters
        basepath = product_params["path"]
        if basepath is None:
            logger.warning("No products downloaded. Aborting!")
            return None
        product_type = product_params["type"].value
        temporal_extent = product_params["temporal_extent"]
        lut = product_params["lut"]

        # Adjust LUT for crop type product
        if product_type == "croptype":
            # add no cropland class
            lut["no_cropland"] = 254

        # get properties and data from input file
        with rasterio.open(basepath, "r") as src:
            labels = src.read(1).astype(np.uint8)
            probs = src.read(2).astype(np.uint8)
            meta = src.meta

        nodata = _get_nodata(product_type)
        if product_type not in colormaps:
            colormaps[product_type] = _get_colormap(product_type, lut)

        # construct dictionary of output files to be generated
        outfiles = {
            "classification": {
                "data": labels,
                "colormap": colormaps[product_type],
                "nodata": nodata,
                "lut": lut,
            },
            "probability": {
                "data": probs,
                "colormap": _get_colormap("probability"),
                "nodata": _get_nodata("probability"),
                "lut": None,
            },
        }

        # write output files
        meta.update(count=1)
        meta.update(dtype=rasterio.uint8)
        for label, settings in outfiles.items():
            # construct final output path
            filename = f"{product}_{label}_{temporal_extent.start_date}_{temporal_extent.end_date}.tif"
            outpath = basepath.parent / filename
            bandnames = [label]
            meta.update(nodata=settings["nodata"])
            with rasterio.open(outpath, "w", **meta) as dst:
                dst.write(settings["data"], indexes=1)
                dst.nodata = settings["nodata"]
                for i, b in enumerate(bandnames):
                    dst.update_tags(i + 1, band_name=b)
                    if settings["lut"] is not None:
                        dst.update_tags(i + 1, lut=settings["lut"])
                if settings["colormap"] is not None:
                    dst.write_colormap(1, settings["colormap"])
            paths[label] = outpath

        final_paths[product] = paths

    return final_paths


############# PRODUCT VISUALIZATION #############


def visualize_classification(rasters, product):
    """Function to visualize a classification raster using matplotlib.

    Parameters
    ----------
    rasters : Dict[str, Dict[str, Path]]
        Dictionary containing all generated rasters.
        Output of function prepare_visualization.
    product : str
        Name of the product you wish to visualize.
        e.g. "cropland"
    """

    filepath = rasters[product]["classification"]

    # Helper function to scale RGB values
    def scale_rgb(color):
        # Scaling only RGB, ignoring alpha
        return tuple(c / 255 for c in color[:3])

    with rasterio.open(filepath, "r") as src:
        arr_classif = src.read().squeeze()
        colormap = src.colormap(1)
        lut = ast.literal_eval(src.tags(1)["lut"])

    # Filter colormap based on LUT (lookup table)
    colormap = {k: v for k, v in colormap.items() if k in lut.values()}

    # Apply RGB scaling
    colormap = {key: scale_rgb(value) for key, value in colormap.items()}

    # Create a custom ListedColormap
    cmap = mpl.colors.ListedColormap([colormap[key] for key in sorted(colormap.keys())])

    fig, ax = plt.subplots()

    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.2, 0.03, 0.5])

    # Get class labels and set colorbar boundaries
    classlabels = list(lut.keys())
    bounds = list(np.unique(arr_classif))  # Class boundaries

    # Define a norm for the colormap
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # plot the raster
    ax.imshow(arr_classif, cmap=cmap, norm=norm)

    # Create a colorbar with class labels
    bounds_colorbar = np.linspace(0, len(classlabels), len(classlabels) + 1)
    norm_colorbar = mpl.colors.BoundaryNorm(bounds_colorbar, cmap.N)
    cb = mpl.colorbar.ColorbarBase(
        ax2,
        cmap=cmap,
        norm=norm_colorbar,
        spacing="proportional",
        boundaries=bounds_colorbar,
        # Middle of each class
        ticks=np.arange(len(classlabels)) + 0.5,
        format="%1i",
    )

    # Set the colorbar ticks and labels
    cb.set_ticks(np.arange(len(classlabels)) + 0.5)
    cb.set_ticklabels(classlabels)

    # Turn off axis
    ax.axis("off")

    # Display the plot
    plt.show()


def visualize_products(rasters, port):
    """
    Function to visualize raster layers using leafmap.
    Only the first band of the input rasters is visualized.

    Parameters
    ----------
    rasters : Dict[str, Dict[str, Path]]
        Dictionary containing all generated rasters.
        Output of function prepare_visualization.
    port : int
        port to use for localtileserver application
        (in case you are working on a remote server, make sure the
        port is forwarded to your local machine)

    Returns:
        leafmap Map instance
    """

    m = leafmap.Map()
    m.add_basemap("Esri.WorldImagery")
    for product, items in rasters.items():
        for label, path in items.items():
            m.add_raster(
                str(path), indexes=[1], layer_name=f"{product}-{label}", port=port
            )

    return m


def show_color_legend(rasters, product):
    """Display the color legend of a product based on its colormap and LUT.
    The latter should be present as metadata in the .tif file.

    Parameters
    ----------
    rasters : Dict[str, Dict[str, Path]]
        Dictionary containing all generated rasters.
        Output of function prepare_visualization.
    product : str
        The product for which to display the color legend.
        Needs to be a key in the rasters dictionary.
    """
    import math

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    if product not in rasters:
        raise ValueError(f"Product {product} not found in rasters.")

    tif_file = rasters[product]["classification"]
    with rasterio.open(tif_file) as src:
        nodata = src.nodata
        colormap = src.colormap(1)
        lut = ast.literal_eval(src.tags(1)["lut"])

    # get rid of all (0, 0, 0, 255) items
    colormap = {k: v for k, v in colormap.items() if v != (0, 0, 0, 255)}

    # apply scaling of RGB values
    for key, value in colormap.items():
        # apply scaling of RGB values
        rgb = [c / 255 for c in value]
        colormap[key] = tuple(rgb)

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    ncols = 1

    raster_values = list(colormap)

    nrows = math.ceil(len(raster_values) / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, raster_value in enumerate(raster_values):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        # Get the name of the class
        if raster_value == nodata:
            name = "No data"
        else:
            name = [k for k, v in lut.items() if v == raster_value][0]

        ax.text(
            text_pos_x,
            y,
            name,
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )

        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=colormap[raster_value],
                edgecolor="0.7",
            )
        )
