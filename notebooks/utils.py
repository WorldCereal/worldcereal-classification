import ast
import copy
import random
import warnings
from calendar import monthrange
from datetime import datetime, timedelta
from typing import List, Tuple

import ipywidgets as widgets
import leafmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from IPython.display import display
from loguru import logger
from matplotlib.patches import Rectangle
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from presto.utils import device
from pyproj import Transformer
from torch.utils.data import DataLoader

from worldcereal.parameters import CropLandParameters, CropTypeParameters
from worldcereal.seasons import get_season_dates_for_extent


class date_slider:
    """Class that provides a slider for selecting a processing period.
    The processing period is fixed in length, amounting to one year.
    The processing period will always start the first day of a month and end the last day of a month.
    """

    def __init__(self, start_date=datetime(2018, 1, 1), end_date=datetime(2024, 1, 1)):

        self.start_date = start_date
        self.end_date = end_date

        dates = pd.date_range(start_date, end_date, freq="MS")
        options = [(date.strftime("%d %b %Y"), date) for date in dates]
        self.interval_slider = widgets.SelectionRangeSlider(
            options=options,
            index=(0, 12),  # Default to a 12-month interval
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
        expected_end = start + pd.DateOffset(months=12)
        if end != expected_end:
            end = start + pd.DateOffset(months=12)
            self.interval_slider.value = (start, end)
        self.selected_range = (start, end - timedelta(days=1))

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
            self.start_date, self.end_date + pd.DateOffset(months=3), freq="4ME"
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
    "confidence": 255,
}


COLORMAP = {
    "cropland": {
        0: (186, 186, 186, 0),  # no cropland
        1: (224, 24, 28, 200),  # cropland
    },
    "confidence": get_probability_cmap(),
}


def _get_nodata(product_type):
    return NODATAVALUE[product_type]


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def generate_distinct_colors(n, min_distance=100):
    colors = []
    while len(colors) < n:
        new_color = generate_random_color()
        if all(color_distance(new_color, c) > min_distance for c in colors):
            colors.append(new_color)
    return colors


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
        else:
            colormap = None

    return colormap


def prepare_visualization(results, processing_period):

    final_paths = {}

    for product, product_params in results.products.items():

        paths = {}

        basepath = product_params["path"]
        if basepath is None:
            logger.warning("No products downloaded. Aborting!")
            return None
        product_type = product_params["type"].value
        if product_type == "cropland":
            lut = {"other": 0, "cropland": 1}
        else:
            lut = product_params["lut"]

        # get properties and data from input file
        with rasterio.open(basepath, "r") as src:
            labels = src.read(1).astype(np.uint8)
            probs = src.read(2).astype(np.uint8)
            meta = src.meta

        nodata = _get_nodata(product_type)
        colormap = _get_colormap(product_type, lut)

        # construct dictionary of output files to be generated
        outfiles = {
            "classification": {
                "data": labels,
                "colormap": colormap,
                "nodata": nodata,
                "lut": lut,
            },
            "confidence": {
                "data": probs,
                "colormap": _get_colormap("confidence"),
                "nodata": _get_nodata("confidence"),
                "lut": None,
            },
        }

        # write output files
        meta.update(count=1)
        for label, settings in outfiles.items():
            # construct final output path
            start = processing_period.start_date.replace("-", "")
            end = processing_period.end_date.replace("-", "")
            filename = f"{product}_{label}_{start}_{end}.tif"
            outpath = basepath.parent / filename
            bandnames = [label]
            with warnings.catch_warnings():  # ignore skimage warnings
                warnings.simplefilter("ignore")
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
