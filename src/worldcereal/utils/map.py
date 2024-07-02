from typing import List
import copy
import geopandas as gpd
import leafmap
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from ipyleaflet import DrawControl, LayersControl, Map, SearchControl, basemaps
from loguru import logger
from openeo_gfmap import BoundingBoxExtent
from shapely import geometry
from shapely.geometry import Polygon, shape


def get_probability_cmap():
    colormap = plt.get_cmap("RdYlGn")
    cmap = {}
    for i in range(101):
        cmap[i] = tuple((np.array(colormap(int(2.55 * i))) * 255).astype(int))
    return cmap


COLORMAP = {
    "active-cropland": {
        0: (232, 55, 39, 255),  # inactive
        1: (77, 216, 39, 255),  # active
    },
    "temporary-crops": {
        0: (186, 186, 186, 0),  # no cropland
        1: (224, 24, 28, 200),  # cropland
    },
    "maize": {
        0: (186, 186, 186, 255),  # other
        1: (252, 207, 5, 255),  # maize
    },
    "croptype": {
        1: (103, 60, 32, 255),  # barley
        2: (252, 207, 5, 255),  # maize
        3: (247, 185, 29, 255),  # millet_sorghum
        4: (186, 186, 186, 0),  # other_crop
        5: (167, 245, 66, 255),  # rapeseed
        6: (85, 218, 218, 255),  # soy
        7: (245, 66, 111, 255),  # sunflower
        8: (186, 113, 53, 255),  # wheat
    },
    "probabilities": get_probability_cmap(),
}

COLORLEGEND = {
    "temporary-crops": {
        "No cropland": (186, 186, 186, 0),  # no cropland
        "Temporary crops": (224, 24, 28, 1),  # cropland
    },
    "croptype": {
        "Barley": (103, 60, 32, 1),
        "Maize": (252, 207, 5, 1),
        "Millet/Sorghum": (247, 185, 29, 1),
        "Other crop": (186, 186, 186, 1),
        "Rapeseed": (167, 245, 66, 1),
        "Soybean": (85, 218, 218, 1),
        "Sunflower": (245, 66, 111, 1),
        "Wheat": (186, 113, 53, 1),
    },
}

NODATAVALUE = {
    "active-cropland": 255,
    "temporary-crops": 255,
    "maize": 255,
    "croptype": 0,
    "probabilities": 255,
}

LOCALTILESERVER_PORT = 8889


def _get_colormap(product):
    if product in COLORMAP.keys():
        colormap = copy.deepcopy(COLORMAP[product])
    else:
        logger.warning(
            (f"Unknown product `{product}`: " "cannot assign colormap"))
        logger.info(f"Supported products: {COLORMAP.keys()}")
        colormap = None

    return colormap


def _get_nodata(product):
    return NODATAVALUE[product]


def get_default_filtersettings():
    return {"kernel_size": 7, "conf_threshold": 80}


def majority_vote(
    prediction: np.ndarray,
    probability: np.ndarray,
    kernel_size: int = 7,
    conf_threshold: int = 30,
    target_excluded_value: int = 255,
    excluded_values: List[int] = [255],
):
    """
    Majority vote is performed using a sliding local kernel.
    For each pixel, the voting of a final class is done from
    neighbours values weighted with the confidence threshold.
    Pixels that have one of the specified excluded values are
    excluded in the voting process and are unchanged.

    The prediction probabilities are reevaluated by taking, for each pixel,
    the average of probabilities of the neighbors that belong to the winning class.
    (For example, if a pixel was voted to class 2 and there are three
    neighbors of that class, then the new probability is the sum of the
    old probabilities of each pixels divided by 3)

    :param prediction: A 2D numpy array containing the predicted
        classification labels.
    :param probability: A 2D numpy array (same dimensions as predictions)
        containing the probabilities of the winning class (ranging between 0 and 100).
    :param kernel_size: The size of the kernel used for the neighbour around the pixel.
    :param conf_threshold: Pixels under this confidence threshold
        do not count into the voting process.
    :param target_excluded_value: Pixels that have a null score for every
        class are turned into this exclusion value
    :param excluded_values: Pixels that have one of the excluded values
        do not count into the voting process and are unchanged.

    :returns: the corrected classification labels and associated probabilities.
    """

    from scipy.signal import convolve2d

    # As the probabilities are in integers between 0 and 100,
    # we use uint16 matrices to store the vote scores
    assert (
        kernel_size <= 25
    ), f"Kernel value cannot be larger than 25 (currently: {kernel_size}) because it might lead to scenarios where the 16-bit count matrix is overflown"

    # Build a class mapping, so classes are converted to indexes and vice-versa
    unique_values = set(np.unique(prediction))
    unique_values = sorted(unique_values - set(excluded_values))
    index_value_lut = [(k, v) for k, v in enumerate(unique_values)]

    counts = np.zeros(
        shape=(*prediction.shape, len(unique_values)), dtype=np.uint16)
    probabilities = np.zeros(
        shape=(*probability.shape, len(unique_values)), dtype=np.uint16
    )

    # Iterates for each classes
    for cls_idx, cls_value in index_value_lut:
        # Take the binary mask of the interest class, and multiplies by the probabilities
        class_mask = ((prediction == cls_value) *
                      probability).astype(np.uint16)

        # Sets to 0 the class scores where the threshold is lower
        class_mask[probability <= conf_threshold] = 0

        # Set to 0 the class scores where the label is excluded
        for excluded_value in excluded_values:
            class_mask[prediction == excluded_value] = 0

        # Binary class mask, used to count HOW MANY neighbours pixels are used for this class
        binary_class_mask = (class_mask > 0).astype(np.uint16)

        # Creates the kernel
        kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint16)

        # Counts around the window the sum of probabilities for that given class
        counts[:, :, cls_idx] = convolve2d(class_mask, kernel, mode="same")

        # Counts the number of neighbors pixels that voted for that given class
        class_voters = convolve2d(binary_class_mask, kernel, mode="same")
        # Remove the 0 values because might create divide by 0 issues
        class_voters[class_voters == 0] = 1

        probabilities[:, :, cls_idx] = np.divide(
            counts[:, :, cls_idx], class_voters)

    # Initializes output array
    aggregated_predictions = np.zeros(
        shape=(counts.shape[0], counts.shape[1]), dtype=np.uint16
    )
    # Initializes probabilities output array
    aggregated_probabilities = np.zeros(
        shape=(counts.shape[0], counts.shape[1]), dtype=np.uint16
    )

    if len(unique_values) > 0:
        # Takes the indices that have the biggest scores
        aggregated_predictions_indices = np.argmax(counts, axis=2)

        # Get the new confidence score for the indices
        aggregated_probabilities = np.take_along_axis(
            probabilities,
            aggregated_predictions_indices.reshape(
                *aggregated_predictions_indices.shape, 1
            ),
            axis=2,
        ).squeeze()

        # Check which pixels have a counts value to 0
        no_score_mask = np.sum(counts, axis=2) == 0

        # convert back to values from indices
        for cls_idx, cls_value in index_value_lut:
            aggregated_predictions[aggregated_predictions_indices == cls_idx] = (
                cls_value
            )
            aggregated_predictions = aggregated_predictions.astype(np.uint16)

        aggregated_predictions[no_score_mask] = target_excluded_value
        aggregated_probabilities[no_score_mask] = target_excluded_value

    # Setting excluded values back to their original values
    for excluded_value in excluded_values:
        aggregated_predictions[prediction == excluded_value] = excluded_value
        aggregated_probabilities[prediction ==
                                 excluded_value] = target_excluded_value

    return aggregated_predictions, aggregated_probabilities


def postprocess_product(
    infile, product=None, colormap=None, nodata=None, filter_settings=None
):
    """
    Function taking care of post-processing of a WorldCereal product,
    including:
    - spatial filter to clean the classification results
    - (in case of a crop type output) deriving a cropland mask
        from the classified map
    - splitting the multi-band raster into individual files
    - assigning a colormap and no data value to each file

    The function assumes the input data consists of two layers:
    1) the classification layer
    2) the probabilities layer, indicating class probability of winning class

    Args:
        infile (str): path to the input file
        product (str): product identifier
        colormap (dict): colormap to assign to the classification layer
        nodata (int): nodata value to assign to the classification layer

    Returns:
        outfiles (dict): {label: path} of output files generated

    """

    # get properties and data from input file
    with rasterio.open(infile, "r") as src:
        labels = src.read(1)
        probs = src.read(2)
        meta = src.meta

    # get colormap and nodata value
    if product is not None:
        nodata = _get_nodata(product)
        colormap = _get_colormap(product)

    if nodata is None:
        nodata = NODATAVALUE

    # run spatial cleaning filter if required
    if filter_settings is None:
        filter_settings = get_default_filtersettings()
    if filter_settings["kernel_size"] > 0:
        newlabels, newprobs = majority_vote(
            labels,
            probs,
            kernel_size=filter_settings["kernel_size"],
            conf_threshold=filter_settings["conf_threshold"],
            target_excluded_value=0,
            excluded_values=[0, nodata],
        )

    # construct dictionary of output files to be generated
    outfiles = {
        "classification": {"data": newlabels, "colormap": colormap, "nodata": nodata},
        "probabilities": {
            "data": probs,
            "colormap": _get_colormap("probabilities"),
            "nodata": _get_nodata("probabilities"),
        },
    }

    # derive cropland mask if required
    if product != "temporary-crops":
        cropland = np.where(newlabels == 0, 1, 0)
        outfiles["croplandmask"] = {
            "data": cropland,
            "colormap": _get_colormap("temporary-crops"),
            "nodata": _get_nodata("temporary-crops"),
        }

    # write output files
    outpaths = {}
    meta.update(count=1)
    for label, settings in outfiles.items():
        outpath = infile.replace(".tif", f"-{label}.tif")
        with rasterio.open(outpath, "w", **meta) as dst:
            dst.write(settings["data"], indexes=1)
            dst.nodata = settings["nodata"]
            if settings["colormap"] is not None:
                dst.write_colormap(1, settings["colormap"])

        outpaths[label] = outpath

    return outpaths


def visualize_products(products, port=LOCALTILESERVER_PORT):
    """
    Function to visualize raster layers using leafmap.
    Only the first band of the input rasters is visualized.

    Args:
        products (dict): dictionary of products to visualize {label: path}
        port (int): port to use for localtileserver application

    Returns:
        leafmap Map instance
    """

    m = leafmap.Map()
    m.add_basemap("Esri.WorldImagery")
    for label, path in products.items():
        m.add_raster(path, indexes=[1], layer_name=label, port=port)
    m.add_colormap(
        "RdYlGn",
        label="Probabilities (%)",
        width=8.0,
        height=0.4,
        orientation="horizontal",
        vmin=0,
        vmax=100,
    )

    return m


def show_color_legend(product):
    import math

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    if product not in COLORLEGEND.keys():
        raise ValueError(
            f"Unknown product `{product}`: cannot generate color legend")

    colors = copy.deepcopy(COLORLEGEND.get(product))
    for key, value in colors.items():
        # apply scaling of RGB values
        rgb = [c / 255 for c in value[:-1]]
        rgb.extend([value[-1]])
        colors[key] = tuple(rgb)

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    ncols = 1

    names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

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

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

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
                facecolor=colors[name],
                edgecolor="0.7",
            )
        )


def get_ui_map():
    from ipyleaflet import basemap_to_tiles

    osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    osm.base = True
    osm.name = "Open street map"

    img = basemap_to_tiles(basemaps.Esri.WorldImagery)
    img.base = True
    img.name = "Satellite imagery"

    m = Map(center=(51.1872, 5.1154), zoom=10, layers=[osm, img])
    m.add_control(LayersControl())

    draw_control = DrawControl()

    draw_control.rectangle = {
        "shapeOptions": {
            "fillColor": "#6be5c3",
            "color": "#00F",
            "fillOpacity": 0.3,
        },
        "drawError": {
            "color": "#dd253b",
            "message": "Oups!"
        },
        "allowIntersection": False,
        "metric": ['km']
    }
    draw_control.circle = {}
    draw_control.polyline = {}
    draw_control.circlemarker = {}
    draw_control.polygon = {}

    m.add_control(draw_control)

    search = SearchControl(
        position="topleft",
        url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
        zoom=20,
    )
    m.add_control(search)

    return m, draw_control


def get_bbox_from_draw(dc, area_limit=25):
    obj = dc.last_draw
    if obj.get("geometry") is not None:
        poly = Polygon(shape(obj.get("geometry")))
        bbox = poly.bounds
    else:
        raise ValueError(
            "Please first draw a rectangle on the map before proceeding."
        )
    print(f"Your area of interest: {bbox}")

    # We convert our bounding box to local UTM projection
    # for further processing
    bbox_utm, epsg = _latlon_to_utm(bbox)
    area = (bbox_utm[2] - bbox_utm[0]) * (bbox_utm[3] - bbox_utm[1]) / 1000000
    print(f"Area of processing extent: {area:.2f} km²")

    if area_limit is not None:
        if area > area_limit:
            spatial_extent = None
            raise ValueError(
                "Area of processing extent is too large. "
                "Please select a smaller area."
            )
    spatial_extent = BoundingBoxExtent(*bbox_utm, epsg)

    return spatial_extent, bbox, poly


def _latlon_to_utm(bbox):
    """This function converts a bounding box defined in lat/lon
    to local UTM coordinates.
    It returns the bounding box in UTM and the epsg code
    of the resulting UTM projection."""

    # convert bounding box to geodataframe
    bbox_poly = geometry.box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")

    # estimate best UTM zone
    crs = bbox_gdf.estimate_utm_crs()
    epsg = int(crs.to_epsg())

    # convert to UTM
    bbox_utm = bbox_gdf.to_crs(crs).total_bounds

    return bbox_utm, epsg
