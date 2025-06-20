import ast
import copy
import logging
import random
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from loguru import logger

logging.getLogger("rasterio").setLevel(logging.ERROR)


def get_probability_cmap():
    colormap = plt.get_cmap("RdYlGn")
    cmap = {}
    for i in range(101):
        cmap[i] = tuple((np.array(colormap(int(2.55 * i))) * 255).astype(int))
    return cmap


NODATAVALUE = {
    "cropland": 255,
    "croptype": 255,
    "cropland-raw": 255,
    "croptype-raw": 255,
    "probability": 255,
}


COLORMAP = {
    "cropland": {
        0: (186, 186, 186, 0),  # no cropland
        1: (224, 24, 28, 200),  # cropland
    },
    "cropland-raw": {
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
    bounds = np.linspace(0, len(classlabels), len(classlabels) + 1)

    # Define a norm for the colormap
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # plot the raster
    ax.imshow(arr_classif, cmap=cmap, norm=norm)

    # Create a colorbar with class labels
    cb = mpl.colorbar.ColorbarBase(
        ax2,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        boundaries=bounds,
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


# Helper function to scale RGB values
def scale_rgb(color):
    # Scaling only RGB, ignoring alpha
    return tuple(c / 255 for c in color[:3])


def visualize_product(
    path: Path,
    product: str,
    lut: Optional[dict] = None,
    write: bool = False,
):
    """
    Visualize a WorldCereal map product using matplotlib.

    Parameters
    ----------
    path : str or Path
        Path to the raster file.
    product : str
        Name of the product to visualize.
    lut : dict, optional
        Lookup table for product classes, if available.
        If None, we assume the default cropland LUT.
    write : bool, optional
        If True, write the classification and probability rasters with colormap to disk.
        Default is False.
    """

    if product not in ["cropland", "croptype", "cropland-raw", "croptype-raw"]:
        raise ValueError(
            f"Product {product} is not supported. Supported products are: "
            "'cropland', 'croptype', 'cropland-raw', 'croptype-raw'."
        )
    if product not in ["cropland", "cropland-raw"]:
        if lut is None:
            raise ValueError(
                f"Look-up table is required for product {product}. "
                "Please provide a valid LUT."
            )
    else:
        if lut is None:
            logger.info(
                f"No LUT provided for product {product}. "
                "Using default LUT for cropland products."
            )
            lut = {
                "no_cropland": 0,
                "cropland": 1,
            }

    # Adjust LUT for crop type product
    if product.startswith("croptype"):
        # add no cropland class
        lut["no_cropland"] = 254

    with rasterio.open(path, "r") as src:
        classification = src.read(1).astype(np.uint8)
        probability = src.read(2).astype(np.uint8)
        meta = src.meta

    nodata = _get_nodata(product)
    colormap = _get_colormap(product, lut)
    colormap = {k: v for k, v in colormap.items() if k in lut.values()}

    if write:
        # Write the classification raster with colormap
        meta.update(count=1, dtype=rasterio.uint8, nodata=nodata)
        outpath = path.parent / f"{path.stem}_classification.tif"
        bandnames = ["classification"]
        with rasterio.open(outpath, "w", **meta) as dst:
            dst.write(classification, indexes=1)
            dst.write_colormap(1, colormap)
            for i, b in enumerate(bandnames):
                dst.update_tags(i + 1, band_name=b)
                dst.update_tags(i + 1, lut=lut)
        # Write the probability raster with colormap
        meta.update(count=1, dtype=rasterio.uint8, nodata=_get_nodata("probability"))
        outpath_prob = path.parent / f"{path.stem}_probability.tif"
        bandnames = ["probability"]
        with rasterio.open(outpath_prob, "w", **meta) as dst:
            dst.write(probability, indexes=1)
            dst.write_colormap(1, _get_colormap("probability"))
            for i, b in enumerate(bandnames):
                dst.update_tags(i + 1, band_name=b)
        logger.info("Visualization saved")

    # Apply RGB scaling
    colormap = {key: scale_rgb(value) for key, value in colormap.items()}

    # Create a custom ListedColormap
    cmap = mpl.colors.ListedColormap([colormap[key] for key in sorted(colormap.keys())])

    fig, ax = plt.subplots()

    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.2, 0.03, 0.5])

    # Get class labels and set colorbar boundaries
    classlabels = list(lut.keys())
    bounds = np.linspace(0, len(classlabels), len(classlabels) + 1)

    # Define a norm for the colormap
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # plot the raster
    ax.imshow(classification, cmap=cmap, norm=norm)

    # Create a colorbar with class labels
    cb = mpl.colorbar.ColorbarBase(
        ax2,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        boundaries=bounds,
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
