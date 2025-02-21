import ast
import copy
import logging
import random
from pathlib import Path
from typing import List

import leafmap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from loguru import logger

from worldcereal.parameters import WorldCereal2021ProductType

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
    "probability": 255,
}

LUTS = {
    "temporarycrops": {
        0: "no_cropland",
        100: "cropland",
    },
    "activecropland": {
        0: "inactive",
        100: "active",
        254: "no_cropland",
    },
    "wintercereals": {
        0: "other",
        100: "winter cereal",
        254: "no_cropland",
    },
    "springcereals": {
        0: "other",
        100: "spring cereal",
        254: "no_cropland",
    },
    "maize": {
        0: "other",
        100: "maize",
        254: "no_cropland",
    },
    "irrigation": {
        0: "rainfed",
        100: "irrigated",
        254: "no_cropland",
    },
}


COLORMAP = {
    "cropland": {
        0: (186, 186, 186, 0),  # no cropland
        1: (224, 24, 28, 200),  # cropland
    },
    "temporarycrops": {
        0: (186, 186, 186, 0),  # no cropland
        100: (224, 24, 28, 200),  # cropland
    },
    "activecropland": {
        0: (232, 55, 39, 255),  # inactive
        100: (77, 216, 39, 255),  # active
        254: (0, 0, 0, 0),  # no cropland
    },
    "wintercereals": {
        0: (186, 186, 186, 255),  # other
        100: (186, 113, 53, 255),  # wheat
        254: (0, 0, 0, 0),  # no cropland
    },
    "springcereals": {
        0: (186, 186, 186, 255),  # other
        100: (186, 113, 53, 255),  # wheat
        254: (0, 0, 0, 0),  # no cropland
    },
    "maize": {
        0: (186, 186, 186, 255),  # other
        100: (252, 207, 5, 255),  # maize
        254: (0, 0, 0, 0),  # no cropland
    },
    "irrigation": {
        0: (255, 132, 132, 255),  # rainfed
        100: (104, 149, 197, 255),  # irrigated
        254: (0, 0, 0, 0),  # no cropland
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


# Helper function to scale RGB values
def scale_rgb(color):
    return tuple(c / 255 for c in color[:3])  # Scale RGB only, ignore alpha


def visualize_2021_products(
    product_paths: List[Path],
    product_type: str,
):

    # Ensure product is a 2021 product type
    if product_type not in WorldCereal2021ProductType.__members__:
        raise ValueError(
            f"Product type {product_type} is not a valid 2021 product type."
        )

    # Get correct LUT and colormap
    colormap = _get_colormap(product_type.lower())
    lut = LUTS[product_type.lower()]

    # Apply RGB scaling
    colormap = {key: scale_rgb(value) for key, value in colormap.items()}

    # Eliminate the metadata file from list of paths
    product_paths = [path for path in product_paths if not path.name.endswith(".json")]

    # Dynamically extract unique values across all rasters
    unique_values = []
    for path in product_paths:
        with rasterio.open(path) as src:
            unique_values.extend(np.unique(src.read(1)))

    unique_values = sorted(list(set(unique_values)))  # Remove duplicates

    # Ensure all unique values exist in colormap
    missing_keys = [val for val in unique_values if val not in colormap]
    if missing_keys:
        raise ValueError(f"Missing colormap entries for values: {missing_keys}")

    # Create colormap dynamically
    cmap = mpl.colors.ListedColormap([colormap[val] for val in unique_values])

    # Define flexible boundaries
    bounds = unique_values + [unique_values[-1] + 1]  # Extend to cover last value
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Prepare figure
    nproducts = len(product_paths)
    fig, axes = plt.subplots(nproducts, 1, figsize=(7, 5 * nproducts))

    # Ensure axes is always a list, even when nproducts == 1
    if nproducts == 1:
        axes = [axes]

    # Plot all figures
    for i, path in enumerate(product_paths):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.uint8)

        # Plot the raster
        im = axes[i].imshow(data, cmap=cmap, norm=norm)
        axes[i].set_title(path.name.split(".")[0])
        axes[i].axis("off")

    # Create a single colorbar
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    cb = plt.colorbar(im, cax=cbar_ax, ticks=unique_values)

    # Set dynamic colorbar labels
    cb.set_ticklabels([lut[val] for val in unique_values])

    # Display the plot
    plt.subplots_adjust(right=0.85)
    plt.show()
