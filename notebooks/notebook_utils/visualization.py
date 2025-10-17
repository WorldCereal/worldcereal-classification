import copy
import logging
import random
from pathlib import Path
from typing import Optional
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
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
    colors = [
        (186, 186, 186),
        (255, 255, 255),
    ]  # grey is reserved for no-cropland, white for no data
    while len(colors) < n + 2:  # +2 for no-cropland and no-data
        new_color = generate_random_color()
        if all(color_distance(new_color, c) > min_distance for c in colors):
            colors.append(new_color)
    return colors[2:]


def _get_colormap(product, lut=None):

    if product in COLORMAP.keys():
        colormap = copy.deepcopy(COLORMAP[product])
    else:
        if lut is not None:
            logger.info((f"Assigning random color map for product {product}. "))
            colormap = {}
            # Generate distinct colors for the LUT
            distinct_colors = generate_distinct_colors(len(lut.values()))
            for code, color in zip(list(lut.values()), distinct_colors):
                colormap[code] = (*color, 255)
        else:
            colormap = None

    return colormap


# Helper function to scale RGB values
def scale_rgb(color):
    # Scaling only RGB, ignoring alpha
    return tuple(c / 255 for c in color[:3])


def rgb_to_hex(r, g, b):
    return f"{r:02x}{g:02x}{b:02x}"


def visualize_product(
    path: Path,
    product: Literal["cropland", "croptype", "cropland-raw", "croptype-raw"],
    lut: Optional[dict] = None,
    interactive_mode: bool = False,
    port: int = 8889,
):
    """
    Visualize a WorldCereal map product using matplotlib.

    Parameters
    ----------
    path : str or Path
        Path to the raster file.
    product : Literal["cropland", "croptype", "cropland-raw", "croptype-raw"]
        Name of the product to visualize.
    lut : dict, optional
        Lookup table for product classes, if available.
        If None, we assume the default cropland LUT.
    interactive_mode : bool, optional
        If True, display the plot interactively using leafmap. Default is False.
    port : int, optional
        Port number for leafmap interactive display. Default is 8889.
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

    with rasterio.open(path, "r") as src:
        classification = src.read(1).astype(np.uint8)
        probability = src.read(2).astype(np.uint8)
        meta = src.meta

    nodata = _get_nodata(product)

    colormap = _get_colormap(product, lut)
    # Adjust LUT and colormap for crop type product
    if product.startswith("croptype"):
        # add no cropland class
        lut["no_cropland"] = 254
        colormap[254] = (186, 186, 186, 255)  # no cropland color
    # add no data color to lut and colormap
    lut["no_data"] = nodata
    colormap[nodata] = (255, 255, 255, 255)  # no data color

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

    # invert lut
    lut_inverted = {v: k for k, v in lut.items()}

    if interactive_mode:

        # We plot using leafmap in interactive mode
        # to allow for zooming and panning
        import os
        import leafmap

        if os.environ.get("JUPYTERHUB_SERVICE_PREFIX") is not None:
            if os.environ["JUPYTERHUB_SERVICE_PREFIX"].endswith("/"):
                # For some reason, leafmap adds too many '/' to the url which is based on JUPYTERHUB_SERVICE_PREFIX
                # We can fix this by simply updating the prefix!
                os.environ["JUPYTERHUB_SERVICE_PREFIX"] = os.environ[
                    "JUPYTERHUB_SERVICE_PREFIX"
                ][:-1]
            os.environ["LOCALTILESERVER_CLIENT_PREFIX"] = (
                f"{os.environ['JUPYTERHUB_SERVICE_PREFIX']}proxy/{{port}}"
            )

        # Create the map
        m = leafmap.Map(draw_control=False)
        m.add_basemap("Esri.WorldImagery")
        # Add classification product
        m.add_raster(
            str(outpath), indexes=[1], layer_name=f"{product}-classification", port=port
        )
        # Add probability product
        m.add_raster(
            str(outpath_prob),
            indexes=[1],
            layer_name=f"{product}-probability",
            port=port,
        )

        # Plot legend separately
        legend_dict = {lut_inverted[k]: rgb_to_hex(*v[:3]) for k, v in colormap.items()}

        m.add_legend(
            title="Classification",
            legend_dict=legend_dict,
            bg_color="rgba(255, 255, 255, 0.5)",
            position="bottomright",
        )

        # Return the map
        return m

    else:
        # We plot using matplotlib in non-interactive mode
        fig, ax = plt.subplots()

        # create a second axes for the colorbar
        ax2 = fig.add_axes([0.95, 0.2, 0.03, 0.5])

        # Colorbar preparation
        # Apply RGB scaling
        colormap = {key: scale_rgb(value) for key, value in colormap.items()}

        # Sort colormap according to keys
        colormap = {key: colormap[key] for key in sorted(colormap.keys())}
        codes = list(colormap.keys())

        # Create a custom ListedColormap
        cmap = mpl.colors.ListedColormap([colormap[key] for key in codes])

        # Get class labels and set colorbar boundaries
        # define boundaries halfway between each code:
        bounds = [(a + b) / 2 for a, b in zip(codes, codes[1:])]
        # extend for under/overflow:
        bounds = [codes[0] - 0.5] + bounds + [codes[-1] + 0.5]

        # Define a norm for the colormap
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # Define tick positions and labels for the colorbar
        tick_positions = np.arange(len(codes)) + 0.5
        uniform_bounds = np.arange(len(codes) + 1)  # [0, 1, 2, ..., len(codes)]
        norm_cb = mpl.colors.BoundaryNorm(uniform_bounds, cmap.N)  # norm for colorbar
        classlabels = [str(lut_inverted.get(code, code)) for code in codes]

        # plot the raster
        ax.imshow(classification, cmap=cmap, norm=norm)

        # Create a colorbar with class labels
        cb = mpl.colorbar.ColorbarBase(
            ax2,
            cmap=cmap,
            norm=norm_cb,
            spacing="uniform",
            boundaries=uniform_bounds,
            ticks=tick_positions,
            format="%1i",
        )

        # Set the colorbar labels
        cb.set_ticklabels(classlabels)

        # Turn off axis
        ax.axis("off")

        # Display the plot
        plt.show()


def extract_zonal_stats(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    band: int = 1,
    stats: list = ["mean"],
    nodata_value: Optional[int] = None,
):
    """Extract zonal statistics using rasterio mask.
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the field geometries.
    raster_path : str
        Path to the raster file.
    band : int, optional
        Band number to extract statistics from. Default is 1.
    stats : list, optional
        List of statistics to calculate. Supported statistics are 'mean', 'sum', 'count',
        'std', 'min', 'max'. Default is ['mean'].
    nodata_value : int, optional
        Value to ignore in the raster data. If None, uses the raster's nodata value.
    Returns
    -------
    results : list of dict
        List of dictionaries containing the calculated statistics for each field."""

    results = []

    with rasterio.open(raster_path) as src:
        # Ensure fields are in same CRS as raster
        fields_reprojected = gdf.to_crs(src.crs)

        for idx, row in fields_reprojected.iterrows():
            try:
                # Mask raster with field geometry
                masked_data, masked_transform = mask(
                    src, [row.geometry], crop=True, nodata=src.nodata
                )

                # Get the specific band
                band_data = masked_data[band - 1]  # bands are 1-indexed

                # Remove nodata values
                valid_data = band_data[band_data != src.nodata]
                if nodata_value is not None:
                    valid_data = valid_data[valid_data != nodata_value]

                if len(valid_data) == 0:
                    results.append({stat: np.nan for stat in stats})
                    continue

                # Calculate statistics
                stat_dict = {}
                for stat in stats:
                    if stat == "mean":
                        stat_dict[stat] = np.mean(valid_data)
                    elif stat == "sum":
                        stat_dict[stat] = np.sum(valid_data)
                    elif stat == "count":
                        stat_dict[stat] = len(valid_data)
                    elif stat == "std":
                        stat_dict[stat] = np.std(valid_data)
                    elif stat == "min":
                        stat_dict[stat] = np.min(valid_data)
                    elif stat == "max":
                        stat_dict[stat] = np.max(valid_data)

                results.append(stat_dict)

            except Exception:
                results.append({stat: np.nan for stat in stats})

    return results
