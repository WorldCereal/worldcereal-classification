import math

import geopandas as gpd
import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import Polygon


def _load_array_bounds_latlon(
    fname,
    bounds,
    rio_gdal_options=None,
    boundless=True,
    fill_value=np.nan,
    nodata_value=None,
):
    rio_gdal_options = rio_gdal_options or {}

    with rasterio.Env(**rio_gdal_options):
        with rasterio.open(fname) as src:
            window = rasterio.windows.from_bounds(*bounds, src.transform)

            vals = np.array(window.flatten())
            if (vals % 1 > 0).any():
                col_off, row_off, width, height = (
                    window.col_off,
                    window.row_off,
                    window.width,
                    window.height,
                )

                width, height = int(math.ceil(width)), int(math.ceil(height))
                col_off, row_off = int(math.floor(col_off + 0.5)), int(
                    math.floor(row_off + 0.5)
                )

                window = rasterio.windows.Window(col_off, row_off, width, height)

            if nodata_value is None:
                nodata_value = src.nodata if src.nodata is not None else np.nan

            if nodata_value is None and boundless is True:
                logger.warning(
                    "Raster has no data value, defaulting boundless"
                    " to False. Specify a nodata_value to read "
                    "boundless."
                )
                boundless = False

            arr = src.read(window=window, boundless=boundless, fill_value=nodata_value)
            arr = arr.astype(
                np.float32
            )  # needed reprojecting with bilinear resampling  # noqa:e501

            if nodata_value is not None:
                arr[arr == nodata_value] = fill_value

    arr = arr[0]
    return arr


def load_reproject(
    filename,
    bounds,
    epsg,
    resolution=10,
    border_buff=0,
    fill_value=0,
    nodata_value=None,
    rio_gdal_options=None,
    resampling=Resampling.nearest,
    dtype=None,
):
    """Read from lat/lon layer and reproject to target CRS.
    Supports float and integer loading.

    Parameters
    ----------
    filename : str or file-like
        Source raster filename or file object.
    bounds : tuple
        (minx, miny, maxx, maxy) in target EPSG coordinates.
    epsg : int
        EPSG code of target CRS.
    resolution : int, default 10
        Output resolution in target CRS units.
    border_buff : int, default 0
        Buffer (in pixels) to trim around the output after reprojection.
    fill_value : numeric, default 0
        Value to assign where nodata occurs after reading source window.
    nodata_value : numeric, optional
        Explicit nodata value for source raster; if None uses src.nodata.
    rio_gdal_options : dict, optional
        Extra environment options for rasterio / GDAL.
    resampling : rasterio.enums.Resampling, default nearest
        Resampling method used during reprojection.
    dtype : numpy dtype, optional
        Desired dtype of output. If None, defaults to float32.
        If provided and resampling is nearest, integer data is preserved.

    Returns
    -------
    np.ndarray
        Reprojected array.
    """
    bbox = gpd.GeoSeries(Polygon.from_bounds(*bounds), crs=CRS.from_epsg(epsg))

    bounds = bbox.buffer(border_buff * resolution).to_crs(epsg=4326).bounds.values[0]
    utm_bounds = bbox.buffer(border_buff * resolution).bounds.values[0].tolist()

    width = max(1, int((utm_bounds[2] - utm_bounds[0]) / resolution))
    height = max(1, int((utm_bounds[3] - utm_bounds[1]) / resolution))

    gim = _load_array_bounds_latlon(
        filename,
        bounds,
        rio_gdal_options=rio_gdal_options,
        fill_value=fill_value,
        nodata_value=nodata_value,
    )

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(bbox.crs.to_epsg())

    src_transform = rasterio.transform.from_bounds(*bounds, gim.shape[1], gim.shape[0])
    dst_transform = rasterio.transform.from_bounds(*utm_bounds, width, height)

    # Determine output dtype
    out_dtype = dtype if dtype is not None else np.float32
    dst = np.zeros((height, width), dtype=out_dtype)

    # Prepare source array for reprojection: force float32 for non-nearest resampling
    if resampling == Resampling.nearest and dtype is not None:
        gim_for_reproject = gim.astype(out_dtype)
    else:
        gim_for_reproject = gim.astype(np.float32)

    reproject(
        gim_for_reproject,
        dst,
        src_transform=src_transform,
        dst_transform=dst_transform,
        src_crs=src_crs,
        dst_crs=dst_crs,
        resampling=resampling,
    )

    if border_buff > 0:
        dst = dst[border_buff:-border_buff, border_buff:-border_buff]

    return dst
