"""Feature computer GFMAP compatible to compute Presto embeddings."""

import copy
import functools
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData
from pyproj import Transformer
from pyproj.crs import CRS
from scipy.ndimage import (
    convolve,
    zoom,
)
from shapely.geometry import Point
from shapely.ops import transform

sys.path.append("feature_deps")

import torch  # noqa: E402

PROMETHEO_WHL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/dependencies/prometheo-0.0.2-py3-none-any.whl"

GFMAP_BAND_MAPPING = {
    "S2-L2A-B02": "B2",
    "S2-L2A-B03": "B3",
    "S2-L2A-B04": "B4",
    "S2-L2A-B05": "B5",
    "S2-L2A-B06": "B6",
    "S2-L2A-B07": "B7",
    "S2-L2A-B08": "B8",
    "S2-L2A-B8A": "B8A",
    "S2-L2A-B11": "B11",
    "S2-L2A-B12": "B12",
    "S1-SIGMA0-VH": "VH",
    "S1-SIGMA0-VV": "VV",
    "AGERA5-TMEAN": "temperature_2m",
    "AGERA5-PRECIP": "total_precipitation",
}

LAT_HARMONIZED_NAME = "GEO-LAT"
LON_HARMONIZED_NAME = "GEO-LON"
EPSG_HARMONIZED_NAME = "GEO-EPSG"


@functools.lru_cache(maxsize=6)
def unpack_prometheo_wheel(wheel_url: str):
    destination_dir = Path.cwd() / "dependencies" / "prometheo"
    destination_dir.mkdir(exist_ok=True, parents=True)

    # Downloads the wheel file
    modelfile, _ = urllib.request.urlretrieve(
        wheel_url, filename=Path.cwd() / Path(wheel_url).name
    )
    with zipfile.ZipFile(modelfile, "r") as zip_ref:
        zip_ref.extractall(destination_dir)
    return destination_dir


@functools.lru_cache(maxsize=6)
def compile_encoder(presto_encoder):
    """Helper function that compiles the encoder of a Presto model
    and performs a warm-up on dummy data. The lru_cache decorator
    ensures caching on compute nodes to be able to actually benefit
    from the compilation process.

    Parameters
    ----------
    presto_encoder : nn.Module
        Encoder part of Presto model to compile

    """

    presto_encoder = torch.compile(presto_encoder)  # type: ignore

    for _ in range(3):
        presto_encoder(
            torch.rand((1, 12, 17)),
            torch.ones((1, 12)).long(),
            torch.rand(1, 2),
        )

    return presto_encoder


def get_output_labels() -> list:
    """Returns the output labels from this UDF, which is the output labels
    of the presto embeddings"""
    return [f"presto_ft_{i}" for i in range(128)]


def evaluate_resolution(inarr: xr.DataArray, epsg: int) -> int:
    """Helper function to get the resolution in meters for
    the input array.

    Parameters
    ----------
    inarr : xr.DataArray
        input array to determine resolution for.

    Returns
    -------
    int
        resolution in meters.
    """

    if epsg == 4326:
        # self.logger.info(
        #     "Converting WGS84 coordinates to EPSG:3857 to determine resolution."
        # )

        transformer = Transformer.from_crs(epsg, 3857, always_xy=True)
        points = [Point(x, y) for x, y in zip(inarr.x.values, inarr.y.values)]
        points = [transform(transformer.transform, point) for point in points]

        resolution = abs(points[1].x - points[0].x)

    else:
        resolution = abs(inarr.x[1].values - inarr.x[0].values)

    # self.logger.info(f"Resolution for computing slope: {resolution}")

    return resolution


def compute_slope(inarr: xr.DataArray, resolution: int) -> xr.DataArray:
    """Computes the slope using the scipy library. The input array should
    have the following bands: 'elevation' And no time dimension. Returns a
    new DataArray containing the new `slope` band.

    Parameters
    ----------
    inarr : xr.DataArray
        input array containing a band 'elevation'.
    resolution : int
        resolution of the input array in meters.

    Returns
    -------
    xr.DataArray
        output array containing 'slope' band in degrees.
    """

    def _rolling_fill(darr, max_iter=2):
        """Helper function that also reflects values inside
        a patch with NaNs."""
        if max_iter == 0:
            return darr
        else:
            max_iter -= 1
        # arr of shape (rows, cols)
        mask = np.isnan(darr)

        if ~np.any(mask):
            return darr

        roll_params = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(roll_params)

        for roll_param in roll_params:
            rolled = np.roll(darr, roll_param, axis=(0, 1))
            darr[mask] = rolled[mask]

        return _rolling_fill(darr, max_iter=max_iter)

    def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
        """Downsamples a 2D NumPy array by a given factor with average resampling and reflect padding.

        Parameters
        ----------
        arr : np.ndarray
            The 2D input array.
        factor : int
            The factor by which to downsample. For example, factor=2 downsamples by 2x.

        Returns
        -------
        np.ndarray
            Downsampled array.
        """

        # Get the original shape of the array
        X, Y = arr.shape

        # Calculate how much padding is needed for each dimension
        pad_X = (
            factor - (X % factor)
        ) % factor  # Ensures padding is only applied if needed
        pad_Y = (
            factor - (Y % factor)
        ) % factor  # Ensures padding is only applied if needed

        # Pad the array using 'reflect' mode
        padded = np.pad(arr, ((0, pad_X), (0, pad_Y)), mode="reflect")

        # Reshape the array to form blocks of size 'factor' x 'factor'
        reshaped = padded.reshape(
            (X + pad_X) // factor, factor, (Y + pad_Y) // factor, factor
        )

        # Take the mean over the factor-sized blocks
        downsampled = np.nanmean(reshaped, axis=(1, 3))

        return downsampled

    dem = inarr.sel(bands="elevation").values
    dem_arr = dem.astype(np.float32)

    # Invalid to NaN and keep track of these pixels
    dem_arr[dem_arr == 65535] = np.nan
    idx_invalid = np.isnan(dem_arr)

    # Fill NaNs with rolling fill
    dem_arr = _rolling_fill(dem_arr)

    # We make sure DEM is at 20m for slope computation
    # compatible with global slope collection
    factor = int(20 / resolution)
    if factor < 1 or factor % 2 != 0:
        raise NotImplementedError(
            f"Unsupported resolution for slope computation: {resolution}"
        )
    dem_arr_downsampled = _downsample(dem_arr, factor)
    x_odd, y_odd = dem_arr.shape[0] % 2 != 0, dem_arr.shape[1] % 2 != 0

    # Mask NaN values in the DEM data
    dem_masked = np.ma.masked_invalid(dem_arr_downsampled)

    # Define convolution kernels for x and y gradients (simple finite difference approximation)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (
        8.0 * 20  # array is now at 20m resolution
    )  # x-derivative kernel

    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (
        8.0 * 20  # array is now at 20m resolution
    )  # y-derivative kernel

    # Apply convolution to compute gradients
    dx = convolve(dem_masked, kernel_x)  # Gradient in the x-direction
    dy = convolve(dem_masked, kernel_y)  # Gradient in the y-direction

    # Reapply the mask to the gradients
    dx = np.ma.masked_where(dem_masked.mask, dx)
    dy = np.ma.masked_where(dem_masked.mask, dy)

    # Calculate the magnitude of the gradient (rise/run)
    gradient_magnitude = np.ma.sqrt(dx**2 + dy**2)

    # Convert gradient magnitude to slope (in degrees)
    slope = np.ma.arctan(gradient_magnitude) * (180 / np.pi)

    # Upsample to original resolution with bilinear interpolation
    mask = slope.mask
    mask = zoom(mask, zoom=factor, order=0)
    slope = zoom(slope, zoom=factor, order=1)
    slope[mask] = 65535

    # Strip one row or column if original array was odd in that dimension
    if x_odd:
        slope = slope[:-1, :]
    if y_odd:
        slope = slope[:, :-1]

    # Fill slope values where the original DEM had NaNs
    slope[idx_invalid] = 65535
    slope[np.isnan(slope)] = 65535
    slope = slope.astype(np.uint16)

    return xr.DataArray(
        slope[None, :, :],
        dims=("bands", "y", "x"),
        coords={
            "bands": ["slope"],
            "y": inarr.y,
            "x": inarr.x,
        },
    )


def select_timestep_from_temporal_features(
    features: xr.DataArray, target_date: Optional[str] = None
) -> xr.DataArray:
    """Select a specific timestep from temporal features based on target date.

    Parameters
    ----------
    features : xr.DataArray
        Temporal features with time dimension preserved.
    target_date : str, optional
        Target date in ISO format (YYYY-MM-DD). If None, selects middle timestep.

    Returns
    -------
    xr.DataArray
        Features for the selected timestep with time dimension removed.
    """
    if target_date is None:
        # Select middle timestep
        mid_idx = len(features.t) // 2
        features = features.isel(t=mid_idx)
    else:
        # Parse target date and find closest timestep
        target_datetime = np.datetime64(target_date)

        # Check if target_datetime is within the temporal extent of features
        min_time = features.t.min().values
        max_time = features.t.max().values

        if target_datetime < min_time or target_datetime > max_time:
            raise ValueError(
                f"Target date {target_date} is outside the temporal extent of features. "
                f"Available time range: {min_time} to {max_time}"
            )

        # Find closest timestep
        features = features.sel(t=target_datetime, method="nearest")

    return features


def execute(inarr: xr.DataArray, parameters: dict, epsg: int) -> xr.DataArray:
    if epsg is None:
        raise ValueError(
            "EPSG code is required for Presto feature extraction, but was "
            "not correctly initialized."
        )
    if "presto_model_url" not in parameters:
        raise ValueError('Missing required parameter "presto_model_url"')

    presto_model_url = parameters.get("presto_model_url")
    # self.logger.info(f'Loading Presto model from "{presto_model_url}"')
    prometheo_wheel_url = parameters.get("prometheo_wheel_url", PROMETHEO_WHL_URL)
    # self.logger.info(f'Loading Prometheo wheel from "{prometheo_wheel_url}"')

    ignore_dependencies = parameters.get("ignore_dependencies", False)
    # if ignore_dependencies:
    # self.logger.info(
    #     "`ignore_dependencies` flag is set to True. Make sure that "
    #     "Presto and its dependencies are available on the runtime "
    #     "environment"
    # )

    # The below is required to avoid flipping of the result
    # when running on OpenEO backend!
    inarr = inarr.transpose("bands", "t", "x", "y")

    # Change the band names
    new_band_names = [GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands]
    inarr = inarr.assign_coords(bands=new_band_names)

    # Handle NaN values in Presto compatible way
    inarr = inarr.fillna(65535)

    if not ignore_dependencies:
        # Unzip the Presto dependencies on the backend
        # self.logger.info("Unpacking prometheo wheel")
        deps_dir = unpack_prometheo_wheel(prometheo_wheel_url)

        # self.logger.info("Appending dependencies")
        sys.path.append(str(deps_dir))

    if "slope" not in inarr.bands:
        # If 'slope' is not present we need to compute it here
        # self.logger.warning("`slope` not found in input array. Computing ...")
        resolution = evaluate_resolution(inarr.isel(t=0), epsg)
        slope = compute_slope(inarr.isel(t=0), resolution)
        slope = slope.expand_dims({"t": inarr.t}, axis=0).astype("float32")

        inarr = xr.concat([inarr.astype("float32"), slope], dim="bands")

    batch_size = parameters.get("batch_size", 256)
    temporal_prediction = parameters.get("temporal_prediction", False)
    target_date = parameters.get("target_date", None)

    # TODO: compile_presto not used for now?
    # compile_presto = parameters.get("compile_presto", False)
    # self.logger.info(f"Compile presto: {compile_presto}")

    # self.logger.info("Loading Presto model for inference")

    # TODO: try to take run_model_inference from worldcereal
    from prometheo.datasets.worldcereal import run_model_inference
    from prometheo.models import Presto
    from prometheo.models.pooling import PoolingMethods
    from prometheo.models.presto.wrapper import load_presto_weights

    presto_model = Presto()
    presto_model = load_presto_weights(presto_model, presto_model_url)

    # self.logger.info("Extracting presto features")
    # Check if we have the expected 12 timesteps
    if len(inarr.t) != 12:
        raise ValueError(f"Can only run Presto on 12 timesteps, got: {len(inarr.t)}")

    # Determine pooling method based on temporal_prediction parameter
    pooling_method = (
        PoolingMethods.TIME if temporal_prediction else PoolingMethods.GLOBAL
    )

    features = run_model_inference(
        inarr,
        presto_model,
        epsg=epsg,
        batch_size=batch_size,
        pooling_method=pooling_method,
    )

    # If temporal prediction, select specific timestep based on target_date
    if temporal_prediction:
        features = select_timestep_from_temporal_features(features, target_date)

    return features


def get_latlons(inarr: xr.DataArray, epsg: int) -> xr.DataArray:
    """Returns the latitude and longitude coordinates of the given array in
    a dataarray. Returns a dataarray with the same width/height of the input
    array, but with two bands, one for latitude and one for longitude. The
    metadata coordinates of the output array are the same as the input
    array, as the array wasn't reprojected but instead new features were
    computed.

    The latitude and longitude band names are standardized to the names
    `LAT_HARMONIZED_NAME` and `LON_HARMONIZED_NAME` respectively.
    """

    lon = inarr.coords["x"]
    lat = inarr.coords["y"]
    lon, lat = np.meshgrid(lon, lat)

    if epsg is None:
        raise Exception(
            "EPSG code was not defined, cannot extract lat/lon array "
            "as the CRS is unknown."
        )

    # If the coordiantes are not in EPSG:4326, we need to reproject them
    if epsg != 4326:
        # Initializes a pyproj reprojection object
        transformer = Transformer.from_crs(
            crs_from=CRS.from_epsg(epsg),
            crs_to=CRS.from_epsg(4326),
            always_xy=True,
        )
        lon, lat = transformer.transform(xx=lon, yy=lat)

    # Create a two channel numpy array of the lat and lons together by stacking
    latlon = np.stack([lat, lon])

    # Repack in a dataarray
    return xr.DataArray(
        latlon,
        dims=["bands", "y", "x"],
        coords={
            "bands": [LAT_HARMONIZED_NAME, LON_HARMONIZED_NAME],
            "y": inarr.coords["y"],
            "x": inarr.coords["x"],
        },
    )


def rescale_s1_backscatter(arr: xr.DataArray) -> xr.DataArray:
    """Rescales the input array from uint16 to float32 decibel values.
    The input array should be in uint16 format, as this optimizes memory usage in Open-EO
    processes. This function is called automatically on the bands of the input array, except
    if the parameter `rescale_s1` is set to False.
    """
    s1_bands = ["S1-SIGMA0-VV", "S1-SIGMA0-VH", "S1-SIGMA0-HV", "S1-SIGMA0-HH"]
    s1_bands_to_select = list(set(arr.bands.values) & set(s1_bands))

    if len(s1_bands_to_select) == 0:
        return arr

    data_to_rescale = arr.sel(bands=s1_bands_to_select).astype(np.float32).data

    # Assert that the values are set between 1 and 65535
    if data_to_rescale.min().item() < 1 or data_to_rescale.max().item() > 65535:
        raise ValueError(
            "The input array should be in uint16 format, with values between 1 and 65535. "
            "This restriction assures that the data was processed according to the S1 fetcher "
            "preprocessor. The user can disable this scaling manually by setting the "
            "`rescale_s1` parameter to False in the feature extractor."
        )

    # Converting back to power values
    data_to_rescale = 20.0 * np.log10(data_to_rescale) - 83.0
    data_to_rescale = np.power(10, data_to_rescale / 10.0)
    data_to_rescale[~np.isfinite(data_to_rescale)] = np.nan

    # Converting power values to decibels
    data_to_rescale = 10.0 * np.log10(data_to_rescale)

    # Change the bands within the array
    arr.loc[dict(bands=s1_bands_to_select)] = data_to_rescale
    return arr


# Below comes the actual UDF part


# Apply the Feature Extraction UDF
def apply_udf_data(udf_data: UdfData) -> UdfData:
    """This is the actual openeo UDF that will be executed by the backend."""

    cube = udf_data.datacube_list[0]
    parameters = copy.deepcopy(udf_data.user_context)

    proj = udf_data.proj
    if proj is not None:
        proj = proj["EPSG"]

    parameters[EPSG_HARMONIZED_NAME] = proj

    arr = cube.get_array().transpose("bands", "t", "y", "x")

    epsg = parameters.pop(EPSG_HARMONIZED_NAME)

    if parameters.get("rescale_s1", True):
        arr = rescale_s1_backscatter(arr)

    arr = execute(inarr=arr, parameters=parameters, epsg=epsg).transpose(
        "bands", "y", "x"
    )

    cube = XarrayDataCube(arr)

    udf_data.datacube_list = [cube]

    return udf_data


# Change band names
def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    return metadata.rename_labels(dimension="bands", target=get_output_labels())
