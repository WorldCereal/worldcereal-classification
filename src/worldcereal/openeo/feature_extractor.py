"""openEO UDF to compute Presto/Prometheo features."""

import copy
import functools
import logging
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData
from pyproj import Transformer
from scipy.ndimage import (
    convolve,
    zoom,
)

sys.path.append("feature_deps")


import torch  # noqa: E402

PROMETHEO_WHL_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/prometheo-0.0.3-py3-none-any.whl"

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

S1_INPUT_BANDS = ["S1-SIGMA0-VV", "S1-SIGMA0-VH"]
NODATA_VALUE = 65535
LAT_HARMONIZED_NAME = "GEO-LAT"
LON_HARMONIZED_NAME = "GEO-LON"
EPSG_HARMONIZED_NAME = "GEO-EPSG"


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
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


@functools.lru_cache(maxsize=1)
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


def extract_presto_embeddings(
    inarr: xr.DataArray, parameters: dict, epsg: int
) -> xr.DataArray:
    """Executes the feature extraction process on the input array."""

    if epsg is None:
        raise ValueError(
            "EPSG code is required for Presto feature extraction, but was "
            "not correctly initialized."
        )
    if "presto_model_url" not in parameters:
        raise ValueError('Missing required parameter "presto_model_url"')

    presto_model_url_raw = parameters.get("presto_model_url")
    if not isinstance(presto_model_url_raw, str) or not presto_model_url_raw:
        raise ValueError(
            'Parameter "presto_model_url" must be provided as a non-empty string'
        )
    presto_model_url = presto_model_url_raw
    logger.info(f'Loading Presto model from "{presto_model_url}"')
    prometheo_wheel_url = parameters.get("prometheo_wheel_url", PROMETHEO_WHL_URL)
    logger.info(f'Loading Prometheo wheel from "{prometheo_wheel_url}"')

    ignore_dependencies = parameters.get("ignore_dependencies", False)
    if ignore_dependencies:
        logger.info(
            "`ignore_dependencies` flag is set to True. Make sure that "
            "Presto and its dependencies are available on the runtime "
            "environment"
        )

    # The below is required to avoid flipping of the result
    # when running on OpenEO backend!
    inarr = inarr.transpose(
        "bands", "t", "x", "y"
    )  # Presto/Prometheo expects xy dimension order

    # Change the band names
    new_band_names = [GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands]
    inarr = inarr.assign_coords(bands=new_band_names)

    # Log pixel statistics
    total_pixels = inarr.size
    num_nan_pixels = np.isnan(inarr.values).sum()
    num_zero_pixels = (inarr.values == 0).sum()
    num_nodatavalue_pixels = (inarr.values == 65535).sum()
    logger.info("Band names: " + ", ".join(inarr.bands.values))
    logger.debug(
        f"Array dtype: {inarr.dtype}, "
        f"Array size: {inarr.shape}, total pixels: {total_pixels}, "
        f"Pixel statistics: NaN pixels = {num_nan_pixels} "
        f"({num_nan_pixels / total_pixels * 100:.2f}%), "
        f"0 pixels = {num_zero_pixels} "
        f"({num_zero_pixels / total_pixels * 100:.2f}%), "
        f"NODATAVALUE pixels = {num_nodatavalue_pixels} "
        f"({num_nodatavalue_pixels / total_pixels * 100:.2f}%)"
    )

    # Log mean value (ignoring NaNs) per band
    for band in inarr.bands.values:
        band_data = inarr.sel(bands=band).values
        mean_value = np.nanmean(band_data)
        logger.debug(f"Band '{band}': Mean value (ignoring NaNs) = {mean_value:.2f}")

    # Handle NaN values in Presto compatible way
    inarr = inarr.fillna(65535)

    if not ignore_dependencies:
        # Unzip the Presto dependencies on the backend
        logger.info("Unpacking prometheo wheel")
        deps_dir = unpack_prometheo_wheel(prometheo_wheel_url)

        logger.info("Appending dependencies")
        sys.path.append(str(deps_dir))

    batch_size = parameters.get("batch_size", 256)
    temporal_prediction = parameters.get("temporal_prediction", False)
    target_date = parameters.get("target_date", None)
    logger.info(
        (
            f"Extracting Presto features with batch size {batch_size}, "
            f"temporal_prediction={temporal_prediction}, "
            f"target_date={target_date}"
        )
    )

    # TODO: compile_presto not used for now?
    # compile_presto = parameters.get("compile_presto", False)
    # self.logger.info(f"Compile presto: {compile_presto}")

    logger.info("Loading Presto model for inference")

    if presto_model_url.endswith(".zip"):
        from worldcereal.utils.models import load_model_artifact

        # Use load model artifact functionality to get path to model weights file
        model_artifact = load_model_artifact(
            presto_model_url,
            encoder_only=True,
        )
        raw_presto_model_url = model_artifact.checkpoint_path
        presto_model_url = str(raw_presto_model_url)

    # TODO: try to take run_model_inference from worldcereal
    from prometheo.datasets.worldcereal import run_model_inference
    from prometheo.models import Presto
    from prometheo.models.pooling import PoolingMethods
    from prometheo.models.presto.wrapper import load_presto_weights

    presto_model = Presto()
    presto_model = load_presto_weights(presto_model, presto_model_url)

    logger.info("Extracting presto features")
    # Check if we have the expected 12 timesteps
    if len(inarr.t) != 12:
        raise ValueError(f"Can only run Presto on 12 timesteps, got: {len(inarr.t)}")

    # Determine pooling method based on temporal_prediction parameter
    pooling_method = (
        PoolingMethods.TIME if temporal_prediction else PoolingMethods.GLOBAL
    )
    logger.info(f"Using pooling method: {pooling_method}")

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

    features = features.transpose(
        "bands", "y", "x"
    )  # openEO expects yx order after the UDF

    return features


# ---------------------------------------------------------------------------
# DEM helpers reused from the legacy pipeline
# ---------------------------------------------------------------------------


class SlopeCalculator:
    """Utility that computes slope layers from DEM inputs."""

    @staticmethod
    def compute(resolution: float, dem: np.ndarray) -> np.ndarray:
        prepared = SlopeCalculator._prepare_dem_array(dem)
        downsampled = SlopeCalculator._downsample_to_20m(prepared, resolution)
        gradient = SlopeCalculator._compute_slope_gradient(downsampled)
        return SlopeCalculator._upsample_to_original(gradient, dem.shape, resolution)

    @staticmethod
    def _prepare_dem_array(dem: np.ndarray) -> np.ndarray:
        dem_arr = dem.astype(np.float32)
        dem_arr[dem_arr == NODATA_VALUE] = np.nan
        return SlopeCalculator._fill_nans(dem_arr)

    @staticmethod
    def _fill_nans(dem_arr: np.ndarray, max_iter: int = 2) -> np.ndarray:
        if max_iter == 0 or not np.any(np.isnan(dem_arr)):
            return dem_arr

        mask = np.isnan(dem_arr)
        roll_params = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(roll_params)

        for shift_x, shift_y in roll_params:
            rolled = np.roll(dem_arr, shift_x, axis=0)
            rolled = np.roll(rolled, shift_y, axis=1)
            dem_arr[mask] = rolled[mask]

        return SlopeCalculator._fill_nans(dem_arr, max_iter - 1)

    @staticmethod
    def _downsample_to_20m(dem_arr: np.ndarray, resolution: float) -> np.ndarray:
        factor = int(20 / resolution)
        if factor < 1 or factor % 2 != 0:
            raise ValueError(
                f"Unsupported resolution for slope computation: {resolution}"
            )

        x_size, y_size = dem_arr.shape
        pad_x = (factor - (x_size % factor)) % factor
        pad_y = (factor - (y_size % factor)) % factor
        padded = np.pad(dem_arr, ((0, pad_x), (0, pad_y)), mode="reflect")

        reshaped = padded.reshape(
            (x_size + pad_x) // factor, factor, (y_size + pad_y) // factor, factor
        )
        return np.nanmean(reshaped, axis=(1, 3))

    @staticmethod
    def _compute_slope_gradient(dem: np.ndarray) -> np.ndarray:
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8.0 * 20)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8.0 * 20)

        dx = convolve(dem, kernel_x, mode="nearest")
        dy = convolve(dem, kernel_y, mode="nearest")
        gradient = np.sqrt(dx**2 + dy**2)
        return np.arctan(gradient) * (180 / np.pi)

    @staticmethod
    def _upsample_to_original(
        slope: np.ndarray, original_shape: Tuple[int, int], resolution: float
    ) -> np.ndarray:
        factor = int(20 / resolution)
        slope_upsampled = zoom(slope, zoom=factor, order=1)

        if original_shape[0] % 2 != 0:
            slope_upsampled = slope_upsampled[:-1, :]
        if original_shape[1] % 2 != 0:
            slope_upsampled = slope_upsampled[:, :-1]

        return slope_upsampled.astype(np.uint16)


class CoordinateTransformer:
    """Minimal helpers for resolution estimation and coordinate transforms."""

    @staticmethod
    def get_resolution(arr: xr.DataArray, epsg: int) -> float:
        if epsg == 4326:
            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            pts = [
                transformer.transform(arr.x.values[i], arr.y.values[0])
                for i in range(2)
            ]
            return abs(pts[1][0] - pts[0][0])
        return abs(float(arr.x.values[1] - arr.x.values[0]))


class DataPreprocessor:
    """Apply harmonization/rescaling expected by the predictor builder."""

    @staticmethod
    def rescale_s1_backscatter(arr: xr.DataArray) -> xr.DataArray:
        present = [b for b in S1_INPUT_BANDS if b in arr.bands.values]
        if not present:
            return arr
        if not np.issubdtype(arr.dtype, np.floating):
            # Allow negative dB values to be written back safely
            arr = arr.astype("float32")
        s1 = arr.sel(bands=present).astype(np.float32)
        data = s1.values
        nodata_mask = data == NODATA_VALUE
        valid_mask = ~nodata_mask
        scaled = np.full_like(data, NODATA_VALUE, dtype=np.float32)
        if np.any(valid_mask):
            DataPreprocessor._validate_s1_data(data[valid_mask])
            power = 20.0 * np.log10(data[valid_mask]) - 83.0
            power = np.power(10, power / 10.0)
            power[~np.isfinite(power)] = np.nan
            scaled[valid_mask] = 10.0 * np.log10(power)
        arr.loc[{"bands": present}] = scaled
        return arr

    @staticmethod
    def _validate_s1_data(data: np.ndarray) -> None:
        if data.min() < 1 or data.max() > NODATA_VALUE:
            raise ValueError(
                "S1 data expected as uint16 in range 1-65535 before rescaling."
            )

    @staticmethod
    def add_slope_band(arr: xr.DataArray, epsg: int) -> xr.DataArray:
        if "slope" in arr.bands.values:
            return arr
        if "COP-DEM" not in arr.bands.values:
            logger.warning("DEM band missing; slope band cannot be created.")
            return arr
        resolution = CoordinateTransformer.get_resolution(arr.isel(t=0), epsg)
        dem = arr.sel(bands="COP-DEM").isel(t=0).values
        slope = SlopeCalculator.compute(resolution, dem)
        slope_da = (
            xr.DataArray(
                slope[None, :, :],
                dims=("bands", "y", "x"),
                coords={"bands": ["slope"], "y": arr.y, "x": arr.x},
            )
            .expand_dims({"t": arr.t})
            .astype("float32")
        )
        return xr.concat([arr.astype("float32"), slope_da], dim="bands")


def _prepare_array(
    arr: xr.DataArray, epsg: int, rescale_s1: bool = True
) -> xr.DataArray:
    if "bands" not in arr.dims:
        raise ValueError("Input DataArray must expose a 'bands' dimension")
    reordered = arr.transpose("bands", "t", "y", "x")
    if rescale_s1:
        reordered = DataPreprocessor.rescale_s1_backscatter(reordered)
    renamed_bands = [
        GFMAP_BAND_MAPPING.get(str(b), str(b)) for b in reordered.bands.values
    ]
    reordered = reordered.assign_coords(bands=renamed_bands)
    reordered = reordered.transpose("bands", "t", "x", "y")
    reordered = DataPreprocessor.add_slope_band(reordered, epsg)
    return reordered.fillna(NODATA_VALUE).astype(np.float32)


# ---------------------------------------------------------------------------
# openEO UDF integration hooks
# ---------------------------------------------------------------------------


def _require_openeo_runtime() -> None:
    sys.path.insert(0, "feature_deps")
    sys.path.insert(0, "worldcereallib")
    sys.path.insert(0, "prometheolib")

    try:
        import prometheo
        import torch

        import worldcereal

        logger.debug(f"Loading worldcereal from {worldcereal.__file__}")
        logger.debug(f"Loading prometheo from {prometheo.__file__}")
        logger.debug(f"Loading torch from {torch.__file__}")
    except ImportError as exc:
        raise ImportError(
            "openEO UDF seasonal inference requires the worldcereal, prometheo, and loguru packages."
        ) from exc


# Below comes the actual UDF part


# Apply the Feature Extraction UDF
def apply_udf_data(udf_data: UdfData) -> UdfData:
    """This is the actual openeo UDF that will be executed by the backend."""

    _require_openeo_runtime()

    cube = udf_data.datacube_list[0]
    parameters = copy.deepcopy(udf_data.user_context)

    proj = udf_data.proj
    if proj is not None:
        proj = proj["EPSG"]

    parameters[EPSG_HARMONIZED_NAME] = proj

    arr = cube.get_array().transpose("bands", "t", "y", "x")

    epsg = parameters.pop(EPSG_HARMONIZED_NAME)
    logger.info(f"EPSG code determined for feature extraction: {epsg}")

    rescale_s1 = parameters.get("rescale_s1", True)
    prepped = _prepare_array(arr, epsg, rescale_s1)

    arr = extract_presto_embeddings(inarr=prepped, parameters=parameters, epsg=epsg)

    cube = XarrayDataCube(arr)

    udf_data.datacube_list = [cube]

    return udf_data


# Change band names
def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    return metadata.rename_labels(
        dimension="bands", target=[f"presto_ft_{i}" for i in range(128)]
    )
