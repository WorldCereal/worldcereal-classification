"""openEO UDF to compute Presto/Prometheo features with clean code structure."""

import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData
from pyproj import Transformer
from scipy.ndimage import convolve, zoom
from shapely.geometry import Point
from shapely.ops import transform

try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            level = record.levelname
            logger.opt(depth=6).log(level, record.getMessage())

    # Replace existing handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(InterceptHandler())

except ImportError:
    # loguru not available, use standard logging
    logger = logging.getLogger(__name__)

_MODULE_CACHE_KEY = f"__model_cache_{__name__}"

# Constants
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

LAT_HARMONIZED_NAME = "GEO-LAT"
LON_HARMONIZED_NAME = "GEO-LON"
EPSG_HARMONIZED_NAME = "GEO-EPSG"

S1_BANDS = ["S1-SIGMA0-VV", "S1-SIGMA0-VH", "S1-SIGMA0-HV", "S1-SIGMA0-HH"]
NODATA_VALUE = 65535

POSTPROCESSING_EXCLUDED_VALUES = [254, 255, 65535]
POSTPROCESSING_NODATA = 255

NUM_THREADS = 2

sys.path.append("feature_deps")

_PROMETHEO_INSTALLED = False

# Global variables for Prometheo imports
Presto = None
load_presto_weights = None
run_model_inference = None
PoolingMethods = None


# =============================================================================
# STANDALONE FUNCTIONS (Work in both apply_udf_data and apply_metadata contexts)
# =============================================================================
def get_model_cache():
    """Get or create module-specific cache."""
    if not hasattr(sys, _MODULE_CACHE_KEY):
        setattr(sys, _MODULE_CACHE_KEY, {})
    return getattr(sys, _MODULE_CACHE_KEY)


def _ensure_prometheo_dependencies():
    """Non-cached dependency check."""
    global \
        _PROMETHEO_INSTALLED, \
        Presto, \
        load_presto_weights, \
        run_model_inference, \
        PoolingMethods

    if _PROMETHEO_INSTALLED:
        return

    try:
        # Try to import first
        from prometheo.datasets.worldcereal import run_model_inference
        from prometheo.models import Presto
        from prometheo.models.pooling import PoolingMethods
        from prometheo.models.presto.wrapper import load_presto_weights

        # They're now available in the global scope
        _PROMETHEO_INSTALLED = True
        return
    except ImportError:
        pass

    # Installation required
    logger.info("Prometheo not available, installing...")
    _install_prometheo()

    # Import immediately after installation - these will be available globally
    from prometheo.datasets.worldcereal import run_model_inference
    from prometheo.models import Presto
    from prometheo.models.pooling import PoolingMethods
    from prometheo.models.presto.wrapper import load_presto_weights

    optimize_pytorch_cpu_performance(NUM_THREADS)
    _PROMETHEO_INSTALLED = True


def _install_prometheo():
    """Non-cached installation function."""
    import tempfile
    import urllib.request
    import zipfile

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Download wheel
        wheel_path, _ = urllib.request.urlretrieve(PROMETHEO_WHL_URL)

        # Extract to temp directory
        with zipfile.ZipFile(wheel_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Add to Python path
        sys.path.append(str(temp_dir))
        logger.info(f"Prometheo installed to {temp_dir}.")

    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.error(f"Failed to install prometheo: {e}")
        raise


def load_torch_model_cached(model_url: str):
    from urllib.parse import unquote

    import torch
    from torch import nn

    def _download_and_unpack(model_url: str):
        import urllib.request

        extract_dir = Path.cwd() / "tmp" / "models" / Path(model_url).stem
        extract_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Downloading modelfile: {model_url}")
        try:
            modelfile, _ = urllib.request.urlretrieve(
                model_url, filename=extract_dir / Path(model_url).name
            )
        except Exception:
            logger.error(f"Failed to download model from: {modelfile}.")
            raise

        try:
            shutil.unpack_archive(modelfile, extract_dir=extract_dir)
        except Exception:
            logger.error("Failed to extract model archive.")

        return Path(extract_dir)

    class LinearHead(nn.Module):
        def __init__(self, in_dim: int, num_classes: int):
            super().__init__()
            self.fc = nn.Linear(in_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    class MLPHead(nn.Module):
        def __init__(
            self,
            in_dim: int,
            num_classes: int,
            hidden_dim: int = 256,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model_url = unquote(model_url)

    cache = get_model_cache()
    if model_url in cache:
        logger.debug(f"PyTorch model cache hit for {model_url}.")
        return cache[model_url]

    logger.info(f"Loading PyTorch model from {model_url}")
    model_dir = _download_and_unpack(model_url)

    # 1) Load config
    cfg_path = model_dir / "config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    head_type = cfg["head_type"]
    in_dim = int(cfg["in_dim"])
    num_classes = int(cfg["num_classes"])
    hidden_dim = int(cfg.get("hidden_dim", 256))
    dropout = float(cfg.get("dropout", 0.2))

    # 2) Build model
    if head_type == "linear":
        model = LinearHead(in_dim, num_classes)
    elif head_type == "mlp":
        model = MLPHead(in_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    # 4) Load state dict
    pt_path = model_dir / (model_dir.stem + ".pt")
    state = torch.load(pt_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()

    # 5) Build LUT
    lut = dict(zip(cfg["classes_list"], range(len(cfg["classes_list"]))))

    result = (model, lut)
    cache[model_url] = result
    return result


def load_presto_weights_cached(presto_model_url: str):
    """Manual caching for Presto weights with dependency check."""
    cache = get_model_cache()
    if presto_model_url in cache:
        logger.debug(f"Presto model cache hit for {presto_model_url}")
        return cache[presto_model_url]

    # Ensure dependencies are available (not cached)
    _ensure_prometheo_dependencies()

    logger.info(f"Loading Presto weights from: {presto_model_url}")

    model = Presto()  # type: ignore
    result = load_presto_weights(model, presto_model_url)  # type: ignore

    cache[presto_model_url] = result
    return result


def get_output_labels(lut_sorted: dict, postprocess_parameters: dict = {}) -> list:
    """Generate output band names from LUT - works in both contexts.
    Parameters
    ----------
    lut_sorted : dict
        Sorted lookup table mapping class names to labels.
    postprocess_parameters : dict
        Postprocessing parameters to determine whether to keep per-class probability bands.
        If not provided, we assume all probabilities are kept."""

    # Determine whether to remove per-class probability bands
    # based on postprocessing parameters
    postprocessing_enabled = postprocess_parameters.get("enabled", True)
    keep_class_probs = postprocess_parameters.get("keep_class_probs", True)
    if postprocessing_enabled and (not keep_class_probs):
        # Only classification and overall probability
        return ["classification", "probability"]
    else:
        # Include per-class probabilities
        class_names = lut_sorted.keys()
        return ["classification", "probability"] + [
            f"probability_{name}" for name in class_names
        ]


def optimize_pytorch_cpu_performance(num_threads):
    """CPU-specific optimizations for Prometheo."""
    import torch

    # Thread configuration

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(
        num_threads
    )  # TODO test setting to 4 due to parallel slope cal ect
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

    logger.info(f"PyTorch CPU:  using {num_threads} threads")

    # CPU-specific optimizations
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    torch.set_grad_enabled(False)  # Disable gradients for inference

    return num_threads


# =============================================================================
# POSTPROCESSING FUNCTIONS
# =============================================================================


def majority_vote(
    base_labels: xr.DataArray,
    max_probabilities: xr.DataArray,
    kernel_size: int,
) -> xr.DataArray:
    """Majority vote is performed using a sliding local kernel.
    For each pixel, the voting of a final class is done by counting
    neighbours values.
    Pixels that have one of the specified excluded values are
    excluded in the voting process and are unchanged.

    The prediction probabilities are reevaluated by taking, for each pixel,
    the average of probabilities of the neighbors that belong to the winning class.
    (For example, if a pixel was voted to class 2 and there are three
    neighbors of that class, then the new probability is the sum of the
    old probabilities of each pixels divided by 3)

    Parameters
    ----------
    base_labels : xr.DataArray
        The original predicted classification labels.
    max_probabilities : xr.DataArray
        The original probabilities of the winning class (ranging between 0 and 100).
    kernel_size : int
        The size of the kernel used for the neighbour around the pixel.

    Returns
    -------
    xr.DataArray
        The cleaned classification labels and associated probabilities.
    """
    from scipy.signal import convolve2d

    prediction = base_labels.values
    probability = max_probabilities.values

    # As the probabilities are in integers between 0 and 100,
    # we use uint16 matrices to store the vote scores
    assert kernel_size <= 25, (
        f"Kernel value cannot be larger than 25 (currently: {kernel_size}) because it might lead to scenarios where the 16-bit count matrix is overflown"
    )

    # Build a class mapping, so classes are converted to indexes and vice-versa
    unique_values = set(np.unique(prediction))
    unique_values = sorted(unique_values - set(POSTPROCESSING_EXCLUDED_VALUES))  # type: ignore
    index_value_lut = [(k, v) for k, v in enumerate(unique_values)]

    counts = np.zeros(shape=(*prediction.shape, len(unique_values)), dtype=np.uint16)
    probabilities = np.zeros(
        shape=(*probability.shape, len(unique_values)), dtype=np.uint16
    )

    # Iterates for each classes
    for cls_idx, cls_value in index_value_lut:
        # Take the binary mask of the interest class, and multiply by the probabilities
        class_mask = ((prediction == cls_value) * probability).astype(np.uint16)

        # Set to 0 the class scores where the label is excluded
        for excluded_value in POSTPROCESSING_EXCLUDED_VALUES:
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

        probabilities[:, :, cls_idx] = np.divide(counts[:, :, cls_idx], class_voters)

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

        # Get the new probabilities of the predictions
        aggregated_probabilities = np.take_along_axis(
            probabilities,
            aggregated_predictions_indices.reshape(
                *aggregated_predictions_indices.shape, 1
            ),
            axis=2,
        ).squeeze()

        # Check which pixels have a counts value equal to 0
        no_score_mask = np.sum(counts, axis=2) == 0

        # convert back to values from indices
        for cls_idx, cls_value in index_value_lut:
            aggregated_predictions[aggregated_predictions_indices == cls_idx] = (
                cls_value
            )
            aggregated_predictions = aggregated_predictions.astype(np.uint16)

        aggregated_predictions[no_score_mask] = POSTPROCESSING_NODATA
        aggregated_probabilities[no_score_mask] = POSTPROCESSING_NODATA

    # Setting excluded values back to their original values
    for excluded_value in POSTPROCESSING_EXCLUDED_VALUES:
        aggregated_predictions[prediction == excluded_value] = excluded_value
        aggregated_probabilities[prediction == excluded_value] = excluded_value

    return xr.DataArray(
        np.stack((aggregated_predictions, aggregated_probabilities)),
        dims=["bands", "y", "x"],
        coords={
            "bands": ["classification", "probability"],
            "y": base_labels.y,
            "x": base_labels.x,
        },
    )


def smooth_probabilities(
    base_labels: xr.DataArray, class_probabilities: xr.DataArray
) -> xr.DataArray:
    """Performs gaussian smoothing on the class probabilities. Requires the
    base labels to keep the pixels that are excluded away from smoothing.
    """
    from scipy.signal import convolve2d

    base_labels_vals = base_labels.values
    probabilities_vals = class_probabilities.values

    excluded_mask = np.in1d(
        base_labels_vals.reshape(-1),
        POSTPROCESSING_EXCLUDED_VALUES,
    ).reshape(*base_labels_vals.shape)

    conv_kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]], dtype=np.int16)

    for class_idx in range(probabilities_vals.shape[0]):
        probabilities_vals[class_idx] = (
            convolve2d(
                probabilities_vals[class_idx],
                conv_kernel,
                mode="same",
                boundary="symm",
            )
            / conv_kernel.sum()
        )
        probabilities_vals[class_idx][excluded_mask] = 0

    # Sum of probabilities should be 1, cast to uint16
    probabilities_vals = np.round(
        probabilities_vals / probabilities_vals.sum(axis=0) * 100.0
    ).astype("uint16")

    return xr.DataArray(
        probabilities_vals,
        coords=class_probabilities.coords,
        dims=class_probabilities.dims,
    )


def reclassify(
    base_labels: xr.DataArray,
    base_max_probs: xr.DataArray,
    probabilities: xr.DataArray,
) -> xr.DataArray:
    base_labels_vals = base_labels.values
    base_max_probs_vals = base_max_probs.values

    excluded_mask = np.in1d(
        base_labels_vals.reshape(-1),
        POSTPROCESSING_EXCLUDED_VALUES,
    ).reshape(*base_labels_vals.shape)

    new_labels_vals = np.argmax(probabilities.values, axis=0)
    new_max_probs_vals = np.max(probabilities.values, axis=0)

    new_labels_vals[excluded_mask] = base_labels_vals[excluded_mask]
    new_max_probs_vals[excluded_mask] = base_max_probs_vals[excluded_mask]

    return xr.DataArray(
        np.stack((new_labels_vals, new_max_probs_vals)),
        dims=["bands", "y", "x"],
        coords={
            "bands": ["classification", "probability"],
            "y": base_labels.y,
            "x": base_labels.x,
        },
    )


# =============================================================================
# ERROR HANDLING - SIMPLE VERSION
# =============================================================================


def create_nan_output_array(
    inarr: xr.DataArray, num_outputs: int, error_info: str = ""
) -> xr.DataArray:
    """Creates a NaN-filled output array with proper dimensions and coordinates.

    Parameters
    ----------
    inarr : xr.DataArray
        Input array to derive dimensions from
    num_outputs : int
        Number of output bands/classes
    error_info : str
        Error information to include in attributes for debugging

    Returns
    -------
    xr.DataArray
        NaN-filled array with proper structure
    """
    logger.error(f"Creating NaN output array due to error: {error_info}")
    logger.error(f"Input array shape: {inarr.shape}, dims: {inarr.dims}")
    logger.error(
        f"Input array coords - bands: {inarr.bands.values}, t: {len(inarr.t)}, x: {len(inarr.x)}, y: {len(inarr.y)}"
    )

    # Create NaN array with same spatial dimensions
    nan_array = np.full(
        (num_outputs, len(inarr.y), len(inarr.x)), np.nan, dtype=np.float32
    )

    # Create output array with proper coordinates
    output_array = xr.DataArray(
        nan_array,
        dims=["bands", "y", "x"],
        coords={
            "bands": list(range(num_outputs)),
            "y": inarr.y,
            "x": inarr.x,
        },
        attrs={"error": error_info},
    )

    return output_array


# =============================================================================
# CLASSES (Main logic for apply_udf_data)
# =============================================================================


class SlopeCalculator:
    """Handles slope computation from elevation data."""

    @staticmethod
    def compute(resolution: float, elevation_data: np.ndarray) -> np.ndarray:
        """Compute slope from elevation data."""
        dem_arr = SlopeCalculator._prepare_dem_array(elevation_data)
        dem_downsampled = SlopeCalculator._downsample_to_20m(dem_arr, resolution)
        slope = SlopeCalculator._compute_slope_gradient(dem_downsampled)
        result = SlopeCalculator._upsample_to_original(slope, dem_arr.shape, resolution)
        return result

    @staticmethod
    def _prepare_dem_array(dem: np.ndarray) -> np.ndarray:
        """Prepare DEM array by handling NaNs and invalid values."""
        dem_arr = dem.astype(np.float32)
        dem_arr[dem_arr == NODATA_VALUE] = np.nan
        return SlopeCalculator._fill_nans(dem_arr)

    @staticmethod
    def _fill_nans(dem_arr: np.ndarray, max_iter: int = 2) -> np.ndarray:
        """Fill NaN values using rolling fill approach."""
        if max_iter == 0 or not np.any(np.isnan(dem_arr)):
            return dem_arr

        mask = np.isnan(dem_arr)
        roll_params = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(roll_params)

        for roll_param in roll_params:
            rolled = np.roll(dem_arr, roll_param, axis=(0, 1))
            dem_arr[mask] = rolled[mask]

        return SlopeCalculator._fill_nans(dem_arr, max_iter - 1)

    @staticmethod
    def _downsample_to_20m(dem_arr: np.ndarray, resolution: float) -> np.ndarray:
        """Downsample DEM to 20m resolution for slope computation."""
        factor = int(20 / resolution)
        if factor < 1 or factor % 2 != 0:
            raise ValueError(f"Unsupported resolution for slope: {resolution}")

        X, Y = dem_arr.shape
        pad_X, pad_Y = (
            (factor - (X % factor)) % factor,
            (factor - (Y % factor)) % factor,
        )
        padded = np.pad(dem_arr, ((0, pad_X), (0, pad_Y)), mode="reflect")

        reshaped = padded.reshape(
            (X + pad_X) // factor, factor, (Y + pad_Y) // factor, factor
        )
        return np.nanmean(reshaped, axis=(1, 3))

    @staticmethod
    def _compute_slope_gradient(dem: np.ndarray) -> np.ndarray:
        """Compute slope gradient using Sobel operators."""
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8.0 * 20)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8.0 * 20)

        dx = convolve(dem, kernel_x)
        dy = convolve(dem, kernel_y)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        return np.arctan(gradient_magnitude) * (180 / np.pi)

    @staticmethod
    def _upsample_to_original(
        slope: np.ndarray, original_shape: Tuple[int, ...], resolution: float
    ) -> np.ndarray:
        """Upsample slope back to original resolution."""
        factor = int(20 / resolution)
        slope_upsampled = zoom(slope, zoom=factor, order=1)

        # Handle odd dimensions
        if original_shape[0] % 2 != 0:
            slope_upsampled = slope_upsampled[:-1, :]
        if original_shape[1] % 2 != 0:
            slope_upsampled = slope_upsampled[:, :-1]

        return slope_upsampled.astype(np.uint16)


class CoordinateTransformer:
    """Handles coordinate transformations and spatial operations."""

    @staticmethod
    def get_resolution(inarr: xr.DataArray, epsg: int) -> float:
        """Calculate resolution in meters."""
        if epsg == 4326:
            return CoordinateTransformer._get_wgs84_resolution(inarr)
        return abs(inarr.x[1].values - inarr.x[0].values)

    @staticmethod
    def _get_wgs84_resolution(inarr: xr.DataArray) -> float:
        """Convert WGS84 coordinates to meters for resolution calculation."""
        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        points = [Point(x, y) for x, y in zip(inarr.x.values, inarr.y.values)]
        points = [transform(transformer.transform, point) for point in points]
        return abs(points[1].x - points[0].x)

    @staticmethod
    def get_lat_lon_array(inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Create latitude/longitude array from coordinates."""
        lon, lat = np.meshgrid(inarr.x.values, inarr.y.values)

        if epsg != 4326:
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            lon, lat = transformer.transform(lon, lat)

        latlon = np.stack([lat, lon])
        return xr.DataArray(
            latlon,
            dims=["bands", "y", "x"],
            coords={
                "bands": [LAT_HARMONIZED_NAME, LON_HARMONIZED_NAME],
                "y": inarr.y,
                "x": inarr.x,
            },
        )


class DataPreprocessor:
    """Handles data preprocessing operations."""

    @staticmethod
    def rescale_s1_backscatter(arr: xr.DataArray) -> xr.DataArray:
        """Rescale Sentinel-1 backscatter from uint16 to dB values."""
        s1_bands_present = [b for b in S1_BANDS if b in arr.bands.values]
        if not s1_bands_present:
            return arr

        s1_data = arr.sel(bands=s1_bands_present).astype(np.float32)
        DataPreprocessor._validate_s1_data(s1_data.values)

        # Convert to power values then to dB
        power_values = 20.0 * np.log10(s1_data.values) - 83.0
        power_values = np.power(10, power_values / 10.0)
        power_values[~np.isfinite(power_values)] = np.nan

        db_values = 10.0 * np.log10(power_values)
        arr.loc[dict(bands=s1_bands_present)] = db_values

        return arr

    @staticmethod
    def _validate_s1_data(data: np.ndarray) -> None:
        """Validate S1 data meets preprocessing requirements."""
        if data.min() < 1 or data.max() > NODATA_VALUE:
            raise ValueError(
                "S1 data should be uint16 format with values 1-65535. "
                "Set 'rescale_s1' to False to disable scaling."
            )


class PrestoFeatureExtractor:
    """Handles Presto feature extraction pipeline."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    def extract(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Extract Presto features from input array."""
        if epsg is None:
            raise ValueError("EPSG code required for Presto feature extraction")

        # ONLY check top level - no nested lookup
        presto_model_url = self.parameters.get("presto_model_url")
        if not presto_model_url:
            logger.error(
                f"Missing presto_model_url. Available keys: {list(self.parameters.keys())}"
            )
            raise ValueError('Missing required parameter "presto_model_url"')

        if len(inarr.t) != 12:
            error_msg = (
                f"Presto requires exactly 12 timesteps, but got {len(inarr.t)}. "
                f"Available timesteps: {inarr.t.values}. "
                f"Patch coordinates - x: {inarr.x.values.tolist()}, y: {inarr.y.values.tolist()}"
            )
            logger.error(error_msg)

            # Return NaN array instead of crashing
            return create_nan_output_array(
                inarr, self.parameters["num_outputs"], error_msg
            )

        inarr = self._preprocess_input(inarr)

        if "slope" not in inarr.bands:
            inarr = self._add_slope_band(inarr, epsg)

        return self._run_presto_inference(inarr, epsg)

    def _preprocess_input(self, inarr: xr.DataArray) -> xr.DataArray:
        """Preprocess input array for Presto."""
        inarr = inarr.transpose("bands", "t", "x", "y")

        # Harmonize band names
        new_bands = [GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands]
        inarr = inarr.assign_coords(bands=new_bands)

        return inarr.fillna(NODATA_VALUE)

    def _add_slope_band(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Compute and add slope band to array."""
        logger.warning("Slope band not found, computing...")
        resolution = CoordinateTransformer.get_resolution(inarr.isel(t=0), epsg)
        elevation_data = inarr.sel(bands="COP-DEM").isel(t=0).values

        slope_array = SlopeCalculator.compute(resolution, elevation_data)
        slope_da = (
            xr.DataArray(
                slope_array[None, :, :],
                dims=("bands", "y", "x"),
                coords={"bands": ["slope"], "y": inarr.y, "x": inarr.x},
            )
            .expand_dims({"t": inarr.t})
            .astype("float32")
        )

        return xr.concat([inarr.astype("float32"), slope_da], dim="bands")

    def _run_presto_inference(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Run Presto model inference with safe dependency handling."""
        # Dependencies are now handled by load_presto_weights_cached
        import gc

        import torch

        _ensure_prometheo_dependencies()

        presto_model_url = self.parameters["presto_model_url"]

        model = load_presto_weights_cached(presto_model_url)

        # Import here to ensure dependencies are available
        pooling_method = (
            PoolingMethods.TIME  # type: ignore
            if self.parameters.get("temporal_prediction")
            else PoolingMethods.GLOBAL  # type: ignore
        )

        logger.info("Running presto inference ...")
        try:
            with torch.inference_mode():
                features = run_model_inference(
                    inarr,
                    model,
                    epsg=epsg,
                    batch_size=self.parameters.get("batch_size", 256),  # TODO optimize?
                    pooling_method=pooling_method,
                )  # type: ignore
            logger.info("Inference completed.")

            if self.parameters.get("temporal_prediction"):
                features = self._select_temporal_features(features)
            return features.transpose("bands", "y", "x")

        finally:
            gc.collect()

    def _select_temporal_features(self, features: xr.DataArray) -> xr.DataArray:
        """Select specific timestep from temporal features."""
        target_date = self.parameters.get("target_date")

        if target_date is None:
            mid_idx = len(features.t) // 2
            return features.isel(t=mid_idx)

        target_dt = np.datetime64(target_date)
        min_time, max_time = features.t.min().values, features.t.max().values

        if target_dt < min_time or target_dt > max_time:
            raise ValueError(
                f"Target date {target_date} outside feature range: {min_time} to {max_time}"
            )

        return features.sel(t=target_dt, method="nearest")


class TorchClassifier:
    """Handles Pytorch-based model inference for classification."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    def predict(self, features: xr.DataArray) -> xr.DataArray:
        """Run classification prediction."""
        classifier_url = self.parameters.get("classifier_url")
        if not classifier_url:
            logger.error(
                f"Missing classifier_url. Available keys: {list(self.parameters.keys())}"
            )
            raise ValueError('Missing required parameter "classifier_url"')

        model, lut = load_torch_model_cached(classifier_url)
        features_flat = self._prepare_features(features)

        logger.info("Running Torch model inference ...")
        predictions = self._run_inference(model, lut, features_flat)
        logger.info("Torch inference completed.")

        return self._reshape_predictions(predictions, features, lut)

    def _prepare_features(self, features: xr.DataArray) -> np.ndarray:
        """Prepare features for inference."""
        return (
            features.transpose("bands", "x", "y")
            .stack(xy=["x", "y"])
            .transpose()
            .values
        )

    def _run_inference(self, model, lut: Dict, features: np.ndarray) -> np.ndarray:
        """Run Torch model inference."""
        import torch

        with torch.inference_mode():
            inputs = torch.from_numpy(features.astype(np.float32))
            outputs = model(inputs)

            all_probabilities = torch.softmax(outputs, dim=1).numpy()
            labels = np.argmax(all_probabilities, axis=1)

        winning_probabilities = np.round(
            np.max(all_probabilities, axis=1) * 100
        ).astype(np.uint8)
        all_probabilities = np.round(all_probabilities * 100).astype(np.uint8)

        return np.hstack(
            [labels[:, None], winning_probabilities[:, None], all_probabilities]
        ).T

    def _reshape_predictions(
        self, predictions: np.ndarray, original_features: xr.DataArray, lut: Dict
    ) -> xr.DataArray:
        """Reshape predictions to match original spatial dimensions."""
        output_labels = get_output_labels(lut)
        x_coords, y_coords = original_features.x.values, original_features.y.values

        reshaped = predictions.reshape(
            (len(output_labels), len(x_coords), len(y_coords))
        )

        return xr.DataArray(
            reshaped,
            dims=["bands", "x", "y"],
            coords={"bands": output_labels, "x": x_coords, "y": y_coords},
        ).transpose("bands", "y", "x")


class Postprocessor:
    """Handles postprocessing of classification results."""

    def __init__(self, parameters: Dict[str, Any], classifier_url: str):
        self.parameters = parameters
        self.classifier_url = classifier_url

    def apply(self, inarr: xr.DataArray) -> xr.DataArray:
        inarr = inarr.transpose(
            "bands", "y", "x"
        )  # Ensure correct dimension order for openEO backend

        _, lookup_table = load_torch_model_cached(self.classifier_url)

        if self.parameters.get("method") == "smooth_probabilities":
            # Cast to float for more accurate gaussian smoothing
            class_probabilities = (
                inarr.isel(bands=slice(2, None)).astype("float32") / 100.0
            )

            # Peform probability smoothing
            class_probabilities = smooth_probabilities(
                inarr.sel(bands="classification"), class_probabilities
            )

            # Reclassify
            new_labels = reclassify(
                inarr.sel(bands="classification"),
                inarr.sel(bands="probability"),
                class_probabilities,
            )

            # Re-apply labels
            class_labels = list(lookup_table.values())

            # Create a final labels array with same dimensions as new_labels
            final_labels = xr.full_like(new_labels, fill_value=65535)
            for idx, label in enumerate(class_labels):
                final_labels.loc[{"bands": "classification"}] = xr.where(
                    new_labels.sel(bands="classification") == idx,
                    label,
                    final_labels.sel(bands="classification"),
                )
            new_labels.sel(bands="classification").values = final_labels.sel(
                bands="classification"
            ).values

            # Append the per-class probabalities if required
            if self.parameters.get("keep_class_probs", False):
                new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

        elif self.parameters.get("method") == "majority_vote":
            kernel_size = self.parameters.get("kernel_size", 5)

            new_labels = majority_vote(
                inarr.sel(bands="classification"),
                inarr.sel(bands="probability"),
                kernel_size=kernel_size,
            )

            # Append the per-class probabalities if required
            if self.parameters.get("keep_class_probs", False):
                class_probabilities = inarr.isel(bands=slice(2, None))
                new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

        else:
            raise ValueError(
                f"Unknown post-processing method: {self.parameters.get('method')}"
            )

        new_labels = new_labels.transpose(
            "bands", "y", "x"
        )  # Ensure correct dimension order for openEO backend

        return new_labels


# =============================================================================
# MAIN UDF FUNCTIONS
# =============================================================================


def run_single_workflow(
    input_array: xr.DataArray,
    epsg: int,
    parameters: Dict[str, Any],
    mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Run a single classification workflow with optional masking."""

    # Preprocess data
    if parameters["feature_parameters"].get("rescale_s1", True):
        logger.info("Rescale s1 ...")
        input_array = DataPreprocessor.rescale_s1_backscatter(input_array)

    # Extract features
    logger.info("Extract Presto embeddings ...")
    feature_extractor = PrestoFeatureExtractor(parameters["feature_parameters"])
    features = feature_extractor.extract(input_array, epsg)
    logger.info("Presto embedding extraction done.")

    # Classify
    logger.info("Torch classification ...")
    classifier = TorchClassifier(parameters["classifier_parameters"])
    classes = classifier.predict(features)
    logger.info("Torch classification done.")

    # Postprocess
    postprocess_parameters: Dict[str, Any] = parameters.get(
        "postprocess_parameters", {}
    )

    if postprocess_parameters.get("enable"):
        logger.info("Postprocessing classification results ...")
        if postprocess_parameters.get("save_intermediate"):
            classes_raw = classes.assign_coords(
                bands=[f"raw_{b}" for b in list(classes.bands.values)]
            )
        postprocessor = Postprocessor(
            postprocess_parameters,
            classifier_url=parameters.get("classifier_parameters", {}).get(
                "classifier_url"
            ),
        )

        classes = postprocessor.apply(classes)
        if postprocess_parameters.get("save_intermediate"):
            classes = xr.concat([classes, classes_raw], dim="bands")
        logger.info("Postprocessing done.")

    # Set masked areas to specific value
    if mask is not None:
        logger.info("`mask` provided, applying to classification results ...")
        classes = classes.where(mask, 254)  # 254 = non-cropland

    return classes


def combine_results(
    croptype_result: xr.DataArray, cropland_result: xr.DataArray
) -> xr.DataArray:
    """Combine crop type results with ALL cropland classification bands."""

    # Rename cropland bands to avoid conflicts
    cropland_bands_renamed = [
        f"cropland_{band}" for band in cropland_result.bands.values
    ]
    cropland_result = cropland_result.assign_coords(bands=cropland_bands_renamed)

    # Rename croptype bands for clarity
    croptype_bands_renamed = [
        f"croptype_{band}" for band in croptype_result.bands.values
    ]
    croptype_result = croptype_result.assign_coords(bands=croptype_bands_renamed)

    # Combine all bands from both results
    combined_bands = list(croptype_bands_renamed) + list(cropland_bands_renamed)
    combined_data = np.concatenate(
        [croptype_result.values, cropland_result.values], axis=0
    )

    result = xr.DataArray(
        combined_data,
        dims=["bands", "y", "x"],
        coords={
            "bands": combined_bands,
            "y": croptype_result.y,
            "x": croptype_result.x,
        },
    )

    return result


def apply_udf_data(udf_data: UdfData) -> UdfData:
    """Main UDF entry point - expects cropland_params and croptype_params in context."""

    input_cube = udf_data.datacube_list[0]
    parameters = udf_data.user_context.copy()

    epsg = udf_data.proj["EPSG"] if udf_data.proj else None
    if epsg is None:
        raise ValueError("EPSG code not found in projection information")

    # Prepare input array
    input_array = input_cube.get_array().transpose("bands", "t", "y", "x")

    # Extract both parameter sets directly from context
    cropland_params = parameters.get("cropland_params", {})
    croptype_params = parameters.get("croptype_params", {})

    # Check if we have both parameter sets for dual workflow
    if cropland_params and croptype_params:
        logger.info(
            "Running combined workflow: cropland masking + croptype mapping ..."
        )

        # Run cropland classification - pass the FLAT parameters
        logger.info("Running cropland classification ...")
        cropland_result = run_single_workflow(input_array, epsg, cropland_params)
        logger.info("Cropland classification done.")

        # Extract cropland mask for masking the crop type classification
        cropland_mask = cropland_result.sel(bands="classification") > 0

        # Run crop type classification with mask
        logger.info("Running crop type classification ...")
        croptype_result = run_single_workflow(
            input_array, epsg, croptype_params, cropland_mask
        )
        logger.info("Croptype classification done.")

        # Combine ALL bands from both results
        result = combine_results(croptype_result, cropland_result)
        result_cube = XarrayDataCube(result)

    else:
        # Single workflow (fallback to original behavior)
        logger.info("Running single workflow ...")
        result = run_single_workflow(input_array, epsg, parameters)
        result_cube = XarrayDataCube(result)

    udf_data.datacube_list = [result_cube]

    return udf_data


def apply_metadata(metadata, context: Dict) -> Any:
    """Update collection metadata for combined output with ALL bands.

    Band naming logic summary (kept for mapping module resilience):
    - Single workflow (either cropland OR croptype parameters only):
        Base bands: classification, probability, probability_<class>
        If save_intermediate: raw_<band> duplicates are appended.
    - Combined workflow (both croptype_params & cropland_params):
        Prefixed bands: croptype_<band> and cropland_<band>
        If save_intermediate: croptype_raw_<band> and cropland_raw_<band> duplicates appended.

    No renaming occurs here beyond prefixing for the combined workflow; logic in
    mapping.py must therefore accept both prefixed and unprefixed forms.
    """
    try:
        # For dual workflow, combine band names from both models
        if "croptype_params" in context and "cropland_params" in context:
            # Get croptype band names
            croptype_classifier_url = context["croptype_params"][
                "classifier_parameters"
            ].get("classifier_url")
            if croptype_classifier_url:
                _, croptype_lut = load_torch_model_cached(croptype_classifier_url)
                postprocess_parameters = context["croptype_params"].get(
                    "postprocess_parameters", {}
                )
                croptype_bands = [
                    f"croptype_{band}"
                    for band in get_output_labels(croptype_lut, postprocess_parameters)
                ]
                if postprocess_parameters.get("save_intermediate", False):
                    croptype_bands += [
                        band.replace("croptype_", "croptype_raw_")
                        for band in croptype_bands
                    ]
            else:
                raise ValueError("No croptype LUT found")

            # Get cropland band names
            cropland_classifier_url = context["cropland_params"][
                "classifier_parameters"
            ].get("classifier_url")
            if cropland_classifier_url:
                _, cropland_lut = load_torch_model_cached(cropland_classifier_url)
                postprocess_parameters = context["cropland_params"].get(
                    "postprocess_parameters", {}
                )
                cropland_bands = [
                    f"cropland_{band}"
                    for band in get_output_labels(cropland_lut, postprocess_parameters)
                ]
                if postprocess_parameters.get("save_intermediate", False):
                    cropland_bands += [
                        band.replace("cropland_", "cropland_raw_")
                        for band in cropland_bands
                    ]
            else:
                raise ValueError("No cropland LUT found")

            output_labels = croptype_bands + cropland_bands

        else:
            # Single workflow
            classifier_url = context["classifier_parameters"].get("classifier_url")
            if classifier_url:
                _, lut_sorted = load_torch_model_cached(classifier_url)
                postprocess_parameters = context.get("postprocess_parameters", {})
                output_labels = get_output_labels(lut_sorted, postprocess_parameters)
                if postprocess_parameters.get("save_intermediate", False):
                    output_labels += [f"raw_{band}" for band in output_labels]
            else:
                raise ValueError("No classifier URL found in context")

        return metadata.rename_labels(dimension="bands", target=output_labels)

    except Exception as e:
        logger.warning(f"Could not load model in metadata context: {e}")
        return metadata
