"""openEO UDF to compute Presto/Prometheo features with clean code structure."""

import functools
import logging
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import xarray as xr
import requests
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData
from pyproj import Transformer
from pyproj.crs import CRS
from scipy.ndimage import convolve, zoom
from shapely.geometry import Point
from shapely.ops import transform

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PROMETHEO_WHL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/dependencies/prometheo-0.0.2-py3-none-any.whl"

GFMAP_BAND_MAPPING = {
    "S2-L2A-B02": "B2", "S2-L2A-B03": "B3", "S2-L2A-B04": "B4",
    "S2-L2A-B05": "B5", "S2-L2A-B06": "B6", "S2-L2A-B07": "B7",
    "S2-L2A-B08": "B8", "S2-L2A-B8A": "B8A", "S2-L2A-B11": "B11",
    "S2-L2A-B12": "B12", "S1-SIGMA0-VH": "VH", "S1-SIGMA0-VV": "VV",
    "AGERA5-TMEAN": "temperature_2m", "AGERA5-PRECIP": "total_precipitation"
}

LAT_HARMONIZED_NAME = "GEO-LAT"
LON_HARMONIZED_NAME = "GEO-LON"
EPSG_HARMONIZED_NAME = "GEO-EPSG"

S1_BANDS = ["S1-SIGMA0-VV", "S1-SIGMA0-VH", "S1-SIGMA0-HV", "S1-SIGMA0-HH"]
NODATA_VALUE = 65535

sys.path.append("feature_deps")
sys.path.append("onnx_deps")
import onnxruntime as ort

# =============================================================================
# STANDALONE FUNCTIONS (Work in both apply_udf_data and apply_metadata contexts)
# =============================================================================

@functools.lru_cache(maxsize=1)
def load_and_prepare_model(model_url: str) -> Tuple[Any, Dict]:
    """Load ONNX model and extract metadata - works in both contexts."""
    logger.info(f"Loading ONNX model from {model_url}")
    
    response = requests.get(model_url, timeout=120)
    model = ort.InferenceSession(response.content)
    
    metadata = model.get_modelmeta().custom_metadata_map
    class_params = eval(metadata["class_params"], {"__builtins__": None}, {})
    
    if "class_names" not in class_params or "class_to_label" not in class_params:
        raise ValueError("Invalid model metadata: missing class_names or class_to_label")
    
    lut = dict(zip(class_params["class_names"], class_params["class_to_label"]))
    sorted_lut = {k: v for k, v in sorted(lut.items(), key=lambda item: item[1])}
    
    return model, sorted_lut


def get_output_labels(lut_sorted: dict) -> list:
    """Generate output band names from LUT - works in both contexts."""
    class_names = lut_sorted.keys()
    return ["classification", "probability"] + [
        f"probability_{name}" for name in class_names
    ]


@functools.lru_cache(maxsize=1)
def unpack_prometheo_wheel(wheel_url: str = PROMETHEO_WHL_URL) -> Path:
    """Download and unpack Prometheo wheel - works in both contexts."""
    destination_dir = Path.cwd() / "dependencies" / "prometheo"
    destination_dir.mkdir(exist_ok=True, parents=True)

    modelfile, _ = urllib.request.urlretrieve(wheel_url)
    with zipfile.ZipFile(modelfile, "r") as zip_ref:
        zip_ref.extractall(destination_dir)
    
    return destination_dir


# =============================================================================
# CLASSES (Main logic for apply_udf_data)
# =============================================================================

class SlopeCalculator:
    """Handles slope computation from elevation data."""
    
    @staticmethod
    def compute(resolution: int, elevation_data: np.ndarray) -> np.ndarray:
        """Compute slope from elevation data."""
        dem_arr = SlopeCalculator._prepare_dem_array(elevation_data)
        dem_downsampled = SlopeCalculator._downsample_to_20m(dem_arr, resolution)
        slope = SlopeCalculator._compute_slope_gradient(dem_downsampled)
        return SlopeCalculator._upsample_to_original(slope, dem_arr.shape, resolution)
    
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
    def _downsample_to_20m(dem_arr: np.ndarray, resolution: int) -> np.ndarray:
        """Downsample DEM to 20m resolution for slope computation."""
        factor = int(20 / resolution)
        if factor < 1 or factor % 2 != 0:
            raise ValueError(f"Unsupported resolution for slope: {resolution}")
        
        X, Y = dem_arr.shape
        pad_X, pad_Y = (factor - (X % factor)) % factor, (factor - (Y % factor)) % factor
        padded = np.pad(dem_arr, ((0, pad_X), (0, pad_Y)), mode="reflect")
        
        reshaped = padded.reshape((X + pad_X) // factor, factor, (Y + pad_Y) // factor, factor)
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
    def _upsample_to_original(slope: np.ndarray, original_shape: Tuple[int, int], 
                            resolution: int) -> np.ndarray:
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
            coords={"bands": [LAT_HARMONIZED_NAME, LON_HARMONIZED_NAME],
                   "y": inarr.y, "x": inarr.x}
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
    
    @staticmethod
    def log_array_statistics(arr: xr.DataArray) -> None:
        """Log comprehensive array statistics."""
        total_pixels = arr.size
        values = arr.values
        
        stats = {
            "NaN": np.isnan(values).sum(),
            "Zero": (values == 0).sum(),
            "NODATA": (values == NODATA_VALUE).sum()
        }
        
        logger.info(f"Bands: {', '.join(arr.bands.values)}")
        logger.info(f"Shape: {arr.shape}, Dtype: {arr.dtype}")
        
        for name, count in stats.items():
            percentage = (count / total_pixels) * 100
            logger.info(f"{name} pixels: {count} ({percentage:.2f}%)")
        
        # Log band means
        for band in arr.bands.values:
            mean_val = np.nanmean(arr.sel(bands=band).values)
            logger.info(f"Band '{band}' mean: {mean_val:.2f}")


class PrestoFeatureExtractor:
    """Handles Presto feature extraction pipeline."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    def extract(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Extract Presto features from input array."""
        self._validate_inputs(inarr, epsg)
        inarr = self._preprocess_input(inarr)
        
        if "slope" not in inarr.bands:
            inarr = self._add_slope_band(inarr, epsg)
        
        return self._run_presto_inference(inarr, epsg)
    
    def _validate_inputs(self, inarr: xr.DataArray, epsg: int) -> None:
        """Validate input parameters and array - SIMPLIFIED to only check top level."""
        if epsg is None:
            raise ValueError("EPSG code required for Presto feature extraction")
        
        # ONLY check top level - no nested lookup
        presto_model_url = self.parameters.get('presto_model_url')
        if not presto_model_url:
            logger.error(f"Missing presto_model_url. Available keys: {list(self.parameters.keys())}")
            raise ValueError('Missing required parameter "presto_model_url"')
            
        if len(inarr.t) != 12:
            raise ValueError(f"Presto requires 12 timesteps, got {len(inarr.t)}")
    
    def _preprocess_input(self, inarr: xr.DataArray) -> xr.DataArray:
        """Preprocess input array for Presto."""
        inarr = inarr.transpose("bands", "t", "x", "y")
        
        # Harmonize band names
        new_bands = [GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands]
        inarr = inarr.assign_coords(bands=new_bands)
        
        DataPreprocessor.log_array_statistics(inarr)
        return inarr.fillna(NODATA_VALUE)
    
    def _add_slope_band(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Compute and add slope band to array."""
        logger.warning("Slope band not found, computing...")
        resolution = CoordinateTransformer.get_resolution(inarr.isel(t=0), epsg)
        elevation_data = inarr.sel(bands="elevation").isel(t=0).values
        
        slope_array = SlopeCalculator.compute(resolution, elevation_data)
        slope_da = xr.DataArray(
            slope_array[None, :, :],
            dims=("bands", "y", "x"),
            coords={"bands": ["slope"], "y": inarr.y, "x": inarr.x}
        ).expand_dims({"t": inarr.t}).astype("float32")
        
        return xr.concat([inarr.astype("float32"), slope_da], dim="bands")
    
    def _run_presto_inference(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Run Presto model inference."""
        # Ensure dependencies are available FIRST
        if not self.parameters.get("ignore_dependencies", False):
            deps_dir = unpack_prometheo_wheel(self.parameters.get("prometheo_wheel_url", PROMETHEO_WHL_URL))
            sys.path.insert(0, str(deps_dir))

        from prometheo.datasets.worldcereal import run_model_inference
        from prometheo.models import Presto
        from prometheo.models.pooling import PoolingMethods
        from prometheo.models.presto.wrapper import load_presto_weights
        
        # Get presto_model_url from top level
        presto_model_url = self.parameters['presto_model_url']
        model = load_presto_weights(Presto(), presto_model_url)
        
        pooling_method = PoolingMethods.TIME if self.parameters.get("temporal_prediction") else PoolingMethods.GLOBAL
        
        features = run_model_inference(
            inarr, model, epsg=epsg,
            batch_size=self.parameters.get("batch_size", 256),
            pooling_method=pooling_method
        )
        
        if self.parameters.get("temporal_prediction"):
            features = self._select_temporal_features(features)
        
        return features.transpose("bands", "y", "x")
    
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


# Simplify ONNXClassifier similarly
class ONNXClassifier:
    """Handles ONNX model inference for classification."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    def predict(self, features: xr.DataArray) -> xr.DataArray:
        """Run classification prediction."""
        classifier_url = self.parameters.get('classifier_url')
        if not classifier_url:
            logger.error(f"Missing classifier_url. Available keys: {list(self.parameters.keys())}")
            raise ValueError('Missing required parameter "classifier_url"')
        
        session, lut = load_and_prepare_model(classifier_url)
        features_flat = self._prepare_features(features)
        
        logger.info(f"Classification input shape: {features_flat.shape}")
        predictions = self._run_inference(session, lut, features_flat)
        logger.info(f"Classification output shape: {predictions.shape}")
        
        return self._reshape_predictions(predictions, features, lut)
    
    def _prepare_features(self, features: xr.DataArray) -> np.ndarray:
        """Prepare features for inference."""
        return features.transpose("bands", "x", "y").stack(xy=["x", "y"]).transpose().values
    
    def _run_inference(self, session: Any, lut: Dict, features: np.ndarray) -> np.ndarray:
        """Run ONNX model inference."""
        outputs = session.run(None, {"features": features})
        
        labels = np.zeros(len(outputs[0]), dtype=np.uint16)
        probabilities = np.zeros(len(outputs[0]), dtype=np.uint8)
        
        for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
            labels[i] = lut[label]
            probabilities[i] = int(round(prob[label] * 100))
        
        class_probs = np.array([[prob[label] for label in lut.keys()] for prob in outputs[1]])
        class_probs = (class_probs * 100).round().astype(np.uint8)
        
        return np.hstack([labels[:, None], probabilities[:, None], class_probs]).T
    
    def _reshape_predictions(self, predictions: np.ndarray, 
                           original_features: xr.DataArray, lut: Dict) -> xr.DataArray:
        """Reshape predictions to match original spatial dimensions."""
        output_labels = get_output_labels(lut)
        x_coords, y_coords = original_features.x.values, original_features.y.values
        
        reshaped = predictions.reshape((len(output_labels), len(x_coords), len(y_coords)))
        
        return xr.DataArray(
            reshaped,
            dims=["bands", "x", "y"],
            coords={"bands": output_labels, "x": x_coords, "y": y_coords}
        ).transpose("bands", "y", "x")



# =============================================================================
# MAIN UDF FUNCTIONS
# =============================================================================
def combine_results(croptype_result: xr.DataArray, cropland_mask: xr.DataArray) -> xr.DataArray:
    """Combine crop type results with cropland mask."""
    # Convert cropland mask to uint16
    cropland_mask_uint16 = cropland_mask.astype(np.uint16)
    
    # Stack all bands: classification, probabilities, cropland_mask
    combined_bands = list(croptype_result.bands.values) + ['cropland_mask']
    combined_data = np.concatenate([
        croptype_result.values,
        cropland_mask_uint16.expand_dims('bands').values
    ], axis=0)
    
    return xr.DataArray(
        combined_data,
        dims=["bands", "y", "x"],
        coords={
            "bands": combined_bands,
            "y": croptype_result.y,
            "x": croptype_result.x,
        }
    )
# Update the main UDF function to handle both workflows
def apply_udf_data(udf_data: UdfData) -> UdfData:
    """Main UDF entry point - expects cropland_params and croptype_params in context."""
    input_cube = udf_data.datacube_list[0]
    parameters = udf_data.user_context.copy()
    epsg = udf_data.proj["EPSG"] if udf_data.proj else None
    
    logger.info(f"Processing with EPSG: {epsg}")
    logger.info(f"Received parameter keys: {list(parameters.keys())}")
    
    # Prepare input array
    input_array = input_cube.get_array().transpose("bands", "t", "y", "x")
    
    # Extract both parameter sets directly from context
    cropland_params = parameters.get('cropland_params', {})
    croptype_params = parameters.get('croptype_params', {})
    
    # Check if we have both parameter sets for dual workflow
    if cropland_params and croptype_params:
        logger.info("Running dual workflow: cropland + croptype")
        
        # DEBUG: Log what parameters we actually have
        logger.info(f"Cropland params keys: {list(cropland_params.keys())}")
        logger.info(f"Croptype params keys: {list(croptype_params.keys())}")
        
        # Run cropland classification - pass the FLAT parameters
        logger.info("Running cropland classification...")
        cropland_result = run_single_workflow(input_array, epsg, cropland_params)
        
        # Extract cropland mask
        cropland_mask = cropland_result.sel(bands='classification') > 0
        logger.info(f"Cropland mask: {np.sum(cropland_mask.values)} cropland pixels")
        
        # Run crop type classification with mask
        logger.info("Running crop type classification...")
        croptype_result = run_single_workflow(input_array, epsg, croptype_params, cropland_mask)
        
        # Combine results
        result = combine_results(croptype_result, cropland_mask)
        result_cube = XarrayDataCube(result)
        
    else:
        # Single workflow (fallback to original behavior)
        logger.info("Running single workflow")
        result = run_single_workflow(input_array, epsg, parameters)
        result_cube = XarrayDataCube(result)
    
    udf_data.datacube_list = [result_cube]
    return udf_data


def run_single_workflow(input_array: xr.DataArray, epsg: int, parameters: Dict, mask: Optional[xr.DataArray] = None) -> xr.DataArray:
    """Run a single classification workflow with optional masking."""
    # Apply mask if provided
    if mask is not None and parameters.get('mask_cropland', True):
        input_array = input_array.where(mask, NODATA_VALUE)
        logger.info("Applied cropland mask")
    
    # Preprocess data
    if parameters.get("rescale_s1", True):
        input_array = DataPreprocessor.rescale_s1_backscatter(input_array)
    
    # Extract features
    feature_extractor = PrestoFeatureExtractor(parameters)
    features = feature_extractor.extract(input_array, epsg)
    
    # Classify
    classifier = ONNXClassifier(parameters)
    return classifier.predict(features)


def apply_metadata(metadata, context: Dict) -> Any:
    """Update collection metadata for combined output."""
    try:
        # Use croptype model to get band names, add cropland_mask at the end
        if 'croptype_params' in context:
            classifier_url = context['croptype_params'].get('classifier_url')
        else:
            classifier_url = context.get('classifier_url')
        
        if classifier_url:
            _, lut_sorted = load_and_prepare_model(classifier_url)
            output_labels = get_output_labels(lut_sorted) + ['cropland_mask']
        else:
            raise ValueError("No classifier URL found in context")
            
    except Exception as e:
        logger.warning(f"Could not load model in metadata context: {e}")
        output_labels = ["classification", "probability", "probability_class1", "probability_class2", "cropland_mask"]
    
    return metadata.rename_labels(dimension="bands", target=output_labels)