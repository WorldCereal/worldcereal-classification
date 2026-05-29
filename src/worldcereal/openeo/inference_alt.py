# /// script
# dependencies = [
#   "onnxruntime>=1.15.1"
# ]
# ///

import sys
from pathlib import Path
import xarray as xr
import numpy as np
import logging
import os
from pathlib import Path   
from typing import Optional, Dict, Tuple
import json
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from openeo.udf.udf_data import UdfData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
base_path = Path(".").resolve()
EPSG_HARMONIZED_NAME = "GEO-EPSG"
CATBOOST_URL = "https://drive.google.com/uc?export=download&id=1YDjYlXF6ulS_7ZNrld_gDou4U7thv7tM"
context = {EPSG_HARMONIZED_NAME: 32631, "ignore_dependencies": True, "classifier_url": CATBOOST_URL}

"""openEO UDF to compute Presto/Prometheo features with clean code structure."""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import requests
import xarray as xr
from openeo.udf.xarraydatacube import XarrayDataCube

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
sys.path.append("onnx_deps")
import onnxruntime as ort  # noqa: E402

_PROMETHEO_INSTALLED = False

Presto = None
load_presto_weights = None
run_model_inference = None
PoolingMethods = None

def get_model_cache():
    """Get or create module-specific cache."""
    if not hasattr(sys, _MODULE_CACHE_KEY):
        setattr(sys, _MODULE_CACHE_KEY, {})
    return getattr(sys, _MODULE_CACHE_KEY)

def load_onnx_model_cached(model_url: str):
    """ONNX loading is fine since it's pure (no side effects)."""

    cache = get_model_cache()
    if model_url in cache:
        logger.debug(f"ONNX model cache hit for {model_url}.")
        return cache[model_url]

    logger.info(f"Loading ONNX model from {model_url}")
    response = requests.get(model_url, timeout=120)

    session_options, providers = optimize_onnx_cpu_performance(NUM_THREADS)

    model = ort.InferenceSession(response.content, session_options, providers=providers)

    metadata = model.get_modelmeta().custom_metadata_map
    class_params = eval(metadata["class_params"], {"__builtins__": None}, {})

    lut = dict(zip(class_params["class_names"], class_params["class_to_label"]))
    sorted_lut = {k: v for k, v in sorted(lut.items(), key=lambda item: item[1])}

    result = (model, sorted_lut)
    cache[model_url] = result
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

def optimize_onnx_cpu_performance(num_threads):
    """CPU-specific ONNX optimizations."""
    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = (
        num_threads  # TODO test setting to 1 due to sequential nature
    )

    # CPU-specific optimizations
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]

    return session_options, providers

import os
import numpy as np
import json
from scipy.interpolate import interp1d 

def interpolate_time_series(X, extreme_threshold=10000):
    N, T, B = X.shape
    X_interp = np.empty_like(X, dtype=float)
    time_idx = np.arange(T)
    for i in range(N):
        for j in range(B):
            y = X[i, :, j].astype(float)
            y[y > extreme_threshold] = np.nan
            valid = ~np.isnan(y)
            if valid.sum() == 0:
                X_interp[i, :, j] = np.nan
            else:
                f = interp1d(time_idx[valid], y[valid], kind="linear",
                            bounds_error=False, fill_value=(y[valid][0], y[valid][-1]))
                X_interp[i, :, j] = f(time_idx)
    return X_interp

def load_folder_data(X):

    bands = ['S2-L2A-B02', 'S2-L2A-B03', 'S2-L2A-B04', 'S2-L2A-B05', 'S2-L2A-B06', 'S2-L2A-B07', 'S2-L2A-B08', 'S2-L2A-B8A', 'S2-L2A-B11', 'S2-L2A-B12', 'S2-L2A-SCL', 'S1-SIGMA0-VV', 'S1-SIGMA0-VH', 'slope', 'elevation', 'AGERA5-TMEAN', 'AGERA5-PRECIP']

    ts_bands_idx = []
    weather_idx = []

    for i, b in enumerate(bands):
        if b.startswith("S2-L2A") or b.startswith("S1-SIGMA0"):
            ts_bands_idx.append(i)
        elif b.startswith("AGERA5") or b in ["slope", "elevation", "lon_x", "lat_x"]:
            weather_idx.append(i)

    # ---- Time-series data
    X_ts = X[:, :, ts_bands_idx]
    ts_bands = [bands[i] for i in ts_bands_idx]

    # ---- Weather data (interpolated per sample)
    X_weather = X[:, :, weather_idx]
    N, T, B_weather = X_weather.shape
    for i in range(N):
        for j in range(B_weather):
            y = X_weather[i, :, j]
            time_idx = np.arange(len(y))
            valid = ~np.isnan(y)
            if valid.sum() == 0:
                X_weather[i, :, j] = np.nan
            else:
                f = interp1d(time_idx[valid], y[valid], kind="linear",
                            bounds_error=False,
                            fill_value=(y[valid][0], y[valid][-1]))
                X_weather[i, :, j] = f(time_idx)
    W = X_weather.copy()
    weather_bands = [bands[i] for i in weather_idx]

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    # Mask by valid time
    # Clean Invalid
    fill_vals = (65530, 65530)
    X = X.astype(np.float32)
    X[X < -9000] = np.nan
    X[X > 65530] = np.nan
    X[np.isin(X, fill_vals)] = np.nan
    # Scale Optical
    # --- Scale optical bands ---
    optical_bands = [f for f in bands if f.startswith("S2-L2A")]
    in_idx = [bands.index(f) for f in bands]
    optical_idx = [in_idx[bands.index(f)] for f in optical_bands]
    X[:, :, optical_idx] /= 10000.0
    # Check S1 Clean zeroes
    # ---- Clean S1 zeros
    for b_idx, band in enumerate(ts_bands):
        if band.startswith("S1-SIGMA0"):
            X[:, :, b_idx][X[:, :, b_idx] == 0] = np.nan
    
    # --------------------------------------------------
    # Interpolate
    # --------------------------------------------------
    X_ts = interpolate_time_series(X_ts)

    # --------------------------------------------------
    # NDVI / NDYI
    # --------------------------------------------------
    idx_red = ts_bands.index("S2-L2A-B04")
    idx_nir = ts_bands.index("S2-L2A-B08")
    idx_green = ts_bands.index("S2-L2A-B03")
    idx_blue = ts_bands.index("S2-L2A-B02")
    idx_swir1 = ts_bands.index("S2-L2A-B11")
    idx_re1 = ts_bands.index("S2-L2A-B05")
    idx_re2 = ts_bands.index("S2-L2A-B06")

    R665 = X_ts[:, :, idx_red]
    R705 = X_ts[:, :, idx_re1]
    R740 = X_ts[:, :, idx_re2]
    R783 = X_ts[:, :, idx_nir]

    ndvi = (X_ts[:, :, idx_nir] - X_ts[:, :, idx_red]) / \
        (X_ts[:, :, idx_nir] + X_ts[:, :, idx_red] + 1e-8)
    ndyi = (X_ts[:, :, idx_green] - X_ts[:, :, idx_blue]) / \
        (X_ts[:, :, idx_green] + X_ts[:, :, idx_blue] + 1e-8)
    gcvi = (X_ts[:, :, idx_nir] / (X_ts[:, :, idx_green] + 1e-8)) - 1
    mndwi = (X_ts[:, :, idx_green] - X_ts[:, :, idx_swir1]) / \
            (X_ts[:, :, idx_green] + X_ts[:, :, idx_swir1] + 1e-8)
    rep = 705 + 35 * (((R665 + R783) / 2 - R705) / (R740 - R705 + 1e-8))

    arrays_to_concat = [X_ts, ndvi[:, :, None], ndyi[:, :, None], gcvi[:, :, None], mndwi[:, :, None], rep[:, :, None]]
    X_ts = np.concatenate(arrays_to_concat, axis=2)
    ts_bands += ["NDVI", "NDYI", "GCVI", "MNDWI", "REP"]
    print("Shapes after processing:", X_ts.shape, W.shape, y.shape)
    
    return X_ts, W, y, ts_bands, weather_bands

def select_bands(X, bands, required):
    idx = [bands.index(b) for b in required if b in bands]
    return X[:, :, idx], [bands[i] for i in idx]

def pad_to_length(X, target_len=24):
    n_samples, seq_len, n_feats = X.shape
    if seq_len == target_len:
        return X
    elif seq_len < target_len:
        X_padded = np.zeros((n_samples, target_len, n_feats), dtype=X.dtype)
        X_padded[:, :seq_len, :] = X
        X_padded[:, seq_len:, :] = X[:, -1:, :]  # repeat last timestep
        return X_padded
    else:  # seq_len > target_len
        return X[:, :target_len, :]

def is_spike(shapelet_data):
    data = np.asarray(shapelet_data)
    if len(data) < 3:
        return False

    max_idx = np.argmax(data)

    if max_idx == 0 or max_idx == len(data)-1:
        return False

    left = data[:max_idx+1]
    if not np.all(np.diff(left) >= 0):
        return False

    right = data[max_idx:]
    if not np.all(np.diff(right) <= 0):
        return False

    return True

class ONNXClassifier:
    """Handles ONNX model inference for classification."""

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

        CATBOOST_URL = "https://drive.google.com/uc?export=download&id=1YDjYlXF6ulS_7ZNrld_gDou4U7thv7tM"
        window_len = 6
        session, lut = load_onnx_model_cached(classifier_url)
        # features_flat = self._prepare_features(features)

        arr = features.transpose('y', 'x', 't', 'bands').values
        logger.info(f"arr_np {arr.shape}")
        arr_shp = arr.shape
        arr = arr.reshape(-1, arr_shp[2], arr_shp[3])  # flatten y*x as first dim
        logger.info(f"final shape {arr.shape}")
        
        crop_bands = {
        "S1-SIGMA0-VH": {"funcs": [is_spike], "min_len": 6, "max_len": 7},
        "NDYI": {"funcs": [is_spike], "min_len": 3, "max_len": 4},
        "MNDWI": {"funcs": [is_spike], "min_len": 6, "max_len": 7},
    }
        
        X, W, y, bands, weather_bands_f = load_folder_data(arr)
        X_sel, bands_sel = select_bands(X, bands, crop_bands)
        X_sel = pad_to_length(X_sel, 24)

        n_samples, T, D = X_sel.shape
        n_windows = T - window_len + 1

        output_labels = get_output_labels(lut)
        window_preds = np.zeros((len(output_labels), n_samples, n_windows), dtype=int)
        final_preds = np.zeros((len(output_labels), n_samples), dtype=int)

        for t0 in range(n_windows):
            windows = X_sel[:, t0:t0+window_len, :]
            windows_flat = windows.reshape(n_samples, -1)
            predictions = self._run_inference(session, lut, windows_flat)
            window_preds[:, :, t0] = predictions

        final_preds[0, :] = (window_preds[0, :, :].sum(axis=1) >= 2).astype(int)
        return self._reshape_predictions(final_preds, features, lut)

    def _prepare_features(self, features: xr.DataArray) -> np.ndarray:
        """Prepare features for inference."""
        return (
            features.transpose("bands", "x", "y")
            .stack(xy=["x", "y"])
            .transpose()
            .values
        )

    def _run_inference(
        self, session: Any, lut: Dict, features: np.ndarray
    ) -> np.ndarray:
        """Run ONNX model inference."""
        # outputs = session.run(None, {"features": features})
        outputs = session.run(None, {"features": features.astype(np.float32)})

        labels = np.zeros(len(outputs[0]), dtype=np.uint16)
        probabilities = np.zeros(len(outputs[0]), dtype=np.uint8)

        for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
            labels[i] = lut[label]
            probabilities[i] = int(round(prob[label] * 100))

        class_probs = np.array(
            [[prob[label] for label in lut.keys()] for prob in outputs[1]]
        )
        class_probs = (class_probs * 100).round().astype(np.uint8)

        return np.hstack([labels[:, None], probabilities[:, None], class_probs]).T

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

def run_single_workflow(
    xarr,
    epsg: int,
    parameters: Dict[str, Any],
    mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Run a single classification workflow with optional masking."""


    logger.info("Onnx classification start ...")
    classifier = ONNXClassifier(parameters["classifier_parameters"])
    y_pred = classifier.predict(xarr)
    logger.info(f"final shape {y_pred.shape}")
    logger.info("Onnx classification Done.")
    return y_pred
  
def apply_udf_data(udf_data: UdfData) -> UdfData:
    """
    Perform cropland mapping using a local execution of Presto models.
    This function is wrapped for use as an OpenEO UDF to handle data cubes.
    
    Parameters:
    -----------
    x : xr.DataArray
        Input data cube with dimensions ('time', 'lat', 'lon', 'bands').
    context : dict
        Context dictionary passed by OpenEO, contains metadata and other details.
    
    Returns:
    --------
    xr.DataArray
        Classified data cube with crop predictions for each pixel.
    """
    input_cube = udf_data.datacube_list[0]
    context = udf_data.user_context.copy()
    epsg = udf_data.proj["EPSG"] if udf_data.proj else None

    if epsg is None:
        raise ValueError("EPSG code not found in projection information")

    xarr = input_cube.get_array()
    band_names = xarr.coords["bands"].values.tolist()
    logger.info(f"BAND NAMES: {band_names}")

    logger.info(f"arr_orr{xarr.shape}")
    if 't' not in xarr.dims:
        xarr = xarr.expand_dims('t')

    output = run_single_workflow(
                    xarr,
                    epsg=32631,
                    parameters={"classifier_parameters":
                        {EPSG_HARMONIZED_NAME: 32631,
                        "ignore_dependencies": True,
                        "classifier_url": CATBOOST_URL}
                    }
                )
    output_cube = XarrayDataCube(output)
    udf_data.datacube_list = [output_cube]
    return udf_data

    # mode = 'inference'
    # regional_crop_graph = {'rapeseed_rape': {'model': context.get('classifier_url', CATBOOST_URL)}}
    # base_path = Path.cwd()

    # for crop, model in regional_crop_graph.items():        
    #     for model_name, model_dict in model.items():
    #         ts_cl = run_single_workflow(
    #                 ts_data_np,
    #                 epsg=32631,
    #                 parameters={"classifier_parameters":
    #                     {EPSG_HARMONIZED_NAME: 32631,
    #                     "ignore_dependencies": True,
    #                     "classifier_url": m['model']}
    #                 }
    #             )
            # Save as xarray
            # return xarray
    # # -----------------------
    # # Run pipeline
    # # -----------------------
    # logger.info("Start Graph")
    # pipeline = CropPipelineGraph(regional_crop_graph)
    # ts_all_cl = []
    # features = np.arange(sliced_arr.shape[2]) # windows, samples, bands
    # logger.info("Start Run")
    # for t in range(sliced_arr.shape[0]): # windows, samples, bands
    #     logger.info("Start Each ts", t)
    #     masks = pipeline.run(timestep=t, spatial_shape=(arr_shape[0], arr_shape[1]), features=features)
    #     ts_all_cl.append(masks)

    # logger.info("End Inf")
    # ts_all_cl = np.stack([np.stack(list(m.values()), axis=0) for m in ts_all_cl], axis=0)
    # logger.info("End Stacking")

    # # Convert result to an xarray DataArray
    # result = xr.DataArray(ts_all_cl, dims=['t', 'y', 'x'],
    #                       coords={'time': np.arange(ts_all_cl.shape[0]),
    #                               'y': arr_orr.coords['y'],
    #                               'x': arr_orr.coords['x']})
    # logger.info("End Xarray")
    # return result