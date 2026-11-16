"""Helpers and definitions for inference udf."""

# /// script
# dependencies = [
#   "onnxruntime>=1.15.1"
# ]
# ///

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
base_path = Path(".").resolve()
EPSG_HARMONIZED_NAME = "GEO-EPSG"

"""openEO UDF to compute Presto/Prometheo features with clean code structure."""
import logging
from pathlib import Path
from typing import Any

import requests
from openeo.udf.udf_data import UdfData  # noqa: F401  (the openEO backend looks for this name)
from openeo.udf.xarraydatacube import XarrayDataCube

try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    class InterceptHandler(logging.Handler):
        """Handle the operation."""
        def emit(self, record):
            """Handle the operation.

            Args:
                record: Input value for ``record`` used by this operation.
            """
            level = record.levelname
            logger.opt(depth=6).log(level, record.getMessage())

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(InterceptHandler())

except ImportError:
    logger = logging.getLogger(__name__)

_MODULE_CACHE_KEY = f"__model_cache_{__name__}"

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
    """Get or create module-specific cache.

    Returns:
        The value returned by ``getattr``.
    """
    if not hasattr(sys, _MODULE_CACHE_KEY):
        setattr(sys, _MODULE_CACHE_KEY, {})
    return getattr(sys, _MODULE_CACHE_KEY)


def _download_with_retry(url: str, min_bytes: int = 2, attempts: int = 5, timeout: int = 120) -> bytes:
    """Downloads `url` and returns its bytes, retrying with backoff on any
    non-200 reply or network error. Many UDF workers fetch the same file
    at once, and a host can answer with 429/5xx or a short error page;
    before this, that error page was handed straight to onnxruntime and
    surfaced as an unhelpful "INVALID_PROTOBUF". Now a persistent failure
    raises an error that names the URL, the HTTP status and the start of
    the body.

    Args:
        url: Input value for ``url`` used by this operation. Expected type: ``str``.
        min_bytes: Input value for ``min_bytes`` used by this operation. Expected type: ``int``.
        attempts: Input value for ``attempts`` used by this operation. Expected type: ``int``.
        timeout: Input value for ``timeout`` used by this operation. Expected type: ``int``.

    Returns:
        Result of the operation. Expected type: ``bytes``.

    Raises:
        RuntimeError: If the operation cannot complete under the required conditions.
    """
    import random as _random
    import time as _time

    last_problem = "no attempt made"
    for attempt in range(attempts):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and len(resp.content) >= min_bytes:
                return resp.content
            last_problem = f"HTTP {resp.status_code}, {len(resp.content)} bytes, body starts with {resp.content[:60]!r}"
        except Exception as e:  # connection reset, timeout, DNS, ...
            last_problem = f"{type(e).__name__}: {e}"
        logger.warning(f"Download attempt {attempt + 1}/{attempts} failed for {url}: {last_problem}")
        _time.sleep(2**attempt + _random.random())
    raise RuntimeError(f"Could not download {url} after {attempts} attempts. Last problem: {last_problem}")


def load_onnx_model_cached(model_url: str):
    """ONNX loading is fine since it's pure (no side effects).

    Args:
        model_url: Input value for ``model_url`` used by this operation. Expected type: ``str``.

    Returns:
        The computed result.
    """

    cache = get_model_cache()
    if model_url in cache:
        logger.debug(f"ONNX model cache hit for {model_url}.")
        return cache[model_url]

    logger.info(f"Loading ONNX model from {model_url}")
    model_bytes = _download_with_retry(model_url, min_bytes=100)

    session_options, providers = optimize_onnx_cpu_performance(NUM_THREADS)

    model = ort.InferenceSession(model_bytes, session_options, providers=providers)

    metadata = model.get_modelmeta().custom_metadata_map
    class_params = eval(metadata["class_params"], {"__builtins__": None}, {})

    lut = dict(zip(class_params["class_names"], class_params["class_to_label"]))
    sorted_lut = {k: v for k, v in sorted(lut.items(), key=lambda item: item[1])}

    result = (model, sorted_lut)
    cache[model_url] = result
    return result


def get_output_labels(lut_sorted: dict, postprocess_parameters: dict = {}) -> list:
    """Generate output band names from LUT - works in both contexts.

    Args:
        lut_sorted: Input value for ``lut_sorted`` used by this operation. Expected type: ``dict``.
        postprocess_parameters: Input value for ``postprocess_parameters`` used by this operation. Expected type: ``dict``.

    Returns:
        Result of the operation. Expected type: ``list``.
    """
    postprocessing_enabled = postprocess_parameters.get("enabled", True)
    keep_class_probs = postprocess_parameters.get("keep_class_probs", True)
    if postprocessing_enabled and (not keep_class_probs):
        return ["classification", "probability"]
    else:
        class_names = lut_sorted.keys()
        return ["classification", "probability"] + [f"probability_{name}" for name in class_names]


def optimize_pytorch_cpu_performance(num_threads):
    """CPU-specific optimizations for Prometheo.

    Args:
        num_threads: Input value for ``num_threads`` used by this operation.

    Returns:
        The computed num threads.
    """
    import torch

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

    logger.info(f"PyTorch CPU:  using {num_threads} threads")

    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    torch.set_grad_enabled(False)

    return num_threads


def optimize_onnx_cpu_performance(num_threads):
    """CPU-specific ONNX optimizations.

    Args:
        num_threads: Input value for ``num_threads`` used by this operation.

    Returns:
        A tuple containing ``session options``, ``providers``.
    """
    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = num_threads

    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]

    return session_options, providers


def interpolate_time_series(X, extreme_threshold=10000):
    """Handle time series.

    Args:
        X: Input feature array or tensor.
        extreme_threshold: Input value for ``extreme_threshold`` used by this operation.

    Returns:
        The computed X interp.
    """
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
                f = interp1d(
                    time_idx[valid], y[valid], kind="linear", bounds_error=False, fill_value=(y[valid][0], y[valid][-1])
                )
                X_interp[i, :, j] = f(time_idx)
    return X_interp


def load_folder_data(X):

    """Load folder data.

    Args:
        X: Input feature array or tensor.

    Returns:
        A tuple containing ``X ts``, ``W``, ``ts bands``, ``weather bands``.
    """
    bands = [
        "S2-L2A-B02",
        "S2-L2A-B03",
        "S2-L2A-B04",
        "S2-L2A-B05",
        "S2-L2A-B06",
        "S2-L2A-B07",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B11",
        "S2-L2A-B12",
        "S2-L2A-SCL",
        "S1-SIGMA0-VV",
        "S1-SIGMA0-VH",
        "slope",
        "elevation",
        "AGERA5-TMEAN",
        "AGERA5-PRECIP",
    ]

    # Real fix: cleaning/scaling now happens on X FIRST, matching
    # dataloader.py's real order, before X_ts/X_weather get extracted.
    # The old code extracted X_ts as an independent copy (numpy fancy
    # indexing with a list copies, it does not view) BEFORE any of this
    # ran -- meaning NaN-cleaning, the /10000 optical scale, and S1
    # handling were being applied to a `X` that nothing downstream ever
    # used again. X_ts silently carried raw, uncleaned, unscaled values
    # all the way to the model.
    fill_vals = (65530, 65530)
    X = X.astype(np.float32)
    X[X < -9000] = np.nan
    X[X > 65530] = np.nan
    X[np.isin(X, fill_vals)] = np.nan

    optical_idx = [i for i, b in enumerate(bands) if b.startswith("S2-L2A")]
    if optical_idx:
        X[:, :, optical_idx] /= 10000.0

    # Real fix: training converts S1 from raw digital numbers to dB here
    # (10*log10(x/10000+eps)) -- this was completely missing from
    # inference before, meaning S1 features were fed to the model on a
    # totally different scale than what it was trained on. Matches
    # dataloader.py's SCALE SENTINEL-1 step exactly, including the same
    # np.nanmax(x)>1 guard so already-dB data isn't double-converted.
    s1_idx = [i for i, b in enumerate(bands) if b.startswith("S1-SIGMA0")]
    eps = 1e-10
    for idx in s1_idx:
        x = X[:, :, idx]
        if np.nanmax(x) > 1:
            x = np.where(x <= 0, np.nan, x)
            x = 10 * np.log10(x / 10000.0 + eps)
        X[:, :, idx] = x

    ts_bands_idx = []
    weather_idx = []

    for i, b in enumerate(bands):
        if b.startswith("S2-L2A") or b.startswith("S1-SIGMA0"):
            ts_bands_idx.append(i)
        elif b.startswith("AGERA5") or b in ["slope", "elevation", "lon_x", "lat_x"]:
            weather_idx.append(i)

    X_ts = X[:, :, ts_bands_idx]
    ts_bands = [bands[i] for i in ts_bands_idx]

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
                f = interp1d(
                    time_idx[valid], y[valid], kind="linear", bounds_error=False, fill_value=(y[valid][0], y[valid][-1])
                )
                X_weather[i, :, j] = f(time_idx)
    W = X_weather.copy()
    weather_bands = [bands[i] for i in weather_idx]

    # croplogic: the training loader (data/dataloader.py) computes the indices on the UN-filled series and gap-fills / smooths everything afterwards; the older UDF
    # gap-filled the reflectances first. croplogic follows training, so the gap filling is moved below the index calculation.

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

    ndvi = (X_ts[:, :, idx_nir] - X_ts[:, :, idx_red]) / (X_ts[:, :, idx_nir] + X_ts[:, :, idx_red] + 1e-8)
    ndyi = (X_ts[:, :, idx_green] - X_ts[:, :, idx_blue]) / (X_ts[:, :, idx_green] + X_ts[:, :, idx_blue] + 1e-8)
    gcvi = (X_ts[:, :, idx_nir] / (X_ts[:, :, idx_green] + 1e-8)) - 1
    mndwi = (X_ts[:, :, idx_green] - X_ts[:, :, idx_swir1]) / (X_ts[:, :, idx_green] + X_ts[:, :, idx_swir1] + 1e-8)

    # Real fix: MCARI was missing entirely -- one of maize's literature
    # bands, silently absent from every maize inference regardless of
    # whether a model's selected_windows actually included it.
    mcari = ((X_ts[:, :, idx_nir] - X_ts[:, :, idx_re1]) / (X_ts[:, :, idx_nir] + X_ts[:, :, idx_re1] + 1e-8)) / (
        X_ts[:, :, idx_re2] + 1e-8
    )

    rep = 705 + 35 * (((R665 + R783) / 2 - R705) / (R740 - R705 + 1e-8))
    # Real fix: training clips physically implausible REP values to NaN
    # -- this range check was missing from inference entirely.
    rep[(rep < 680) | (rep > 760)] = np.nan

    arrays_to_concat = [
        X_ts,
        ndvi[:, :, None],
        ndyi[:, :, None],
        gcvi[:, :, None],
        mndwi[:, :, None],
        rep[:, :, None],
        mcari[:, :, None],
    ]
    X_ts = np.concatenate(arrays_to_concat, axis=2)
    ts_bands += ["NDVI", "NDYI", "GCVI", "MNDWI", "REP", "MCARI"]
    X_ts = interpolate_time_series(X_ts)

    # Real fix: training smooths the interpolated series with a
    # Savitzky-Golay filter -- the import for this already existed at
    # the top of this file, but the actual call was missing, so
    # inference was feeding noisier, unsmoothed series to the model
    # than what it was ever trained on.
    X_ts = savgol_filter(X_ts, window_length=7, polyorder=2, axis=1, mode="nearest")

    # croplogic: BHDI (Barley Heading Date Index, Ashourloo et al. 2022) = (green - blue) + (green - red), computed on the smoothed reflectances exactly like croplogic/data.py does after the old loader
    _g, _b, _r = (X_ts[:, :, ts_bands.index("S2-L2A-" + k)] for k in ("B03", "B02", "B04"))
    X_ts = np.concatenate([X_ts, ((_g - _b) + (_g - _r))[:, :, None].astype(X_ts.dtype)], axis=2)
    ts_bands.append("BHDI")

    print("Shapes after processing:", X_ts.shape, W.shape)

    return X_ts, W, ts_bands, weather_bands


def select_bands(X, bands, required):
    """Select bands.

    Args:
        X: Input feature array or tensor.
        bands: Band names or band indices to use.
        required: Input value for ``required`` used by this operation.

    Returns:
        A tuple of computed values.
    """
    idx = [bands.index(b) for b in required if b in bands]
    return X[:, :, idx], [bands[i] for i in idx]


def smooth_classification(result: xr.DataArray, kernel: int) -> xr.DataArray:
    """Majority filter on the 'classification' band: a pixel becomes class 1 when at least half of the
    kernel x kernel pixels around it are class 1. kernel <= 1 switches smoothing off.

    apply_neighborhood gives this function a chunk WITH an overlap border of kernel // 2 pixels and
    removes that border from the result afterwards, so pixels at the edge of the visible chunk see
    their real neighbours and there are no seams between chunks. The 'probability' band (the raw vote
    share) is left untouched.

    Args:
        result: Result produced by an earlier operation. Expected type: ``xr.DataArray``.
        kernel: Input value for ``kernel`` used by this operation. Expected type: ``int``.

    Returns:
        Result of the operation. Expected type: ``xr.DataArray``.
    """
    if kernel <= 1:
        return result
    from scipy.ndimage import uniform_filter

    positive = (result.sel(bands="classification").values > 0).astype(np.float32)
    smoothed = uniform_filter(positive, size=kernel, mode="nearest") >= 0.5
    out = result.copy(deep=True)
    out.loc[{"bands": "classification"}] = smoothed.astype(result.dtype)
    return out


def keep_detection_time_for_crop_pixels(result: xr.DataArray) -> xr.DataArray:
    """The 'detection_time' band only has a value where the (cleaned) classification says "target crop";
    everywhere else it is NaN (no data), so it can be shown or averaged without masking by hand.

    Args:
        result: Result produced by an earlier operation. Expected type: ``xr.DataArray``.

    Returns:
        Result of the operation. Expected type: ``xr.DataArray``.
    """
    out = result.copy(deep=True)
    detection_time = out.sel(bands="detection_time").values.copy()
    detection_time[out.sel(bands="classification").values < 1] = np.nan
    out.loc[{"bands": "detection_time"}] = detection_time
    return out


SERIES_LEN = 36  # keep equal to series_len in pipeline_config.json (pipeline.py checks this before any cloud job)


def pad_to_length(X, target_len=SERIES_LEN):
    """Handle to length.

    Args:
        X: Input feature array or tensor.
        target_len: Input value for ``target_len`` used by this operation.

    Returns:
        The computed result of pad to length.
    """
    n_samples, seq_len, n_feats = X.shape
    if seq_len == target_len:
        return X
    elif seq_len < target_len:
        X_padded = np.zeros((n_samples, target_len, n_feats), dtype=X.dtype)
        X_padded[:, :seq_len, :] = X
        X_padded[:, seq_len:, :] = X[:, -1:, :]
        return X_padded
    else:
        return X[:, :target_len, :]


def is_spike(shapelet_data):
    """Report whether spike.

    Args:
        shapelet_data: Input value for ``shapelet_data`` used by this operation.

    Returns:
        The computed result of is spike.
    """
    data = np.asarray(shapelet_data)
    if len(data) < 3:
        return False

    max_idx = np.argmax(data)

    if max_idx == 0 or max_idx == len(data) - 1:
        return False

    left = data[: max_idx + 1]
    if not np.all(np.diff(left) >= 0):
        return False

    right = data[max_idx:]
    if not np.all(np.diff(right) <= 0):
        return False

    return True


def apply_data_cutoff(arr, t_values, cutoff_date):
    """In-season emulation. arr: (pixels, time, bands) raw cube values; t_values: the cube's time labels
    (dekad composites, chronological, each label = first day of its dekad). A dekad counts as COMPLETE on
    cutoff_date when the NEXT dekad starts on or before cutoff_date + 1 day (for the last step: its label
    + 10 days). Every dekad that is not complete is set to NaN BEFORE the gap filling and smoothing, so the
    model sees exactly what a user would have had on that date -- nothing from later observations leaks in.
    Returns (arr, n_available) with n_available = number of leading complete dekads.
    cutoff_date None -> arr unchanged, n_available = number of time steps.

    Args:
        arr: Input value for ``arr`` used by this operation.
        t_values: Input value for ``t_values`` used by this operation.
        cutoff_date: Input value for ``cutoff_date`` used by this operation.

    Returns:
        A tuple containing ``arr``, ``n available``.
    """
    n_steps = arr.shape[1]
    if not cutoff_date:
        return arr, n_steps
    t = np.asarray(t_values).astype("datetime64[D]")
    limit = np.datetime64(str(cutoff_date), "D") + np.timedelta64(1, "D")
    next_start = np.concatenate([t[1:], t[-1:] + np.timedelta64(10, "D")])
    available = next_start <= limit
    arr = arr.copy()
    arr[:, ~available, :] = np.nan
    n_available = int(np.cumprod(available.astype(int)).sum())
    return arr, n_available


def n_valid_windows(n_available, window_len=6, series_len=SERIES_LEN):
    """Windows that lie completely inside the observed part of the (first series_len) steps.

    Args:
        n_available: Input value for ``n_available`` used by this operation.
        window_len: Input value for ``window_len`` used by this operation.
        series_len: Input value for ``series_len`` used by this operation.

    Returns:
        The value returned by ``int``.
    """
    return int(max(min(n_available, series_len) - window_len + 1, 0))


class ONNXClassifier:
    """Handles ONNX model inference for classification."""

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize the instance.

        Args:
            parameters: Input value for ``parameters`` used by this operation. Expected type: ``Dict[str, Any]``.
        """
        self.parameters = parameters

    def predict(self, features: xr.DataArray) -> xr.DataArray:
        """Run classification prediction.

        Args:
            features: Feature values or feature names used by the operation. Expected type: ``xr.DataArray``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.

        Raises:
            ValueError: If the operation cannot complete under the required conditions.
        """
        classifier_url = self.parameters.get("classifier_url")
        if not classifier_url:
            logger.error(f"Missing classifier_url. Available keys: {list(self.parameters.keys())}")
            raise ValueError('Missing required parameter "classifier_url"')

        window_len = 6
        session, lut = load_onnx_model_cached(classifier_url)

        arr = features.transpose("y", "x", "t", "bands").values
        logger.info(f"arr_np {arr.shape}")
        arr_shp = arr.shape
        arr = arr.reshape(-1, arr_shp[2], arr_shp[3])
        logger.info(f"final shape {arr.shape}")

        # In-season emulation: only data that was complete on data_cutoff_date is used (None = whole series)
        data_cutoff_date = self.parameters.get("data_cutoff_date")
        arr, n_available = apply_data_cutoff(arr, features.coords["t"].values, data_cutoff_date)
        if data_cutoff_date:
            logger.info(f"Data cutoff {data_cutoff_date}: {n_available} complete dekads available")

        # This is the WHOLE-PIPELINE version: after fixing
        # 1_train_models.py to actually train on selected_windows'
        # bands (not the full set), inference needs to select the
        # SAME reduced band set, dynamically, since it now genuinely
        # varies per country/crop -- not a hardcoded list, and not
        # "use everything" either. Reads the model's own real
        # selected_windows from model_json_url.
        model_json_url = self.parameters.get("model_json_url")
        if not model_json_url:
            raise ValueError(
                "Missing model_json_url -- required to know which bands this "
                "model was actually trained on. Pass it in the UDF context."
            )
        crop_bands = self._build_crop_bands_from_model_json(model_json_url)
        logger.info(f"Selected bands from model_json_url: {list(crop_bands.keys())}")

        X, W, bands, weather_bands_f = load_folder_data(arr)
        X_sel, bands_sel = select_bands(X, bands, crop_bands)
        X_sel = pad_to_length(X_sel, SERIES_LEN)

        n_samples, T, D = X_sel.shape
        n_windows = T - window_len + 1

        output_labels = get_output_labels(lut)
        window_preds = np.zeros((len(output_labels), n_samples, n_windows), dtype=int)

        # Window range from the model JSON ("window_range", written by scripts/2b_find_window_margin.py): only the windows
        # between the first and last selected position (+ the chosen margin) are predicted. Missing -> every window (old behaviour).
        first_w, last_w = self._window_range_from_model_json(model_json_url, n_windows)
        logger.info(f"Window range: windows {first_w}..{last_w} of {n_windows}")

        # with a cutoff, windows that reach into the unobserved part are not predicted (they stay 0 = no vote)
        for t0 in range(min(n_windows, n_valid_windows(n_available, window_len, SERIES_LEN))):
            if t0 < first_w or t0 > last_w:
                continue
            windows = X_sel[:, t0 : t0 + window_len, :]
            windows_flat = windows.reshape(n_samples, -1)
            predictions = self._run_inference(session, lut, windows_flat)
            window_preds[:, :, t0] = predictions

        # Cleaning that happens right after the model has predicted every window:
        #  votes         = in how many of the n_windows time windows this pixel was classified as the target crop
        #  min_windows   = cut-off: target crop only if at least this many windows say so (1 = the old "any window" rule)
        #  smooth_kernel = majority filter size in pixels (1 = off); needs the matching overlap in apply_neighborhood
        # Both come from the job's context (see inference_job_builder.py); the defaults reproduce the old behaviour.
        min_windows = int(self.parameters.get("min_windows", 1))
        smooth_kernel = int(self.parameters.get("smooth_kernel", 1))
        logger.info(
            f"Cleaning: target crop needs >= {min_windows} of {n_windows} windows, smoothing kernel {smooth_kernel}"
        )

        # which windows said "target crop" for every pixel: (n_samples, n_windows)
        positive = window_preds[0, :, :] > 0
        votes = positive.sum(axis=1)

        # Detection time: the MEDIAN index of the windows in which the crop was detected.
        # Window k starts at time step k (steps are 10-day dekads) and spans steps k..k+5, so index 0 is the
        # first window and n_windows-1 (18) the last. A pixel detected in windows 3 and 5 gets 4.
        # An even number of detections can give a half value (windows 3 and 4 -> 3.5).
        window_index = np.arange(n_windows, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # pixels with no detection: empty slice -> NaN
            detection_time = np.nanmedian(np.where(positive, window_index[None, :], np.nan), axis=1)

        # The three output bands:
        #  1 classification  : 1 = target crop (>= min_windows windows, then smoothed), 0 = not
        #  2 probability     : the RAW vote share in percent (0-100) -- unaffected by the cut-off and the smoothing
        #  3 detection_time  : median window index of the detections; NaN (no data) where the pixel is not the crop
        final_labels = ["classification", "probability", "detection_time"]
        final_preds = np.full((len(final_labels), n_samples), np.nan, dtype=np.float32)
        final_preds[0, :] = votes >= min_windows
        final_preds[1, :] = np.rint(100.0 * votes / n_windows)
        final_preds[2, :] = detection_time
        result = self._reshape_predictions(final_preds, features, lut, labels=final_labels)
        result = smooth_classification(result, smooth_kernel)
        return keep_detection_time_for_crop_pixels(result)

    def _window_range_from_model_json(self, model_json_url, n_windows):
        """(first, last) window index allowed to vote, from the model JSON's 'window_range'; whole series if it has none.

        Args:
            model_json_url: Input value for ``model_json_url`` used by this operation.
            n_windows: Input value for ``n_windows`` used by this operation.

        Returns:
            A tuple containing ``first``, ``last``.
        """
        cache = get_model_cache()
        cache_key = f"window_range::{model_json_url}::{n_windows}"
        if cache_key in cache:
            return cache[cache_key]
        if str(model_json_url).startswith("http"):
            model_json = json.loads(_download_with_retry(model_json_url, min_bytes=2, timeout=60))
        else:
            with open(model_json_url, "r") as f:
                model_json = json.load(f)
        first, last = 0, n_windows - 1
        wr = model_json.get("window_range") if isinstance(model_json, dict) else None
        if wr and wr.get("first_window") is not None and wr.get("last_window") is not None:
            f0, l0 = int(wr["first_window"]), int(wr["last_window"])
            if 0 <= f0 <= l0:
                first, last = f0, min(l0, n_windows - 1)
        cache[cache_key] = (first, last)
        return first, last

    def _build_crop_bands_from_model_json(self, model_json_url):
        """Fetches the deployed model's own saved country model JSON
        (selected_windows) and builds crop_bands from the REAL,
        unique band list it was actually trained on -- this is now
        genuinely correct, since 1_train_models.py has been fixed to
        actually train the window classifiers on this same reduced
        set, not the full band set.

        Args:
            model_json_url: Input value for ``model_json_url`` used by this operation.

        Returns:
            The computed crop bands.

        Raises:
            ValueError: If the operation cannot complete under the required conditions.
        """
        cache = get_model_cache()
        cache_key = f"crop_bands::{model_json_url}"
        if cache_key in cache:
            return cache[cache_key]

        if str(model_json_url).startswith("http"):
            model_json = json.loads(_download_with_retry(model_json_url, min_bytes=2, timeout=60))
        else:
            with open(model_json_url, "r") as f:
                model_json = json.load(f)

        selected_windows = model_json.get("selected_windows", [])
        if not selected_windows:
            raise ValueError(
                f"model_json_url={model_json_url} has no selected_windows -- "
                f"cannot build crop_bands from an empty/invalid model JSON."
            )

        seen = set()
        ordered_bands = []
        for sw in selected_windows:
            band = sw["band"]
            if band not in seen:
                seen.add(band)
                ordered_bands.append(band)

        crop_bands = {band: {"funcs": [is_spike], "min_len": 3, "max_len": 8} for band in ordered_bands}

        cache[cache_key] = crop_bands
        return crop_bands

    def _prepare_features(self, features: xr.DataArray) -> np.ndarray:
        """Prepare features for inference.

        Args:
            features: Feature values or feature names used by the operation. Expected type: ``xr.DataArray``.

        Returns:
            Result of the operation. Expected type: ``np.ndarray``.
        """
        return features.transpose("bands", "x", "y").stack(xy=["x", "y"]).transpose().values

    def _run_inference(self, session: Any, lut: Dict, features: np.ndarray) -> np.ndarray:
        """Run ONNX model inference.

        Args:
            session: Input value for ``session`` used by this operation. Expected type: ``Any``.
            lut: Input value for ``lut`` used by this operation. Expected type: ``Dict``.
            features: Feature values or feature names used by the operation. Expected type: ``np.ndarray``.

        Returns:
            Result of the operation. Expected type: ``np.ndarray``.
        """
        outputs = session.run(None, {"features": features.astype(np.float32)})

        labels = np.zeros(len(outputs[0]), dtype=np.uint16)
        probabilities = np.zeros(len(outputs[0]), dtype=np.uint8)

        for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
            labels[i] = lut[label]
            probabilities[i] = int(round(prob[label] * 100))

        class_probs = np.array([[prob[label] for label in lut.keys()] for prob in outputs[1]])
        class_probs = (class_probs * 100).round().astype(np.uint8)

        return np.hstack([labels[:, None], probabilities[:, None], class_probs]).T

    def _reshape_predictions(
        self, predictions: np.ndarray, original_features: xr.DataArray, lut: Dict, labels=None
    ) -> xr.DataArray:
        """Reshape predictions to match original spatial dimensions.

        Args:
            predictions: Input value for ``predictions`` used by this operation. Expected type: ``np.ndarray``.
            original_features: Input value for ``original_features`` used by this operation. Expected type: ``xr.DataArray``.
            lut: Input value for ``lut`` used by this operation. Expected type: ``Dict``.
            labels: Labels associated with the input samples.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.
        """
        output_labels = labels or get_output_labels(lut)
        x_coords, y_coords = original_features.x.values, original_features.y.values

        reshaped = predictions.reshape((len(output_labels), len(y_coords), len(x_coords)))

        return xr.DataArray(
            reshaped,
            dims=["bands", "y", "x"],
            coords={"bands": output_labels, "y": y_coords, "x": x_coords},
        )


"""openEO UDF to compute Presto/Prometheo features with clean code structure."""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube
from pyproj import Transformer
from scipy.ndimage import convolve, zoom
from shapely.geometry import Point
from shapely.ops import transform

try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    class InterceptHandler(logging.Handler):
        """Handle the operation."""
        def emit(self, record):
            """Handle the operation.

            Args:
                record: Input value for ``record`` used by this operation.
            """
            level = record.levelname
            logger.opt(depth=6).log(level, record.getMessage())

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(InterceptHandler())

except ImportError:
    logger = logging.getLogger(__name__)

_MODULE_CACHE_KEY = f"__model_cache_{__name__}"

PROMETHEO_WHL_URL = "https://s3.waw3-1.cloudferro.com/project_dependencies/worldcereal/prometheo-0.0.5-py3-none-any.whl"

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

_PROMETHEO_INSTALLED = False

Presto = None
load_presto_weights = None
run_model_inference = None
PoolingMethods = None


def get_model_cache():
    """Get or create module-specific cache.

    Returns:
        The value returned by ``getattr``.
    """
    if not hasattr(sys, _MODULE_CACHE_KEY):
        setattr(sys, _MODULE_CACHE_KEY, {})
    return getattr(sys, _MODULE_CACHE_KEY)


def _ensure_prometheo_dependencies():
    """Non-cached dependency check."""
    global _PROMETHEO_INSTALLED, Presto, load_presto_weights, run_model_inference, PoolingMethods

    if _PROMETHEO_INSTALLED:
        return

    try:
        from prometheo.datasets.worldcereal import run_model_inference
        from prometheo.models import Presto
        from prometheo.models.pooling import PoolingMethods
        from prometheo.models.presto.wrapper import load_presto_weights

        _PROMETHEO_INSTALLED = True
        return
    except ImportError:
        pass

    logger.info("Prometheo not available, installing...")
    _install_prometheo()

    from prometheo.datasets.worldcereal import run_model_inference
    from prometheo.models import Presto
    from prometheo.models.pooling import PoolingMethods
    from prometheo.models.presto.wrapper import load_presto_weights

    optimize_pytorch_cpu_performance(NUM_THREADS)
    _PROMETHEO_INSTALLED = True


def _install_prometheo():
    """Non-cached installation function."""
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    temp_dir = Path(tempfile.mkdtemp())
    try:
        wheel_path, _ = urllib.request.urlretrieve(PROMETHEO_WHL_URL)

        with zipfile.ZipFile(wheel_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        sys.path.append(str(temp_dir))
        logger.info(f"Prometheo installed to {temp_dir}.")

    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.error(f"Failed to install prometheo: {e}")
        raise


def load_presto_weights_cached(presto_model_url: str):
    """Manual caching for Presto weights with dependency check.

    Args:
        presto_model_url: Input value for ``presto_model_url`` used by this operation. Expected type: ``str``.

    Returns:
        The computed result.
    """
    cache = get_model_cache()
    if presto_model_url in cache:
        logger.debug(f"Presto model cache hit for {presto_model_url}")
        return cache[presto_model_url]

    _ensure_prometheo_dependencies()

    logger.info(f"Loading Presto weights from: {presto_model_url}")

    model = Presto()  # type: ignore
    result = load_presto_weights(model, presto_model_url)  # type: ignore

    cache[presto_model_url] = result
    return result


def optimize_pytorch_cpu_performance(num_threads):
    """CPU-specific optimizations for Prometheo.

    Args:
        num_threads: Input value for ``num_threads`` used by this operation.

    Returns:
        The computed num threads.
    """
    import torch

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

    logger.info(f"PyTorch CPU:  using {num_threads} threads")

    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    torch.set_grad_enabled(False)

    return num_threads


def optimize_onnx_cpu_performance(num_threads):
    """CPU-specific ONNX optimizations.

    Args:
        num_threads: Input value for ``num_threads`` used by this operation.

    Returns:
        A tuple containing ``session options``, ``providers``.
    """
    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = num_threads

    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]

    return session_options, providers


def majority_vote(
    base_labels: xr.DataArray,
    max_probabilities: xr.DataArray,
    kernel_size: int,
) -> xr.DataArray:
    """Majority vote is performed using a sliding local kernel.

    Args:
        base_labels: Input value for ``base_labels`` used by this operation. Expected type: ``xr.DataArray``.
        max_probabilities: Input value for ``max_probabilities`` used by this operation. Expected type: ``xr.DataArray``.
        kernel_size: Input value for ``kernel_size`` used by this operation. Expected type: ``int``.

    Returns:
        Result of the operation. Expected type: ``xr.DataArray``.
    """
    from scipy.signal import convolve2d

    prediction = base_labels.values
    probability = max_probabilities.values

    assert kernel_size <= 25, (
        f"Kernel value cannot be larger than 25 (currently: {kernel_size}) because it might lead to scenarios where the 16-bit count matrix is overflown"
    )

    unique_values = set(np.unique(prediction))
    unique_values = sorted(unique_values - set(POSTPROCESSING_EXCLUDED_VALUES))  # type: ignore
    index_value_lut = [(k, v) for k, v in enumerate(unique_values)]

    counts = np.zeros(shape=(*prediction.shape, len(unique_values)), dtype=np.uint16)
    probabilities = np.zeros(shape=(*probability.shape, len(unique_values)), dtype=np.uint16)

    for cls_idx, cls_value in index_value_lut:
        class_mask = ((prediction == cls_value) * probability).astype(np.uint16)

        for excluded_value in POSTPROCESSING_EXCLUDED_VALUES:
            class_mask[prediction == excluded_value] = 0

        binary_class_mask = (class_mask > 0).astype(np.uint16)

        kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint16)

        counts[:, :, cls_idx] = convolve2d(class_mask, kernel, mode="same")

        class_voters = convolve2d(binary_class_mask, kernel, mode="same")
        class_voters[class_voters == 0] = 1

        probabilities[:, :, cls_idx] = np.divide(counts[:, :, cls_idx], class_voters)

    aggregated_predictions = np.zeros(shape=(counts.shape[0], counts.shape[1]), dtype=np.uint16)
    aggregated_probabilities = np.zeros(shape=(counts.shape[0], counts.shape[1]), dtype=np.uint16)

    if len(unique_values) > 0:
        aggregated_predictions_indices = np.argmax(counts, axis=2)

        aggregated_probabilities = np.take_along_axis(
            probabilities,
            aggregated_predictions_indices.reshape(*aggregated_predictions_indices.shape, 1),
            axis=2,
        ).squeeze()

        no_score_mask = np.sum(counts, axis=2) == 0

        for cls_idx, cls_value in index_value_lut:
            aggregated_predictions[aggregated_predictions_indices == cls_idx] = cls_value
            aggregated_predictions = aggregated_predictions.astype(np.uint16)

        aggregated_predictions[no_score_mask] = POSTPROCESSING_NODATA
        aggregated_probabilities[no_score_mask] = POSTPROCESSING_NODATA

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


def smooth_probabilities(base_labels: xr.DataArray, class_probabilities: xr.DataArray) -> xr.DataArray:
    """Performs gaussian smoothing on the class probabilities.

    Args:
        base_labels: Input value for ``base_labels`` used by this operation. Expected type: ``xr.DataArray``.
        class_probabilities: Input value for ``class_probabilities`` used by this operation. Expected type: ``xr.DataArray``.

    Returns:
        Result of the operation. Expected type: ``xr.DataArray``.
    """
    from scipy.signal import convolve2d

    base_labels_vals = base_labels.values
    probabilities_vals = class_probabilities.values

    excluded_mask = np.isin(
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

    probabilities_vals = np.round(probabilities_vals / probabilities_vals.sum(axis=0) * 100.0).astype("uint16")

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
    """Handle the operation.

    Args:
        base_labels: Input value for ``base_labels`` used by this operation. Expected type: ``xr.DataArray``.
        base_max_probs: Input value for ``base_max_probs`` used by this operation. Expected type: ``xr.DataArray``.
        probabilities: Input value for ``probabilities`` used by this operation. Expected type: ``xr.DataArray``.

    Returns:
        Result of the operation. Expected type: ``xr.DataArray``.
    """
    base_labels_vals = base_labels.values
    base_max_probs_vals = base_max_probs.values

    excluded_mask = np.isin(
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


def create_nan_output_array(inarr: xr.DataArray, num_outputs: int, error_info: str = "") -> xr.DataArray:
    """Creates a NaN-filled output array with proper dimensions and coordinates.

    Args:
        inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.
        num_outputs: Input value for ``num_outputs`` used by this operation. Expected type: ``int``.
        error_info: Input value for ``error_info`` used by this operation. Expected type: ``str``.

    Returns:
        Result of the operation. Expected type: ``xr.DataArray``.
    """
    logger.error(f"Creating NaN output array due to error: {error_info}")
    logger.error(f"Input array shape: {inarr.shape}, dims: {inarr.dims}")
    logger.error(
        f"Input array coords - bands: {inarr.bands.values}, t: {len(inarr.t)}, x: {len(inarr.x)}, y: {len(inarr.y)}"
    )

    nan_array = np.full((num_outputs, len(inarr.y), len(inarr.x)), np.nan, dtype=np.float32)

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


class SlopeCalculator:
    """Handles slope computation from elevation data."""

    @staticmethod
    def compute(resolution: float, elevation_data: np.ndarray) -> np.ndarray:
        """Compute slope from elevation data.

        Args:
            resolution: Input value for ``resolution`` used by this operation. Expected type: ``float``.
            elevation_data: Input value for ``elevation_data`` used by this operation. Expected type: ``np.ndarray``.

        Returns:
            Result of the operation. Expected type: ``np.ndarray``.
        """
        dem_arr = SlopeCalculator._prepare_dem_array(elevation_data)
        dem_downsampled = SlopeCalculator._downsample_to_20m(dem_arr, resolution)
        slope = SlopeCalculator._compute_slope_gradient(dem_downsampled)
        result = SlopeCalculator._upsample_to_original(slope, dem_arr.shape, resolution)
        return result

    @staticmethod
    def _prepare_dem_array(dem: np.ndarray) -> np.ndarray:
        """Prepare DEM array by handling NaNs and invalid values.

        Args:
            dem: Input value for ``dem`` used by this operation. Expected type: ``np.ndarray``.

        Returns:
            Result of the operation. Expected type: ``np.ndarray``.
        """
        dem_arr = dem.astype(np.float32)
        dem_arr[dem_arr == NODATA_VALUE] = np.nan
        return SlopeCalculator._fill_nans(dem_arr)

    @staticmethod
    def _fill_nans(dem_arr: np.ndarray, max_iter: int = 2) -> np.ndarray:
        """Fill NaN values using rolling fill approach.

        Args:
            dem_arr: Input value for ``dem_arr`` used by this operation. Expected type: ``np.ndarray``.
            max_iter: Input value for ``max_iter`` used by this operation. Expected type: ``int``.

        Returns:
            Result of the operation. Expected type: ``np.ndarray``.
        """
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
        """Downsample DEM to 20m resolution for slope computation.

        Args:
            dem_arr: Input value for ``dem_arr`` used by this operation. Expected type: ``np.ndarray``.
            resolution: Input value for ``resolution`` used by this operation. Expected type: ``float``.

        Returns:
            Result of the operation. Expected type: ``np.ndarray``.

        Raises:
            ValueError: If the operation cannot complete under the required conditions.
        """
        factor = int(20 / resolution)
        if factor < 1 or factor % 2 != 0:
            raise ValueError(f"Unsupported resolution for slope: {resolution}")

        X, Y = dem_arr.shape
        pad_X, pad_Y = (
            (factor - (X % factor)) % factor,
            (factor - (Y % factor)) % factor,
        )
        padded = np.pad(dem_arr, ((0, pad_X), (0, pad_Y)), mode="reflect")

        reshaped = padded.reshape((X + pad_X) // factor, factor, (Y + pad_Y) // factor, factor)
        return np.nanmean(reshaped, axis=(1, 3))

    @staticmethod
    def _compute_slope_gradient(dem: np.ndarray) -> np.ndarray:
        """Compute slope gradient using Sobel operators.

        Args:
            dem: Input value for ``dem`` used by this operation. Expected type: ``np.ndarray``.

        Returns:
            Result of the operation. Expected type: ``np.ndarray``.
        """
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8.0 * 20)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8.0 * 20)

        dx = convolve(dem, kernel_x)
        dy = convolve(dem, kernel_y)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        return np.arctan(gradient_magnitude) * (180 / np.pi)

    @staticmethod
    def _upsample_to_original(slope: np.ndarray, original_shape: Tuple[int, ...], resolution: float) -> np.ndarray:
        """Upsample slope back to original resolution.

        Args:
            slope: Input value for ``slope`` used by this operation. Expected type: ``np.ndarray``.
            original_shape: Input value for ``original_shape`` used by this operation. Expected type: ``Tuple[int, ...]``.
            resolution: Input value for ``resolution`` used by this operation. Expected type: ``float``.

        Returns:
            Result of the operation. Expected type: ``np.ndarray``.
        """
        factor = int(20 / resolution)
        slope_upsampled = zoom(slope, zoom=factor, order=1)

        if original_shape[0] % 2 != 0:
            slope_upsampled = slope_upsampled[:-1, :]
        if original_shape[1] % 2 != 0:
            slope_upsampled = slope_upsampled[:, :-1]

        return slope_upsampled.astype(np.uint16)


class CoordinateTransformer:
    """Handles coordinate transformations and spatial operations."""

    @staticmethod
    def get_resolution(inarr: xr.DataArray, epsg: int) -> float:
        """Calculate resolution in meters.

        Args:
            inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.
            epsg: Input value for ``epsg`` used by this operation. Expected type: ``int``.

        Returns:
            Result of the operation. Expected type: ``float``.
        """
        if epsg == 4326:
            return CoordinateTransformer._get_wgs84_resolution(inarr)
        return abs(inarr.x[1].values - inarr.x[0].values)

    @staticmethod
    def _get_wgs84_resolution(inarr: xr.DataArray) -> float:
        """Convert WGS84 coordinates to meters for resolution calculation.

        Args:
            inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.

        Returns:
            Result of the operation. Expected type: ``float``.
        """
        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        points = [Point(x, y) for x, y in zip(inarr.x.values, inarr.y.values)]
        points = [transform(transformer.transform, point) for point in points]
        return abs(points[1].x - points[0].x)

    @staticmethod
    def get_lat_lon_array(inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Create latitude/longitude array from coordinates.

        Args:
            inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.
            epsg: Input value for ``epsg`` used by this operation. Expected type: ``int``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.
        """
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
        """Rescale Sentinel-1 backscatter from uint16 to dB values.

        Args:
            arr: Input value for ``arr`` used by this operation. Expected type: ``xr.DataArray``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.
        """
        s1_bands_present = [b for b in S1_BANDS if b in arr.bands.values]
        if not s1_bands_present:
            return arr

        s1_data = arr.sel(bands=s1_bands_present).astype(np.float32)
        DataPreprocessor._validate_s1_data(s1_data.values)

        power_values = 20.0 * np.log10(s1_data.values) - 83.0
        power_values = np.power(10, power_values / 10.0)
        power_values[~np.isfinite(power_values)] = np.nan

        db_values = 10.0 * np.log10(power_values)
        arr.loc[dict(bands=s1_bands_present)] = db_values

        return arr

    @staticmethod
    def _validate_s1_data(data: np.ndarray) -> None:
        """Validate S1 data meets preprocessing requirements.

        Args:
            data: Input data to process. Expected type: ``np.ndarray``.

        Raises:
            ValueError: If the operation cannot complete under the required conditions.
        """
        if data.min() < 1 or data.max() > NODATA_VALUE:
            raise ValueError(
                "S1 data should be uint16 format with values 1-65535. Set 'rescale_s1' to False to disable scaling."
            )


class PrestoFeatureExtractor:
    """Handles Presto feature extraction pipeline."""

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize the instance.

        Args:
            parameters: Input value for ``parameters`` used by this operation. Expected type: ``Dict[str, Any]``.
        """
        self.parameters = parameters

    def extract(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Extract Presto features from input array.

        Args:
            inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.
            epsg: Input value for ``epsg`` used by this operation. Expected type: ``int``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.

        Raises:
            ValueError: If the operation cannot complete under the required conditions.
        """
        if epsg is None:
            raise ValueError("EPSG code required for Presto feature extraction")

        presto_model_url = self.parameters.get("presto_model_url")
        if not presto_model_url:
            logger.error(f"Missing presto_model_url. Available keys: {list(self.parameters.keys())}")
            raise ValueError('Missing required parameter "presto_model_url"')

        if len(inarr.t) != 12:
            error_msg = (
                f"Presto requires exactly 12 timesteps, but got {len(inarr.t)}. "
                f"Available timesteps: {inarr.t.values}. "
                f"Patch coordinates - x: {inarr.x.values.tolist()}, y: {inarr.y.values.tolist()}"
            )
            logger.error(error_msg)

            return create_nan_output_array(inarr, self.parameters["num_outputs"], error_msg)

        inarr = self._preprocess_input(inarr)

        if "slope" not in inarr.bands:
            inarr = self._add_slope_band(inarr, epsg)

        return self._run_presto_inference(inarr, epsg)

    def _preprocess_input(self, inarr: xr.DataArray) -> xr.DataArray:
        """Preprocess input array for Presto.

        Args:
            inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.
        """
        inarr = inarr.transpose("bands", "t", "x", "y")

        new_bands = [GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands]
        inarr = inarr.assign_coords(bands=new_bands)

        return inarr.fillna(NODATA_VALUE)

    def _add_slope_band(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Compute and add slope band to array.

        Args:
            inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.
            epsg: Input value for ``epsg`` used by this operation. Expected type: ``int``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.
        """
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
        """Run Presto model inference with safe dependency handling.

        Args:
            inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.
            epsg: Input value for ``epsg`` used by this operation. Expected type: ``int``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.
        """
        import gc

        import torch

        _ensure_prometheo_dependencies()

        presto_model_url = self.parameters["presto_model_url"]

        model = load_presto_weights_cached(presto_model_url)

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
                    batch_size=self.parameters.get("batch_size", 256),
                    pooling_method=pooling_method,
                )  # type: ignore
            logger.info("Inference completed.")

            if self.parameters.get("temporal_prediction"):
                features = self._select_temporal_features(features)
            return features.transpose("bands", "y", "x")

        finally:
            gc.collect()

    def _select_temporal_features(self, features: xr.DataArray) -> xr.DataArray:
        """Select specific timestep from temporal features.

        Args:
            features: Feature values or feature names used by the operation. Expected type: ``xr.DataArray``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.

        Raises:
            ValueError: If the operation cannot complete under the required conditions.
        """
        target_date = self.parameters.get("target_date")

        if target_date is None:
            mid_idx = len(features.t) // 2
            return features.isel(t=mid_idx)

        target_dt = np.datetime64(target_date)
        min_time, max_time = features.t.min().values, features.t.max().values

        if target_dt < min_time or target_dt > max_time:
            raise ValueError(f"Target date {target_date} outside feature range: {min_time} to {max_time}")

        return features.sel(t=target_dt, method="nearest")


class Postprocessor:
    """Handles postprocessing of classification results."""

    def __init__(self, parameters: Dict[str, Any], classifier_url: str):
        """Initialize the instance.

        Args:
            parameters: Input value for ``parameters`` used by this operation. Expected type: ``Dict[str, Any]``.
            classifier_url: Input value for ``classifier_url`` used by this operation. Expected type: ``str``.
        """
        self.parameters = parameters
        self.classifier_url = classifier_url

    def apply(self, inarr: xr.DataArray) -> xr.DataArray:
        """Apply the operation.

        Args:
            inarr: Input value for ``inarr`` used by this operation. Expected type: ``xr.DataArray``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.

        Raises:
            ValueError: If the operation cannot complete under the required conditions.
        """
        inarr = inarr.transpose("bands", "y", "x")

        _, lookup_table = load_onnx_model_cached(self.classifier_url)

        if self.parameters.get("method") == "smooth_probabilities":
            class_probabilities = inarr.isel(bands=slice(2, None)).astype("float32") / 100.0

            class_probabilities = smooth_probabilities(inarr.sel(bands="classification"), class_probabilities)

            new_labels = reclassify(
                inarr.sel(bands="classification"),
                inarr.sel(bands="probability"),
                class_probabilities,
            )

            class_labels = list(lookup_table.values())

            final_labels = xr.full_like(new_labels, fill_value=65535)
            for idx, label in enumerate(class_labels):
                final_labels.loc[{"bands": "classification"}] = xr.where(
                    new_labels.sel(bands="classification") == idx,
                    label,
                    final_labels.sel(bands="classification"),
                )
            new_labels.sel(bands="classification").values = final_labels.sel(bands="classification").values

            if self.parameters.get("keep_class_probs", False):
                new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

        elif self.parameters.get("method") == "majority_vote":
            kernel_size = self.parameters.get("kernel_size", 5)

            new_labels = majority_vote(
                inarr.sel(bands="classification"),
                inarr.sel(bands="probability"),
                kernel_size=kernel_size,
            )

            if self.parameters.get("keep_class_probs", False):
                class_probabilities = inarr.isel(bands=slice(2, None))
                new_labels = xr.concat([new_labels, class_probabilities], dim="bands")

        else:
            raise ValueError(f"Unknown post-processing method: {self.parameters.get('method')}")

        new_labels = new_labels.transpose("bands", "y", "x")

        return new_labels


# ====================================================================================================================================================
# croplogic classifier: a multi-class windowed CatBoost model exported to ONNX. Same recipe as croplogic/models.py: every valid 6-dekad window of a pixel is
# one example (features = the window's selected bands, time-major, flattened); the crop is detected by the unit's tuned rule:
#   count     >= param windows (of all valid windows) are predicted as the crop
#   fraction  >= param of the crop's SELECTED windows are predicted as the crop
#   prob      the mean crop probability over the crop's SELECTED windows >= param
# The settings arrive in the UDF context under "croplogic" (written by scripts/deploy/17_openeo_package.py): bands, window_len, class_index, rule, param, sel_starts, valid_starts.
# ====================================================================================================================================================
def load_onnx_session_cached(model_url: str):
    """Load onnx session cached.

    Args:
        model_url: Input value for ``model_url`` used by this operation. Expected type: ``str``.

    Returns:
        The computed result of load onnx session cached.
    """
    cache = get_model_cache()
    key = "croplogic::" + str(model_url)
    if key in cache:
        return cache[key]
    model_bytes = _download_with_retry(model_url, min_bytes=100)
    session_options, providers = optimize_onnx_cpu_performance(NUM_THREADS)
    cache[key] = ort.InferenceSession(model_bytes, session_options, providers=providers)
    return cache[key]


class ONNXClassifierWindowed:
    """Handle the operation."""
    def __init__(self, parameters: Dict[str, Any]):
        """Initialize the instance.

        Args:
            parameters: Input value for ``parameters`` used by this operation. Expected type: ``Dict[str, Any]``.
        """
        self.parameters = parameters

    @staticmethod
    def _proba(session, w):
        """Handle the operation.

        Args:
            session: Input value for ``session`` used by this operation.
            w: Input value for ``w`` used by this operation.

        Returns:
            The input values as a NumPy array.
        """
        out = session.run(None, {session.get_inputs()[0].name: w.astype(np.float32)})
        p = out[1]
        if isinstance(p, list):  # zipmap output: [{class: prob}, ...]
            keys = sorted(p[0].keys())
            return np.array([[d[k] for k in keys] for d in p], dtype=np.float32)
        return np.asarray(p, dtype=np.float32)

    def predict(self, features: xr.DataArray) -> xr.DataArray:
        """Predict the operation.

        Args:
            features: Feature values or feature names used by the operation. Expected type: ``xr.DataArray``.

        Returns:
            Result of the operation. Expected type: ``xr.DataArray``.

        Raises:
            ValueError: If the operation cannot complete under the required conditions.
        """
        cfg = self.parameters.get("croplogic")
        if not cfg:
            raise ValueError('Missing "croplogic" settings in the UDF context')
        session = load_onnx_session_cached(self.parameters["classifier_url"])
        arr = features.transpose("y", "x", "t", "bands").values
        ny, nx = arr.shape[0], arr.shape[1]
        arr = arr.reshape(-1, arr.shape[2], arr.shape[3])
        X, W, bands, _ = load_folder_data(arr)
        missing = [b for b in cfg["bands"] if b not in bands]
        if missing:
            raise ValueError(f"bands {missing} are not in the processed cube ({bands})")
        X_sel = pad_to_length(X[:, :, [bands.index(b) for b in cfg["bands"]]], SERIES_LEN)
        n, L = X_sel.shape[0], int(cfg["window_len"])
        n_windows = SERIES_LEN - L + 1
        # one crop (old settings) or several crops of the same model ("crops": [{name, class_index, rule, param, sel_starts, valid_starts}, ...]): the model is run once per window
        specs = cfg.get("crops") or [
            {
                "name": None,
                "class_index": cfg["class_index"],
                "rule": cfg["rule"],
                "param": cfg["param"],
                "sel_starts": cfg["sel_starts"],
                "valid_starts": cfg.get("valid_starts"),
            }
        ]
        for sp in specs:
            sp["_valid"] = [int(v) for v in (sp.get("valid_starts") or range(n_windows)) if 0 <= int(v) < n_windows]
        cache = {}
        for t0 in sorted(set(v for sp in specs for v in sp["_valid"])):
            w = X_sel[:, t0 : t0 + L, :].reshape(n, -1)
            ok = ~np.isnan(w).any(axis=1)  # a window with a missing value is not used (as in training)
            if ok.any():
                cache[t0] = (np.where(ok)[0], self._proba(session, w[ok]))
        results = []
        for sp in specs:
            k = int(sp["class_index"])
            rule = sp["rule"]
            param = float(sp["param"])
            sel = set(int(v) for v in sp["sel_starts"])
            votes = np.zeros(n, int)
            sel_n = np.zeros(n, int)
            sel_votes = np.zeros(n, int)
            sel_psum = np.zeros(n)
            n_valid = np.zeros(n, int)
            hit_pos = np.zeros((n, n_windows), bool)
            for t0 in sp["_valid"]:
                if t0 not in cache:
                    continue
                idx, proba = cache[t0]
                hit = proba.argmax(axis=1) == k
                n_valid[idx] += 1
                votes[idx[hit]] += 1
                hit_pos[idx[hit], t0] = True
                if t0 in sel:
                    sel_n[idx] += 1
                    sel_votes[idx[hit]] += 1
                    sel_psum[idx] += proba[:, k]
            if rule == "count":
                det = votes >= param
                score = np.where(n_valid > 0, votes / np.maximum(n_valid, 1), 0.0)
            elif rule == "fraction":
                det = (sel_n > 0) & (sel_votes >= param * sel_n - 1e-9)
                score = np.where(sel_n > 0, sel_votes / np.maximum(sel_n, 1), 0.0)
            elif rule == "prob":
                det = (sel_n > 0) & (sel_psum >= param * sel_n - 1e-9)
                score = np.where(sel_n > 0, sel_psum / np.maximum(sel_n, 1), 0.0)
            else:
                raise ValueError(f"unknown rule {rule}")
            pos = np.arange(n_windows, dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                dtime = np.nanmedian(np.where(hit_pos, pos[None, :], np.nan), axis=1)
            out = np.full((3, n), np.nan, dtype=np.float32)
            out[0] = det
            out[1] = np.rint(100.0 * score)
            out[2] = dtime
            labels = ["classification", "probability", "detection_time"]
            result = xr.DataArray(
                out.reshape(3, ny, nx),
                dims=["bands", "y", "x"],
                coords={"bands": labels, "y": features.y.values, "x": features.x.values},
            )
            result = smooth_classification(result, int(self.parameters.get("smooth_kernel", 1)))
            result = keep_detection_time_for_crop_pixels(result)
            if sp.get(
                "name"
            ):  # several crops: bands are named classification_<crop>, probability_<crop>, detection_time_<crop>
                result = result.assign_coords(bands=[f"{l}_{sp['name']}" for l in labels])
            results.append(result)
        return results[0] if len(results) == 1 else xr.concat(results, dim="bands")


def run_single_workflow(
    xarr, epsg: int, parameters: Dict[str, Any], mask: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """Run single workflow.

    Args:
        xarr: Input value for ``xarr`` used by this operation.
        epsg: Input value for ``epsg`` used by this operation. Expected type: ``int``.
        parameters: Input value for ``parameters`` used by this operation. Expected type: ``Dict[str, Any]``.
        mask: Input value for ``mask`` used by this operation. Expected type: ``Optional[xr.DataArray]``.

    Returns:
        Result of the operation. Expected type: ``xr.DataArray``.
    """
    logger.info("=== WORKFLOW START ===")
    logger.info(f"INPUT shape={xarr.shape}, dims={xarr.dims}")
    logger.info(f"INPUT bands={xarr.coords['bands'].values.tolist()}")

    logger.info("=== ONNX CLASSIFICATION ===")
    classifier = (ONNXClassifierWindowed if parameters["classifier_parameters"].get("croplogic") else ONNXClassifier)(
        parameters["classifier_parameters"]
    )
    classes = classifier.predict(xarr)
    logger.info(f"RAW shape={classes.shape}, dims={classes.dims}")
    logger.info(f"RAW bands={classes.coords['bands'].values.tolist()}")

    for b in classes.coords["bands"].values.tolist():
        v = classes.sel(bands=b).values
        finite = v[np.isfinite(v)]
        if finite.size:
            logger.info(
                f"RAW {b}: dtype={v.dtype}, min={finite.min()}, max={finite.max()}, mean={finite.mean()}, unique={np.unique(finite)[:30].tolist()}"
            )

    logger.info("=== POSTPROCESSING ===")
    postprocess_parameters: Dict[str, Any] = parameters.get("postprocess_parameters", {})
    postprocess_parameters["enable"] = False
    postprocess_parameters["save_intermediate"] = False
    postprocess_parameters["method"] = "majority_vote"
    logger.info(f"POSTPROCESS PARAMETERS={postprocess_parameters}")

    if postprocess_parameters.get("enable"):
        postprocessor = Postprocessor(
            postprocess_parameters, classifier_url=parameters.get("classifier_parameters", {}).get("classifier_url")
        )
        classes = postprocessor.apply(classes)
        logger.info("Postprocessing done.")

    logger.info("=== POSTPROCESS OUTPUT ===")
    logger.info(f"POST shape={classes.shape}, dims={classes.dims}")
    logger.info(f"POST bands={classes.coords['bands'].values.tolist()}")

    for b in classes.coords["bands"].values.tolist():
        v = classes.sel(bands=b).values
        finite = v[np.isfinite(v)]
        if finite.size:
            logger.info(
                f"POST {b}: dtype={v.dtype}, min={finite.min()}, max={finite.max()}, mean={finite.mean()}, unique={np.unique(finite)[:30].tolist()}"
            )

    if mask is not None:
        logger.info("=== MASK ===")
        classes = classes.where(mask, 254)
        logger.info("Mask applied; masked pixels=254")

    logger.info("=== FINAL OUTPUT ===")
    logger.info(f"FINAL shape={classes.shape}, dims={classes.dims}")
    logger.info(f"FINAL bands={classes.coords['bands'].values.tolist()}")

    for b in classes.coords["bands"].values.tolist():
        v = classes.sel(bands=b).values
        finite = v[np.isfinite(v)]
        if finite.size:
            logger.info(
                f"FINAL {b}: min={finite.min()}, max={finite.max()}, mean={finite.mean()}, unique={np.unique(finite)[:30].tolist()}"
            )

    logger.info("=== WORKFLOW END ===")
    return classes


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """Real fix: converted from the old apply_udf_data(udf_data) signature
    to apply_datacube(cube, context) -- the documented, modern openEO
    UDF entry point. Confirmed against the official openEO Python
    Client documentation: context here is guaranteed to arrive as a
    direct function parameter. apply_udf_data's signature relied on
    context being populated into udf_data.user_context, which this
    backend was NOT reliably doing (confirmed directly: a real job
    failed with "Available keys: []" despite inference_job_builder.py
    correctly passing context into apply_neighborhood). Switching to
    the documented signature removes that unreliable indirection
    entirely, rather than working around it.

    Args:
        cube: Input value for ``cube`` used by this operation. Expected type: ``XarrayDataCube``.
        context: Input value for ``context`` used by this operation. Expected type: ``dict``.

    Returns:
        Result of the operation. Expected type: ``XarrayDataCube``.

    Raises:
        ValueError: If the operation cannot complete under the required conditions.
    """
    xarr = cube.get_array()

    logger.info(f"BAND NAMES: {xarr.coords['bands'].values.tolist()}")
    logger.info(f"Input shape: {xarr.shape}")
    logger.info(f"Input dims: {xarr.dims}")

    context = context or {}

    # EPSG comes from context now too, not udf_data.proj -- inference_job_builder.py
    # already puts it there ("EPSG": target_epsg), so this is one reliable
    # source instead of two separate indirect ones.
    epsg = context.get("EPSG")
    if epsg is None:
        raise ValueError(
            f"Missing EPSG in context. Available keys: {list(context.keys())}. "
            f"This UDF must be called with context={{'EPSG': ...}} via apply_neighborhood."
        )

    classifier_url = context.get("classifier_url")
    if not classifier_url:
        raise ValueError(
            f"Missing classifier_url in context. Available keys: {list(context.keys())}. "
            f"This UDF must be called with context={{'classifier_url': ...}} "
            f"via apply_neighborhood -- no default model is used."
        )

    logger.info(f"Using classifier_url from real context: {classifier_url}")

    output = run_single_workflow(
        xarr,
        epsg=epsg,
        parameters={
            "classifier_parameters": {
                EPSG_HARMONIZED_NAME: epsg,
                "ignore_dependencies": True,
                "classifier_url": classifier_url,
                "model_json_url": context.get("model_json_url"),
                "croplogic": context.get("croplogic"),  # croplogic model settings (bands, class index, rule, windows)
                # cleaning right after prediction (defaults = the old behaviour: any 1 window, no smoothing)
                "min_windows": context.get("min_windows", 1),
                "smooth_kernel": context.get("smooth_kernel", 1),
                # in-season emulation (None = use the whole series)
                "data_cutoff_date": context.get("data_cutoff_date"),
            }
        },
    )

    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output dims: {output.dims}")
    logger.info(f"Output bands: {output.coords['bands'].values.tolist()}")

    # Give the output a real time label taken from the input cube. Without a time
    # dimension, the backend stamps every output tile with the fixed date 2020-01-01.
    # That date lies outside the cube's own time range whenever the job's temporal
    # extent is not 2019/2020, and the cropland mask step then silently drops every
    # tile -- the job "finishes" but writes 0 files (confirmed with tiny test jobs:
    # no-mask worked, masked-with-this-fix worked, masked-without-it did not).
    if "t" in xarr.coords and xarr.sizes.get("t", 0) > 0:
        output = output.expand_dims(t=[xarr.coords["t"].values[0]]).transpose("t", "bands", "y", "x")
        logger.info(f"Output now carries time label {output.coords['t'].values[0]} (dims {output.dims})")

    return XarrayDataCube(output)
