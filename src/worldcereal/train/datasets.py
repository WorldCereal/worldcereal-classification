import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import resources
from math import floor
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from loguru import logger
from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    Predictors,
)
from torch.utils.data import Dataset, Sampler

from worldcereal.seasons import season_doys_to_dates_refyear
from worldcereal.train import (
    GLOBAL_SEASON_IDS,
    MIN_EDGE_BUFFER,
    OUTLIER_COLUMNS,
    SEASONALITY_COLUMN_MAP,
    SEASONALITY_LAT_RANGE,
    SEASONALITY_LON_RANGE,
    SEASONALITY_LOOKUP_COLUMNS,
    SEASONALITY_LOOKUP_FILENAME,
    SEASONALITY_LOOKUP_PACKAGE,
    SEASONALITY_LOOKUP_PATH,
)
from worldcereal.train import predictors as _predictor_utils
from worldcereal.train.collation import ATTR_KEYS_ALLOW_PARTIAL_NONE
from worldcereal.train.seasonal import (
    SeasonWindow,
    align_to_composite_window,
    coerce_date_for_year,
    date_in_season,
    enumerate_composite_slots,
    in_season_window,
    season_window_from_dates,
)

# Re-export helper functions so legacy imports remain valid.
generate_predictor = _predictor_utils.generate_predictor
run_model_inference = _predictor_utils.run_model_inference

# minimum distance from valid_position to the edges when augmenting
# we need to define it globally so that it can be used in process_parquet as well
SeasonCalendarMode = Literal["calendar", "custom", "auto", "off"]
SeasonEngine = Literal["manual", "calendar", "off"]

_SEASONALITY_LOOKUP_TABLE: Optional[pd.DataFrame] = None


def _seasonality_lookup_context():
    """Return a context manager pointing to the seasonality lookup parquet."""

    if SEASONALITY_LOOKUP_PATH.exists():
        return nullcontext(SEASONALITY_LOOKUP_PATH)
    return resources.path(SEASONALITY_LOOKUP_PACKAGE, SEASONALITY_LOOKUP_FILENAME)


def _is_lc_only_dataset(ref_id: str) -> bool:
    """Return True for LC-only datasets whose ref_id ends in ``_100`` or ``_101``.

    These datasets carry no crop-type annotation by design, so their seasonal
    masks should always be considered fully valid regardless of the sample's
    valid_time relative to any crop calendar.
    """
    return bool(re.search(r"_10[01]$", ref_id))


SAMPLE_ATTR_COLUMNS: Tuple[str, ...] = (
    "lat",
    "lon",
    "ref_id",
    "sample_id",
    "region",
    "finetune_class",
    "downstream_class",
    "landcover_label",
    "croptype_label",
    "valid_time",
    "quality_score_lc",
    "quality_score_ct",
    "sample_weight_lc",
    "sample_weight_ct",
    *OUTLIER_COLUMNS.values(),
)


_LABEL_DATETIME_COLUMNS: Tuple[str, ...] = (
    "valid_time",
    "valid_date",
)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (float, int, np.floating, np.integer)):
        if value == NODATAVALUE:
            return True
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return True
    if pd.isna(value):
        return True
    return False


def _extract_float(value: Any) -> Optional[float]:
    if _is_missing_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_datetime64(value: Any) -> Optional[np.datetime64]:
    if _is_missing_value(value):
        return None
    try:
        ts = pd.to_datetime(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return np.datetime64(ts, "D")


def _resolve_label_datetime(row: Mapping[str, Any]) -> Optional[np.datetime64]:
    """Pick the first non-missing label timestamp available in the sample row."""

    for column in _LABEL_DATETIME_COLUMNS:
        if column not in row:
            continue
        candidate = _safe_datetime64(row.get(column))
        if candidate is not None:
            return candidate
    return None


def _timestamps_to_datetime_array(timestamps: np.ndarray) -> np.ndarray:
    """Convert array of (day, month, year) to np.datetime64[D] array.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of shape (N, 3) or (3,) with (day, month, year) entries.

    Returns
    -------
    np.ndarray
        Array of shape (N,) with dtype np.datetime64[D].
    """
    timestamps = np.asarray(timestamps)
    if timestamps.ndim == 1:
        timestamps = timestamps.reshape(-1, 3)
    dates = [
        np.datetime64(f"{int(year):04d}-{int(month):02d}-{int(day):02d}", "D")
        for day, month, year in timestamps.tolist()
    ]
    return np.array(dates, dtype="datetime64[D]")


def _default_season_mask(num_timesteps: int, num_seasons: int) -> np.ndarray:
    num_seasons = max(1, num_seasons)
    return np.ones((num_seasons, num_timesteps), dtype=bool)


def _resolve_season_engine(
    mode: SeasonCalendarMode, has_manual_windows: bool
) -> SeasonEngine:
    if mode == "custom":
        if not has_manual_windows:
            raise ValueError(
                "season_calendar_mode='custom' requires explicit season_windows to be provided."
            )
        return "manual"
    if mode == "calendar":
        return "calendar"
    if mode == "auto":
        return "manual" if has_manual_windows else "calendar"
    if mode == "off":
        return "off"
    raise ValueError(f"Unknown season_calendar_mode: {mode}")


def _normalize_season_windows(
    season_windows: Optional[Mapping[str, Tuple[Any, Any]]],
) -> Dict[str, SeasonWindow]:
    normalized: Dict[str, SeasonWindow] = {}
    if not season_windows:
        return normalized

    for season, window in season_windows.items():
        if window is None:
            raise ValueError(
                f"Season window for '{season}' must be a (start, end) tuple, got None."
            )
        try:
            start_value, end_value = window
        except (TypeError, ValueError):
            raise ValueError(
                f"Season window for '{season}' must be an iterable of two datetime-like values."
            ) from None

        start_dt = _safe_datetime64(start_value)
        end_dt = _safe_datetime64(end_value)
        if start_dt is None or end_dt is None:
            raise ValueError(
                f"Season window for '{season}' must contain valid datetime-like values."
            )

        start_np = start_dt.astype("datetime64[D]")
        end_np = end_dt.astype("datetime64[D]")
        try:
            normalized[season] = season_window_from_dates(start_np, end_np)
        except ValueError as exc:
            raise ValueError(f"Season window for '{season}': {exc}") from exc

    return normalized


def _label_datetime_series(frame: pd.DataFrame) -> pd.Series:
    """Return the first available label datetime per row as a pandas Series."""

    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for column in _LABEL_DATETIME_COLUMNS:
        if column not in frame.columns:
            continue
        # Replace NODATAVALUE with NaN before converting to datetime
        column_data = frame[column].replace(NODATAVALUE, np.nan)
        candidate = pd.to_datetime(column_data, errors="coerce")
        result = result.where(result.notna(), candidate)
        if result.notna().all():
            break
    return result


def _filter_frame_by_manual_windows(
    dataframe: pd.DataFrame,
    season_windows: Mapping[str, SeasonWindow],
    *,
    context: str,
    timestep_freq: Literal["month", "dekad"],
) -> Tuple[pd.DataFrame, int]:
    if not season_windows:
        return dataframe, 0

    label_datetimes = _label_datetime_series(dataframe)
    if label_datetimes.isna().all():
        logger.warning(
            f"{context}: All samples are missing valid_time information. "
            "Season windows will be used to create season masks but not for sample filtering."
        )
        return dataframe, 0

    keep_mask = pd.Series(False, index=dataframe.index, dtype=bool)
    for season_name, window in season_windows.items():
        keep_mask |= label_datetimes.apply(
            lambda ts, w=window: (
                pd.notna(ts) and in_season_window(ts, w, freq=timestep_freq)
            )
        )

    missing = label_datetimes.isna().sum()
    if missing:
        logger.warning(
            f"{context}: Dropping {int(missing)} samples missing valid_time while enforcing manual season window(s)."
        )
        keep_mask &= label_datetimes.notna()

    dropped = int((~keep_mask).sum())
    if dropped:
        ranges = ", ".join(
            f"{season}: {window.start_month:02d}-{window.start_day:02d} -> {window.end_month:02d}-{window.end_day:02d}"
            for season, window in season_windows.items()
        )
        logger.info(
            f"{context}: Removed {dropped} samples outside manual season window(s): {ranges}"
        )

    retained = int(keep_mask.sum())
    if retained == 0:
        raise ValueError(
            f"{context}: No samples remain after enforcing manual season window(s)."
        )

    filtered = dataframe.loc[keep_mask].copy().reset_index(drop=True)
    return filtered, dropped


def _ensure_seasonality_lookup() -> pd.DataFrame:
    """Load and cache the seasonality lookup table indexed by lat/lon centers."""

    global _SEASONALITY_LOOKUP_TABLE
    if _SEASONALITY_LOOKUP_TABLE is not None:
        return _SEASONALITY_LOOKUP_TABLE

    try:
        with _seasonality_lookup_context() as lookup_path:
            table = pd.read_parquet(lookup_path)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise FileNotFoundError(
            "Seasonality lookup parquet not found at "
            f"{SEASONALITY_LOOKUP_PATH} or within package "
            f"'{SEASONALITY_LOOKUP_PACKAGE}'."
        ) from exc
    required = {"lat", "lon", *SEASONALITY_LOOKUP_COLUMNS}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Seasonality lookup parquet is missing required columns: {sorted(missing)}"
        )

    table = table.astype({"lat": np.float64, "lon": np.float64})
    table = table.set_index(["lat", "lon"])
    if not table.index.is_unique:
        raise ValueError("Seasonality lookup index must be unique per lat/lon cell.")

    _SEASONALITY_LOOKUP_TABLE = table[list(SEASONALITY_LOOKUP_COLUMNS)].sort_index()
    return _SEASONALITY_LOOKUP_TABLE


def _snap_coordinate_to_grid(value: float, bounds: Tuple[float, float]) -> float:
    """Snap a coordinate to the 0.5° grid center used by the lookup."""

    min_value, max_value = bounds
    if value is None:
        raise ValueError("Cannot snap a missing coordinate to the seasonality grid.")
    clamped = max(min(float(value), max_value), min_value)
    return (floor(clamped * 2.0) / 2.0) + 0.25


def _snap_latlon_to_calendar_grid(lat: float, lon: float) -> Tuple[float, float]:
    lat_center = _snap_coordinate_to_grid(lat, SEASONALITY_LAT_RANGE)
    lon_center = _snap_coordinate_to_grid(lon, SEASONALITY_LON_RANGE)
    return lat_center, lon_center


def get_class_weights(
    labels: Union[np.ndarray[Any, Any], Sequence[Hashable]],
    method: str = "balanced",  # 'balanced', 'log', 'effective', or 'none'
    clip_range: Optional[tuple] = None,  # e.g. (0.2, 10.0)
    normalize: bool = True,
    counts_override: Optional[Mapping[Any, int]] = None,
    verbose: bool = True,
    pool_name: str = "",
) -> Dict[Hashable, float]:
    """
    Compute class weights for classification tasks.

    Args:
        labels: array of integer class labels.
        method:
            - 'balanced' : inverse frequency (sklearn-style)
            - 'log'      : log-scaled inverse frequency
            - 'effective': effective number of samples (Cui et al.)
            - 'none'     : uniform weights
        clip_range: tuple (min, max) to clip weights.
        normalize: whether to rescale weights to mean = 1.

    Returns:
        class_weights_dict: dict mapping class index → weight
    """
    if counts_override is not None:
        counts: Dict[Hashable, int] = {k: int(v) for k, v in counts_override.items()}
    else:
        counts = {k: int(v) for k, v in Counter(labels).items()}

    classes = sorted(counts.keys(), key=lambda value: str(value))
    total_samples = sum(counts.values())
    num_classes = len(classes)

    if num_classes == 0:
        return {}

    freq = np.array([counts[c] for c in classes], dtype=np.float64)
    freq = np.maximum(freq, 1.0)  # safety

    if method == "balanced":
        weights = total_samples / (num_classes * freq)

    elif method == "log":
        inv_freq = 1.0 / freq
        weights = np.log1p(inv_freq / np.mean(inv_freq))

    elif method == "effective":
        # Effective number of samples (Class-Balanced Loss, Cui et al. 2019).
        # Beta is derived from the total sample count so the saturation point
        # (1 / (1 - beta)) scales with the dataset rather than being fixed at
        # 1000 (beta=0.999), which causes all classes to saturate and receive
        # near-identical weights when class sizes are in the millions.
        beta = total_samples / (total_samples + 1)
        effective_num = (1.0 - np.power(beta, freq)) / (1.0 - beta)
        weights = 1.0 / effective_num

    elif method == "none":
        weights = np.ones_like(freq)

    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize:
        logger.debug("Normalizing weights to mean = 1")
        weights = weights / weights.mean()

    if clip_range:
        logger.info(f"Clipping weights to range {clip_range}")
        weights = np.clip(weights, clip_range[0], clip_range[1])

    rounded = np.round(weights, 3)
    result = {cls: float(weight) for cls, weight in zip(classes, rounded.tolist())}
    if verbose:
        display = {
            str(cls): float(weight) for cls, weight in zip(classes, rounded.tolist())
        }
        prefix = f"[{pool_name}] " if pool_name else ""
        logger.info(f"{prefix}Class weights ({method}): {display}")
    return result


def _stringify_weight_dict(weights: Dict[Hashable, float]) -> Dict[str, float]:
    """Convert potentially mixed-type keys to strings for deterministic indexing."""

    return {str(key): float(value) for key, value in weights.items()}


def _get_normalized_weights(
    labels: np.ndarray,
    method: str,
    clip_range: Optional[Tuple[float, float]],
    pool_name: str = "",
) -> np.ndarray:
    """Return a per-sample float64 weight array.

    Weights are computed via ``get_class_weights`` which normalizes to mean=1
    first, then clips to *clip_range*.  This order guarantees that the clip
    bounds are respected in the final output — values returned here are always
    within *clip_range* (when provided).
    """
    w_dict = _stringify_weight_dict(
        get_class_weights(
            labels,
            method=method,
            clip_range=clip_range,
            normalize=True,
            pool_name=pool_name,
        )
    )
    return np.array([w_dict[str(lbl)] for lbl in labels], dtype=np.float64)


def _get_spatial_density_weights(
    spatial_bins: np.ndarray,
    method: str,
    clip_range: Optional[Tuple[float, float]],
    min_samples_per_bin: int = 50,
) -> np.ndarray:
    """Per-sample spatial-density weight, with a sparse-bin override.

    Computes density weights for all bins via `_get_normalized_weights`,
    then overrides bins below *min_samples_per_bin* to the dataset-mean weight
    (= 1.0). Without this protection, singleton/sparse bins would be assigned
    extreme weights from the inverse-density formula (e.g. a 1-sample bin in
    the Arctic can pin the density factor to its upper clip),
    blowing up the multiplicative composition with class weights downstream.

    Mirrors the analogous safeguard in `_get_per_bin_class_weights`:
    a bin is considered too sparse to estimate a stable density weight from,
    and is treated as "average density" instead.

    ``min_samples_per_bin=1`` is a no-op for the sparse-bin override (every
    non-empty bin trivially has >=1 sample); values <1 are rejected.
    """
    if min_samples_per_bin < 1:
        raise ValueError(f"min_samples_per_bin must be >= 1, got {min_samples_per_bin}")
    sp_arr = _get_normalized_weights(
        spatial_bins, method, clip_range, pool_name="spatial-density"
    )

    if min_samples_per_bin > 1:
        bin_counts = pd.Series(spatial_bins).value_counts()
        sparse_bins = set(bin_counts.index[bin_counts < min_samples_per_bin])
        if sparse_bins:
            sparse_mask = pd.Series(spatial_bins).isin(sparse_bins).to_numpy()
            n_sparse_samples = int(sparse_mask.sum())
            sp_arr = sp_arr.copy()
            sp_arr[sparse_mask] = 1.0
            logger.info(
                f"_get_spatial_density_weights: {len(sparse_bins)}/"
                f"{bin_counts.size} bins ({n_sparse_samples}/{len(spatial_bins)} "
                f"samples) below min_samples_per_bin={min_samples_per_bin}; "
                "density factor set to 1.0 for those samples."
            )
    return sp_arr


def _spatial_bins_from_latlon(
    latitudes: pd.Series, longitudes: pd.Series, bin_size: float
) -> np.ndarray:
    """Quantize latitude/longitude pairs into coarse grid bins."""

    if bin_size <= 0:
        raise ValueError("spatial_bin_size_degrees must be a positive number")

    lat_array = latitudes.to_numpy(dtype=np.float64, copy=True)
    lon_array = longitudes.to_numpy(dtype=np.float64, copy=True)

    if np.isnan(lat_array).any() or np.isnan(lon_array).any():
        raise ValueError(
            "Latitude/longitude contain missing values; cannot build spatial bins"
        )

    lat_bins = np.floor((lat_array + 90.0) / bin_size).astype(np.int64)
    lon_bins = np.floor((lon_array + 180.0) / bin_size).astype(np.int64)

    lat_str = lat_bins.astype(str)
    lon_str = lon_bins.astype(str)
    return np.char.add(np.char.add(lat_str, "_"), lon_str)


def _spatial_bins_from_h3(
    latitudes: pd.Series, longitudes: pd.Series, resolution: int
) -> np.ndarray:
    """Quantize latitude/longitude pairs into Uber H3 hexagonal cell bins.

    Each (lat, lon) is mapped to its H3 cell index (as a string) at the given
    *resolution*. Unlike fixed-degree lat/lon bins -- whose ground area shrinks
    dramatically toward the poles (a 5x5 degree cell near 60N covers roughly
    half the area of one at the equator) -- H3 cells are approximately
    equal-area hexagons. This gives more geographically consistent per-bin
    class balancing, so co-located classes (e.g. wheat vs oats in Europe) are
    balanced against each other within comparable-sized neighbourhoods rather
    than within latitude-distorted rectangles.

    Parameters
    ----------
    latitudes, longitudes : pd.Series
        Sample coordinates in degrees.
    resolution : int
        H3 resolution (0 = coarsest ... 15 = finest). As a rough guide:
        res 1 ~ 610,000 km2, res 2 ~ 86,000 km2, res 3 ~ 12,000 km2 per cell.

    Returns
    -------
    np.ndarray
        Array of H3 cell-index strings, one per sample.
    """
    if not isinstance(resolution, (int, np.integer)) or not (0 <= resolution <= 15):
        raise ValueError(f"h3 resolution must be an int in [0, 15], got {resolution!r}")

    try:
        import h3
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "bin_type='h3' requires the 'h3' package (>=4). Install it via "
            "`pip install h3` or `conda install h3-py`."
        ) from exc

    lat_array = latitudes.to_numpy(dtype=np.float64, copy=True)
    lon_array = longitudes.to_numpy(dtype=np.float64, copy=True)

    if np.isnan(lat_array).any() or np.isnan(lon_array).any():
        raise ValueError(
            "Latitude/longitude contain missing values; cannot build spatial bins"
        )

    # Support both the h3 v4 API (latlng_to_cell) and the legacy v3 API
    # (geo_to_h3) so the function works across environments.
    if hasattr(h3, "latlng_to_cell"):
        _to_cell = h3.latlng_to_cell  # h3 >= 4
    elif hasattr(h3, "geo_to_h3"):
        _to_cell = h3.geo_to_h3  # h3 < 4
    else:  # pragma: no cover - unexpected h3 build
        raise ImportError(
            "Installed 'h3' package exposes neither latlng_to_cell nor geo_to_h3."
        )

    cells = [
        _to_cell(float(lat), float(lon), resolution)
        for lat, lon in zip(lat_array, lon_array)
    ]
    return np.asarray(cells, dtype=object).astype(str)


def _get_per_bin_class_weights(
    labels: np.ndarray,
    bins: np.ndarray,
    method: str,
    clip_range: Optional[Tuple[float, float]],
    min_samples_per_bin: int = 50,
    pool_name: str = "",
    min_samples_per_class_per_bin: Optional[int] = None,
) -> np.ndarray:
    """Per-sample class weights computed *within* each spatial bin.

    For each unique value in *bins*, class weights are derived from the
    within-bin label distribution via `get_class_weights` (with
    ``normalize=True`` and no clipping).  Bins containing fewer than
    *min_samples_per_bin* samples fall back to globally-computed class
    weights — within-bin counts are too sparse to estimate a stable class
    distribution otherwise.

    Within a bin, classes with fewer than *min_samples_per_class_per_bin*
    samples are excluded from the within-bin weight computation entirely
    (they don't count toward ``num_classes`` either, removing the
    destabilising effect of rare classes on dense ones via the ``k_bin``
    factor in the ``balanced`` recipe). Samples of those filtered classes
    inherit the global class weight individually. ``None`` (default)
    auto-derives as ``max(1, min_samples_per_bin // 10)``.
    """
    if labels.shape != bins.shape:
        raise ValueError(
            f"labels and bins must have the same shape; got "
            f"labels={labels.shape}, bins={bins.shape}"
        )
    if min_samples_per_bin < 1:
        raise ValueError(f"min_samples_per_bin must be >= 1, got {min_samples_per_bin}")
    if min_samples_per_class_per_bin is None:
        min_samples_per_class_per_bin = max(1, min_samples_per_bin // 10)
    if min_samples_per_class_per_bin < 1:
        raise ValueError(
            f"min_samples_per_class_per_bin must be >= 1, got "
            f"{min_samples_per_class_per_bin}"
        )

    # Clip the GLOBAL fallback weights (used for sparse bins and rare in-bin
    # classes) to clip_range. Without this, a globally-rare class could inject
    # an extreme unclipped weight that inflates the array mean and distorts the
    # final mean-normalisation for every other sample.
    global_w_dict = _stringify_weight_dict(
        get_class_weights(
            labels,
            method=method,
            clip_range=clip_range,
            normalize=True,
            pool_name=pool_name,
        )
    )

    weights = np.empty(len(labels), dtype=np.float64)
    unique_bins = np.unique(bins)
    n_fallback_bins = 0
    n_fallback_samples = 0
    n_class_filtered_samples = 0

    for bin_id in unique_bins:
        mask = bins == bin_id
        bin_labels = labels[mask]

        if len(bin_labels) < min_samples_per_bin:
            n_fallback_bins += 1
            n_fallback_samples += int(mask.sum())
            weights[mask] = np.array(
                [global_w_dict[str(lbl)] for lbl in bin_labels],
                dtype=np.float64,
            )
            continue

        # Per-class filtering: drop classes with too few samples in this bin.
        class_counts = pd.Series(bin_labels).value_counts()
        well_represented = set(
            class_counts.index[class_counts >= min_samples_per_class_per_bin]
        )
        if not well_represented:
            # No class meets the per-class threshold; fall back wholesale.
            weights[mask] = np.array(
                [global_w_dict[str(lbl)] for lbl in bin_labels],
                dtype=np.float64,
            )
            continue

        kept_mask = np.array(
            [lbl in well_represented for lbl in bin_labels], dtype=bool
        )
        filtered_labels = bin_labels[kept_mask]
        bin_w_dict = _stringify_weight_dict(
            get_class_weights(
                filtered_labels,
                method=method,
                clip_range=None,
                normalize=True,
                verbose=False,
            )
        )
        per_sample = np.array(
            [
                (
                    bin_w_dict[str(lbl)]
                    if lbl in well_represented
                    else global_w_dict[str(lbl)]
                )
                for lbl in bin_labels
            ],
            dtype=np.float64,
        )
        weights[mask] = per_sample
        n_class_filtered_samples += int((~kept_mask).sum())

    prefix = (
        f"_get_per_bin_class_weights[{pool_name}]"
        if pool_name
        else "_get_per_bin_class_weights"
    )
    if n_fallback_bins > 0:
        logger.info(
            f"{prefix}: {n_fallback_bins}/{len(unique_bins)} "
            f"bins ({n_fallback_samples}/{len(labels)} samples) below "
            f"min_samples_per_bin={min_samples_per_bin}; using global class "
            "weights for those samples."
        )
    if n_class_filtered_samples > 0:
        logger.info(
            f"{prefix}: {n_class_filtered_samples}/{len(labels)} samples "
            f"belong to a class with < min_samples_per_class_per_bin="
            f"{min_samples_per_class_per_bin} in their bin; using global "
            "class weights for those samples (other classes in their bin "
            "compute as if those samples weren't there)."
        )

    mean = weights.mean()
    if mean > 0:
        weights = weights / mean

    if clip_range is not None:
        weights = np.clip(weights, clip_range[0], clip_range[1])

    return weights


def _get_smoothed_per_bin_class_weights(
    labels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    bin_size: float,
    method: str,
    clip_range: Optional[Tuple[float, float]],
    min_samples_per_bin: int = 50,
    pool_name: str = "",
    min_samples_per_class_per_bin: Optional[int] = None,
) -> np.ndarray:
    """Per-sample class weights with bilinear interpolation between bin centers.

    Smooths the step changes that ``_get_per_bin_class_weights`` produces at
    bin boundaries. For each sample, the weight for its class is computed as
    the bilinear blend of the per-class weight from the four neighbouring
    bin centers, using the sample's continuous (lat, lon) position.

    Bins below *min_samples_per_bin* and (bin, class) pairs below
    *min_samples_per_class_per_bin* (default ``max(1, min_samples_per_bin //
    10)``) are treated as missing — both are excluded from the per-bin
    class-weight computation, and the bilinear kernel renormalises over the
    remaining valid corners. Samples with all four corners NaN fall back to the
    global class weight.

    A sample exactly at a bin centre yields the same weight as hard
    binning; a sample at the corner between four bins yields the average
    of the four corner weights.
    """
    if labels.shape != latitudes.shape or labels.shape != longitudes.shape:
        raise ValueError(
            "labels, latitudes, longitudes must have the same shape; got "
            f"{labels.shape}, {latitudes.shape}, {longitudes.shape}"
        )
    if min_samples_per_bin < 1:
        raise ValueError(f"min_samples_per_bin must be >= 1, got {min_samples_per_bin}")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be > 0, got {bin_size}")

    str_labels = labels.astype(str)

    # Hard bin assignment (same as _spatial_bins_from_latlon)
    lat_bin = np.floor((latitudes + 90.0) / bin_size).astype(np.int64)
    lon_bin = np.floor((longitudes + 180.0) / bin_size).astype(np.int64)

    # Continuous bin-centre coordinates for bilinear interpolation.
    u_cont = (latitudes + 90.0) / bin_size - 0.5
    v_cont = (longitudes + 180.0) / bin_size - 0.5
    u_floor = np.floor(u_cont).astype(np.int64)
    v_floor = np.floor(v_cont).astype(np.int64)
    fu = (u_cont - u_floor).astype(np.float64)
    fv = (v_cont - v_floor).astype(np.float64)

    # Build the per-class lookup grid (with sparse-bin and per-class filtering).
    all_lat_idx = np.concatenate([u_floor, u_floor + 1])
    all_lon_idx = np.concatenate([v_floor, v_floor + 1])
    lat_min, lat_max = int(all_lat_idx.min()), int(all_lat_idx.max())
    lon_min, lon_max = int(all_lon_idx.min()), int(all_lon_idx.max())
    grid, class_to_idx, global_w_dict, n_dense_bins, n_total_bins = (
        _build_per_class_weight_grid(
            str_labels,
            lat_bin,
            lon_bin,
            method,
            min_samples_per_bin,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            min_samples_per_class_per_bin=min_samples_per_class_per_bin,
            pool_name=pool_name,
        )
    )
    n_lat = grid.shape[1]
    n_lon = grid.shape[2]

    sample_class_idx = np.array([class_to_idx[c] for c in str_labels], dtype=np.int64)
    sample_global_w = np.array([global_w_dict[c] for c in str_labels], dtype=np.float64)

    def _lookup_corner(li_offset: int, lo_offset: int) -> Tuple[np.ndarray, np.ndarray]:
        gi = u_floor + li_offset - lat_min
        gj = v_floor + lo_offset - lon_min
        in_bounds = (gi >= 0) & (gi < n_lat) & (gj >= 0) & (gj < n_lon)
        gi_clip = np.clip(gi, 0, n_lat - 1)
        gj_clip = np.clip(gj, 0, n_lon - 1)
        vals = grid[sample_class_idx, gi_clip, gj_clip]
        valid = in_bounds & ~np.isnan(vals)
        # Replace NaN with 0 so the masked-out contribution doesn't pollute math.
        return np.where(valid, vals, 0.0), valid.astype(np.float64)

    v_sw, m_sw = _lookup_corner(0, 0)
    v_nw, m_nw = _lookup_corner(1, 0)
    v_se, m_se = _lookup_corner(0, 1)
    v_ne, m_ne = _lookup_corner(1, 1)

    # Valid-only kernel: weighted sum over real corners; renormalise by the
    # weight of valid corners only. Samples with no valid corner at all fall
    # back to the global class weight.
    b_sw = (1 - fu) * (1 - fv)
    b_nw = fu * (1 - fv)
    b_se = (1 - fu) * fv
    b_ne = fu * fv
    weighted_sum = b_sw * v_sw + b_nw * v_nw + b_se * v_se + b_ne * v_ne
    weight_sum = b_sw * m_sw + b_nw * m_nw + b_se * m_se + b_ne * m_ne
    weights = np.where(
        weight_sum > 0,
        weighted_sum / np.maximum(weight_sum, 1e-12),
        sample_global_w,
    )

    mean = weights.mean()
    if mean > 0:
        weights = weights / mean
    if clip_range is not None:
        weights = np.clip(weights, clip_range[0], clip_range[1])

    prefix = (
        f"_get_smoothed_per_bin_class_weights[{pool_name}]"
        if pool_name
        else "_get_smoothed_per_bin_class_weights"
    )
    n_sparse_bins = n_total_bins - n_dense_bins
    logger.info(
        f"{prefix}: bilinear interpolation across {n_dense_bins}/{n_total_bins} "
        f"dense bins (min_samples_per_bin={min_samples_per_bin}); "
        f"{n_sparse_bins} sparse bins and absent (bin, class) corners fall back "
        "to global class weights."
    )

    return weights


def _build_per_class_weight_grid(
    str_labels: np.ndarray,
    lat_bin: np.ndarray,
    lon_bin: np.ndarray,
    method: str,
    min_samples_per_bin: int,
    grid_lat_min: int,
    grid_lat_max: int,
    grid_lon_min: int,
    grid_lon_max: int,
    min_samples_per_class_per_bin: Optional[int] = None,
    pool_name: str = "",
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float], int, int]:
    """Build a (n_classes, n_lat, n_lon) per-class lookup grid.

    Cells with no data (sparse bin, class absent in dense bin, or class with
    fewer than *min_samples_per_class_per_bin* samples in its bin) are NaN.
    Returns the grid plus class index map, global weight dict, and
    (n_dense_bins, n_total_bins) for logging.

    *min_samples_per_class_per_bin* defaults to ``max(1, min_samples_per_bin //
    10)`` — within each dense bin, classes below this in-bin count are dropped
    from the per-bin weight computation entirely (no entry in the grid for
    that class in that bin), so they don't contribute to the ``num_classes``
    factor for the remaining classes.
    """
    if min_samples_per_class_per_bin is None:
        min_samples_per_class_per_bin = max(1, min_samples_per_bin // 10)

    df = pd.DataFrame({"lat_bin": lat_bin, "lon_bin": lon_bin, "label": str_labels})
    bin_groups = df.groupby(["lat_bin", "lon_bin"])
    bin_weights: Dict[Tuple[int, int], Dict[str, float]] = {}
    for (li, lo), group in bin_groups:
        if len(group) < min_samples_per_bin:
            continue
        bin_labels = group["label"].to_numpy()
        class_counts = pd.Series(bin_labels).value_counts()
        well_represented = set(
            class_counts.index[class_counts >= min_samples_per_class_per_bin]
        )
        if not well_represented:
            continue
        kept = bin_labels[np.isin(bin_labels, list(well_represented))]
        bin_weights[(li, lo)] = _stringify_weight_dict(
            get_class_weights(
                kept,
                method=method,
                clip_range=None,
                normalize=True,
                verbose=False,
            )
        )
    n_total_bins = bin_groups.ngroups
    n_dense_bins = len(bin_weights)

    global_w_dict = _stringify_weight_dict(
        get_class_weights(
            str_labels,
            method=method,
            clip_range=None,
            normalize=True,
            pool_name=pool_name,
        )
    )
    unique_classes = sorted(set(str_labels))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}

    n_lat = grid_lat_max - grid_lat_min + 1
    n_lon = grid_lon_max - grid_lon_min + 1
    grid = np.full((len(unique_classes), n_lat, n_lon), np.nan, dtype=np.float64)
    for (li, lo), wdict in bin_weights.items():
        gi = li - grid_lat_min
        gj = lo - grid_lon_min
        if not (0 <= gi < n_lat and 0 <= gj < n_lon):
            continue
        for cls, w in wdict.items():
            if cls in class_to_idx:
                grid[class_to_idx[cls], gi, gj] = w
    return grid, class_to_idx, global_w_dict, n_dense_bins, n_total_bins


@dataclass
class SensorMaskingConfig:
    """Configuration for simulating real-world missing data scenarios.

    Probabilities are applied independently per sample. Values are in [0,1].
    Set config to None or enabled=False to disable masking.

    Invariant: masking never leaves a sample with S1 and S2 both fully
    masked-or-missing. If the drawn masks (combined with gaps already present
    in the data) would wipe both sensors, one synthetically masked timestep is
    restored — preferring S2, and never a sensor whose full elimination is
    intentional (s1_full_dropout_prob or s2_cloud_timestep_prob of 1.0).

    Attributes
    ----------
    enable: bool
        Master switch.
    s1_full_dropout_prob: float
        Probability that all S1 timesteps (VV & VH) are missing (e.g. prolonged platform outage).
    s1_timestep_dropout_prob: float
        Probability applied per timestep to drop S1 values (sporadic acquisition gaps).
    s2_cloud_timestep_prob: float
        Probability applied per timestep to cloud-mask S2 (all optical bands) individually.
    s2_cloud_block_prob: float
        Probability to create a contiguous cloud block of S2 masked timesteps.
    s2_cloud_block_min: int
        Minimum length of the contiguous S2 cloud block.
    s2_cloud_block_max: int
        Maximum length of the contiguous S2 cloud block.
    meteo_timestep_dropout_prob: float
        Probability applied per timestep to mask meteorological data.
    dem_dropout_prob: float
        Probability to mask DEM (rare but possible missing elevation ancillary data).
    seed: Optional[int]
        Optional random seed for reproducibility at dataset construction time.
    """

    enable: bool = False
    s1_full_dropout_prob: float = 0.0
    s1_timestep_dropout_prob: float = 0.0
    s2_cloud_timestep_prob: float = 0.0
    s2_cloud_block_prob: float = 0.0
    s2_cloud_block_min: int = 2
    s2_cloud_block_max: int = 5
    meteo_timestep_dropout_prob: float = 0.0
    dem_dropout_prob: float = 0.0
    seed: Optional[int] = None

    def validate(self, num_timesteps: int):
        if self.s2_cloud_block_min > self.s2_cloud_block_max:
            raise ValueError(
                "s2_cloud_block_min cannot be greater than s2_cloud_block_max"
            )
        if self.s2_cloud_block_max > num_timesteps:
            raise ValueError("s2_cloud_block_max cannot exceed num_timesteps")
        if self.s1_full_dropout_prob >= 1.0 and self.s2_cloud_timestep_prob >= 1.0:
            raise ValueError(
                "s1_full_dropout_prob and s2_cloud_timestep_prob cannot both be 1.0: "
                "every sample would end up with S1 and S2 fully masked"
            )
        for name in [
            "s1_full_dropout_prob",
            "s1_timestep_dropout_prob",
            "s2_cloud_timestep_prob",
            "s2_cloud_block_prob",
            "meteo_timestep_dropout_prob",
            "dem_dropout_prob",
        ]:
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {v}")


class WorldCerealDataset(Dataset):
    BAND_MAPPING = {
        "OPTICAL-B02-ts{}-10m": "B2",
        "OPTICAL-B03-ts{}-10m": "B3",
        "OPTICAL-B04-ts{}-10m": "B4",
        "OPTICAL-B05-ts{}-20m": "B5",
        "OPTICAL-B06-ts{}-20m": "B6",
        "OPTICAL-B07-ts{}-20m": "B7",
        "OPTICAL-B08-ts{}-10m": "B8",
        "OPTICAL-B8A-ts{}-20m": "B8A",
        "OPTICAL-B11-ts{}-20m": "B11",
        "OPTICAL-B12-ts{}-20m": "B12",
        "SAR-VH-ts{}-20m": "VH",
        "SAR-VV-ts{}-20m": "VV",
        "METEO-precipitation_flux-ts{}-100m": "precipitation",
        "METEO-temperature_mean-ts{}-100m": "temperature",
        "DEM-alt-20m": "elevation",
        "DEM-slo-20m": "slope",
    }
    # Column templates probed by the joint S1/S2 availability checks.
    S1_S2_COLUMN_TEMPLATES = [
        k for k in BAND_MAPPING if k.startswith(("SAR-", "OPTICAL-"))
    ]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_timesteps: int = 12,
        timestep_freq: Literal["month", "dekad"] = "month",
        task_type: Literal["ssl", "binary", "multiclass"] = "ssl",
        num_outputs: Optional[int] = None,
        augment: bool = False,
        masking_config: Optional[SensorMaskingConfig] = None,
        min_season_coverage: float = 1.0,
        remove_samples_without_s1_s2: bool = False,
    ):
        """WorldCereal base dataset. This dataset is typically used for
        self-supervised learning.

        Parameters
        ----------
        dataframe : pd.DataFrame
            input dataframe containing the data
        num_timesteps : int, optional
            number of timesteps for a sample, by default 12
        timestep_freq : str, optional. Should be one of ['month', 'dekad']
            frequency of the timesteps, by default "month"
        task_type : str, optional. One of ['ssl', 'binary', 'multiclass', 'regression']
            type of the task, by default self-supervised learning "ssl"
        num_outputs : int, optional
            number of outputs for the task, by default None. If task_type is 'ssl',
            the value of this parameter is ignored.
        augment : bool, optional
            whether to augment the data, by default False
        masking_config : Optional[SensorMaskingConfig], optional
            configuration for sensor masking during training, by default None.
        min_season_coverage : float, optional
            Minimum fraction of a season's composite slots that must fall inside
            the selected timestep window for the season mask to be enabled.
            1.0 (default) enforces full coverage — every slot must be present,
            matching the original strict behaviour used for val/test. For the
            training split with augmentation enabled, pass a lower value (e.g.
            0.5) so that seasons only partially shifted out of the window by
            random augmentation still contribute supervision signal.
        remove_samples_without_s1_s2 : bool, optional
            If True, guarantee every emitted sample has S1 or S2 data:
            (1) at construction, rows with no S1/S2 data in any admissible
            timestep window (one containing `valid_position`) are dropped, and
            (2) at sampling time, a selected window without S1/S2 data is
            re-positioned onto an admissible window that has some. Samples
            violating this produce zero encoder tokens (e.g. OlmoEarth raises
            `num_encoded_tokens is 0`) and break the never-both-fully-missing
            invariant of the sensor masking guard. By default False, in which
            case a warning is logged when such rows are present.
        """
        self.dataframe = dataframe.copy()
        numeric_cols = self.dataframe.select_dtypes(include="number").columns
        self.dataframe[numeric_cols] = self.dataframe[numeric_cols].fillna(NODATAVALUE)
        self.num_timesteps = num_timesteps

        if timestep_freq not in ["month", "dekad"]:
            raise NotImplementedError(
                f"timestep_freq should be one of ['month', 'dekad']. Got `{timestep_freq}`"
            )
        self.timestep_freq = timestep_freq
        self.task_type = task_type
        self.num_outputs = num_outputs
        self.is_ssl = task_type == "ssl"
        self.augment = augment
        self.masking_config = masking_config
        if not (0.0 < min_season_coverage <= 1.0):
            raise ValueError(
                f"min_season_coverage must be in (0, 1]; got {min_season_coverage}"
            )
        self.min_season_coverage = min_season_coverage

        if self.min_season_coverage < 1.0:
            logger.info(
                f"Using min_season_coverage={self.min_season_coverage} with augment={self.augment}"
            )

        self.remove_samples_without_s1_s2 = remove_samples_without_s1_s2

        masking_enabled = False
        if self.masking_config:
            if self.masking_config.seed is not None:
                # set a per-dataset RNG seed (numpy global for simplicity)
                np.random.seed(self.masking_config.seed)
            self.masking_config.validate(self.num_timesteps)
            masking_enabled = self.masking_config.enable
            if masking_enabled:
                logger.info(
                    "Sensor masking enabled for this dataset with config: {}".format(
                        self.masking_config
                    )
                )
            else:
                logger.info(
                    "Sensor masking config provided but enable=False; masking disabled."
                )

        if self.remove_samples_without_s1_s2 or masking_enabled:
            self._check_joint_s1_s2_availability()

    def _check_joint_s1_s2_availability(self):
        """Find samples the joint S1/S2 masking guard cannot repair.

        A sample with no S1 and no S2 data anywhere in its timeseries will
        always violate the never-both-fully-missing invariant, regardless of
        how masking is drawn. Depending on `remove_samples_without_s1_s2`,
        such samples are either dropped from the dataset or reported with a
        warning.

        The check is window-aware: a row is flagged when no admissible window
        (one containing `valid_position` and fitting within the available
        timesteps) holds any S1/S2 data. A row can have data in its full
        timeseries yet still be unusable if none of it is reachable from
        `valid_position`. Since admissible windows slide by one timestep,
        their union is the contiguous range ``[first_min, first_max + T)``,
        so checking that range for data is exact.
        """
        if (
            "available_timesteps" not in self.dataframe.columns
            or "valid_position" not in self.dataframe.columns
        ):
            return

        # Stream the per-(row, timestep) S1/S2 data presence in row chunks.
        # Keeping a dense ``len(dataframe) x max_ts`` matrix around is too
        # expensive for large training frames and can OOM before training starts.
        max_ts = 0
        while any(
            template.format(max_ts) in self.dataframe.columns
            for template in self.S1_S2_COLUMN_TEMPLATES
        ):
            max_ts += 1
        if max_ts == 0:
            return
        timestep_cols = [
            [
                col
                for template in self.S1_S2_COLUMN_TEMPLATES
                if (col := template.format(t)) in self.dataframe.columns
            ]
            for t in range(max_ts)
        ]

        T = self.num_timesteps
        avail = self.dataframe["available_timesteps"].astype(int).to_numpy()
        vp = self.dataframe["valid_position"].astype(int).to_numpy()
        first_min = np.maximum(0, vp - T + 1)
        first_max = np.minimum(vp, avail - T)
        last_exclusive = np.maximum(first_max, first_min) + T
        bad = np.ones(len(self.dataframe), dtype=bool)
        chunk_size = 50_000

        for start in range(0, len(self.dataframe), chunk_size):
            stop = min(start + chunk_size, len(self.dataframe))
            chunk_bad = np.ones(stop - start, dtype=bool)
            chunk_first = first_min[start:stop]
            chunk_last = last_exclusive[start:stop]
            chunk_df = self.dataframe.iloc[start:stop]
            for t in range(max_ts):
                cols = timestep_cols[t]
                if not cols:
                    continue
                relevant = (chunk_first <= t) & (t < chunk_last)
                if not relevant.any():
                    continue
                has_data = (chunk_df[cols].to_numpy(copy=False) != NODATAVALUE).any(
                    axis=1
                )
                chunk_bad &= ~(relevant & has_data)
                if not chunk_bad.any():
                    break
            bad[start:stop] = chunk_bad

        num_bad = int(bad.sum())
        if not num_bad:
            return
        if self.remove_samples_without_s1_s2:
            self.dataframe = self.dataframe.loc[~bad].reset_index(drop=True)
            logger.warning(
                f"Removed {num_bad}/{len(bad)} sample(s) with no S1 and no S2 "
                "data in any admissible timestep window "
                "(remove_samples_without_s1_s2=True)."
            )
        else:
            logger.warning(
                f"{num_bad}/{len(self)} sample(s) have no S1 and no S2 data in any "
                "admissible timestep window; the joint S1/S2 masking guard cannot "
                "restore data for these. Consider removing them with "
                "remove_samples_without_s1_s2=True."
            )

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        row = pd.Series.to_dict(self.dataframe.iloc[idx, :])
        timestep_positions, _ = self.get_timestep_positions(row)
        return Predictors(**self.get_inputs(row, timestep_positions))

    def get_timestep_positions(
        self,
        row_d: Dict,
        min_edge_buffer: int = MIN_EDGE_BUFFER,
    ) -> Tuple[List[int], int]:
        available_timesteps = int(row_d["available_timesteps"])
        valid_position = int(row_d["valid_position"])

        # Get the center point to use for extracting a sequence of timesteps
        center_point = self._get_center_point(
            available_timesteps, valid_position, self.augment, min_edge_buffer
        )

        # Determine the timestep positions to extract
        last_timestep = min(available_timesteps, center_point + self.num_timesteps // 2)
        first_timestep = max(0, last_timestep - self.num_timesteps)
        timestep_positions = list(range(first_timestep, last_timestep))

        # Sanity check to make sure we will extract the correct number of timesteps
        if len(timestep_positions) != self.num_timesteps:
            raise ValueError(
                (
                    "Acquired timestep positions do not have correct length: "
                    f"required {self.num_timesteps}, got {len(timestep_positions)}"
                )
            )

        # Sanity check to make sure valid_position is still within the extracted timesteps
        assert valid_position in timestep_positions, (
            f"Valid position {valid_position} not in timestep positions {timestep_positions}"
        )

        # A window without any S1/S2 data produces a sample the model cannot
        # encode (zero valid tokens), even when the row has data elsewhere in
        # its timeseries. Re-position the window onto data when possible.
        if self.remove_samples_without_s1_s2 and not self._window_has_s1_s2(
            row_d, timestep_positions
        ):
            fallback = self._find_window_with_s1_s2(
                row_d, available_timesteps, valid_position, timestep_positions
            )
            if fallback is not None:
                timestep_positions = fallback

        return timestep_positions, valid_position

    def _window_has_s1_s2(self, row_d: Dict, timestep_positions: List[int]) -> bool:
        """Whether any S1 or S2 band has data at any of the given timesteps."""
        for t in timestep_positions:
            for template in self.S1_S2_COLUMN_TEMPLATES:
                value = row_d.get(template.format(t))
                if value is not None and value != NODATAVALUE:
                    return True
        return False

    def _find_window_with_s1_s2(
        self,
        row_d: Dict,
        available_timesteps: int,
        valid_position: int,
        current_positions: List[int],
    ) -> Optional[List[int]]:
        """Find an alternative window containing S1/S2 data.

        Scans every admissible window (one that still contains
        `valid_position` and fits within the available timesteps) and returns
        one with S1/S2 data: a random one when augmenting, otherwise the one
        closest to the originally selected window. Returns None if no
        admissible window has data — such rows are removed at construction
        when `remove_samples_without_s1_s2` is enabled.
        """
        T = self.num_timesteps
        first_min = max(0, valid_position - T + 1)
        first_max = min(valid_position, available_timesteps - T)
        candidates = [
            positions
            for first in range(first_min, first_max + 1)
            if self._window_has_s1_s2(row_d, positions := list(range(first, first + T)))
        ]
        if not candidates:
            return None
        if self.augment:
            return candidates[np.random.randint(len(candidates))]
        return min(candidates, key=lambda pos: abs(pos[0] - current_positions[0]))

    def _get_center_point(
        self, available_timesteps, valid_position, augment, min_edge_buffer
    ):
        """Helper method to decide on the center point based on which to
        extract the timesteps."""

        if not augment or available_timesteps == self.num_timesteps:
            #  check if the valid position is too close to the start_date and force shifting it
            if valid_position < self.num_timesteps // 2:
                center_point = self.num_timesteps // 2
            #  or too close to the end_date
            elif valid_position > (available_timesteps - self.num_timesteps // 2):
                center_point = available_timesteps - self.num_timesteps // 2
            else:
                # Center the timesteps around the valid position
                center_point = valid_position
        else:
            if self.is_ssl:
                # Take a random center point enabling horizontal jittering
                center_point = int(
                    np.random.choice(
                        range(
                            self.num_timesteps // 2,
                            (available_timesteps - self.num_timesteps // 2),
                        ),
                        1,
                    )[0]
                )
            else:
                # Randomly shift the center point but make sure the resulting range
                # well includes the valid position

                min_center_point = max(
                    self.num_timesteps // 2,
                    valid_position + max(1, min_edge_buffer) - self.num_timesteps // 2,
                )
                max_center_point = min(
                    available_timesteps - self.num_timesteps // 2,
                    valid_position - max(1, min_edge_buffer) + self.num_timesteps // 2,
                )

                center_point = np.random.randint(
                    min_center_point, max_center_point + 1
                )  # max_center_point included

        return center_point

    def _get_timestamps(self, row: Dict, timestep_positions: List[int]) -> np.ndarray:
        """
        Generate an array of dates based on the specified compositing window.
        """
        # adjust start date depending on the compositing window
        start_date = np.datetime64(row["start_date"], "D")
        end_date = np.datetime64(row["end_date"], "D")

        # Generate date vector depending on the compositing window
        if self.timestep_freq == "dekad":
            days, months, years = get_dekad_timestamp_components(start_date, end_date)
        elif self.timestep_freq == "month":
            days, months, years = get_monthly_timestamp_components(start_date, end_date)
        else:
            raise ValueError(f"Unknown compositing window: {self.timestep_freq}")

        return np.stack(
            [
                days[timestep_positions],
                months[timestep_positions],
                years[timestep_positions],
            ],
            axis=1,
        )

    def get_inputs(self, row_d: Dict, timestep_positions: List[int]) -> dict:
        # Get latlons which need to have spatial dims
        latlon = np.reshape(
            np.array([row_d["lat"], row_d["lon"]], dtype=np.float32), (1, 1, 2)
        )

        # Get timestamps belonging to each timestep
        timestamps = self._get_timestamps(row_d, timestep_positions)

        # Initialize inputs
        s1, s2, meteo, dem = self.initialize_inputs()

        # Fill inputs
        for src_attr, dst_atr in self.BAND_MAPPING.items():
            keys = [src_attr.format(t) for t in timestep_positions]
            values = np.array([float(row_d[key]) for key in keys], dtype=np.float32)
            idx_valid = values != NODATAVALUE
            if dst_atr in S2_BANDS:
                s2[..., S2_BANDS.index(dst_atr)] = values
            elif dst_atr in S1_BANDS:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
                s1[..., S1_BANDS.index(dst_atr)] = values
            elif dst_atr == "precipitation":
                # scaling, and AgERA5 is in mm, prometheo convention expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
                meteo[..., METEO_BANDS.index(dst_atr)] = values
            elif dst_atr == "temperature":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
                meteo[..., METEO_BANDS.index(dst_atr)] = values
            elif dst_atr in DEM_BANDS:
                values = values[0]  # dem is not temporal
                dem[..., DEM_BANDS.index(dst_atr)] = values
            else:
                raise ValueError(f"Unknown band {dst_atr}")

        # Apply masking if configured
        if self.masking_config and self.masking_config.enable:
            s1, s2, meteo, dem = self._apply_masking(s1, s2, meteo, dem)
        return dict(
            s1=s1, s2=s2, meteo=meteo, dem=dem, latlon=latlon, timestamps=timestamps
        )

    def initialize_inputs(self):
        s1 = np.full(
            (1, 1, self.num_timesteps, len(S1_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(S1_BANDS)]
        s2 = np.full(
            (1, 1, self.num_timesteps, len(S2_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(S2_BANDS)]
        meteo = np.full(
            (1, 1, self.num_timesteps, len(METEO_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(METEO_BANDS)]
        dem = np.full(
            (1, 1, len(DEM_BANDS)), fill_value=NODATAVALUE, dtype=np.float32
        )  # [H, W, len(DEM_BANDS)]

        return s1, s2, meteo, dem

    def _apply_masking(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        meteo: np.ndarray,
        dem: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply sensor/timestep masking according to the masking_config.

        Rules applied in order:
        1. Full S1 dropout (overrides timestep dropout).
        2. Per-timestep S1 dropout.
        3. S2 contiguous cloud block.
        4. Per-timestep S2 cloud dropout.
        5. Joint S1/S2 guard: if S1 and S2 would both end up fully
           masked-or-missing, one synthetically masked timestep is restored.
        6. Per-timestep meteo dropout.
        7. DEM dropout.

        The S1 and S2 dropouts are drawn as boolean masks first so the joint
        guard can repair them before any values are overwritten.
        """
        # Guard: if masking_config is None (should not happen when enable checked)
        if self.masking_config is None:
            return s1, s2, meteo, dem
        cfg: SensorMaskingConfig = self.masking_config  # type narrowing for mypy
        T = self.num_timesteps

        # Timesteps already missing in the input data (all bands NODATAVALUE)
        s1_missing = np.all(s1[0, 0] == NODATAVALUE, axis=-1)
        s2_missing = np.all(s2[0, 0] == NODATAVALUE, axis=-1)

        # 1. Full S1 dropout / 2. per-timestep S1 dropout
        s1_drop = np.zeros(T, dtype=bool)
        if np.random.rand() < cfg.s1_full_dropout_prob:
            s1_drop[:] = True
        elif cfg.s1_timestep_dropout_prob > 0:
            s1_drop = np.random.rand(T) < cfg.s1_timestep_dropout_prob

        # 3. S2 contiguous cloud block / 4. per-timestep S2 cloud dropout
        s2_drop = np.zeros(T, dtype=bool)
        if cfg.s2_cloud_block_prob > 0 and np.random.rand() < cfg.s2_cloud_block_prob:
            block_len = np.random.randint(
                cfg.s2_cloud_block_min, cfg.s2_cloud_block_max + 1
            )
            if block_len >= T:
                s2_drop[:] = True
            else:
                start = np.random.randint(0, T - block_len + 1)
                s2_drop[start : start + block_len] = True
        if cfg.s2_cloud_timestep_prob > 0:
            s2_drop |= np.random.rand(T) < cfg.s2_cloud_timestep_prob

        # 5. Joint S1/S2 guard
        s1_drop, s2_drop = self._rescue_joint_s1_s2_wipe(
            s1_drop, s2_drop, s1_missing, s2_missing
        )

        if s1_drop.any():
            s1[..., s1_drop, :] = NODATAVALUE
        if s2_drop.any():
            s2[..., s2_drop, :] = NODATAVALUE

        # 6. Meteo per-timestep dropout
        if cfg.meteo_timestep_dropout_prob > 0:
            meteo_mask = np.random.rand(T) < cfg.meteo_timestep_dropout_prob
            if meteo_mask.any():
                meteo[..., meteo_mask, :] = NODATAVALUE
                # logger.debug(
                #     f"Applied meteo timestep dropout on {meteo_mask.sum()} timesteps"
                # )

        # 7. DEM dropout
        if cfg.dem_dropout_prob > 0 and np.random.rand() < cfg.dem_dropout_prob:
            dem[:] = NODATAVALUE
            # logger.debug("Applied DEM dropout")

        return s1, s2, meteo, dem

    def _rescue_joint_s1_s2_wipe(
        self,
        s1_drop: np.ndarray,
        s2_drop: np.ndarray,
        s1_missing: np.ndarray,
        s2_missing: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure S1 and S2 never end up both fully masked-or-missing.

        If the drawn dropout masks, combined with gaps already present in the
        data, would leave neither sensor with a single valid timestep, one
        synthetically dropped timestep is restored. S2 is restored in
        preference to S1 because the dominant violating path is the explicit
        full-S1-dropout draw, whose semantics should stay intact. A sensor
        whose full elimination is intentional (dropout probability of 1.0,
        e.g. disable_s1/disable_s2 experiments) is never restored.
        """
        cfg: SensorMaskingConfig = self.masking_config  # type: ignore[assignment]
        if not (s1_drop | s1_missing).all() or not (s2_drop | s2_missing).all():
            return s1_drop, s2_drop

        candidates = [
            (
                s2_drop,
                np.flatnonzero(s2_drop & ~s2_missing),
                cfg.s2_cloud_timestep_prob >= 1.0,
            ),
            (
                s1_drop,
                np.flatnonzero(s1_drop & ~s1_missing),
                cfg.s1_full_dropout_prob >= 1.0,
            ),
        ]
        for drop, restorable, intentional in candidates:
            if restorable.size and not intentional:
                drop[np.random.choice(restorable)] = False
                return s1_drop, s2_drop

        logger.warning(
            "Sample has S1 and S2 fully masked-or-missing and no restorable "
            "timesteps; the joint S1/S2 guard cannot be enforced for this sample."
        )
        return s1_drop, s2_drop

    def _build_sample_attrs(
        self,
        row: Mapping[str, Any],
        valid_position: int,
        timestamps: np.ndarray,
        season_ids: Sequence[str],
        season_windows: Optional[Mapping[str, SeasonWindow]],
        season_engine: SeasonEngine,
        derive_from_calendar: bool,
        label_datetime: Optional[np.datetime64] = None,
    ) -> Dict[str, Any]:
        """Build sample attributes dictionary.

        Parameters
        ----------
        row : Mapping[str, Any]
            input row from the dataframe
        valid_position : int
            Valid position index within the time series
        timestamps : np.ndarray
            Array of timestamps corresponding to the time series
        season_ids : Sequence[str]
            List of season identifiers
        season_windows : Optional[Mapping[str, SeasonWindow]]
            Optional mapping with explicit (start, end) datetimes per season id.
        season_engine : SeasonEngine
            Execution strategy for season metadata (manual/calendar/off).
        derive_from_calendar : bool
            Whether to derive season information from official crop calendars
        label_datetime : Optional[np.datetime64], optional
            Label datetime, by default None

        Returns
        -------
        Dict[str, Any]
            Sample attributes dictionary
        """
        attrs: Dict[str, Any] = {}
        for key in SAMPLE_ATTR_COLUMNS:
            value = row.get(key)
            if key == "region":
                # Always emit "region" with a fallback so _collate_attrs sees a
                # consistent key across all samples in a batch, even when some
                # rows have a missing/NaN region value. Read from the configured
                # region_column (may differ from "region" if parameterised).
                value = row.get(self._region_column)
                attrs["region"] = (
                    str(value)
                    if (value is not None and not _is_missing_value(value))
                    else "unknown"
                )
            elif key in row and not _is_missing_value(value):
                attrs[key] = value

        label_task = row.get("label_task")
        if label_task is not None and not _is_missing_value(label_task):
            attrs["label_task"] = label_task

        attrs["valid_position"] = int(valid_position)

        if season_engine == "off":
            attrs["season_masks"] = None
            attrs["in_seasons"] = None
            return attrs

        season_masks, in_seasons = self._compute_season_metadata(
            row=row,
            timestamps=timestamps,
            season_ids=season_ids,
            season_windows=season_windows,
            derive_from_calendar=derive_from_calendar,
            label_datetime=label_datetime,
        )
        attrs["season_masks"] = season_masks
        if in_seasons is not None:
            attrs["in_seasons"] = in_seasons

        return attrs

    def _compute_season_metadata(
        self,
        row: Mapping[str, Any],
        timestamps: np.ndarray,
        season_ids: Sequence[str],
        season_windows: Optional[Mapping[str, SeasonWindow]],
        derive_from_calendar: bool,
        label_datetime: Optional[np.datetime64],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resolve per-season temporal masks for a single sample.

        This helper consolidates the three supported season sources:
        1. Manual windows (`season_windows`): we normalize the dates and build masks
           by repeating those month/day windows for every year present in
           ``timestamps``. When manual windows are provided the configuration is
           treated as fully manual—every requested season id must have a window
           and no crop-calendar lookup happens.
        2. Calendar-driven windows (`derive_from_calendar=True`): for each requested
           season id we query ``worldcereal.seasons`` using the sample lat/lon and
           an inferred target year (label datetime when available, otherwise the
           final timestep's year). This path requires lat/lon coordinates and a
           label datetime so we can determine whether the label falls inside the
           season window.
        3. Global defaults: when neither of the above are specified we fall back to
           ``GLOBAL_SEASON_IDS``.

        The method returns a boolean mask ``(num_seasons, num_timesteps)`` and,
        when ``label_datetime`` is provided, a boolean vector indicating whether
        that label falls inside each season window. Masks are only emitted for
        seasons where the sample covers *every* composite timestep inside the
        season window (monthly or dekadal); partial coverage results in an
        all-False mask and a False entry in ``in_seasons``.
        """

        timestamps_arr = np.asarray(timestamps)
        normalized_windows = season_windows or {}
        # Prefer the user's explicit `season_ids`; if not provided, fall back to the
        # manual window keys, and only if both are missing do we use the global defaults.
        if season_ids:
            target_seasons: Tuple[str, ...] = tuple(season_ids)
        elif normalized_windows:
            target_seasons = tuple(normalized_windows.keys())
        else:
            target_seasons = GLOBAL_SEASON_IDS

        num_seasons = len(target_seasons)
        if timestamps_arr.size == 0:
            masks = _default_season_mask(0, num_seasons)
            return masks, None

        num_timesteps = int(timestamps_arr.shape[0])
        composite_dates = _timestamps_to_datetime_array(timestamps_arr)
        if composite_dates.size == 0:
            masks = _default_season_mask(num_timesteps, num_seasons)
            in_seasons = (
                np.zeros(num_seasons, dtype=bool)
                if label_datetime is not None
                else None
            )
            return masks, in_seasons

        # LC-only datasets (ref_id ending _100/_101) carry no crop-type annotation
        # by design.  Their valid_time is meaningless relative to any crop season,
        # so we short-circuit the calendar/window logic and return all-True masks.
        # This ensures nocrop samples from LC datasets always provide supervision
        # signal for the crop-type head without being gated by season boundaries.
        if _is_lc_only_dataset(str(row.get("ref_id", ""))):
            all_true = np.ones((num_seasons, num_timesteps), dtype=bool)
            in_seasons = (
                np.ones(num_seasons, dtype=bool) if label_datetime is not None else None
            )
            return all_true, in_seasons

        lat = _extract_float(row.get("lat"))
        lon = _extract_float(row.get("lon"))

        manual_season_names = set(normalized_windows.keys())
        if manual_season_names:
            # When users provide explicit windows we treat the configuration as fully manual;
            # any season listed in `season_ids` must therefore have an accompanying window.
            missing_manual = [
                season for season in target_seasons if season not in manual_season_names
            ]
            if missing_manual:
                raise ValueError(
                    "season_ids references seasons without manual windows; mixing "
                    "manual and calendar seasons is not supported. Missing windows for "
                    f"{tuple(missing_manual)}."
                )
            has_calendar_seasons = False
        else:
            has_calendar_seasons = bool(target_seasons)
        if has_calendar_seasons and not derive_from_calendar:
            missing = tuple(
                season for season in target_seasons if season not in manual_season_names
            )
            raise ValueError(
                "Season calendar derivation disabled but required for seasons "
                f"{missing}. Configure season_calendar_mode='calendar' or provide windows."
            )
        if derive_from_calendar and has_calendar_seasons and label_datetime is None:
            sample_id = row.get("sample_id")
            raise ValueError(
                "WorldCerealDataset requires a label datetime for season metadata"
                + (f" (sample_id={sample_id})" if sample_id is not None else "")
            )
        if has_calendar_seasons and derive_from_calendar:
            if lat is None or lon is None:
                sample_id = row.get("sample_id")
                raise ValueError(
                    "WorldCerealDataset requires lat/lon coordinates to derive season calendars"
                    + (f" (sample_id={sample_id})" if sample_id is not None else "")
                )

        if label_datetime is not None:
            target_year = pd.Timestamp(label_datetime).year
        else:
            target_year = int(timestamps_arr[-1][2])

        season_mask_list: List[np.ndarray] = []
        in_flags: List[bool] = []
        for season in target_seasons:
            manual_window = normalized_windows.get(season)
            if manual_window is not None:
                mask, in_flag = self._season_mask_from_window(
                    window=manual_window,
                    composite_dates=composite_dates,
                    label_datetime=label_datetime,
                )
            else:
                mask, in_flag = self._season_mask_from_calendar(
                    season_name=season,
                    composite_dates=composite_dates,
                    target_year=target_year,
                    row=row,
                    label_datetime=label_datetime,
                    lat=lat,
                    lon=lon,
                )

            season_mask_list.append(mask)
            in_flags.append(in_flag)

        if season_mask_list:
            masks_arr = np.stack(season_mask_list, axis=0)
        else:
            masks_arr = np.zeros((0, num_timesteps), dtype=bool)

        in_seasons = (
            np.array(in_flags, dtype=bool) if label_datetime is not None else None
        )
        return masks_arr, in_seasons

    def _season_mask_from_window(
        self,
        window: SeasonWindow,
        composite_dates: np.ndarray,
        label_datetime: Optional[np.datetime64],
    ) -> Tuple[np.ndarray, bool]:
        """Build a mask for a manual ``SeasonWindow`` when the sample spans it."""
        mask = np.zeros(composite_dates.shape, dtype=bool)
        if composite_dates.size == 0:
            return mask, False

        years = composite_dates.astype("datetime64[Y]").astype(int) + 1970
        base_years = set(int(y) for y in np.unique(years).tolist())
        if window.year_offset > 0 and base_years:
            shifted = {year - window.year_offset for year in base_years}
            base_years.update(shifted)

        cycles: List[Tuple[np.datetime64, np.datetime64]] = []
        for year in sorted(base_years):
            start_dt = coerce_date_for_year(year, window.start_month, window.start_day)
            end_year = year + window.year_offset
            end_dt = coerce_date_for_year(end_year, window.end_month, window.end_day)
            start_aligned = align_to_composite_window(start_dt, self.timestep_freq)
            end_aligned = align_to_composite_window(end_dt, self.timestep_freq)
            if end_aligned < start_aligned:
                continue
            slots = enumerate_composite_slots(
                start_aligned, end_aligned, self.timestep_freq
            )
            if not slots:
                continue
            cycle_mask = (composite_dates >= start_aligned) & (
                composite_dates <= end_aligned
            )
            n_required = max(1, round(len(slots) * self.min_season_coverage))
            if int(cycle_mask.sum()) < n_required:
                continue
            # Store the raw cycle bounds so the in_season check can compare the
            # exact label against composite-aligned edges (date_in_season),
            # consistent with the season-alignment pre-filter and the mask.
            cycles.append((start_dt, end_dt))
            mask |= cycle_mask

        in_flag = label_datetime is not None and any(
            date_in_season(label_datetime, start_dt, end_dt, freq=self.timestep_freq)
            for start_dt, end_dt in cycles
        )

        return mask.astype(bool, copy=False), in_flag

    def _season_mask_from_calendar(
        self,
        season_name: str,
        composite_dates: np.ndarray,
        target_year: int,
        row: Mapping[str, Any],
        label_datetime: Optional[np.datetime64],
        lat: Optional[float],
        lon: Optional[float],
    ) -> Tuple[np.ndarray, bool]:
        """Build a mask from calendar dates only when every timestep is present."""
        if lat is None or lon is None:
            sample_id = row.get("sample_id", "n/a")
            raise RuntimeError(
                "Season calendar derivation was requested but lat/lon were missing "
                f"(sample_id={sample_id}, season={season_name})."
            )

        try:
            start_dt, end_dt = self._season_context_for(
                season_name,
                row,
                target_year,
                lat,
                lon,
                label_datetime=label_datetime,
            )
        except ValueError as exc:
            # Nodata DOY values for this season at this location — treat as
            # "season not available" rather than crashing.  Return an all-zeros
            # mask so the sample is simply not supervised for this season.
            sample_id = row.get("sample_id", "n/a")
            logger.error(
                f"Season '{season_name}' unavailable for sample {sample_id}: {exc}. "
                f"Returning empty mask."
            )
            return np.zeros_like(composite_dates, dtype=bool), False
        except Exception as exc:  # pragma: no cover - guard rare failures
            sample_id = row.get("sample_id", "n/a")
            raise RuntimeError(
                f"Failed to derive season '{season_name}' for sample {sample_id}: {exc}"
            ) from exc

        start_aligned = align_to_composite_window(start_dt, self.timestep_freq)
        end_aligned = align_to_composite_window(end_dt, self.timestep_freq)
        slots = enumerate_composite_slots(
            start_aligned, end_aligned, self.timestep_freq
        )
        partial_mask = (composite_dates >= start_aligned) & (
            composite_dates <= end_aligned
        )
        n_required = max(1, round(len(slots) * self.min_season_coverage))
        meets_threshold = int(partial_mask.sum()) >= n_required
        mask = (
            partial_mask
            if meets_threshold
            else np.zeros_like(composite_dates, dtype=bool)
        )
        in_flag = (
            date_in_season(label_datetime, start_dt, end_dt, freq=self.timestep_freq)
            if (label_datetime is not None and meets_threshold)
            else False
        )
        return mask.astype(bool, copy=False), in_flag

    def _season_context_for(
        self,
        season_id: str,
        row: Mapping[str, Any],
        year: int,
        lat: float,
        lon: float,
        label_datetime: Optional[np.datetime64] = None,
    ) -> Tuple[np.datetime64, np.datetime64]:
        """Fetch (start, end) dates for a season/grid cell from the lookup."""

        lat_center, lon_center = _snap_latlon_to_calendar_grid(lat, lon)
        try:
            sos_col, eos_col = SEASONALITY_COLUMN_MAP[season_id]
        except KeyError as exc:
            raise ValueError(
                f"Season '{season_id}' is not available in the seasonality lookup. "
                f"Known seasons: {sorted(SEASONALITY_COLUMN_MAP)}"
            ) from exc

        table = _ensure_seasonality_lookup()
        if sos_col not in table.columns or eos_col not in table.columns:
            raise ValueError(
                f"Season '{season_id}' requires columns ({sos_col}, {eos_col}) "
                "but they are not present in the seasonality lookup parquet. "
                "Regenerate the parquet with the required bands included."
            )
        try:
            doy_row = table.loc[(lat_center, lon_center)]
        except KeyError as exc:  # pragma: no cover - unexpected gaps
            lat_vals = table.index.get_level_values("lat").to_numpy()
            lon_vals = table.index.get_level_values("lon").to_numpy()
            if lat_vals.size == 0:
                raise ValueError(
                    "No seasonality record found for snapped lat/lon "
                    f"({lat_center}, {lon_center})."
                ) from exc
            distances = (lat_vals - lat_center) ** 2 + (lon_vals - lon_center) ** 2
            best_idx = int(distances.argmin())
            fallback_key = (lat_vals[best_idx], lon_vals[best_idx])
            logger.error(
                f"Seasonality lookup missing ({lat_center}, {lon_center}); using nearest cell ({fallback_key[0]}, {fallback_key[1]})."
            )
            lat_center, lon_center = float(fallback_key[0]), float(fallback_key[1])
            doy_row = table.iloc[best_idx]

        sos_doy = int(doy_row[sos_col])
        eos_doy = int(doy_row[eos_col])
        if sos_doy <= 0 or eos_doy <= 0:
            sample_id = row.get("sample_id", "n/a")
            raise ValueError(
                "Seasonality lookup returned nodata DOY values for "
                f"season '{season_id}' (sample_id={sample_id})."
            )

        # For year-crossing seasons (SOS DOY > EOS DOY), season_doys_to_dates_refyear
        # places the EOS in ref_year. When target_year is derived from label_datetime.year,
        # this is only correct if the label falls early in the year (before/at EOS DOY).
        # If the label falls later (after EOS DOY), the relevant season instance ends in
        # year+1, so we must increment ref_year accordingly.
        ref_year = year
        if sos_doy > eos_doy and label_datetime is not None:
            label_doy = pd.Timestamp(label_datetime).day_of_year
            if label_doy > eos_doy:
                ref_year = year + 1

        start_dt, end_dt = season_doys_to_dates_refyear(sos_doy, eos_doy, ref_year)
        return (
            np.datetime64(start_dt, "D"),
            np.datetime64(end_dt, "D"),
        )


class WorldCerealLabelledDataset(WorldCerealDataset):
    def __init__(
        self,
        dataframe,
        task_type: Literal["binary", "multiclass"] = "binary",
        num_outputs: int = 1,
        emit_label_tensor: bool = True,
        classes_list: Union[np.ndarray, List[str]] = [],
        time_explicit: bool = False,
        label_jitter: int = 0,  # ± timesteps to jitter true label pos, for time_explicit only
        label_window: int = 0,  # ± timesteps to expand around label pos (true or moved), for time_explicit only
        return_sample_id: bool = False,
        season_calendar_mode: SeasonCalendarMode = "auto",
        season_ids: Optional[Sequence[str]] = None,
        season_windows: Optional[Mapping[str, Tuple[Any, Any]]] = None,
        region_column: str = "region",
        **kwargs,
    ):
        """Labelled version of WorldCerealDataset for supervised training.
        Additional arguments are explained below.

        Parameters
        ----------
        num_outputs : int, optional
            number of outputs to supervise training on, by default 1
        emit_label_tensor : bool, optional
            whether to emit the label tensor in __getitem__, by default True.
            If False, no label tensor is created or returned.
        classes_list : List, optional
            list of column names in the dataframe containing class labels for multiclass tasks,
            used to extract labels from each row of the dataframe, by default []
        time_explicit : bool, optional
            if True, labels respect the full temporal dimension
            to have temporally explicit outputs, by default False
        label_jitter : int, optional
            ± timesteps to jitter true label pos, for time_explicit only, by default 0.
            Only used if `time_explicit` is True.
        label_window : int, optional
            ± timesteps to expand around label pos (true or moved), for time_explicit only, by default 0.
            Only used if `time_explicit` is True.
        return_sample_id : bool, optional
            whether to return the sample_id in the output, by default False.
            If True, the sample_id will be included in the output as a separate element.
        season_calendar_mode : {"calendar", "custom", "auto", "off"}, optional
            Controls how `season_masks`/`in_seasons` are derived.
            - "calendar": use `worldcereal.seasons` lookup for every season id.
            - "custom": require `season_windows` and use them directly.
            - "auto" (default): use custom windows when provided, otherwise fall back to calendars.
            - "off": skip season metadata entirely (attrs contain None).
        season_ids : Optional[Sequence[str]], optional
            For calendar-driven modes, select which official seasons to fetch
            (defaults to `GLOBAL_SEASON_IDS`). When `season_windows` is provided,
            these ids must be a subset of the manual window keys—mixing manual
            windows with unspecified calendar ids is not supported.
        season_windows : Optional[Mapping[str, Tuple[Any, Any]]], optional
            Optional mapping from season identifier to a `(start, end)` datetime-like
            tuple. When specified, those seasons bypass crop calendars and use the
            provided windows directly. Dates repeat annually—only the month/day
            information (with an optional cross-year span) is used when building
            masks, so a window defined for 2021 will apply to samples from any year.
        """
        assert task_type in [
            "binary",
            "multiclass",
        ], f"Invalid task type `{task_type}` for labelled dataset"

        super().__init__(
            dataframe,
            task_type=task_type,
            num_outputs=num_outputs,
            **kwargs,
        )
        self.classes_list = classes_list
        self.emit_label_tensor = emit_label_tensor
        self.time_explicit = time_explicit
        self.label_jitter = label_jitter
        self.label_window = label_window
        self.return_sample_id = return_sample_id
        self._region_column = region_column
        self._season_windows: Dict[str, SeasonWindow] = _normalize_season_windows(
            season_windows
        )
        self._season_engine: SeasonEngine = _resolve_season_engine(
            season_calendar_mode, bool(self._season_windows)
        )
        if season_ids:
            resolved_seasons = tuple(season_ids)
        elif self._season_windows:
            resolved_seasons = tuple(self._season_windows.keys())
        else:
            resolved_seasons = GLOBAL_SEASON_IDS
        self._season_ids: Tuple[str, ...] = resolved_seasons

        if self.return_sample_id and "sample_id" not in self.dataframe.columns:
            raise ValueError(
                "`return_sample_id` is True, but 'sample_id' column not found in dataframe."
            )

        if self._season_engine == "manual" and self._season_windows:
            filtered_df, dropped = _filter_frame_by_manual_windows(
                self.dataframe,
                self._season_windows,
                context=self.__class__.__name__,
                timestep_freq=self.timestep_freq,
            )
            if dropped:
                logger.info(
                    f"{self.__class__.__name__}: proceeding with {len(filtered_df)} samples after enforcing manual season window(s)."
                )
            self.dataframe = filtered_df

        # Vectorized batch-fetch cache: precomputes numpy views of the expensive
        # dataframe-derived pieces so whole training batches can be assembled
        # without per-sample pandas access. Unsupported configurations fall back
        # to the canonical per-sample path below.
        self._batch_cache: Optional[Dict[str, Any]] = None
        self._batch_rng: Optional[np.random.Generator] = None
        self._batch_rng_key: Optional[Tuple[int, int]] = None
        try:
            self._build_fast_batch_cache()
        except Exception as exc:  # noqa: BLE001 - never break dataset construction
            logger.warning(
                f"Fast batched fetching disabled ({exc!r}); "
                "falling back to per-sample loading."
            )
            self._batch_cache = None

    def _decode_task_index(self, idx: int) -> Tuple[int, Optional[str]]:
        """Decode sampler virtual indices into dataframe rows and task overrides."""
        # During task-configurable training, SeasonalTaskBatchSampler tells each sample
        # which head it should supervise (landcover or croptype) by shifting its
        # index: indices in [N..2N) are landcover, [2N..3N) are croptype.
        # We decode the real row index and the intended task here, so the
        # dataframe itself never needs to be mutated (important for multi-worker
        # data loading). Plain [0..N) indices are used during validation/test.
        n = len(self.dataframe)
        if idx >= 2 * n:
            return idx - 2 * n, "croptype"
        if idx >= n:
            return idx - n, "landcover"
        return idx, None

    def _sample_from_row(
        self, row: Dict[str, Any], task_override: Optional[str] = None
    ) -> Tuple[Predictors, Dict[str, Any]]:
        """Build one labelled sample using the canonical per-sample helpers."""
        timestep_positions, valid_position = self.get_timestep_positions(row)
        relative_valid = valid_position - timestep_positions[0]
        inputs = self.get_inputs(row, timestep_positions)

        if self.emit_label_tensor:
            label = self.get_label(
                row,
                task_type=self.task_type,
                classes_list=self.classes_list,
                valid_position=relative_valid,
            )
        else:
            label = None

        attrs = self._build_sample_attrs(
            row=row,
            valid_position=relative_valid,
            timestamps=inputs["timestamps"],
            season_ids=self._season_ids,
            season_windows=self._season_windows,
            season_engine=self._season_engine,
            derive_from_calendar=self._season_engine == "calendar",
            label_datetime=_resolve_label_datetime(row),
        )

        # Sampler-driven task assignment always overrides the dataframe value so
        # that each sample supervises exactly one head per batch.
        if task_override is not None:
            attrs["label_task"] = task_override

        return Predictors(**inputs, label=label), attrs

    def __getitem__(self, idx):
        real_idx, task_override = self._decode_task_index(int(idx))
        row = pd.Series.to_dict(self.dataframe.iloc[real_idx, :])
        return self._sample_from_row(row, task_override)

    def initialize_label(self):
        tsteps = self.num_timesteps if self.time_explicit else 1
        label = np.full(
            (1, 1, tsteps, 1),
            fill_value=NODATAVALUE,
            dtype=np.int32,
        )  # [H, W, T or 1, 1]

        return label

    def get_label(
        self,
        row_d: Dict,
        task_type: Literal["binary", "multiclass"] = "binary",
        classes_list: Optional[List] = None,
        valid_position: Optional[
            Union[int, Sequence[int]]
        ] = None,  # TO DO: this can also be a list of positions
    ) -> np.ndarray:
        """Get the label for the given row. Label is a 2D array based on
        the number of timesteps and number of outputs. If time_explicit is False,
        the number of timesteps will be set to 1.

        Parameters
        ----------
        row_d : Dict
            input row as a dictionary
        task_type : str, optional
            task type to infer labels from, by default "binary"
        classes_list : Optional[List], optional
            list of column names in the dataframe containing class labels for multiclass tasks,
            must be provided if task_type is "multiclass", by default None
        valid_position : int, optional
            the ‘true’ timestep index where the label lives, by default None.
            If provided and `time_explicit` is True,
            only the label at the corresponding timestep will be
            set while other timesteps will be set to NODATAVALUE.
            We’ll optionally jitter it and/or expand it into a small time‐window.

        Returns
        -------
        np.ndarray
            label array
        """

        label = self.initialize_label()
        T = self.num_timesteps

        # 1) determine base position (single int) or all-positions if not time_explicit
        base_idxs: List[int]
        if not self.time_explicit:
            base_idxs = [0]
        else:
            if valid_position is None:
                # putting label at every timestep
                base_idxs = list(range(T))
            elif isinstance(valid_position, (list, tuple, np.ndarray)):
                # bring into a flat Python list of ints
                if isinstance(valid_position, np.ndarray):
                    seq: List[int] = valid_position.astype(int).tolist()
                else:
                    seq = [int(x) for x in valid_position]
                # Apply either jittering or label_window, but not both
                if self.label_jitter > 0 and self.label_window > 0:
                    apply_jitter = np.random.choice([True, False])
                else:
                    apply_jitter = self.label_jitter > 0

                if apply_jitter:
                    # one global jitter shift
                    shift = np.random.randint(-self.label_jitter, self.label_jitter + 1)
                    seq = [int(np.clip(p + shift, 0, T - 1)) for p in seq]
                elif self.label_window > 0:
                    # one contiguous window around the min→max of seq
                    mn = min(seq)
                    mx = max(seq)
                    start = max(0, mn - self.label_window)
                    end = min(T - 1, mx + self.label_window)
                    base_idxs = list(range(start, end + 1))
                else:
                    base_idxs = seq
            else:
                # apply jitter
                # scalar valid_position must be an int here
                assert isinstance(valid_position, int), (
                    f"Expected single int valid_position, got {type(valid_position)}"
                )
                p = valid_position
                if self.label_jitter > 0:
                    shift = np.random.randint(-self.label_jitter, self.label_jitter + 1)
                    p = int(np.clip(p + shift, 0, T - 1))
                # apply window expansion
                if self.label_window > 0:
                    start = max(0, p - self.label_window)
                    end = min(T - 1, p + self.label_window)
                    base_idxs = list(range(start, end + 1))
                else:
                    base_idxs = [p]

        valid_idx = np.array(base_idxs, dtype=int)

        # 2) set the labels at those indices
        if task_type == "binary":
            label[0, 0, valid_idx, 0] = int(
                not row_d["finetune_class"].startswith("not_")
            )
        elif task_type == "multiclass":
            if not classes_list:
                raise ValueError("classes_list should be provided for multiclass task")
            label[0, 0, valid_idx, 0] = classes_list.index(row_d["finetune_class"])

        return label

    # ------------------------------------------------------------------
    @staticmethod
    def _factorize_attr_column(col: pd.Series) -> Tuple[np.ndarray, List[Any]]:
        """Factorize an object column into codes plus a unique-value list.

        Returns codes shifted by +1 so that 0 maps to the missing value (None).
        Storing per-row small int codes instead of object arrays keeps
        DataLoader workers from touching (and copy-on-write duplicating) the
        large string pools of the parent process.
        """
        codes, uniques = pd.factorize(col, use_na_sentinel=True)
        uniq_list: List[Any] = [None] + [
            None if _is_missing_value(u) else u for u in uniques
        ]
        return (codes + 1).astype(np.int32), uniq_list

    def _build_fast_batch_cache(self) -> None:
        """Precompute numpy arrays enabling vectorized `__getitems__`."""
        df = self.dataframe
        n_rows = len(df)
        if n_rows == 0:
            return
        if self.timestep_freq != "month":
            logger.info(
                "Fast batched fetching supports timestep_freq='month' only; "
                "using per-sample loading."
            )
            return
        if self._season_engine == "manual":
            logger.info(
                "Fast batched fetching does not support manual season windows; "
                "using per-sample loading."
            )
            return
        required_cols = {"available_timesteps", "valid_position", "start_date"}
        if not required_cols.issubset(df.columns):
            return

        avail = df["available_timesteps"].to_numpy(dtype=np.int64)
        vp = df["valid_position"].to_numpy(dtype=np.int64)
        if (avail < self.num_timesteps).any():
            logger.warning(
                "Fast batched fetching disabled: some rows have fewer available "
                "timesteps than num_timesteps; using per-sample loading."
            )
            return

        # ---- band cubes -------------------------------------------------
        max_ts = 0
        while any(
            template.format(max_ts) in df.columns
            for template in self.S1_S2_COLUMN_TEMPLATES
        ):
            max_ts += 1
        if max_ts == 0 or int(avail.max()) > max_ts:
            return

        temporal_templates = {
            src: dst for src, dst in self.BAND_MAPPING.items() if dst not in DEM_BANDS
        }
        for src in temporal_templates:
            for t in range(int(avail.max())):
                if src.format(t) not in df.columns:
                    logger.warning(
                        f"Fast batched fetching disabled: missing column "
                        f"{src.format(t)!r}; using per-sample loading."
                    )
                    return
        for src, dst in self.BAND_MAPPING.items():
            if dst in DEM_BANDS and src not in df.columns:
                logger.warning(
                    f"Fast batched fetching disabled: missing column {src!r}; "
                    "using per-sample loading."
                )
                return
        if "lat" not in df.columns or "lon" not in df.columns:
            return

        s2_src = [
            (src, S2_BANDS.index(dst))
            for src, dst in self.BAND_MAPPING.items()
            if dst in S2_BANDS
        ]
        s1_src = [
            (src, S1_BANDS.index(dst))
            for src, dst in self.BAND_MAPPING.items()
            if dst in S1_BANDS
        ]

        s2 = np.full((n_rows, max_ts, len(s2_src)), NODATAVALUE, dtype=np.float32)
        for j, (src, _) in enumerate(s2_src):
            for t in range(max_ts):
                col = src.format(t)
                if col in df.columns:
                    s2[:, t, j] = df[col].to_numpy(dtype=np.float32)
        s2_dst_idx = np.array([dst for _, dst in s2_src], dtype=np.int64)

        s1 = np.full((n_rows, max_ts, len(S1_BANDS)), NODATAVALUE, dtype=np.float32)
        for src, dst_idx in s1_src:
            for t in range(max_ts):
                col = src.format(t)
                if col in df.columns:
                    s1[:, t, dst_idx] = df[col].to_numpy(dtype=np.float32)

        # Joint S1/S2 data presence per (row, timestep) — computed on raw
        # values, before the dB conversion below (the NODATAVALUE sentinel is
        # preserved by all transforms, so this matches _window_has_s1_s2).
        presence = (s2 != NODATAVALUE).any(axis=-1) | (s1 != NODATAVALUE).any(axis=-1)

        # S1 dB conversion (valid positive values only, as in get_inputs)
        s1_valid = (s1 != NODATAVALUE) & (s1 > 0)
        s1[s1_valid] = 20.0 * np.log10(s1[s1_valid]) - 83.0

        meteo = np.full(
            (n_rows, max_ts, len(METEO_BANDS)), NODATAVALUE, dtype=np.float32
        )
        for src, dst in self.BAND_MAPPING.items():
            if dst not in METEO_BANDS:
                continue
            dst_idx = METEO_BANDS.index(dst)
            scale = 100.0 * 1000.0 if dst == "precipitation" else 100.0
            for t in range(max_ts):
                col = src.format(t)
                if col in df.columns:
                    vals = df[col].to_numpy(dtype=np.float32)
                    valid = vals != NODATAVALUE
                    out = vals.copy()
                    out[valid] = vals[valid] / scale
                    meteo[:, t, dst_idx] = out

        dem = np.full((n_rows, len(DEM_BANDS)), NODATAVALUE, dtype=np.float32)
        for src, dst in self.BAND_MAPPING.items():
            if dst in DEM_BANDS:
                dem[:, DEM_BANDS.index(dst)] = df[src].to_numpy(dtype=np.float32)

        latlon = np.stack(
            [
                df["lat"].to_numpy(dtype=np.float32),
                df["lon"].to_numpy(dtype=np.float32),
            ],
            axis=1,
        )

        # ---- timestamps --------------------------------------------------
        start_dates = pd.to_datetime(df["start_date"]).to_numpy()
        start_month_num = start_dates.astype("datetime64[M]").astype(np.int64)

        # ---- label metadata ----------------------------------------------
        label_dt = _label_datetime_series(df)
        cache: Dict[str, Any] = {
            "avail": avail,
            "vp": vp,
            "max_ts": max_ts,
            "s2": s2,
            "s2_dst_idx": s2_dst_idx,
            "s1": s1,
            "meteo": meteo,
            "dem": dem,
            "latlon": latlon,
            "presence": presence,
            "start_month_num": start_month_num,
        }

        # ---- season-calendar metadata (fixed per row/season) --------------
        if self._season_engine == "calendar" and self._season_ids:
            if label_dt.isna().any():
                logger.warning(
                    "Fast batched fetching disabled: some samples lack a label "
                    "datetime required for season calendars; using per-sample "
                    "loading."
                )
                return
            season_meta = self._build_season_calendar_cache(df, label_dt)
            if season_meta is None:
                return
            cache.update(season_meta)

        # ---- label values --------------------------------------------------
        if self.emit_label_tensor:
            if "finetune_class" not in df.columns:
                logger.warning(
                    "Fast batched fetching disabled: emit_label_tensor requires "
                    "a 'finetune_class' column; using per-sample loading."
                )
                return
            ft_codes, ft_uniques = self._factorize_attr_column(df["finetune_class"])
            cache["finetune_codes"] = ft_codes
            cache["finetune_uniques"] = ft_uniques

        # ---- attrs ----------------------------------------------------------
        attr_specs: List[Tuple[str, str, Any, Any]] = []
        for key in SAMPLE_ATTR_COLUMNS:
            if key == "region":
                continue
            if key not in df.columns:
                continue
            col = df[key]
            if pd.api.types.is_numeric_dtype(col.dtype):
                vals = col.to_numpy()
                # Match the per-sample path's collate dtypes: row scalars are
                # boxed to python float/int, which default_collate turns into
                # float64 / int64 tensors.
                if np.issubdtype(vals.dtype, np.floating):
                    vals = vals.astype(np.float64)
                elif np.issubdtype(vals.dtype, np.integer):
                    vals = vals.astype(np.int64)
                missing = vals == NODATAVALUE
                if np.issubdtype(vals.dtype, np.floating):
                    missing = missing | np.isnan(vals)
                attr_specs.append(
                    (key, "num", vals, missing if missing.any() else None)
                )
            else:
                codes, uniques = self._factorize_attr_column(col)
                attr_specs.append((key, "obj", codes, uniques))
        cache["attr_specs"] = attr_specs

        if self._region_column in df.columns:
            region_codes, region_uniques = self._factorize_attr_column(
                df[self._region_column]
            )
            region_uniques = [
                "unknown" if u is None else str(u) for u in region_uniques
            ]
        else:
            region_codes = np.zeros(n_rows, dtype=np.int32)
            region_uniques = ["unknown"]
        cache["region_codes"] = region_codes
        cache["region_uniques"] = region_uniques

        if "label_task" in df.columns:
            lt_codes, lt_uniques = self._factorize_attr_column(df["label_task"])
            cache["label_task_codes"] = lt_codes
            cache["label_task_uniques"] = lt_uniques
        else:
            cache["label_task_codes"] = None
            cache["label_task_uniques"] = None

        self._batch_cache = cache
        logger.info(
            f"{self.__class__.__name__}: fast batched fetching enabled "
            f"({n_rows} samples, {max_ts} timesteps)."
        )

    def _build_season_calendar_cache(
        self, df: pd.DataFrame, label_dt: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Precompute per-row, per-season calendar windows (month numbers)."""
        n_rows = len(df)
        seasons = tuple(self._season_ids)
        for season in seasons:
            if season not in SEASONALITY_COLUMN_MAP:
                logger.warning(
                    f"Fast batched fetching disabled: season {season!r} not in "
                    "seasonality lookup; using per-sample loading."
                )
                return None

        table = _ensure_seasonality_lookup()
        lat = df["lat"].to_numpy(dtype=np.float64)
        lon = df["lon"].to_numpy(dtype=np.float64)
        lat_c = (
            np.floor(np.clip(lat, *SEASONALITY_LAT_RANGE) * 2.0) / 2.0
        ) + 0.25
        lon_c = (
            np.floor(np.clip(lon, *SEASONALITY_LON_RANGE) * 2.0) / 2.0
        ) + 0.25

        key_index = pd.MultiIndex.from_arrays([lat_c, lon_c], names=["lat", "lon"])
        joined = table.reindex(key_index)

        # Nearest-cell fallback for grid cells absent from the lookup (mirrors
        # the per-sample KeyError fallback, logged once per unique cell).
        missing_rows = joined[list(SEASONALITY_LOOKUP_COLUMNS)].isna().all(axis=1)
        if missing_rows.to_numpy().any():
            lat_vals = table.index.get_level_values("lat").to_numpy()
            lon_vals = table.index.get_level_values("lon").to_numpy()
            missing_pos = np.flatnonzero(missing_rows.to_numpy())
            missing_cells = {
                (float(lat_c[i]), float(lon_c[i])) for i in missing_pos
            }
            cell_to_row = {}
            for cell_lat, cell_lon in missing_cells:
                distances = (lat_vals - cell_lat) ** 2 + (lon_vals - cell_lon) ** 2
                best_idx = int(distances.argmin())
                cell_to_row[(cell_lat, cell_lon)] = best_idx
                logger.error(
                    f"Seasonality lookup missing ({cell_lat}, {cell_lon}); using "
                    f"nearest cell ({lat_vals[best_idx]}, {lon_vals[best_idx]})."
                )
            joined = joined.reset_index(drop=True)
            for i in missing_pos:
                joined.iloc[i] = table.iloc[
                    cell_to_row[(float(lat_c[i]), float(lon_c[i]))]
                ]

        label_days = label_dt.to_numpy().astype("datetime64[D]")
        label_month_num = label_days.astype("datetime64[M]").astype(np.int64)
        label_year = label_days.astype("datetime64[Y]").astype(np.int64) + 1970
        year_start = label_days.astype("datetime64[Y]").astype("datetime64[D]")
        label_doy = (label_days - year_start).astype(np.int64) + 1

        num_seasons = len(seasons)
        season_start_m = np.zeros((n_rows, num_seasons), dtype=np.int64)
        season_end_m = np.zeros((n_rows, num_seasons), dtype=np.int64)
        season_n_slots = np.ones((n_rows, num_seasons), dtype=np.int64)
        season_invalid = np.zeros((n_rows, num_seasons), dtype=bool)
        season_in_raw = np.zeros((n_rows, num_seasons), dtype=bool)

        for s_idx, season in enumerate(seasons):
            sos_col, eos_col = SEASONALITY_COLUMN_MAP[season]
            if sos_col not in joined.columns or eos_col not in joined.columns:
                logger.warning(
                    f"Fast batched fetching disabled: seasonality lookup lacks "
                    f"columns for season {season!r}; using per-sample loading."
                )
                return None
            sos = joined[sos_col].to_numpy(dtype=np.float64)
            eos = joined[eos_col].to_numpy(dtype=np.float64)
            invalid = (
                ~np.isfinite(sos) | ~np.isfinite(eos) | (sos <= 0) | (eos <= 0)
            )
            sos_i = np.where(invalid, 1, sos).astype(np.int64)
            eos_i = np.where(invalid, 1, eos).astype(np.int64)

            # For year-crossing seasons, shift ref year when the label falls
            # after EOS (mirrors _season_context_for).
            ref_year = label_year + (
                (sos_i > eos_i) & (label_doy > eos_i)
            ).astype(np.int64)

            # season_doys_to_dates_refyear, vectorized:
            #   end = Jan 1 of ref_year + eos days; start = end - duration
            ref_year_start = (ref_year - 1970).astype("datetime64[Y]").astype(
                "datetime64[D]"
            )
            end_date = ref_year_start + eos_i.astype("timedelta64[D]")
            duration = np.where(sos_i < eos_i, eos_i - sos_i, eos_i + 365 - sos_i)
            start_date = end_date - duration.astype("timedelta64[D]")

            start_m = start_date.astype("datetime64[M]").astype(np.int64)
            end_m = end_date.astype("datetime64[M]").astype(np.int64)

            season_start_m[:, s_idx] = start_m
            season_end_m[:, s_idx] = end_m
            season_n_slots[:, s_idx] = np.maximum(end_m - start_m + 1, 1)
            season_invalid[:, s_idx] = invalid
            # date_in_season with freq='month': label month within [start, end]
            season_in_raw[:, s_idx] = (
                (label_month_num >= start_m)
                & (label_month_num <= end_m)
                & ~invalid
            )

        ref_ids = (
            df["ref_id"].astype(str).to_numpy()
            if "ref_id" in df.columns
            else np.array([""] * n_rows)
        )
        lc_only = np.array([_is_lc_only_dataset(r) for r in ref_ids], dtype=bool)

        return {
            "season_start_m": season_start_m,
            "season_end_m": season_end_m,
            "season_n_slots": season_n_slots,
            "season_invalid": season_invalid,
            "season_in_raw": season_in_raw,
            "lc_only": lc_only,
        }

    def _fast_rng(self) -> np.random.Generator:
        """Per-worker numpy Generator, reseeded per DataLoader epoch.

        Uses the DataLoader worker seed (derived from the torch base seed, so
        `seed_everything` keeps runs reproducible) to decorrelate augmentation
        and masking draws across workers and epochs.
        """
        info = torch.utils.data.get_worker_info()
        key = (
            (info.id, int(info.seed))
            if info is not None
            else (-1, -1)
        )
        if self._batch_rng is None or self._batch_rng_key != key:
            if info is not None:
                seed = int(info.seed) % (2**32)
            else:
                seed = int(np.random.randint(0, 2**32 - 1))
            self._batch_rng = np.random.default_rng(seed)
            self._batch_rng_key = key
        return self._batch_rng

    def __getitems__(self, indices):
        if self._batch_cache is None:
            return [self[int(i)] for i in indices]
        return self._fast_fetch_batch(np.asarray(indices, dtype=np.int64))

    def _fast_fetch_batch(self, idx: np.ndarray):
        """Assemble a fully collated (Predictors, attrs) batch with array ops."""
        c = self._batch_cache
        assert c is not None
        n = len(self.dataframe)
        T = self.num_timesteps
        rng = self._fast_rng()

        override_ct = idx >= 2 * n
        override_lc = (idx >= n) & ~override_ct
        rows = np.where(override_ct, idx - 2 * n, np.where(override_lc, idx - n, idx))
        B = len(rows)

        avail = c["avail"][rows]
        vp = c["vp"][rows]

        # ---- timestep window (mirrors _get_center_point) -------------------
        half = T // 2
        det_center = np.clip(vp, half, avail - half)
        if self.augment and not self.is_ssl:
            buf = max(1, MIN_EDGE_BUFFER)
            min_c = np.maximum(half, vp + buf - half)
            max_c = np.minimum(avail - half, vp - buf + half)
            needs_rand = avail != T
            if (needs_rand & (min_c > max_c)).any():
                raise ValueError(
                    "low >= high while drawing augmented center point; "
                    "run _filter_temporally_invalid_rows on the dataframe."
                )
            safe_min = np.where(needs_rand, min_c, 0)
            safe_max = np.where(needs_rand, max_c, 0)
            rand_center = rng.integers(safe_min, safe_max + 1)
            center = np.where(needs_rand, rand_center, det_center)
        else:
            center = det_center
        last = np.minimum(avail, center + half)
        first = np.maximum(0, last - T)

        if ((vp < first) | (vp >= first + T)).any():
            bad = int(np.flatnonzero((vp < first) | (vp >= first + T))[0])
            raise AssertionError(
                f"Valid position {vp[bad]} not in timestep window starting at "
                f"{first[bad]}"
            )

        # ---- window rescue for empty S1/S2 windows -------------------------
        if self.remove_samples_without_s1_s2:
            pres = c["presence"][rows]
            csum = np.zeros((B, c["max_ts"] + 1), dtype=np.int64)
            csum[:, 1:] = np.cumsum(pres, axis=1)
            arange_b = np.arange(B)
            window_has = (csum[arange_b, first + T] - csum[arange_b, first]) > 0
            for i in np.flatnonzero(~window_has):
                first_min = max(0, vp[i] - T + 1)
                first_max = min(vp[i], avail[i] - T)
                cands = [
                    f
                    for f in range(first_min, first_max + 1)
                    if pres[i, f : f + T].any()
                ]
                if not cands:
                    continue
                if self.augment:
                    first[i] = cands[int(rng.integers(len(cands)))]
                else:
                    first[i] = min(cands, key=lambda f, cur=first[i]: abs(f - cur))

        tidx = first[:, None] + np.arange(T)[None, :]
        rows_col = rows[:, None]

        s2_win = c["s2"][rows_col, tidx]  # (B, T, 10)
        s1_win = c["s1"][rows_col, tidx]  # (B, T, 2)
        meteo_win = c["meteo"][rows_col, tidx]  # (B, T, 2)

        s2_full = np.full(
            (B, 1, 1, T, len(S2_BANDS)), NODATAVALUE, dtype=np.float32
        )
        s2_full[:, 0, 0][:, :, c["s2_dst_idx"]] = s2_win
        s1_full = s1_win.reshape(B, 1, 1, T, len(S1_BANDS)).copy()
        meteo_full = meteo_win.reshape(B, 1, 1, T, len(METEO_BANDS)).copy()
        dem_full = c["dem"][rows].reshape(B, 1, 1, len(DEM_BANDS)).copy()
        latlon_full = c["latlon"][rows].reshape(B, 1, 1, 2).copy()

        # ---- sensor masking (mirrors _apply_masking) ------------------------
        if self.masking_config and self.masking_config.enable:
            cfg = self.masking_config
            s1_missing = np.all(s1_win == NODATAVALUE, axis=-1)
            s2_missing = np.all(s2_win == NODATAVALUE, axis=-1)

            s1_full_drop = rng.random(B) < cfg.s1_full_dropout_prob
            if cfg.s1_timestep_dropout_prob > 0:
                s1_drop = rng.random((B, T)) < cfg.s1_timestep_dropout_prob
            else:
                s1_drop = np.zeros((B, T), dtype=bool)
            s1_drop[s1_full_drop] = True

            s2_drop = np.zeros((B, T), dtype=bool)
            if cfg.s2_cloud_block_prob > 0:
                has_block = rng.random(B) < cfg.s2_cloud_block_prob
                block_len = rng.integers(
                    cfg.s2_cloud_block_min, cfg.s2_cloud_block_max + 1, size=B
                )
                full_block = block_len >= T
                capped_len = np.minimum(block_len, T)
                start = rng.integers(0, T - capped_len + 1)
                offs = np.arange(T)[None, :]
                block_mask = (offs >= start[:, None]) & (
                    offs < (start + capped_len)[:, None]
                )
                block_mask[full_block] = True
                s2_drop |= block_mask & has_block[:, None]
            if cfg.s2_cloud_timestep_prob > 0:
                s2_drop |= rng.random((B, T)) < cfg.s2_cloud_timestep_prob

            # Joint S1/S2 guard (rare; per-row like _rescue_joint_s1_s2_wipe)
            violated = ((s1_drop | s1_missing).all(axis=1)) & (
                (s2_drop | s2_missing).all(axis=1)
            )
            for i in np.flatnonzero(violated):
                restorable = np.flatnonzero(s2_drop[i] & ~s2_missing[i])
                if restorable.size and cfg.s2_cloud_timestep_prob < 1.0:
                    s2_drop[i, int(rng.choice(restorable))] = False
                    continue
                restorable = np.flatnonzero(s1_drop[i] & ~s1_missing[i])
                if restorable.size and cfg.s1_full_dropout_prob < 1.0:
                    s1_drop[i, int(rng.choice(restorable))] = False
                    continue
                logger.warning(
                    "Sample has S1 and S2 fully masked-or-missing and no "
                    "restorable timesteps; the joint S1/S2 guard cannot be "
                    "enforced for this sample."
                )

            s1_view = s1_full[:, 0, 0]
            s2_view = s2_full[:, 0, 0]
            s1_view[s1_drop] = NODATAVALUE
            s2_view[s2_drop] = NODATAVALUE

            if cfg.meteo_timestep_dropout_prob > 0:
                meteo_mask = rng.random((B, T)) < cfg.meteo_timestep_dropout_prob
                meteo_full[:, 0, 0][meteo_mask] = NODATAVALUE

            if cfg.dem_dropout_prob > 0:
                dem_drop = rng.random(B) < cfg.dem_dropout_prob
                dem_full[dem_drop] = NODATAVALUE

        # ---- timestamps ------------------------------------------------------
        month_num = c["start_month_num"][rows][:, None] + tidx  # (B, T)
        timestamps = np.empty((B, T, 3), dtype=np.int64)
        timestamps[:, :, 0] = 1
        timestamps[:, :, 1] = (month_num % 12) + 1
        timestamps[:, :, 2] = month_num // 12 + 1970

        relative_valid = (vp - first).astype(np.int64)

        # ---- label tensor ----------------------------------------------------
        label_tensor = None
        if self.emit_label_tensor:
            codes = c["finetune_codes"][rows]
            uniques = c["finetune_uniques"]
            if self.task_type == "binary":
                lut = np.array(
                    [
                        0 if (u is not None and str(u).startswith("not_")) else 1
                        for u in uniques
                    ],
                    dtype=np.int32,
                )
            else:
                lut = np.array(
                    [
                        self.classes_list.index(u) if u is not None else -1
                        for u in uniques
                    ],
                    dtype=np.int32,
                )
            values = lut[codes]
            if (codes == 0).any():
                raise ValueError(
                    "Missing finetune_class value encountered while building "
                    "labels for a batch."
                )
            tsteps = T if self.time_explicit else 1
            label = np.full((B, 1, 1, tsteps, 1), NODATAVALUE, dtype=np.int32)
            if self.time_explicit:
                p = relative_valid.copy()
                if self.label_jitter > 0:
                    shift = rng.integers(
                        -self.label_jitter, self.label_jitter + 1, size=B
                    )
                    p = np.clip(p + shift, 0, T - 1)
                if self.label_window > 0:
                    w_start = np.maximum(0, p - self.label_window)
                    w_end = np.minimum(T - 1, p + self.label_window)
                    grid = np.arange(T)[None, :]
                    in_window = (grid >= w_start[:, None]) & (grid <= w_end[:, None])
                    label[:, 0, 0, :, 0] = np.where(
                        in_window, values[:, None], NODATAVALUE
                    )
                else:
                    label[np.arange(B), 0, 0, p, 0] = values
            else:
                label[:, 0, 0, 0, 0] = values
            label_tensor = torch.from_numpy(label)

        # ---- attrs -----------------------------------------------------------
        attrs: Dict[str, Any] = {}
        for key, kind, data, extra in c["attr_specs"]:
            if kind == "num":
                vals = data[rows]
                miss = extra[rows] if extra is not None else None
                if miss is not None and miss.any():
                    if miss.all():
                        attrs[key] = None
                    elif key in ATTR_KEYS_ALLOW_PARTIAL_NONE:
                        attrs[key] = [
                            None if m else v
                            for v, m in zip(vals.tolist(), miss.tolist())
                        ]
                    else:
                        raise ValueError(
                            f"_collate_attrs received None values for key "
                            f"'{key}' at indices "
                            f"{np.flatnonzero(miss).tolist()}"
                        )
                else:
                    attrs[key] = torch.from_numpy(vals)
            else:
                codes = data[rows]
                values_list = [extra[k] for k in codes.tolist()]
                if all(v is None for v in values_list):
                    attrs[key] = None
                elif any(v is None for v in values_list):
                    if key in ATTR_KEYS_ALLOW_PARTIAL_NONE:
                        attrs[key] = values_list
                    else:
                        missing_indices = [
                            i for i, v in enumerate(values_list) if v is None
                        ]
                        raise ValueError(
                            f"_collate_attrs received None values for key "
                            f"'{key}' at indices {missing_indices}"
                        )
                else:
                    attrs[key] = values_list

        attrs["region"] = [
            c["region_uniques"][k] for k in c["region_codes"][rows].tolist()
        ]

        lt_codes = c["label_task_codes"]
        if lt_codes is not None:
            label_task = [
                c["label_task_uniques"][k] for k in lt_codes[rows].tolist()
            ]
        else:
            label_task = [None] * B
        has_override = override_ct.any() or override_lc.any()
        if has_override:
            for i in np.flatnonzero(override_lc):
                label_task[i] = "landcover"
            for i in np.flatnonzero(override_ct):
                label_task[i] = "croptype"
        if any(v is not None for v in label_task):
            attrs["label_task"] = label_task

        attrs["valid_position"] = torch.from_numpy(relative_valid)

        # ---- season metadata -------------------------------------------------
        if self._season_engine == "off":
            attrs["season_masks"] = None
            attrs["in_seasons"] = None
        else:
            start_m = c["season_start_m"][rows]  # (B, S)
            end_m = c["season_end_m"][rows]
            n_slots = c["season_n_slots"][rows]
            invalid = c["season_invalid"][rows]
            in_raw = c["season_in_raw"][rows]
            lc_only = c["lc_only"][rows]

            month_grid = month_num[:, None, :]  # (B, 1, T)
            masks = (month_grid >= start_m[:, :, None]) & (
                month_grid <= end_m[:, :, None]
            )
            n_required = np.maximum(
                1, np.rint(n_slots * self.min_season_coverage)
            ).astype(np.int64)
            meets = (masks.sum(axis=-1) >= n_required) & ~invalid
            masks &= meets[:, :, None]
            in_seasons = in_raw & meets
            masks[lc_only] = True
            in_seasons[lc_only] = True
            attrs["season_masks"] = masks
            attrs["in_seasons"] = in_seasons

        predictors = Predictors(
            s1=torch.from_numpy(s1_full),
            s2=torch.from_numpy(s2_full),
            meteo=torch.from_numpy(meteo_full),
            dem=torch.from_numpy(dem_full),
            latlon=torch.from_numpy(latlon_full),
            timestamps=torch.from_numpy(timestamps),
            label=label_tensor,
        )
        return predictors, attrs


    def get_task_batch_sampler(
        self,
        *,
        batch_size: int,
        landcover_column: str = "landcover_label",
        croptype_column: str = "croptype_label",
        class_weight_method: str = "balanced",
        clip_range: Optional[Tuple[float, float]] = (0.1, 10.0),
        spatial_bin_size_degrees: Optional[float] = None,
        spatial_weight_method: str = "log",
        class_balancing_scope: Literal["global", "per_bin"] = "global",
        min_samples_per_bin: int = 50,
        smoothing: Literal["none", "bilinear"] = "none",
        min_samples_per_class_per_bin: Optional[int] = None,
        num_batches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        class_weight_multipliers: Optional[Mapping[str, Any]] = None,
        region_column: str = "region",
        bin_type: Literal["degrees", "h3"] = "degrees",
        h3_resolution: int = 2,
        tasks: Sequence[str] = ("landcover", "croptype"),
        task_ratios: Optional[Mapping[str, float]] = None,
    ) -> "SeasonalTaskBatchSampler":
        """Build a task-configurable seasonal batch sampler.

        ``tasks`` selects the active task pools to draw into each batch while
        preserving the same class, spatial, per-bin, multiplier, and generator
        balancing logic.
        """
        return SeasonalTaskBatchSampler(
            dataframe=self.dataframe,
            batch_size=batch_size,
            landcover_column=landcover_column,
            croptype_column=croptype_column,
            class_weight_method=class_weight_method,
            clip_range=clip_range,
            spatial_bin_size_degrees=spatial_bin_size_degrees,
            spatial_weight_method=spatial_weight_method,
            class_balancing_scope=class_balancing_scope,
            min_samples_per_bin=min_samples_per_bin,
            smoothing=smoothing,
            min_samples_per_class_per_bin=min_samples_per_class_per_bin,
            num_batches=num_batches,
            generator=generator,
            class_weight_multipliers=class_weight_multipliers,
            region_column=region_column,
            bin_type=bin_type,
            h3_resolution=h3_resolution,
            tasks=tasks,
            task_ratios=task_ratios,
        )


def _normalize_class_weight_multipliers(
    mults: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Accept per-class OR per-region multipliers; return canonical form.

    - Per-class:  ``{"soy_soybeans": 0.6, "fibre_crops": 0.6}``
      → wrapped as ``{"*": {"soy_soybeans": 0.6, "fibre_crops": 0.6}}``
    - Per-region: ``{"Eastern Asia": {"temporary_crops": 1.5}, ...}``
      → returned as-is (with all values coerced to float).
    - Hybrid via "*": ``{"*": {...defaults...}, "Eastern Asia": {...overrides...}}``
      → works naturally; lookup precedence is region-specific then wildcard "*".

    Mixed (some top-level values dicts, others scalars) is rejected.
    """
    if not mults:
        return {}
    values_are_dicts = [isinstance(v, Mapping) for v in mults.values()]
    if all(values_are_dicts):
        return {
            str(reg): {str(c): float(w) for c, w in cmults.items()}
            for reg, cmults in mults.items()
        }
    if not any(values_are_dicts):
        return {"*": {str(c): float(w) for c, w in mults.items()}}
    raise ValueError(
        "class_weight_multipliers must be either per-class (class→float) "
        "or per-region (region→{class→float}), not mixed."
    )


def apply_class_weight_multipliers(
    sampler_weights: np.ndarray,
    labels: np.ndarray,
    regions: Optional[np.ndarray],
    mults_canonical: Dict[str, Dict[str, float]],
    pool_name: str = "",
) -> np.ndarray:
    """Multiply ``sampler_weights`` by per-region/per-class multipliers.

    Parameters
    ----------
    sampler_weights:
        Per-sample float64 weight array of length N.
    labels:
        String label array of length N (e.g. ``landcover_label`` or
        ``croptype_label`` values).
    regions:
        Optional string region array of length N.  ``None`` or a missing
        value causes the wildcard ``"*"`` entry to be used.
    mults_canonical:
        Canonical multiplier dict as returned by
        ``_normalize_class_weight_multipliers`` — keys are region names (or
        ``"*"``), values are ``{class_name: float}`` dicts.
    pool_name:
        Optional label for log messages.

    Returns
    -------
    np.ndarray
        ``sampler_weights * mult_arr`` (a new array; input is not mutated).
    """
    if not mults_canonical:
        return sampler_weights

    wildcard = mults_canonical.get("*", {})
    label_set = set(labels.tolist())

    def _lookup(reg: Optional[str], lbl: str) -> float:
        if reg is not None:
            reg_dict = mults_canonical.get(reg)
            if reg_dict is not None and lbl in reg_dict:
                return reg_dict[lbl]
        return wildcard.get(lbl, 1.0)

    regs_iter = regions if regions is not None else [None] * len(labels)
    mult_arr = np.array(
        [_lookup(r, lbl) for r, lbl in zip(regs_iter, labels)],
        dtype=np.float64,
    )
    if np.allclose(mult_arr, 1.0):
        return sampler_weights

    applied = {
        reg: {c: w for c, w in cmults.items() if c in label_set}
        for reg, cmults in mults_canonical.items()
    }
    applied = {k: v for k, v in applied.items() if v}
    prefix = f"[{pool_name}] " if pool_name else ""
    logger.info(f"{prefix}Applied class weight multipliers: {applied}")

    # Warn about classes that don't appear in the label set.
    for reg, cmults in mults_canonical.items():
        for c in cmults:
            if c not in label_set:
                logger.debug(
                    f"{prefix}class_weight_multipliers[{reg!r}][{c!r}] not found "
                    "in label set; ignoring."
                )

    return sampler_weights * mult_arr


class SeasonalTaskBatchSampler(Sampler):
    """Batch sampler for seasonal landcover/croptype finetuning.

    By default each batch is composed of exactly ``batch_size // 2``
    **LC-assigned** samples and ``batch_size - batch_size // 2`` **CT-assigned**
    samples for default two-task training. Callers may
    instead pass ``tasks=("landcover",)`` or ``tasks=("croptype",)`` for
    single-task batches, or ``task_ratios`` for explicit task proportions.

    Task assignment is encoded into the virtual index returned to the DataLoader so that
    :meth:`WorldCerealLabelledDataset.__getitem__` can override ``label_task``
    without mutating the dataframe or touching shared worker state:

    * ``[N, 2N)``  → LC-assigned  (``real_idx = idx - N``,   ``label_task = "landcover"``)
    * ``[2N, 3N)`` → CT-assigned  (``real_idx = idx - 2N``,  ``label_task = "croptype"``)
    * ``[0, N)``   → natural idx  (val/test; ``label_task`` read from dataframe)

    **LC pool**: all *N* training samples, weighted by the ``landcover_label``
    class distribution over the full training set.

    **CT pool**: only samples that carry a valid (non-null / non-ignore)
    ``croptype_label``, weighted by the ``croptype_label`` class distribution
    over that subset.  The "no-crop" class is included naturally.

    Sample weighting is composed from two **orthogonal** factors:

    * **Class-weight source** (``class_balancing_scope``):

      - ``"global"`` (default) — class weights computed once over the full
        training set via `_get_normalized_weights`.
      - ``"per_bin"`` — class weights computed *within* each lat/lon bin via
        `_get_per_bin_class_weights`. Bins with fewer than
        ``min_samples_per_bin`` samples fall back to global class weights.
        Requires ``spatial_bin_size_degrees`` to be set and lat/lon columns
        to be present.

        With ``smoothing="bilinear"`` (per_bin scope only), each sample's
        class weight is the bilinear blend of the four neighbouring bin
        centres' weights (see `_get_smoothed_per_bin_class_weights`).

    * **Spatial-density factor** (``spatial_weight_method``):

      Applied multiplicatively to whichever class weights are produced above,
      whenever ``spatial_bin_size_degrees`` is set and
      ``spatial_weight_method != "none"``.  Normalised to mean = 1 and clipped
      to *clip_range* independently before multiplication, so the two factors
      cannot dominate each other due to differences in absolute range.
      Bins with fewer than ``min_samples_per_bin`` samples are treated as
      "average density" (weight = 1.0) so that singleton/sparse bins cannot
      pin the density factor to its upper clip and blow up the composed
      sampling distribution (see `_get_spatial_density_weights`).
    """

    _LC_OFFSET: int = 1  # virtual offset factor for LC pool
    _CT_OFFSET: int = 2  # virtual offset factor for CT pool

    def __init__(
        self,
        dataframe: pd.DataFrame,
        batch_size: int,
        *,
        landcover_column: str = "landcover_label",
        croptype_column: str = "croptype_label",
        class_weight_method: str = "balanced",
        clip_range: Optional[Tuple[float, float]] = (0.1, 10.0),
        spatial_bin_size_degrees: Optional[float] = None,
        spatial_weight_method: str = "log",
        class_balancing_scope: Literal["global", "per_bin"] = "global",
        min_samples_per_bin: int = 50,
        smoothing: Literal["none", "bilinear"] = "none",
        min_samples_per_class_per_bin: Optional[int] = None,
        num_batches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        class_weight_multipliers: Optional[Mapping[str, Any]] = None,
        region_column: str = "region",
        bin_type: Literal["degrees", "h3"] = "degrees",
        h3_resolution: int = 2,
        tasks: Optional[Sequence[str]] = None,
        task_ratios: Optional[Mapping[str, float]] = None,
    ) -> None:
        import math

        valid_scopes = ("global", "per_bin")
        if class_balancing_scope not in valid_scopes:
            raise ValueError(
                f"class_balancing_scope must be one of {valid_scopes}, got "
                f"'{class_balancing_scope}'"
            )
        valid_smoothing = ("none", "bilinear")
        if smoothing not in valid_smoothing:
            raise ValueError(
                f"smoothing must be one of {valid_smoothing}, got '{smoothing}'"
            )
        if smoothing != "none" and class_balancing_scope != "per_bin":
            raise ValueError(
                f"smoothing='{smoothing}' only meaningful with "
                "class_balancing_scope='per_bin'."
            )

        N = len(dataframe)
        self._n = N
        self._batch_size = batch_size
        self._generator = generator
        self._num_batches = (
            num_batches if num_batches is not None else math.ceil(N / batch_size)
        )
        self._tasks = self._normalize_tasks(tasks)
        self._task_ratios = self._normalize_task_ratios(self._tasks, task_ratios)
        self._task_batch_counts = self._compute_task_batch_counts(
            batch_size, self._tasks, self._task_ratios
        )

        lc_labels = dataframe[landcover_column].astype(str).to_numpy()

        # ---- CT pool: samples with a valid (non-null, non-ignore) croptype label ----
        ct_valid = dataframe[croptype_column].notna() & (
            dataframe[croptype_column].astype(str) != "ignore"
        )
        ct_real_indices = np.where(ct_valid.to_numpy())[0]
        has_ct_pool = len(ct_real_indices) > 0
        if not has_ct_pool and "croptype" in self._tasks:
            raise ValueError(
                f"No samples with a valid '{croptype_column}' found; "
                "SeasonalTaskBatchSampler requires at least one CT-eligible sample."
            )
        ct_labels = dataframe.loc[ct_valid, croptype_column].astype(str).to_numpy()

        # ---- Compute spatial bins (degree grid or H3 cells) ----
        # bin_type selects the binning scheme:
        #   - "degrees": legacy lat/lon grid, cell size = spatial_bin_size_degrees
        #   - "h3": Uber H3 (approximately equal-area) hexagons at h3_resolution
        valid_bin_types = ("degrees", "h3")
        if bin_type not in valid_bin_types:
            raise ValueError(
                f"bin_type must be one of {valid_bin_types}, got '{bin_type}'"
            )
        spatial_bins: Optional[np.ndarray] = None
        has_latlon = "lat" in dataframe.columns and "lon" in dataframe.columns
        if has_latlon:
            if bin_type == "h3":
                spatial_bins = _spatial_bins_from_h3(
                    dataframe["lat"], dataframe["lon"], h3_resolution
                )
            elif spatial_bin_size_degrees is not None:
                spatial_bins = _spatial_bins_from_latlon(
                    dataframe["lat"], dataframe["lon"], spatial_bin_size_degrees
                )

        # ---- Class weights: source controlled by class_balancing_scope ----
        if class_balancing_scope == "per_bin":
            if spatial_bins is None:
                raise ValueError(
                    "class_balancing_scope='per_bin' requires lat/lon columns "
                    "and either bin_type='h3' or spatial_bin_size_degrees to be "
                    "set."
                )
            if smoothing == "bilinear":
                if bin_type != "degrees":
                    raise ValueError(
                        "smoothing='bilinear' is only supported with "
                        "bin_type='degrees'; use smoothing='none' for H3 bins."
                    )
                # bilinear path is degree-grid specific; spatial_bin_size_degrees
                # is guaranteed set when bin_type='degrees' and bins were built.
                assert spatial_bin_size_degrees is not None
                lat_arr = dataframe["lat"].to_numpy(dtype=np.float64)
                lon_arr = dataframe["lon"].to_numpy(dtype=np.float64)
                lc_class_arr = _get_smoothed_per_bin_class_weights(
                    lc_labels,
                    lat_arr,
                    lon_arr,
                    spatial_bin_size_degrees,
                    method=class_weight_method,
                    clip_range=clip_range,
                    min_samples_per_bin=min_samples_per_bin,
                    min_samples_per_class_per_bin=min_samples_per_class_per_bin,
                    pool_name="LC",
                )
                ct_class_arr = (
                    _get_smoothed_per_bin_class_weights(
                        ct_labels,
                        lat_arr[ct_real_indices],
                        lon_arr[ct_real_indices],
                        spatial_bin_size_degrees,
                        method=class_weight_method,
                        clip_range=clip_range,
                        min_samples_per_bin=min_samples_per_bin,
                        min_samples_per_class_per_bin=min_samples_per_class_per_bin,
                        pool_name="CT",
                    )
                    if has_ct_pool
                    else np.array([], dtype=np.float64)
                )
            else:
                lc_class_arr = _get_per_bin_class_weights(
                    lc_labels,
                    spatial_bins,
                    method=class_weight_method,
                    clip_range=clip_range,
                    min_samples_per_bin=min_samples_per_bin,
                    min_samples_per_class_per_bin=min_samples_per_class_per_bin,
                    pool_name="LC",
                )
                ct_class_arr = (
                    _get_per_bin_class_weights(
                        ct_labels,
                        spatial_bins[ct_real_indices],
                        method=class_weight_method,
                        clip_range=clip_range,
                        min_samples_per_bin=min_samples_per_bin,
                        min_samples_per_class_per_bin=min_samples_per_class_per_bin,
                        pool_name="CT",
                    )
                    if has_ct_pool
                    else np.array([], dtype=np.float64)
                )
        else:
            lc_class_arr = _get_normalized_weights(
                lc_labels,
                method=class_weight_method,
                clip_range=clip_range,
                pool_name="LC",
            )
            ct_class_arr = (
                _get_normalized_weights(
                    ct_labels,
                    method=class_weight_method,
                    clip_range=clip_range,
                    pool_name="CT",
                )
                if has_ct_pool
                else np.array([], dtype=np.float64)
            )

        # ---- Optional class weight multipliers (per-class or per-region) ----
        # Accepts either {class: float} (legacy) or {region: {class: float}}.
        # Wildcard region "*" applies to all regions; region-specific entries take
        # precedence over the wildcard. Routed to LC or CT pool by label membership.
        if class_weight_multipliers:
            mults_canonical = _normalize_class_weight_multipliers(
                class_weight_multipliers
            )
            has_region = region_column in dataframe.columns
            lc_regions = (
                dataframe[region_column].astype(str).to_numpy() if has_region else None
            )
            ct_regions = lc_regions[ct_real_indices] if lc_regions is not None else None

            lc_class_arr = apply_class_weight_multipliers(
                lc_class_arr, lc_labels, lc_regions, mults_canonical, pool_name="LC"
            )
            if has_ct_pool:
                ct_class_arr = apply_class_weight_multipliers(
                    ct_class_arr, ct_labels, ct_regions, mults_canonical, pool_name="CT"
                )

            # Re-normalise to mean=1 and re-clip AFTER multipliers so a per-region
            # or per-class multiplier can never push a class beyond the configured
            # clip ceiling/floor. Without this a multiplier > 1 stacked on an
            # already-high per-bin weight would exceed clip_range[1] and
            # re-introduce exactly the over-sampling the clip is meant to bound
            # (the root cause of e.g. oats over-commission into wheat).
            if clip_range is not None:
                lc_mean = lc_class_arr.mean()
                if lc_mean > 0:
                    lc_class_arr = lc_class_arr / lc_mean
                lc_class_arr = np.clip(lc_class_arr, clip_range[0], clip_range[1])
                if has_ct_pool:
                    ct_mean = ct_class_arr.mean()
                    if ct_mean > 0:
                        ct_class_arr = ct_class_arr / ct_mean
                    ct_class_arr = np.clip(ct_class_arr, clip_range[0], clip_range[1])
                logger.info(
                    "SeasonalTaskBatchSampler: re-normalised and re-clipped class "
                    f"weights to {clip_range} after applying multipliers."
                )

            # Warn once about classes that don't appear in either pool.
            lc_label_set = set(lc_labels.tolist())
            ct_label_set = set(ct_labels.tolist())
            all_labels = lc_label_set | ct_label_set
            for reg, cmults in mults_canonical.items():
                for c in cmults:
                    if c not in all_labels:
                        logger.warning(
                            f"class_weight_multipliers[{reg!r}][{c!r}] not found "
                            "in LC or CT labels; ignoring."
                        )

        # ---- Spatial-density factor: independent, applied uniformly ----
        # Both class and spatial arrays are independently normalised to mean=1
        # and re-clipped to clip_range (see _get_normalized_weights) before
        # being multiplied, so neither can dominate due to differences in
        # absolute weight range. spatial_weight_method='none' or no bins
        # means the density factor is skipped. Sparse bins (< min_samples_per_bin)
        # get density weight = 1.0 to prevent singleton bins from blowing up
        # the multiplicative composition.
        density_applied = spatial_bins is not None and spatial_weight_method != "none"
        if density_applied:
            sp_arr = _get_spatial_density_weights(
                spatial_bins,
                method=spatial_weight_method,
                clip_range=clip_range,
                min_samples_per_bin=min_samples_per_bin,
            )
            lc_class_arr = lc_class_arr * sp_arr  # full N-sample array
            if has_ct_pool:
                ct_class_arr = ct_class_arr * sp_arr[ct_real_indices]  # CT subset

        density_msg = (
            f"× spatial-density factor (method={spatial_weight_method})"
            if density_applied
            else "(no spatial-density factor)"
        )
        if class_balancing_scope == "per_bin":
            if smoothing == "bilinear":
                smoothing_msg = " smoothing=bilinear"
            else:
                smoothing_msg = ""
            logger.info(
                "SeasonalTaskBatchSampler: per-bin class balancing "
                f"(method={class_weight_method}, "
                f"min_samples_per_bin={min_samples_per_bin}{smoothing_msg}) "
                f"{density_msg}."
            )
        else:
            logger.info(
                "SeasonalTaskBatchSampler: global class balancing "
                f"(method={class_weight_method}) "
                f"{density_msg}."
            )

        # Convert to float64 probability tensors for torch.multinomial
        self._lc_probs = torch.as_tensor(
            lc_class_arr / lc_class_arr.sum(), dtype=torch.float64
        )
        self._ct_probs = torch.as_tensor(
            ct_class_arr / ct_class_arr.sum() if has_ct_pool else ct_class_arr,
            dtype=torch.float64,
        )

        # Virtual-index tensors: LC → [N, 2N), CT → [2N, 3N)
        self._lc_virtual = torch.arange(N, dtype=torch.long) + N
        self._ct_virtual = torch.as_tensor(ct_real_indices, dtype=torch.long) + 2 * N

        lc_class_counts: Dict[str, int] = Counter(lc_labels.tolist())
        ct_class_counts: Dict[str, int] = Counter(ct_labels.tolist())
        lc_count = self._task_batch_counts.get("landcover", 0)
        ct_count = self._task_batch_counts.get("croptype", 0)
        logger.info(
            f"SeasonalTaskBatchSampler: N={N}, LC pool={N}, CT pool={len(ct_real_indices)}, "
            f"batch_size={batch_size} (LC={lc_count}, CT={ct_count}), "
            f"num_batches={self._num_batches}, tasks={self._tasks}, "
            f"task_ratios={self._task_ratios}"
        )
        logger.info(
            f"SeasonalTaskBatchSampler LC class distribution: {lc_class_counts}"
        )
        logger.info(
            f"SeasonalTaskBatchSampler CT class distribution: {ct_class_counts}"
        )

    @staticmethod
    def _normalize_tasks(tasks: Optional[Sequence[str]]) -> Tuple[str, ...]:
        if tasks is None:
            return ("landcover", "croptype")
        aliases = {"lc": "landcover", "ct": "croptype"}
        normalized = tuple(aliases.get(str(task), str(task)) for task in tasks)
        valid = {"landcover", "croptype"}
        invalid = [task for task in normalized if task not in valid]
        if invalid:
            raise ValueError(f"tasks must contain only {sorted(valid)}, got {invalid}")
        if not normalized:
            raise ValueError("tasks must contain at least one active task.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"tasks must not contain duplicates, got {normalized}")
        return normalized

    @staticmethod
    def _normalize_task_ratios(
        tasks: Tuple[str, ...], task_ratios: Optional[Mapping[str, float]]
    ) -> Dict[str, float]:
        if task_ratios is None:
            ratio = 1.0 / len(tasks)
            return {task: ratio for task in tasks}
        aliases = {"lc": "landcover", "ct": "croptype"}
        normalized = {
            aliases.get(str(task), str(task)): float(ratio)
            for task, ratio in task_ratios.items()
        }
        task_set = set(tasks)
        if set(normalized) != task_set:
            raise ValueError(
                "task_ratios keys must exactly match active tasks; "
                f"got keys={sorted(normalized)} tasks={sorted(task_set)}"
            )
        if any(ratio < 0 for ratio in normalized.values()):
            raise ValueError(f"task_ratios must be non-negative, got {normalized}")
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError(
                f"task_ratios must sum to a positive value, got {normalized}"
            )
        return {task: ratio / total for task, ratio in normalized.items()}

    @staticmethod
    def _compute_task_batch_counts(
        batch_size: int, tasks: Tuple[str, ...], task_ratios: Mapping[str, float]
    ) -> Dict[str, int]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if tasks == ("landcover", "croptype") and task_ratios == {
            "landcover": 0.5,
            "croptype": 0.5,
        }:
            return {
                "landcover": batch_size // 2,
                "croptype": batch_size - batch_size // 2,
            }

        raw_counts = {task: batch_size * task_ratios[task] for task in tasks}
        counts = {task: int(np.floor(count)) for task, count in raw_counts.items()}
        remaining = batch_size - sum(counts.values())
        order = sorted(
            tasks, key=lambda task: raw_counts[task] - counts[task], reverse=True
        )
        for task in order[:remaining]:
            counts[task] += 1
        return counts

    def __iter__(self):
        lc_count = self._task_batch_counts.get("landcover", 0)
        ct_count = self._task_batch_counts.get("croptype", 0)
        all_lc_drawn = (
            torch.multinomial(
                self._lc_probs,
                lc_count * self._num_batches,
                replacement=True,
                generator=self._generator,
            )
            if lc_count
            else None
        )
        all_ct_drawn = (
            torch.multinomial(
                self._ct_probs,
                ct_count * self._num_batches,
                replacement=True,
                generator=self._generator,
            )
            if ct_count
            else None
        )
        for i in range(self._num_batches):
            parts = []
            if lc_count and all_lc_drawn is not None:
                lc_drawn = all_lc_drawn[i * lc_count : (i + 1) * lc_count]
                parts.append(self._lc_virtual[lc_drawn])
            if ct_count and all_ct_drawn is not None:
                ct_drawn = all_ct_drawn[i * ct_count : (i + 1) * ct_count]
                parts.append(self._ct_virtual[ct_drawn])
            batch = torch.cat(parts)
            perm = torch.randperm(self._batch_size, generator=self._generator)
            yield batch[perm].tolist()

    def __len__(self) -> int:
        return self._num_batches


class WorldCerealTrainingDataset(WorldCerealDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_timesteps: int = 12,
        timestep_freq: Literal["month", "dekad"] = "month",
        task_type: Literal["ssl", "binary", "multiclass"] = "ssl",
        num_outputs: Optional[int] = None,
        augment: bool = False,
        masking_config: Optional[SensorMaskingConfig] = None,
        repeats: int = 1,
        season_ids: Optional[Sequence[str]] = None,
        season_windows: Optional[Mapping[str, Tuple[Any, Any]]] = None,
        season_calendar_mode: SeasonCalendarMode = "auto",
        min_season_coverage: float = 1.0,
        remove_samples_without_s1_s2: bool = False,
        region_column: str = "region",
    ):
        """WorldCereal training dataset. This dataset is typically used for
        computing embeddings for downstream training."""
        super().__init__(
            dataframe=dataframe,
            num_timesteps=num_timesteps,
            timestep_freq=timestep_freq,
            task_type=task_type,
            num_outputs=num_outputs,
            augment=augment,
            masking_config=masking_config,
            min_season_coverage=min_season_coverage,
            remove_samples_without_s1_s2=remove_samples_without_s1_s2,
        )

        self._season_windows: Dict[str, SeasonWindow] = _normalize_season_windows(
            season_windows
        )
        self._season_engine: SeasonEngine = _resolve_season_engine(
            season_calendar_mode, bool(self._season_windows)
        )
        if season_ids:
            resolved_seasons = tuple(season_ids)
        elif self._season_windows:
            resolved_seasons = tuple(self._season_windows.keys())
        else:
            resolved_seasons = GLOBAL_SEASON_IDS
        self._season_ids: Tuple[str, ...] = resolved_seasons

        if self._season_engine == "manual" and self._season_windows:
            filtered_df, dropped = _filter_frame_by_manual_windows(
                self.dataframe,
                self._season_windows,
                context=self.__class__.__name__,
                timestep_freq=self.timestep_freq,
            )
            if dropped:
                logger.info(
                    f"{self.__class__.__name__}: proceeding with {len(filtered_df)} samples after enforcing manual season window(s)."
                )
            self.dataframe = filtered_df

        repeats = _check_augmentation_settings(augment, masking_config, repeats)

        self._region_column = region_column

        base_indices = list(range(len(self.dataframe)))
        self.indices = base_indices * repeats
        self._repeats = repeats

    def __len__(self):
        # Return total repeated length, not the base dataframe length
        return len(self.indices)

    def __iter__(self):
        for idx in self.indices:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        # Map incoming idx to the original dataframe index
        real_idx = self.indices[idx]

        row = pd.Series.to_dict(self.dataframe.iloc[real_idx, :])

        # Draw the timestep window exactly ONCE. Calling
        # super().__getitem__() and then get_timestep_positions() again would
        # draw two *independent* random windows when augment=True, so
        # valid_position (and anything derived from it) would describe a
        # different window than the inputs actually extracted.
        timestep_positions, valid_position = self.get_timestep_positions(row)
        relative_valid = valid_position - timestep_positions[0]
        sample = Predictors(**self.get_inputs(row, timestep_positions))

        attrs = self._build_sample_attrs(
            row=row,
            valid_position=relative_valid,
            timestamps=sample.timestamps,
            season_ids=self._season_ids,
            season_windows=self._season_windows,
            season_engine=self._season_engine,
            derive_from_calendar=self._season_engine == "calendar",
            label_datetime=_resolve_label_datetime(row),
        )

        return sample, attrs


def _check_augmentation_settings(
    augment: bool, masking_config: Optional[SensorMaskingConfig], repeats: int
) -> int:
    """
    Check augmentation/masking settings. If no augmentation or masking is
    enabled but repeats > 1, set repeats to 1 and log a warning.
    If augmentation or masking is enabled but repeats = 1, log a warning
    suggesting to increase repeats for more variability.
    """
    some_augmentation = augment or (masking_config and masking_config.enable)
    if repeats == 1 and some_augmentation:
        logger.warning(
            "Dataset augmentation or masking is enabled but repeats=1. "
            "Consider setting repeats > 1 to increase training variability."
        )
    elif repeats > 1 and not some_augmentation:
        logger.warning(
            "Dataset is repeated but not augmented which is useless; "
            "consider setting `augment=True` or `masking_config` for training. "
            "Setting repeats=1 instead."
        )
        repeats = 1
    elif repeats > 1:
        logger.info(
            f"Dataset will be repeated {repeats} times for training with augmentation/masking."
        )

    return repeats


def get_dekad_timestamp_components(
    start_date: np.datetime64, end_date: np.datetime64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dekad (10-day period) timestamp components (day, month, year) between a start and end date.

    Parameters
    ----------
    start_date : np.datetime64
        The starting date from which to generate dekad timestamps.
    end_date : np.datetime64
        The ending date up to which dekad timestamps are generated (inclusive).

    Returns
    -------
    days : np.ndarray
        Array of day components for each dekad timestamp.
    months : np.ndarray
        Array of month components for each dekad timestamp.
    years : np.ndarray
        Array of year components for each dekad timestamp.
    """

    # Align start and end dates to the dekad window
    start_date = align_to_composite_window(start_date, "dekad")
    end_date = align_to_composite_window(end_date, "dekad")

    # Extract year, month, and day
    year = start_date.astype("object").year
    month = start_date.astype("object").month
    day = start_date.astype("object").day

    year_end = end_date.astype("object").year
    month_end = end_date.astype("object").month
    day_end = end_date.astype("object").day

    days, months, years = [day], [month], [year]
    while f"{year}-{month}-{day}" != f"{year_end}-{month_end}-{day_end}":
        if day < 21:
            day += 10
        else:
            month = month + 1 if month < 12 else 1
            year = year + 1 if month == 1 else year
            day = 1
        days.append(day)
        months.append(month)
        years.append(year)
    return np.array(days), np.array(months), np.array(years)


def get_monthly_timestamp_components(
    start_date: np.datetime64, end_date: np.datetime64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate monthly timestamp components (day, month, year) between a start and end date.

    Parameters
    ----------
    start_date : np.datetime64
        The starting date from which to generate month timestamps.
    end_date : np.datetime64
        The ending date up to which to generate month timestamps.

    Returns
    -------
    days : np.ndarray
        Array of day components for each month timestamp.
    months : np.ndarray
        Array of month components for each month timestamp.
    years : np.ndarray
        Array of year components for each month timestamp.
    """

    # Align start and end dates to the first day of the month
    start_date = align_to_composite_window(start_date, "month")
    end_date = align_to_composite_window(end_date, "month")

    # Truncate to month precision (year and month only, day is dropped)
    start_month = start_date.astype("datetime64[M]")
    end_month = end_date.astype("datetime64[M]")
    num_timesteps = (end_month - start_month).astype(int) + 1

    # generate date vector based on the number of timesteps
    date_vector = start_month + np.arange(num_timesteps, dtype="timedelta64[M]")

    # generate day, month and year vectors with numpy operations
    days = np.ones(len(date_vector), dtype=int)
    months = (date_vector.astype("datetime64[M]").astype(int) % 12) + 1
    years = (date_vector.astype("datetime64[Y]").astype(int)) + 1970
    return days, months, years
