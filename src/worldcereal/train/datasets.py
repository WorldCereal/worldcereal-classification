import calendar
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
from torch.utils.data import Dataset, WeightedRandomSampler

from worldcereal.seasons import season_doys_to_dates_refyear
from worldcereal.train import (
    GLOBAL_SEASON_IDS,
    MIN_EDGE_BUFFER,
    SEASONALITY_COLUMN_MAP,
    SEASONALITY_LAT_RANGE,
    SEASONALITY_LON_RANGE,
    SEASONALITY_LOOKUP_COLUMNS,
    SEASONALITY_LOOKUP_FILENAME,
    SEASONALITY_LOOKUP_PACKAGE,
    SEASONALITY_LOOKUP_PATH,
)
from worldcereal.train import predictors as _predictor_utils
from worldcereal.train.seasonal import align_to_composite_window

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


@dataclass(frozen=True)
class SeasonWindow:
    start_month: int
    start_day: int
    end_month: int
    end_day: int
    year_offset: int = 0


SAMPLE_ATTR_COLUMNS: Tuple[str, ...] = (
    "lat",
    "lon",
    "ref_id",
    "sample_id",
    "finetune_class",
    "downstream_class",
    "landcover_label",
    "croptype_label",
    "valid_time",
    "quality_score_lc",
    "quality_score_ct",
    "confidence_nonoutlier",
    "anomaly_flag",
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


def _coerce_date_for_year(year: int, month: int, day: int) -> np.datetime64:
    """Build a numpy datetime64, clamping the day to the month's max if needed."""

    last_day = calendar.monthrange(year, month)[1]
    safe_day = min(day, last_day)
    return np.datetime64(f"{year:04d}-{month:02d}-{safe_day:02d}", "D")


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
        if end_np < start_np:
            raise ValueError(
                f"Season window for '{season}' has end date {end_dt} before start date {start_dt}."
            )

        start_ts = pd.Timestamp(start_np)
        end_ts = pd.Timestamp(end_np)
        year_offset = end_ts.year - start_ts.year
        if year_offset < 0:
            raise ValueError(
                f"Season window for '{season}' has end date {end_dt} before start date {start_dt}."
            )
        if year_offset > 1:
            raise ValueError(
                f"Season window for '{season}' spans more than one year; only single-year crossings are supported."
            )

        normalized[season] = SeasonWindow(
            start_month=start_ts.month,
            start_day=start_ts.day,
            end_month=end_ts.month,
            end_day=end_ts.day,
            year_offset=year_offset,
        )

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


def _timestamp_in_season_window(
    timestamp: Optional[pd.Timestamp],
    *,
    window: SeasonWindow,
) -> bool:
    if timestamp is None or pd.isna(timestamp):
        return False

    ts = pd.Timestamp(timestamp)

    if window.year_offset not in (0, 1):
        raise ValueError("Season windows must span at most 12 months.")

    if window.year_offset == 0:
        start_year = ts.year
    else:
        ts_tuple = (ts.month, ts.day)
        end_tuple = (window.end_month, window.end_day)
        start_year = ts.year if ts_tuple > end_tuple else ts.year - 1

    start_dt = pd.Timestamp(
        _coerce_date_for_year(start_year, window.start_month, window.start_day)
    )
    end_dt = pd.Timestamp(
        _coerce_date_for_year(
            start_year + window.year_offset, window.end_month, window.end_day
        )
    )
    return start_dt <= ts <= end_dt


def _filter_frame_by_manual_windows(
    dataframe: pd.DataFrame,
    season_windows: Mapping[str, SeasonWindow],
    *,
    context: str,
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
            lambda ts: _timestamp_in_season_window(ts, window=window)
        )

    missing = label_datetimes.isna().sum()
    if missing:
        logger.warning(
            "%s: Dropping %d samples missing valid_time while enforcing manual season window(s).",
            context,
            int(missing),
        )
        keep_mask &= label_datetimes.notna()

    dropped = int((~keep_mask).sum())
    if dropped:
        ranges = ", ".join(
            f"{season}: {window.start_month:02d}-{window.start_day:02d} -> {window.end_month:02d}-{window.end_day:02d}"
            for season, window in season_windows.items()
        )
        logger.info(
            "%s: Removed %d samples outside manual season window(s): %s",
            context,
            dropped,
            ranges,
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
        # Effective number of samples (Class-Balanced Loss)
        # beta close to 1.0 -> smoother, less extreme weights
        beta = 0.999
        effective_num = (1.0 - np.power(beta, freq)) / (1.0 - beta)
        weights = 1.0 / effective_num

    elif method == "none":
        weights = np.ones_like(freq)

    else:
        raise ValueError(f"Unknown method: {method}")

    if clip_range:
        logger.info(f"Clipping weights to range {clip_range}")
        weights = np.clip(weights, clip_range[0], clip_range[1])

    if normalize:
        logger.info("Renormalizing weights to mean = 1")
        weights = weights / weights.mean()

    rounded = np.round(weights, 3)
    return {cls: float(weight) for cls, weight in zip(classes, rounded.tolist())}


def _stringify_weight_dict(weights: Dict[Hashable, float]) -> Dict[str, float]:
    """Convert potentially mixed-type keys to strings for deterministic indexing."""

    return {str(key): float(value) for key, value in weights.items()}


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


@dataclass
class SensorMaskingConfig:
    """Configuration for simulating real-world missing data scenarios.

    Probabilities are applied independently per sample. Values are in [0,1].
    Set config to None or enabled=False to disable masking.

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

    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_timesteps: int = 12,
        timestep_freq: Literal["month", "dekad"] = "month",
        task_type: Literal["ssl", "binary", "multiclass"] = "ssl",
        num_outputs: Optional[int] = None,
        augment: bool = False,
        masking_config: Optional[SensorMaskingConfig] = None,
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
        """
        self.dataframe = dataframe.replace({np.nan: NODATAVALUE})
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
        if self.masking_config:
            if self.masking_config.seed is not None:
                # set a per-dataset RNG seed (numpy global for simplicity)
                np.random.seed(self.masking_config.seed)
            self.masking_config.validate(self.num_timesteps)
            if self.masking_config.enable:
                logger.info(
                    "Sensor masking enabled for this dataset with config: {}".format(
                        self.masking_config
                    )
                )
            else:
                logger.info(
                    "Sensor masking config provided but enable=False; masking disabled."
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

        return timestep_positions, valid_position

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
        5. Per-timestep meteo dropout.
        6. DEM dropout.
        """
        # Guard: if masking_config is None (should not happen when enable checked)
        if self.masking_config is None:
            return s1, s2, meteo, dem
        cfg: SensorMaskingConfig = self.masking_config  # type narrowing for mypy
        T = self.num_timesteps
        # 1. Full S1 dropout
        if np.random.rand() < cfg.s1_full_dropout_prob:
            s1[:] = NODATAVALUE
            # logger.debug("Applied full S1 dropout")
        else:
            # 2. Per-timestep S1 dropout
            if cfg.s1_timestep_dropout_prob > 0:
                s1_mask = np.random.rand(T) < cfg.s1_timestep_dropout_prob
                if s1_mask.any():
                    s1[..., s1_mask, :] = NODATAVALUE
                    # logger.debug(
                    #     f"Applied S1 timestep dropout on {s1_mask.sum()} of {T} timesteps"
                    # )

        # 3. S2 contiguous cloud block
        if cfg.s2_cloud_block_prob > 0 and np.random.rand() < cfg.s2_cloud_block_prob:
            block_len = np.random.randint(
                cfg.s2_cloud_block_min, cfg.s2_cloud_block_max + 1
            )
            if block_len >= T:
                start = 0
                end = T
            else:
                start = np.random.randint(0, T - block_len + 1)
                end = start + block_len
            s2[..., start:end, :] = NODATAVALUE
            # logger.debug(
            #     f"Applied S2 cloud block dropout from timestep {start} to {end - 1} (len={block_len})"
            # )

        # 4. Per-timestep S2 cloud dropout (skip already-masked timesteps)
        if cfg.s2_cloud_timestep_prob > 0:
            s2_mask = np.random.rand(T) < cfg.s2_cloud_timestep_prob
            # Avoid double logging of block; still mask independent timesteps not in block
            newly_masked = s2_mask & (s2[0, 0, :, 0] != NODATAVALUE)
            if newly_masked.any():
                s2[..., newly_masked, :] = NODATAVALUE
                # logger.debug(
                #     f"Applied S2 per-timestep cloud masking on {newly_masked.sum()} timesteps"
                # )

        # 5. Meteo per-timestep dropout
        if cfg.meteo_timestep_dropout_prob > 0:
            meteo_mask = np.random.rand(T) < cfg.meteo_timestep_dropout_prob
            if meteo_mask.any():
                meteo[..., meteo_mask, :] = NODATAVALUE
                # logger.debug(
                #     f"Applied meteo timestep dropout on {meteo_mask.sum()} timesteps"
                # )

        # 6. DEM dropout
        if cfg.dem_dropout_prob > 0 and np.random.rand() < cfg.dem_dropout_prob:
            dem[:] = NODATAVALUE
            # logger.debug("Applied DEM dropout")

        return s1, s2, meteo, dem

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
            if key in row and not _is_missing_value(value):
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
            start_dt = _coerce_date_for_year(year, window.start_month, window.start_day)
            end_year = year + window.year_offset
            end_dt = _coerce_date_for_year(end_year, window.end_month, window.end_day)
            start_aligned = align_to_composite_window(start_dt, self.timestep_freq)
            end_aligned = align_to_composite_window(end_dt, self.timestep_freq)
            if end_aligned < start_aligned:
                continue
            slots = self._enumerate_composite_slots(start_aligned, end_aligned)
            if not slots:
                continue
            if not self._has_full_window_coverage(composite_dates, slots):
                continue
            cycles.append((start_aligned, end_aligned))
            mask |= (composite_dates >= start_aligned) & (
                composite_dates <= end_aligned
            )

        in_flag = False
        if label_datetime is not None:
            for start_aligned, end_aligned in cycles:
                if start_aligned <= label_datetime <= end_aligned:
                    in_flag = True
                    break

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
                season_name, row, target_year, lat, lon
            )
        except Exception as exc:  # pragma: no cover - guard rare failures
            sample_id = row.get("sample_id", "n/a")
            raise RuntimeError(
                f"Failed to derive season '{season_name}' for sample {sample_id}: {exc}"
            ) from exc

        start_aligned = align_to_composite_window(start_dt, self.timestep_freq)
        end_aligned = align_to_composite_window(end_dt, self.timestep_freq)
        slots = self._enumerate_composite_slots(start_aligned, end_aligned)
        has_full_coverage = self._has_full_window_coverage(composite_dates, slots)
        if has_full_coverage:
            mask = (composite_dates >= start_aligned) & (composite_dates <= end_aligned)
        else:
            mask = np.zeros_like(composite_dates, dtype=bool)
        if label_datetime is not None and has_full_coverage:
            in_flag = bool(start_dt <= label_datetime <= end_dt)
        else:
            in_flag = False
        return mask.astype(bool, copy=False), in_flag

    def _has_full_window_coverage(
        self, composite_dates: np.ndarray, slots: Sequence[np.datetime64]
    ) -> bool:
        """Return True when every composite slot in ``slots`` exists in the sample."""
        if not slots or composite_dates.size == 0:
            return False
        available = set(
            composite_dates.astype("datetime64[D]").astype(np.int64).tolist()
        )
        required = set(
            np.array(list(slots), dtype="datetime64[D]").astype(np.int64).tolist()
        )
        return required.issubset(available)

    def _enumerate_composite_slots(
        self, start: np.datetime64, end: np.datetime64
    ) -> List[np.datetime64]:
        if end < start:
            return []
        slots: List[np.datetime64] = []
        current = start
        while True:
            slots.append(current)
            if current >= end:
                break
            current = self._advance_composite_slot(current)
        return slots

    def _advance_composite_slot(self, current: np.datetime64) -> np.datetime64:
        current = current.astype("datetime64[D]")
        if self.timestep_freq == "month":
            month_step = current.astype("datetime64[M]") + np.timedelta64(1, "M")
            return month_step.astype("datetime64[D]")
        if self.timestep_freq == "dekad":
            month_start = current.astype("datetime64[M]")
            # Offset from first day determines which dekad we are in.
            offset_days = (current - month_start).astype(int)
            if offset_days == 0:
                return current + np.timedelta64(10, "D")
            if offset_days == 10:
                return current + np.timedelta64(10, "D")
            if offset_days == 20:
                next_month = month_start + np.timedelta64(1, "M")
                return next_month.astype("datetime64[D]")
            raise ValueError("Dekad slots must align to days 1, 11, or 21.")
        raise ValueError(f"Unknown timestep frequency '{self.timestep_freq}'")

    def _season_context_for(
        self,
        season_id: str,
        row: Mapping[str, Any],
        year: int,
        lat: float,
        lon: float,
    ) -> Tuple[np.datetime64, np.datetime64]:
        """Fetch (start, end) dates for a season/grid cell from the lookup."""

        lat_center, lon_center = _snap_latlon_to_calendar_grid(lat, lon)
        try:
            sos_col, eos_col = SEASONALITY_COLUMN_MAP[season_id]
        except KeyError as exc:
            raise ValueError(
                f"Season '{season_id}' is not available in the seasonality lookup."
            ) from exc

        table = _ensure_seasonality_lookup()
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
            logger.debug(
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

        start_dt, end_dt = season_doys_to_dates_refyear(sos_doy, eos_doy, year)
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
            )
            if dropped:
                logger.info(
                    f"{self.__class__.__name__}: proceeding with {len(filtered_df)} samples after enforcing manual season window(s)."
                )
            self.dataframe = filtered_df

    def __getitem__(self, idx):
        row = pd.Series.to_dict(self.dataframe.iloc[idx, :])
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

        predictors = Predictors(**inputs, label=label)
        return predictors, attrs

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

    def get_balanced_sampler(
        self,
        method: str = "balanced",
        clip_range: Optional[tuple] = None,  # e.g. (0.2, 10.0)
        normalize: bool = True,
        generator: Optional[Any] = None,
        sampling_class: str = "finetune_class",
    ) -> "WeightedRandomSampler":
        """
        Build a WeightedRandomSampler so that rare classes (from `balancing_class`)
        are upsampled and common classes downsampled.
        max_upsample:
            maximum upsampling factor for the rarest class (e.g. 10 means
            no class will be sampled >10× more than its frequency).
        sampling_class:
            column name in the dataframe to use for balancing.
            Default is `finetune_class`, which is the class label
            used in the training. `balancing_class` can be used as well.
        """
        # extract the sampling class (strings or ints)
        bc_vals = self.dataframe[sampling_class].values

        logger.info("Computing class weights ...")
        class_weights = get_class_weights(
            bc_vals, method, clip_range=clip_range, normalize=normalize
        )
        logger.info(f"Class weights: {class_weights}")

        # per‐sample weight
        sample_weights = np.ones_like(bc_vals).astype(np.float32)
        for k, v in class_weights.items():
            sample_weights[bc_vals == k] = v

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return sampler

    def get_task_balanced_sampler(
        self,
        *,
        task_column: str = "label_task",
        task_weight_method: str = "balanced",
        class_column_map: Optional[Mapping[str, str]] = None,
        class_weight_method: str = "balanced",
        spatial_group_column: Optional[str] = None,
        spatial_bin_size_degrees: Optional[float] = None,
        spatial_weight_method: str = "log",
        clip_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        generator: Optional[Any] = None,
        fallback_sampling_class: str = "finetune_class",
    ) -> "WeightedRandomSampler":
        """Build weights that balance tasks, classes, and spatial density."""

        def _log_weight_stats(name: str, values: np.ndarray) -> None:
            if values.size == 0:
                logger.warning(f"{name}: no values to describe")
                return
            stats = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
            logger.info(f"{name} stats: {stats}")

        if task_column not in self.dataframe.columns:
            raise ValueError(f"Task column '{task_column}' not found in dataframe")

        task_series = self.dataframe[task_column]
        if task_series.isna().any():
            raise ValueError(
                "Task column contains missing values; balancing requires explicit task IDs"
            )

        tasks = task_series.astype(str).to_numpy()
        unique_tasks = sorted(set(tasks.tolist()))
        task_counts = {task: int((tasks == task).sum()) for task in unique_tasks}
        logger.info(
            f"Building task-balanced sampler for {len(tasks)} samples across tasks: {task_counts}"
        )

        effective_task_counts = dict(task_counts)
        if {"landcover", "croptype"}.issubset(effective_task_counts):
            effective_task_counts["landcover"] = len(tasks)
            logger.info(
                f"Treating landcover as supervising all samples for task weighting: {effective_task_counts}"
            )

        # Task-level weights
        task_weights = _stringify_weight_dict(
            get_class_weights(
                tasks,
                method=task_weight_method,
                clip_range=None,
                normalize=normalize,
                counts_override=effective_task_counts,
            )
        )
        logger.info(f"Task weights per task: {task_weights}")
        task_weight_vec = np.array(
            [task_weights[str(task)] for task in tasks], dtype=np.float64
        )
        _log_weight_stats("Task weight vector", task_weight_vec)

        # Class-level weights (within each task)
        class_weight_vec = np.ones_like(task_weight_vec)
        class_column_map = class_column_map or {}
        for task_name in unique_tasks:
            class_column = class_column_map.get(task_name, fallback_sampling_class)
            if class_column not in self.dataframe.columns:
                raise ValueError(
                    f"Class column '{class_column}' required for task '{task_name}' but not found in dataframe"
                )

            mask = tasks == task_name
            if not np.any(mask):
                continue

            class_values = self.dataframe.loc[mask, class_column].astype(str).to_numpy()
            class_counts = Counter(class_values.tolist())
            class_weights = _stringify_weight_dict(
                get_class_weights(
                    class_values,
                    method=class_weight_method,
                    clip_range=(0.1, 10.0),
                    normalize=normalize,
                )
            )
            logger.info(f"[Task={task_name}] class counts: {class_counts}")
            logger.info(f"[Task={task_name}] class weights: {class_weights}")

            class_weight_vec[mask] = np.array(
                [class_weights[str(value)] for value in class_values], dtype=np.float64
            )
        _log_weight_stats("Class weight vector", class_weight_vec)

        # Spatial weights (down-weight dense regions, up-weight sparse ones)
        spatial_weight_vec = np.ones_like(task_weight_vec)
        if spatial_group_column or spatial_bin_size_degrees is not None:
            if spatial_group_column:
                if spatial_group_column not in self.dataframe.columns:
                    raise ValueError(
                        f"Spatial group column '{spatial_group_column}' not found in dataframe"
                    )
                spatial_groups = (
                    self.dataframe[spatial_group_column].astype(str).to_numpy()
                )
                if pd.isna(spatial_groups).any():
                    raise ValueError(
                        f"Spatial group column '{spatial_group_column}' contains missing values"
                    )
            else:
                if spatial_bin_size_degrees is None:
                    raise ValueError(
                        "spatial_bin_size_degrees must be provided when spatial_group_column is not set"
                    )
                if (
                    "lat" not in self.dataframe.columns
                    or "lon" not in self.dataframe.columns
                ):
                    raise ValueError(
                        "Latitude/longitude columns are required to compute spatial bins"
                    )
                spatial_groups = _spatial_bins_from_latlon(
                    self.dataframe["lat"],
                    self.dataframe["lon"],
                    spatial_bin_size_degrees,
                )

            spatial_weights = _stringify_weight_dict(
                get_class_weights(
                    spatial_groups,
                    method=spatial_weight_method,
                    clip_range=(0.1, 10.0),
                    normalize=normalize,
                )
            )
            spatial_weight_vec = np.array(
                [spatial_weights[str(group)] for group in spatial_groups],
                dtype=np.float64,
            )
            logger.info(f"Spatial weights: {spatial_weights}")
        _log_weight_stats("Spatial weight vector", spatial_weight_vec)

        combined = task_weight_vec * class_weight_vec * spatial_weight_vec
        if clip_range is not None:
            combined = np.clip(combined, clip_range[0], clip_range[1])
        else:
            combined = np.clip(combined, 1e-6, None)
        _log_weight_stats("Combined sampling weights", combined)

        weight_tensor = torch.as_tensor(combined, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=weight_tensor.tolist(),
            num_samples=len(combined),
            replacement=True,
            generator=generator,
        )
        sampler.weights = weight_tensor  # type: ignore[attr-defined]
        return sampler


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
            )
            if dropped:
                logger.info(
                    f"{self.__class__.__name__}: proceeding with {len(filtered_df)} samples after enforcing manual season window(s)."
                )
            self.dataframe = filtered_df

        repeats = _check_augmentation_settings(augment, masking_config, repeats)

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

        # Get the sample
        sample = super().__getitem__(real_idx)
        row_series = self.dataframe.iloc[real_idx, :]
        row = pd.Series.to_dict(row_series)
        timestep_positions, valid_position = self.get_timestep_positions(row)
        relative_valid = valid_position - timestep_positions[0]

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
