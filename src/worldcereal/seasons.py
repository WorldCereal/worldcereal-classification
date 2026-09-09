"""
This module provides functions to retrieve crop calendar seasonality information
from the WorldCereal seasonality lookup geoparquet.
This geoparquet represents a simplified version of the global WorldCereal crop calendars.

Originally, crop calendars in WorldCereal have always been represented by the DOY (day of year) metric.
Due to the circular nature of this metric however and the fact that some seasons cross calendar years,
it is not always straightforward to compute the start and end dates of a season from DOY values.

To address this, we introduced the concept of "season dekads" as a more robust representation of crop calendars.
Each month has 3 dekads:
1-10: first dekad of the month
11-20: second dekad of the month
21-last day of the month: third dekad of the month

Dekads are expressed on a 3-year scale, where dekads 1-36 represent the first year, 37-72 the second year, and 73-108 the third year.
We chose a 3 year scale because for a given target year for map generation, a season can start in the year before and end in the year after,
so we need to be able to represent all three years.

The main function to be used to access the seasonality information is `get_season_dates_for_extent`, 
which returns a TemporalContext object with the start and end dates of the season for a given extent and year.

"""


import datetime
import json
import math
from typing import (
    Callable,
    Dict,
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
from loguru import logger
from openeo_gfmap import BoundingBoxExtent, TemporalContext

from worldcereal import SUPPORTED_SEASONS
from worldcereal.data import cropcalendars

_SEASONALITY_LOOKUP_TABLE: Optional[pd.DataFrame] = None
DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES = 5.0
DEFAULT_MAX_DEKAD_DIFFERENCE = 7

# Seasons whose union defines the processing period of a production grid cell.
PROCESSING_SEASONS: Tuple[str, ...] = ("tc-s1", "tc-s2")
PROCESSING_PERIOD_MONTHS = 12


def ensure_seasonality_lookup_table() -> pd.DataFrame:
    """Load and cache the seasonality lookup table indexed by lat/lon centers."""

    global _SEASONALITY_LOOKUP_TABLE
    if _SEASONALITY_LOOKUP_TABLE is not None:
        return _SEASONALITY_LOOKUP_TABLE

    table = cropcalendars.load_seasonality_lookup()
    required = {"lat", "lon", *cropcalendars.SEASONALITY_LOOKUP_COLUMNS}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Seasonality lookup parquet is missing required columns: {sorted(missing)}"
        )

    table = table.astype({"lat": np.float64, "lon": np.float64})
    table = table.set_index(["lat", "lon"])
    if not table.index.is_unique:
        raise ValueError("Seasonality lookup index must be unique per lat/lon cell.")

    _SEASONALITY_LOOKUP_TABLE = table[
        list(cropcalendars.SEASONALITY_LOOKUP_COLUMNS)
    ].sort_index()
    return _SEASONALITY_LOOKUP_TABLE


def _snap_coordinate_to_lookup_grid(
    value: float, bounds: Tuple[float, float]
) -> float:
    """Snap a coordinate to the 0.5 deg grid center used by the lookup."""

    min_value, max_value = bounds
    clamped = max(min(float(value), max_value), min_value)
    return (math.floor(clamped * 2.0) / 2.0) + 0.25


def resolve_cropcalendar_columns(
    season_id: str,
) -> Tuple[str, str]:
    """Resolve season identifier and parameter to SOS/EOS parquet columns."""

    try:
        sos_dekad_col, eos_dekad_col = cropcalendars.SEASONALITY_COLUMN_MAP[season_id]
    except KeyError as exc:
        raise ValueError(
            f"Season '{season_id}' is not available in the seasonality lookup. "
            f"Known seasons: {sorted(cropcalendars.SEASONALITY_COLUMN_MAP)}"
        ) from exc

    return (
        sos_dekad_col,
        eos_dekad_col,
    )


def _extent_to_wgs84_bounds(extent: BoundingBoxExtent) -> Tuple[float, float, float, float]:
    """Return extent bounds in EPSG:4326 as (west, south, east, north)."""

    west, south, east, north = (
        float(extent.west),
        float(extent.south),
        float(extent.east),
        float(extent.north),
    )
    epsg = int(getattr(extent, "epsg", 4326))
    if epsg == 4326:
        return west, south, east, north

    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValueError(
            "Extent EPSG is not 4326 and pyproj is not available to reproject "
            f"(epsg={epsg})."
        ) from exc

    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    corners = [
        transformer.transform(west, south),
        transformer.transform(west, north),
        transformer.transform(east, south),
        transformer.transform(east, north),
    ]
    lons = [lon for lon, _ in corners]
    lats = [lat for _, lat in corners]
    return min(lons), min(lats), max(lons), max(lats)


def _lookup_row_for_point(
    lat: float,
    lon: float,
    *,
    fallback_to_nearest: bool = True,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
) -> pd.Series:
    """Return the single lookup row representing a point.

    The point is snapped to the lookup's 0.5 deg grid centers. When the snapped
    cell is missing and ``fallback_to_nearest`` is enabled, the nearest available
    cell within ``max_fallback_distance_degrees`` is returned instead.
    """

    if (
        not math.isfinite(max_fallback_distance_degrees)
        or max_fallback_distance_degrees < 0
    ):
        raise ValueError(
            "max_fallback_distance_degrees must be a finite non-negative value"
        )

    table = ensure_seasonality_lookup_table()
    lat_center = _snap_coordinate_to_lookup_grid(
        lat, cropcalendars.SEASONALITY_LAT_RANGE
    )
    lon_center = _snap_coordinate_to_lookup_grid(
        lon, cropcalendars.SEASONALITY_LON_RANGE
    )

    try:
        return table.loc[(lat_center, lon_center)]
    except KeyError as exc:
        lat_vals = table.index.get_level_values("lat").to_numpy()
        lon_vals = table.index.get_level_values("lon").to_numpy()
        if not fallback_to_nearest or lat_vals.size == 0:
            raise ValueError(
                "No seasonality record found for snapped lat/lon "
                f"({lat_center}, {lon_center})."
            ) from exc

        distances = (lat_vals - lat_center) ** 2 + (lon_vals - lon_center) ** 2
        best_idx = int(distances.argmin())
        fallback_distance = math.sqrt(float(distances[best_idx]))
        if fallback_distance > max_fallback_distance_degrees:
            raise ValueError(
                "Nearest seasonality lookup cell is too far from snapped "
                f"lat/lon ({lat_center}, {lon_center}): distance is "
                f"{fallback_distance:.3f} degrees, maximum allowed is "
                f"{max_fallback_distance_degrees:.3f} degrees."
            ) from exc
        logger.error(
            f"Seasonality lookup missing ({lat_center}, {lon_center}); "
            f"using nearest cell ({lat_vals[best_idx]}, {lon_vals[best_idx]})."
        )
        return table.iloc[best_idx]


def _in_extent_lookup_rows(extent: BoundingBoxExtent) -> pd.DataFrame:
    """Return all lookup rows whose cell center falls inside the extent."""

    west, south, east, north = _extent_to_wgs84_bounds(extent)
    table = ensure_seasonality_lookup_table()

    lat_vals = table.index.get_level_values("lat").to_numpy()
    lon_vals = table.index.get_level_values("lon").to_numpy()
    mask_lat = (lat_vals >= south) & (lat_vals <= north)
    if west <= east:
        mask_lon = (lon_vals >= west) & (lon_vals <= east)
    else:
        mask_lon = (lon_vals >= west) | (lon_vals <= east)

    return table.iloc[np.flatnonzero(mask_lat & mask_lon)]


def _require_season_columns(season_id: str, table: pd.DataFrame) -> Tuple[str, str]:
    """Resolve and validate the SOS/EOS columns of a season."""

    sos_col, eos_col = resolve_cropcalendar_columns(season_id)
    if sos_col not in table.columns or eos_col not in table.columns:
        raise ValueError(
            f"Season '{season_id}' requires columns ({sos_col}, {eos_col}) "
            "but they are not present in the seasonality lookup parquet."
        )
    return sos_col, eos_col


def fetch_cropcalendar_dekad_point(
    season_id: str,
    lat: float,
    lon: float,
    *,
    fallback_to_nearest: bool = True,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
) -> Tuple[int, int]:
    """Fetch (SOS, EOS) dekad values for one point from the global parquet lookup.

    The input point is snapped to the lookup's 0.5 deg grid centers before querying.
    If the snapped cell is missing and ``fallback_to_nearest`` is enabled, the
    nearest available lookup cell is used.

    Parameters
    ----------
    season_id : str
        Season identifier (e.g. ``tc-s1``, ``tc-s2``, ``tc-annual``).
    lat, lon : float
        Input point coordinates.
    fallback_to_nearest : bool, default True
        Whether to use the nearest available lookup cell when the snapped cell
        is not present in the table.
    max_fallback_distance_degrees : float, default 5.0
        Maximum Euclidean distance in latitude/longitude degrees between the
        snapped coordinate and a fallback lookup cell.
    """

    sos_col, eos_col = _require_season_columns(
        season_id, ensure_seasonality_lookup_table()
    )
    row = _lookup_row_for_point(
        lat,
        lon,
        fallback_to_nearest=fallback_to_nearest,
        max_fallback_distance_degrees=max_fallback_distance_degrees,
    )

    sos_value = int(row[sos_col])
    eos_value = int(row[eos_col])
    if sos_value <= 0 or eos_value <= 0:
        logger.warning(
            "Seasonality lookup returned nodata values for "
            f"season '{season_id}'."
        )
        raise ValueError(
            "Seasonality lookup returned nodata values for "
            f"season '{season_id}'."
        )

    if sos_value > 108 or eos_value > 108:
        logger.warning(
            "Seasonality lookup returned invalid dekad values for "
            f"season '{season_id}': SOS={sos_value}, EOS={eos_value}. "
            "Valid dekad range is 1-108."
        )
        raise ValueError(
            "Seasonality lookup returned invalid dekad values for "
            f"season '{season_id}': SOS={sos_value}, EOS={eos_value}. "
            "Valid dekad range is 1-108."
        )

    return sos_value, eos_value


def fetch_cropcalendar_dekad_points_batch(
    season_id: str,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized counterpart of `fetch_cropcalendar_dekad_point` for many points.

    Every point is snapped to the lookup grid and, when the snapped cell is
    missing, resolved via nearest-cell fallback (mirroring the per-point
    fallback of `fetch_cropcalendar_dekad_point`, but batched across all
    input points at once for speed).

    Parameters
    ----------
    season_id : str
        Season identifier (e.g. ``tc-s1``, ``tc-s2``, ``tc-annual``).
    lats, lons : np.ndarray
        Input point coordinates.
    max_fallback_distance_degrees : float, default 5.0
        Maximum Euclidean distance in latitude/longitude degrees between a
        snapped coordinate and its fallback lookup cell.

    Returns
    -------
    sos, eos : np.ndarray (int64)
        SOS/EOS dekad values, one per input point. Entries where `invalid`
        is True hold a placeholder value (1) and must not be used directly.
    invalid : np.ndarray (bool)
        True where the lookup returned nodata, an out-of-range dekad, or (when
        no lookup cell was within `max_fallback_distance_degrees`) no usable
        fallback was found.
    """
    sos_col, eos_col = resolve_cropcalendar_columns(season_id)
    table = ensure_seasonality_lookup_table()
    if sos_col not in table.columns or eos_col not in table.columns:
        raise ValueError(
            f"Season '{season_id}' requires columns ({sos_col}, {eos_col}) "
            "but they are not present in the seasonality lookup parquet."
        )

    lat_arr = np.asarray(lats, dtype=np.float64)
    lon_arr = np.asarray(lons, dtype=np.float64)
    lat_c = (
        np.floor(np.clip(lat_arr, *cropcalendars.SEASONALITY_LAT_RANGE) * 2.0) / 2.0
    ) + 0.25
    lon_c = (
        np.floor(np.clip(lon_arr, *cropcalendars.SEASONALITY_LON_RANGE) * 2.0) / 2.0
    ) + 0.25

    key_index = pd.MultiIndex.from_arrays([lat_c, lon_c], names=["lat", "lon"])
    joined = table.reindex(key_index)

    missing = joined[sos_col].isna().to_numpy() | joined[eos_col].isna().to_numpy()
    if missing.any():
        lat_vals = table.index.get_level_values("lat").to_numpy()
        lon_vals = table.index.get_level_values("lon").to_numpy()
        joined = joined.reset_index(drop=True)
        missing_pos = np.flatnonzero(missing)
        missing_cells = {(float(lat_c[i]), float(lon_c[i])) for i in missing_pos}
        for cell_lat, cell_lon in missing_cells:
            distances = (lat_vals - cell_lat) ** 2 + (lon_vals - cell_lon) ** 2
            best_idx = int(distances.argmin())
            fallback_distance = math.sqrt(float(distances[best_idx]))
            cell_mask = missing & (lat_c == cell_lat) & (lon_c == cell_lon)
            if fallback_distance > max_fallback_distance_degrees:
                logger.error(
                    "Nearest seasonality lookup cell is too far from snapped "
                    f"lat/lon ({cell_lat}, {cell_lon}): distance is "
                    f"{fallback_distance:.3f} degrees, maximum allowed is "
                    f"{max_fallback_distance_degrees:.3f} degrees."
                )
                continue
            logger.error(
                f"Seasonality lookup missing ({cell_lat}, {cell_lon}); using "
                f"nearest cell ({lat_vals[best_idx]}, {lon_vals[best_idx]})."
            )
            joined.iloc[np.flatnonzero(cell_mask)] = table.iloc[best_idx]

    sos = joined[sos_col].to_numpy(dtype=np.float64)
    eos = joined[eos_col].to_numpy(dtype=np.float64)
    invalid = (
        ~np.isfinite(sos)
        | ~np.isfinite(eos)
        | (sos <= 0)
        | (eos <= 0)
        | (sos > 108)
        | (eos > 108)
    )
    sos_i = np.where(invalid, 1, sos).astype(np.int64)
    eos_i = np.where(invalid, 1, eos).astype(np.int64)
    return sos_i, eos_i, invalid


def _valid_dekad_mask(sos_arr: np.ndarray, eos_arr: np.ndarray) -> np.ndarray:
    """Mask of entries holding usable dekad values for both SOS and EOS."""

    return (
        (sos_arr > 0) & (sos_arr <= 108) & (eos_arr > 0) & (eos_arr <= 108)
    )


def dekad_medoid_index(values: np.ndarray) -> int:
    """Index of the observed row that best represents all rows.

    `values` holds one row per lookup point and one column per dekad variable
    (SOS and EOS of every season), so seasons are represented jointly by simply
    widening the array. The medoid is the row minimising the summed L1 distance
    to all other rows.

    Aggregating each variable independently (e.g. with a median per column) can
    yield a season that occurs nowhere in the extent, and whose length is an
    artefact of mixing distinct seasonality regimes. The medoid instead returns
    an actually observed row, so both the season lengths and the combination of
    seasons always remain realistic.

    Distances are plain absolute differences on the 1-108 dekad scale, which
    keeps the year an observation is anchored to meaningful: dekad 1 and dekad
    37 are the same time of year but a full year apart, and are treated as such.
    """

    values = np.atleast_2d(np.asarray(values, dtype=np.int64))
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("dekad_medoid_index requires a non-empty 2D array.")

    cost = np.zeros(values.shape[0], dtype=np.int64)
    for column in values.T:
        cost += np.abs(column[:, None] - column[None, :]).sum(axis=1)

    # Ties are broken on (cost, then each column in order) for reproducibility.
    keys = tuple(values[:, ::-1].T) + (cost,)
    return int(np.lexsort(keys)[0])


def _jointly_valid_seasons(
    season_ids: Sequence[str],
    valid_masks: Mapping[str, np.ndarray],
) -> Tuple[List[str], Optional[np.ndarray]]:
    """Largest set of seasons that are jointly valid on at least one row.

    Seasons are added greedily, best covered first, so that regions where a
    season simply does not exist (nodata) drop that season instead of losing all
    candidate rows.
    """

    ordered = sorted(season_ids, key=lambda sid: -int(valid_masks[sid].sum()))
    retained: List[str] = []
    mask: Optional[np.ndarray] = None
    for season_id in ordered:
        candidate = (
            valid_masks[season_id] if mask is None else mask & valid_masks[season_id]
        )
        if not candidate.any():
            continue
        retained.append(season_id)
        mask = candidate

    return [sid for sid in season_ids if sid in retained], mask


def _check_extent_homogeneity(
    rows: pd.DataFrame,
    columns: Mapping[str, Tuple[str, str]],
    max_dekad_difference: int,
    on_heterogeneity: Literal["warn", "raise"],
) -> None:
    """Report seasons whose dekad spread inside the extent is too large."""

    problems = []
    for season_id, (sos_col, eos_col) in columns.items():
        for label, column in (("SOS", sos_col), ("EOS", eos_col)):
            values = rows[column].to_numpy(dtype=np.int64)
            spread = int(values.max() - values.min())
            if spread > max_dekad_difference:
                problems.append(f"{season_id} {label} spans {spread} dekads")

    if not problems:
        return

    message = (
        "Seasonality inside the extent is heterogeneous "
        f"({'; '.join(problems)}; maximum allowed is {max_dekad_difference} "
        "dekads), so no single crop calendar represents it well. Consider "
        "downsizing your area of interest."
    )
    if on_heterogeneity == "raise":
        raise ValueError(message)
    logger.warning(message)


def fetch_cropcalendar_dekads_extent(
    season_ids: Sequence[str],
    extent: BoundingBoxExtent,
    *,
    fallback_to_nearest: bool = True,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
    max_dekad_difference: int = DEFAULT_MAX_DEKAD_DIFFERENCE,
    on_heterogeneity: Literal["warn", "raise"] = "warn",
) -> Dict[str, Tuple[int, int]]:
    """Fetch representative (SOS, EOS) dekads for several seasons at once.

    All returned seasons are read from a *single* lookup point, the joint medoid
    over every requested season. Selecting each season independently could pick
    season 1 from one seasonality regime and season 2 from another, yielding a
    cropping calendar that exists nowhere in the extent.

    Seasons that hold nodata everywhere inside the extent are dropped from the
    result rather than invalidating the whole selection.
    """

    if not season_ids:
        raise ValueError("At least one season must be requested.")

    table = ensure_seasonality_lookup_table()
    columns = {sid: _require_season_columns(sid, table) for sid in season_ids}

    rows = _in_extent_lookup_rows(extent)
    if rows.empty or not _any_season_valid(rows, columns):
        if not fallback_to_nearest:
            raise ValueError(
                "No valid crop-calendar dekad values found inside extent for "
                f"seasons {list(season_ids)}."
            )
        logger.info(
            "No valid crop-calendar dekad values found inside extent; "
            "falling back to nearest lookup point."
        )
        centroid_lat, centroid_lon = _extent_centroid(extent)
        rows = _lookup_row_for_point(
            centroid_lat,
            centroid_lon,
            fallback_to_nearest=True,
            max_fallback_distance_degrees=max_fallback_distance_degrees,
        ).to_frame().T

    valid_masks = {
        sid: _valid_dekad_mask(
            rows[sos_col].to_numpy(dtype=np.int64),
            rows[eos_col].to_numpy(dtype=np.int64),
        )
        for sid, (sos_col, eos_col) in columns.items()
    }
    retained, row_mask = _jointly_valid_seasons(list(season_ids), valid_masks)
    if not retained or row_mask is None:
        raise ValueError(
            "No valid crop-calendar dekad values found for seasons "
            f"{list(season_ids)}."
        )

    dropped = [sid for sid in season_ids if sid not in retained]
    if dropped:
        logger.warning(
            f"Seasons {dropped} hold no valid crop-calendar values inside the "
            "extent and are dropped."
        )

    candidates = rows.iloc[np.flatnonzero(row_mask)]
    retained_columns = {sid: columns[sid] for sid in retained}
    _check_extent_homogeneity(
        candidates, retained_columns, max_dekad_difference, on_heterogeneity
    )

    values = np.column_stack(
        [
            candidates[column].to_numpy(dtype=np.int64)
            for sos_col, eos_col in retained_columns.values()
            for column in (sos_col, eos_col)
        ]
    )
    medoid = candidates.iloc[dekad_medoid_index(values)]
    return {
        sid: (int(medoid[sos_col]), int(medoid[eos_col]))
        for sid, (sos_col, eos_col) in retained_columns.items()
    }


def _any_season_valid(
    rows: pd.DataFrame, columns: Mapping[str, Tuple[str, str]]
) -> bool:
    """Whether at least one season has a usable value on at least one row."""

    return any(
        _valid_dekad_mask(
            rows[sos_col].to_numpy(dtype=np.int64),
            rows[eos_col].to_numpy(dtype=np.int64),
        ).any()
        for sos_col, eos_col in columns.values()
    )


def _extent_centroid(extent: BoundingBoxExtent) -> Tuple[float, float]:
    """Centroid of an extent in EPSG:4326, handling dateline-crossing bounds."""

    west, south, east, north = _extent_to_wgs84_bounds(extent)
    if west <= east:
        centroid_lon = (west + east) / 2.0
    else:
        span = ((east + 360.0) - west) % 360.0
        centroid_lon = ((west + span / 2.0 + 180.0) % 360.0) - 180.0
    return (south + north) / 2.0, centroid_lon


def get_seasons_for_extent(
    extent: BoundingBoxExtent,
    year: int,
    seasons: Sequence[str] = PROCESSING_SEASONS,
    *,
    max_dekad_difference: int = DEFAULT_MAX_DEKAD_DIFFERENCE,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
    on_heterogeneity: Literal["warn", "raise"] = "warn",
) -> Dict[str, TemporalContext]:
    """Retrieve the season windows of an extent, jointly across all seasons.

    All returned seasons originate from the same lookup point, so the resulting
    cropping calendar is one that really occurs inside the extent instead of a
    mix of seasons taken from different seasonality regimes.

    Args:
        extent (BoundingBoxExtent): extent for which to infer dates
        year (int): target year
        seasons (Sequence[str]): season identifiers to infer
        max_dekad_difference (int): maximum dekad spread inside the extent before
            the extent is reported as heterogeneous. Defaults to 7.
        on_heterogeneity ("warn" | "raise"): whether a too heterogeneous extent
            is only logged or rejected outright.

    Raises:
        ValueError: invalid season specified, no valid crop calendar found, or
            the extent is too heterogeneous while ``on_heterogeneity="raise"``.

    Returns:
        Dict[str, TemporalContext]: inferred temporal range per season.
    """

    unsupported = [season for season in seasons if season not in SUPPORTED_SEASONS]
    if unsupported:
        raise ValueError(f"Season `{unsupported[0]}` not supported!")

    dekads = fetch_cropcalendar_dekads_extent(
        seasons,
        extent,
        fallback_to_nearest=True,
        max_fallback_distance_degrees=max_fallback_distance_degrees,
        max_dekad_difference=max_dekad_difference,
        on_heterogeneity=on_heterogeneity,
    )

    return {
        season: TemporalContext(
            season_dekad_to_date(sos, target_year=year, mode="first").strftime(
                "%Y-%m-%d"
            ),
            season_dekad_to_date(eos, target_year=year, mode="last").strftime(
                "%Y-%m-%d"
            ),
        )
        for season, (sos, eos) in dekads.items()
    }


def get_season_dates_for_extent(
    extent: BoundingBoxExtent,
    year: int,
    season: str = "tc-annual",
    max_dekad_difference: int = DEFAULT_MAX_DEKAD_DIFFERENCE,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
) -> TemporalContext:
    """Function to retrieve seasonality for a specific year based on WorldCereal
    crop calendars for a given extent and season.

    Single-season convenience wrapper around `get_seasons_for_extent`. Use that
    function directly when several seasons are needed, so they are selected
    jointly instead of independently.

    More explanation on the concept of "dekads" can be found at the top of this file.

    Args:
        extent (BoundingBoxExtent): extent for which to infer dates
        year (int): target year
        season (str): season identifier for which to infer dates. Defaults to `tc-annual`
        max_dekad_difference (int): maximum difference in seasonality for all pixels
                in extent before logging a warning. Defaults to 7.

    Raises:
        ValueError: invalid season specified or no valid crop calendar found

    Returns:
        TemporalContext: inferred temporal range
    """

    contexts = get_seasons_for_extent(
        extent,
        year,
        [season],
        max_dekad_difference=max_dekad_difference,
        max_fallback_distance_degrees=max_fallback_distance_degrees,
    )
    return contexts[season]


def _row_spatial_extent_from_grid_row(row: pd.Series) -> BoundingBoxExtent:
    """Build a BoundingBoxExtent from a production-grid row.

    Supported row schemas (in this order):
    1) ``geometry_utm_wkt`` + ``epsg_utm``
    2) ``xmin``, ``ymin``, ``xmax``, ``ymax`` + (``epsg_utm`` or ``epsg``)
    3) ``geometry`` + (``epsg_utm`` or ``epsg``)
    """

    if "geometry_utm_wkt" in row and pd.notna(row.get("geometry_utm_wkt")):
        try:
            from shapely import wkt as shapely_wkt
        except ImportError as exc:  # pragma: no cover
            raise ValueError(
                "shapely is required to parse 'geometry_utm_wkt' rows."
            ) from exc
        if "epsg_utm" not in row or pd.isna(row.get("epsg_utm")):
            raise ValueError(
                "Row contains 'geometry_utm_wkt' but no valid 'epsg_utm'."
            )
        geom = shapely_wkt.loads(str(row["geometry_utm_wkt"]))
        return BoundingBoxExtent(*geom.bounds, epsg=int(row["epsg_utm"]))

    bbox_cols = {"xmin", "ymin", "xmax", "ymax"}
    if bbox_cols.issubset(set(row.index)):
        epsg = row.get("epsg_utm", row.get("epsg", None))
        if epsg is None or pd.isna(epsg):
            raise ValueError(
                "Row has xmin/ymin/xmax/ymax but no valid epsg_utm/epsg column."
            )
        return BoundingBoxExtent(
            west=float(row["xmin"]),
            south=float(row["ymin"]),
            east=float(row["xmax"]),
            north=float(row["ymax"]),
            epsg=int(epsg),
        )

    if "geometry" in row and pd.notna(row.get("geometry")):
        epsg = row.get("epsg_utm", row.get("epsg", None))
        if epsg is None or pd.isna(epsg):
            raise ValueError("Row has 'geometry' but no valid epsg_utm/epsg column.")
        geom = row["geometry"]
        return BoundingBoxExtent(*geom.bounds, epsg=int(epsg))

    raise ValueError(
        "Cannot infer spatial extent from row. Expected one of: "
        "(geometry_utm_wkt + epsg_utm), "
        "(xmin/ymin/xmax/ymax + epsg), or "
        "(geometry + epsg)."
    )


def _month_index(value: Union[str, datetime.date, pd.Timestamp]) -> int:
    """Map a date to a continuous month index (year * 12 + month - 1)."""

    ts = pd.Timestamp(value)
    return ts.year * 12 + (ts.month - 1)


def _month_start_date(month_index: int) -> datetime.date:
    """First calendar day of the month identified by `month_index`."""

    year, month = divmod(month_index, 12)
    return datetime.date(year, month + 1, 1)


def _month_end_date(month_index: int) -> datetime.date:
    """Last calendar day of the month identified by `month_index`."""

    return (pd.Timestamp(_month_start_date(month_index)) + pd.offsets.MonthEnd(1)).date()


def consolidate_processing_period(
    season_windows: Mapping[str, Sequence[str]],
    *,
    period_months: int = PROCESSING_PERIOD_MONTHS,
) -> TemporalContext:
    """Derive a processing period of exactly `period_months` from season windows.

    The period is anchored on the union of all provided season windows, so start
    and end dates can never contradict the seasons they are supposed to cover.
    Because season windows always start on the first and end on the last day of a
    month, the consolidation is done on whole months, which makes the resulting
    period exactly `period_months` long by construction.

    When the union is shorter than `period_months`, it is padded symmetrically
    (any odd month goes to the end). When it is longer, it is trimmed
    symmetrically and a warning is raised, since at least one season will then
    only be partially covered.

    Args:
        season_windows: mapping of season id to a ``(start_date, end_date)`` pair
            of ``YYYY-MM-DD`` strings.
        period_months: required length of the processing period, in months.

    Returns:
        TemporalContext: consolidated processing period.
    """

    if not season_windows:
        raise ValueError(
            "Cannot consolidate a processing period without any season window."
        )
    if period_months <= 0:
        raise ValueError("period_months must be a strictly positive number of months.")

    first_month = min(_month_index(window[0]) for window in season_windows.values())
    last_month = max(_month_index(window[1]) for window in season_windows.values())
    span = last_month - first_month + 1

    if span > period_months:
        logger.warning(
            f"Union of seasons {sorted(season_windows)} spans {span} months, which "
            f"exceeds the required processing period of {period_months} months. "
            "The period is trimmed symmetrically; seasons will be partially covered."
        )
        first_month += (span - period_months) // 2
    elif span < period_months:
        first_month -= (period_months - span) // 2
    last_month = first_month + period_months - 1

    return TemporalContext(
        _month_start_date(first_month).strftime("%Y-%m-%d"),
        _month_end_date(last_month).strftime("%Y-%m-%d"),
    )


def clip_season_windows_to_period(
    season_windows: Mapping[str, Sequence[str]],
    processing_period: TemporalContext,
) -> dict:
    """Clip season windows to `processing_period` so no season sticks out of it.

    Seasons that end up fully outside the period are dropped, since they cannot
    be produced from the data covered by the processing period.

    Args:
        season_windows: mapping of season id to a ``(start_date, end_date)`` pair
            of ``YYYY-MM-DD`` strings.
        processing_period: processing period the seasons have to fit in.

    Returns:
        dict: clipped season windows, keyed by season id.
    """

    period_first = _month_index(processing_period.start_date)
    period_last = _month_index(processing_period.end_date)

    clipped = {}
    for season_id, (start, end) in season_windows.items():
        first = max(_month_index(start), period_first)
        last = min(_month_index(end), period_last)
        if first > last:
            logger.warning(
                f"Season '{season_id}' ({start} - {end}) lies fully outside the "
                f"processing period {processing_period.start_date} - {processing_period.end_date} "
                "and is dropped."
            )
            continue
        clipped[season_id] = [
            _month_start_date(first).strftime("%Y-%m-%d"),
            _month_end_date(last).strftime("%Y-%m-%d"),
        ]
        if clipped[season_id] != [start, end]:
            logger.warning(
                f"Season '{season_id}' is clipped from ({start} - {end}) to "
                f"({clipped[season_id][0]} - {clipped[season_id][1]}) to fit the "
                "processing period."
            )

    return clipped


def enrich_production_grid_from_crop_calendars(
    grid_df: pd.DataFrame,
    year: int,
    *,
    get_seasons: bool = True,
    extent_resolver: Optional[Callable[[pd.Series], BoundingBoxExtent]] = None,
    seasons: Sequence[str] = PROCESSING_SEASONS,
    period_months: int = PROCESSING_PERIOD_MONTHS,
    max_dekad_difference: int = DEFAULT_MAX_DEKAD_DIFFERENCE,
    on_heterogeneity: Literal["warn", "raise"] = "raise",
) -> pd.DataFrame:
    """Enrich a production grid with temporal extent and optional season metadata.

    All seasons of a grid cell are resolved jointly from a single crop-calendar
    point, and ``start_date`` / ``end_date`` are then derived from their union
    via `consolidate_processing_period`. This guarantees a processing period of
    exactly ``period_months`` months that is always consistent with the seasons
    it has to cover. The season windows are subsequently clipped to that period,
    so seasons and processing period always remain mutually consistent.

    Cells whose seasonality is too heterogeneous to be represented by a single
    crop calendar are rejected by default, since silently producing them would
    yield a calendar that applies nowhere in the cell.

    This function writes:
    - ``start_date`` and ``end_date``, consolidated from the season windows.
    - optionally ``season_ids`` and ``season_windows`` (JSON string).
    """

    resolver = extent_resolver or _row_spatial_extent_from_grid_row
    result = grid_df.copy()

    for idx, row in result.iterrows():
        extent = resolver(row)

        season_contexts = get_seasons_for_extent(
            extent,
            year,
            seasons,
            max_dekad_difference=max_dekad_difference,
            on_heterogeneity=on_heterogeneity,
        )
        season_windows = {
            season: [context.start_date, context.end_date]
            for season, context in season_contexts.items()
        }

        processing_ctx = consolidate_processing_period(
            season_windows, period_months=period_months
        )
        result.loc[idx, "start_date"] = processing_ctx.start_date
        result.loc[idx, "end_date"] = processing_ctx.end_date

        if not get_seasons:
            continue

        season_windows = clip_season_windows_to_period(season_windows, processing_ctx)
        result.loc[idx, "season_ids"] = ",".join(sorted(season_windows))
        result.loc[idx, "season_windows"] = json.dumps(season_windows, sort_keys=True)

    return result


def season_dekad_to_date(
    dekad: Union[int, np.ndarray],
    target_year: Union[int, np.ndarray] = 2000,
    mode: Optional[Literal["first", "last"]] = None,
) -> Union[datetime.date, np.ndarray]:
    """Convert dekad (1-108) to date(s) in a 3-year window around target_year.

    Accepts either scalars (returns a single `datetime.date`) or numpy arrays
    (returns an array of `datetime64[D]`), so the same implementation serves
    both per-sample and vectorized/batched code paths.

    Attention: this function always returns the first day of the month
    (``mode="first"``) or the last day of the month (``mode="last"``) for the
    dekad.
    """
    if mode not in ("first", "last"):
        raise ValueError("mode must be 'first' or 'last'")

    is_scalar = np.isscalar(dekad) and np.isscalar(target_year)
    dekad_arr = np.atleast_1d(np.asarray(dekad, dtype=np.int64))
    year_arr = np.atleast_1d(np.asarray(target_year, dtype=np.int64))

    over36 = dekad_arr > 36
    year_offset = np.where(over36, (dekad_arr - 1) // 36, 0)
    year_adj = (year_arr - 1) + year_offset
    dekad_adj = np.where(over36, dekad_arr - year_offset * 36, dekad_arr)
    month = (dekad_adj - 1) // 3 + 1
    dk = (dekad_adj - 1) % 3 + 1

    if mode == "first":
        is_third = dk == 3
        is_dec = is_third & (month == 12)
        month = np.where(is_third, np.where(month == 12, 1, month + 1), month)
        year_adj = np.where(is_dec, year_adj + 1, year_adj)
        day = np.ones_like(month)
    else:  # mode == "last"
        is_first = dk == 1
        is_jan = is_first & (month == 1)
        month = np.where(is_first, np.where(month == 1, 12, month - 1), month)
        year_adj = np.where(is_jan, year_adj - 1, year_adj)
        leap = ((year_adj % 4 == 0) & (year_adj % 100 != 0)) | (year_adj % 400 == 0)
        days_in_month = np.array(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        )[month - 1]
        day = np.where((month == 2) & leap, 29, days_in_month)

    year_epoch = (year_adj - 1970).astype("datetime64[Y]")
    month_date = year_epoch.astype("datetime64[M]") + (month - 1).astype(
        "timedelta64[M]"
    )
    dates = month_date.astype("datetime64[D]") + (day - 1).astype("timedelta64[D]")

    if is_scalar:
        return dates[0].item()
    return dates