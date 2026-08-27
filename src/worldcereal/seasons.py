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
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from openeo_gfmap import BoundingBoxExtent, TemporalContext

from worldcereal import SUPPORTED_SEASONS
from worldcereal.data import cropcalendars

_SEASONALITY_LOOKUP_TABLE: Optional[pd.DataFrame] = None
DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES = 5.0


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

    if not math.isfinite(max_fallback_distance_degrees) or max_fallback_distance_degrees < 0:
        raise ValueError(
            "max_fallback_distance_degrees must be a finite non-negative value"
        )

    sos_col, eos_col = resolve_cropcalendar_columns(season_id)

    table = ensure_seasonality_lookup_table()
    if sos_col not in table.columns or eos_col not in table.columns:
        raise ValueError(
            f"Season '{season_id}' requires columns ({sos_col}, {eos_col}) "
            "but they are not present in the seasonality lookup parquet."
        )

    lat_center = _snap_coordinate_to_lookup_grid(
        lat, cropcalendars.SEASONALITY_LAT_RANGE
    )
    lon_center = _snap_coordinate_to_lookup_grid(
        lon, cropcalendars.SEASONALITY_LON_RANGE
    )

    try:
        row = table.loc[(lat_center, lon_center)]
    except KeyError as exc:
        if not fallback_to_nearest:
            raise ValueError(
                "No seasonality record found for snapped lat/lon "
                f"({lat_center}, {lon_center})."
            ) from exc

        lat_vals = table.index.get_level_values("lat").to_numpy()
        lon_vals = table.index.get_level_values("lon").to_numpy()
        if lat_vals.size == 0:
            raise ValueError(
                "No seasonality record found for snapped lat/lon "
                f"({lat_center}, {lon_center})."
            ) from exc

        distances = (lat_vals - lat_center) ** 2 + (lon_vals - lon_center) ** 2
        best_idx = int(distances.argmin())
        fallback_key = (float(lat_vals[best_idx]), float(lon_vals[best_idx]))
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
            f"using nearest cell ({fallback_key[0]}, {fallback_key[1]})."
        )
        row = table.iloc[best_idx]

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


def fetch_cropcalendar_dekad_extent(
    season_id: str,
    extent: BoundingBoxExtent,
    *,
    fallback_to_nearest: bool = True,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
) -> Tuple[int, int]:
    """Fetch median (SOS, EOS) dekad values for an extent from parquet points.

    The function selects lookup points whose lat/lon fall within the extent
    (reprojected to EPSG:4326 when needed), filters nodata values, and returns
    median SOS/EOS dekads. If no valid points are found in the extent and
    ``fallback_to_nearest`` is enabled, it falls back to the extent centroid.
    """

    sos_med, eos_med, _, _ = _fetch_cropcalendar_dekad_extent_stats(
        season_id=season_id,
        extent=extent,
        fallback_to_nearest=fallback_to_nearest,
        max_fallback_distance_degrees=max_fallback_distance_degrees,
    )
    return sos_med, eos_med


def _collect_cropcalendar_dekad_extent_values(
    season_id: str,
    extent: BoundingBoxExtent,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect valid in-extent SOS/EOS dekad arrays for a season."""

    west, south, east, north = _extent_to_wgs84_bounds(extent)
    table = ensure_seasonality_lookup_table()

    sos_col, eos_col = resolve_cropcalendar_columns(season_id)
    if sos_col not in table.columns or eos_col not in table.columns:
        raise ValueError(
            f"Season '{season_id}' requires columns ({sos_col}, {eos_col}) "
            "but they are not present in the seasonality lookup parquet."
        )

    lat_vals = table.index.get_level_values("lat").to_numpy()
    lon_vals = table.index.get_level_values("lon").to_numpy()
    mask_lat = (lat_vals >= south) & (lat_vals <= north)
    if west <= east:
        mask_lon = (lon_vals >= west) & (lon_vals <= east)
    else:
        mask_lon = (lon_vals >= west) | (lon_vals <= east)
    mask = mask_lat & mask_lon

    rows = table.iloc[np.flatnonzero(mask)]
    if rows.empty:
        logger.info(
            "No crop-calendar lookup points found inside extent.")
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    sos_arr = rows[sos_col].to_numpy(dtype=np.int64)
    eos_arr = rows[eos_col].to_numpy(dtype=np.int64)
    valid = (sos_arr > 0) & (sos_arr <= 108) & (eos_arr > 0) & (eos_arr <= 108)
    sos_valid = sos_arr[valid]
    eos_valid = eos_arr[valid]
    if sos_valid.size == 0 or eos_valid.size == 0:
        logger.warning(
            "No valid crop-calendar dekad values found inside extent for "
            f"season '{season_id}' (west={west}, south={south}, east={east}, north={north})."
        )
    return sos_valid, eos_valid


def _fetch_cropcalendar_dekad_extent_stats(
    season_id: str,
    extent: BoundingBoxExtent,
    *,
    fallback_to_nearest: bool = True,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Return median dekads and in-extent dekad arrays for a season."""

    sos_arr, eos_arr = _collect_cropcalendar_dekad_extent_values(season_id, extent)
    if sos_arr.size and eos_arr.size:
        sos_med = int(np.rint(np.median(sos_arr)))
        eos_med = int(np.rint(np.median(eos_arr)))
        return sos_med, eos_med, sos_arr, eos_arr

    if not fallback_to_nearest:
        raise ValueError(
            "No valid crop-calendar dekad values found inside extent for "
            f"season '{season_id}'."
        )

    # Fallback: sample the extent centroid using nearest-cell point lookup.
    logger.info(
        "No valid crop-calendar dekad values found inside extent; "
        "falling back to nearest point."
    )
    west, south, east, north = _extent_to_wgs84_bounds(extent)
    if west <= east:
        centroid_lon = (west + east) / 2.0
    else:
        # Dateline-crossing extent: midpoint on wrapped interval.
        span = ((east + 360.0) - west) % 360.0
        centroid_lon = ((west + span / 2.0 + 180.0) % 360.0) - 180.0
    centroid_lat = (south + north) / 2.0
    sos_med, eos_med = fetch_cropcalendar_dekad_point(
        season_id=season_id,
        lat=centroid_lat,
        lon=centroid_lon,
        fallback_to_nearest=True,
        max_fallback_distance_degrees=max_fallback_distance_degrees,
    )
    # Expose fallback-derived values in the returned arrays as well, so callers
    # don't keep receiving empty arrays after a successful fallback.
    sos_arr = np.array([sos_med], dtype=np.int64)
    eos_arr = np.array([eos_med], dtype=np.int64)
    return sos_med, eos_med, sos_arr, eos_arr


def get_season_dates_for_extent(
    extent: BoundingBoxExtent,
    year: int,
    season: str = "tc-annual",
    max_dekad_difference: int = 7,
    max_fallback_distance_degrees: float = DEFAULT_MAX_FALLBACK_DISTANCE_DEGREES,
) -> TemporalContext:

    """Function to retrieve seasonality for a specific year based on WorldCereal
        crop calendars for a given extent and season.
        
        Uses `fetch_cropcalendar_dekad_extent` for the median SOS/EOS dekads and
            converts those to dates via `season_dekad_to_date`.
        
        More explanation on the concept of "dekads" can be found at the top of this file.
    
        Args:
            extent (BoundingBoxExtent): extent for which to infer dates
            year (int): target year
            season (str): season identifier for which to infer dates. Defaults to `tc-annual`
            max_dekad_difference (int): maximum difference in seasonality for all pixels
                    in extent before raising a warning. Defaults to 7.
    
        Raises:
            ValueError: invalid season specified
            Warning: raised when seasonality difference is too large within the extent
    
        Returns:
            TemporalContext: inferred temporal range
        """

    if season not in SUPPORTED_SEASONS:
        raise ValueError(f"Season `{season}` not supported!")

    sos_dekad, eos_dekad, sos_arr, eos_arr = _fetch_cropcalendar_dekad_extent_stats(
        season_id=season,
        extent=extent,
        fallback_to_nearest=True,
        max_fallback_distance_degrees=max_fallback_distance_degrees,
    )

    if sos_arr.size and eos_arr.size:
        sos_diff = int(sos_arr.max() - sos_arr.min())
        eos_diff = int(eos_arr.max() - eos_arr.min())
        warning = False
        if sos_diff > max_dekad_difference:
            logger.warning(
                "Seasonality variability for SOS is large: "
                f"{sos_diff} dekads (> {max_dekad_difference})."
            )
            warning = True
        if eos_diff > max_dekad_difference:
            logger.warning(
                "Seasonality variability for EOS is large: "
                f"{eos_diff} dekads (> {max_dekad_difference})."
            )
            warning = True
        if warning:
            logger.warning(
                "Computation of median crop calendars may be inaccurate. "
                "Consider downsizing your area of interest for more accurate results."
            )

    start_date = season_dekad_to_date(sos_dekad, target_year=year, mode="first")
    end_date = season_dekad_to_date(eos_dekad, target_year=year, mode="last")

    return TemporalContext(
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )


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


def enrich_production_grid_from_crop_calendars(
    grid_df: pd.DataFrame,
    year: int,
    *,
    get_seasons: bool = True,
    extent_resolver: Optional[Callable[[pd.Series], BoundingBoxExtent]] = None,
) -> pd.DataFrame:
    """Enrich a production grid with temporal extent and optional season metadata.

    This function writes:
    - ``start_date`` and ``end_date`` from ``tc-annual`` crop calendars.
    - optionally ``season_ids`` and ``season_windows`` (JSON string) seasons 1 and 2.
    """

    resolver = extent_resolver or _row_spatial_extent_from_grid_row
    result = grid_df.copy()

    for idx, row in result.iterrows():
        extent = resolver(row)
        annual_ctx = get_season_dates_for_extent(extent, year, "tc-annual")
        result.loc[idx, "start_date"] = annual_ctx.start_date
        result.loc[idx, "end_date"] = annual_ctx.end_date

        if not get_seasons:
            continue

        season_windows = {}
        for season in ["tc-s1", "tc-s2"]:
            season_ctx = get_season_dates_for_extent(extent, year, season)
            season_windows[season] = [season_ctx.start_date, season_ctx.end_date]

        season_ids = sorted(season_windows.keys())
        result.loc[idx, "season_ids"] = ",".join(season_ids)
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