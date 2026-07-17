import datetime
import importlib.resources as pkg_resources
import math
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from openeo_gfmap import BoundingBoxExtent, TemporalContext

from worldcereal import SEASONAL_MAPPING, SUPPORTED_SEASONS
from worldcereal.data import cropcalendars

# from worldcereal.utils import aez as aezloader
from worldcereal.utils.geoloader import load_reproject


class NoSeasonError(Exception):
    pass


class SeasonMaxDiffError(Exception):
    pass


_SEASONALITY_LOOKUP_TABLE: Optional[pd.DataFrame] = None


def _ensure_seasonality_lookup_table() -> pd.DataFrame:
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


def _resolve_cropcalendar_columns(
    season_id: str, parameter: Literal["doy", "dekad"]
) -> Tuple[str, str]:
    """Resolve season identifier and parameter to SOS/EOS parquet columns."""

    if parameter not in {"doy", "dekad"}:
        raise ValueError(
            f"parameter must be one of ('doy', 'dekad'), got '{parameter}'"
        )

    try:
        sos_doy_col, eos_doy_col = cropcalendars.SEASONALITY_COLUMN_MAP[season_id]
    except KeyError as exc:
        raise ValueError(
            f"Season '{season_id}' is not available in the seasonality lookup. "
            f"Known seasons: {sorted(cropcalendars.SEASONALITY_COLUMN_MAP)}"
        ) from exc

    return (
        sos_doy_col.replace("_doy", f"_{parameter}"),
        eos_doy_col.replace("_doy", f"_{parameter}"),
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


def _fetch_cropcalendar_point(
    season_id: str,
    lat: float,
    lon: float,
    parameter: Literal["doy", "dekad"] = "doy",
    *,
    fallback_to_nearest: bool = True,
) -> Tuple[int, int]:
    """Fetch (SOS, EOS) values for one point from the global parquet lookup.

    The input point is snapped to the lookup's 0.5 deg grid centers before querying.
    If the snapped cell is missing and ``fallback_to_nearest`` is enabled, the
    nearest available lookup cell is used.

    Parameters
    ----------
    season_id : str
        Season identifier (e.g. ``tc-s1``, ``tc-s2``, ``tc-annual``).
    lat, lon : float
        Input point coordinates.
    parameter : {"doy", "dekad"}, default "doy"
        Which crop-calendar representation to fetch.
    fallback_to_nearest : bool, default True
        Whether to use the nearest available lookup cell when the snapped cell
        is not present in the table.
    """

    sos_col, eos_col = _resolve_cropcalendar_columns(season_id, parameter)

    table = _ensure_seasonality_lookup_table()
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
        logger.error(
            f"Seasonality lookup missing ({lat_center}, {lon_center}); "
            f"using nearest cell ({fallback_key[0]}, {fallback_key[1]})."
        )
        row = table.iloc[best_idx]

    sos_value = int(row[sos_col])
    eos_value = int(row[eos_col])
    if sos_value <= 0 or eos_value <= 0:
        raise ValueError(
            "Seasonality lookup returned nodata values for "
            f"season '{season_id}' and parameter '{parameter}'."
        )

    return sos_value, eos_value


def fetch_cropcalendar_dekad_extent(
    season_id: str,
    extent: BoundingBoxExtent,
    *,
    fallback_to_nearest: bool = True,
) -> Tuple[int, int]:
    """Fetch median (SOS, EOS) dekad values for an extent from parquet points.

    The function selects lookup points whose lat/lon fall within the extent
    (reprojected to EPSG:4326 when needed), filters nodata values, and returns
    median SOS/EOS dekads. If no valid points are found in the extent and
    ``fallback_to_nearest`` is enabled, it falls back to the extent centroid.
    """

    west, south, east, north = _extent_to_wgs84_bounds(extent)
    table = _ensure_seasonality_lookup_table()

    sos_col, eos_col = _resolve_cropcalendar_columns(season_id, "dekad")
    if sos_col not in table.columns or eos_col not in table.columns:
        raise ValueError(
            f"Season '{season_id}' requires columns ({sos_col}, {eos_col}) "
            "but they are not present in the seasonality lookup parquet."
        )

    lat_vals = table.index.get_level_values("lat").to_numpy()
    lon_vals = table.index.get_level_values("lon").to_numpy()
    mask_lat = (lat_vals >= south) & (lat_vals <= north)
    # Support extents crossing the antimeridian (west > east).
    if west <= east:
        mask_lon = (lon_vals >= west) & (lon_vals <= east)
    else:
        mask_lon = (lon_vals >= west) | (lon_vals <= east)
    mask = mask_lat & mask_lon

    rows = table.iloc[np.flatnonzero(mask)]
    if not rows.empty:
        sos_arr = rows[sos_col].to_numpy(dtype=np.int64)
        eos_arr = rows[eos_col].to_numpy(dtype=np.int64)
        valid = (sos_arr > 0) & (sos_arr <= 108) & (eos_arr > 0) & (eos_arr <= 108)
        sos_arr = sos_arr[valid]
        eos_arr = eos_arr[valid]
        if sos_arr.size and eos_arr.size:
            sos_med = int(np.rint(np.median(sos_arr)))
            eos_med = int(np.rint(np.median(eos_arr)))
            return sos_med, eos_med

    if not fallback_to_nearest:
        raise ValueError(
            "No valid crop-calendar dekad values found inside extent for "
            f"season '{season_id}'."
        )

    # Fallback: sample the extent centroid using nearest-cell point lookup.
    if west <= east:
        centroid_lon = (west + east) / 2.0
    else:
        # Dateline-crossing extent: midpoint on wrapped interval.
        span = ((east + 360.0) - west) % 360.0
        centroid_lon = ((west + span / 2.0 + 180.0) % 360.0) - 180.0
    centroid_lat = (south + north) / 2.0
    return fetch_cropcalendar_dekad_point(
        season_id=season_id,
        lat=centroid_lat,
        lon=centroid_lon,
        fallback_to_nearest=True,
    )


def fetch_cropcalendar_doy_point(
    season_id: str,
    lat: float,
    lon: float,
    *,
    fallback_to_nearest: bool = True,
) -> Tuple[int, int]:
    """Fetch (SOS, EOS) DOY values for one point from the global parquet lookup."""

    sos_value, eos_value = _fetch_cropcalendar_point(
        season_id=season_id,
        lat=lat,
        lon=lon,
        parameter="doy",
        fallback_to_nearest=fallback_to_nearest,
    )

    if (sos_value > 366 or eos_value > 366):
        raise ValueError(
            "Seasonality lookup returned invalid DOY values for "
            f"season '{season_id}': SOS={sos_value}, EOS={eos_value}. "
            "Valid DOY range is 1-366."
        )
    
    return sos_value, eos_value


def fetch_cropcalendar_dekad_point(
    season_id: str,
    lat: float,
    lon: float,
    *,
    fallback_to_nearest: bool = True,
) -> Tuple[int, int]:
    """Fetch (SOS, EOS) dekad values for one point from the global parquet lookup."""

    sos_value, eos_value = _fetch_cropcalendar_point(
        season_id=season_id,
        lat=lat,
        lon=lon,
        parameter="dekad",
        fallback_to_nearest=fallback_to_nearest,
    )

    if sos_value > 108 or eos_value > 108:
        raise ValueError(
            "Seasonality lookup returned invalid dekad values for "
            f"season '{season_id}': SOS={sos_value}, EOS={eos_value}. "
            "Valid dekad range is 1-108."
        )

    return sos_value, eos_value


def doy_to_angle(day_of_year, total_days=365):
    return 2 * math.pi * (day_of_year / total_days)


def angle_to_doy(angle, total_days=365):
    return (angle / (2 * math.pi)) * total_days


def max_doy_difference(doy_array):
    """
    Method to check the max difference in days between all DOY values
    in an array taking into account wrap-around effects due to the circular nature.
    Optimized for integer DOY arrays.
    """
    # Ensure we're working with integers for efficiency
    doy_array = np.asarray(doy_array, dtype=np.int32)

    # For small arrays, use the full pairwise approach
    if len(doy_array) <= 1000:
        doy_array = np.expand_dims(doy_array, axis=1)
        x, y = np.meshgrid(doy_array, doy_array.T)

        days_in_year = 365  # True for crop calendars

        # Calculate the direct difference
        direct_difference = np.abs(x - y)

        # Calculate the wrap-around difference
        wrap_around_difference = days_in_year - direct_difference

        # Determine the minimum difference
        effective_difference = np.minimum(direct_difference, wrap_around_difference)

        return int(effective_difference.max())

    else:
        # For large arrays, use a more efficient approach
        # Find min and max, then check if the gap is larger going the other way
        min_doy = int(np.min(doy_array))
        max_doy = int(np.max(doy_array))

        # Direct span
        direct_span = max_doy - min_doy

        # Wrap-around span (going the other direction around the circle)
        wrap_span = (365 - max_doy) + min_doy

        # The maximum difference is the smaller of the two spans
        return int(min(direct_span, wrap_span))


def circular_median_day_of_year(doy_array, total_days=365):
    """Compute the circular median DOY (exact) using a weighted histogram approach.

    The circular median is the day-of-year (DOY) that minimizes the sum of
    circular distances to all observations, where circular distance between two
    DOYs d1 and d2 is ``min(|d1-d2|, total_days - |d1-d2|)``.

    Implementation details:
    * Filters invalid entries (<=0 or > ``total_days``).
    * Collapses duplicates via a histogram (frequency weighting) so runtime depends
      on the number of distinct DOYs (k ≤ 365) instead of raw sample size (n).
    * Uses a fully vectorized k×k distance matrix over unique DOYs to obtain the
      exact weighted 1-median solution on the circle.

    Contract:
    * Returns an ``int`` DOY in [1, total_days] when at least one valid value is present.
    * Raises ``ValueError`` if, after filtering, there are no valid DOY values.

    Parameters
    ----------
    doy_array : array-like
        Input day-of-year values (integers expected). May include zeros or other invalid
        values that will be filtered out.
    total_days : int, default 365
        Length of the circular period (e.g. 365 for non-leap-year crop calendars).

    Returns
    -------
    int
        Circular median DOY.

    Raises
    ------
    ValueError
        If no valid DOY values remain after filtering.
    """
    vals = np.asarray(doy_array, dtype=np.int32)
    vals = vals[(vals > 0) & (vals <= total_days)]
    if vals.size == 0:
        raise ValueError(
            "circular_median_day_of_year: no valid DOY values (after filtering) to compute median."
        )
    if vals.size == 1:
        return int(vals[0])

    # Histogram counts for each DOY (1..total_days). Index 0 unused.
    counts = np.bincount(vals, minlength=total_days + 1)[1:]
    nonzero = np.nonzero(counts)[0] + 1  # actual DOYs present
    weights = counts[nonzero - 1].astype(np.int64)

    if nonzero.size == 1:
        return int(nonzero[0])

    # Vectorized pairwise circular distances between candidate DOYs and observed DOYs.
    cand = nonzero.reshape(-1, 1)
    other = nonzero.reshape(1, -1)
    direct = np.abs(cand - other)
    circ = np.minimum(direct, total_days - direct)
    # Weighted sum of distances for each candidate median.
    total_dist = (circ * weights).sum(axis=1)
    median_idx = int(np.argmin(total_dist))
    return int(nonzero[median_idx])


def doy_from_tiff(season: str, kind: str, bounds: tuple, epsg: int, resolution: int):
    """Function to read SOS/EOS DOY from TIFF

    Optimized to return integer arrays for maximum efficiency. Missing/nodata
    values are represented as 0 - filter these out with array[array > 0].

    Args:
        season (str): crop season to process
        kind (str): which DOY to read (SOS/EOS)
        bounds (tuple): the bounds to read
        epsg (int): epsg in which bounds are expressed
        resolution (int): resolution in meters of resulting array

    Raises:
        FileNotFoundError: when required TIFF file was not found
        ValueError: when requested season is not supported
        ValueError: when `kind` value is not supported

    Returns:
        np.ndarray: resulting DOY array as uint16 integers (1-365), with 0 for nodata
    """

    if epsg == 4326:
        raise ValueError(
            "EPSG 4326 not supported for DOY data. Use a projected CRS instead."
        )

    if season not in SUPPORTED_SEASONS:
        raise ValueError(f"Season `{season}` not supported.")
    else:
        season = SEASONAL_MAPPING[season]

    if kind not in ["SOS", "EOS"]:
        raise ValueError(
            ("Only `SOS` and `EOS` are valid " f"values of `kind`. Got: `{kind}`")
        )

    doy_file = season + f"_{kind}_WGS84.tif"

    if not pkg_resources.is_resource(cropcalendars, doy_file):
        raise FileNotFoundError(
            ("Required season DOY file " f"`{doy_file}` not found.")
        )

    with pkg_resources.open_binary(cropcalendars, doy_file) as doy_file:  # type: ignore
        # Use integer-optimized loading for DOY data (1-365 values)
        # Keep as integers throughout - much more memory efficient
        doy_data = load_reproject(
            doy_file,
            bounds,
            epsg,
            resolution,
            nodata_value=0,
            fill_value=0,
            dtype=np.uint16,
        )

    return doy_data


def season_doys_to_dates_refyear(sos: int, eos: int, ref_year: int):
    """Funtion to transform SOS and EOS from DOY
    to exact dates, making use of the reference year
    in which EOS should be located.

    Args:
        sos (int): DOY from SOS
        eos (int): DOY from EOS
        ref_year (int): ref year to match

    Returns:
        tuple: (start_date, end_date)
    """

    # We can derive the end date from ref_year and EOS
    end_date = datetime.datetime(ref_year, 1, 1) + pd.Timedelta(days=eos)

    if sos < eos:
        """
        Straightforward case where entire season
        is in calendar year
        """
        seasonduration = eos - sos

    else:
        """
        Nasty case where we cross a calendar year.
        """

        # Correct DOY for the year crossing
        eos += 365
        seasonduration = eos - sos

    # Now we can compute start date from end date and
    # season duration
    start_date = end_date - pd.Timedelta(days=seasonduration)

    return start_date, end_date


def get_season_dates_for_extent(
    extent: BoundingBoxExtent,
    year: int,
    season: str = "tc-annual",
    max_seasonality_difference: int = 60,
) -> TemporalContext:
    """Function to retrieve seasonality for a specific year based on WorldCereal
    crop calendars for a given extent and season.

    Args:
        extent (BoundingBoxExtent): extent for which to infer dates
        year (int): year in which the end of season needs to be
        season (str): season identifier for which to infer dates. Defaults to `tc-annual`
        max_seasonality_difference (int): maximum difference in seasonality for all pixels
                in extent before raising an exception. Defaults to 60.

    Raises:
        ValueError: invalid season specified
        SeasonMaxDiffError: raised when seasonality difference is too large

    Returns:
        TemporalContext: inferred temporal range
    """

    if season not in SUPPORTED_SEASONS:
        raise ValueError(f"Season `{season}` not supported!")

    bounds = (extent.west, extent.south, extent.east, extent.north)
    epsg = extent.epsg

    # Get DOY of SOS and EOS for the target season (now returns integers directly)
    sos_doy = doy_from_tiff(season, "SOS", bounds, epsg, 10000).flatten()
    eos_doy = doy_from_tiff(season, "EOS", bounds, epsg, 10000).flatten()

    # Check if we have seasonality - filter out nodata values (0)
    if not (sos_doy > 0).any():
        logger.error("No start of season information available for this location!")
        raise NoSeasonError(f"No valid SOS DOY found for season `{season}`")
    if not (eos_doy > 0).any():
        logger.error("No end of season information available for this location!")
        raise NoSeasonError(f"No valid EOS DOY found for season `{season}`")

    # Only consider valid seasonality pixels (DOY > 0)
    # Already integers, much more efficient!
    sos_doy = sos_doy[sos_doy > 0].astype(np.int32)
    eos_doy = eos_doy[eos_doy > 0].astype(np.int32)

    # Check max seasonality difference
    seasonality_difference_sos = max_doy_difference(sos_doy)
    seasonality_difference_eos = max_doy_difference(eos_doy)
    warning = False
    if seasonality_difference_sos > max_seasonality_difference:
        logger.warning(
            f"Seasonality difference for SOS is large: {seasonality_difference_sos} days"
        )
        warning = True
    if seasonality_difference_eos > max_seasonality_difference:
        logger.warning(
            f"Seasonality difference for EOS is large: {seasonality_difference_eos} days"
        )
        warning = True

    if warning:
        logger.warning(
            "Computation of median crop calendars will be inaccurate. Consider downsizing your area of interest for more accurate results."
        )

    # Compute median DOY
    sos_doy_median = circular_median_day_of_year(sos_doy)
    eos_doy_median = circular_median_day_of_year(eos_doy)

    # Get the seasonality dates
    start_date, end_date = season_doys_to_dates_refyear(
        sos_doy_median, eos_doy_median, year
    )

    return TemporalContext(
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )
