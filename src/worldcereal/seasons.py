import datetime
import importlib.resources as pkg_resources
import math

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


def doy_to_date_after(doy: int, after_date: datetime.datetime):
    """Function to convert a DOY to an actual date which
    needs to be after another date.

    Args:
        doy (int): input DOY
        after_date (datetime.datetime): date that needs to preceed DOY

    Returns:
        str: inferred date (yyyy-mm-dd)
    """

    year = after_date.year
    # DOY is 1-based so day 1 should correspond to January 1st
    doy_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=doy - 1)

    if doy_date >= after_date:
        pass
    else:
        doy_date = pd.to_datetime(f"{year+1}-01-01") + pd.Timedelta(days=doy - 1)

    doy_date = doy_date.strftime("%Y-%m-%d")
    # logger.info(f'Inferred date from DOY: {doy_date}')

    return doy_date


def season_doys_to_dates(
    sos: int, eos: int, sample_date: str, allow_outside: bool = False
):
    """Funtion to transform SOS and EOS from DOY
    to exact dates, making use of a sample_date
    to match the season to.
    NOTE: if sample_date is not within the resulting
    season and `allow_outside` is False, an exception
    is thrown.

    Args:
        sos (int): DOY from SOS
        eos (int): DOY from EOS
        sample_date (str): date to match (yyyy-mm-dd)
        allow_outside (bool): if True, do not fail when
            sample date is outside season

    Returns:
        tuple: (start_date, end_date), matched season
    """

    sample_date = pd.to_datetime(sample_date)

    if eos > sos:
        """
        Straightforward case in which season
        is entirely in one calendar year
        """
        ref_year = sample_date.year  # type: ignore

    else:
        """
        Nasty case where we cross a calendar year.
        There's three options. We take the season
        with the center closest to the sample_date
        parameter
        """

        # Correct DOY for the year crossing
        eos += 365

        base_year = sample_date.year  # type: ignore
        timediff = 365
        ref_year = base_year

        for year in [base_year - 1, base_year, base_year + 1]:
            start = datetime.datetime(year, 1, 1) + pd.Timedelta(days=sos - 1)
            end = datetime.datetime(year, 1, 1) + pd.Timedelta(days=eos - 1)

            seasonmid = start + (end - start) / 2

            if abs(seasonmid - sample_date).days < timediff:
                timediff = abs(seasonmid - sample_date).days
                ref_year = year

    # We found our true ref year, now get the SOS/EOS dates

    start_date = datetime.datetime(ref_year, 1, 1) + pd.Timedelta(days=sos - 1)

    end_date = datetime.datetime(ref_year, 1, 1) + pd.Timedelta(days=eos - 1)

    if not allow_outside:
        if sample_date < start_date:
            raise ValueError(
                (
                    f"Sample date ({sample_date}) "
                    " is before season start_date "
                    f"({start_date})!"
                )
            )
        if sample_date > end_date:
            raise ValueError(
                (
                    f"Sample date ({sample_date}) "
                    " is after season end_date "
                    f"({end_date})!"
                )
            )

    return (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))


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


def get_processing_dates_for_extent(
    extent: BoundingBoxExtent,
    year: int,
    season: str = "tc-annual",
    max_seasonality_difference: int = 60,
) -> TemporalContext:
    """Function to retrieve required temporal range of input products for a
    given extent, season and year. Based on the requested season's end date
    a temporal range is inferred that spans an entire year.

    Args:
        extent (BoundingBoxExtent): extent for which to infer dates
        year (int): year in which the end of season needs to be
        season (str): season identifier for which to infer dates. Defaults to tc-annual
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

    # Get DOY of EOS for the target season (now returns integers directly)
    eos_doy = doy_from_tiff(season, "EOS", bounds, epsg, 10000).flatten()

    # Check if we have seasonality - filter out nodata values (0)
    if not (eos_doy > 0).any():
        raise NoSeasonError(f"No valid EOS DOY found for season `{season}`")

    # Only consider valid seasonality pixels (DOY > 0)
    # Already integers, much more efficient!
    eos_doy = eos_doy[eos_doy > 0].astype(np.int32)

    # Check max seasonality difference
    seasonality_difference = max_doy_difference(eos_doy)
    if seasonality_difference > max_seasonality_difference:
        raise SeasonMaxDiffError(
            f"Seasonality difference too large: {seasonality_difference} days"
        )

    # Compute median DOY
    eos_doy_median = circular_median_day_of_year(eos_doy)

    # We can derive the end date from year and EOS
    end_date = datetime.datetime(year, 1, 1) + pd.Timedelta(days=eos_doy_median)

    # And get start date by subtracting a year
    start_date = end_date - pd.Timedelta(days=364)

    print(f"Derived the following period for processing: {start_date} " f"- {end_date}")

    return TemporalContext(
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )
