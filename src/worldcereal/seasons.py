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
    """Method to check the max difference in days between all DOY values
    in an array taking into account wrap-around effects due to the circular nature
    """

    doy_array = np.expand_dims(doy_array, axis=1)
    x, y = np.meshgrid(doy_array, doy_array.T)

    days_in_year = 365  # True for crop calendars

    # Step 2: Calculate the direct difference
    direct_difference = np.abs(x - y)

    # Step 3: Calculate the wrap-around difference
    wrap_around_difference = days_in_year - direct_difference

    # Step 4: Determine the minimum difference
    effective_difference = np.min(
        np.stack([direct_difference, wrap_around_difference]), axis=0
    )

    # Step 5: Determine the maximum difference for all combinations

    return effective_difference.max()


def circular_median_day_of_year(doy_array, total_days=365):
    """This function computes the median doy from a given array
    taking into account its circular nature. Still has to be used with caution!
    """
    angles = np.array([doy_to_angle(day, total_days) for day in doy_array])

    def circular_distance(a, b):
        return np.angle(np.exp(1j * (a - b)))

    median_angle = None
    min_distance = float("inf")

    for angle in angles:
        total_distance = np.sum(
            [abs(circular_distance(angle, other)) for other in angles]
        )
        if total_distance < min_distance:
            min_distance = total_distance
            median_angle = angle

    median_day = round(angle_to_doy(median_angle, total_days))
    median_day = median_day % total_days
    if median_day < 0:
        median_day += total_days

    return median_day


def doy_from_tiff(season: str, kind: str, bounds: tuple, epsg: int, resolution: int):
    """Function to read SOS/EOS DOY from TIFF

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
        np.ndarray: resulting DOY array
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
        doy_data = load_reproject(
            doy_file, bounds, epsg, resolution, nodata_value=0, fill_value=np.nan
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
    doy_date = pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=doy)

    if doy_date >= after_date:
        pass
    else:
        doy_date = pd.to_datetime(f"{year+1}-01-01") + pd.Timedelta(days=doy)

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

    # Get DOY of SOS and EOS for the target season
    sos_doy = doy_from_tiff(season, "SOS", bounds, epsg, 10000).flatten()
    eos_doy = doy_from_tiff(season, "EOS", bounds, epsg, 10000).flatten()

    # Check if we have seasonality
    if not np.isfinite(sos_doy).any():
        raise NoSeasonError(f"No valid SOS DOY found for season `{season}`")
    if not np.isfinite(eos_doy).any():
        raise NoSeasonError(f"No valid EOS DOY found for season `{season}`")

    # Only consider valid seasonality pixels
    sos_doy = sos_doy[np.isfinite(sos_doy)]
    eos_doy = eos_doy[np.isfinite(eos_doy)]

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

    # Get DOY of EOS for the target season
    eos_doy = doy_from_tiff(season, "EOS", bounds, epsg, 10000).flatten()

    # Check if we have seasonality
    if not np.isfinite(eos_doy).any():
        raise NoSeasonError(f"No valid EOS DOY found for season `{season}`")

    # Only consider valid seasonality pixels
    eos_doy = eos_doy[np.isfinite(eos_doy)]

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
