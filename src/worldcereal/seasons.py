import datetime
import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
from loguru import logger

from worldcereal import SEASONAL_MAPPING, SUPPORTED_SEASONS, BoundingBoxExtent
from worldcereal.data import cropcalendars

# from worldcereal.utils import aez as aezloader
from worldcereal.utils.geoloader import load_reproject


class NoSeasonError(Exception):
    pass


def doy_from_tiff(season: str, kind: str, bounds: tuple, epsg: str, resolution: int):
    """Function to read SOS/EOS DOY from TIFF

    Args:
        season (str): crop season to process
        kind (str): which DOY to read (SOS/EOS)
        bounds (tuple): the bounds to read
        epsg (str): epsg in which bounds are expressed
        resolution (int): resolution in meters of resulting array

    Raises:
        RuntimeError: when required TIFF file was not found
        ValueError: when requested season is not supported
        ValueError: when `kind` value is not supported

    Returns:
        np.ndarray: resulting DOY array
    """

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
        raise RuntimeError(("Required season DOY file " f"`{doy_file}` not found."))

    logger.info(f"Loading DOY data from: {doy_file}")

    with pkg_resources.open_binary(cropcalendars, doy_file) as doy_file:
        doy_data = load_reproject(doy_file, bounds, epsg, resolution)

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
        ref_year = sample_date.year

    else:
        """
        Nasty case where we cross a calendar year.
        There's three options. We take the season
        with the center closest to the sample_date
        parameter
        """

        # Correct DOY for the year crossing
        eos += 365

        base_year = sample_date.year
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


def get_processing_dates_for_extent(extent: BoundingBoxExtent, season: str, year: int):
    """Function to retrieve required temporal range of input products for a
    given extent, season and year. Based on the requested season's end date
    a temporal range is inferred that spans an entire year.

    Args:
        extent (BoundingBoxExtent): extent for which to infer dates
        season (str): season identifier for which to infer dates
        year (int): year in which the end of season needs to be

    Raises:
        ValueError: invalid season specified

    Returns:
        (start_date, end_date): tuple of date strings specifying
        start and end date to process
    """

    if season not in SUPPORTED_SEASONS:
        raise ValueError(f"Season `{season}` not supported!")

    bounds = (extent.east, extent.south, extent.west, extent.north)
    epsg = extent.epsg

    # Get DOY of EOS for the target season
    eos_doy = doy_from_tiff(season, "EOS", bounds, epsg, 10000)

    # Get the median EOS
    eos_doy_median = np.median(eos_doy)

    # We can derive the end date from year and EOS
    end_date = datetime.datetime(year, 1, 1) + pd.Timedelta(days=eos_doy_median)

    # And get start date by subtracting a year
    start_date = end_date - pd.Timedelta(days=364)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
