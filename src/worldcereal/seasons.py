import datetime
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from loguru import logger
import numpy as np
import pandas as pd
from satio.geoloader import load_reproject

from worldcereal import (SUPPORTED_SEASONS,
                         SEASONAL_MAPPING,
                         SEASON_PRIOR_BUFFER)
from worldcereal.resources import cropcalendars
from worldcereal.utils import aez as aezloader


class NoSeasonError(Exception):
    pass


def doy_from_tiff(season: str, kind: str,
                  bounds: tuple, epsg: str, resolution: int):
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
        raise ValueError(f'Season `{season}` not supported.')
    else:
        season = SEASONAL_MAPPING[season]

    if kind not in ['SOS', 'EOS']:
        raise ValueError(('Only `SOS` and `EOS` are valid '
                          f'values of `kind`. Got: `{kind}`'))

    doy_file = season + f'_{kind}_WGS84.tif'

    if not pkg_resources.is_resource(cropcalendars, doy_file):
        raise RuntimeError(('Required season DOY file '
                            f'`{doy_file}` not found.'))

    logger.info(f'Loading DOY data from: {doy_file}')

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
    doy_date = pd.to_datetime(
        f'{year}-01-01') + pd.Timedelta(days=doy)

    if doy_date >= after_date:
        pass
    else:
        doy_date = pd.to_datetime(
            f'{year+1}-01-01') + pd.Timedelta(days=doy)

    doy_date = doy_date.strftime('%Y-%m-%d')
    # logger.info(f'Inferred date from DOY: {doy_date}')

    return doy_date


def season_doys_to_dates(sos: int, eos: int,
                         sample_date: str,
                         allow_outside: bool = False):
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
        '''
        Straightforward case in which season
        is entirely in one calendar year
        '''
        ref_year = sample_date.year

    else:
        '''
        Nasty case where we cross a calendar year.
        There's three options. We take the season
        with the center closest to the sample_date
        parameter
        '''

        # Correct DOY for the year crossing
        eos += 365

        base_year = sample_date.year
        timediff = 365
        ref_year = base_year

        for year in [base_year - 1, base_year, base_year + 1]:

            start = (datetime.datetime(year, 1, 1) + pd.Timedelta(
                days=sos - 1))
            end = (datetime.datetime(year, 1, 1) + pd.Timedelta(
                days=eos - 1))

            seasonmid = start + (end - start) / 2

            if abs(seasonmid - sample_date).days < timediff:
                timediff = abs(seasonmid - sample_date).days
                ref_year = year

    # We found our true ref year, now get the SOS/EOS dates

    start_date = (datetime.datetime(ref_year, 1, 1)
                  + pd.Timedelta(days=sos - 1))

    end_date = (datetime.datetime(ref_year, 1, 1)
                + pd.Timedelta(days=eos - 1))

    if not allow_outside:

        if sample_date < start_date:
            raise ValueError((f'Sample date ({sample_date}) '
                              ' is before season start_date '
                              f'({start_date})!'))
        if sample_date > end_date:
            raise ValueError((f'Sample date ({sample_date}) '
                              ' is after season end_date '
                              f'({end_date})!'))

    return (start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'))


def season_doys_to_dates_refyear(sos: int, eos: int,
                                 ref_year: int):
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
    end_date = (datetime.datetime(ref_year, 1, 1)
                + pd.Timedelta(days=eos))

    if sos < eos:
        '''
        Straightforward case where entire season
        is in calendar year
        '''
        seasonduration = eos - sos + 1

    else:
        '''
        Nasty case where we cross a calendar year.
        '''

        # Correct DOY for the year crossing
        eos += 365
        seasonduration = eos - sos + 1

    # Now we can compute start date from end date and
    # season duration
    start_date = end_date - pd.Timedelta(days=seasonduration)

    return (start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'))


def _get_cropland_processing_dates(aez_id, year):
    # Load AEZ
    aez = aezloader.load().to_crs(epsg=4326)

    m1eos = aez[aez['zoneID'] == int(aez_id)]['m1eos_max'].values[0]
    m2eos = aez[aez['zoneID'] == int(aez_id)]['m2eos_max'].values[0]
    wweos = aez[aez['zoneID'] == int(aez_id)]['wweos_max'].values[0]

    end_doy = np.maximum(m1eos, wweos)
    if m2eos is not None and not np.isnan(m2eos) and m2eos != 0:
        end_doy = np.maximum(end_doy, m2eos)

    end_date = pd.to_datetime(f'{year}-01-01') + pd.Timedelta(
        days=end_doy)
    start_date = end_date - pd.Timedelta(days=365)

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    logger.info(f'Derived start_date: {start_date}')
    logger.info(f'Derived end_date: {end_date}')

    return start_date, end_date


def _get_croptype_processing_dates(season, aez_id, year):
    # Load AEZ
    aez = aezloader.load().to_crs(epsg=4326)

    sos = aez[aez['zoneID'] == int(aez_id)][
        f'{SEASONAL_MAPPING[season].lower()}sos_min'].values[0]
    eos = aez[aez['zoneID'] == int(aez_id)][
        f'{SEASONAL_MAPPING[season].lower()}eos_max'].values[0]

    for doy in [sos, eos]:
        if doy is None or np.isnan(doy):
            raise NoSeasonError(('Could not get valid SOS/EOS '
                                 f'value for season `{season}`!'))

    start_date, end_date = season_doys_to_dates_refyear(sos, eos, year)

    # Now need to add a season-specific buffer before SOS
    prior_buffer = SEASON_PRIOR_BUFFER[season]
    if prior_buffer > 0:
        start_date = (pd.to_datetime(start_date) -
                      pd.Timedelta(
            days=prior_buffer)).strftime('%Y-%m-%d')

    logger.info(f'Derived start_date: {start_date}')
    logger.info(f'Derived end_date: {end_date}')

    return start_date, end_date


def get_processing_dates(season, aez_id, year):
    """Function to retrieve required temporal range
    of input products.

    Args:
        season (str): season identifier for which to infer dates
        aez_id (int): ID of the AEZ for which to infer dates
        year (int): year in which the end of season needs to be

    Raises:
        ValueError: invalid season specified

    Returns:
        (start_date, end_date): tuple of date strings specifying
        start and end date to process
    """

    if season not in SUPPORTED_SEASONS:
        raise ValueError(f'Season `{season}` not supported!')

    elif season == 'tc-annual':
        return _get_cropland_processing_dates(aez_id, year)
    else:
        return _get_croptype_processing_dates(season, aez_id, year)


def infer_season_dates(season, bounds, epsg,
                       sample_date, resolution=10000,
                       allow_outside=False):

    # Extract SOS doy
    sos_doy = doy_from_tiff(season, 'SOS', bounds, epsg, resolution)

    if sos_doy.size != 1:
        raise ValueError(('Extracted SOS DOY data has size '
                          f'`{sos_doy.size}` while only size '
                          '1 is supported'))
    elif int(sos_doy) == 0:
        raise NoSeasonError(('Extracted SOS DOY is 0, meaning '
                             'no seasonality for this crop is '
                             'available in this region'))
    else:
        sos_doy = int(sos_doy)

    # Extract EOS day
    eos_doy = doy_from_tiff(season, 'EOS', bounds, epsg, resolution)

    if eos_doy.size != 1:
        raise ValueError(('Extracted EOS DOY data has size '
                          f'`{eos_doy.size}` while only size '
                          '1 is supported'))
    elif int(eos_doy) == 0:
        raise NoSeasonError(('Extracted EOS DOY is 0, meaning '
                             'no seasonality for this crop is '
                             'available in this region'))
    else:
        eos_doy = int(eos_doy)

    return season_doys_to_dates(sos_doy, eos_doy, sample_date,
                                allow_outside=allow_outside)
