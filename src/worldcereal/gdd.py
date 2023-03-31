from loguru import logger
import numpy as np
import pandas as pd
from satio.timeseries import Timeseries
import xarray as xr

from worldcereal.seasons import (doy_from_tiff,
                                 doy_to_date_after)
from worldcereal.utils import SkipBlockError, BlockTooColdError
from worldcereal.utils import aez
from worldcereal import SEASONAL_MAPPING

GDD_ACCUM_THRESHOLD = 300


def computeGDD(tcube: xr.DataArray,
               tbase: int = 0,
               upper_limit: int = None,
               units: str = "C"):
    """
    Compute GDD value for each timestep in the tcube

    INPUTS
    - tcube - A xarray.dataarray.dataArray or numpy.ndarray object
              with X, Y and Timestamp as coordinates
    - tbase - Integer. Base temperature to the GDD computation
    - upper_limit - Max temperature to not accumulate temperature
    - units - Temperature units
    """

    if units == "C":
        tcube = tcube - 273.15
    elif units == "K":
        pass
    else:
        raise RuntimeError(("`units` should be one of "
                            f"['C', 'K'] but got {units}."))

    if "ndarray" in str(type(tcube)):
        gdd = tcube.copy()
        if upper_limit:
            gdd[gdd >= upper_limit] = upper_limit
        gdd = gdd - tbase
        gdd[gdd < 0] = 0

    else:
        gdd = tcube.copy()
        gdd.values[gdd.values <= tbase] = 0
        if upper_limit:
            gdd.values[gdd.values >= upper_limit] = upper_limit
        gdd.values = gdd.values - tbase
        gdd.values[gdd.values < 0] = 0

    return gdd


def winterwheatGDD(band):
    out = computeGDD(band, tbase=0)
    return out


def maizeGDD(band):
    out = computeGDD(band, tbase=10)
    return out


class GDDcomputer:

    def __init__(self,
                 agera5coll,
                 tbase: int,
                 bounds: tuple = None,
                 epsg: int = None,
                 upper_limit: int = None,
                 units: str = "C",
                 start_date: str = None,
                 end_date: str = None,
                 aez_id: int = None):

        self.coll = agera5coll
        self.tbase = tbase
        self.bounds = bounds
        self.epsg = epsg
        self.upper_limit = upper_limit
        self.units = units
        self.start_date = start_date
        self.end_date = end_date
        self.aez_id = aez_id

    def load_temperature(self):
        logger.info('Loading temperature_mean time series.')
        ts = self.coll.load_timeseries('temperature_mean',
                                       resolution=10000)

        if self.start_date is not None:
            idx = np.where(ts.timestamps >= pd.to_datetime(
                self.start_date).to_pydatetime())[0]
            ts.data = ts.data[:, idx, ...]
            ts.timestamps = ts.timestamps[idx]

        if self.end_date is not None:
            idx = np.where(ts.timestamps <= pd.to_datetime(
                self.end_date).to_pydatetime())[0]
            ts.data = ts.data[:, idx, ...]
            ts.timestamps = ts.timestamps[idx]

        return ts

    def check_within_aez(self):

        if self.aez_id is None:
            logger.warning(('Cannot check AEZ intersection '
                            'because no AEZ ID was specified.'))
            return True

        from shapely.geometry import box
        aez_df = aez.load().to_crs(epsg=self.epsg)
        aez_id = aez_df.set_index('zoneID').loc[self.aez_id]

        if aez_id.geometry.contains(box(*self.bounds)):
            return True
        else:
            return False

    def get_closest_aez_sos(self, season, block_sos_doy):
        '''Method to return the AEZ SOS that is closest
        to the pixel-based SOS out of the available
        min, avg, max SOS.
        '''
        aez_df = aez.load().to_crs(epsg=self.epsg)
        aez_id = aez_df.set_index('zoneID').loc[self.aez_id]

        sos_min = aez_id[f'{SEASONAL_MAPPING[season].lower()}sos_min']
        sos_avg = aez_id[f'{SEASONAL_MAPPING[season].lower()}sos_avg']
        sos_max = aez_id[f'{SEASONAL_MAPPING[season].lower()}sos_max']

        all_sos = np.array([sos_min, sos_avg, sos_max])
        sos_diff = np.abs(all_sos - block_sos_doy)

        sos = all_sos[np.argmin(sos_diff)]

        return sos

    def get_aez_sos(self, season, which='min'):
        '''Method to return a specific AEZ SOS [min, avg, max]
        '''

        if self.aez_id is None:
            raise ValueError('`aez_id` not specified while we need it.')

        aez_df = aez.load().to_crs(epsg=self.epsg)
        aez_id = aez_df.set_index('zoneID').loc[self.aez_id]

        sos_min = aez_id[f'{SEASONAL_MAPPING[season].lower()}sos_{which}']

        return int(sos_min)

    def get_sos_date(self, season, after_date):
        '''Method to retrieve the actual SOS date from
        which to start accumulating GDD
        '''

        if type(after_date) == str:
            after_date = pd.to_datetime(after_date)

        # Infer pixel-based SOS
        sos_doy = doy_from_tiff(season,
                                'SOS',
                                self.bounds,
                                self.epsg,
                                10000)
        if sos_doy.size != 1:
            raise ValueError(('Extracted SOS DOY data has size '
                              f'`{sos_doy.size}` while only size '
                              '1 is supported'))

        if int(sos_doy) == 0:
            logger.warning(('Extracted SOS DOY is 0. We take '
                            'average AEZ-based SOS for GDD computation'))
            sos_doy = self.get_aez_sos(season, which='avg')
            sos_date = doy_to_date_after(sos_doy, after_date)

        elif not self.check_within_aez():
            '''This typically means the block was assigned to AEZ with
            largest tile intersection while block itself is actually
            (partly) outside the AEZ. This can lead to problems with the
            pixel-based crop calendars being outside the input date range
            as determined by the AEZ stats. As a workaround we take the
            closest AEZ crop calendar in this case.
            '''
            logger.warning(('Block is not (entirely) within AEZ '
                            'so we take AEZ-based SOS for '
                            'GDD computation'))

            # Infer closest AEZ-based SOS
            sos_doy = self.get_closest_aez_sos(season, int(sos_doy))
            sos_date = doy_to_date_after(sos_doy, after_date)

        else:
            # In principle we can use the pixel-based calendar. However
            # it is possible that the pixel-based SOS is before the
            # earliest SOS in AEZ and then we won't have enough ARD.
            # We use SOSmin of AEZ in this case
            sos_doy = int(sos_doy)
            sos_date = doy_to_date_after(sos_doy, after_date)

            if self.aez_id is not None:
                # Get the closest AEZ SOS
                aez_sos_min = self.get_aez_sos(season, which='min')
                aez_sos_date = doy_to_date_after(aez_sos_min, after_date)

                if pd.to_datetime(sos_date) < pd.to_datetime(aez_sos_date):
                    logger.warning((f'Pixel-based SOS ({sos_date}) is before '
                                    f'AEZ SOSmin ({aez_sos_date}). '
                                    'Taking AEZ version!'))
                    sos_date = aez_sos_date

        return sos_date

    def compute_accumulated_gdd(self, season: str = None):
        '''
        Main method to compute accumulated GDD.
        If the optional `season` argument is provided
        then accumulation will be done from the respective SOS onward.
        Otherwise accumulation is done over the entire time series.
        '''
        logger.info('Start computation accumulated GDD.')

        # Load temperature time series
        temperature_ts = self.load_temperature()

        # if all Nan's -> block is situated completely over the ocean
        if np.all(np.isnan(temperature_ts.data)):
            raise SkipBlockError(('Accumulated GDD resulted in all NaNs, '
                                  'block probably not located over land!'))

        # If a season is specified we need to determine
        # the start date for accumulation
        if season is not None:

            if self.bounds is None:
                raise ValueError(('`bounds` cannot be None when '
                                  'requesting a specific season'))
            if self.epsg is None:
                raise ValueError(('`epsg` cannot be None when '
                                  'requesting a specific season'))

            # Get SOS date to start GDD accumulation
            sos_date = self.get_sos_date(season=season,
                                         after_date=temperature_ts.timestamps[0])

            logger.info(f'Accumulating GDD from: {sos_date}')

            # Finally, put all temperature before the sos_date
            # low enough so they won't contribute to accumulation
            idx = np.where(temperature_ts.timestamps < pd.to_datetime(
                sos_date).to_pydatetime())[0]

            temperature_ts.data[:, idx, ...] = -100

        # Compute accumulated GDD
        gdd = computeGDD(temperature_ts.data, tbase=self.tbase,
                         upper_limit=self.upper_limit, units=self.units)
        accumulated_gdd = gdd.cumsum(axis=1)

        if accumulated_gdd.max() < GDD_ACCUM_THRESHOLD:
            raise BlockTooColdError(
                (f'Accumulated GDD reaches only {accumulated_gdd.max()} deg '
                 f'which is less than the min required {GDD_ACCUM_THRESHOLD} '
                 'deg to develop a crop. Cannot proceed '
                 'with GDD accumulation!'))

        accumulated_gdd_ts = Timeseries(
            data=accumulated_gdd,
            timestamps=temperature_ts.timestamps,
            bands=['accumulated_gdd'],
            attrs=dict(
                tbase=self.tbase
            )
        )
        logger.info('Computation accumulated GDD finished.')
        return accumulated_gdd_ts
