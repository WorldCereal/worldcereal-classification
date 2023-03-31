import copy
from typing import Dict

from loguru import logger
import numpy as np
import pandas as pd
import satio
from satio.features import Features, BaseFeaturesProcessor
from satio.features import multitemporal_speckle
from satio.utils import TaskTimer
from satio.utils.resample import downsample_n
from skimage.transform import resize
import xarray as xr


from worldcereal.utils.masking import (scl_mask, pixel_qa_mask,
                                       binary_mask,
                                       SCL_MASK_VALUES,
                                       flaglocalminima)
from worldcereal.utils.resize import AgERA5_resize
from worldcereal.utils import SkipBlockError
from worldcereal.features.feat_dem import elev_from_dem
from worldcereal.features.feat_irr import (surface_soil_moisture,
                                           theoretical_boundaries,
                                           soil_moisture, calc_str_1)

L2A_BANDS_10M = ['B02', 'B03', 'B04', 'B08']
L2A_BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL',
                 'sunAzimuthAngles', 'sunZenithAngles',
                 'viewAzimuthMean', 'viewZenithMean']
L2A_BANDS_DICT = {10: L2A_BANDS_10M,
                  20: L2A_BANDS_20M}
L8_THERMAL_BANDS_DICT = {10: ['ST-B10', 'QA-PIXEL', 'PIXEL-QA']}
SIGMA0_BANDS_DICT = {20: ['VV', 'VH']}
AGERA5_BANDS_DICT = {
    100: [
        'dewpoint_temperature', 'precipitation_flux',
        'solar_radiation_flux', 'temperature_max',
        'temperature_mean', 'temperature_min',
        'vapour_pressure', 'wind_speed'
    ]
}


def apply_gdd_normalization(band_ts, settings):
    """Function that swaps time-coordinate of time series
    to accumulated GDD, normalizing for phenological stages.

    WARNING: once this GDD normalization is done, the timestamps
    are replaced by GDD fake timestamps. Use with caution!

    Args:
        band_ts (satio.Timeseries): input time series to be normalized
        settings (dict): The settings containing GDD normalization specs
    """

    dtype = band_ts.data.dtype
    orig_start = band_ts.timestamps[0]
    orig_end = band_ts.timestamps[-1]

    if 'normalize_gdd' not in settings.keys():
        raise RuntimeError(('GDD normzalization was invoked '
                            'while the required settings '
                            'were not specified'))

    # Make a deepcopy of the settings
    settings = copy.deepcopy(settings)

    # Get the precomputed accumulated GDD
    accumulated_gdd = settings['normalize_gdd']['accumulated_gdd']

    # Check dimensionality
    if accumulated_gdd.data.shape[2] != 1:
        raise NotImplementedError(
            ('Currently it is required that accumulated GDD '
             'timeseries is one-dimensional in x and y while the actual '
             f'dimension is: {accumulated_gdd.data.shape[2]}'))

    # Get the desired gdd bin resolution. Defaults to 100 degrees
    gdd_bins = settings['normalize_gdd'].get('gdd_bins', 100)

    # Resample the daily accumulated GDD to the available timestamps
    accumulated_gdd_resampled = accumulated_gdd.select_timestamps(
        band_ts.timestamps
    )

    # Setup temporary DataArray to do the required manipulations
    band_ts_da = xr.DataArray(data=band_ts.data,
                              coords={'t': band_ts.timestamps},
                              dims=['band', 't', 'x', 'y']).astype(np.float32)

    # Swap the time coordinate for accumulated GDD
    band_ts_da['t'] = accumulated_gdd_resampled.data.ravel()

    # Create a fake datetime list based on GDD
    # so timeseries functionality keeps working
    gdd_time = np.array([(pd.to_datetime(band_ts.timestamps[0])
                          + pd.Timedelta(days=x)).to_pydatetime()
                         for x in band_ts_da['t'].values])

    # # Update the data into the time series object
    band_ts.data = band_ts_da.values.astype(dtype)
    band_ts.timestamps = gdd_time

    # Now we want to bin the GDDs, for which we use
    # the satio compositing functionality. We take
    # over the existing compositing settings where needed
    gdd_composite_settings = settings.get('composite', {})
    gdd_composite_settings['start'] = band_ts.timestamps[0].strftime(
        '%Y-%m-%d')
    gdd_composite_settings['end'] = band_ts.timestamps[-1].strftime(
        '%Y-%m-%d')
    gdd_composite_settings['freq'] = gdd_bins
    gdd_composite_settings['window'] = gdd_bins

    band_ts = band_ts.composite(**gdd_composite_settings)
    band_ts.attrs['accumulated_gdd'] = [
        (x - band_ts.timestamps[0]).days for x in band_ts.timestamps]

    # Finally, because we might have a ridiculous time series span
    # due to the steps above, we reindex the GDD time series to match
    # the original span.
    # pylint: disable=no-member
    final_timestamps = pd.date_range(
        orig_start,
        orig_end,
        periods=len(band_ts.timestamps)).to_pydatetime()
    band_ts.timestamps = final_timestamps

    return band_ts


class _FeaturesTimer():

    def __init__(self, *resolutions):

        self.load = {}
        self.rsi = {}
        self.composite = {}
        self.interpolate = {}
        self.features = {}
        self.text_features = {}
        self.seasons = {}
        self.speckle = {}

        for r in resolutions:
            self.load[r] = TaskTimer(
                f'{r}m loading', unit='seconds')
            self.rsi[r] = TaskTimer(
                f'{r}m rsi calculation', unit='seconds')
            self.composite[r] = TaskTimer(
                f'{r}m compositing', unit='seconds')
            self.interpolate[r] = TaskTimer(
                f'{r}m interpolation', unit='seconds')
            self.features[r] = TaskTimer(
                f'{r}m features computation', unit='seconds')
            self.text_features[r] = TaskTimer(
                f'{r}m texture features', unit='seconds')
            self.seasons[r] = TaskTimer(
                f'{r}m season detection', unit='seconds')
            self.speckle[r] = TaskTimer(
                f'{r}m speckle filtering', unit='seconds')


class WorldCerealFeaturesProcessor(BaseFeaturesProcessor):
    '''WorldCereal has its own default FeaturesProcessor
    although at the moment it is exactly the same as the
    satio one.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class L2AFeaturesProcessor(WorldCerealFeaturesProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = _FeaturesTimer(10, 20)

    @ property
    def _reflectance(self):
        return True

    @ property
    def supported_bands(self):
        return L2A_BANDS_DICT

    @ property
    def supported_rsis(self):
        if self._supported_rsis is None:
            rsis_dict = {}
            rsi_res = {r: self._rsi_meta[r]['native_res']
                       for r in self._rsi_meta.keys()}
            rsis_dict[10] = [v for v, r in rsi_res.items() if r == 10]
            rsis_dict[20] = [v for v, r in rsi_res.items() if r == 20]
            self._supported_rsis = rsis_dict

        return self._supported_rsis

    def preprocess_data(self,
                        timeseries: 'satio.timeseries.Timeseries',
                        resolution: int,
                        mask: np.ndarray = None,
                        composite: bool = True,
                        interpolate=None,
                        settings_override: Dict = None):
        """
        Pre-processing of loaded timeseries object. Includes masking,
        compositing and interpolation.
        """
        settings = settings_override or self.settings
        newtimeseries = None

        for band in timeseries.bands:
            band_ts = timeseries.select_bands([band])
            if mask is not None:
                # mask 'nan' values. for uint16 we use 0 as nodata value
                # mask values that are False are marked as invalid
                if isinstance(mask, dict):
                    mask = mask[resolution]

                band_ts = band_ts.mask(mask)

            # drop all no data frames
            band_ts = band_ts.drop_nodata()

            composite_settings = settings.get('composite')
            if (composite_settings is not None) & composite:
                self.timer.composite[resolution].start()
                logger.info(f"{band}: compositing")
                band_ts = band_ts.composite(**composite_settings)
                self.timer.composite[resolution].stop()

            # If required, we do here the GDD normalization
            if 'normalize_gdd' in settings.keys():
                logger.info(f'{band}: applying GDD normalization')
                band_ts = apply_gdd_normalization(band_ts, settings)

            if interpolate is None:
                interpolate = settings.get('interpolate', True)
            if interpolate:
                self.timer.interpolate[resolution].start()
                logger.info(f'{band}: interpolating')
                band_ts = band_ts.interpolate()
                self.timer.interpolate[resolution].stop()

            if newtimeseries is None:
                newtimeseries = band_ts
            else:
                newtimeseries = newtimeseries.merge(band_ts)

        return newtimeseries

    def load_data(self,
                  resolution,
                  timeseries=None,
                  no_data=0,
                  dtype=None):
        """
        Load Timeseries from the collection and merge with `timeseries` if
        given.
        `dtype` allows optional explicit casting of the loaded data
        """
        collection = self.collection

        # check which bands are needed for the rsis at this resolution
        rsis = self.rsis[resolution]
        rsi_bands = {'all': []}
        for rsi in rsis:
            rsi_bands['all'].extend(self._rsi_meta[rsi]['bands'])
        # remove duplicates
        rsi_bands['all'] = list(dict.fromkeys(rsi_bands['all']))
        rsi_bands['10'] = [b for b in rsi_bands['all']
                           if b in self.supported_bands[10]]
        rsi_bands['20'] = [b for b in rsi_bands['all']
                           if b in self.supported_bands[20]]
        loaded_bands = timeseries.bands if timeseries is not None else []

        # if resolution == 20 -> check if you need 10 m bands for certain rsi's
        # because these need to be loaded first and downsampled
        if resolution == 20:
            # add bands required for SMAD feature
            smad = self.settings.get('smad', None)
            if smad is not None:
                for b in smad['bands_10']:
                    if b not in rsi_bands['10']:
                        rsi_bands['10'].append(b)
                for b in smad['bands_20']:
                    if b not in rsi_bands['20']:
                        rsi_bands['20'].append(b)
            bands_to_load = [b for b in rsi_bands['10']
                             if b not in loaded_bands]
            if(len(bands_to_load) > 0):
                self.timer.load[10].start()
                logger.info(f'Loading bands: {bands_to_load}')
                bands_ts = collection.load_timeseries(*bands_to_load)
                bands_ts.data[bands_ts.data == no_data] = 0
                if dtype is not None:
                    bands_ts.data = bands_ts.data.astype(dtype)
                bands_ts = bands_ts.downsample()
                if timeseries is None:
                    timeseries = bands_ts
                else:
                    timeseries = timeseries.merge(bands_ts)
                self.timer.load[10].stop()

        # now the required bands...
        bands = self.bands[resolution].copy()
        # add those for rsi...
        bands.extend([b for b in rsi_bands['{}'.format(resolution)]
                      if b not in bands])
        loaded_bands = timeseries.bands if timeseries is not None else []
        bands_to_load = [b for b in bands if b not in loaded_bands]

        if(len(bands_to_load) > 0):
            self.timer.load[resolution].start()
            logger.info(f'Loading bands: {bands_to_load}')
            bands_ts = collection.load_timeseries(*bands_to_load)
            bands_ts.data[bands_ts.data == no_data] = 0
            if dtype is not None:
                bands_ts.data = bands_ts.data.astype(dtype)
            if timeseries is None:
                timeseries = bands_ts
            else:
                timeseries = timeseries.merge(bands_ts)
            self.timer.load[resolution].stop()
        else:
            logger.info("Did not find bands to "
                        f"load for resolution: {resolution}")

        return timeseries

    def load_mask(self):

        logger.info(f"L2A collection products: {self.collection.df.shape[0]}")

        self.timer.load[20].start()
        logger.info(f'SCL: loading')
        scl_ts = self.collection.load_timeseries('SCL')
        self.timer.load[20].stop()

        scl_ts = scl_ts.upsample()

        logger.info(f"SCL: preparing mask")
        mask, obs, valid_before, valid_after = scl_mask(
            scl_ts.data, **self.settings['mask'])

        mask_20 = downsample_n(mask.astype(np.uint16), 1) == 1

        mask_dict = {10: mask,
                     20: mask_20}

        return mask_dict, obs, valid_before, valid_after

    def load_multitemporal_mask(self, prior_mask=None):
        '''
        Method to flag undetected clouds/shadows using multitemporal
        gap search approach.
        Cfr. work of Dominique Haesen (VITO)
        '''

        logger.info(('Performing multitemporal '
                     'cloud/shadow filtering ...'))

        # Load the raw bands needed to compute NDVI
        logger.info(f"Loading bands: ['B04', 'B08']")
        ts = self.collection.load_timeseries('B04', 'B08')

        # Make sure data is in uint16
        ts.data = ts.data.astype(np.uint16)

        # If a prior mask is provided, apply it to the timeseries
        if prior_mask is not None:
            if isinstance(prior_mask, dict):
                prior_mask = prior_mask[10]
            ts_masked = ts.mask(prior_mask, drop_nodata=False)
        else:
            ts_masked = ts

        # Convert to float32 and replace missing values (0)
        # with NaN
        ts_masked.data = ts_masked.data.astype(np.float32)
        ts_masked.data[ts_masked.data == 0] = np.nan

        # Compute NDVI
        ndvi = ((ts_masked['B08'].data - ts_masked['B04'].data) /
                (ts_masked['B08'].data + ts_masked['B04'].data))[0, ...]

        # Make a DataArray for easy daily resampling
        ndvi_da = xr.DataArray(data=ndvi,
                               coords={'time': ts_masked.timestamps},
                               dims=['time', 'x', 'y'])

        # Resample to daily, missing data will be NaN
        daily_daterange = pd.date_range(
            ts_masked.timestamps[0],
            ts_masked.timestamps[-1] + pd.Timedelta(days=1),
            freq='D').floor('D')
        ndvi_daily = ndvi_da.reindex(time=daily_daterange,
                                     method='bfill', tolerance='1D')

        # Run multitemporal dip detection
        # Need to do it in slices, to avoid memory issues
        step = 256
        for idx in np.r_[:ndvi_daily.values.shape[1]:step]:
            for idy in np.r_[:ndvi_daily.values.shape[2]:step]:
                logger.debug((f"idx: {idx} - {idx+step} "
                              f"| idy: {idy} - {idy+step}"))
                ndvi_daily.values[
                    :, idx:idx+step, idy:idy+step] = flaglocalminima(
                    ndvi_daily.values[:, idx:idx+step, idy:idy+step],
                    maxdip=0.01,
                    maxdif=0.1,
                    maxgap=60,
                    maxpasses=5)

        # Subset on the original timestamps
        ndvi_cleaned = ndvi_daily.sel(time=ts.timestamps,
                                      method='ffill',
                                      tolerance='1D')

        # Extract the mask: True is invalid, False is valid
        mask = np.isnan(ndvi_cleaned.values)

        # Apply erosion/dilation to reduce speckle effect
        # DISABLED BECAUSE IF REMOVES WAY TOO MUCH !!!
        # erode_r = self.settings['mask'].get('erode_r', None)
        # dilate_r = self.settings['mask'].get('dilate_r', None)
        # erode_r = 3
        # dilate_r = 11
        # mask = erode_mask(mask, erode_r=erode_r)
        # mask = dilate_mask(mask, dilate_r=dilate_r)

        # Invert the mask
        mask = ~mask

        # Create the 20m mask as well
        mask_20 = downsample_n(mask.astype(np.uint16), 1) == 1

        mask_dict = {10: mask,
                     20: mask_20}

        # Return the enhanced mask and the original raw
        # timeseries of B04 and B08 so we don't need
        # to load them again if needed
        return mask_dict, ts

    def compute_features(self,
                         chunck_size=None,
                         upsample_order=1,
                         augment=False,
                         **kwargs):

        timer = self.timer
        settings = self.settings
        features_meta = self._features_meta

        # check if texture features need to be computed
        # and store features_meta of those separately
        if 'texture' in features_meta.keys():
            text_feat = True
            text_feat_meta = features_meta['texture']
            del features_meta['texture']
        else:
            text_feat = False

        # if no other features remain at this point -> abort!
        if not bool(features_meta):
            raise ValueError('At least one other feature required'
                             'other than texture. Aborting...')

        # check if pheno features need to be computed
        # and store features_meta of those in separate variable
        seas_feat = False
        pheno_feat_meta = {}
        pheno_keys = ['pheno_mult_season', 'pheno_single_season']
        for p_key in pheno_keys:
            if p_key in features_meta.keys():
                seas_feat = True
                pheno_feat_meta.update({p_key: features_meta[p_key]})
                del features_meta[p_key]

        # check whether all bands and rsis are represented
        # in features_meta. If not, drop the ones not needed!
        check = False
        for key, value in features_meta.items():
            if value.get("bands", None) is None:
                check = True
        if not check:
            feat_bands = []
            for key, value in features_meta.items():
                feat_bands.extend(value.get("bands", []))
            to_remove_b = [b for b in settings['bands']
                           if b not in feat_bands]
            for b in to_remove_b:
                settings['bands'].remove(b)
            to_remove_r = [r for r in settings['rsis']
                           if r not in feat_bands]
            for r in to_remove_r:
                settings['rsis'].remove(r)

        # if seasons need to be detected:
        # make sure the RSI required for this is included in the rsi list
        # to be computed!
        seasonSettings = settings.get('seasons', {})
        if bool(seasonSettings) or (seas_feat):
            rsi_seas = seasonSettings.get('rsi', 'evi')
            if rsi_seas not in settings['rsis']:
                settings['rsis'].append(rsi_seas)
            # determine at which resolution the season detection
            # needs to be done
            if rsi_seas in self.supported_rsis[10]:
                resolution_seas = 10
            else:
                resolution_seas = 20
        else:
            seasons = None
            resolution_seas = 0

        # get mask
        mask, obs, valid_before, valid_after = self.load_mask()

        # Add additional clouds for more robust classifiers
        if augment:
            mask = self.augment(mask)

        #  ---------------------------------
        #  get advanced multitemporal mask
        #  ---------------------------------

        if bool(self.settings['mask'].get('multitemporal', False)):
            mask, ts = self.load_multitemporal_mask(prior_mask=mask)

            # Need to adjust valid_after for additionally masked obs
            valid_after = (mask[10].sum(axis=0) / obs * 100).astype(int)

        else:
            ts = None

        #  ---------------------------
        #  Handle 10m data
        #  ---------------------------
        resolution = 10
        ts = self.load_data(resolution, timeseries=ts)

        # feature computuation at 10 m
        if ts is not None:
            features_10m, ts_proc = self.compute_features_10m(ts,
                                                              chunck_size,
                                                              mask)

            # if season detection needs to be done on 10m -> do it now!
            if resolution_seas == 10:
                seasons = self.get_seasons(ts,
                                           mask, seasonSettings,
                                           resolution_seas)

                # compute pheno features
                phen_feat_10m = self.compute_phen_features(seasons,
                                                           pheno_feat_meta,
                                                           resolution)

                # merge with other 10m features
                if features_10m is not None:
                    if phen_feat_10m is not None:
                        features_10m = features_10m.merge(phen_feat_10m)
                else:
                    features_10m = phen_feat_10m
        else:
            features_10m = None

        #  ---------------------------
        #  Handle 20m data
        #  ---------------------------
        resolution = 20
        rsis = self.rsis[resolution]
        if (len(self.bands[resolution]) > 0) or (len(rsis) > 0):
            features_20m, ts = self.compute_features_20m(ts, ts_proc,
                                                         chunck_size,
                                                         mask)

            # if season detection needs to be done on 20m -> do it now!
            if resolution_seas == 20:
                seasons = self.get_seasons(ts,
                                           mask, seasonSettings,
                                           resolution_seas)

                # compute pheno features
                phen_feat_20m = self.compute_phen_features(seasons,
                                                           pheno_feat_meta,
                                                           resolution)

                # merge with other 20m features
                if features_20m is not None:
                    if phen_feat_20m is not None:
                        features_20m = features_20m.merge(phen_feat_20m)
                else:
                    features_20m = phen_feat_20m

            # upsample 20m features to 10m and merge them
            if features_10m is None:
                features_10m = features_20m.upsample(order=upsample_order)
            elif features_20m is not None:
                features_10m = features_10m.merge(
                    features_20m.upsample(order=upsample_order))

        #  ---------------------------
        #  Texture
        #  ---------------------------

        # optionally compute texture features based on
        # computed features
        if text_feat:
            logger.info('Computing texture features')
            timer.text_features[10].start()
            inputFeat = features_10m.select(text_feat_meta['features'])
            params = text_feat_meta['parameters']
            # if desired, run PCA first
            if 'pca' in text_feat_meta.keys():
                inputFeat = inputFeat.pca(text_feat_meta['pca'],
                                          scaling_range=params.get(
                    'scaling_range', None))

            text_features = inputFeat.texture(
                win=params.get('win', 2),
                d=params.get('d', [1]),
                theta=params.get('theta', [0, np.pi/4]),
                levels=params.get('levels', 256),
                metrics=params.get('metrics', ('contrast',)),
                avg=params.get('avg', True),
                scaling_range=params.get('scaling_range', {}))

            features_10m = features_10m.merge(text_features)

            timer.text_features[10].stop()

        #  ---------------------------
        #  Meta
        #  ---------------------------

        # add meta features
        features_10m = features_10m.merge(
            Features(np.array([obs, valid_after]),
                     names=['l2a_obs',
                            'l2a_obs_percentvalid']))
        for r in [10, 20]:
            timer.load[r].log()
            timer.rsi[r].log()
            timer.composite[r].log()
            timer.interpolate[r].log()
            timer.features[r].log()
            timer.text_features[r].log()
            timer.seasons[r].log()

        return features_10m

    def compute_features_10m(self, ts, chunk_size, mask):
        resolution = 10
        timer = self.timer
        features_meta = self._features_meta
        rsi_meta = self._rsi_meta

        # first pre-process the timeseries
        ts_proc = self.preprocess_data(ts, resolution, mask=mask)

        # now, we compute RSI's
        rsis = self.rsis[resolution]
        if len(rsis) > 0:
            logger.info(f"{resolution}m: computing rsis")
            timer.rsi[resolution].start()
            ts_rsi = ts_proc.compute_rsis(*rsis, rsi_meta=rsi_meta)
            timer.rsi[resolution].stop()

            #########################################
            # Perform smoothing on NDVI if requested
            smooth_ndvi = bool(self.settings.get('smooth_ndvi', False))

            if 'ndvi' in ts_rsi.bands and smooth_ndvi:
                logger.info('NDVI smoothing requested!')

                # compute required rsi ...
                ndvi = ts.compute_rsis('ndvi', rsi_meta=rsi_meta,
                                       interpolate_inf=False)
                ndvi = self.preprocess_data(
                    ndvi, resolution, mask=mask,
                    interpolate=False)

                # RSI 0 means no data
                ndvi.data[ndvi.data == 0] = np.nan

                # Run whittaker smoother
                smoothed_ndvi = self.smooth_whittaker(ndvi)

                # Overwrite in original satio.Timeseries
                ndvi.data = np.expand_dims(smoothed_ndvi.values, axis=0)

                # Replace in ts_rsi
                ts_rsi = ts_rsi.select_bands([band for band in ts_rsi.bands
                                              if 'ndvi' not in band])
                ts_rsi = ts_rsi.merge(ndvi)

            #########################

        # start feature calculation
        timer.features[resolution].start()

        # compute 10m band features and scale to reflectance
        features_10m = None
        bands = self.bands[resolution]
        if len(bands) > 0:
            logger.info(f"{resolution}m: computing bands features")
            features_10m = ts_proc.select_bands(
                bands).features_from_dict(
                    resolution,
                    features_meta=features_meta,
                    chunk_size=chunk_size)

            # because bands features are calculated from uint16 bands
            # scale them to reflectance values
            # (fft features should not be scaled though)
            features_10m.data /= 10000

        # compute RSI features
        logger.info(f"{resolution}m: computing rsi features")
        if len(rsis) > 0:
            if features_10m is not None:
                features_10m = features_10m.merge(
                    ts_rsi.features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size))
            else:
                features_10m = ts_rsi.features_from_dict(
                    resolution,
                    features_meta=features_meta,
                    chunk_size=chunk_size
                )

        timer.features[resolution].stop()

        return features_10m, ts_proc

    def compute_features_20m(self, ts, ts_proc,
                             chunk_size, mask):

        resolution = 20
        timer = self.timer
        features_meta = self._features_meta
        rsi_meta = self._rsi_meta

        # load raw data
        if ts is not None:
            # downsample 10 m bands to 20m
            ts = ts.downsample()
            ts_proc = ts_proc.downsample()
            # load raw 20m bands
            ts = self.load_data(resolution, timeseries=ts)
        else:
            ts = self.load_data(resolution)

        # now pre-process the timeseries
        # (only those which were not pre-processed at 10 m resolution)
        bands = self.bands[resolution].copy()
        # add those bands required to calculate the requested RSI's
        rsis = self.rsis[resolution]
        for rsi in rsis:
            rsi_bands = rsi_meta[rsi]['bands']
            bands.extend([b for b in rsi_bands if b not in bands])
        # remove 10m bands (if any)
        if ts_proc is not None:
            bands = [b for b in bands if b not in ts_proc.bands]
        # start pre-processing
        if len(bands) > 0:
            if ts_proc is not None:
                ts_proc = ts_proc.merge(self.preprocess_data(
                    ts.select_bands(bands),
                    resolution, mask=mask))
            else:
                ts_proc = self.preprocess_data(
                    ts.select_bands(bands),
                    resolution, mask=mask)

        # compute rsis
        if len(rsis) > 0:
            logger.info(f"{resolution}m: computing rsis")
            timer.rsi[resolution].start()
            ts_rsi = ts_proc.compute_rsis(*rsis, rsi_meta=rsi_meta)
            timer.rsi[resolution].stop()

        # start feature computation
        timer.features[resolution].start()

        # compute features of rsis first
        # (before dropping 10m bands)
        if len(rsis) > 0:
            logger.info(f"{resolution}m: computing rsis features")
            features_20m = ts_rsi.features_from_dict(
                resolution,
                features_meta=features_meta,
                chunk_size=chunk_size)
        else:
            features_20m = None

        # compute SMAD feature if requested
        if self.settings.get('smad', None) is not None:
            logger.info('Computing SMAD feature')
            smadsettings = self.settings['smad']
            smadbands = smadsettings['bands_10'].copy()
            smadbands.extend(smadsettings['bands_20'].copy())
            tssmad = [ts_proc.select_bands([b]).data[0] / 10000
                      for b in smadbands]
            # compute the actual features
            if features_20m is not None:
                features_20m = features_20m.merge(
                    smadsettings['function'](*tssmad))
            else:
                features_20m = smadsettings['function'](*tssmad)

        # compute 20m bands features
        if len(self.bands[resolution]) > 0:
            logger.info(f"{resolution}m: computing bands features")
            # reduce timeseries to 20m bands only
            ts_proc = ts_proc.select_bands(self.bands[resolution])
            features_20m_bands = ts_proc.features_from_dict(
                resolution,
                features_meta=features_meta,
                chunk_size=chunk_size)

            # scale bands features  (fft features should not be scaled though)
            features_20m_bands.data /= 10000

            # merge with other features
            if features_20m is None:
                features_20m = features_20m_bands
            else:
                features_20m = features_20m.merge(features_20m_bands)

        timer.features[resolution].stop()

        return features_20m, ts

    def get_seasons(self, ts, mask, seasonSettings, resolution):

        rsi_meta = self._rsi_meta
        rsi_seas = seasonSettings.get('rsi', 'evi')
        timer = self.timer

        logger.info(f"{resolution}m: detecting growing seasons")
        timer.seasons[resolution].start()

        '''
        Season detection needs specific settings.
        NO gdd normalization and daily compositing so smoothing
        and season detection happens on original time series.
        '''
        season_override_settings = {
            'composite': self.settings['composite']
        }

        # get parameters
        max_seasons = seasonSettings.get('max_seasons', 5)
        amp_thr1 = seasonSettings.get('amp_thr1', 0.1)
        amp_thr2 = seasonSettings.get('amp_thr2', 0.35)
        min_window = seasonSettings.get('min_window', 10)
        max_window = seasonSettings.get('max_window', 185)
        partial_start = seasonSettings.get('partial_start', False)
        partial_end = seasonSettings.get('partial_end', False)
        smooth_rsi = seasonSettings.get('smooth_rsi', False)

        # compute required rsi ...
        rsi = ts.compute_rsis(rsi_seas, rsi_meta=rsi_meta,
                              interpolate_inf=False)

        # ATTENTION: if whittaker smoothing is disabled
        # interpolate needs to be True in the preprocessing
        # of the RSI!!
        rse_interpolate = False if smooth_rsi else True

        rsi = self.preprocess_data(
            rsi, resolution, mask=mask,
            interpolate=rse_interpolate,
            settings_override=season_override_settings)

        # ###################################################
        # Whittaker smoothing

        if smooth_rsi:
            # RSI 0 means no data
            rsi.data[rsi.data == 0] = np.nan

            # Run whittaker smoother
            smoothed_ts = self.smooth_whittaker(rsi)

            # Overwrite in original satio.Timeseries
            rsi.data = np.expand_dims(smoothed_ts.values, axis=0)

        # ####################################################

        # run season detection
        seasons = rsi.detect_seasons(max_seasons=max_seasons,
                                     amp_thr1=amp_thr1,
                                     amp_thr2=amp_thr2,
                                     min_window=min_window,
                                     max_window=max_window,
                                     partial_start=partial_start,
                                     partial_end=partial_end)

        timer.seasons[resolution].stop()

        return seasons

    def smooth_whittaker(self, rsi):
        '''Method to smooth a RSI using Whittaker smoother
        '''

        minvalue = -0.2 if rsi.bands[0] == 'ndvi' else 0

        # SETTINGS
        lmbda = 100  # Very strong smoothing for season detection
        passes = 3
        dokeepmaxima = True

        # Smooth the RSI before detecting seasons
        logger.info(('Performing Whittaker smoothing ...'))

        # Make a DataArray for easy daily resampling
        rsi_da = xr.DataArray(data=rsi.data[0, ...],
                              coords={'time': rsi.timestamps},
                              dims=['time', 'x', 'y'])

        # Resample to daily
        daily_daterange = pd.date_range(
            rsi.timestamps[0],
            rsi.timestamps[-1] + pd.Timedelta(days=1),
            freq='D').floor('D')
        rsi_daily = rsi_da.reindex(time=daily_daterange,
                                   method='bfill', tolerance='1D')

        # Run whittaker smoother
        # Need to do it in slices, to avoid memory issues
        from worldcereal.utils.masking import whittaker
        step = 256
        for idx in np.r_[:rsi_daily.values.shape[1]:step]:
            for idy in np.r_[:rsi_daily.values.shape[2]:step]:
                logger.debug((f"idx: {idx} - {idx+step} "
                              f"| idy: {idy} - {idy+step}"))
                rsi_daily.values[
                    :, idx:idx+step,
                    idy:idy+step] = whittaker(
                        lmbda=lmbda,
                        npdatacube=rsi_daily.values[
                            :, idx:idx+step, idy:idy+step],
                    minimumdatavalue=minvalue,
                    maximumdatavalue=1,
                    passes=passes,
                    dokeepmaxima=dokeepmaxima)

        # Subset on the original timestamps
        rsi_origts = rsi_daily.sel(time=rsi.timestamps,
                                   method='ffill',
                                   tolerance='1D')

        return rsi_origts

    def compute_phen_features(self, seasons, pheno_feat_meta, resolution):

        # TODO: for now, all pheno features are only computed on the RSI
        # from which the seasons were derived. This should be extended
        # towards multiple bands or rsis, but requires some more thinking...
        # The idea is that you specify in the features_meta for which bands
        # the single_season features need to be computed and you make sure
        # these timeseries are imported in this function

        timer = self.timer
        phen_feat = None

        logger.info(f"{resolution}m: computing pheno features")
        if 'pheno_mult_season' in pheno_feat_meta.keys():
            timer.features[resolution].start()
            phen_feat = seasons.pheno_mult_season_features(resolution)
            timer.features[resolution].stop()

        if 'pheno_single_season' in pheno_feat_meta.keys():
            timer.features[resolution].start()
            sel_mode = pheno_feat_meta['pheno_single_season'][
                'select_season']['mode']
            sel_param = pheno_feat_meta['pheno_single_season'][
                'select_season']['param']
            if phen_feat is None:
                phen_feat = seasons.pheno_single_season_features(sel_mode,
                                                                 sel_param,
                                                                 resolution)
            else:
                phen_feat = phen_feat.merge(
                    seasons.pheno_single_season_features(sel_mode,
                                                         sel_param,
                                                         resolution))
            timer.features[resolution].stop()

        return phen_feat

    def augment(self, mask, max_additional_invalid=0.20):
        '''Augmentation method for the cloud/shadow mask.
        The aim is to make more robust classifiers by making timeseries
        more representative for multiple years. We accomplish this by
        randomly removing some additional valid data to simulate random
        additional clouds/shadows.
        By default, up to 20% of valid observations will be removed
        '''

        #  Values False or 0 in the mask will be set to nodata_value.

        logger.info('Augmenting existing cloud mask ...')

        if isinstance(mask, dict):
            mask = mask[10]

        augmented_mask = np.copy(mask)

        # Get the valid acquisitions which, defined as having at least
        # one valid pixel.
        valid_acquisitions = np.where(np.any(mask, axis=(1, 2)))[0]

        # Get a random additional cloud percentage
        additional_invalid = np.random.uniform(0, max_additional_invalid)
        additional_invalid = max(1, int(additional_invalid *
                                        len(valid_acquisitions)))

        # Derive randome invalid acquisitions from these
        additional_invalid_acq = np.random.choice(
            valid_acquisitions,
            size=additional_invalid,
            replace=False)

        logger.info((f'Removed {len(additional_invalid_acq)} valid '
                     'acquisitions from the '
                     f'available {len(valid_acquisitions)}.'))

        # Put these acuisitions to invalid
        augmented_mask[additional_invalid_acq, ...] = False

        # Create the 20m mask as well
        mask_20 = downsample_n(augmented_mask.astype(np.uint16), 1) == 1

        mask_dict = {10: augmented_mask,
                     20: mask_20}

        return mask_dict


class TSL2AFeaturesProcessor(L2AFeaturesProcessor):

    def load_mask(self):
        from satio.timeseries import load_timeseries

        nodata = 32767  # Terrascope S2-TOC nodata value

        logger.info(("Terrascope L2A collection "
                     f"products: {self.collection.df.shape[0]}"))

        self.timer.load[20].start()
        logger.info(f'SCL: loading')
        scl_ts = load_timeseries(self.collection, 'SCENECLASSIFICATION')
        scl_ts.data[scl_ts.data == nodata] = 0
        self.timer.load[20].stop()

        scl_ts = scl_ts.upsample()

        logger.info(f"SCL: preparing mask")
        mask, obs, valid_before, valid_after = scl_mask(
            scl_ts.data, **self.settings['mask'])

        mask_20 = downsample_n(mask.astype(np.uint16), 1) == 1

        mask_dict = {10: mask,
                     20: mask_20}

        return mask_dict, obs, valid_before, valid_after

    def load_data(self,
                  resolution,
                  timeseries=None):
        """
        Load Timeseries from the collection and merge with `timeseries` if
        given.
        We need to override the default method to account for specific no data
        and explicitly cast dtype
        """

        return super().load_data(resolution,
                                 timeseries=timeseries,
                                 no_data=32767,
                                 dtype=np.uint16)


class WorldCerealOpticalFeaturesProcessor(L2AFeaturesProcessor):

    @property
    def supported_bands(self):
        return {10: L2A_BANDS_10M,
                20: ['B05', 'B06', 'B07', 'B11', 'B12', 'MASK']}

    def load_mask(self):

        nr_L8 = self.collection.df.path.str.contains('LC08').sum()
        nr_S2 = self.collection.df.path.str.contains('MSIL2A').sum()

        logger.info(("WorldCereal OPTICAL collection "
                     f"products (Sentinel-2): {nr_S2}"))
        logger.info(("WorldCereal OPTICAL collection "
                     f"products (Landsat 8): {nr_L8}"))

        self.timer.load[20].start()
        logger.info(f'MASK: loading')
        scl_ts = self.collection.load_timeseries('MASK')
        self.timer.load[20].stop()

        scl_ts = scl_ts.upsample()

        logger.info(f"MASK: preparing mask")
        mask, obs, valid_before, valid_after = binary_mask(
            scl_ts.data, **self.settings['mask'])

        mask_20 = downsample_n(mask.astype(np.uint16), 1) == 1

        mask_dict = {10: mask,
                     20: mask_20}

        return mask_dict, obs, valid_before, valid_after


class L8ThermalFeaturesProcessor(WorldCerealFeaturesProcessor):
    '''
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = _FeaturesTimer(10)

    @property
    def supported_bands(self):
        return L8_THERMAL_BANDS_DICT

    @property
    def _reflectance(self):
        return False

    @property
    def supported_rsis(self):

        if self._supported_rsis is None:
            rsis_dict = {}
            rsis_dict[10] = [r for r in self._rsi_meta.keys()]
            self._supported_rsis = rsis_dict

        return self._supported_rsis

    def preprocess_data(self,
                        timeseries: 'satio.timeseries.Timeseries',
                        resolution: int,
                        mask: np.ndarray = None,
                        composite: bool = True,
                        interpolate: bool = True):
        """
        Pre-processing of loaded timeseries object. Includes masking,
        compositing and interpolation.
        """
        settings = self.settings
        newtimeseries = None

        for band in timeseries.bands:
            band_ts = timeseries.select_bands([band])
            if mask is not None:
                # mask 'nan' values. for uint16 we use 0 as nodata value
                # mask values that are False are marked as invalid
                if isinstance(mask, dict):
                    mask = mask[resolution]

                band_ts = band_ts.mask(mask)

            # drop all no data frames
            band_ts = band_ts.drop_nodata()

            composite_settings = settings.get('composite')
            if (composite_settings is not None) & composite:
                self.timer.composite[resolution].start()
                logger.info(f"{band}: compositing")
                band_ts = band_ts.composite(**composite_settings)
                self.timer.composite[resolution].stop()

            if interpolate:
                self.timer.interpolate[resolution].start()
                logger.info(f'{band}: interpolating')
                band_ts = band_ts.interpolate()
                self.timer.interpolate[resolution].stop()

            if newtimeseries is None:
                newtimeseries = band_ts
            else:
                newtimeseries = newtimeseries.merge(band_ts)

        return newtimeseries

    def load_data(self,
                  resolution=10,
                  timeseries=None,
                  no_data=0,
                  dtype=None):
        """
        Load Timeseries from the collection and merge with `timeseries` if
        given.
        `dtype` allows optional explicit casting of the loaded data
        """
        collection = self.collection

        # check which bands are needed for the rsis at this resolution
        rsis = self.rsis[resolution]
        rsi_bands = {'all': []}
        for rsi in rsis:
            rsi_bands['all'].extend(self._rsi_meta[rsi]['bands'])
        # remove duplicates
        rsi_bands['all'] = list(dict.fromkeys(rsi_bands['all']))
        rsi_bands['10'] = [b for b in rsi_bands['all']
                           if b in self.supported_bands[10]]

        bands = self.bands[resolution].copy()
        # add those for rsi...
        bands.extend([b for b in rsi_bands['{}'.format(resolution)]
                      if b not in bands])
        loaded_bands = timeseries.bands if timeseries is not None else []
        bands_to_load = [b for b in bands if b not in loaded_bands]

        if(len(bands_to_load) > 0):
            self.timer.load[resolution].start()
            logger.info(f'Loading bands: {bands_to_load}')
            bands_ts = collection.load_timeseries(*bands_to_load)

            bands_ts.data[bands_ts.data == no_data] = 0
            if dtype is not None:
                bands_ts.data = bands_ts.data.astype(dtype)
            if timeseries is None:
                timeseries = bands_ts
            else:
                timeseries = timeseries.merge(bands_ts)
            self.timer.load[resolution].stop()
        else:
            logger.info("Did not find bands to "
                        f"load for resolution: {resolution}")

        return timeseries

    def load_mask(self):

        logger.info(f"L8 collection products: {self.collection.df.shape[0]}")

        self.timer.load[10].start()
        logger.info(f'QA-PIXEL: loading')
        try:
            pixel_qa_ts = self.collection.load_timeseries('QA-PIXEL')
        except Exception:
            # tiledcollection
            pixel_qa_ts = self.collection.load_timeseries('PIXEL-QA')

        self.timer.load[10].stop()

        logger.info(f"QA-PIXEL: preparing mask")
        mask, obs, invalid_before, invalid_after = pixel_qa_mask(
            pixel_qa_ts.data, **self.settings['mask'])

        mask_dict = {10: mask}

        return mask_dict, obs, invalid_before, invalid_after

    def compute_lst_ta(self, ts_proc, resolution,
                       features_meta, chunk_size):

        # get air temperature timeseries
        settings = self.settings
        METEOcol = settings.get('METEOcol', None)
        if METEOcol is None:
            raise ValueError(
                'Need METEO collection to compute LST-Ta feature')

        airtemp = METEOcol.load_timeseries('temperature_mean',
                                           resolution=100)

        # resample to exact timestamps of L8
        airtemp = airtemp.select_timestamps(
            timestamps=ts_proc.timestamps,
            resample=True)

        # resample to spatial resolution of L8
        try:
            required_xdim = ts_proc.data.shape[3]
            required_ydim = ts_proc.data.shape[2]

            airtemp.data = AgERA5_resize(airtemp.data,
                                         shape=(required_ydim,
                                                required_xdim))
        except AttributeError:
            # Likely a TrainingCollection
            airtemp.data = AgERA5_resize(airtemp.data, scaling=64)

        # compute the time series
        ts_vals = ts_proc.data[0, ...] - airtemp.data[0, ...]
        ts_vals = np.expand_dims(ts_vals, axis=0)
        # if no LST information is available -> result should be 0
        ts_vals[ts_proc.data == 0] = 0
        ts = ts_proc._clone(data=ts_vals, bands=['lst_ta'])

        # calculate features
        features = ts.features_from_dict(
            resolution,
            features_meta=features_meta,
            chunk_size=chunk_size)

        return features

    def compute_features(self,
                         chunk_size=None,
                         **kwargs):

        timer = self.timer
        features_meta = self._features_meta
        rsi_meta = self._rsi_meta

        mask, obs, invalid_before, invalid_after = self.load_mask()

        # 10m processing
        resolution = 10
        ts = None
        ts_proc = None
        ts = self.load_data()

        if ts is not None:
            # check which rsis are required
            # (lst_ta) is done in separate function
            rsis = self.rsis[resolution]
            if 'lst_ta' in rsis:
                lst_ta_feat = True
                rsis.remove('lst_ta')
            else:
                lst_ta_feat = False

            # pre-process the timeseries
            ts_proc = self.preprocess_data(ts, resolution, mask=mask)

            # compute other rsis than lst-ta
            if len(rsis) > 0:
                logger.info(f"{resolution}m: computing rsis")
                timer.rsi[resolution].start()
                ts_rsi = ts_proc.compute_rsis(*rsis, rsi_meta=rsi_meta)
                timer.rsi[resolution].stop()

            # start feature calculation
            timer.features[resolution].start()

            # compute 10m features
            if len(self.bands[resolution]) > 0:
                logger.info(f"{resolution}m: computing bands features")
                features_10m = ts_proc.select_bands(
                    self.bands[resolution]).features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size)
            else:
                features_10m = None

            # add RSI features
            logger.info(f"{resolution}m: computing rsi features")
            # first others than lst-ta
            if len(rsis) > 0:
                if features_10m is not None:
                    features_10m = features_10m.merge(
                        ts_rsi.features_from_dict(
                            resolution,
                            features_meta=features_meta,
                            chunk_size=chunk_size))
                else:
                    features_10m = ts_rsi.features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size)

            # computing LST - airtemp feature
            if lst_ta_feat:
                if features_10m is not None:
                    features_10m = features_10m.merge(
                        self.compute_lst_ta(
                            ts_proc, resolution,
                            features_meta, chunk_size))
                else:
                    features_10m = self.compute_lst_ta(
                        ts_proc, resolution,
                        features_meta, chunk_size)

            timer.features[resolution].stop()

            # add meta features
            features_10m = features_10m.merge(
                Features(np.array([obs, invalid_before, invalid_after]),
                         names=['L8_observations',
                                'L8_invalid_before',
                                'L8_invalid_after']))

        else:
            features_10m = None

        timer.load[resolution].log()
        timer.rsi[resolution].log()
        timer.composite[resolution].log()
        timer.interpolate[resolution].log()
        timer.features[resolution].log()

        return features_10m


class WorldCerealThermalFeaturesProcessor(L8ThermalFeaturesProcessor):

    @property
    def supported_bands(self):
        return {10: ['B10', 'QA_PIXEL']}

    def load_mask(self):
        logger.info(f"WorldCereal THERMAL collection "
                    f"products: {self.collection.df.shape[0]}")

        self.timer.load[10].start()
        logger.info(f'QA_PIXEL: loading')
        qa_ts = self.collection.load_timeseries('QA_PIXEL', resample=True)
        self.timer.load[10].stop()

        logger.info(f"QA_PIXEL: preparing mask")
        mask, obs, invalid_before, invalid_after = pixel_qa_mask(
            qa_ts.data, **self.settings['mask'])

        mask_dict = {10: mask}

        return mask_dict, obs, invalid_before, invalid_after

    def load_data(self,
                  resolution=10,
                  timeseries=None,
                  no_data=0,
                  dtype=np.float32):
        """
        Load Timeseries from the collection and merge with `timeseries` if
        given.
        `dtype` allows optional explicit casting of the loaded data
        """
        collection = self.collection

        # check which bands are needed for the rsis at this resolution
        rsis = self.rsis[resolution]
        rsi_bands = {'all': []}
        for rsi in rsis:
            rsi_bands['all'].extend(self._rsi_meta[rsi]['bands'])
        # remove duplicates
        rsi_bands['all'] = list(dict.fromkeys(rsi_bands['all']))
        if 'ST-B10' in rsi_bands['all']:
            rsi_bands['all'].remove('ST-B10')
            rsi_bands['all'].append('B10')
        rsi_bands['10'] = [b for b in rsi_bands['all']
                           if b in self.supported_bands[10]]

        bands = self.bands[resolution].copy()
        # add those for rsi...
        bands.extend([b for b in rsi_bands['{}'.format(resolution)]
                      if b not in bands])
        loaded_bands = timeseries.bands if timeseries is not None else []
        bands_to_load = [b for b in bands if b not in loaded_bands]

        if(len(bands_to_load) > 0):
            self.timer.load[resolution].start()
            logger.info(f'Loading bands: {bands_to_load}')
            bands_ts = collection.load_timeseries(*bands_to_load,
                                                  resample=True)

            bands_ts.data[bands_ts.data == no_data] = 0
            if dtype is not None:
                bands_ts.data = bands_ts.data.astype(dtype)
            if timeseries is None:
                timeseries = bands_ts
            else:
                timeseries = timeseries.merge(bands_ts)
            self.timer.load[resolution].stop()
        else:
            logger.info("Did not find bands to "
                        f"load for resolution: {resolution}")

        return timeseries


class SARFeaturesProcessor(WorldCerealFeaturesProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = _FeaturesTimer(20)

    @ property
    def supported_bands(self):
        return SIGMA0_BANDS_DICT

    @ property
    def supported_rsis(self):
        if self._supported_rsis is None:
            rsis_dict = {}
            rsis_dict[20] = [r for r in self._rsi_meta.keys()]
            self._supported_rsis = rsis_dict

        return self._supported_rsis

    @ property
    def _reflectance(self):
        return False

    def load_data(self,
                  resolution=20,
                  timeseries=None):
        """
        Load Timeseries from the collection and merge with `timeseries` if
        given.
        """

        collection = self.collection

        bands = self.bands[resolution]
        loaded_bands = timeseries.bands if timeseries is not None else []
        bands_to_load = [b for b in bands if b not in loaded_bands]

        logger.info(f"SIGMA0 collection products: {collection.df.shape[0]}")

        if(len(bands_to_load) > 0):
            self.timer.load[resolution].start()

            # Put `mask_and_scale` to true to get physical dB values
            # TODO: a future disk collection does not have this attribute!
            logger.info(f'Loading bands: {bands_to_load}')
            bands_ts = collection.load_timeseries(*bands_to_load,
                                                  mask_and_scale=True)
            if timeseries is None:
                timeseries = bands_ts
            else:
                timeseries = timeseries.merge(bands_ts)
            self.timer.load[resolution].stop()

        return timeseries

    def preprocess_data(self,
                        timeseries: 'satio.timeseries.Timeseries',
                        resolution: int = 20,
                        mask: 'satio.timeseries.Timeseries' = None,
                        speckle_filter: bool = True,
                        composite: bool = True,
                        interpolate=None):
        """
        Pre-processing of loaded timeseries object. Includes masking,
        speckle filtering, compositing and interpolation.
        """

        def _to_db(pwr):
            return 10 * np.log10(pwr)

        def _to_pwr(db):
            return np.power(10, db / 10)

        settings = self.settings
        newtimeseries = None

        # number of obs needs to be calculated here
        self._obs = None
        self._obs_notmasked = None

        for band in timeseries.bands:
            band_ts = timeseries.select_bands([band])

            # get number of observations for first band loaded
            if self._obs is None:
                self._obs = band_ts.data[0]
                self._obs = np.isfinite(self._obs)
                self._obs = self._obs.sum(axis=0)

            if mask is not None:

                # mask 'nan' values. for uint16 we use 0 as nodata value
                # mask values that are False are marked as invalid
                if isinstance(mask, dict):
                    mask = mask[resolution]

                # Subset the mask on the needed timestamps
                mask_subset = mask.select_timestamps(band_ts.timestamps)

                # Apply mask
                nr_valid_before = np.isfinite(band_ts.data[0]).sum(axis=0)
                band_ts = band_ts.mask(mask_subset.data.astype(bool))

                if self._obs_notmasked is None:
                    nr_valid_after = np.isfinite(band_ts.data[0]).sum(axis=0)
                    self._obs_notmasked = (nr_valid_after /
                                           nr_valid_before * 100)

            # drop all no data frames
            band_ts = band_ts.drop_nodata()

            # Before doing manupulations
            # first need to get rid of dB
            data_lin = _to_pwr(band_ts.data)

            if speckle_filter:
                logger.info(f"{band}: speckle filtering")
                idx_nodata = np.isnan(data_lin)
                data_lin[idx_nodata] = 0  # Speckle filter uses 0 as nodata
                self.timer.speckle[resolution].start()
                for band_idx in range(data_lin.shape[0]):
                    data_lin[band_idx] = multitemporal_speckle(
                        data_lin[band_idx])
                data_lin[idx_nodata] = np.nan
                self.timer.speckle[resolution].stop()

            band_ts.data = data_lin

            composite_settings = settings.get('composite')
            if (composite_settings is not None) & composite:
                self.timer.composite[resolution].start()
                logger.info(f"{band}: compositing")
                band_ts = band_ts.composite(**composite_settings)
                self.timer.composite[resolution].stop()

            # If required, we do here the GDD normalization
            if 'normalize_gdd' in self.settings.keys():
                logger.info(f'{band}: applying GDD normalization')
                band_ts = apply_gdd_normalization(band_ts, self.settings)

            if interpolate is None:
                interpolate = settings.get('interpolate', True)
            if interpolate:
                self.timer.interpolate[resolution].start()
                logger.info(f'{band}: interpolating')
                band_ts = band_ts.interpolate()
                self.timer.interpolate[resolution].stop()

            # Finally back to dB
            data_db = _to_db(band_ts.data)
            band_ts.data = data_db

            if newtimeseries is None:
                newtimeseries = band_ts
            else:
                newtimeseries = newtimeseries.merge(band_ts)

        return newtimeseries

    def load_mask(self, resolution=20):

        from satio.timeseries import Timeseries

        precip_threshold = 10  # mm

        settings = self.settings.get('mask', None)
        precip_threshold = settings.get('precipitation_threshold',
                                        precip_threshold)
        AgERA5 = settings.get('METEOcol')

        self.timer.load[20].start()
        logger.info(f'METEO: loading precipitation_flux')
        precip_ts = AgERA5.load_timeseries('precipitation_flux',
                                           resolution=resolution)
        # if all Nan's -> block is situated completely over the ocean
        if np.all(np.isnan(precip_ts.data)):
            raise SkipBlockError(
                ('Loaded precipitation_flux resulted in all '
                 'NaNs, block probably not located over land!'))

        # Make the mask
        new_data = precip_ts.data[0, ...]
        new_data[new_data <= precip_threshold] = 1
        new_data[new_data > precip_threshold] = 0

        # Back to Timeseries for further handling
        mask = Timeseries(np.expand_dims(new_data, axis=0),
                          precip_ts.timestamps, ['mask'])

        self.timer.load[20].stop()

        return mask

    def compute_features(self,
                         chunk_size=None,
                         upsample_order=1,
                         augment=False,
                         **kwargs):

        lproc = self
        features_meta = self._features_meta
        rsi_meta = self._rsi_meta

        msk_settings = self.settings.get('mask', None)
        if msk_settings:
            if msk_settings.get('METEOcol', None):
                mask = self.load_mask()
            else:
                raise ValueError(('An `METEOcol` is needed '
                                  'in settings when S1 masking '
                                  'is requested'))
        else:
            mask = None

        # check if texture features need to be computed
        # and store features_meta of those separately
        if 'texture' in features_meta.keys():
            text_feat = True
            text_feat_meta = features_meta['texture']
            del features_meta['texture']
        else:
            text_feat = False

        # if no other features remain at this point -> abort!
        if not bool(features_meta):
            raise ValueError('At least one other feature required'
                             'other than texture. Aborting...')

        # 20m processing
        resolution = 20

        ts = lproc.load_data(resolution)

        if ts is not None:

            if augment:
                ts = self.augment(ts)

            # first pre-process the timeseries
            ts = lproc.preprocess_data(ts, resolution, mask=mask)

            # now we compute rsis
            rsis = lproc.rsis[resolution]
            if len(rsis) > 0:
                logger.info(f"{resolution}m: computing rsis")
                self.timer.rsi[resolution].start()
                ts_rsi = ts.compute_rsis(*rsis,
                                         rsi_meta=rsi_meta,
                                         bands_scaling=1)
                self.timer.rsi[resolution].stop()

            # start feature calculation
            self.timer.features[resolution].start()
            features = None
            bands = self.bands[resolution]
            if len(bands) > 0:
                logger.info(f"{resolution}m: computing bands features")
                features = ts.select_bands(
                    bands).features_from_dict(resolution,
                                              features_meta=features_meta,
                                              chunk_size=chunk_size)

            # add RSI features
            logger.info(f"{resolution}m: computing rsi features")
            if len(rsis) > 0:
                if features is not None:
                    features = features.merge(ts_rsi.features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size))
                else:
                    features = ts_rsi.features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size)

            self.timer.features[resolution].stop()

            # optionally compute texture features based on
            # computed features
            if text_feat:
                logger.info('Computing texture features')
                self.timer.text_features[resolution].start()
                inputFeat = features.select(text_feat_meta['features'])
                params = text_feat_meta['parameters']
                # if desired, run PCA first
                if 'pca' in text_feat_meta.keys():
                    inputFeat = inputFeat.pca(text_feat_meta['pca'],
                                              scaling_range=params.get(
                        'scaling_range', None))

                features = features.merge(inputFeat.texture(
                    win=params.get('win', 2),
                    d=params.get('d', [1]),
                    theta=params.get('theta', [0, np.pi/4]),
                    levels=params.get('levels', 256),
                    metrics=params.get('metrics', ('contrast',)),
                    avg=params.get('avg', True),
                    scaling_range=params.get('scaling_range', {})))

                self.timer.text_features[resolution].stop()

            # add meta features
            features = features.merge(
                Features(np.array([self._obs]),
                         names=['sigma0_obs']))
            if msk_settings:
                features = features.merge(
                    Features(np.array([self._obs_notmasked]),
                             names=['sigma0_obs_percentvalid']))

            # Now finally, we go to 10m
            features = features.upsample(order=upsample_order)

        else:
            features = None

        for r in [20]:
            self.timer.load[r].log()
            self.timer.rsi[r].log()
            self.timer.composite[r].log()
            self.timer.interpolate[r].log()
            self.timer.features[r].log()
            self.timer.text_features[r].log()

        return features

    def augment(self, ts, max_additional_invalid=0.33):
        '''Augmentation method for SAR.
        The aim is to make more robust classifiers by making timeseries
        more representative for multiple years. We accomplish this by
        randomly removing some additional valid data to simulate random
        missing SAR Data.
        By default, up to 33% of valid observations will be removed
        '''

        logger.info('Augmenting SAR data ...')

        # Get the valid acquisitions which, defined as having at least
        # one valid pixel.
        valid_acquisitions = np.arange(ts.data.shape[1])

        # Get a random additional removal percentage
        additional_invalid = np.random.uniform(0, max_additional_invalid)
        additional_invalid = max(1, int(additional_invalid *
                                        len(valid_acquisitions)))

        # Derive randome invalid acquisitions from these
        additional_invalid_acq = np.random.choice(
            valid_acquisitions,
            size=additional_invalid,
            replace=False)

        logger.info((f'Removed {len(additional_invalid_acq)} valid '
                     'acquisitions from the '
                     f'available {len(valid_acquisitions)}.'))

        # Put these acuisitions to invalid
        ts.data[:, additional_invalid_acq, ...] = np.nan

        return ts


class TSSIGMA0FeaturesProcessor(SARFeaturesProcessor):

    def load_data(self,
                  resolution=20,
                  timeseries=None):
        """
        Override of default method to cope with
        Terrascope specific data
        """

        def _to_db(pwr):
            return 10 * np.log10(pwr)

        collection = self.collection

        bands = self.bands[resolution]
        loaded_bands = timeseries.bands if timeseries is not None else []
        bands_to_load = [b for b in bands if b not in loaded_bands]

        logger.info(f"SIGMA0 collection products: {collection.df.shape[0]}")

        if(len(bands_to_load) > 0):
            self.timer.load[resolution].start()

            # The raw data we load needs to be transformed to
            # dB and downsampled to 20m for compatibility
            # with the usual Sigma0/Gamma0 products
            logger.info(f'Loading bands: {bands_to_load}')
            bands_ts = collection.load_timeseries(*bands_to_load)

            if resolution == 20:
                # Need to downsample. Because values
                # are in float, need to use imresize method
                bands_ts = bands_ts.imresize(scaling=0.5)

            bands_ts.data = _to_db(bands_ts.data)

            if timeseries is None:
                timeseries = bands_ts
            else:
                timeseries = timeseries.merge(bands_ts)
            self.timer.load[resolution].stop()

        return timeseries


class TSSIGMA0TiledFeaturesProcessor(SARFeaturesProcessor):

    def load_data(self,
                  resolution=20,
                  timeseries=None):
        """
        Override of default method to cope with
        tiled Terrascope specific data
        """

        nodata = 65535

        collection = self.collection

        bands = self.bands[resolution]
        loaded_bands = timeseries.bands if timeseries is not None else []
        bands_to_load = [b for b in bands if b not in loaded_bands]

        logger.info(f"SIGMA0 collection products: {collection.df.shape[0]}")

        if(len(bands_to_load) > 0):
            self.timer.load[resolution].start()

            # Load the raw data
            logger.info(f'Loading bands: {bands_to_load}')
            bands_ts = collection.load_timeseries(*bands_to_load)

            # Convert to float32, unfortunately
            bands_ts.data = bands_ts.data.astype(np.float32)

            # Apply nodata
            bands_ts.data[bands_ts.data == nodata] = np.nan

            # The data is scaled. Need to unscale to real dB values
            bands_ts.data = 20 * np.log10(bands_ts.data) - 83

            if timeseries is None:
                timeseries = bands_ts
            else:
                timeseries = timeseries.merge(bands_ts)
            self.timer.load[resolution].stop()

        return timeseries


class WorldCerealSARFeaturesProcessor(SARFeaturesProcessor):

    def load_data(self,
                  resolution=20,
                  timeseries=None):
        """
        Override of default method to cope with
        WorldCereal specific data
        """

        # Outside-swath pixels are 65535
        nodata = 65535

        collection = self.collection

        bands = self.bands[resolution]
        loaded_bands = timeseries.bands if timeseries is not None else []
        bands_to_load = [b for b in bands if b not in loaded_bands]

        logger.info(("WorldCereal SIGMA0 collection products: "
                     f"{collection.df.shape[0]}"))

        if(len(bands_to_load) > 0):
            self.timer.load[resolution].start()

            # First load the raw data
            logger.info(f'Loading bands: {bands_to_load}')
            bands_ts = collection.load_timeseries(*bands_to_load)
            bands_ts.data = bands_ts.data.astype(np.float32)

            # Apply nodata
            bands_ts.data[bands_ts.data == nodata] = np.nan

            '''
            OTB-processed backscatter puts all pixels that have
            backscatter below the noise level to "0". This is different
            from SNAP which is what the system is trained on.
            We need to be remove those 0 values with a HACK, by assigning
            low backscatter values to these 0 pixels.
            '''

            # Identify below-noise pixels
            idx_belownoise = bands_ts.data == 0

            # Get random low number between DN 50-300 (~[-50dB, -33dB])
            logger.debug('Introducing random noise ...')
            random_low = np.random.randint(50, 300, size=idx_belownoise.sum())

            # Replace below noise with low value
            bands_ts.data[idx_belownoise] = random_low

            # Remove product-specific scaling towards true dB
            bands_ts.data = 20 * np.log10(bands_ts.data) - 83

            if timeseries is None:
                timeseries = bands_ts
            else:
                timeseries = timeseries.merge(bands_ts)
            self.timer.load[resolution].stop()

        return timeseries


class AgERA5FeaturesProcessor(WorldCerealFeaturesProcessor):

    def __init__(self, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = _FeaturesTimer(100, 20)

    @ property
    def supported_bands(self):
        return AGERA5_BANDS_DICT

    @ property
    def _reflectance(self):
        return False

    @ property
    def _optical_fp(self):
        return L2AFeaturesProcessor

    @ property
    def _thermal_fp(self):
        return L8ThermalFeaturesProcessor

    @ property
    def supported_rsis(self):
        if self._supported_rsis is None:
            rsis_dict = {}
            rsi_res = {r: self._rsi_meta[r]['native_res']
                       for r in self._rsi_meta.keys()}
            rsis_dict[100] = [v for v, r in rsi_res.items() if r == 100]
            rsis_dict[20] = [v for v, r in rsi_res.items() if r == 20]
            self._supported_rsis = rsis_dict

        return self._supported_rsis

    def preprocess_data(self,
                        timeseries,
                        resolution=100,
                        composite=True):
        """
        Pre-processing of loaded timeseries object. Includes compositing.
        As the data is daily without missing values, no interpolation is needed
        """
        settings = self.settings
        newtimeseries = None

        for band in timeseries.bands:
            band_ts = timeseries.select_bands([band])

            # drop all no data frames
            band_ts = band_ts.drop_nodata()

            composite_settings = settings.get('composite')
            if (composite_settings is not None) & composite:
                self.timer.composite[resolution].start()
                logger.info(f"{band}: compositing")
                band_ts = band_ts.composite(**composite_settings)
                self.timer.composite[resolution].stop()

            # If required, we do here the GDD normalization
            if 'normalize_gdd' in self.settings.keys():
                logger.info(f'{band}: applying GDD normalization')
                band_ts = apply_gdd_normalization(band_ts, self.settings)

            if newtimeseries is None:
                newtimeseries = band_ts
            else:
                newtimeseries = newtimeseries.merge(band_ts)

        return newtimeseries

    def load_data(self,
                  resolution=100,
                  timeseries=None):
        """
        Load Timeseries from the collection and merge with `timeseries` if
        given.
        """

        collection = self.collection

        bands = self.bands[resolution].copy()
        # check which bands are needed for the desired rsis
        # for now only 100 m rsis supported!
        rsis = self.rsis[100].copy()
        rsis.extend(self.rsis[20].copy())
        rsi_bands = []
        for rsi in rsis:
            rsi_bands.extend(self._rsi_meta[rsi]['bands'])
        # add to bands and remove duplicates
        bands.extend(rsi_bands)
        bands = list(dict.fromkeys(bands))
        # remove all bands not supported
        bands = [b for b in bands if b in self.supported_bands[resolution]]
        # check which are available already
        loaded_bands = timeseries.bands if timeseries is not None else []
        bands_to_load = [b for b in bands if b not in loaded_bands]

        logger.info(f"AgERA5 collection products: {collection.df.shape[0]}")

        if(len(bands_to_load) > 0):
            self.timer.load[resolution].start()
            logger.info(f'Loading bands: {bands_to_load}')
            bands_ts = collection.load_timeseries(*bands_to_load,
                                                  resolution=resolution)
            if timeseries is None:
                timeseries = bands_ts
            else:
                timeseries = timeseries.merge(bands_ts)
            self.timer.load[resolution].stop()
        else:
            logger.info("Did not find bands to "
                        f"load for resolution: {resolution}")

        return timeseries

    def compute_et0(self, ts, chunk_size):

        from satio.timeseries import Timeseries

        resolution = 100
        logger.info(f"{resolution}m: computing et0")
        features_meta = self._features_meta
        rsi_meta = self._rsi_meta

        _, nt, nx, ny = ts.data.shape

        # get latitude
        bounds = self.settings.get('bounds', None)
        epsg = self.settings.get('epsg', None)
        if ((bounds is None) or (epsg is None)):
            raise ValueError('Specify bounds and epsg in settings'
                             ' for ET0/ET computation!')
        lat = Features.from_latlon(
            bounds,
            epsg,
            resolution=resolution).select(['lat'])
        # in case of resolution mismatch (for training collection),
        # we downsample lat information
        if lat.data.shape[2] > nx:
            out_shape = ts.data.shape[-2:]
            lat = resize(np.squeeze(lat.data),
                         out_shape, order=1,
                         mode='edge')

        # convert to timeseries and add to ts
        lat = np.broadcast_to(np.squeeze(lat.data),
                              (nt, nx, ny))
        lat = np.expand_dims(lat, axis=0)
        ts = ts.merge(
            Timeseries(lat, ts.timestamps, ['lat']))

        # get elevation from dem
        demcol = self.settings.get('demcol', None)
        if demcol is None:
            raise ValueError('Specify DEM collection in settings'
                             ' for ET0/ET computation!')
        elev = elev_from_dem(demcol, resolution=20)
        # downsample to meteo spatial resolution
        out_shape = ts.data.shape[-2:]
        elev = resize(np.squeeze(elev.data),
                      out_shape, order=1,
                      mode='edge')
        # convert to timeseries and add to ts
        elev = np.broadcast_to(elev[np.newaxis, ...],
                               (nt, nx, ny))
        elev = np.expand_dims(elev, axis=0)
        ts = ts.merge(
            Timeseries(elev, ts.timestamps, ['elev']))

        # get day of year from ts.timestamps
        times = ts.timestamps
        doy = np.asarray(
            [t.timetuple().tm_yday for t in times],
            dtype=np.float32)
        # convert to timeseries and add to ts
        doy = np.broadcast_to(doy[:, np.newaxis, np.newaxis],
                              (nt, nx, ny))
        doy = np.expand_dims(doy, axis=0)
        ts = ts.merge(
            Timeseries(doy, ts.timestamps, ['doy']))

        # compute et0 timeseries
        et0ts = ts.compute_rsis('et0',
                                rsi_meta=rsi_meta,
                                bands_scaling=1)

        # calculate et0 features
        et0feat = et0ts.features_from_dict(resolution,
                                           features_meta=features_meta,
                                           chunk_size=chunk_size)

        return et0feat, et0ts

    def compute_sm(self, s2_ts, meteots, l8b10, chunk_size):

        resolution = 20
        logger.info(f"{resolution}m: computing soil moisture")

        # select AgERA5 data based on S2 availability
        timestamps = s2_ts.timestamps
        meteots = meteots.select_timestamps(timestamps)

        # Calculate the theoretical boundaries using meteo data only
        boundaries = theoretical_boundaries(meteots)
        # upsample to 20m
        boundaries = self._upsample_meteo_ts(boundaries)

        # get NDVI at 20m
        ndvi = s2_ts.select_bands(['ndvi']).data[0]

        # Calculate Soil Moisture and Soil Moisture Stress based on NDVI, LST,
        # and theoretical boundaries
        smts, smstress = soil_moisture(ndvi, l8b10, boundaries)

        # compute soil moisture features
        settings = self.settings
        features_meta = self._features_meta
        if 'sm' in settings['rsis']:
            smfeat = smts.features_from_dict(resolution,
                                             features_meta=features_meta,
                                             chunk_size=chunk_size)
        else:
            smfeat = None
        if 'smstress' in settings['rsis']:
            smstressfeat = smstress.features_from_dict(
                resolution,
                features_meta=features_meta,
                chunk_size=chunk_size)
            if smfeat is not None:
                # merge features
                smfeat = smfeat.merge(smstressfeat)
            else:
                smfeat = smstressfeat

        return smfeat, smts

    def compute_et(self, et0ts, s2_ts_proc, chunk_size):

        resolution = 20
        logger.info(f"{resolution}m: computing et")
        features_meta = self._features_meta
        settings = self.settings

        # get et settings
        etset = settings.get('et', None)
        if etset is None:
            raise ValueError('Need et settings for ET computation!')
        method = etset.get('method', None)

        if 'ndvi' in method:
            ndvi = s2_ts_proc.select_bands(['ndvi']).data[0, ...]
            # compute kc factor
            if method == 'ndvi_linear':
                kc = 1.4571 * ndvi - 0.1725
            elif method == 'ndvi_cubic':
                ndvi_min = np.nanmin(ndvi, axis=0)
                ndvi_max = np.nanmax(ndvi, axis=0)
                ndvi_norm = ((ndvi - ndvi_min) /
                             (ndvi_max - ndvi_min))
                kc = (0.176
                      + 1.325 * ndvi_norm
                      - 1.466 * ndvi_norm**2
                      + 1.146 * ndvi_norm**3)
            else:
                raise ValueError('Requested ET method not available!')
        else:
            raise ValueError('Requested ET method not available!')

        # compute et
        etts_val = np.expand_dims(kc * et0ts.data[0, ...], axis=0)
        etts = et0ts._clone(data=etts_val, bands=['et'])

        # calculate features
        etfeat = etts.features_from_dict(resolution,
                                         features_meta=features_meta,
                                         chunk_size=chunk_size)

        return etfeat, etts

    def compute_features_100m(self, ts, chunk_size):
        resolution = 100
        timer = self.timer
        features_meta = self._features_meta
        rsi_meta = self._rsi_meta
        rsis = self.rsis[resolution]

        # et0 happens through a separate function prior to
        # pre-processing...
        if "et0" in rsis:
            timer.rsi[resolution].start()
            features_100m, ts_rsi = self.compute_et0(ts, chunk_size)
            rsis.remove('et0')
            timer.rsi[resolution].stop()
        else:
            features_100m = None
            ts_rsi = None

        # now pre-process the timeseries
        ts = self.preprocess_data(ts, resolution)

        # pre-process et0 if needed
        if ts_rsi is not None:
            ts_rsi = self.preprocess_data(ts_rsi,
                                          resolution)

        # compute other rsis if needed
        if len(rsis) > 0:
            logger.info(f"{resolution}m: computing rsis")
            timer.rsi[resolution].start()
            if ts_rsi is None:
                ts_rsi = ts.compute_rsis(*rsis,
                                         rsi_meta=rsi_meta,
                                         bands_scaling=1)
            else:
                ts_rsi = ts_rsi.merge(
                    ts.compute_rsis(*rsis,
                                    rsi_meta=rsi_meta,
                                    bands_scaling=1))
            timer.rsi[resolution].stop()

        # start feature calculation
        timer.features[resolution].start()

        # compute 100m features
        if len(self.bands[resolution]) > 0:
            logger.info(f"{resolution}m: computing bands features")
            if features_100m is not None:
                features_100m = features_100m.merge(ts.select_bands(
                    self.bands[resolution]).features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size))
            else:
                features_100m = ts.select_bands(
                    self.bands[resolution]).features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size)

        if len(rsis) > 0:
            logger.info(f"{resolution}m: computing rsi features")
            if features_100m is not None:
                features_100m = features_100m.merge(
                    ts_rsi.select_bands(rsis).features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size)
                )
            else:
                features_100m = ts_rsi.select_bands(rsis).features_from_dict(
                    resolution,
                    features_meta=features_meta,
                    chunk_size=chunk_size)

        timer.features[resolution].stop()

        return features_100m, ts, ts_rsi

    def _prepare_s2_ts(self, list_indices):

        settings = self.settings

        # retrieve S2 collection to be used
        col = settings.get('optical_col', None)
        if col is None:
            raise ValueError(
                'Sentinel-2 collection not specified in METEO settings.'
                'Cannot continue!')

        # define feature processor settings (ndvi is always needed)
        s2set = {'bands': [],
                 'rsis': ["ndvi"],
                 'composite': {
            'freq': settings['composite'].get('freq', 10),
            'window': None,
            'start': settings['composite'].get('start', None),
            'end': settings['composite'].get('end', None)},
            'mask': {
            'erode_r': 3,
            'dilate_r': 13,
            'mask_values': SCL_MASK_VALUES,
            'multitemporal': True}
        }

        # extend settings if ssm is required
        if ('ssm' in list_indices) or ('ssm_adj' in list_indices):
            s2set['rsis'] = ["ndvi", "mndwi", "str_1"]

            s2_rsi_meta = {
                'str_1': {'bands': ['B11'],
                          'clamp_range': False,
                          'scale': 1,
                          'native_res': 20,
                          'func': calc_str_1},
            }
            process_20m = True
        else:
            s2_rsi_meta = None
            process_20m = False

        # define the feature processor
        s2_fp = self._optical_fp(col, s2set, rsi_meta=s2_rsi_meta)

        # load mask
        mask, _, _, _ = s2_fp.load_mask()
        ts = None
        if s2set.get('multitemporal', False):
            mask, ts = s2_fp.load_multitemporal_mask(prior_mask=mask)

        # load 10m data
        ts = s2_fp.load_data(10, timeseries=ts)

        # pre-process 10m data
        ts_proc = s2_fp.preprocess_data(ts, 10, mask=mask)

        # compute rsis at 10m
        rsi = ts_proc.compute_rsis(*s2_fp.rsis[10])

        # resample all to 20m
        rsi = rsi.imresize(scaling=0.5)

        if process_20m:
            # downsample 10m ts
            ts = ts.downsample()
            ts10names = ts.bands.copy()
            ts_proc = ts_proc.downsample()
            # load 20m bands
            ts = s2_fp.load_data(20, timeseries=ts)
            # pre-process the time series
            # (only those not already pre-processed)
            bands_20m = [b for b in ts.bands if b not in ts10names]
            ts_proc = ts_proc.merge(
                s2_fp.preprocess_data(
                    ts.select_bands(bands_20m), 20, mask=mask))

            # compute rsi at 20m
            rsi = rsi.merge(ts_proc.compute_rsis(*s2_fp.rsis[20],
                                                 rsi_meta=s2_rsi_meta))

        return rsi

    def compute_prdef(self, etts, precip,
                      smcor, chunk_size):

        resolution = 20
        features_meta = self._features_meta
        logger.info(f"{resolution}m: computing prdef")

        prdeffeat = None
        for cor_name, sm in smcor.items():
            if sm is None:
                sm = 1
            else:
                sm = sm.data
            prdef_val = (etts.data - precip.data) * sm
            prdef = etts._clone(data=prdef_val,
                                bands=[f'prdef{cor_name}'])

            # compute requested features
            if prdeffeat is None:
                prdeffeat = prdef.features_from_dict(
                    resolution,
                    features_meta=features_meta,
                    chunk_size=chunk_size)
            else:
                prdeffeat = prdeffeat.merge(prdef.features_from_dict(
                    resolution,
                    features_meta=features_meta,
                    chunk_size=chunk_size))

        return prdeffeat

    def compute_ssm(self, s2_rsi,
                    meteo_ts, chunk_size):

        resolution = 20
        logger.info(f"{resolution}m: computing ssm")
        features_meta = self._features_meta
        settings = self.settings

        # resample precipitation to 20m
        precip = self._upsample_meteo_ts(meteo_ts,
                                         bands=['precipitation_flux'])
        precip_data = precip.data[0]
        # get the s2 data you need
        ndvi = s2_rsi.select_bands(['ndvi']).data[0]
        mndwi = s2_rsi.select_bands(['mndwi']).data[0]
        str_1 = s2_rsi.select_bands(['str_1']).data[0]
        timestamps = s2_rsi.timestamps

        # compute surface soil moisture time series
        ssm, ssm_adj = surface_soil_moisture(
            ndvi, str_1, mndwi,
            timestamps,
            precip_data)

        # compute features from these time series
        resolution = 20
        if 'ssm' in settings['rsis']:
            ssmfeat = ssm.features_from_dict(
                resolution,
                features_meta=features_meta,
                chunk_size=chunk_size)
        else:
            ssmfeat = None
        if 'ssm_adj' in settings['rsis']:
            if ssmfeat is not None:
                ssmfeat = ssmfeat.merge(
                    ssm_adj.features_from_dict(
                        resolution,
                        features_meta=features_meta,
                        chunk_size=chunk_size))
            else:
                ssmfeat = ssm_adj.features_from_dict(
                    resolution,
                    features_meta=features_meta,
                    chunk_size=chunk_size)

        return ssmfeat, ssm, precip

    def _upsample_meteo_ts(self, input, bands=None):

        # get timeseries
        if bands is not None:
            input = input.select_bands(bands)

        # upsample data to required spatial dimensions
        try:
            required_xdim = int((self.collection.bounds[2] -
                                 self.collection.bounds[0]) / 20)
            required_ydim = int((self.collection.bounds[3] -
                                 self.collection.bounds[1]) / 20)
            input.data = AgERA5_resize(input.data, shape=(
                required_ydim, required_xdim))
        except AttributeError:
            # Likely a TrainingCollection
            # Get the bounds of the sample
            bounds = self.bounds
            # Get desired output shape
            outdim = int((bounds[2] - bounds[0]) / 20)
            # Resize
            input.data = AgERA5_resize(
                input.data,
                scaling=int(outdim / input.data.shape[-1]))

        return input

    def _prepare_thermal_data(self):

        settings = self.settings

        # set settings of feature processor
        l8settings = dict(bands=["ST-B10"],
                          composite=dict(
                              freq=10,
            window=None,
            mode='median',
            start=settings['composite'].get('start', None),
            end=settings['composite'].get('end', None)),
            mask=dict(erode_r=3,
                      dilate_r=13,
                      max_invalid_ratio=1))

        # get thermal collection
        l8col = settings.get('tir_col', None)
        if l8col is None:
            raise ValueError('Need L8Collection for '
                             'computing soil moisture!')

        # define features processor
        fp8 = self._thermal_fp(l8col, l8settings)
        mask, _, _, _ = fp8.load_mask()
        ts8 = fp8.load_data()
        ts8 = fp8.preprocess_data(ts8, mask=mask, resolution=10)
        # resample to 20m
        ts8 = ts8.downsample()

        return ts8.select_bands(['ST-B10'])

    def compute_features_20m(self, ts, ts_proc,
                             ts_rsi, chunk_size):

        resolution = 20
        timer = self.timer
        settings = self.settings

        # start feature calculation
        timer.features[resolution].start()
        features_20m = None

        # check for which features S2 data is needed
        s2_rsis = ['ssm', 'ssm_adj', 'et', 'sm', 'smstress', 'prdef']
        s2_check = [i for i in s2_rsis
                    if i in self.rsis[resolution]]
        if ('prdef' in self.rsis[resolution]) and (
                settings['prdef']['corr'] == 'ssm'):
            s2_check.append('ssm')
        if len(s2_check) > 0:
            # prepare S2 data
            s2_rsi = self._prepare_s2_ts(s2_check)

        # start with surface soil moisture
        # needed when the feature is requested directly
        ssm_check = [i for i in ['ssm', 'ssm_adj']
                     if i in self.rsis[resolution]]
        # or to correct precip deficit
        if ('prdef' in self.rsis[resolution]) & (
                settings['prdef']['corr'] == 'ssm'):
            ssm_check.append('ssm')
        if len(ssm_check) > 0:
            # need to compute surface soil moisture based on S2
            features_20m, ssmts, precip = self.compute_ssm(
                s2_rsi, ts_proc, chunk_size)
        else:
            precip = None

        # compute soil moisture
        # needed when the feature is requested directly
        sm_check = [i for i in ['sm', 'smstress']
                    if i in self.rsis[resolution]]
        # or to correct precip deficit
        if ('prdef' in self.rsis[resolution]) and (
                'sm' in settings['prdef']['corr']):
            sm_check.append('sm')
        if len(sm_check) > 0:
            # prepare L8 thermal data
            therm_b10 = self._prepare_thermal_data()
            smfeat, smts = self.compute_sm(s2_rsi, ts,
                                           therm_b10, chunk_size)
            # merge with other features
            if features_20m is None:
                features_20m = smfeat
            else:
                features_20m = features_20m.merge(smfeat)

        # now actual ET if needed
        etcheck = [i for i in ['et', 'prdef']
                   if i in self.rsis[resolution]]
        if len(etcheck) > 0:
            # resample et0 to 20m
            et0ts = self._upsample_meteo_ts(ts_rsi, bands=['et0'])
            # compute ET
            etfeat, etts = self.compute_et(et0ts, s2_rsi,
                                           chunk_size)
            # merge with other features
            if features_20m is None:
                features_20m = etfeat
            else:
                features_20m = features_20m.merge(etfeat)

            if 'prdef' in etcheck:
                # if precip not ready yet, prepare
                if precip is None:
                    precip = self._upsample_meteo_ts(
                        ts_proc, bands=['precipitation_flux'])
                # check which soil moisture feature
                # should be used for correction
                corrections = settings['prdef'].get('corr', None)
                if corrections is None:
                    logger.warning('No soil moisture correction applied to'
                                   ' precipitation deficit!')
                    sm_cor = {'Nocor': None}
                else:
                    sm_cor = {}
                    if 'ssm' in corrections:
                        sm_cor['Ssm'] = ssmts
                    if 'sm' in corrections:
                        sm_cor['Sm'] = smts
                    if 'None' in corrections:
                        sm_cor['Nocor'] = None

                # compute precipitation deficit
                if features_20m is not None:
                    features_20m = features_20m.merge(
                        self.compute_prdef(etts, precip,
                                           sm_cor, chunk_size))
                else:
                    features_20m = self.compute_prdef(etts, precip,
                                                      sm_cor, chunk_size)

        # stop timer
        timer.features[resolution].stop()

        return features_20m

    def _upsample_meteo_features(self, features):

        try:
            required_xdim = int((self.collection.bounds[2] -
                                 self.collection.bounds[0]) / 20)
            required_ydim = int((self.collection.bounds[3] -
                                 self.collection.bounds[1]) / 20)

            while ((features.shape[1] < required_ydim)
                    or (features.shape[2] < required_xdim)):
                features = features.upsample()
            # Now cut to exact shape
            features.data = features.data[:,
                                          :required_ydim,
                                          :required_xdim]

        except AttributeError:
            # Suppose a TrainingCollection
            # Get the bounds of the sample
            bounds = self.bounds
            # Get desired output shape
            outdim = int((bounds[2] - bounds[0]) / 20)
            # Nr of required upsamplings
            upsampling = int(np.log2(outdim))
            # Resize the AgERA5 data
            features = features.upsample(times=upsampling)

        return features

    def compute_features(self,
                         chunk_size=None,
                         upsample_order=1,
                         **kwargs):

        timer = self.timer

        # 100m processing
        resolution = 100
        ts = None
        ts_proc = None
        ts = self.load_data(resolution)

        if ts is not None:
            features, ts_proc, ts_rsi = self.compute_features_100m(ts,
                                                                          chunk_size)  # NOQA

            # upsample features to 20m
            features = self._upsample_meteo_features(features)
        else:
            features = None

        # start 20m processing
        resolution = 20
        rsis = self.rsis[resolution]
        if len(rsis) > 0:
            if features is not None:
                features = features.merge(
                    self.compute_features_20m(ts, ts_proc,
                                              ts_rsi,
                                              chunk_size)
                )

            else:
                features = self.compute_features_20m(ts, ts_proc,
                                                     ts_rsi,
                                                     chunk_size)

        for r in [20, 100]:
            timer.load[r].log()
            timer.composite[r].log()
            timer.features[r].log()
            timer.rsi[r].log()

        # Now finally, we go to 10m
        features = features.upsample(order=upsample_order)

        return features

    @property
    def bounds(self):
        if hasattr(self.collection, 'bounds'):
            return self.collection.bounds
        else:
            bounds = self.collection.df.bounds.values[0]
            if type(bounds) == str:
                bounds = eval(bounds)
        return bounds


class TSAgERA5FeaturesProcessor(AgERA5FeaturesProcessor):

    @ property
    def _optical_fp(self):
        return TSL2AFeaturesProcessor

    @ property
    def _thermal_fp(self):
        return L8ThermalFeaturesProcessor


class WorldCerealAgERA5FeaturesProcessor(AgERA5FeaturesProcessor):

    @ property
    def _optical_fp(self):
        return WorldCerealOpticalFeaturesProcessor

    @ property
    def _thermal_fp(self):
        return WorldCerealThermalFeaturesProcessor
