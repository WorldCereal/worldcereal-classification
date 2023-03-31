import atexit
import copy
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
import json
import gc
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, List

import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from satio.features import Features
from satio.geoloader import load_reproject
from satio.utils import run_parallel, TaskTimer, rasterize
from satio.utils.errors import EmptyCollection
from satio.utils.geotiff import write_geotiff, get_rasterio_profile
from shapely.geometry import Polygon
from skimage.segmentation import felzenszwalb
import tensorflow as tf
import zarr

from worldcereal import (GDDTLIMIT, SEASON_POST_BUFFER, SEASON_PRIOR_BUFFER,
                         SUPPORTED_SEASONS, TBASE)
from worldcereal.classification.models import WorldCerealModel
from worldcereal.clf import WorldCerealClassifier, majority_filter, mask
from worldcereal.collections import WorldCerealProductCollection
from worldcereal.features.settings import (get_cropland_features_meta,
                                           get_default_ignore_def_feat,
                                           get_default_rsi_meta)
from worldcereal.fp import (AgERA5FeaturesProcessor, L2AFeaturesProcessor,
                            L8ThermalFeaturesProcessor, SARFeaturesProcessor,
                            WorldCerealAgERA5FeaturesProcessor,
                            WorldCerealOpticalFeaturesProcessor,
                            WorldCerealSARFeaturesProcessor,
                            WorldCerealThermalFeaturesProcessor)
from worldcereal.gdd import GDDcomputer
from worldcereal.postprocess import COLORMAP, NODATA
from worldcereal.resources import biomes
from worldcereal.seasons import infer_season_dates, season_doys_to_dates
from worldcereal.utils import (AUXDATA_PATH, BIOME_RASTERS, get_coll_maxgap,
                               REALM_RASTERS, BlockTooColdError,
                               conflicts, io, is_real_feature)
from worldcereal.utils.aez import group_from_id
from worldcereal.utils.filterfunctions import irr_max_ndvi_filter
from worldcereal.utils.io import (check_collection, load_features_geotiff,
                                  save_features_geotiff)
from worldcereal.utils.scalers import minmaxscaler


def _clean(path):
    """Helper function to cleanup a path

    Args:
        path (str): path to file or directory to remove
    """
    if Path(path).is_file():
        Path(path).unlink()
    elif Path(path).is_dir():
        shutil.rmtree(path)


class BlockProcessor:

    supported_sources = ['OPTICAL',
                         'SAR',
                         'TIR',
                         'METEO',
                         'DEM',
                         'WorldCover'
                         ]

    def __init__(self,
                 collections_dict,
                 settings,
                 rsi_meta,
                 features_meta,
                 ignore_def_feat,
                 bounds,
                 epsg,
                 custom_fps: Dict = None,
                 featresolution: int = 10,
                 avg_segm: bool = False,
                 augment: bool = False,
                 segm_feat: List = None,
                 aez_id: int = None):

        self.collections = collections_dict
        self.bounds = bounds
        self.epsg = epsg
        self.aez_id = aez_id

        self._feat_handles = custom_fps or {
            'OPTICAL': L2AFeaturesProcessor,
            'SAR': SARFeaturesProcessor,
            'TIR': L8ThermalFeaturesProcessor,
            'METEO': AgERA5FeaturesProcessor,
            'DEM': Features.from_dem,
            'WorldCover': Features.from_worldcover
        }

        self.settings = settings
        self.rsi_meta = rsi_meta
        self.features_meta = features_meta
        self.ignore_def_feat = ignore_def_feat

        self.featresolution = featresolution
        self.avg_segm = avg_segm
        self.segm_feat = segm_feat
        self.augment = augment

        self._check_sources()

    def _check_sources(self):

        for s in self.collections.keys():
            if s not in self.supported_sources:
                raise ValueError(f"{s} is not a supported source. "
                                 "Supported sources: "
                                 f"'{self.supported_sources}'.")

    def add_raster_feat(self,
                        feat: Features,
                        feat_name: str,
                        raster_path: str,
                        resolution=10):
        """
        Add features from global latlon rasters/vrts
        """
        # we use border_buff to avoid blocks artefact on meteo data

        def _read_file(raster_file):
            arr = load_reproject(
                raster_file, self.bounds, self.epsg,
                resolution=resolution, border_buff=resolution,
                fill_value=0,
                resampling=Resampling.bilinear)

            if 'realm' in feat_name:
                arr = (arr > 0).astype(np.uint8)

            return feat.add(arr, [feat_name])

        if AUXDATA_PATH is not None:
            raster_file = Path(AUXDATA_PATH) / 'biomes' / raster_path
            if raster_file.is_file():
                logger.info(f'Loading biome data from: {raster_file}')
                return _read_file(raster_file)
            else:
                if 'realm' not in feat_name:
                    logger.warning((f'No {feat_name} file found '
                                    f'in auxdata folder `{AUXDATA_PATH}`. '
                                    'Using simplified version from package!'))
        else:
            if 'realm' not in feat_name:
                logger.warning(('No auxdata folder specified. Using '
                                f'simplified {feat_name} version '
                                'from package!'))

        with pkg_resources.open_binary(biomes, raster_path) as raster_file:
            return _read_file(raster_file)

    def get_features(self) -> Features:

        collections = self.collections
        feat_handles = self._feat_handles
        settings = self.settings
        rsi_meta = self.rsi_meta
        features_meta = self.features_meta
        ignore_def_feat = self.ignore_def_feat

        sources = [s for s in collections.keys()
                   if s not in ['LABELS', 'DEM',
                                'WorldCover']]

        # now run all other feature computations
        features = []

        # add METEOcol to SAR settings if available and needed
        if settings.get('SAR', None):
            if settings['SAR'].get('mask', None):
                if settings['SAR']['mask'].get('METEOcol', None) == '':
                    if 'METEO' in sources:
                        settings['SAR']['mask'][
                            'METEOcol'] = collections['METEO']
                    else:
                        raise ValueError('When masking of SAR is required,'
                                         ' you need to specify a METEO '
                                         'collection')

        # add dem, bounds and epsg to METEO settings if available and needed
        if settings.get('METEO', None):
            if settings['METEO'].get('rsis', None):
                etrsis = ['et0', 'et', 'prdef']
                etcheck = [i for i in etrsis
                           if i in settings['METEO']['rsis']]
                if len(etcheck) > 0:
                    if 'DEM' in collections.keys():
                        settings['METEO']['demcol'] = collections['DEM']
                        settings['METEO']['bounds'] = self.bounds
                        settings['METEO']['epsg'] = self.epsg
                    else:
                        raise ValueError(
                            'When ET0, ET or prdef needs to be computed,'
                            'a DEM collection is required!')

        # add OPTICAL and TIR collections to METEO settings
        # if ET or prdef required
        if settings.get('METEO', None):
            if settings['METEO'].get('rsis', None):
                etrsis = ['et', 'prdef', 'sm', 'smstress',
                          'ssm', 'ssm_adj']
                etcheck = [i for i in etrsis
                           if i in settings['METEO']['rsis']]
                if len(etcheck) > 0:
                    if 'OPTICAL' in collections.keys():
                        settings[
                            'METEO']['optical_col'] = collections['OPTICAL']
                    else:
                        raise ValueError(
                            'When ET, SM or prdef needs to be computed,'
                            'an OPTICAL collection is required!')

                    # if SM, you also need tir collection
                    if ('sm' in etcheck) or ('smstress' in etcheck):
                        if 'TIR' not in collections.keys():
                            raise ValueError('Need TIR collection for '
                                             ' computation of SM!')
                        else:
                            settings['METEO']['tir_col'] = collections['TIR']

        # add METEO collection to TIR settings if lst_ta required
        if settings.get('TIR', None):
            if settings['TIR'].get('rsis', None):
                if 'lst_ta' in settings['TIR']['rsis']:
                    if 'METEO' in collections.keys():
                        settings['TIR']['METEOcol'] = collections['METEO']
                    else:
                        raise ValueError('lst_ta feature required -> '
                                         'provide METEO collection!')

        # Compute accumulated GDD and add to settings if needed
        accumulated_gdd = None
        for s in sources:
            if 'normalize_gdd' in settings.get(s, {}).keys():
                if accumulated_gdd is None:
                    if 'METEO' not in collections:
                        raise RuntimeError(('An `METEO` collection '
                                            'is required when GDD '
                                            'normalization is requested.'))
                    # Compute accumulated GDD
                    tbase = settings[s]['normalize_gdd']['tbase']
                    tlimit = settings[s]['normalize_gdd']['tlimit']
                    season = settings[s]['normalize_gdd']['season']
                    start_date = settings[s]['composite']['start']
                    end_date = settings[s]['composite']['end']
                    gddcomp = GDDcomputer(collections['METEO'],
                                          tbase=tbase,
                                          upper_limit=tlimit,
                                          bounds=self.bounds,
                                          epsg=self.epsg,
                                          start_date=start_date,
                                          end_date=end_date,
                                          aez_id=self.aez_id)
                    accumulated_gdd = gddcomp.compute_accumulated_gdd(
                        season=season
                    )
                else:
                    if settings[s]['normalize_gdd']['tbase'] != tbase:
                        raise RuntimeError(
                            (f'The `Tbase` value of {s} '
                             f'is different from the Tbase value {tbase}'
                             'that was used for another source. These '
                             'should be equal.'))

                # Add accumulated gdd to settings
                settings[s]['normalize_gdd'][
                    'accumulated_gdd'] = accumulated_gdd

        for s in sources:
            source_settings = settings.get(s, {})
            if source_settings:
                feat = feat_handles[s](
                    collections[s],
                    source_settings,
                    rsi_meta=rsi_meta.get(s, {}),
                    features_meta=features_meta.get(s,
                                                    {}),
                    ignore_def_features=ignore_def_feat.get(
                        s, False)).compute_features(augment=self.augment)

                # append source to feature names
                featnames = feat.names
                featnames = [s + '-' + f for f in featnames]
                feat.names = featnames
                features.append(feat)

        if 'DEM' in collections.keys():
            feat = feat_handles['DEM'](
                collections['DEM'],
                settings.get('dem', {}),
                resolution=self.featresolution)
            features.append(feat)

        if 'WorldCover' in collections.keys():
            feat = feat_handles['WorldCover'](
                collections['WorldCover'],
                resolution=self.featresolution)
            features.append(feat)

        features = Features.from_features(*features)

        if self.avg_segm:
            logger.info('Running segmentation...')
            # select the features for segmentation
            if self.segm_feat is None:
                raise ValueError('Need list of features to '
                                 'perform the segmentation on!')
            elif len(self.segm_feat) < 3:
                raise ValueError('You need at least 3 features'
                                 ' to perform segmentation.')
            sfeat = features.select(self.segm_feat)
            # scale the features
            scaledFeat = []
            for feat in sfeat.names:
                f = sfeat.select([feat]).data
                scaledData = minmaxscaler(f, feat)
                scaledFeat.append(Features(scaledData, [feat]))
            scaledFeat = Features.from_features(*scaledFeat)
            # replace nan's by zeros
            scaledFeat.data[np.isnan(scaledFeat.data)] = 0
            # run PCA if more than 3 features selected
            if len(sfeat.names) > 3:
                scaledFeat = scaledFeat.pca(num_components=3)
            # transform data into right shape
            inputs = np.moveaxis(scaledFeat.data, 0, -1)
            # apply segmentation algorithms
            segments_fz = felzenszwalb(inputs, scale=1, sigma=0.5,
                                       min_size=10)
            logger.info('Segmentation done!')
            logger.info('Computing average features per segment')
            newdata = features.data.copy()
            for segm in np.unique(segments_fz):
                msk = segments_fz == segm
                newdata[:, msk] = np.repeat(np.nanmean(
                    features.data[:, msk], axis=1, keepdims=True),
                    msk.sum(), axis=1)
            features.data = newdata
            logger.info('Done!')

        # Add lat/lon features
        features = features.add_latlon(self.bounds,
                                       self.epsg,
                                       resolution=self.featresolution)

        # Add pixelids
        features = features.add_pixelids()

        # Add biomes features
        for feat_name, raster_path in BIOME_RASTERS.items():
            features = self.add_raster_feat(
                features, feat_name, raster_path,
                resolution=self.featresolution)

        # Add realms features
        for feat_name, raster_path in REALM_RASTERS.items():
            features = self.add_raster_feat(
                features, feat_name, raster_path)

        return features


class TrainingBlockProcessor(BlockProcessor):

    supported_sources = ['OPTICAL',
                         'SAR',
                         'DEM',
                         'WorldCover',
                         'METEO',
                         'LABELS',
                         'TIR']

    def __init__(self,
                 collections_dict,
                 settings,
                 rsi_meta,
                 features_meta,
                 ignore_def_feat,
                 bounds,
                 epsg,
                 location_id,
                 ref_id,
                 start_date,
                 end_date,
                 **kwargs):

        super().__init__(collections_dict,
                         settings,
                         rsi_meta,
                         features_meta,
                         ignore_def_feat,
                         bounds,
                         epsg,
                         **kwargs)

        self.location_id = location_id
        self.ref_id = ref_id
        self.start_date = start_date
        self.end_date = end_date

    def _get_labels(self):

        location_id = self.location_id

        labels_collection = self.collections['LABELS']
        labels = labels_collection.filter_location(location_id)

        # Load the labels
        labelsdata = labels.load()

        labelfeatures = Features(data=labelsdata['CT'], names=[
            'CT'], dtype='uint16')
        labelfeatures = labelfeatures.merge(Features(data=labelsdata['LC'],
                                                     names=['LC'],
                                                     dtype='uint16'),
                                            Features(data=labelsdata['IRR'],
                                                     names=['IRR'],
                                                     dtype='uint16'))

        if self.featresolution == 20:
            labelfeatures = labelfeatures.downsample_categorical()

        return labelfeatures

    def get_features(self):

        start_date = self.start_date
        end_date = self.end_date
        settings = self.settings
        features_meta = self.features_meta

        # Now make sure start_date and end_date are set in the settings
        # of all collections for this sample
        for coll in settings.keys():
            settings[coll]['composite']['start'] = start_date
            settings[coll]['composite']['end'] = end_date

        if 'sen2agri_temp_feat' in features_meta.get('OPTICAL', {}):
            features_meta['OPTICAL'][
                'sen2agri_temp_feat'][
                    'parameters']['time_start'] = start_date

        if 'sumdiv' in features_meta.get('METEO', {}):
            length_season = (pd.to_datetime(end_date) -
                             pd.to_datetime(start_date))
            features_meta['METEO']['sumdiv'][
                'parameters']['div'] = length_season.days

        features = super().get_features()

        # add labels
        features = features.merge(self._get_labels())

        # add location_id
        features = features.add_attribute(self.location_id,
                                          'location_id')
        features = features.add_attribute(self.ref_id, 'ref_id')

        # add Potapov crop layer
        try:
            base_path = Path('/vitodata/vegteam/landcover_products/')
            raster_file = (base_path / 'global_cropland_expansion' /
                           'global_cropland_extension_2019_tiled.vrt')
            arr = load_reproject(raster_file, self.bounds, self.epsg,
                                 resolution=10, fill_value=0,
                                 resampling=Resampling.nearest)
            features = features.add(arr, ['POTAPOV-LABEL-10m'])
        except Exception:
            logger.warning(('Could not load global cropland '
                            'expansion product: skipping feature!'))

        # add the split category
        # check which source is present
        if 'OPTICAL' in self.collections.keys():
            source = 'OPTICAL'
        elif 'SAR' in self.collections.keys():
            source = 'SAR'
        elif 'TIR' in self.collections.keys():
            source = 'TIR'
        else:
            source = 'METEO'
        features = features.add_attribute(
            self.collections[source].df['split'].values[0],
            'split')

        return features


class TrainingChain:
    """
    Trains a model based on the training collections and settings provided.

    Attributes
    ----------
    collections_dict : dict
        Supported keys: ['OPTICAL', 'GAMMA0', 'SAR', 'METEO', 'DEM', 'LABELS'].
        'OPTICAL' and 'LABELS' are required and should be instances of
        L2ATrainingCollection and PatchLabelsTrainingCollection respectively.

    settings : Dict
        Specifies parameters for the features computation.
        Dictionary with the same keys as `collections_dict` except for
        'LABELS'. It specifies different FeaturesProcessor parameters, as
        well as providing features functions and features names.

    rsi_meta : Dict
        Optionally provides information on how to calculate particular RSIs.
        Dictionary with same keys as "collections_dict". If a certain key
        is not provided, then the default in satio will be
        selected for that source.

    features_meta : Dict
        Optionally provides information on how to calculate certain features.
        Dictionary with same keys as "collections_dict". If a certain key
        is not provided, then the default in satio will be selected for that
        source.

    location_ids : List
        List of location_ids from which to build the training dataframe

    geometry : Polygon
        shapely polygon in lat lon to retrieve location_ids.

    trainingdb : geopandas.GeoDataFrame
        gdf of the training locations with required attributes

    aez : List
        should be a list of tuples (geodataframe, name, value_column).
        If value_column is None, the default will be the 'Index'. These
        layers will be added as a rasterized feature and one hot encoded.

    season: str
        specification of the season for which this chain is valid.
        Currently one of [WW, M1, M2, annual, custom].

    start_date: str (yyyy-mm-dd)
        an optional custom start date for which to run the chain.
        Is only taken into account for season `custom`

    end_date: str (yyyy-mm-dd)
        an optional custom end date for which to run the chain.
        Is only taken into account for season `custom`

    prior_buffer: int
        an optional buffer (days) to subtract from season start.
        A negative number means the start will be postponed.

    post_buffer: int
        an optional buffer (days) to add to the season end.
        A negative number means the end will be advanced.

    avg_segm: boolean
        whether or not to compute the average of features per segment

    segm_feat: list
        list of features to be used for segmentation

    featresolution: int (optional)
        resolution of computed features (default = 10)

    """

    def __init__(self,
                 collections_dict: Dict,
                 settings: Dict,
                 rsi_meta: Dict = None,
                 features_meta: Dict = None,
                 ignore_def_feat: Dict = None,
                 location_ids: List = None,
                 geometry: Polygon = None,
                 trainingdb: gpd.GeoDataFrame = None,
                 aez: gpd.GeoDataFrame = None,
                 basedir: str = None,
                 model: WorldCerealModel = None,
                 season: str = 'custom',
                 start_date: str = None,
                 end_date: str = None,
                 prior_buffer: int = None,
                 post_buffer: int = None,
                 avg_segm: bool = False,
                 segm_feat: List = None,
                 featresolution: int = 10):

        # -------------------------------------------------------------
        # FIRST DO THE NECESSARY CHECKS IF WE HAVE EVERYTHING WE NEED

        if season not in SUPPORTED_SEASONS:
            raise ValueError(f'Unrecognized `season` parameter: {season}')

        if season != 'custom' and (
                start_date is not None or end_date is not None):
            logger.warning((f'Using season `{season}`. '
                            'start_date and end_date arguments are ignored'))

        if season == 'custom' and (start_date is None or end_date is None):
            raise ValueError(('When using `custom` season, start_date and '
                              'end_date arguments are required'))

        if (location_ids is None) & (geometry is None):
            raise ValueError("One argument between `location_ids` "
                             "and `geometry` should be provided. If both are "
                             "provided `geometry` is ignored.")

        if trainingdb is None:
            raise ValueError('`trainingdb` cannot be None')

        # -------------------------------------------------------------

        self.collections = collections_dict
        self.settings = settings
        self.rsi_meta = rsi_meta or get_default_rsi_meta()
        self.features_meta = features_meta or get_cropland_features_meta()
        self.ignore_def_feat = ignore_def_feat or get_default_ignore_def_feat()
        self.season = season
        self.start_date = start_date
        self.end_date = end_date
        self.prior_buffer = prior_buffer or SEASON_PRIOR_BUFFER[season]
        self.post_buffer = post_buffer or SEASON_POST_BUFFER[season]
        self.trainingdb = trainingdb
        self.location_ids = (location_ids if location_ids is not None
                             else self._get_geometry_locations(geometry))
        self.aez = aez

        self.avg_segm = avg_segm
        self.segm_feat = segm_feat

        self._basedir = basedir
        if basedir is not None:
            os.makedirs(basedir, exist_ok=True)
            logger.add(Path(basedir) / 'main.log', mode='w')

        if model is not None:
            if not isinstance(model, WorldCerealModel):
                raise ValueError(('`model` should be instance '
                                  'of WorldCerealModel but got: '
                                  f'{type(model)}.'))
            self.model = model

        self.featresolution = featresolution
        self.timer = _ChainTimer()

    def _get_geometry_locations(self, geometry):
        trainingdb = self.trainingdb
        location_ids = trainingdb[trainingdb.intersects(
            geometry)].location_id.values
        return location_ids

    def _filter_collections(self, location_id, bounds=None, epsg=None):

        if (bounds is None) | (epsg is None):
            bounds, epsg = self._get_location_bounds(location_id)

        new_collections = self.collections.copy()
        for s in self.collections.keys():
            if s in ['OPTICAL', 'SAR', 'TIR', 'METEO']:
                try:
                    new_collections[s] = (new_collections[s]
                                          .filter_location(location_id))
                except EmptyCollection:
                    _ = new_collections.pop(s)

            elif s in ['DEM', 'WorldCover']:
                new_collections[s] = (new_collections[s]
                                      .filter_bounds(bounds, epsg))
            else:
                pass

        return new_collections

    def _get_location_features(self, location_id, custom_fps=None,
                               augment=False):

        bounds, epsg = self._get_location_bounds(location_id)
        collections = self._filter_collections(location_id, bounds, epsg)
        logger.debug(f"Processing location_id {location_id}")

        # ----------------------------------------------
        # INFER START DATE AND END DATE FOR THIS SAMPLE
        # ----------------------------------------------

        sample = self.trainingdb.set_index('location_id').loc[location_id]
        ref_id = sample['ref_id']

        # Sample validity time
        valid_date = sample.validityTi

        if self.season == 'custom':

            sos = int(pd.to_datetime(self.start_date).strftime('%j'))
            eos = int(pd.to_datetime(self.end_date).strftime('%j'))

            start_date, end_date = season_doys_to_dates(
                sos, eos, valid_date, allow_outside=True)

        else:
            logger.info(('Retrieving processing period from '
                         'pixel-based crop calendars...'))
            start_date, end_date = infer_season_dates(self.season,
                                                      bounds, epsg,
                                                      valid_date)

        # Subtract optional buffer prior to season start
        if self.prior_buffer != 0:
            logger.warning(('Applying prior buffer of: '
                            f'{self.prior_buffer} days'))
            start_date = (pd.to_datetime(start_date) -
                          pd.Timedelta(
                days=self.prior_buffer)).strftime('%Y-%m-%d')

        # Add optional buffer after season end
        if self.post_buffer != 0:
            logger.warning(('Applying post buffer of: '
                            f'{self.post_buffer} days'))
            end_date = (pd.to_datetime(end_date) +
                        pd.Timedelta(
                days=self.post_buffer)).strftime('%Y-%m-%d')

        # We have our start_date and end_date; now check
        # if this range is covered by the sample
        adjusted = False
        max_diff = 90  # max 3 months difference allowed
        if pd.to_datetime(sample['start_date']) > pd.to_datetime(start_date):
            if self.season == 'tc-annual':
                adjusted = True
                # Temporary (?) fix for annual cropland
                logger.warning(('Default annual cropland range not covered'
                                ' by sample. Adjusting to sample range!'))
                # start_date = _random_start_date(valid_date,
                #                                 sample['start_date'])
                new_start_date = pd.to_datetime(sample['start_date'])
                if ((new_start_date - pd.to_datetime(start_date)) >
                        pd.Timedelta(days=max_diff)):
                    raise ValueError((f'Needed start_date ({start_date}) '
                                      f' is more than {max_diff} days '
                                      'before sample start_date '
                                      f'({sample["start_date"]})!'))
                else:
                    start_date = new_start_date

                end_date = start_date + pd.Timedelta(days=364)
                start_date = start_date.strftime('%Y-%m-%d')
                end_date = end_date.strftime('%Y-%m-%d')
            else:
                raise ValueError((f'Needed start_date ({start_date}) '
                                  ' is before sample start_date '
                                  f'({sample["start_date"]})!'))
        if pd.to_datetime(sample['end_date']) < pd.to_datetime(end_date):
            if self.season == 'tc-annual':
                # Temporary (?) fix for annual cropland
                logger.warning(('Default annual cropland range not covered'
                                ' by sample. Adjusting to sample range!'))
                # end_date = _random_end_date(valid_date,
                #                             sample['end_date'])
                new_end_date = pd.to_datetime(sample['end_date'])
                if ((pd.to_datetime(end_date) - new_end_date) >
                        pd.Timedelta(days=max_diff)):
                    raise ValueError((f'Needed end_date ({end_date}) '
                                      f' is more than {max_diff} days '
                                      'after sample end_date '
                                      f'({sample["end_date"]})!'))
                else:
                    end_date = new_end_date

                start_date = end_date - pd.Timedelta(days=364)
                start_date = start_date.strftime('%Y-%m-%d')
                end_date = end_date.strftime('%Y-%m-%d')
                adjusted = True
            else:
                raise ValueError((f'Needed end_date ({end_date}) '
                                  ' is after sample end_date '
                                  f'({sample["end_date"]})!'))
        if (adjusted and pd.to_datetime(sample['start_date'])
                > pd.to_datetime(start_date)):
            raise ValueError((f'Adjusted start_date ({start_date}) '
                              ' is before sample start_date '
                              f'({sample["start_date"]})!'))

        logger.debug(('Start and end of season to be used for this sample: '
                      f'{start_date} - {end_date}'))

        # -----------------------------------------------

        settings = copy.deepcopy(self.settings)
        rsi_meta = self.rsi_meta
        features_meta = self.features_meta
        ignore_def_feat = self.ignore_def_feat
        featresolution = self.featresolution

        train_processor = TrainingBlockProcessor(
            collections,
            settings,
            rsi_meta,
            features_meta,
            ignore_def_feat,
            bounds,
            epsg,
            location_id,
            ref_id,
            start_date,
            end_date,
            custom_fps=custom_fps,
            featresolution=featresolution,
            augment=augment,
            avg_segm=self.avg_segm,
            segm_feat=self.segm_feat)

        loc_feat = train_processor.get_features()

        if self.aez is not None:
            groupid = int(
                rasterize(self.aez, bounds, epsg,
                          resolution=int(bounds[2] - bounds[0]),  # One pixel in patch  # NOQA
                          value_column='groupID',
                          fill_value=-9999,
                          dtype=np.int32).ravel()[0])
            zoneid = int(
                rasterize(self.aez, bounds, epsg,
                          resolution=int(bounds[2] - bounds[0]),  # One pixel in patch  # NOQA
                          value_column='zoneID',
                          fill_value=-9999,
                          dtype=np.int32).ravel()[0])

            loc_feat = loc_feat.add_attribute(groupid,
                                              'aez_groupid')
            loc_feat = loc_feat.add_attribute(zoneid,
                                              'aez_zoneid')
        else:
            loc_feat = loc_feat.add_attribute(None, 'aez_groupid')
            loc_feat = loc_feat.add_attribute(None, 'aez_zoneid')

        return loc_feat

    def _get_location_bounds(self, location_id):
        trainingdb = self.trainingdb
        trainingdb_row = trainingdb[
            trainingdb.location_id == location_id].iloc[0]
        epsg = int(trainingdb_row['epsg'])
        bounds = trainingdb_row['bounds']

        if isinstance(bounds, str):
            bounds = eval(bounds)
        bounds = np.array(bounds)

        return bounds, epsg

    def _get_feat(self, location_id, ft_selection=None, custom_fps=None,
                  augment=False):
        try:
            features = self._get_location_features(location_id,
                                                   custom_fps=custom_fps,
                                                   augment=augment)

            if ft_selection is not None:
                if type(ft_selection) is not list:
                    raise ValueError('`ft_selection` should be a list!')
                features = features.select(ft_selection)

        except Exception as e:
            # raise e
            features = None
            logger.error(f"Error processing location_id = {location_id}: "
                         f"{e}.")
        return features

    def select_fts_with_meta(self, features: Features, ft_selection):
        if type(ft_selection) is not list:
            raise ValueError('`ft_selection` should be a list!')
        all_fts = features.names
        real_fts = [ft for ft in all_fts if self.is_real_feature(ft)]
        real_selected_fts = [ft for ft in real_fts if ft in ft_selection]
        meta_fts = list(set(all_fts) - set(real_fts))
        selected_fts = real_selected_fts + meta_fts

        return features.select(selected_fts)

    @ staticmethod
    def is_real_feature(ft_name):
        return is_real_feature(ft_name)

    def save_settings(self, outdir=None):

        def _make_serial(d):
            if isinstance(d, dict):
                return {_make_serial(k): _make_serial(v)
                        for k, v in d.items()}
            elif isinstance(d, list):
                return [_make_serial(i) for i in d]
            elif callable(d):
                return d.__name__
            else:
                return d

        outdir = outdir or self._basedir

        logger.info(f'Saving settings to: {outdir}')

        with open(Path(outdir) / 'settings.json', 'w') as f:
            json.dump(_make_serial(self.settings), f)
        with open(Path(outdir) / 'rsi_meta.json', 'w') as f:
            json.dump(_make_serial(self.rsi_meta), f)
        with open(Path(outdir) / 'features_meta.json', 'w') as f:
            json.dump(_make_serial(self.features_meta), f)
        with open(Path(outdir) / 'ignore_def_feat.json', 'w') as f:
            json.dump(_make_serial(self.ignore_def_feat), f)

    def train(self):
        raise NotImplementedError

    def predict(self, location_id):
        if self.model is None:
            raise ValueError(('No model assigned to '
                              'this chain.'))

        self.clf = WorldCerealClassifier(self.model)

        # Compute the features
        starttime = time.time()
        features = self._get_feat(location_id)
        time_feat = time.time() - starttime

        if features is None:
            return None, None, None
        else:
            # Make the prediction and return together with
            # the features
            starttime = time.time()
            prediction, confidence = self.clf.predict(features)
            time_pred = time.time() - starttime
            return prediction, confidence, features, (time_feat, time_pred)


class PatchTrainingChain(TrainingChain):

    def __init__(self, windowsize=64, **kwargs):

        self.chaintype = 'patch'
        self.windowsize = windowsize
        super().__init__(**kwargs)

    def get_training_patches(self,
                             sparkcontext=None,
                             ft_selection=None,
                             **kwargs
                             ):

        logger.info((f'Processing {len(self.location_ids)} samples ...'))

        # Save the current settings
        self.save_settings()

        if sparkcontext is None:
            df = self._get_training_patches_local(
                ft_selection=ft_selection,
                **kwargs)
            return df
        else:
            self._get_training_patches_spark(sparkcontext,
                                             ft_selection=ft_selection,
                                             **kwargs)

    def _get_training_patches_spark(self,
                                    sparkcontext,
                                    ft_selection=None,
                                    outputbasename='inputpatches',
                                    outputformat='tfrecord',
                                    nrparts=None):
        '''
        Method to get and save patch features on spark
        '''

        if self._basedir is None:
            raise ValueError(('Cannot run export job without '
                              'specifying `basedir`.'))

        if outputformat not in ['tfrecord', 'parquet']:
            raise ValueError(f'Unrecognized outputformat: `{outputformat}`')

        outdir = Path(self._basedir)

        logger.info('Running jobs on spark ...')
        rdd = sparkcontext.parallelize(self.location_ids,
                                       len(self.location_ids)).map(
            lambda location_id: self._get_feat(
                location_id,
            )).filter(lambda ft: ft is not None)

        # Check if we actually have something left
        if rdd.isEmpty():
            logger.warning('No samples left in RDD -> aborting job')
            if (Path(outdir) / str(outputbasename)).is_dir():
                (Path(outdir) / str(outputbasename)).unlink()
            return
        # Convert features to spark SQL rows
        rdd = rdd.map(lambda ft: self._convert_ft_to_row(
            ft,
            windowsize=self.windowsize,
            ft_selection=ft_selection,
            on_spark=True))

        # Convert to dataframe
        output_df = rdd.toDF()

        # Get feature names in the rdd
        present_fts = output_df.schema.names
        present_fts_file = (Path(self._basedir) /
                            'present_features.txt')
        if present_fts_file.is_file():
            present_fts_file.unlink()
        with open(present_fts_file, 'w') as f:
            for ft in present_fts:
                f.write(str(ft) + '\n')

        # Amount of output files
        nrparts = nrparts or int(output_df.count() / 500 + 1)

        if outputformat == 'tfrecord':
            self._rdd_to_tfrecords(output_df, nrparts,
                                   outdir, str(outputbasename))
        elif outputformat == 'parquet':
            self._rdd_to_parquet(output_df, nrparts,
                                 outdir, str(outputbasename))

    def _get_training_patches_local(self,
                                    ft_selection=None,
                                    max_workers=2,
                                    save=True,
                                    debug=False,
                                    **kwargs):

        def get_feat_df(location_id):
            try:
                features = self._get_location_features(location_id)

                self._convert_ft_to_row(features,
                                        windowsize=self.windowsize,
                                        ft_selection=ft_selection,
                                        on_spark=False,
                                        scalefactor=1)
            except Exception as e:
                if debug:
                    raise e
                logger.error(f"Error processing location_id = {location_id}: "
                             f"{e}.")
                return None

        results = run_parallel(get_feat_df,
                               self.location_ids,
                               max_workers=max_workers)

        train_df = pd.DataFrame(results)

        if save:

            # Get feature names in the rdd
            present_fts = train_df.columns.tolist()
            present_fts_file = (Path(self._basedir) /
                                'present_features.txt')
            if present_fts_file.is_file():
                present_fts_file.unlink()
            with open(present_fts_file, 'w') as f:
                for ft in present_fts:
                    f.write(str(ft) + '\n')

            train_df_file = (Path(self._basedir)
                             / 'training_df.csv')
            logger.info(f'Saving training df to: {train_df_file}')
            train_df.to_csv(train_df_file)

        return train_df

    def _convert_ft_to_row(self, ft, windowsize=64,
                           ft_selection=None, scalefactor=10000,
                           on_spark=True):
        '''
        helper function to transform the resulting features into a
        proper spark dataframe row for saving
        :param ft: features to save
        :param windowsize: the patch size that is being used
        :param scalefactor: integer factor used to scale the values
        :return: spark SQL row
        '''
        if on_spark:
            from pyspark.sql.types import Row

        location_id = ft.attrs['location_id']
        ref_id = ft.attrs['ref_id']
        split = ft.attrs['split']
        aez_groupid = ft.attrs['aez_groupid']
        aez_zoneid = ft.attrs['aez_zoneid']

        if ft_selection is not None:
            ft = self.select_fts_with_meta(ft,
                                           ft_selection)

        converted_ft = {}

        for ft_name in ft.names:
            # Get np.ndarray data of this feature
            ftdata = ft[ft_name]

            if on_spark:
                dtype = int
            else:
                dtype = ftdata.dtype

            converted_ft[ft_name] = (ftdata.reshape(
                (windowsize*windowsize)) * scalefactor).astype(
                    dtype).tolist()

        if on_spark:
            return Row(**converted_ft, location_id=location_id, split=split,
                       aez_groupid=aez_groupid, aez_zoneid=aez_zoneid,
                       ref_id=ref_id)
        else:
            return dict(**converted_ft, location_id=location_id, split=split,
                        aez_groupid=aez_groupid, aez_zoneid=aez_zoneid,
                        ref_id=ref_id)

    @ staticmethod
    def _rdd_to_tfrecords(sparkdf, nrparts, outdir, outname):
        logger.info('Saving results to tfrecords ...')
        sparkdf.repartition(nrparts).write.format('tfrecords')\
            .mode('overwrite').option('recordType', 'SequenceExample')\
            .option("codec", "org.apache.hadoop.io.compress.GzipCodec")\
            .save('file://' + os.path.join(outdir, outname))

    @ staticmethod
    def _rdd_to_parquet(sparkdf, nrparts, outdir, outname):
        logger.info('Saving results to parquet ...')
        sparkdf.repartition(nrparts).write.format('parquet')\
            .mode('overwrite')\
            .save('file://' + os.path.join(outdir, outname))


class PixelTrainingChain(TrainingChain):

    def __init__(self, **kwargs):

        self.chaintype = 'pixel'
        super().__init__(**kwargs)

    @ staticmethod
    def _get_df(features):
        if features is None:
            return None
        else:
            return features.df

    @ staticmethod
    def _identity(df):
        return df

    def get_training_df(self, sparkcontext=None, **kwargs):

        # timer = self.timer
        # timer.trainingdf.start()

        # Save the current settings
        self.save_settings()

        if sparkcontext is None:
            train_df = self._get_training_df_local(**kwargs)
        else:
            train_df = self._get_training_df_spark(sparkcontext,
                                                   **kwargs)
        # timer.trainingdf.stop()
        # if self.location_ids is not None:
        #     timer.trainingdf._total = (timer.trainingdf._total /
        #                                len(self.location_ids))
        # timer.trainingdf.log()

        return train_df

    def _get_training_df_local(self,
                               ft_selection=None,
                               label='LC',
                               max_pixels=1,
                               augment=False,
                               max_workers=2,
                               save=True,
                               debug=False,
                               seed=1234,
                               format='csv',
                               custom_fps=None,
                               filter_function=None):

        if format not in ['csv', 'parquet']:
            raise ValueError(f'Unsupported output format: `{format}`')

        if filter_function is None:
            filter_function = self._identity
        elif filter_function == 'irr_max_ndvi_filter':
            filter_function = irr_max_ndvi_filter
        else:
            raise ValueError('Unknown filter function requested!')

        def get_feat_df(location_id):
            try:
                features = self._get_location_features(location_id,
                                                       custom_fps=custom_fps,
                                                       augment=augment)

                if ft_selection is not None:
                    features = self.select_fts_with_meta(features,
                                                         ft_selection)

            except Exception as e:
                if debug:
                    raise e
                features = None
                logger.error(f"Error processing location_id = {location_id}: "
                             f"{e}.")
            return self.sample_pixels(
                filter_function(self._get_df(features)),
                nrpixels=max_pixels,
                label=label,
                seed=seed)

        results = run_parallel(get_feat_df,
                               self.location_ids,
                               max_workers=max_workers)
        results = [result for result in results if result is not None]

        train_df = pd.concat(results, ignore_index=True)
        train_df = train_df.rename(columns={label: 'OUTPUT'})

        if save:
            if format == 'csv':
                train_df_file = (Path(self._basedir)
                                 / f'training_df_{label}.csv')
                logger.info(f'Saving training df to: {train_df_file}')
                train_df.to_csv(train_df_file)
            if format == 'parquet':
                train_df_file = (Path(self._basedir)
                                 / f'training_df_{label}.parquet')
                logger.info(f'Saving training df to: {train_df_file}')
                train_df.to_parquet(train_df_file)
        return train_df

    def _get_training_df_spark(self,
                               sparkcontext,
                               ft_selection=None,
                               label='LC',
                               max_pixels=1,
                               augment=False,
                               save=True,
                               seed=1234,
                               format='csv',
                               custom_fps=None,
                               filter_function=None,
                               **kwargs
                               ):

        if format not in ['csv', 'parquet']:
            raise ValueError(f'Unsupported output format: `{format}`')

        if filter_function is None:
            filter_function = self._identity
        elif filter_function == 'irr_max_ndvi_filter':
            filter_function = irr_max_ndvi_filter
        else:
            raise ValueError('Unknown filter function requested!')

        logger.info('Running feature extraction on spark ...')
        rdd = sparkcontext.parallelize(self.location_ids,
                                       len(self.location_ids)
                                       ).persist().map(
            lambda location_id: self._get_feat(
                location_id, ft_selection=ft_selection,
                custom_fps=custom_fps,
                augment=augment
            )).filter(lambda t: t is not None).map(
                self._get_df).map(
                    lambda df: filter_function(df)).map(
                    lambda df: self.sample_pixels(
                        df, nrpixels=max_pixels,
                        label=label, seed=seed)).filter(
                            lambda t: t is not None)

        # Check if we actually have something left
        if rdd.isEmpty():
            logger.warning('No samples left in RDD -> aborting job')
            return

        dfs = rdd.collect()
        train_df = pd.concat(dfs, ignore_index=True)
        train_df = train_df.rename(columns={label: 'OUTPUT'})
        rdd.unpersist()

        if save:
            if format == 'csv':
                train_df_file = (Path(self._basedir)
                                 / f'training_df_{label}.csv')
                logger.info(f'Saving training df to: {train_df_file}')
                train_df.to_csv(train_df_file)
            if format == 'parquet':
                train_df_file = (Path(self._basedir)
                                 / f'training_df_{label}.parquet')
                logger.info(f'Saving training df to: {train_df_file}')
                train_df.to_parquet(train_df_file)

        return train_df

    def train_one_vs_all(self,
                         traindf,
                         targetlabels,
                         nodatalabel=0):
        """Method to train a binary pixel-based classifier
        on a training dataframe

        Args:
            traindf (pd.DataFrame or str): training dataframe or path to csv
            targetlabels (str or list): the target classes to identify
            nodatalabel (int, optional): Target label corresponding to unknown.
            Defaults to 0.

        Raises:
            ValueError: if no model has been assigned to the trainingchain
            ValueError: if training dataframe is not a dataframe or string
        """
        timer = self.timer

        if self.model is None:
            raise ValueError('No model assigned yet.')

        if type(traindf) is str:
            traindf = pd.read_csv(traindf, index_col=0)
        elif type(traindf) is pd.DataFrame:
            pass
        else:
            raise ValueError(('`traindf` should be either '
                              ' a path to a csv or a '
                              'pd.DataFrame instance. Got: '
                              f'{type(traindf)}'))

        inputfeatures = self.model.feature_names

        # Check if all required features are in the DF
        for ft in inputfeatures:
            if ft not in traindf.columns.tolist():
                logger.debug(f'traindf shape: {traindf.shape}')
                raise ValueError((f'Expected input feature `{ft}` '
                                  'not found in trainingDF.'))

        logger.info(('Start training on '
                     f'{len(inputfeatures)} features ...'))

        traindfcal = traindf[traindf['split'] == 'CAL']
        traindfval = traindf[traindf['split'] == 'VAL']

        inputscal = traindfcal[inputfeatures].values
        outputscal = traindfcal['OUTPUT'].values
        inputsval = traindfval[inputfeatures].values
        outputsval = traindfval['OUTPUT'].values

        inputsval, outputsval = io.drop_nan(
            *io.drop_unknown(inputsval, outputsval))
        inputscal, outputscal = io.drop_nan(
            *io.drop_unknown(inputscal, outputscal))

        outputscal = io.convert_to_binary(
            outputscal,
            target_labels=targetlabels)
        outputsval = io.convert_to_binary(
            outputsval,
            target_labels=targetlabels)

        timer.trainModel.start()
        self.model.train(inputscal, outputscal)
        timer.trainModel.stop()
        timer.evalModel.start()
        self.model.evaluate(inputsval, outputsval)
        timer.evalModel.stop()
        modelfile = (Path(self._basedir) /
                     self.model.modeltype)
        self.model.save(modelfile)

        timer.trainModel.log()
        timer.evalModel.log()

    @ staticmethod
    def sample_pixels(df: pd.DataFrame,
                      nrpixels=1,
                      label='LC',
                      nodata_value=0,
                      seed=1234):
        '''
        Method to sample one or more valid pixels
        from the training dataframe
        '''
        if df is None:
            return None

        np.random.seed(seed)

        labeldata = df[label].values

        if label != 'LC':
            # also filter on annual/perennial cropland and grassland
            idxvalid = np.where((labeldata != nodata_value)
                                & (df['LC'].isin([10, 11, 12, 13])))[0]
        else:
            idxvalid = np.where(labeldata != nodata_value)[0]

        if len(idxvalid) == 0:
            return None
        elif len(idxvalid) > nrpixels:

            # Make DataFrame with valid pixels
            labeldf = pd.DataFrame(index=idxvalid, data=labeldata[idxvalid],
                                   columns=['label'])

            # Get the pixel count for smallest class
            smallest = labeldf.value_counts().min()

            # Get the to be sampled pixel amount per class
            sample_pixels = np.min([nrpixels, smallest])

            # Do the class sampling
            sampled = labeldf.groupby('label').sample(sample_pixels)

            # Get the indexes of sampled pixels
            idxchoice = sampled.index.to_list()

            # Get the subset DF
            df_sample = df.iloc[idxchoice, :]
        else:
            df_sample = df.iloc[idxvalid, :]

        return df_sample


class SegmentTrainingChain(TrainingChain):

    def __init__(self, **kwargs):

        self.chaintype = 'segment'
        super().__init__(**kwargs)

        raise NotImplementedError


class ClassificationProcessor:

    supported_sources = ['OPTICAL',
                         'SAR',
                         'METEO',
                         'DEM',
                         'TIR']

    def __init__(self,
                 output_folder,
                 models: Dict,
                 collections: Dict,
                 season: str = 'tc-annual',
                 settings: Dict = None,
                 rsi_meta: Dict = None,
                 features_meta: Dict = None,
                 ignore_def_feat: Dict = None,
                 gdd_normalization: bool = False,
                 fps: Dict = None,
                 aez: int = None,
                 start_date='20190101',
                 end_date='20200101',
                 save_confidence=True,
                 save_meta=True,
                 save_features=False,
                 decision_threshold: float = 0.5,
                 filtersettings: Dict = None,
                 featresolution: int = 10,
                 avg_segm: bool = False,
                 segm_feat: List = None,
                 use_existing_features: bool = False,
                 features_dir: str = None,
                 **kwargs):

        logger.debug("Initializing ClassificationProcessor")

        self.collections = {k: v for k, v in collections.items()
                            if v is not None}
        self.aez = aez
        self.output = Path(output_folder)
        self.models = models
        self.filtersettings = filtersettings
        self.decision_threshold = decision_threshold
        self.season = season
        self.settings = settings
        self.rsi_meta = rsi_meta
        self.features_meta = features_meta
        self.ignore_def_feat = ignore_def_feat
        self.gdd_normalization = gdd_normalization
        self._start_date = start_date
        self._end_date = end_date
        self._save_features = save_features
        self._save_confidence = save_confidence
        self._save_meta = save_meta
        self.featresolution = featresolution
        self.use_existing_features = use_existing_features
        self.avg_segm = avg_segm
        self.segm_feat = segm_feat or ['OPTICAL-ndvi-p10-10m',
                                       'OPTICAL-ndvi-difminmax-10m',
                                       'OPTICAL-ndvi-ts1-10m',
                                       'OPTICAL-ndvi-ts3-10m',
                                       'OPTICAL-ndvi-ts5-10m']

        if features_dir is not None:
            self.features_dir = Path(features_dir)
        else:
            self.features_dir = self.output

        self._check_sources()

        # Configure the featuresprocessors
        fps = fps or {}
        self._fps = {
            'OPTICAL': fps.get('OPTICAL', WorldCerealOpticalFeaturesProcessor),
            'SAR': fps.get('SAR', WorldCerealSARFeaturesProcessor),
            'METEO': fps.get('METEO', WorldCerealAgERA5FeaturesProcessor),
            'DEM': fps.get('DEM', Features.from_dem),
            'TIR': fps.get('TIR', WorldCerealThermalFeaturesProcessor)
        }

        # Now make sure start_date and end_date are set in the settings
        # of all collections
        if self.settings is not None:
            self._add_settings_dates()

            # Add optional GDD normalization to the settings as well
            if self.gdd_normalization:
                for coll in self.settings.keys():
                    self.settings[coll]['normalize_gdd'] = dict(
                        tbase=TBASE[season],
                        tlimit=GDDTLIMIT[season],
                        season=season
                    )

        logger.debug("ClassificationProcessor initialized")

    def _add_settings_dates(self):
        for coll in self.settings.keys():
            self.settings[coll]['composite']['start'] = self._start_date
            self.settings[coll]['composite']['end'] = self._end_date

        if 'sen2agri_temp_feat' in self.features_meta.get('OPTICAL', {}):
            self.features_meta['OPTICAL'][
                'sen2agri_temp_feat'][
                    'parameters']['time_start'] = self._start_date

        if 'sumdiv' in self.features_meta.get('METEO', {}):
            length_season = (pd.to_datetime(self._end_date) -
                             pd.to_datetime(self._start_date))
            self.features_meta['METEO']['sumdiv'][
                'parameters']['div'] = length_season.days

    def _check_sources(self):

        for s in self.collections.keys():
            if s not in self.supported_sources:
                raise ValueError(f"{s} is not a supported source. "
                                 "Supported sources: "
                                 f"'{self.supported_sources}'.")

    def _filtered_collections(self, collections, tile, bounds, epsg):

        # Filter collections spatially
        data_collections = {k: v.filter_bounds(bounds, epsg)
                            for k, v in collections.items()}

        # Filter temporal collections
        # Subtract and add 1 day to be certain
        # it's included in the filtering
        collstart = (pd.to_datetime(self._start_date) -
                     pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        collend = (pd.to_datetime(self._end_date) +
                   pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        for collection in data_collections:
            if collection != 'DEM':
                data_collections[collection] = (
                    data_collections[collection].filter_dates(
                        collstart,
                        collend
                    )
                )

        # If we have OPTICAL/SAR/TIR product, additionally filter on the
        # tile
        for coll in ['OPTICAL', 'SAR', 'TIR']:
            if coll in data_collections:
                data_collections[coll] = (
                    data_collections[coll].filter_tiles(tile))

        # Filter DEM for tile (different method)
        if 'DEM' in data_collections:
            data_collections['DEM'] = data_collections['DEM'].filter_tile(tile)

        # Filter on SCL 60m which we do
        # BEFORE checking collection completeness
        for coll in ['OPTICAL', 'TIR']:
            if coll in data_collections:
                data_collections[coll] = data_collections[coll].filter_nodata()

                # HACK: for now: disable L8 if already too many S2.
                # otherwise we run into memory issues
                if coll == 'OPTICAL':
                    nr_obs_S2 = data_collections[coll].df.path.str.contains(
                        'MSIL2A').sum()
                    nr_obs_L8 = data_collections[coll].df.path.str.contains(
                        'LC08').sum()
                    if nr_obs_S2 > 100 and nr_obs_L8 > 0:
                        logger.warning(
                            'More than 100 S2 products: disabling L8!')
                        S2coll = data_collections[coll]
                        S2coll.df = S2coll.df[
                            S2coll.df.path.str.contains('MSIL2A')]
                        data_collections[coll] = S2coll

        # Check for abnormal gaps in collections
        for collid, coll in data_collections.items():
            if 'DEM' not in collid:
                _ = check_collection(coll, collid,
                                     self._start_date, self._end_date,
                                     fail_threshold=get_coll_maxgap(collid))

        # Check if collections or not empty
        for collid, coll in data_collections.items():
            if 'DEM' not in collid:
                if coll.df.shape[0] == 0:
                    raise EmptyCollection(
                        ('Got an empty collection for '
                         f'`{collid}`. Cannot process block!'))

        logger.info('-' * 50)

        return data_collections

    def get_features(self, tile, bounds, epsg,
                     settings_override=None,
                     rsi_meta_override=None,
                     features_meta_override=None,
                     ignore_def_feat_override=None,
                     collections_override=None
                     ) -> Features:

        def num_to_feat(data, names, shapex, shapey):

            data = [np.full((shapex, shapey), d) for d in data]
            data = np.stack(data, axis=0)
            return Features(data, names, dtype=np.int16)

        settings = settings_override or self.settings
        rsi_meta = rsi_meta_override or self.rsi_meta
        features_meta = features_meta_override or self.features_meta
        ignore_def_feat = ignore_def_feat_override or self.ignore_def_feat
        collections = collections_override or self.collections

        logger.info("Filtering collections")
        data_collections = self._filtered_collections(collections,
                                                      tile, bounds, epsg)

        logger.info("Starting features computation")

        # Compute the features
        features = BlockProcessor(
            data_collections,
            settings,
            rsi_meta,
            features_meta,
            ignore_def_feat,
            bounds,
            epsg,
            custom_fps=self._fps,
            featresolution=self.featresolution,
            avg_segm=self.avg_segm,
            segm_feat=self.segm_feat,
            aez_id=self.aez).get_features()

        # Adding meta features related to gaps in collections
        for collection in data_collections:
            if 'DEM' not in collection:
                coll_gaps, names = check_collection(
                    data_collections[collection],
                    collection,
                    self._start_date, self._end_date)
                coll_gaps_feat = num_to_feat(coll_gaps, names,
                                             features.data.shape[1],
                                             features.data.shape[2])
                features = features.merge(coll_gaps_feat)

        # Put feature data on disk
        logger.info('Saving to zarr on disk')
        ft_dir = tempfile.TemporaryDirectory(dir='.').name
        fn = str(Path(ft_dir) / f'features_{tile}_{bounds}.zarr')
        atexit.register(_clean, ft_dir)
        z = zarr.open(fn, mode='w',
                      shape=features.data.shape,
                      chunks=(128, 128), dtype=np.float32)
        z[:] = features.data
        logger.info('Switching features data to disk version')
        features.data = z

        return features

    def get_features_fromfile(self, output_folder, tile, block_id, aez,
                              producttag):

        fname = (output_folder / str(tile) /
                 f'{self._end_date[0:4]}_{self.season}' /
                 f'features_{producttag}' /
                 f"{tile}_{aez}_{block_id:03d}_features.tif")

        if fname.is_file():
            # We can load the existing features from file
            logger.info(f'Loading existing features from: {fname}')
            features = load_features_geotiff(fname)

            if not np.any(~np.isnan(features.data)):
                # if we only get nan's, there's something wrong...
                features = None

            return features
        else:
            return None

    def process(self,
                tile,
                bounds,
                epsg,
                block_id):

        # Set output folder
        output_folder = self.output / 'blocks'

        # Compute the features
        features = None
        if self.use_existing_features:
            try:
                features = self.get_features_fromfile(
                    self.features_dir / 'blocks',
                    tile, block_id,
                    self.aez, 'cropland')
            except Exception as e:
                logger.warning(e)
                logger.warning('Reading features from file '
                               'failed, recomputing...')

        # Compute the features if we don't have them yet
        if features is None:
            features = self.get_features(tile, bounds, epsg)

            # Add AEZ GROUP and ID as features
            aez_id_ft = Features.from_constant(self.aez, 'aez_zoneid',
                                               features.data.shape)
            aez_group_ft = Features.from_constant(group_from_id(self.aez),
                                                  'aez_groupid',
                                                  features.data.shape)
            features = features.merge(aez_id_ft, aez_group_ft)

            # Save the features
            if self._save_features:
                self.save_features(features, self.features_dir / 'blocks',
                                   tile, block_id, bounds, epsg,
                                   self.aez, 'cropland')

        # Run the classification
        self.classify(features, output_folder, tile, bounds,
                      epsg, block_id)

    def classify(self, features, output_folder, tile, bounds,
                 epsg, block_id, models=None,
                 mask=None, decision_threshold=None):
        # Now do the predictions based on all provided models
        # and write result to disk

        if decision_threshold is None:
            decision_threshold = self.decision_threshold

        if models is not None:
            models = models.items()
        else:
            models = self.models.items()

        try:

            for product, model in models:

                if product == 'springwheat' or product == 'springcereals':
                    from worldcereal.utils.aez import load
                    aez_df = load()
                    if aez_df.set_index('zoneID').loc[
                            self.aez]['trigger_sw'] == 0:
                        logger.warning(f'`{product}` detector skipped based '
                                       'on `trigger_sw` flag for this AEZ.')
                        continue

                # Load the model
                wcmodel = WorldCerealModel.from_config(model)

                # Check if the model needs SAR. If we don't have it
                # try to load optical-onlyh model
                if (any(['SAR' in ft for ft in wcmodel.feature_names]) and not
                        any(['SAR' in ft for ft in features.names])):
                    logger.warning(('Default model needs SAR but we dont have it. '
                                    'Trying to load optical-only model ...'))
                    wcmodel = self._get_opticalonly_model(model)

                # Create a WorldCerealClassifier from the model
                self.classifier = WorldCerealClassifier(
                    wcmodel,
                    filtersettings=self.filtersettings,
                    maskdata=mask)

                # get correct colormap and nodatavalue
                colormap = COLORMAP.get(product.split('-')[0], None)
                nodata = NODATA.get(product.split('-')[0], None)

                # Get prediction and confidence
                prediction, confidence = self._make_prediction(
                    features,
                    decision_threshold=decision_threshold,
                    model=model,
                    maskdata=mask)

                # Remap nodata for output product
                if nodata is not None:
                    prediction[prediction == 255] = nodata
                    confidence[confidence == 255] = nodata

                self.save(prediction, tile, bounds, epsg,
                          self.aez, output_folder, block_id=block_id,
                          product=product, tag='classification',
                          colormap=colormap, nodata=nodata)

                if self._save_confidence:
                    self.save(confidence, tile, bounds, epsg,
                              self.aez, output_folder, block_id=block_id,
                              product=product, tag='confidence',
                              nodata=nodata)

                if self._save_meta:
                    self.save_meta(features, output_folder, product,
                                   tile, block_id, bounds, epsg,
                                   self.aez)

                # Clear session to avoid memory leak
                tf.keras.backend.clear_session()
                prediction = None
                confidence = None
                gc.collect()
        except Exception as e:
            logger.error(f"Error predicting for {tile} - {bounds} "
                         f"- {block_id}: {e}")
            raise

    def _make_prediction(self, features, decision_threshold, model, maskdata):

        # Get prediction and confidence
        prediction, confidence = self.classifier.predict(
            features,
            threshold=decision_threshold,
            nodatavalue=255)

        MIN_SIGMA0_OBS = 10
        if ('SAR-sigma0_obs' in features.names and
            features.select(
                ['SAR-sigma0_obs']).data.min() < MIN_SIGMA0_OBS):
            opticalmodel = self._get_opticalonly_model(model)

            if opticalmodel is not None:
                prediction, confidence = self._merge_with_optical_predictions(  # NOQA
                    opticalmodel, features,
                    decision_threshold, maskdata,
                    prediction, confidence,
                    MIN_SIGMA0_OBS)

        return prediction, confidence

    @staticmethod
    def _get_opticalonly_model(model):
        opticalmodelpath = ('/'.join(model.split('/')[:-1])
                            + '-OPTICAL/config.json')
        modelparts = model.split('/')
        opticalmodelpaths = []
        for i in range(len(modelparts)-2, 0, -1):
            opticalmodelpaths.append(('/'.join(modelparts[:i]) + '/' + '/'.join(
                [p + '-OPTICAL' for p in modelparts[i:-1]])
                + '/' + modelparts[-1]))
        # one other variation for AEZ (group) models
        opticalmodelpath_pt1 = ('/'.join(model.split('/')[:-2])
                                + '-OPTICAL')
        opticalmodelpaths.append(('/'.join([opticalmodelpath_pt1,
                                            modelparts[-2],
                                            modelparts[-1]])))
        for opticalmodelpath in opticalmodelpaths:
            try:
                opticalmodel = WorldCerealModel.from_config(opticalmodelpath)
                return opticalmodel
            except Exception:
                continue

        logger.warning(('Tried to find an OPTICAL-only model '
                        'but did not succeed.'))
        return None

    def _merge_with_optical_predictions(self, opticalmodel, features,
                                        threshold, mask,
                                        prev_prediction, prev_confidence,
                                        sigma0_threshold):
        """Method to blend existing prediction with
        optical-only predictions.
        """

        updatemask = (features.select(['SAR-sigma0_obs']).data[0, ...]
                      < sigma0_threshold).astype(np.uint8)
        affected_pixels = updatemask.sum()

        logger.warning((f'Low S1 coverage detected in {affected_pixels} '
                        'pixels! Using optical-only model ...'))

        classifier = WorldCerealClassifier(
            opticalmodel,
            filtersettings=self.filtersettings,
            maskdata=mask)
        prediction, confidence = classifier.predict(
            features,
            threshold=threshold,
            nodatavalue=255)

        new_prediction = ((1 - updatemask) * prev_prediction
                          + updatemask * prediction)
        new_confidence = ((1 - updatemask) * prev_confidence
                          + updatemask * confidence)

        new_prediction[prev_prediction == 255] = 255  # Keep nodata
        new_confidence[prev_confidence == 255] = 255  # Keep nodata

        return (new_prediction.astype(np.uint8),
                new_confidence.astype(np.uint8))

    def save(self, predictiondata, tile, bounds,
             epsg, aez, save_folder, block_id=999, product='product',
             tag='classification', colormap=None,
             nodata=None):

        fname = self._product_fname(save_folder,
                                    f'{self._end_date[0:4]}_{self.season}',
                                    product,
                                    tag,
                                    tile,
                                    block_id,
                                    aez)
        fname.parent.mkdir(parents=True, exist_ok=True)
        self._to_geotiff(predictiondata, bounds, epsg, fname,
                         colormap=colormap, nodata=nodata)

    @ staticmethod
    def _to_geotiff(data, bounds, epsg, filename,
                    band_names=[], colormap=None,
                    nodata=None):

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        profile = get_rasterio_profile(data, bounds, epsg)
        logger.debug(f"Saving {filename}...")

        # Write to temporary file in current directory
        tempfile = f'./{Path(filename).name}'
        atexit.register(_clean, tempfile)

        write_geotiff(data, profile, tempfile,
                      band_names=band_names, colormap=colormap,
                      nodata=nodata)

        # Now copy the file
        logger.info('Copying to final destination ...')
        shutil.move(tempfile, filename)

    @ staticmethod
    def _product_fname(output_folder, season, product,
                       tag, tile, block_id, aez):
        product_filename = (output_folder / str(tile) / season / product /
                            tag / f"{tile}_{aez}_{block_id:03d}_{tag}.tif")
        return product_filename

    @ staticmethod
    def is_real_feature(ft_name):
        return is_real_feature(ft_name)

    def save_features(self, features, output_folder,
                      tile, block_id, bounds, epsg, aez,
                      producttag,
                      compression='deflate-uint16-lsb11-z9'):
        logger.info('Saving features ...')

        fts = ([ft for ft in features.names
                if ft not in ['pixelids']])
        features = features.select(fts)

        fname = (output_folder / str(tile) /
                 f'{self._end_date[0:4]}_{self.season}' /
                 f'features_{producttag}' /
                 f"{tile}_{aez}_{block_id:03d}_features.tif")
        fname.parent.mkdir(parents=True, exist_ok=True)

        _ = save_features_geotiff(features, bounds=bounds, epsg=epsg,
                                  filename=fname, compress_tag=compression)

    def save_meta(self, features, output_folder, product,
                  tile, block_id, bounds, epsg, aez):
        logger.info('Saving metadata ...')
        meta_ft_names = [
            ft_name for ft_name in features.names
            if not self.is_real_feature(ft_name) and
            ft_name not in ['lat', 'lon', 'pixelids',
                            'aez_zoneid', 'aez_groupid']]
        if len(meta_ft_names) == 0:
            logger.warning('No metafeatures found.')
            return
        meta_fts = features.select(meta_ft_names)
        meta_fts.data = meta_fts.data.astype(np.uint8)
        fname = self._product_fname(
            output_folder, f'{self._end_date[0:4]}_{self.season}',
            product, 'metafeatures', tile, block_id, aez)
        fname.parent.mkdir(parents=True, exist_ok=True)
        self._to_geotiff(meta_fts.data, bounds, epsg, fname,
                         band_names=meta_fts.names)

    def generate_fixed_output(self, value, output_folder, tile, bounds,
                              epsg, block_id, models, mask=None):
        """Method to produce a dummy output block with a fixed value,
        e.g. when the block is too cold to grow a crop.

        Args:
            value (int): Value to burn into output block product
            output_folder (str): path to folder for output product
            tile (str): S2 tile
            bounds (tuple): bounds of the product
            epsg (int): epsg code of the product
            block_id (int): ID of the block
            models (dict): dictionary containing product models
            mask (np.ndarray, optional): optional mask. Defaults to None.
        """

        models = models.items()

        for product, _ in models:

            if product == 'springwheat' or product == 'springcereals':
                from worldcereal.utils.aez import load
                aez_df = load()
                if aez_df.set_index('zoneID').loc[
                        self.aez]['trigger_sw'] == 0:
                    logger.warning(f'`{product}` detector skipped based '
                                   'on `trigger_sw` flag for this AEZ.')
                    continue

            # get correct colormap and nodatavalue
            colormap = COLORMAP.get(product.split('-')[0], None)
            nodata = NODATA.get(product.split('-')[0], None)

            # Generate the output products
            outshape = (int((bounds[3] - bounds[1]) / 10),
                        int((bounds[2] - bounds[0]) / 10))
            outdata = (np.ones(outshape) * value).astype(np.uint8)
            outconf = (np.ones(outshape) * 100).astype(np.uint8)

            # Mask if needed
            if mask is not None:
                outdata[mask != 100] = nodata
                outconf[mask != 100] = nodata

            self.save(outdata, tile, bounds, epsg,
                      self.aez, output_folder, block_id=block_id,
                      product=product, tag='classification',
                      colormap=colormap, nodata=nodata)

            if self._save_confidence:
                self.save(outconf, tile, bounds, epsg,
                          self.aez, output_folder, block_id=block_id,
                          product=product, tag='confidence',
                          nodata=nodata)


class CropTypeProcessor(ClassificationProcessor):
    def __init__(self, *args,
                 season='custom',
                 gdd_normalization=True,
                 cropland_mask=None,
                 active_marker=False,
                 irrigation=False,
                 irrcollections=None,
                 irrparameters=None,
                 irrmodels=None,
                 **kwargs):

        if season == 'custom':
            logger.warning(('Using `custom` season. Are '
                            'you sure this is what you want?'))

        super().__init__(*args,
                         season=season,
                         gdd_normalization=gdd_normalization,
                         **kwargs)

        self.active_marker = active_marker
        self.cropland_mask = cropland_mask
        self.irrigation = irrigation
        self.irrcollections = irrcollections
        self.irrparameters = irrparameters
        self.irrmodels = irrmodels

        self._check_season()

    def _check_season(self):
        if self.season not in SUPPORTED_SEASONS:
            raise ValueError(f'Unknown season `{self.season}`')

    def process(self,
                tile,
                bounds,
                epsg,
                block_id):
        ''' Override parent method for season-specific
        processing.
        '''

        # Set output folder
        output_folder = self.output / 'blocks'

        # Check if we need to load the cropland mask
        if self.cropland_mask is not None:

            # Make a WorldCerealProductCollection for cropland
            year = self._end_date[0:4]
            croplandcoll = WorldCerealProductCollection(
                self.cropland_mask, int(year),
                'tc-annual', self.aez, 'temporarycrops').filter_tile(
                    tile).filter_bounds(bounds, epsg)

            try:
                maskdata = croplandcoll.load()
                logger.info(('Sucessfully loaded cropland mask '
                             f'from {year}.'))
            except Exception:
                # Try one year earlier
                croplandcoll = WorldCerealProductCollection(
                    self.cropland_mask, int(year)-1,
                    'tc-annual', self.aez, 'temporarycrops').filter_tile(
                    tile).filter_bounds(bounds, epsg)
                try:
                    maskdata = croplandcoll.load()
                    logger.info(('Sucessfully loaded cropland mask '
                                 f'from {int(year)-1}.'))
                except Exception:
                    logger.error('Failed loading required cropland mask!')
                    raise

            # If there is no cropland in this block
            # we log it here for proper handling later on
            if np.sum(maskdata == 100) == 0:
                nocropland = True
            else:
                nocropland = False

        else:
            maskdata = None
            nocropland = False

        if not nocropland:
            try:
                try:
                    features = None
                    if self.use_existing_features:
                        try:
                            features = self.get_features_fromfile(
                                self.features_dir / 'blocks',
                                tile, block_id,
                                self.aez, 'croptype')
                        except Exception:
                            logger.warning('Failed to read features from file, '
                                           'recomputing...')

                    # Compute the features if we don't have them yet
                    if features is None:
                        features = self.get_features(tile, bounds, epsg)

                        if self._save_features:
                            self.save_features(
                                features, self.features_dir / 'blocks',
                                tile, block_id, bounds, epsg,
                                self.aez, 'croptype')

                    # Run the classification
                    self.classify(features, output_folder, tile, bounds,
                                  epsg, block_id, mask=maskdata)
                except BlockTooColdError as e:
                    # We need to disable GDD normalization for active_cropland
                    # and burn a fixed 0-value (other) in the output product
                    logger.warning(e)
                    self._switch_blocktoocold_settings()
                    features = self.get_features(tile, bounds, epsg)
                    logger.info(('Generating output product with '
                                 'fixed 0 value ...'))
                    self.generate_fixed_output(0, output_folder, tile, bounds,
                                               epsg, block_id, self.models,
                                               mask=maskdata)

                # Optionally retrieve active marker
                if self.active_marker:
                    activemarker_ft = 'OPTICAL-evi-nSeas-10m'
                    if activemarker_ft not in features.names:
                        raise ValueError(
                            ('The active marker needs the '
                                f'`{activemarker_ft}` feature which is '
                                'not available in the computed features.'))
                    nr_seasons = features.select(
                        ['OPTICAL-evi-nSeas-10m']).data[0, ...]
                    nr_seasons = np.round(nr_seasons).astype(np.uint8)
                    nr_seasons[nr_seasons > 0] = 1
                    nr_seasons[nr_seasons != 1] = 0

                    colormap = COLORMAP['activecropland']
                    nodatavalue = NODATA['activecropland']

                    if self.cropland_mask is not None:
                        nr_seasons = mask(nr_seasons, maskdata,
                                          maskedvalue=255)

                    # apply post-classification majority filtering
                    if self.filtersettings['kernelsize'] > 0:
                        nr_seasons = majority_filter(
                            nr_seasons,
                            self.filtersettings['kernelsize'],
                            no_data_value=255)

                    # To correct dtype
                    nr_seasons = nr_seasons.astype(np.uint8)

                    # # Remap values and nodata for output product
                    nr_seasons[nr_seasons == 1] = 100  # Active cropland
                    nr_seasons[nr_seasons == 255] = nodatavalue

                    self.save(nr_seasons, tile, bounds, epsg, self.aez,
                              output_folder, block_id=block_id,
                              product='activecropland', tag='classification',
                              colormap=colormap, nodata=nodatavalue)

                    if self._save_meta:
                        activecropland_metafeatures = [
                            ft for ft in features.names if 'OPTICAL' in ft
                        ]
                        self.save_meta(
                            features.select(activecropland_metafeatures),
                            output_folder, 'activecropland',
                            tile, block_id, bounds, epsg,
                            self.aez)

            except Exception as e:
                logger.error(f"Error predicting crop type for "
                             f"{tile} - {bounds} "
                             f"- {block_id}: {e}")
                raise

            if self.irrigation:

                if self.irrmodels is None:
                    raise ValueError('No dedicated irrigation models have been'
                                     ' supplied for the irrigation detection!')

                if self.irrparameters is None:
                    raise ValueError(
                        'No dedicated irrigation parameters have '
                        'been provided for irrigation feature computation!')

                logger.info('Preparing settings for irrigation')
                irr_settings = self.irrparameters.get('settings', None)
                if irr_settings is not None:
                    # Now make sure start_date and end_date are set in settings
                    # of all collections
                    start_date = self.settings['OPTICAL']['composite'].get('start')
                    end_date = self.settings['OPTICAL']['composite'].get('end')
                    for coll in irr_settings.keys():
                        irr_settings[coll]['composite']['start'] = start_date
                        irr_settings[coll]['composite']['end'] = end_date

                irr_features_meta = self.irrparameters.get('features_meta', None)
                if (irr_settings is not None) and (irr_features_meta is not None):
                    if 'sumdiv' in irr_features_meta.get('METEO', {}):
                        length_season = (pd.to_datetime(end_date) -
                                         pd.to_datetime(start_date))
                        irr_features_meta['METEO']['sumdiv'][
                            'parameters']['div'] = length_season.days
                irr_rsi_meta = self.irrparameters.get('rsi_meta', None)
                irr_ignore_def_feat = self.irrparameters.get(
                    'ignore_def_feat', None)
                irr_collections = {k: v for k, v in self.irrcollections.items()
                                   if v is not None}

                irrfeatures = None
                if self.use_existing_features:
                    try:
                        irrfeatures = self.get_features_fromfile(
                            self.features_dir / 'blocks',
                            tile, block_id,
                            self.aez, 'irrigation')
                    except Exception:
                        logger.warning('Failed to read features '
                                       'from file, recomputing...')

                # Compute the features if we don't have them yet
                if irrfeatures is None:
                    logger.info('Computing irrigation features')
                    irrfeatures = self.get_features(
                        tile, bounds, epsg,
                        settings_override=irr_settings,
                        features_meta_override=irr_features_meta,
                        rsi_meta_override=irr_rsi_meta,
                        ignore_def_feat_override=irr_ignore_def_feat,
                        collections_override=irr_collections)

                    if self._save_features:
                        self.save_features(irrfeatures,
                                           self.features_dir / 'blocks',
                                           tile, block_id, bounds, epsg,
                                           self.aez, 'irrigation')

                # Run the classification
                self.classify(irrfeatures, output_folder, tile, bounds,
                              epsg, block_id,
                              models=self.irrmodels, mask=maskdata,
                              decision_threshold=0.7)

            # Finally, because we might have created more than
            # one propduct, try to resolve conflicts in the maps
            self.resolve_conflicts(tile, block_id)

        else:
            # Handle the case where we have no cropland
            logger.info(('No cropland pixels found in this block: '
                         'generating output products with '
                         'fixed nodata value 255 ...'))
            self.generate_fixed_output(255, output_folder, tile, bounds,
                                       epsg, block_id, self.models,
                                       mask=maskdata)

            if self.active_marker:
                colormap = COLORMAP['activecropland']
                nodatavalue = NODATA['activecropland']
                nr_seasons = np.ones_like(maskdata).astype(np.uint8)
                nr_seasons *= 255

                self.save(nr_seasons, tile, bounds, epsg, self.aez,
                          output_folder, block_id=block_id,
                          product='activecropland', tag='classification',
                          colormap=colormap, nodata=nodatavalue)

            if self.irrigation:
                self.generate_fixed_output(255, output_folder, tile, bounds,
                                           epsg, block_id, self.irrmodels,
                                           mask=maskdata)

    def _switch_blocktoocold_settings(self):
        """Force updated settings that will only compute active_cropland
        in case of temperatures that are too low to grow the crop(s) of
        interest.
        """
        from worldcereal.features.settings import get_active_crop_parameters

        logger.warning(('Overriding settings for active_cropland '
                        'computation only'))
        params = get_active_crop_parameters()
        self.settings = params['settings']
        self.features_meta = params['features_meta']

        # Add start and end dates to the new settings
        self._add_settings_dates()

    def resolve_conflicts(self, tile, block_id):

        block_folder = self.output / 'blocks'

        conflictresolver = conflicts.ConflictResolver(
            block_folder, tile, block_id,
            self.aez, self.season, self._end_date[0:4])
        conflictresolver.run()


class _ChainTimer():

    def __init__(self):

        self.trainingdf = TaskTimer('building training df per sample',
                                    unit='seconds')
        self.trainModel = TaskTimer('model training',
                                    unit='seconds')
        self.evalModel = TaskTimer('model evaluation',
                                   unit='seconds')
