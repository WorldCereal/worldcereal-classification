from datetime import datetime
import glob
import json
from pathlib import Path
from typing import Dict, List, TypeVar

from dateutil.parser import parse
from loguru import logger
import numpy as np
import pandas as pd
import requests
import satio
from satio.collections import (TerrascopeSigma0Collection,
                               TrainingCollection,
                               DiskCollection,
                               FolderCollection,
                               S2Collection,
                               AgERA5Collection)
from satio.geoloader import LandsatLoader, ParallelLoader
from satio.grid import tile_to_epsg

from worldcereal.geoloader import (EWOCS3ParallelLoader,
                                   EWOCGDALWarpLoader,
                                   _get_s3_keys)
from worldcereal.fp import TSSIGMA0TiledFeaturesProcessor as TSS0FeatProc
from worldcereal.fp import L8ThermalFeaturesProcessor as L8FeatProc
from worldcereal.fp import WorldCerealSARFeaturesProcessor as WCS0FP
from worldcereal.fp import WorldCerealOpticalFeaturesProcessor as WCOPTICALFP
from worldcereal.fp import WorldCerealThermalFeaturesProcessor as WCTHERMALFP
from worldcereal.utils import SkipBlockError
from worldcereal.utils.masking import pixel_qa_mask, select_valid_obs

_TS_S2_BASEURL = ('https://services.terrascope.be/catalogue/'
                  'products?collection=urn%3Aeop%3AVITO%3A'
                  'TERRASCOPE_S2')
_TS_S1_BASEURL = ('https://services.terrascope.be/catalogue/'
                  'products?collection=urn%3Aeop%3AVITO%3A'
                  'CGS_S1_GRD_SIGMA0_L1')

L8_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "MASK"]


class TerrascopeSigma0TiledCollection(TerrascopeSigma0Collection):

    @property
    def supported_resolutions(self):
        return [20]

    @property
    def loader(self):
        if self._loader is None:
            self._loader = ParallelLoader(fill_value=65535)
        return self._loader

    def features_processor(self,
                           settings: Dict,
                           rsi_meta: Dict = {},
                           features_meta: Dict = {},
                           ignore_def_features: bool = False) -> TSS0FeatProc:
        return TSS0FeatProc(self,
                            settings,
                            rsi_meta=rsi_meta,
                            features_meta=features_meta,
                            ignore_def_features=ignore_def_features)

    @staticmethod
    def get_terrascope_filenames(terrascope_product):
        filenames_list = glob.glob(str(Path(terrascope_product) / '*.tif'),
                                   recursive=True)
        filenames = {}
        resolution = 20

        for f in filenames_list:

            band = str(Path(f).stem).split('_')[-2].split('.')[0]

            filenames[band] = {resolution: f}

        return filenames

    @staticmethod
    def get_terrascope_band_filename(filenames, band, resolution=20):
        """Returns filename for band (TERRASCOPE product)
        closest to target resolution"""

        bands_20m = ['VV', 'VH']

        if band not in bands_20m:
            raise ValueError("Band {} not supported.".format(band))

        return filenames[band][resolution]


class WorldCerealSigma0TiledCollection(DiskCollection):

    sensor = 'S1'
    processing_level = 'SIGMA0'

    @property
    def supported_bands(self):
        return ['VV', 'VH']

    @property
    def supported_resolutions(self):
        return [20]

    def get_band_filenames(self, band, resolution):
        if self._filenames is None:
            self._filenames = self.df['path'].apply(
                lambda x: self.get_worldcereal_filenames(x))

        tif_filenames = self._filenames.apply(
            lambda x: self.get_worldcereal_band_filename(x, band, resolution))

        return tif_filenames.values.tolist()

    @classmethod
    def from_folder(cls, folder, s2grid=None):
        df = cls.build_products_df(folder)
        df = df.sort_values('date', ascending=True)
        collection = cls(df, s2grid=s2grid)
        return collection

    def features_processor(self,
                           settings: Dict,
                           rsi_meta: Dict = {},
                           features_meta: Dict = {},
                           ignore_def_features: bool = False) -> WCS0FP:
        return WCS0FP(self,
                      settings,
                      rsi_meta=rsi_meta,
                      features_meta=features_meta,
                      ignore_def_features=ignore_def_features)

    @staticmethod
    def get_worldcereal_filenames(worldcereal_product):

        filenames = {}
        resolution = 20

        for band in ['VV', 'VH']:
            # Construct path to band file
            f = (worldcereal_product + '/' +
                 Path(worldcereal_product).stem +
                 f'_SIGMA0_{band}.tif')

            filenames[band] = {resolution: f}

        return filenames

    @staticmethod
    def get_worldcereal_band_filename(filenames, band, resolution=20):
        """Returns filename for band (WORLDCEREAL product)
        """

        bands_20m = ['VV', 'VH']

        if band not in bands_20m:
            raise ValueError("Band {} not supported.".format(band))

        return filenames[band][resolution]

    @classmethod
    def build_products_df(cls, folder):

        products = []
        zones = [Path(x).stem for x in glob.glob(str(Path(folder) / '*'))]
        for zone in zones:
            tilespart1 = [Path(x).stem for x in glob.glob(
                str(Path(folder) / zone / '*'))]
            for tilepart1 in tilespart1:
                tilespart2 = [Path(x).stem for x in glob.glob(
                    str(Path(folder) / zone / tilepart1 / '*'))]
                for tilepart2 in tilespart2:
                    tileproducts = [x for x in glob.glob(
                        str(Path(folder) / zone /
                            tilepart1 / tilepart2 / '*' / '*' / '*'))]

                    for tileproduct in tileproducts:
                        products.append(tileproduct)

        entries = [cls.sentinel1_entry(f) for f in products]
        if len(entries):
            df = pd.DataFrame(entries)
        else:
            df = pd.DataFrame([], columns=['date',
                                           'tile',
                                           'path',
                                           'level'])

        return df

    @staticmethod
    def sentinel1_entry(filename):
        """
        """
        basename = Path(filename).stem
        date = basename.split('_')[1]
        tile = Path(filename).stem.split('_')[-1]

        entry = dict(date=datetime.strptime(date, '%Y%m%dT%H%M%S'),
                     tile=tile,
                     level='SIGMA0',
                     path=filename)

        return entry


class WorldCerealOpticalTiledCollection(S2Collection):

    processing_level = 'ATCOR'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FOR NOW OVERRIDE TO ONLY TAKE S2!
        # logger.info('Landsat-8 OPTICAL deactivated!')
        self.df = self.df[self.df.path.str.contains('MSIL2A')]

    @ property
    def supported_bands(self):
        return ["B02", "B03", "B04", "B05", "B06",
                "B07", "B08", "B8A", "B11", "B12", "MASK"]

    def get_band_filenames(self, band, resolution):
        if self._filenames is None:
            self._filenames = self.df['path'].apply(
                lambda x: self.get_worldcereal_filenames(x))

        tif_filenames = self._filenames.apply(
            lambda x: self.get_worldcereal_band_filename(x, band, resolution))

        return tif_filenames.values.tolist()

    @classmethod
    def from_folder(cls, folder, s2grid=None):
        df = cls.build_products_df(folder)
        df = df.sort_values('date', ascending=True)
        collection = cls(df, s2grid=s2grid)
        return collection

    def features_processor(self,
                           settings: Dict,
                           rsi_meta: Dict = {},
                           features_meta: Dict = {},
                           ignore_def_features: bool = False) -> WCOPTICALFP:
        return WCOPTICALFP(self,
                           settings,
                           rsi_meta=rsi_meta,
                           features_meta=features_meta,
                           ignore_def_features=ignore_def_features)

    def load_timeseries(self,
                        *bands,
                        **kwargs):
        '''Need to override parent method to cope with missing
        bands in L8 data that can be part of the timeseries.

        and to correct currently faulty values
        '''
        from satio.timeseries import load_timeseries

        # Get the requested bands we can load from
        # both S2 and L8
        combined_to_load = [b for b in list(bands)
                            if b in L8_BANDS]

        if combined_to_load:
            # Load combined S2/L8 bands
            combined_ts = load_timeseries(self, *combined_to_load, **kwargs)
        else:
            combined_ts = None

        # Get the requested bands we can load
        # exclusively from S2
        S2_to_load = [b for b in list(bands)
                            if b not in L8_BANDS]

        if S2_to_load:
            # Load exclusive S2 bands
            S2_coll = self._clone()
            S2_coll.df = self.df[self.df.path.str.contains('MSIL2A')]
            S2_ts = load_timeseries(S2_coll, *S2_to_load, **kwargs)
        else:
            S2_ts = None

        if S2_ts is not None and combined_ts is not None:
            timeseries = combined_ts.merge(S2_ts, fill_missing_timestamps=True)
        elif combined_ts is not None:
            timeseries = combined_ts
        else:
            timeseries = S2_ts

        # Put faulty values to nodata
        timeseries.data[timeseries.data > 65000] = 0

        return timeseries

    @staticmethod
    def get_worldcereal_filenames(worldcereal_product):

        filenames = {}
        bands_10m = ['B02', 'B03', 'B04', 'B08']
        bands_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12',
                     'MASK']

        allbands = bands_10m + bands_20m

        for band in allbands:
            if 'L1T' in worldcereal_product and band not in L8_BANDS:
                # Band not available for L8 products
                continue

            # Construct path to band file
            f = (worldcereal_product + '/' +
                 Path(worldcereal_product).stem.replace(
                     'MSIL2A', 'L2A'
                 ).replace('L1T', 'L2SP') +
                 f'_{band}.tif')

            if band in bands_10m:
                filenames[band] = {10: f}
            else:
                filenames[band] = {20: f}

        return filenames

    @staticmethod
    def get_worldcereal_band_filename(filenames, band, resolution):
        bands_10m = ['B02', 'B03', 'B04', 'B08']
        bands_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12',
                     'MASK']

        if band in bands_10m:
            resolution = 10
        elif band in bands_20m:
            resolution = 20
        else:
            raise ValueError("Band {} not supported.".format(band))

        return filenames[band][resolution]

    def filter_nodata(self,
                      resolution=20,
                      mask_th=0.05,
                      min_keep_fraction=0.75,
                      min_acquisitions=75,
                      min_offswath_acquisitions=3):
        """
        Load mask and filter collection based on the relative amount
        of valid pixels. Minimum is specified by `mask_th`.
        e.g. a frame is valid if the mean of the binary mask
        is above `mask_th`. Or in other words,
        keep frames with more than `mask_th * 100` % valid pixels.

        `min_keep_fraction` is the minimum fraction of obs that will be
        retained, if needed by relaxing the the mask_th
        until the minimum required amount of acquisitions is met.

        `min_acquisitions` is the minimum amount of within-swath
        acquisitions that are needed to even start filter_nodata process.

        `min_offswath_acquisitions` is the minimum amount of within-swath
        acquisitions below which a SkipBlockError will be raised
        """
        mask_20 = self.load(['MASK'], resolution=resolution)['MASK']

        within_swath = (mask_20 != 255).sum(axis=(1, 2), dtype=np.float32)

        if (within_swath > 0).values.sum() < min_offswath_acquisitions:
            # No acquisitions with within-swath data
            # should skip this block gracefully
            raise SkipBlockError((f'Less than {min_offswath_acquisitions} '
                                  'off-swath acquisitions found -> safe to '
                                  'skip block'))

        valid_ids = select_valid_obs(mask_20, mask_th,
                                     min_keep_fraction, within_swath,
                                     min_acquisitions=min_acquisitions)

        return self._clone(df=self.df.iloc[valid_ids])

    @ classmethod
    def build_products_df(cls, folder):

        products = []
        zones = [Path(x).stem for x in glob.glob(str(Path(folder) / '*'))]
        for zone in zones:
            tilespart1 = [Path(x).stem for x in glob.glob(
                str(Path(folder) / zone / '*'))]
            for tilepart1 in tilespart1:
                tilespart2 = [Path(x).stem for x in glob.glob(
                    str(Path(folder) / zone / tilepart1 / '*'))]
                for tilepart2 in tilespart2:
                    tileproducts = [x for x in glob.glob(
                        str(Path(folder) / zone /
                            tilepart1 / tilepart2 / '*' / '*' / '*'))]

                    for tileproduct in tileproducts:
                        filenames = glob.glob(
                            str(Path(tileproduct) / '*.tif'))
                        bands = [Path(x).stem.split('_')[
                            -1].split('.')[0] for x in filenames]
                        if 'MASK' in bands:
                            products.append(tileproduct)
                        else:
                            logger.warning(
                                ('No MASK file found for: '
                                 f'`{tileproduct}` -> Ignoring product'))

        entries = [cls.sentinel2_entry(f) for f in products]
        if len(entries):
            df = pd.DataFrame(entries)
        else:
            df = pd.DataFrame([], columns=['date',
                                           'tile',
                                           'path',
                                           'level'])

        return df

    @ staticmethod
    def sentinel2_entry(filename):
        """
        """
        basename = Path(filename).stem
        date = basename.split('_')[2]
        tile = Path(filename).stem.split('_')[-1]

        entry = dict(date=pd.to_datetime(date),
                     tile=tile,
                     level='SMAC',
                     path=filename)

        return entry


class WorldCerealThermalTiledCollection(DiskCollection):

    sensor = 'L8'
    processing_level = 'L2SP'

    def __init__(self, df, s2grid=None):
        super().__init__(df, s2grid=s2grid)
        self.df = self.df[self.df.level == self.processing_level]

    @ property
    def supported_bands(self):
        return ['B10', 'QA_PIXEL']

    @ property
    def supported_resolutions(self):
        return [10, 30]

    @property
    def loader(self):
        if self._loader is None:
            self._loader = LandsatLoader()
        return self._loader

    @loader.setter
    def loader(self, value):
        self._loader = value

    @classmethod
    def from_folder(cls, folder, s2grid=None):
        df = cls.build_products_df(folder)
        df = df.sort_values('date', ascending=True)
        collection = cls(df, s2grid=s2grid)
        return collection

    def filter_nodata(self,
                      resolution=30,
                      mask_th=0.05,
                      min_keep_fraction=0.75,
                      min_acquisitions=75):
        """
        filter collection based on the relative amount
        of valid pixels. Minimum is specified by `mask_th`.
        e.g. a frame is valid if the mean of the binary mask
        is above `mask_th`. Or in other words,
        keep frames with more than `mask_th * 100` % valid pixels.

        `min_keep_fraction` is the minimum fraction of obs that will be
        retained, if needed by relaxing the the mask_th
        until the minimum required amount of acquisitions is met.

        `min_acquisitions` is the minimum of amount of within-swath
        acquisitions that are needed to even start filter_nodata process.
        """
        qa_30 = self.load(['QA_PIXEL'], resolution=resolution)['QA_PIXEL']

        # Get a binary mask
        mask, _, _, _ = pixel_qa_mask(qa_30)

        # Count within swath pixels per obs
        within_swath = (qa_30 != 1).sum(axis=(1, 2)).astype(float)

        # Get relative amount of valid obs
        mask = mask.values.astype(int)

        valid_ids = select_valid_obs(mask, mask_th,
                                     min_keep_fraction, within_swath,
                                     min_acquisitions=min_acquisitions)

        return self._clone(df=self.df.iloc[valid_ids])

    def features_processor(self,
                           settings: Dict,
                           rsi_meta: Dict = {},
                           features_meta: Dict = {},
                           ignore_def_features: bool = False) -> WCTHERMALFP:
        return WCTHERMALFP(self,
                           settings,
                           rsi_meta=rsi_meta,
                           features_meta=features_meta,
                           ignore_def_features=ignore_def_features)

    def get_band_filenames(self, band, resolution):
        if self._filenames is None:
            self._filenames = self.df['path'].apply(
                lambda x: self.get_worldcereal_filenames(x))

        tif_filenames = self._filenames.apply(
            lambda x: self.get_worldcereal_band_filename(x, band, resolution))

        return tif_filenames.values.tolist()

    @staticmethod
    def get_worldcereal_filenames(worldcerealTherm_product):

        filenames = {}
        resolution = 30

        for band in ['B10', 'QA_PIXEL']:
            # Construct path to band file
            f = (worldcerealTherm_product + '/' +
                 Path(worldcerealTherm_product).stem.replace('L1T', 'L2SP') +
                 f'_{band}.tif')

            filenames[band] = {resolution: f}

        return filenames

    @staticmethod
    def get_worldcereal_band_filename(filenames, band, resolution=30):
        """Returns filename for band
        closest to target resolution"""

        bands_30m = ['B10', 'QA_PIXEL']
        if band not in bands_30m:
            raise ValueError("Band {} not supported.".format(band))

        return filenames[band][resolution]

    @ classmethod
    def build_products_df(cls, folder):

        products = []
        zones = [Path(x).stem for x in glob.glob(str(Path(folder) / '*'))]
        for zone in zones:
            tilespart1 = [Path(x).stem for x in glob.glob(
                str(Path(folder) / zone / '*'))]
            for tilepart1 in tilespart1:
                tilespart2 = [Path(x).stem for x in glob.glob(
                    str(Path(folder) / zone / tilepart1 / '*'))]
                for tilepart2 in tilespart2:
                    tileproducts = [x for x in glob.glob(
                        str(Path(folder) / zone /
                            tilepart1 / tilepart2 / '*' / '*' / '*'))]

                    for tileproduct in tileproducts:
                        filenames = glob.glob(
                            str(Path(tileproduct) / '*.tif'))
                        bands = ['_'.join(str(Path(x).stem).split('_')[
                            5:]).split('.')[0]
                            for x in filenames]
                        if 'QA_PIXEL' in bands:
                            products.append(tileproduct)
                        else:
                            logger.warning(
                                ('No QA_PIXEL file found for: '
                                 f'`{tileproduct}` -> Ignoring product'))

        entries = [cls.landsat8_entry(f) for f in products]
        if len(entries):
            df = pd.DataFrame(entries)
        else:
            df = pd.DataFrame([], columns=['date',
                                           'tile',
                                           'path',
                                           'level'])

        return df

    @ staticmethod
    def landsat8_entry(filename):
        """
        """
        basename = Path(filename).stem
        date = basename.split('_')[2]
        tile = Path(filename).stem.split('_')[-1]

        entry = dict(date=pd.to_datetime(date),
                     tile=tile,
                     level='L2SP',
                     path=filename)

        return entry

    def load_timeseries(self, *bands, **kwargs):
        '''
        Override default method to deal
        with scaling
        '''

        L8SCALING = {
            'B10': {
                'scale': 0.00341802,
                'offset': 149.0,
                'nodata': 0
            }
        }

        from satio.timeseries import load_timeseries
        from satio.timeseries import Timeseries

        ts = load_timeseries(self, *bands, **kwargs)

        scaled_ts = None

        for band in bands:
            scaling = L8SCALING.get(band, None)
            if scaling is not None:
                ts.data = ts.data.astype(np.float32)
                ts_banddata = ts[band].data
                ts_banddata[ts_banddata == scaling['nodata']] = np.nan
                ts_banddata = ((ts_banddata * scaling['scale'])
                               + scaling['offset'])
            else:
                ts_banddata = ts[band].data

            ts_band = Timeseries(
                data=ts_banddata,
                timestamps=ts.timestamps,
                bands=[band],
                attrs=ts.attrs
            )
            scaled_ts = (ts_band if scaled_ts is None
                         else scaled_ts.merge(ts_band))

        return scaled_ts


class L8ThermalTiledCollection(DiskCollection):

    sensor = 'L8'
    processing_level = 'L2SP'

    def __init__(self, df, s2grid=None):
        super().__init__(df, s2grid=s2grid)
        self.df = self.df[self.df.level == self.processing_level]

    @ property
    def supported_bands(self):
        return ['ST-B10', 'PIXEL-QA']

    @ property
    def supported_resolutions(self):
        return [10]

    @ classmethod
    def from_folders(cls, *folders, s2grid=None):
        df = build_L8ThermalTiled_products_df(*folders)
        df = df.sort_values('date', ascending=True)
        collection = cls(df, s2grid=s2grid)
        return collection

    def filter_nodata(self,
                      resolution=10,
                      mask_th=0.05,
                      min_keep_fraction=0.7,
                      min_acquisitions=75):
        """
        filter collection based on the relative amount
        of valid pixels. Minimum is specified by `mask_th`.
        e.g. a frame is valid if the mean of the binary mask
        is above `mask_th`. Or in other words,
        keep frames with more than `mask_th * 100` % valid pixels.

         `min_keep_fraction` is the minimum fraction of obs that will be
        retained, if needed by relaxing the the mask_th
        until the minimum required amount of acquisitions is met.

        `min_acquisitions` is the minimum of amount of within-swath
        acquisitions that are needed to even start filter_nodata process.
        """
        pixel_qa_10 = self.load(['PIXEL-QA'],
                                resolution=resolution)['PIXEL-QA']

        # Get a binary mask
        mask, _, _, _ = pixel_qa_mask(pixel_qa_10)

        # Count within swath pixels per obs
        within_swath = (pixel_qa_10 != 1).sum(axis=(1, 2)).astype(float)

        # Get relative amount of valid obs
        mask = mask.values.astype(int)

        valid_ids = select_valid_obs(mask, mask_th,
                                     min_keep_fraction, within_swath,
                                     min_acquisitions=min_acquisitions)

        return self._clone(df=self.df.iloc[valid_ids])

    def features_processor(self,
                           settings: Dict,
                           rsi_meta: Dict = {},
                           features_meta: Dict = {},
                           ignore_def_features: bool = False) -> L8FeatProc:
        return L8FeatProc(self,
                          settings,
                          rsi_meta=rsi_meta,
                          features_meta=features_meta,
                          ignore_def_features=ignore_def_features)

    def get_band_filenames(self, band, resolution):
        if self._filenames is None:
            self._filenames = self.df['path'].apply(
                lambda x: self.get_filenames(x))

        tif_filenames = self._filenames.apply(
            lambda x: self.get_band_filename(x, band, resolution))

        return tif_filenames.values.tolist()

    @staticmethod
    def get_filenames(L8_tiled_product):
        filenames_list = glob.glob(str(Path(L8_tiled_product) / '*.tif'),
                                   recursive=True)
        filenames = {}
        resolution = 10

        for f in filenames_list:

            band = str(Path(f).stem).split('_')[-2]

            filenames[band] = {resolution: f}

        return filenames

    @staticmethod
    def get_band_filename(filenames, band, resolution=10):
        """Returns filename for band
        closest to target resolution"""

        bands_10m = ['ST-B10', 'PIXEL-QA']

        if band not in bands_10m:
            raise ValueError("Band {} not supported.".format(band))

        return filenames[band][resolution]


class L8ThermalTrainingCollection(TrainingCollection):

    sensor = 'L8'
    processing_level = 'L2'

    def features_processor(self,
                           settings: Dict,
                           rsi_meta: Dict = {},
                           features_meta: Dict = {},
                           ignore_def_features: bool = False) -> L8FeatProc:
        return L8FeatProc(self,
                          settings,
                          rsi_meta=rsi_meta,
                          features_meta=features_meta,
                          ignore_def_features=ignore_def_features)

    def load(self,
             bands: List = None,
             resolution: int = None,
             location_id: int = None,
             mask_and_scale: bool = False) -> Dict:
        '''
        Override of parent class method to support native
        GEE-processed data
        '''

        if location_id is None:
            # get first location available
            location_id = self.location_ids[0]

        row = self.df[self.df.location_id == location_id].iloc[0]
        path = row['path']

        if not Path(path).is_dir():
            raise FileNotFoundError(f"{path} not found.")

        # Here we split up the worldcover and worldcereal approaches
        if self._dataformat == 'worldcereal':
            xarrs = self._load_worldcereal(row, bands,
                                           mask_and_scale=mask_and_scale)
        else:
            xarrs = self._load_ewoco(row, bands)

        # properly scale the data
        if ((bands[0] == 'ST-B10') and
                (xarrs[bands[0]].attrs['source'] == 'googleearthengine')):
            logger.warning('Applying manual scaling to L8 data!!')
            scaling = xarrs[bands[0]].attrs['scale_factor']
            offset = 149
            xarrs[bands[0]].data = np.round(
                (xarrs[bands[0]].data * scaling) + offset).astype(np.uint16)

        # replace bytestrings by strings in timestamp if needed
        if isinstance(xarrs[bands[0]].timestamp.values[0], bytes):
            xarrs[bands[0]] = xarrs[bands[0]].assign_coords(
                timestamp=[s.decode("utf-8") for s in
                           xarrs[bands[0]].timestamp.values])

        return xarrs


class WorldCerealDEMCollection(FolderCollection):
    '''Override native DEMCollection to be able to work
    with paths referring to s3 objects
    '''

    def __init__(self, folder, loader=None, s2grid=None):

        self.folder = folder

        self._loader = (loader if loader is not None
                        else ParallelLoader())
        self._s2grid = s2grid
        self._bounds = None
        self._epsg = None
        self._tile = None

    def _filename(self, tile, *args, **kwargs):
        return self.folder + f'/dem_{tile}.tif'


class AgERA5YearlyCollection(AgERA5Collection):

    sensor = 'AgERA5'

    def __init__(self, *args, **kwargs):
        """
        collection csv/dataframe should have the paths
        in column 'path' with /vsis3/ prefix.
        e.g.
        date      |    path
        20200101   /vsis3/ewoc-agera5-yearly/2020
        """
        super().__init__(*args, **kwargs)
        self._loader = EWOCGDALWarpLoader()

    def filter_dates(self, start_date, end_date):
        start_year = int(start_date[:4])
        end_year = int(end_date[:4]) + 1

        df = self.df[(self.df.date >= f'{start_year}0101')
                     & (self.df.date < f'{end_year}0101')]
        return self._clone(df=df, start_date=start_date, end_date=end_date)

    def get_band_filenames(self, band, resolution=None):
        filenames = self.df.apply(
            lambda x:
            f"{x.path}/AgERA5_{band}_"
            f'{x.date.year}.tif',
            axis=1)
        return filenames.values.tolist()

    def load_timeseries(self,
                        *bands,
                        resolution=100,
                        resampling='cubic',
                        **kwargs):
        return super().load_timeseries(*bands,
                                       resolution=resolution,
                                       resampling=resampling,
                                       **kwargs)


WCPC = TypeVar('WCPC', bound='WorldCerealProductCollection')


class WorldCerealProductCollection:
    """Collection to load WorldCereal products

    The "folder" can be either a base folder on disk
    or a s3 bucket prefix.
    """

    def __init__(self, folder, year, season, aez, product,
                 loader=None, s2grid=None,
                 access_key_id=None, secret_access_key=None):
        """Initialize WorldCerealProductCollection

        Args:
            folder (str): path to local products or s3 prefix
            year (int): reference year of the products
            season (str): season identifier of the products
            aez (int): AEZ id of the products
            product (str): Product ID
            access_key_id (str, optional): optional bucket access key id
            secret_access_key (str, optional): optional bucket secret key
        """

        self.year = year
        self.season = season
        self.aez = aez
        self.product = product

        if str(folder).startswith('s3'):
            # Setup an S3 collection
            self.folder = folder
            self.s3collection = True

            if access_key_id is None or secret_access_key is None:
                CLF_ACCESS_KEY_ID, CLF_SECRET_ACCESS_KEY = _get_s3_keys()
            self._loader = (loader if loader is not None
                            else EWOCS3ParallelLoader(
                                access_key_id=CLF_ACCESS_KEY_ID,
                                secret_access_key=CLF_SECRET_ACCESS_KEY,
                                fill_value=0))
        else:
            # Setup a disk collection
            self.folder = Path(folder)
            self.s3collection = False
            self._loader = (loader if loader is not None
                            else ParallelLoader())

        self._s2grid = s2grid
        self._bounds = None
        self._epsg = None
        self._tile = None

    @property
    def s2grid(self):
        if self._s2grid is None:
            self._s2grid = satio.layers.load('s2grid')
        return self._s2grid

    def _clone(self, bounds=None, epsg=None, tile=None) -> WCPC:
        new_cls = self.__class__(self.folder,
                                 self.year, self.season,
                                 self.aez, self.product,
                                 self._loader,
                                 self._s2grid)

        new_cls._bounds = bounds if bounds is not None else self._bounds
        new_cls._epsg = epsg if epsg is not None else self._epsg
        new_cls._tile = tile if tile is not None else self._tile

        return new_cls

    def filter_tile(self, tile) -> WCPC:
        return self._clone(tile=tile)

    def filter_bounds(self, bounds, epsg) -> WCPC:
        return self._clone(bounds=bounds, epsg=epsg)

    def load(self):

        tile, bounds = self._tile, self._bounds

        if (tile is None) | (bounds is None):
            raise ValueError(
                "`tile` and `bounds` need to be set first "
                "before loading data, by using the `filter_tile` "
                "and `filter_bounds` methods.")

        filename = self._filename(tile)
        arr = self._loader._load_array_bounds(filename, bounds)

        return arr

    @property
    def loader(self):
        if self._loader is None:
            self._loader = ParallelLoader()
        return self._loader

    @loader.setter
    def loader(self, value):
        self._loader = value

    def _filename(self, tile):

        if self.s3collection:
            return self._objectkey_bucket(tile)
        else:
            return self._filename_disk(tile)

    def _filename_disk(self, tile):

        filename = (self.folder / tile /
                    f'{self.year}_{self.season}' /
                    f'{self.year}_{self.season}'
                    f'_{self.aez}_{self.product}'
                    f'_classification_{tile}.tif')

        if not filename.is_file():
            raise FileNotFoundError(
                ('Could not find requested WorldCereal '
                 f'product: `{filename}`'))

        return filename

    def _objectkey_bucket(self, tile):
        objectkey = (
            '/'.join([self.folder, tile,
                      f'{self.year}_{self.season}',
                      f'{self.year}_{self.season}'
                      f'_{self.aez}_{self.product}'
                      f'_classification_{tile}.tif']))
        return objectkey


def get_S2_products(tile, startdate, enddate):
    logger.info(f'Getting S2 products for tile {tile} ...')

    # Make a query to the catalog to retrieve an example file
    response = requests.get((f'{_TS_S2_BASEURL}_TOC_V2&tileId={tile}'
                             '&cloudCover=[0,95]'
                             f'&start={startdate}&end={enddate}'
                             '&accessedFrom=MEP'))
    response = json.loads(response.text)

    if response['totalResults'] == 0:
        logger.warning('No Sentinel-2 products found.')
        return []

    products = []
    for acquisition in response['features']:
        for band in acquisition['properties']['links']['data']:
            if 'B08' in band['title']:
                products.append(str(Path(band['href'][7:]).parent))

    while len(products) != response['totalResults']:
        response = requests.get(
            response['properties']['links']['next'][0]['href'])
        response = json.loads(response.text)
        for acquisition in response['features']:
            for band in acquisition['properties']['links']['data']:
                if 'B08' in band['title']:
                    products.append(str(Path(band['href'][7:]).parent))
    return products


def get_S1_products(bbox, orbitpass, startdate, enddate):
    logger.info(f'Getting S1 products for bbox: {bbox}')
    logger.info(f'Considering orbit pass: {orbitpass}')

    if orbitpass not in ['ASCENDING', 'DESCENDING']:
        raise ValueError('Orbit pass should be one of ASCENDING/DESCENDING')

    # Make a query to the catalog to retrieve an example file
    response = requests.get((f'{_TS_S1_BASEURL}&bbox={bbox}'
                             f'&start={startdate}&end={enddate}'
                             f'&orbitDirection={orbitpass}'
                             '&accessedFrom=MEP'))
    response = json.loads(response.text)

    if response['totalResults'] == 0:
        logger.warning(f'No {orbitpass} Sentinel-1 products found.')
        return []

    products = []
    for acquisition in response['features']:
        for band in acquisition['properties']['links']['data']:
            if 'VV' in band['title']:
                products.append(str(Path(band['href'][7:]).parent))

    while len(products) != response['totalResults']:
        response = requests.get(
            response['properties']['links']['next'][0]['href'])
        response = json.loads(response.text)
        for acquisition in response['features']:
            for band in acquisition['properties']['links']['data']:
                if 'VV' in band['title']:
                    products.append(str(Path(band['href'][7:]).parent))

    return products


def get_s2_product_date(product):
    return pd.to_datetime(str(Path(product).stem).split('_')[1])


def get_s1_product_date(product):
    return pd.to_datetime(str(Path(product).stem).split('_')[5])


def create_products_df(products, tile, epsg, sensor='s2'):
    df = pd.DataFrame(columns=['product', 'date', 'path'],
                      index=None)

    if sensor == 's2':
        df['date'] = [get_s2_product_date(x) for x in products]
    elif sensor == 's1':
        df['date'] = [get_s1_product_date(x) for x in products]
    else:
        raise ValueError(f'Unrecognized sensor: {sensor}')

    df['product'] = [str(Path(product).stem) for product in products]
    df['path'] = products
    df['tile'] = tile
    df['epsg'] = epsg
    df = df.sort_values('date', axis=0)

    return df


def landsat8_entry(filename):
    """
    """
    basename = Path(filename).stem
    date = basename.split('_')[2]
    tile = Path(filename).parts[-2]

    entry = dict(date=datetime.strptime(date, '%Y%m%d'),
                 tile=tile,
                 level='L2SP',
                 path=filename)

    return entry


def _build_L8ThermalTiled_products_df(folder, tiles=None):

    products = []
    if tiles is None:
        tiles = [Path(x).stem for x in glob.glob(str(Path(folder) / '*'))]
    for tile in tiles:
        tileproducts = [Path(x).stem for x in glob.glob(
            str(Path(folder) / tile / '*'))]

        for tileproduct in tileproducts:
            filenames = glob.glob(str(Path(folder) / tile /
                                      tileproduct / '*.tif'))
            bands = [Path(x).stem.split('_')[-2] for x in filenames]
            if 'PIXEL-QA' in bands and 'ST-B10' in bands:
                products.append(str(Path(folder) / tile / tileproduct))

    entries = [landsat8_entry(f) for f in products]
    if len(entries):
        df = pd.DataFrame(entries)
    else:
        df = pd.DataFrame([], columns=['date',
                                       'tile',
                                       'path',
                                       'level'])

    return df


def build_L8ThermalTiled_products_df(*folders):
    dfs_list = [_build_L8ThermalTiled_products_df(folder) for
                folder in folders]
    return pd.concat(dfs_list, axis=0)


def _get_creo_optical_entry(fn):
    product = fn.split('/')[-1]
    split = product.split('_')
    tile = split[-1]
    entry = dict(
        product=product,
        date=parse(split[2]),
        path=fn,
        tile=tile,
        epsg=tile_to_epsg(tile),
        level='L2A'
    )

    return entry


def _get_bucket_products(url):
    prefix = 'WORLDCEREAL_PREPROC/OPTICAL/'
    resp = requests.get(url)
    body = resp.content.decode()
    body = body.split('\n')
    files_keys = filter(lambda x: prefix in x, body)
    products = list(set(map(lambda x: url + "/".join(x.split("/")[:-1]),
                            files_keys)))
    return products


def build_opticalcreo_products_df(url):
    products = _get_bucket_products(url)
    entries = list(map(_get_creo_optical_entry, products))
    return pd.DataFrame(entries)


if __name__ == '__main__':
    # url = ('https://cf2.cloudferro.com:8080/swift/v1'
    #        '/AUTH_b33f63f311844f2fbf62c5741ff0f734/world-cereal/')
    # coll = WorldCerealOpticalCREOCollection.from_url(url)

    coll = WorldCerealSigma0TiledCollection.from_folder(
        '/data/worldcereal/runs/largescaletest/WORLDCEREAL_PREPROC/SAR'
    )

    from satio.grid import get_blocks_gdf
    blocks = get_blocks_gdf(['30SVH'], coll._s2grid)

    b = blocks[blocks.block_id == 99]
    coll = coll.filter_bounds(b['bounds'].values[0], int(b.epsg)).filter_dates(
        '2018-11-10', '2019-11-10')

    # b = blocks.iloc[0]
    # coll = coll.filter_bounds(b.bounds, b.epsg).filter_dates(
    #     '2018-09-04', '2018-09-15')
    # f = coll.get_band_filenames('MASK')[0]
    # src = rasterio.open(f)
    # pass
    coll = WorldCerealSigma0TiledCollection.from_folder(
        '/data/worldcereal/runs/largescaletest/WORLDCEREAL_PREPROC/SAR'
    ).filter_bounds(
        bounds=(300000, 4189760, 310240, 4200000),
        epsg=32630)
    print(coll)
