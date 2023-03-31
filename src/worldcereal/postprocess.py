import glob
import os
from typing import Dict
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from loguru import logger
import matplotlib.cm
import numpy as np
import rasterio
import satio
from satio.utils import random_string
from satio import layers
from satio.utils.buildvrt import build_vrt

import worldcereal
from worldcereal import metadata
from worldcereal import SUPPORTED_SEASONS
from worldcereal.utils import aez


def get_confidence_cmap():
    colormap = matplotlib.cm.get_cmap('RdYlGn')
    cmap = {}
    for i in range(101):
        cmap[i] = tuple((np.array(colormap(int(2.55 * i))) * 255).astype(int))
    return cmap


COLORMAP = {
    'activecropland': {
        0: (232, 55, 39, 255),  # inactive
        100: (77, 216, 39, 255),  # active
    },
    'temporarycrops': {
        0: (186, 186, 186, 0),  # no temporary crops
        100: (224, 24, 28, 200),  # temporary crops
    },
    'winterwheat': {
        0: (186, 186, 186, 255),  # other
        100: (186, 113, 53, 255),  # wheat
        # 250: (2, 11, 74, 255)   # block too cold
    },
    'wintercereals': {
        0: (186, 186, 186, 255),  # other
        100: (186, 113, 53, 255),  # wheat
        # 250: (2, 11, 74, 255)   # block too cold
    },
    'springwheat': {
        0: (186, 186, 186, 255),  # other
        100: (186, 113, 53, 255),  # wheat
        # 250: (2, 11, 74, 255)   # block too cold
    },
    'springcereals': {
        0: (186, 186, 186, 255),  # other
        100: (186, 113, 53, 255),  # wheat
        # 250: (2, 11, 74, 255)   # block too cold
    },
    'wheat': {
        0: (186, 186, 186, 255),  # other
        100: (186, 113, 53, 255),  # wheat
        # 250: (2, 11, 74, 255)   # block too cold
    },
    'cereals': {
        0: (186, 186, 186, 255),  # other
        100: (186, 113, 53, 255),  # wheat
        # 250: (2, 11, 74, 255)   # block too cold
    },
    'maize': {
        0: (186, 186, 186, 255),  # other
        100: (252, 207, 5, 255),  # maize
        # 250: (2, 11, 74, 255),   # block too cold
    },
    'rapeseed': {
        0: (186, 186, 186, 255),  # other
        100: (167, 245, 66, 255),  # rapeseed
        # 250: (2, 11, 74, 255),   # block too cold
    },
    'sunflower': {
        0: (186, 186, 186, 255),  # other
        100: (245, 66, 111, 255),  # sunflower
        # 250: (2, 11, 74, 255),   # block too cold
    },
    'irrigation': {
        0: (255, 132, 132, 255),  # rainfed
        100: (104, 149, 197, 255)  # irrigated
    },
    'confidence': get_confidence_cmap(),
    'metafeatures': None,
    'inputfeatures': None
}
NODATA = {
    'activecropland': 255,
    'temporarycrops': 255,
    'winterwheat': 255,
    'springwheat': 255,
    'wintercereals': 255,
    'springcereals': 255,
    'wheat': 255,
    'cereals': 255,
    'maize': 255,
    'sunflower': 255,
    'rapeseed': 255,
    'irrigation': 255,
    'confidence': 255,
    'metafeatures': 255,
    'inputfeatures': None
}

S2_GRID = layers.load('s2grid')


def gdal_translate_cog(fn, out_fn, tile, gdal_cachemax=2000,
                       nodata=None, resampling='MODE',
                       overview_resampling='MODE'):
    '''Method to convert an existing TIF or VRT to COG using
    system gdal_translate (>= 3.1)
    '''

    logger.debug(f'Using `{resampling}` resampling method')
    logger.debug(f'Using `{overview_resampling}` overview_resampling method')

    if nodata is None:
        nodata = 'none'

    # Get the bounds for this tile
    bounds = S2_GRID.set_index('tile').loc[tile].bounds
    bounds = [bounds[0], bounds[3], bounds[2], bounds[1]]
    bounds = ' '.join([str(x) for x in bounds])

    bin = str(Path(sys.executable).parent / 'gdal_translate')

    if not Path(bin).is_file():
        # Fall-back to system gdal
        bin = '/bin/gdal_translate'
        if not Path(bin).is_file():
            raise FileNotFoundError(
                'Could not find a GDAL installation.')
    os.environ['GDAL_CACHEMAX'] = f"{gdal_cachemax}"
    cmd = (f"{bin} "
           "-of COG "
           f"-a_nodata {nodata} "
           f"-projwin {bounds} "
           "-co COMPRESS=DEFLATE "
           "-co BLOCKSIZE=1024 "
           f"-co RESAMPLING={resampling} "
           f"-co OVERVIEW_RESAMPLING={overview_resampling} "
           "-co OVERVIEWS=IGNORE_EXISTING "
           f"-mo WORLDCEREAL_VERSION={worldcereal.__version__} "
           f"-mo SATIO_VERSION={satio.__version__} "
           ""
           f"{fn} {out_fn}")
    logger.debug(cmd)
    p = subprocess.run(cmd.split())
    if p.returncode != 0:
        raise IOError("COG creation failed")


class PostProcessor:

    def __init__(self, blocks_folder: str, output_folder: str,
                 tile: str, year: int, season: str,
                 aez_id: int, products: list = None,
                 parameters: Dict = None):
        """Prostprocessing methods

        Args:
            blocks_folder (str): input directory where blocks are stored
            output_folder (str): output directory to store postprocessed COGs
            tile (str): Sentinel-2 tile for which COG will
                        be generated (e.g. 31UFS)
            year (int): year for which product is valid (e.g. 2020)
            season (str): season for which product is valid (e.g. `winter`)
            aez_id (int): Zone ID of the AEZ of the product (e.g. 46172)
            products (list): list of products for which to generate COGs
                             (e.g. ['wheat', 'maize'])
            parameters (Dict): Postprocessing parameters

        """

        self.blocks_folder = Path(blocks_folder)
        self.output_folder = Path(output_folder)
        self.products = products
        self.tile = tile
        self.active_marker = parameters.get('active_marker', False)
        if self.active_marker:
            self.products.extend(['activecropland'])
        self.parameters = parameters

        self.year = year
        self.aez = aez_id
        self.season = season
        self.tempfolder = Path(tempfile.TemporaryDirectory().name)

        logger.info('-' * 50)
        logger.info('Starting PostProcessor:')
        self._check_season(season)
        self._check_aez(aez_id)
        self._check_products(products)

        pass

    def run(self, generate_metadata=True, **kwargs):

        for product in self.products:

            if product == 'springwheat' or product == 'springcereals':
                from worldcereal.utils.aez import load
                aez_df = load()
                if aez_df.set_index('zoneID').loc[
                        self.aez]['trigger_sw'] == 0:
                    logger.warning(f'`{product}` postprocessing skipped based '
                                   'on `trigger_sw` flag for this AEZ.')
                    continue

            for name in ['classification',
                         'confidence',
                         'metafeatures',
                         #  'inputfeatures'
                         ]:
                # for activecropland, there is no confidence layer!
                if (product == 'activecropland') and (name == 'confidence'):
                    continue
                self._postprocess_product(product, name, **kwargs)

            if generate_metadata:
                self.generate_metadata(product, **kwargs)

    def _postprocess_product(self, product, name,
                             skip_processed=False,
                             in_memory=True,
                             debug=False, **kwargs):

        tif_folder = (self.blocks_folder / self.tile /
                      f'{self.year}_{self.season}' /
                      product / name)
        out_fn = (self.output_folder / self.tile /
                  f'{self.year}_{self.season}' /
                  '_'.join([str(self.year), self.season,
                            str(self.aez), product,
                            name, self.tile + '.tif']))

        out_fn.parent.mkdir(parents=True, exist_ok=True)

        # If we're looking at the classification itself we'll get
        # the options from the product.
        if out_fn.is_file() and skip_processed:
            logger.warning(f'{out_fn} already exists, skipping')
        elif not Path(tif_folder).is_dir():
            logger.warning(f'{tif_folder} does not exist, skipping')
        else:
            if out_fn.is_file() and not skip_processed:
                logger.warning(f'{out_fn} exists, removing')
                out_fn.unlink()

        identifier = name if name in COLORMAP.keys() else product
        colormap = self._get_colormap(identifier)
        nodata = self._get_nodata(identifier)
        resampling = self._get_resampling(identifier)
        overview_resampling = self._get_resampling(identifier)

        self._create_cog(tif_folder, out_fn,
                         colormap, nodata,
                         debug=debug, in_memory=in_memory,
                         resampling=resampling,
                         overview_resampling=overview_resampling)

    def _get_colormap(self, identifier):
        if identifier.split('-')[0] in COLORMAP.keys():
            colormap = COLORMAP[identifier.split('-')[0]]
        else:
            logger.warning((f'Unknown identifier `{identifier}`: '
                            'cannot assign colormap'))
            colormap = None

        return colormap

    def _get_nodata(self, identifier):
        if identifier.split('-')[0] in NODATA.keys():
            nodata = NODATA[identifier.split('-')[0]]
        else:
            logger.warning((f'Unknown identifier `{identifier}`: '
                            'cannot assign nodata value'))
            nodata = None

        return nodata

    def _get_resampling(self, identifier):
        if identifier.split('-')[0] == 'confidence':
            return 'AVERAGE'
        else:
            return 'MODE'

    def _create_cog(self, tif_folder, out_fn,
                    colormap, nodata,
                    debug=False, in_memory=True,
                    **options):
        try:
            tif_files = glob.glob(str(tif_folder / '*.tif'))

            if len(tif_files) == 0:
                msg = ('Cannot create COG: No TIF files '
                       f'found in `{tif_folder}`.')
                if 'classification' in str(out_fn):
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)
                return

            ex_file = tif_files[0]
            with rasterio.open(ex_file) as src:
                band_names = eval(src.get_tag_item('bands'))

            # Write to temporary file in current directory
            tempfile = self.tempfolder / f'{Path(out_fn).name}'

            self._cog_from_folder(tif_folder,
                                  tempfile,
                                  colormap=colormap,
                                  nodata=nodata,
                                  in_memory=in_memory,
                                  band_names=band_names,
                                  **options)

            # Now copy the file
            logger.info('Copying to final destination ...')
            shutil.copy2(tempfile, out_fn)
        except Exception as e:
            logger.error(f'{e}')
            if debug:
                raise

    def _cog_from_folder(self,
                         folder,
                         cog_filename,
                         create_folder=True,
                         nodata=None,
                         colormap=None,
                         band_names=None,
                         in_memory=True,
                         **options):

        cog_filename = Path(cog_filename)
        vrt_fname = None

        if cog_filename.is_file():
            cog_filename.unlink()

        if create_folder:
            cog_filename.parent.mkdir(parents=True, exist_ok=True)

        try:
            vrt_fname = self.tempfolder / f'_tmpvrt_{random_string()}.vrt'
            logger.info(f"Generating {cog_filename} from {folder}")

            # Infer tile bounds so output file has always
            # full tile extent
            bounds = S2_GRID.set_index('tile').loc[self.tile].bounds
            build_vrt(vrt_fname, folder, outputBounds=bounds,
                      nodata_value=nodata)

            # Set band names and colormap to vrt
            # before converting to COG
            if band_names is not None and band_names != []:
                with rasterio.open(vrt_fname, 'r+') as src:
                    src.update_tags(bands=band_names)
                    for i, b in enumerate(band_names):
                        src.update_tags(i + 1, band_name=b)

            if colormap is not None:
                with rasterio.open(vrt_fname, 'r+') as src:
                    src.write_colormap(1, colormap)

            # Add software versions as metadata
            with rasterio.open(vrt_fname, 'r+') as src:
                src.update_tags(
                    WORLDCEREAL_VERSION=worldcereal.__version__,
                    SATIO_VERSION=satio.__version__
                )

            # Convert to COG using system gdal_translate
            gdal_translate_cog(vrt_fname,
                               cog_filename,
                               self.tile,
                               nodata=nodata,
                               **options)

        except Exception as e:
            logger.error(f"Error on creating COG from folder {folder}:\n{e}")
            raise
        finally:
            if Path(vrt_fname).is_file():
                Path(vrt_fname).unlink()

    def generate_metadata(self, product, skip_processed=True,
                          user='0000', public=True, **kwargs):

        logger.info('Generating metadata ...')

        metadatafile = (
            self.output_folder / self.tile /
            f'{self.year}_{self.season}' /
            '_'.join([str(self.year), self.season,
                      str(self.aez),
                      product, 'metadata', self.tile + '.json']))

        classificationfile = (
            self.output_folder / self.tile /
            f'{self.year}_{self.season}' /
            '_'.join([str(self.year), self.season,
                      str(self.aez), product,
                      'classification', self.tile + '.tif']))

        if metadatafile.is_file() and skip_processed:
            logger.warning(f'{metadatafile} already exists, skipping')

        if not classificationfile.is_file():
            msg = (f'Classification file {classificationfile} does not '
                   'exist so metadata cannot be generated.')
            logger.error(msg)
            return

        metadata.generate(metadatafile, classificationfile,
                          product, user, public, self.parameters)

    @staticmethod
    def _check_season(season):
        if season not in SUPPORTED_SEASONS:
            raise ValueError(f'Season `{season}` not supported.')
        logger.info(f'Season: {season}')

    @staticmethod
    def _check_aez(aez_id):
        aez_df = aez.load()
        if aez_id not in aez_df['zoneID'].unique():
            raise ValueError(f'AEZ id `{aez_id}` unknown.')
        logger.info(f'AEZ Zone ID: {aez_id}')

    @staticmethod
    def _check_products(products):
        logger.info('-' * 25)
        logger.info('Considering following products:')
        for product in products:
            logger.info(product)
        logger.info('-' * 25)
