import glob
from pathlib import Path
import shutil

from loguru import logger
import numpy as np
import rasterio
from satio.utils.geotiff import write_geotiff
import xarray as xr

from worldcereal.utils import aez


class ConflictResolver:

    def __init__(self, block_folder, tile,
                 block_id, aez_id, season, year,
                 nodata=255):
        self.block_folder = block_folder
        self.tile = tile
        self.block_id = block_id
        self.aez_id = aez_id
        self.season = season
        self.year = year
        self.nodata = nodata
        self.target_value = 100
        self.marker_value = 100
        self.aez_layer = aez.load()

    # TODO: find overlapping seasons!

    def run(self):
        logger.info('-' * 50)
        logger.info('Resolving conflicts ...')

        products = self._list_products()

        if products is None:
            logger.warning('No products to fix.')
            return

        logger.info(f'Found {len(products.keys())} products to verify.')

        # First get irrigation product
        irrigation = products.pop('irrigation', None)

        # Check if activecropland product was found and correct it
        activecropland = products.pop('activecropland', None)
        if activecropland is not None:
            self.harmonize_activecropland(
                activecropland['classification'], products)

        # Harmonize irr with corrected active marker
        if (activecropland is not None) & (irrigation is not None):
            self.harmonize_irrigation(irrigation['classification'],
                                      activecropland['classification'])

        # # Harmonize the different products
        # # only needed if more than one crop types and one of them is maize
        if 'maize' in products.keys() and len(products) > 1:
            self.harmonize_products(products)

        logger.info('Harmonization process finished!')

    def _list_products(self, include_overlapping_seasons=False):

        # TODO: NEED TO LOOK FOR PRODUCTS FROM OVERLAPPING SEASONS!

        if include_overlapping_seasons:
            raise NotImplementedError('Overlapping seasons not suppported.')

        # self.aez_layer.set_index('zoneID').loc[self.aez_id]

        # from worldcereal.seasons import get_processing_dates
        # from worldcereal import SUPPORTED_SEASONS
        # seasons_to_check = [s for s in SUPPORTED_SEASONS if
        #  'annual' not in s and 'custom' not in s]

        year_season = f'{self.year}_{self.season}'
        base_folder = self.block_folder / self.tile / year_season

        products_folders = glob.glob(str(base_folder / '*'))
        # avoid including feature directories here
        products_folders = [pf for pf in products_folders if 'features_' not in Path(pf).name]
        products = [Path(x).name for x in products_folders]

        if len(products) == 0:
            logger.warning(f'No products found under `{base_folder}`!')
            return None

        product_list = {}

        for folder, product in zip(products_folders, products):
            if product == 'activecropland':
                product_list['activecropland'] = {
                    'classification': Path(folder) / 'classification' / '_'.join([self.tile, str(self.aez_id), f'{self.block_id:03d}', 'classification.tif'])  # NOQA
                }
            else:
                product_list[product] = {
                    'classification': Path(folder) / 'classification' / '_'.join([self.tile, str(self.aez_id), f'{self.block_id:03d}', 'classification.tif']),  # NOQA
                    'confidence': Path(folder) / 'confidence' / '_'.join([self.tile, str(self.aez_id), f'{self.block_id:03d}', 'confidence.tif']),  # NOQA
                }

        for k1, v1 in product_list.items():
            for k2, file in v1.items():
                if not file.is_file():
                    raise FileNotFoundError(f'Expected file `{file} not found.')  # NOQA

        return product_list

    def harmonize_products(self, products):

        # TODO: LOOK AT PRODUCTS FROM OVERLAPPING SEASONS

        # TODO: make code more generic
        # (now focused on getting rid of conflicts between maize
        # and other crop types)

        logger.info(f'Harmonizing products: {list(products.keys())}')

        # Load all crop type maps
        class_cube, conf_cube = self._load_product_cubes(products)
        class_cube = class_cube.to_array(dim='products')
        conf_cube = conf_cube.to_array(dim='confidence').astype(np.int16)

        # Compute max confidence across products for each pixel
        # (ignoring no data values)
        conf_cube.values[conf_cube.values == self.nodata] = -999
        max_conf = conf_cube.max(dim='confidence')
        max_conf.values[max_conf.values == -999] = self.nodata
        conf_cube.values[conf_cube.values == -999] = self.nodata

        # Make a copy of the original class cube
        harmonized_class = class_cube.copy(deep=True)

        # locate conflicts
        conflicts = class_cube.copy(deep=True)
        conflicts.values[conflicts.values == self.nodata] = 0
        conflicts = conflicts.sum(dim='products')
        conflicts = conflicts > 100
        # only if maize is involved in conflicts, it should be corrected
        conflicts = xr.where(class_cube.sel(products='maize') != 100,
                             False, conflicts)
        logger.info(f'Found {conflicts.values.sum()} pixels with conflicts!')

        if conflicts.values.sum() > 0:
            # locate conflicts where maize has highest confidence
            maize = xr.where(conflicts &
                             (conf_cube.sel(confidence='maize') == max_conf),
                             True,
                             False)
            # adjust maize in conflict pixels
            harmonized_class.loc['maize'] = xr.where(
                conflicts & (~maize), 0, class_cube.sel(products='maize'))
            # adjust other pixels in conflict pixels
            other_products = [p for p in list(products.keys()) if 'maize' not in p]
            for op in other_products:
                harmonized_class.loc[op] = xr.where(
                    conflicts & maize, 0, class_cube.sel(products=op))

            # save corrected classification
            for product in list(products.keys()):
                self._to_geotiff(harmonized_class.sel(products=product).values,
                                 products[product]['classification'])

    def harmonize_activecropland(self, activecropland, products):

        # TODO: LOOK AT PRODUCTS FROM OVERLAPPING SEASONS

        logger.info('Correcting active cropland marker ...')

        # Load active marker data
        activecropland_data = self._load_file(activecropland)

        # Load all crop type maps
        classification_cube, _ = self._load_product_cubes(products)

        # Find out in which pixels there was a crop detected
        crop_detected = (classification_cube.to_array(
            dim='products') == self.target_value).sum(dim='products') > 0

        # If we detected a crop, we need to put active marker True
        nr_correct = ((activecropland_data == self.marker_value) & crop_detected).values.sum()  # NOQA
        nr_adj = ((activecropland_data != self.marker_value) & crop_detected).values.sum()  # NOQA
        logger.info((f'{nr_adj} crop pixels have been corrected to active, '
                     f'{nr_correct} were already correct.'))
        activecropland_data[crop_detected] = self.marker_value

        self._to_geotiff(activecropland_data, activecropland)

    def harmonize_irrigation(self, irrigation, activecropland):

        logger.info('Correcting irrigation product ...')

        # Load active marker data
        activecropland_data = self._load_file(activecropland)

        # Load irrigation data
        irrigation_data = self._load_file(irrigation)

        # Find out in which pixels there was irrigation detected
        irr_detected = irrigation_data == self.target_value

        # If we detected irr, but active marker is False, we need to put irr to rainfed
        to_adj = (activecropland_data != self.marker_value) & irr_detected
        irrigation_data[to_adj] = 0

        nr_adj = np.sum(to_adj)
        logger.info((f'{nr_adj} irr pixels have been corrected to rainfed'))

        self._to_geotiff(irrigation_data, irrigation)

    def _load_product_cubes(self, products):
        logger.info('Loading classification and confidence cubes ...')
        classification_cube = xr.Dataset()
        confidence_cube = xr.Dataset()

        for product in products.keys():
            classification_cube[product] = (
                ['x', 'y'], self._load_file(
                    products[product]['classification']))
            confidence_cube[product] = (
                ['x', 'y'], self._load_file(
                    products[product]['confidence']))

        return classification_cube, confidence_cube

    @staticmethod
    def _load_file(file):
        with rasterio.open(file) as src:
            return src.read(1)

    @ staticmethod
    def _to_geotiff(data, filename):

        with rasterio.open(filename) as src:
            colormap = src.colormap(1)
            profile = src.profile
            bands = eval(src.tags()['bands'])
            nodata = src.nodata

        logger.debug(f"Saving corrected file: {filename}")

        # Write to temporary file in current directory
        tempfile = f'./{Path(filename).name}'

        write_geotiff(data, profile, tempfile,
                      band_names=bands, colormap=colormap,
                      nodata=nodata)

        # Now copy the file
        logger.info('Copying to final destination ...')
        shutil.move(tempfile, filename)

        logger.info('Done saving.')
