from datetime import datetime, timezone
import json
import os

from loguru import logger
from pathlib import Path
import rasterio
import satio
from shapely.geometry import box

import worldcereal
from worldcereal.utils.aez import group_from_id
from worldcereal.classification.models import WorldCerealModel
from worldcereal.utils import COLL_MAXGAP


TEMPLATE = (Path(__file__).parent / 'resources'
            / 'STAC_metadata_template.json')


def _get_product_title(product):

    if product == 'temporarycrops':
        title = 'ESA WorldCereal temporary crop extent product'
    elif product in ['maize', 'wintercereals', 'winterwheat',
                     'springcereals', 'springwheat']:
        title = f'ESA WorldCereal seasonal crop type product ({product})'
    elif product == 'irrigation':
        title = 'ESA WorldCereal seasonal irrigation product'
    elif product == 'activecropland':
        title = 'ESA Worldcereal seasonal active cropland marker'
    else:
        title = 'Unknown product'

    return title


def _get_product_description(product, season):

    if product == 'temporarycrops':
        descr = ('Pixel-based, binary classification product at 10m spatial resolution distinguishing '
                 'temporary crops from all other types of land cover. '
                 'Pixels labeled as temporary crops have at least one crop with a less-than-one-year '
                 'growing cycle, sowed/planted and fully harvestable within the 12 months after '
                 'the sowing/planting date. Some crops that remain in the field for more than one '
                 'year are also considered as temporary crops, such as sugar cane and cassava. '
                 'The WorldCereal temporary crop maps exclude perennial crops as well as '
                 '(temporary) pastures.')

    elif product in ['maize', 'wintercereals', 'winterwheat',
                     'springcereals', 'springwheat']:
        descr = ('Pixel-based, binary classification product at 10m spatial resolution indicating '
                 f'the presence of {product} in the {season} season. '
                 'This product is generated within one month after the growing season. '
                 'Areas not identified as annual cropland by the '
                 'annual cropland extent product are ignored.')
    elif product == 'irrigation':
        descr = ('Pixel-based, binary classification product at 10m spatial resolution indicating '
                 f'whether or not a pixel has been actively irrigated in the {season} season. '
                 'Actively irrigated areas are defined as areas where scheduled irrigation is '
                 'needed for a crop to grow given the climatic conditions, or in other words, '
                 'when naturally occuring rainfall is not enough to produce the crop. '
                 'Supplementary irrigation is not included in this definition. '
                 'This product is generated within one month after the growing season. '
                 'Areas not identified as annual cropland by the '
                 'annual cropland extent product are ignored.')
    elif product == 'activecropland':
        descr = ('Pixel-based, binary classification product at 10m spatial resolution indicating '
                 f'whether or not any crop has been grown during the {season} season. '
                 'This product is generated within one month after the end of the growing season. '
                 'Areas not identified as annual cropland by the '
                 'annual cropland extent product are ignored.')
    else:
        descr = 'Unknown'

    return descr


def _get_modelname(models, product):
    modelpath = models[product]
    if 'Zone' in modelpath or 'Group' in modelpath:
        return '_'.join(Path(models[product]).parts[-3:-1])
    elif 'realm' in modelpath.lower():
        return '_'.join(Path(models[product]).parts[-3:-1])
    else:
        return Path(modelpath).parent.name


def generate(metadatafile, classificationfile, product,
             user, public, parameters):

    meta = json.load(open(TEMPLATE, 'r'))

    with rasterio.open(classificationfile) as src:

        # Immediately transform bbox to WGS84
        bbox = list(rasterio.warp.transform_bounds(
            src.crs, 'epsg:4326', *src.bounds))
        coordinates = [[list(x) for x in box(*bbox).exterior.coords[:]]]
        epsg = src.crs.to_epsg()
        shape = list(src.shape)

    tile = Path(classificationfile).stem.split('_')[-1]

    id = Path(classificationfile).stem
    meta['id'] = id
    meta['bbox'] = bbox
    meta['geometry']['coordinates'] = coordinates
    meta['collection'] = f'esa-worldcereal-{product}'
    meta['links'][0]['href'] = str(metadatafile)

    # Properties
    meta['properties']['title'] = _get_product_title(product)
    meta['properties']['description'] = _get_product_description(
        product, parameters["season"])
    meta['properties']['datetime'] = datetime.fromtimestamp(os.path.getmtime(
        classificationfile), tz=timezone.utc).isoformat(
            timespec='milliseconds').replace("+00:00", "Z")
    meta['properties']['created'] = datetime.fromtimestamp(os.path.getmtime(
        classificationfile), tz=timezone.utc).isoformat(
            timespec='milliseconds').replace("+00:00", "Z")
    meta['properties']['updated'] = datetime.fromtimestamp(os.path.getmtime(
        classificationfile), tz=timezone.utc).isoformat(
            timespec='milliseconds').replace("+00:00", "Z")

    meta['properties']['tile_collection_id'] = (
        f'{parameters["year"]}_{parameters["season"]}_'
        f'{parameters["aez"]}_{product}_{user}')

    meta['properties']['license'] = 'TBD'
    meta['properties']['rd:sat_id'] = 'Sentinel-2,Sentinel-1'
    meta['properties']['mgrs:utm_zone'] = int(tile[:2])
    meta['properties']['mgrs:latitude_band'] = tile[2]
    meta['properties']['mgrs:grid_square'] = tile[3:]
    meta['properties']['proj:epsg'] = epsg
    meta['properties']['proj:shape'] = shape
    meta['properties']['instruments'] = ["Sentinel-2 MSI",
                                         "Sentinel-1 C-band SAR"]
    meta['properties']['start_datetime'] = parameters['start_date']
    meta['properties']['end_datetime'] = parameters['end_date']
    meta['properties']['season'] = parameters['season']
    meta['properties']['aez_id'] = int(parameters['aez'])
    meta['properties']['aez_group'] = group_from_id(parameters['aez'])
    meta['properties']['inputcollections_maxgap'] = COLL_MAXGAP

    # get information on the model
    if product in parameters['models'].keys():
        meta['properties']['model'] = _get_modelname(parameters['models'], product)
        model = WorldCerealModel.from_config(parameters['models'][product])
        ref_ids = model.config.get('training_refids', None)
    elif (parameters.get('irrmodels', None) is not None) and (
            product in parameters['irrmodels'].keys()):
        meta['properties']['model'] = _get_modelname(parameters['irrmodels'], product)
        model = WorldCerealModel.from_config(parameters['irrmodels'][product])
        ref_ids = model.config.get('training_refids', None)
    else:
        # for instance for activecropland there is no model
        meta['properties']['model'] = 'None'
        ref_ids = None

    if ref_ids is not None:
        meta['properties']['training_refids'] = ref_ids

    meta['properties']['users'] = [user]
    meta['properties']['product'] = product
    meta['properties']['type'] = 'map'
    meta['properties']['public'] = str(public).lower()
    meta['properties']['related_products'] = [meta['id']]
    meta['properties']['software_version'] = {
        'worldcereal': worldcereal.__version__,
        'satio': satio.__version__
    }

    # Assets
    meta['assets']['product']['href'] = str(classificationfile)
    meta['assets']['product']['title'] = f'{product} classification'

    metafeatures = str(classificationfile).replace(
        'classification', 'metafeatures'
    )
    if Path(metafeatures).is_file():
        meta['assets']['metafeatures']['href'] = metafeatures
        meta['assets']['metafeatures']['title'] = f'{product} meta-features'
    else:
        meta['assets'].pop('metafeatures')

    confidence = str(classificationfile).replace(
        'classification', 'confidence'
    )
    if Path(confidence).is_file():
        meta['assets']['confidence']['href'] = confidence
        meta['assets']['confidence']['title'] = f'{product} confidence'
    else:
        meta['assets'].pop('confidence')

    with open(metadatafile, 'w') as f:
        json.dump(meta, f, indent=4)

    logger.info(f'Metadata file saved to: {metadatafile}')


def generate_for_merged_aez(metadatafile, classificationfile,
                            examplemetafile):

    meta = json.load(open(examplemetafile, 'r'))

    with rasterio.open(classificationfile) as src:

        # Immediately transform bbox to WGS84
        bbox = list(rasterio.warp.transform_bounds(
            src.crs, 'epsg:4326', *src.bounds))
        coordinates = [list(x) for x in box(*bbox).exterior.coords[:]]
        epsg = src.crs.to_epsg()
        shape = list(src.shape)

    id = Path(classificationfile).stem
    meta['id'] = id
    meta['bbox'] = bbox
    meta['geometry']['coordinates'] = coordinates
    meta['links'][0]['href'] = str(metadatafile)

    # Properties
    meta['properties'].pop('mgrs:utm_zone')
    meta['properties'].pop('mgrs:latitude_band')
    meta['properties'].pop('mgrs:grid_square')

    meta['properties']['proj:epsg'] = epsg
    meta['properties']['proj:shape'] = shape

    # Assets
    meta['assets']['product']['href'] = str(classificationfile)

    metafeatures = str(classificationfile).replace(
        'classification', 'metafeatures'
    )
    if Path(metafeatures).is_file():
        meta['assets']['metafeatures']['href'] = metafeatures
    else:
        if 'metafeatures' in meta['assets'].keys():
            meta['assets'].pop('metafeatures')

    confidence = str(classificationfile).replace(
        'classification', 'confidence'
    )
    if Path(confidence).is_file():
        meta['assets']['confidence']['href'] = confidence
    else:
        if 'confidence' in meta['assets'].keys():
            meta['assets'].pop('confidence')
    with open(metadatafile, 'w') as f:
        json.dump(meta, f, indent=4)

    logger.info(f'Metadata file saved to: {metadatafile}')


def update_paths(infile, outfile, key):

    meta = json.load(open(infile, 'r'))

    # Make the path adjustments
    metalink = key.replace('classification', 'metadata').replace(
        '.tif', '.json'
    )
    classificationlink = key
    metafeatureslink = key.replace('classification', 'metafeatures')
    confidencelink = key.replace('classification', 'confidence')

    meta['links'][0]['href'] = metalink

    # Assets
    meta['assets']['product']['href'] = classificationlink
    if 'metafeatures' in meta['assets'].keys():
        meta['assets']['metafeatures']['href'] = metafeatureslink
    if 'confidence' in meta['assets'].keys():
        meta['assets']['confidence']['href'] = confidencelink

    with open(outfile, 'w') as f:
        json.dump(meta, f, indent=4)
