from pathlib import Path
import sys
import os
import numpy as np
from loguru import logger
import geopandas as gpd
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
from satio.collections import BaseCollection

from worldcereal.utils.aez import group_from_id
from worldcereal import resources


AUXDATA_PATH = os.environ.get('EWOC_AUXDATA', None)
COLL_MAXGAP = int(os.environ.get('EWOC_COLL_MAXGAP', 60))


LC_VALUES = {
    'Unknown': 0,
    'Cropland': 10,
    'Annual cropland': 11,
    'Perennial cropland': 12,
    'Grassland': 13,
    'Herbaceous vegetation': 20,
    'Shrubland': 30,
    'Deciduous forest': 40,
    'Evergreen forest': 41,
    'Mixed/unknown forest': 42,
    'Bare/Sparse vegetation': 50,
    'Builtup/urban': 60,
    'Water': 70,
    'Snow/ice': 80,
    'Non-cropland': 99
}

LC_COLORS = {
    'Unknown': [255/255., 255/255., 255/255.],
    'Cropland': [244/255., 233/255., 12/255.],
    'Annual cropland': [244/255., 178/255., 12/255.],
    'Perennial cropland': [137/255., 100/255., 5/255.],
    'Grassland': [134/255., 229/255., 25/255.],
    'Herbaceous vegetation': [56/255., 237/255., 153/255.],
    'Shrubland': [149/255., 249/255., 234/255.],
    'Deciduous forest': [21/255., 155/255., 17/255.],
    'Evergreen forest': [12/255., 119/255., 8/255.],
    'Mixed/unknown forest': [6/255., 73/255., 4/255.],
    'Bare/Sparse vegetation': [73/255., 56/255., 4/255.],
    'Builtup/urban': [201/255., 201/255., 201/255.],
    'Water': [14/255., 194/255., 234/255.],
    'Snow/ice': [187/255., 238/255., 249/255.],
    'Non-cropland': [163/255., 78/255., 109/255.]
}

LC_MAPPER = {
    0: 0,
    10: 1,
    11: 2,
    12: 3,
    13: 4,
    20: 5,
    30: 6,
    40: 7,
    41: 8,
    42: 9,
    50: 10,
    60: 11,
    70: 12,
    80: 13,
    99: 14
}


BIOME_RASTERS = {f"biome{i:02d}": f"biome_{i:02d}.tif"
                 for i in range(1, 14)}
REALM_RASTERS = {f"realm{i:02d}": f"realm_{i:02d}.tif"
                 for i in range(9)}


def get_coll_maxgap(collection_id):
    '''Get the collection-specific maximum allowed gap. If no
    collection-specific environment variable has been set for this gap
    this function returns the default value.
    '''
    return int(os.environ.get(f'EWOC_COLL_MAXGAP_{collection_id}',
                              COLL_MAXGAP))


def is_real_feature(ft_name):
    '''
    Helper function to distinguish real input features
    from meta-features.
    '''
    if ((len(ft_name.split('-')) == 4) or ('DEM' in ft_name)
            or ('biome' in ft_name) or ('realm' in ft_name)):
        return True
    else:
        return False


def get_sensor_config(feature_names, pattern=''):

    sensorconfig = {}
    sensorconfig['aux'] = dict(
        channels=len([x for x in feature_names if ('DEM' in x or 'biome' in x)
                      and pattern in x and '-ts' not in x]),
        positions=[i for i, ft in enumerate(feature_names)
                   if ('DEM' in ft or 'biome' in ft)
                   and pattern in ft and '-ts' not in ft]
    )

    return sensorconfig


def get_sensor_config_timeseries(feature_names):
    all_sensors = [x.split('-')[0] for x in feature_names if '-ts' in x]
    unique_sensors = list(set(all_sensors))

    sensorconfig = {}
    for sensor in unique_sensors:
        allsensorfts = [x for x in feature_names if sensor in x if '-ts' in x]
        sensorchannels = list(set([x.split('-')[1] for x in allsensorfts]))
        sensorconfig[sensor] = {'channels': sensorchannels}

        for channel in sensorchannels:
            channelts = [ft for ft in allsensorfts if channel in ft]
            tsteps = sorted([int(ft.split('-')[2][2:]) for ft in channelts])
            positions = []
            for tstep in tsteps:
                positions += [i for i, ft in enumerate(feature_names)
                              if sensor in ft and channel
                              in ft and f'-ts{tstep}-' in ft]

            sensorconfig[sensor][channel] = dict(
                positions=positions
            )

    return sensorconfig


def check_url(url):
    """Method to check if a URL is valid

    Args:
        url (str): URL to check

    Returns:
        bool: True if URL is valid, False if not
    """

    import urllib

    try:
        status_code = urllib.request.urlopen(url).getcode()
        return status_code == 200
    except urllib.error.HTTPError:
        # logger.debug(f'URLError: {e.reason}')
        return False


def _get_best_model_fromurl(parentmodel, aez_id):
    '''Helper function to find locally trained models.
    This is the version that looks on Artifactory!
    '''

    model_baselink = '/'.join(parentmodel.split('/')[:-1])
    aez_group = group_from_id(aez_id)

    groupmodel = f'{model_baselink}/Group_{aez_group}/config.json'
    zonemodel = f'{model_baselink}/Zone_{aez_id}/config.json'

    if check_url(zonemodel):
        logger.info(f'Found a zone-specific model: {zonemodel}')
        return str(zonemodel)
    elif check_url(groupmodel):
        logger.info(f'Found a group-specific model: {groupmodel}')
        return str(groupmodel)
    else:
        logger.info(f'Only found a parent model: {parentmodel}')
        return str(parentmodel)
    # return str(parentmodel)


def _get_best_model_fromfile(parentmodel, aez_id):
    '''Helper function to find locally trained models
    '''

    model_basedir = Path(parentmodel).parent
    aez_group = group_from_id(aez_id)

    groupmodel = model_basedir / f'Group_{aez_group}' / 'config.json'
    zonemodel = model_basedir / f'Zone_{aez_id}' / 'config.json'

    if zonemodel.is_file():
        logger.info(f'Found a zone-specific model: {zonemodel}')
        return str(zonemodel)
    elif groupmodel.is_file():
        logger.info(f'Found a group-specific model: {groupmodel}')
        return str(groupmodel)
    else:
        logger.info(f'Only found a parent model: {parentmodel}')
        return str(parentmodel)
    # return str(parentmodel)


def _get_realm_model_fromfile(basemodelpath, realm_id):
    '''Helper function to find locally trained models
    '''

    realm_model = Path(basemodelpath) / f'Realm_{realm_id}' / 'config.json'

    if not realm_model.is_file():
        raise FileNotFoundError(
            f'Could not find required realm_model: `{realm_model}`')
    else:
        logger.info(f'Found realm model: {realm_model}')
        return str(realm_model)


def _get_realm_model_fromurl(basemodelpath, realm_id):
    '''Helper function to find locally trained models
    '''

    realm_model = '/'.join(basemodelpath.split('/')) + f'/Realm_{realm_id}/config.json'

    if check_url(realm_model):
        logger.info(f'Found realm model: {realm_model}')
        return str(realm_model)
    else:
        raise FileNotFoundError(
            f'Could not find required realm_model: `{realm_model}`')


def get_best_model(basemodelpath, aez_id=None, realm_id=None,
                   use_local_models=False):

    if 'realm' in basemodelpath.lower():
        # Realm model track
        realm_id = int(realm_id)

        # Ensure compatibility: strip config.json from path
        basemodelpath = str(basemodelpath).replace('config.json', '')

        if basemodelpath.startswith('http'):
            return _get_realm_model_fromurl(basemodelpath, realm_id)
        else:
            return _get_realm_model_fromfile(basemodelpath, realm_id)

    else:
        # AEZ model track
        aez_id = int(aez_id)
        if not use_local_models:
            return basemodelpath
        else:
            if basemodelpath.startswith('http'):
                return _get_best_model_fromurl(basemodelpath, aez_id)
            else:
                return _get_best_model_fromfile(basemodelpath, aez_id)


def get_matching_realm_id(geometry):
    '''Function to get corresponding REALM ID for a geometry
    '''

    with pkg_resources.open_binary(resources, 'realms.gpkg') as realms_file:
        # logger.info(f'Loading AEZ ...')
        realm_df = gpd.read_file(realms_file, driver='GPKG')

    # Cast the Realm ID to INT
    realm_df['REALM'] = realm_df['REALM'].astype(int)

    matching_realm = realm_df[realm_df.intersects(geometry)].copy()
    matching_realm['geometry'] = matching_realm.buffer(0)

    if matching_realm.shape[0] == 0:
        raise ValueError('Could not find any intersecting REALM.')
    elif matching_realm.shape[0] > 1:
        logger.warning(('Found more than one intersecting REALM. '
                        'Taking largest intersection.'))
        matching_realm = matching_realm.set_geometry(
            matching_realm.intersection(geometry))
        matching_realm['area'] = matching_realm.to_crs(epsg=3857).area
        matching_realm = matching_realm.sort_values('area', ascending=False)
        matching_realm = matching_realm.iloc[0, :]
    else:
        matching_realm = matching_realm.iloc[0, :]

    logger.info((f'Matched REALM: {matching_realm.REALM}'))

    return int(matching_realm.REALM)


def needs_s3_loader(coll: BaseCollection):
    """Function to check if the collection's loader
    needs to be updated by a direct s3 geoloader.

    Args:
        coll (BaseCollection): the Satio collection to check

    Returns:
        Bool: True if loader needs to be updated
    """
    if hasattr(coll, 'df'):
        if (coll.df.path.iloc[0].startswith('s3') or
                coll.df.path.iloc[0].startswith('/vsis3')):
            return True
        else:
            return False
    elif hasattr(coll, 'folder'):
        if (str(coll.folder).startswith('s3') or
                str(coll.folder).startswith('/vsis3')):
            return True
        else:
            return False
    else:
        raise RuntimeError((f'Collection `{coll}` does not have a `df` '
                            'or `folder` attribute.'))


def setup_logging():

    def _no_error(record):
        return record["level"].name != "ERROR"

    # Remove all current loggers
    logger.remove()

    # Add a stderr logger for ERROR messages
    logger.add(sys.stderr, level='ERROR')

    # Add a stdout logger for other messages
    logger.add(sys.stdout, filter=_no_error)


def probability_to_binary(probabilities, threshold):
    """Method to transform 2-class probabilities into a binary
    prediction array based on provided threshold

    Args:
        probabilities (np.ndarray): 2D array (samples X probability)
        threshold (float): decision threshold for binarization

    Returns:
        1D array (uint8): resulting binary class array
    """
    if not 0 <= threshold <= 1:
        raise ValueError('`threshold` should be in [0, 1] range.')

    prob_1 = probabilities[:, 1]  # Probability of class 1
    binary = (prob_1 >= threshold).astype(np.uint8)

    return binary


def probability_to_confidence(probabilities):
    """Method to transform 2-class probabilities into confidence
       array. Confidence is 0 when both probabilities are 50%.
       Confidence is 1 when probabilities are 0% and 100%
    Args:
        probabilities (np.ndarray): 2D array (samples X probability)

    Returns:
        1D array (float): resulting confidence array
    """

    confidence = np.absolute((probabilities[:, 0] - 0.5) / 0.5)

    return confidence


class SkipBlockError(Exception):
    pass


class BlockTooColdError(Exception):
    pass
