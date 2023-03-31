try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
import geopandas as gpd
from loguru import logger

from worldcereal import resources


def load():

    with pkg_resources.open_binary(resources, 'AEZ.geojson') as aez_file:
        # logger.info(f'Loading AEZ ...')
        aez = gpd.read_file(aez_file, driver='GeoJSON')

    return aez


def ids_from_group(aez_group):

    aez = load()
    aez = aez[aez['groupID'] == aez_group]
    zone_ids = list(aez['zoneID'].values)

    try:
        zone_ids = [int(x) for x in zone_ids]
    except Exception:
        pass

    if len(zone_ids) == 0:
        raise RuntimeError(('No matching AEZ IDs for '
                            f'Group `{aez_group}`'))

    logger.info((f'Found {len(zone_ids)} matching '
                 f'AEZ IDs for Group {aez_group}.'))

    return zone_ids


def group_from_id(aez_id):

    aez = load()
    aez = aez[aez['zoneID'] == aez_id]
    aez_group = list(aez['groupID'].values)

    try:
        aez_group = [int(x) for x in aez_group]
    except Exception:
        pass

    if len(aez_group) == 0:
        raise RuntimeError(('No matching AEZ group for '
                            f'zoneID `{aez_id}`'))

    elif len(aez_group) > 1:
        raise RuntimeError(('Multiple matching AEZ groups for '
                            f'zoneID `{aez_id}`'))

    logger.info((f'Matched AEZ Group {aez_group[0]} '
                 f'to AEZ id {aez_id}.'))

    return int(aez_group[0])


def get_matching_aez_id(geometry):
    '''Function to get corresponding AEZ ID for a geometry
    '''

    aez = load().to_crs(epsg=4326)

    # Cast the Zone and Group IDs to INT
    aez['zoneID'] = aez['zoneID'].astype(int)
    aez['groupID'] = aez['groupID'].astype(int)

    matching_aez = aez[aez.intersects(geometry)].copy()
    matching_aez['geometry'] = matching_aez.buffer(0)

    if matching_aez.shape[0] == 0:
        raise ValueError('Could not find any intersecting AEZ.')
    elif matching_aez.shape[0] > 1:
        # logger.warning(('Found more than one intersecting AEZ. '
        #                 'Taking largest intersection.'))
        matching_aez = matching_aez.set_geometry(
            matching_aez.intersection(geometry))
        matching_aez['area'] = matching_aez.to_crs(epsg=3857).area
        matching_aez = matching_aez.sort_values('area', ascending=False)
        matching_aez = matching_aez.iloc[0, :]
    else:
        matching_aez = matching_aez.iloc[0, :]

    # logger.debug((f'Matched AEZ: zoneID = {matching_aez.zoneID} | '
    #              f'groupID = {matching_aez.groupID}'))

    return int(matching_aez.zoneID)


def check_supported_aez(aez_id):
    '''Method to check whether an aez ID
    is supported or not. Will raise an error if not.
    '''
    aez = load()

    if aez_id not in aez['zoneID'].astype(int).tolist():
        raise ValueError(f'AEZ ID `{aez_id}` not supported!')


if __name__ == '__main__':
    print(load())
