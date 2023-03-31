import tempfile
import pytest
import pandas as pd

from satio.features import Features
from worldcereal.processors import (ClassificationProcessor,
                                    CropTypeProcessor,
                                    BlockProcessor)
from worldcereal.fp import (WorldCerealAgERA5FeaturesProcessor,
                            WorldCerealOpticalFeaturesProcessor,
                            WorldCerealSARFeaturesProcessor,
                            WorldCerealThermalFeaturesProcessor)
from worldcereal.features.settings import (get_cropland_catboost_parameters,
                                           get_croptype_catboost_parameters)


FPS = {
    'OPTICAL': WorldCerealOpticalFeaturesProcessor,
    'SAR': WorldCerealSARFeaturesProcessor,
    'METEO': WorldCerealAgERA5FeaturesProcessor,
    'TIR': WorldCerealThermalFeaturesProcessor,
    'DEM': Features.from_dem
}


def get_blockprocessor(WorldCerealOpticalTiled_collection,
                       WorldCerealSigma0Tiled_collection,
                       WorldCerealAgERA5Yearly_collection,
                       dem_collection):

    bounds = (661440, 9648800, 661500, 9648860)
    epsg = 32736
    settings = get_cropland_catboost_parameters()['settings']
    features_meta = get_cropland_catboost_parameters()['features_meta']

    start_date = '2019-08-10'
    end_date = '2019-09-05'
    tile = '36MXB'

    for coll in settings.keys():
        settings[coll]['composite']['start'] = start_date
        settings[coll]['composite']['end'] = end_date

    if 'sen2agri_temp_feat' in features_meta.get('OPTICAL', {}):
        features_meta['OPTICAL'][
            'sen2agri_temp_feat'][
                'parameters']['time_start'] = start_date

    # Filter collections spatially
    collections = dict(
        OPTICAL=WorldCerealOpticalTiled_collection,
        SAR=WorldCerealSigma0Tiled_collection,
        METEO=WorldCerealAgERA5Yearly_collection,
        DEM=dem_collection
    )
    data_collections = {k: v.filter_bounds(bounds, epsg)
                        for k, v in collections.items()}

    # Filter temporal collections
    # Subtract and add 1 day to be certain
    # it's included in the filtering
    collstart = (pd.to_datetime(start_date) -
                 pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    collend = (pd.to_datetime(end_date) +
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

    blockprocessor = BlockProcessor(
        data_collections,
        settings=settings,
        rsi_meta=get_cropland_catboost_parameters()['rsi_meta'],
        features_meta=features_meta,
        ignore_def_feat=get_cropland_catboost_parameters()['ignore_def_feat'],
        bounds=bounds,
        epsg=epsg,
        featresolution=10,
        aez_id=12048,
        custom_fps=FPS)

    return blockprocessor


def test_block_classifier_temporarycrops(
        WorldCerealOpticalTiled_collection,
        WorldCerealSigma0Tiled_collection,
        WorldCerealAgERA5Yearly_collection,
        dem_collection):

    start_date = '2019-08-10'
    end_date = '2019-09-05'

    # Create tmp directory and tmp model file in it
    output_folder = tempfile.TemporaryDirectory().name

    models = {'temporarycrops': 'https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v750/cropland_detector_WorldCerealPixelCatBoost_v750-realms/Realm_5/config.json'}  # NOQA

    # Setup the classification processor
    chain = ClassificationProcessor(
        output_folder,
        models=models,
        season='tc-annual',
        aez=12048,
        collections=dict(
            OPTICAL=WorldCerealOpticalTiled_collection,
            SAR=WorldCerealSigma0Tiled_collection,
            METEO=WorldCerealAgERA5Yearly_collection,
            DEM=dem_collection
        ),
        settings=get_cropland_catboost_parameters()['settings'],
        features_meta=get_cropland_catboost_parameters()['features_meta'],
        rsi_meta=get_cropland_catboost_parameters()['rsi_meta'],
        ignore_def_feat=get_cropland_catboost_parameters()['ignore_def_feat'],
        fps=FPS,
        start_date=start_date.strip('-'),
        end_date=end_date.strip('-'),
        save_features=True,
        save_meta=True,
        save_confidence=True)

    chain.process('36MXB', (661440, 9648800, 661500, 9648860), 32736, 1)


def test_block_classifier_temporarycrops_nosar(
        WorldCerealOpticalTiled_collection,
        WorldCerealAgERA5Yearly_collection,
        dem_collection):
    '''Same as test above but processor should automatically switch
    to optical-only model in absence of a SAR collection
    '''

    start_date = '2019-08-10'
    end_date = '2019-09-05'

    # Create tmp directory and tmp model file in it
    output_folder = tempfile.TemporaryDirectory().name

    models = {'temporarycrops': 'https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v750/cropland_detector_WorldCerealPixelCatBoost_v750-realms/Realm_5/config.json'}  # NOQA

    # Setup the classification processor
    chain = ClassificationProcessor(
        output_folder,
        models=models,
        season='tc-annual',
        aez=12048,
        collections=dict(
            OPTICAL=WorldCerealOpticalTiled_collection,
            METEO=WorldCerealAgERA5Yearly_collection,
            DEM=dem_collection
        ),
        settings=get_cropland_catboost_parameters()['settings'],
        features_meta=get_cropland_catboost_parameters()['features_meta'],
        rsi_meta=get_cropland_catboost_parameters()['rsi_meta'],
        ignore_def_feat=get_cropland_catboost_parameters()['ignore_def_feat'],
        fps=FPS,
        start_date=start_date.strip('-'),
        end_date=end_date.strip('-'),
        save_features=True,
        save_meta=True,
        save_confidence=True)

    chain.process('36MXB', (661440, 9648800, 661500, 9648860), 32736, 1)


@pytest.mark.xfail
def test_block_classifier_tcmaizemain(
        WorldCerealOpticalTiled_collection,
        WorldCerealSigma0Tiled_collection,
        WorldCerealAgERA5Yearly_collection,
        dem_collection):
    '''Expected to fail on github because of no GDAL
    installation.
    '''

    start_date = '2018-10-01'
    end_date = '2018-12-31'

    # Create tmp directory and tmp model file in it
    output_folder = tempfile.TemporaryDirectory().name

    models = {'maize': 'https://artifactory.vgt.vito.be:443/auxdata-public/worldcereal/models/WorldCerealPixelCatBoost/v720/maize_detector_WorldCerealPixelCatBoost_v720/config.json'}  # NOQA

    # Setup the classification processor
    chain = CropTypeProcessor(
        output_folder,
        models=models,
        season='tc-maize-main',
        aez=12048,
        collections=dict(
            OPTICAL=WorldCerealOpticalTiled_collection,
            SAR=WorldCerealSigma0Tiled_collection,
            METEO=WorldCerealAgERA5Yearly_collection,
            DEM=dem_collection
        ),
        settings=get_croptype_catboost_parameters()['settings'],
        features_meta=get_croptype_catboost_parameters()['features_meta'],
        rsi_meta=get_croptype_catboost_parameters()['rsi_meta'],
        ignore_def_feat=get_croptype_catboost_parameters()['ignore_def_feat'],
        fps=FPS,
        start_date=start_date.strip('-'),
        end_date=end_date.strip('-'),
        save_features=True,
        save_meta=True,
        save_confidence=True)

    chain.process('36MXB', (661440, 9648800, 661500, 9648860), 32736, 1)


def test_blockprocessor(WorldCerealOpticalTiled_collection,
                        WorldCerealSigma0Tiled_collection,
                        WorldCerealAgERA5Yearly_collection,
                        dem_collection):

    blockprocessor = get_blockprocessor(WorldCerealOpticalTiled_collection,
                                        WorldCerealSigma0Tiled_collection,
                                        WorldCerealAgERA5Yearly_collection,
                                        dem_collection)
    _ = blockprocessor.get_features()
