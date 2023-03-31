import copy
import json
import os
from pathlib import Path
from typing import List

from loguru import logger
from satio import layers
from satio.collections import AgERA5Collection
from satio.grid import S2TileBlocks
from satio.utils.logs import (exitlogs, proclogs)

from worldcereal.seasons import get_processing_dates, NoSeasonError
from worldcereal.utils.aez import (get_matching_aez_id, group_from_id,
                                   check_supported_aez)
from worldcereal.utils import (get_best_model, needs_s3_loader,
                               get_matching_realm_id,
                               SkipBlockError, setup_logging)
from worldcereal.fp import (WorldCerealAgERA5FeaturesProcessor,
                            WorldCerealOpticalFeaturesProcessor,
                            WorldCerealSARFeaturesProcessor,
                            WorldCerealThermalFeaturesProcessor)
from worldcereal.processors import (ClassificationProcessor,
                                    CropTypeProcessor)
from worldcereal.collections import (WorldCerealOpticalTiledCollection,
                                     WorldCerealSigma0TiledCollection,
                                     WorldCerealThermalTiledCollection,
                                     WorldCerealDEMCollection,
                                     AgERA5YearlyCollection)
from worldcereal.features.settings import (get_irr_parameters,
                                           get_cropland_catboost_parameters,
                                           get_croptype_catboost_parameters,
                                           get_active_crop_parameters)
from worldcereal.geoloader import make_s3_collection
from worldcereal.postprocess import PostProcessor

S2_GRID = layers.load('s2grid')

FEATURESETTINGS_LUT = {
    # 'cropland': get_cropland_tsteps_parameters(),
    'cropland': get_cropland_catboost_parameters(),
    'croptype': get_croptype_catboost_parameters(),
    'irrigation': get_irr_parameters(),
    'active_crop': get_active_crop_parameters()
}

SUPPORTED_BLOCK_SIZES = [512, 1024]
DEFAULT_BLOCK_SIZE = 512


def clip_bounds(bounds, clipsize):
    '''Cut out a part of the block based on the clipsize.
    clipsize = resulting size of the block (in meters)
    '''
    bounds = [bounds[0], bounds[1], bounds[0] + clipsize,
              bounds[1] + clipsize]
    return bounds


def _get_block_size():
    # First try to get block_size from environment var
    try:
        block_size = int(os.environ['EWOC_BLOCKSIZE'])
        logger.info(('Found `EWOC_BLOCKSIZE` environment '
                     f'variable: {block_size}'))
    except KeyError:
        block_size = DEFAULT_BLOCK_SIZE
        logger.warning(('No `block_size` specified. Using '
                        f'default value of: {block_size}'))
    return block_size


def get_processing_blocks(tiles, parameters,
                          debug=False, block_size=None,
                          blocks: List = None):
    '''Construct elementary processing blocks dataframe

    '''

    if block_size is not None:
        logger.info(f'Using provided `block_size` of: {block_size}')
    else:
        block_size = _get_block_size()
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(('Got an unsupported `block_size` value '
                          f'of `{block_size}`. Should be one of '
                          f'{SUPPORTED_BLOCK_SIZES}.'))

    splitter = S2TileBlocks(block_size, s2grid=S2_GRID)
    processingblocks = splitter.blocks(*tiles)

    # If not blocks subset is provided, take by default all
    if type(blocks) == int:
        blocks = [blocks]
    blocks = blocks or list(range(len(processingblocks)))

    # Subset on the desired blocks
    processingblocks = processingblocks.loc[blocks]

    # Add the parameters to each block
    # this is some onconventional Pandas stuff
    processingblocks['parameters'] = processingblocks.apply(
        lambda x: copy.deepcopy(parameters), axis=1)

    if debug:
        # reduce to one block per tile
        processingblocks = processingblocks.iloc[0:1, :]

        # reduce blocks extent for testing
        processingblocks["bounds"] = processingblocks["bounds"].apply(
            lambda x: clip_bounds(x, 2000))

    return processingblocks


def parse_featuresettings(featuresettings):
    if featuresettings not in FEATURESETTINGS_LUT.keys():
        raise ValueError('Unknown featuresettings describer: '
                         f'`{featuresettings}`')

    return FEATURESETTINGS_LUT[featuresettings]


def get_worldcereal_collections(inputpaths, yearly_meteo):
    """Function to create WorldCereal satio-based collections
    from folders

    Args:
        inputpaths (dict): dictionary with keys the different
                           input sources and values the root path
                           to the input data
        yearly_meteo (bool): whether or not to expect yearly composites
                            of meteo as opposed to daily files.
    Returns:
        (dict[collection], dict[fp]): the resulting collections
        and features processors to be used.
    """

    # ----------------------------
    # Create the collections
    # ----------------------------
    logger.info('Initializing collections ...')
    collections = {}

    if 'OPTICAL' in inputpaths.keys():
        collections['OPTICAL'] = WorldCerealOpticalTiledCollection.from_path(
            inputpaths['OPTICAL'])
    if 'SAR' in inputpaths.keys():
        collections['SAR'] = WorldCerealSigma0TiledCollection.from_path(
            inputpaths['SAR'])
    if 'DEM' in inputpaths.keys():
        collections['DEM'] = WorldCerealDEMCollection(
            folder=inputpaths['DEM'])
    if 'METEO' in inputpaths.keys():
        if yearly_meteo:
            collections['METEO'] = AgERA5YearlyCollection.from_path(
                inputpaths['METEO'])
        else:
            collections['METEO'] = AgERA5Collection.from_path(
                inputpaths['METEO'])
    if 'TIR' in inputpaths.keys():
        collections['TIR'] = WorldCerealThermalTiledCollection.from_path(
            inputpaths['TIR'])

    for collID, coll in collections.items():
        if needs_s3_loader(coll):
            logger.info(f'Using s3 bucket collection for `{collID}` ...')
            collections[collID] = make_s3_collection(coll)

    fps = {
        'OPTICAL': WorldCerealOpticalFeaturesProcessor,
        'SAR': WorldCerealSARFeaturesProcessor,
        'METEO': WorldCerealAgERA5FeaturesProcessor,
        'TIR': WorldCerealThermalFeaturesProcessor
    }

    return collections, fps


def get_best_irr_model(parentmodel, aez_id):
    '''Helper function to find locally trained irr models
    '''
    modelname = Path(parentmodel).parent.stem
    model_basedir = Path(parentmodel).parents[2]
    aez_group = group_from_id(aez_id)

    groupmodel = (model_basedir / f'Group_{aez_group}' /
                  str(modelname) / 'config.json')
    zonemodel = (model_basedir / f'Zone_{aez_id}' /
                 str(modelname) / 'config.json')

    if zonemodel.is_file():
        logger.info(f'Found a zone-specific model: {zonemodel}')
        return str(zonemodel)
    elif groupmodel.is_file():
        logger.info(f'Found a group-specific model: {groupmodel}')
        return str(groupmodel)
    else:
        logger.info(f'Only found a parent model: {parentmodel}')
        return str(parentmodel)


def _run_block(output_folder, processing_tuple):
    '''Main WorldCereal products pipeline for 1 processing block

    '''
    oldmask = os.umask(0o002)

    block_id, block = processing_tuple

    # Get the processing parameters
    parameters = block.parameters
    aez_id = parameters['aez']
    realm_id = parameters['realm']
    season = parameters['season']
    models = parameters['models']
    yearly_meteo = parameters['yearly_meteo']
    use_existing_features = parameters.get('use_existing_features', False)
    features_dir = parameters.get('features_dir', None)
    use_local_models = parameters.get('localmodels', True)
    featuresettings = parameters['featuresettings']
    filtersettings = parameters['filtersettings']
    decision_threshold = parameters.get('decision_threshold', 0.5)
    segment = parameters.get('segment', False)
    segment_feat = parameters.get('segment_feat', None)
    save_confidence = parameters.get('save_confidence', True)
    save_meta = parameters.get('save_meta', True)
    save_features = parameters.get('save_features', False)
    active_marker = parameters.get('active_marker', False)
    cropland_mask = parameters.get('cropland_mask', None)
    irrigation = parameters.get('irrigation', False)
    irrparameters = parameters.get('irrparameters', None)
    irrmodels = parameters.get('irrmodels', None)

    logger.info('-'*50)
    logger.info(f'Starting processing block: {block_id}')
    logger.info('-'*50)
    logger.info('PARAMETERS:')
    for parameter, value in parameters.items():
        logger.info(f'{parameter}: {value}')

    # Get collections
    collections, fps = get_worldcereal_collections(parameters['inputs'],
                                                   yearly_meteo)
    irrcollections = copy.deepcopy(collections) if irrigation else None
    collections.pop('TIR', None)  # Default collections should not contain TIR

    # Get best model
    for model in models.keys():
        models[model] = get_best_model(
            models[model], aez_id=aez_id,
            realm_id=realm_id,
            use_local_models=use_local_models
        )

    # Initialize processor
    if season == 'tc-annual':
        processor = ClassificationProcessor
    else:
        processor = CropTypeProcessor

    # Initialize processing chain
    chain = processor(
        output_folder,
        models=models,
        season=season,
        aez=aez_id,
        collections=collections,
        settings=featuresettings['settings'],
        features_meta=featuresettings['features_meta'],
        rsi_meta=featuresettings['rsi_meta'],
        ignore_def_feat=featuresettings['ignore_def_feat'],
        gdd_normalization=featuresettings['gddnormalization'],
        fps=fps,
        start_date=parameters['start_date'].strip('-'),
        end_date=parameters['end_date'].strip('-'),
        active_marker=active_marker,
        cropland_mask=cropland_mask,
        save_features=save_features,
        save_meta=save_meta,
        avg_segm=segment,
        segm_feat=segment_feat,
        use_existing_features=use_existing_features,
        features_dir=features_dir,
        save_confidence=save_confidence,
        decision_threshold=decision_threshold,
        filtersettings=filtersettings,
        irrcollections=irrcollections,
        irrigation=irrigation,
        irrparameters=irrparameters,
        irrmodels=irrmodels)

    # Run pipeline
    chain.process(block.tile, block.bounds,
                  block.epsg, block.block_id)

    os.umask(oldmask)


def run_tile(tile: str,
             configfile: str,
             outputfolder: str,
             blocks: List = None,
             block_size: int = None,
             skip_processed: bool = True,
             debug: bool = False,
             process: bool = True,
             postprocess: bool = True,
             raise_exceptions: bool = False,
             yearly_meteo: bool = True,
             use_existing_features: bool = True,
             user: str = '0000',
             public: bool = True,
             aez_id: int = None,
             sparkcontext=None,
             force_start_date=None,
             force_end_date=None):
    """Generates WorldCereal products.

    Args:
        tile (str): MGRS tile ID to process. Example: '31UFS'
        configfile (str): path to config.json containing processing settings
        outputfolder (str): path to use for saving products and logs
        blocks (List, optional): Block ids of the blocks to process
                from the given tile. Should be a sequence of integers
                between 0 and the total nr of blocks (depending on block size).
                If not provided, all blocks will be processed.
        block_size (int, optional): The size of the processing blocks. If
                not provided, we try to take it from the environment variable
                "EWOC_BLOCKSIZE" or else use the default value 512.
        skip_processed (bool, optional): Skip already processed blocks
                by checking the existlogs folder. Defaults to True.
        debug (bool, optional): Run in debug mode, processing only
                one part of one block. Defaults to False.
        process (bool, optional): If False, skip block processing
        postprocess (bool, optional): If False, skip post-processing to COG
        raise_exceptions (bool, optional): If True, immediately raise any
                unexpected exception instead of silently failing.
        yearly_meteo (bool, optional): If True, use the AgERA5 collection
                suited to work with yearly composites of daily meteo
                data instead of daily files.
        use_existing_features (bool, optional): If True, processor will
                attempt to load the features from a file if they are
                available. Otherwise, normal feature computation will be done.
        user (str, optional): User ID which will be written to STAC metadata
                defaults to "0000"
        public (bool, optional): Intended visibility of the of the created
                products. If True, STAC metadata will set public visibiliy
                flag to True, otherwise False.
        aez_id (int, optional): If provided, the AEZ ID will be enforced
                instead of automatically derived from the Sentinel-2 tile ID.
        sparkcontext (optional): Optional sparkcontext to parallellize
                block processing using spark.
        force_start_date (str, optional): if set, start_date will be overruled
                by this value
        force_end_date (str, optional): if set, end date will be overruled by
                this value

    Returns:
        return_code (int): indicative of processing result
                0: all bocks successfully processed
                1: one or more blocks purposefully skipped
                2: one or more blocks silently failed unexpectedly

    Raises:
        Exception: if one of the blocks unexpectedly failed to process


    """

    # Setup custom logging
    setup_logging()

    # For now, initiate the return_code as sucessful
    return_code = 0

    # Load the config file
    if not Path(configfile).is_file():
        raise FileNotFoundError(
            'Required config file '
            f'`{configfile}` not found. Cannot continue.')
    config = json.load(open(configfile, 'r'))

    # Get processing parameters
    parameters = config['parameters']

    # Get the right feature settings
    parameters['featuresettings'] = parse_featuresettings(
        parameters['featuresettings'])
    if parameters.get("irrigation", False):
        parameters['irrparameters'] = parse_featuresettings(
            'irrigation')

    # Add models and inputs as parameters
    parameters['models'] = config['models']
    parameters['inputs'] = config['inputs']

    # Get year and season to process
    year = parameters['year']
    season = parameters['season']

    # Store which meteo collection method we need to use
    parameters['yearly_meteo'] = yearly_meteo

    # Store whether or not to attempt loading existing features
    parameters['use_existing_features'] = use_existing_features

    # Determine AEZ ID and add as parameter
    if aez_id is None:
        aez_id = get_matching_aez_id(
            S2_GRID[S2_GRID.tile == tile].geometry.values[0])
    else:
        # Check if AEZ ID is valid
        check_supported_aez(aez_id)
        logger.warning(f'Enforcing AEZ ID {aez_id}!')
    parameters['aez'] = aez_id

    # Determine realm ID
    realm_id = get_matching_realm_id(
        S2_GRID[S2_GRID.tile == tile].geometry.values[0])
    parameters['realm'] = realm_id

    # Get processing dates
    try:
        start_date, end_date = get_processing_dates(season, aez_id, year)
        if force_start_date is not None:
            logger.warning(('Overriding `start_date` from '
                            f'`{start_date}` to `{force_start_date}`'))
            start_date = force_start_date
        if force_end_date is not None:
            logger.warning(('Overriding `end_date` from '
                            f'`{end_date}` to `{force_end_date}`'))
            end_date = force_end_date
        parameters['start_date'] = start_date
        parameters['end_date'] = end_date
    except NoSeasonError:
        logger.error(f'No valid `{season}` season found for this tile.')
        if raise_exceptions:
            raise
        return 2  # Unexpected failure code

    if process:

        # ----------------------------
        # Get processing blocks
        # ----------------------------
        blocks = get_processing_blocks([tile], parameters, debug,
                                       block_size=block_size,
                                       blocks=blocks)

        # ----------------------------
        # Create processing tuples
        # ----------------------------
        logger.info('Getting processing tuples ...')
        processing_tuples = [(f'{row.tile}_{row.block_id:03d}', row)
                             for row in blocks.itertuples()]

        # Setup logging folders
        outputfolder = Path(outputfolder)
        exitlogs_folder = outputfolder / 'exitlogs' / f'{year}_{season}'
        proclogs_folder = outputfolder / 'proclogs' / f'{year}_{season}'

        @proclogs(proclogs_folder, level='DEBUG')
        @exitlogs(exitlogs_folder, skip_processed=skip_processed,
                  raise_exceptions=raise_exceptions)
        def _log_run_block(processing_tuple):
            try:
                _run_block(outputfolder, processing_tuple)
                return 0  # Succesfully processed block
            except SkipBlockError as e:
                '''In case a SkipBlockError is raised we have a good
                reason for not being able to process this block.
                Don't fail with an error, but log the reason and stop.
                '''
                logger.warning(e)
                return 1  # Purposefully skipped block

        if sparkcontext is not None:
            logger.info('Starting spark parallellization ...')
            sparkcontext.parallelize(
                processing_tuples,
                len(processing_tuples)).foreach(_log_run_block)

        else:
            logger.info('Running in serial ...')
            # TODO: allow processing in a few threads
            for tup in processing_tuples:
                result = _log_run_block(tup)
                if result is not None:
                    return_code = max(return_code, result)
                else:
                    return_code = 2

    # ----------------------------
    # Do post-processing
    # ----------------------------

    if postprocess:

        oldmask = os.umask(0o002)

        logger.info('Start post-processing to COGs...')
        cog_folder = Path(outputfolder) / 'cogs'

        # Get best model
        products = list(config['models'].keys())
        for model in parameters['models'].keys():

            parameters['models'][model] = get_best_model(
                parameters['models'][model], aez_id=aez_id,
                realm_id=realm_id,
                use_local_models=parameters.get(
                    'localmodels', True)
            )

        if (parameters.get('irrigation', False)) and (
                parameters.get('irrmodels', None) is not None):
            # for model in parameters['irrmodels'].keys():
            # if parameters['localmodels']:
            #     parameters['irrmodels'][model] = get_best_model(
            #         parameters['irrmodels'][model], aez_id)
            products.extend(list(parameters['irrmodels'].keys()))

        # Initialize postprocessor
        postprocessor = PostProcessor(
            Path(outputfolder) / 'blocks',
            cog_folder, tile, year, season, aez_id,
            products=products,
            parameters=parameters)

        # Postprocess products
        postprocessor.run(generate_metadata=True,
                          user=user, public=public,
                          skip_processed=skip_processed,
                          in_memory=False, debug=debug)

        os.umask(oldmask)

    logger.success('Finished!')

    return return_code
