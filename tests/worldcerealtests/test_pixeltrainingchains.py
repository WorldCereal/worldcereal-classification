import tempfile
from loguru import logger

from worldcereal.processors import PixelTrainingChain
from worldcereal.classification.models import WorldCerealRFModel
from worldcereal.utils import aez
from worldcereal.features.settings import (get_default_settings,
                                           get_default_rsi_meta,
                                           get_default_ignore_def_feat,
                                           get_cropland_tsteps_parameters,
                                           get_croptype_catboost_parameters,
                                           get_croptype_catboost_features_meta)


def test_pixeltrainingchain_local_custom(L2ATraining_collection,
                                         Sigma0Training_collection,
                                         AgERA5Training_collection,
                                         dem_collection,
                                         worldcover_collection,
                                         PatchLabelsTraining_collection,
                                         worldcereal_training_df,
                                         selected_features):
    '''
    A test for a complete pixel-based training chain
    to detect cropland vs non-cropland on a custom
    start and end date without using spark
    '''

    season = 'custom'
    start_date = '2019-01-01'
    end_date = '2019-10-31'

    #  We can only use the two samples that contain
    #  this custom season
    location_ids = ['0000280849BAC91C', '000028085BF08E7E']

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name

    # Initialize a model
    rf_parameters = {'n_estimators': 100}
    rfmodel = WorldCerealRFModel(feature_names=selected_features,
                                 parameters=rf_parameters,
                                 basedir=basedir)

    # Get processing settings
    settings = get_croptype_catboost_parameters()['settings']
    rsi_meta = get_croptype_catboost_parameters()['rsi_meta']
    features_meta = get_croptype_catboost_parameters()['features_meta']
    ignore_def_feat = get_croptype_catboost_parameters()['ignore_def_feat']

    trainingchain = PixelTrainingChain(
        model=rfmodel,
        basedir=tempfile.mkdtemp(),
        collections_dict={'OPTICAL': L2ATraining_collection,
                          'SAR': Sigma0Training_collection,
                          'METEO': AgERA5Training_collection,
                          'LABELS': PatchLabelsTraining_collection,
                          'DEM': dem_collection,
                          'WorldCover': worldcover_collection},
        trainingdb=worldcereal_training_df,
        settings=settings,
        rsi_meta=rsi_meta,
        features_meta=features_meta,
        ignore_def_feat=ignore_def_feat,
        location_ids=location_ids,
        season=season,
        start_date=start_date,
        end_date=end_date
    )

    trainingdf = trainingchain.get_training_df(sparkcontext=None,
                                               label='LC',
                                               save=False,
                                               max_pixels=3,
                                               debug=True
                                               )

    # Train the model
    trainingchain.train_one_vs_all(trainingdf, targetlabels=11)


def test_pixelchain_local_winterwheat_gdd(L2ATraining_collection,
                                          Sigma0Training_collection,
                                          AgERA5Training_collection,
                                          dem_collection,
                                          PatchLabelsTraining_collection,
                                          worldcereal_training_df,
                                          selected_features):
    '''
    A test for a complete pixel-based training chain
    to detect winter wheat without using spark, and using
    gdd normalization
    '''

    season = 'winter'
    targetlabel = 1110

    # Get processing settings
    gdd_settings = get_croptype_catboost_parameters()['settings']
    rsi_meta = get_croptype_catboost_parameters()['rsi_meta']
    features_meta = get_croptype_catboost_parameters()['features_meta']
    ignore_def_feat = get_croptype_catboost_parameters()['ignore_def_feat']

    # Update default settings to include GDD normalization
    for s in gdd_settings.keys():
        gdd_settings[s]['normalize_gdd'] = dict(tbase=0,
                                                tlimit=25,
                                                season=season)

    features_meta = get_croptype_catboost_features_meta()

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name

    # Initialize a model
    rf_parameters = {'n_estimators': 10}
    rfmodel = WorldCerealRFModel(feature_names=selected_features,
                                 parameters=rf_parameters,
                                 basedir=basedir)

    trainingchain = PixelTrainingChain(
        model=rfmodel,
        basedir=tempfile.mkdtemp(),
        collections_dict={'OPTICAL': L2ATraining_collection,
                          'SAR': Sigma0Training_collection,
                          'METEO': AgERA5Training_collection,
                          'DEM': dem_collection,
                          'LABELS': PatchLabelsTraining_collection},
        trainingdb=worldcereal_training_df,
        location_ids=['0000280849BAC91C', '000028085BF08E7E',
                      '00002806639B424A'],
        settings=gdd_settings,
        features_meta=features_meta,
        rsi_meta=rsi_meta,
        ignore_def_feat=ignore_def_feat,
        season=season,
    )

    trainingdf = trainingchain.get_training_df(sparkcontext=None,
                                               label='CT',
                                               save=False,
                                               max_pixels=3,
                                               debug=True
                                               )

    # Train the model
    trainingchain.train_one_vs_all(trainingdf, targetlabels=targetlabel)


def test_pixeltrainingchain_local_winterwheat(L2ATraining_collection,
                                              Sigma0Training_collection,
                                              AgERA5Training_collection,
                                              dem_collection,
                                              PatchLabelsTraining_collection,
                                              worldcereal_training_df,
                                              selected_features):

    season = 'winter'
    targetlabel = 1110

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name

    # Initialize a model
    rf_parameters = {'n_estimators': 10}
    rfmodel = WorldCerealRFModel(feature_names=selected_features,
                                 parameters=rf_parameters,
                                 basedir=basedir)

    # Get processing settings
    settings = get_croptype_catboost_parameters()['settings']
    rsi_meta = get_croptype_catboost_parameters()['rsi_meta']
    features_meta = get_croptype_catboost_parameters()['features_meta']
    ignore_def_feat = get_croptype_catboost_parameters()['ignore_def_feat']

    trainingchain = PixelTrainingChain(
        model=rfmodel,
        basedir=tempfile.mkdtemp(),
        collections_dict={'OPTICAL': L2ATraining_collection,
                          'SAR': Sigma0Training_collection,
                          'METEO': AgERA5Training_collection,
                          'DEM': dem_collection,
                          'LABELS': PatchLabelsTraining_collection},
        trainingdb=worldcereal_training_df,
        settings=settings,
        rsi_meta=rsi_meta,
        features_meta=features_meta,
        ignore_def_feat=ignore_def_feat,
        location_ids=['0000280849BAC91C', '000028085BF08E7E',
                      '00002806639B424A'],
        season=season,
    )

    trainingdf = trainingchain.get_training_df(sparkcontext=None,
                                               label='CT',
                                               save=False,
                                               max_pixels=3,
                                               debug=True
                                               )

    # Train the model
    trainingchain.train_one_vs_all(trainingdf, targetlabels=targetlabel)


def test_pixeltrainingchain_local_summer1(L2ATraining_collection,
                                          Sigma0Training_collection,
                                          AgERA5Training_collection,
                                          dem_collection,
                                          PatchLabelsTraining_collection,
                                          worldcereal_training_df,
                                          selected_features):

    season = 'summer1'
    targetlabel = 1200

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name

    # Initialize a model
    rf_parameters = {'n_estimators': 10}
    rfmodel = WorldCerealRFModel(feature_names=selected_features,
                                 parameters=rf_parameters,
                                 basedir=basedir)

    # Get processing settings
    settings = get_croptype_catboost_parameters()['settings']
    rsi_meta = get_croptype_catboost_parameters()['rsi_meta']
    features_meta = get_croptype_catboost_parameters()['features_meta']
    ignore_def_feat = get_croptype_catboost_parameters()['ignore_def_feat']

    trainingchain = PixelTrainingChain(
        model=rfmodel,
        basedir=tempfile.mkdtemp(),
        collections_dict={'OPTICAL': L2ATraining_collection,
                          'SAR': Sigma0Training_collection,
                          'METEO': AgERA5Training_collection,
                          'DEM': dem_collection,
                          'LABELS': PatchLabelsTraining_collection},
        trainingdb=worldcereal_training_df,
        settings=settings,
        rsi_meta=rsi_meta,
        features_meta=features_meta,
        ignore_def_feat=ignore_def_feat,
        location_ids=['0000280849BAC91C', '000028085BF08E7E',
                      '00002806639B424A'],
        season=season,
    )

    trainingdf = trainingchain.get_training_df(sparkcontext=None,
                                               label='CT',
                                               save=False,
                                               max_pixels=3,
                                               debug=True
                                               )

    # Train the model
    trainingchain.train_one_vs_all(trainingdf, targetlabels=targetlabel)


def test_pixeltrainingchain_local_summer2(L2ATraining_collection,
                                          Sigma0Training_collection,
                                          AgERA5Training_collection,
                                          dem_collection,
                                          PatchLabelsTraining_collection,
                                          worldcereal_training_df):

    season = 'summer2'

    trainingchain = PixelTrainingChain(
        basedir=tempfile.mkdtemp(),
        collections_dict={'OPTICAL': L2ATraining_collection,
                          'SAR': Sigma0Training_collection,
                          'METEO': AgERA5Training_collection,
                          'DEM': dem_collection,
                          'LABELS': PatchLabelsTraining_collection},
        trainingdb=worldcereal_training_df,
        settings=get_default_settings(),
        rsi_meta=get_default_rsi_meta(),
        ignore_def_feat=get_default_ignore_def_feat(),
        location_ids=['2021_TZA_COPERNICUS-GEOGLAM_POLY_1101950'],
        season=season,
    )

    _ = trainingchain.get_training_df(sparkcontext=None,
                                      label='CT',
                                      save=False,
                                      max_pixels=3,
                                      debug=True)


def test_pixeltrainingchain_local_cropland(L2ATraining_collection,
                                           Sigma0Training_collection,
                                           AgERA5Training_collection,
                                           dem_collection,
                                           PatchLabelsTraining_collection,
                                           worldcereal_training_df,
                                           selected_features):

    season = 'annual'
    targetlabel = 11

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name

    # Initialize a model
    rf_parameters = {'n_estimators': 10}
    rfmodel = WorldCerealRFModel(feature_names=selected_features,
                                 parameters=rf_parameters,
                                 basedir=basedir)

    # Get processing settings
    settings = get_croptype_catboost_parameters()['settings']
    rsi_meta = get_croptype_catboost_parameters()['rsi_meta']
    features_meta = get_croptype_catboost_parameters()['features_meta']
    ignore_def_feat = get_croptype_catboost_parameters()['ignore_def_feat']

    trainingchain = PixelTrainingChain(
        model=rfmodel,
        basedir=tempfile.mkdtemp(),
        collections_dict={'OPTICAL': L2ATraining_collection,
                          'SAR': Sigma0Training_collection,
                          'METEO': AgERA5Training_collection,
                          'DEM': dem_collection,
                          'LABELS': PatchLabelsTraining_collection},
        trainingdb=worldcereal_training_df,
        settings=settings,
        features_meta=features_meta,
        rsi_meta=rsi_meta,
        ignore_def_feat=ignore_def_feat,
        location_ids=worldcereal_training_df['location_id'].tolist(),
        season=season,
    )

    trainingdf = trainingchain.get_training_df(sparkcontext=None,
                                               label='LC',
                                               save=True,
                                               max_pixels=3,
                                               debug=True
                                               )

    # Train the model
    trainingchain.train_one_vs_all(trainingdf, targetlabels=targetlabel)


def test_pixeltrainingchain_local_cropland_tanzania(
        L2ATraining_collection,
        Sigma0Training_collection,
        AgERA5Training_collection,
        dem_collection,
        PatchLabelsTraining_collection,
        worldcereal_training_df):

    season = 'annual'

    # Create tmp directory and tmp model file in it
    basedir = tempfile.TemporaryDirectory().name

    # Get processing settings
    settings = get_cropland_tsteps_parameters()['settings']
    rsi_meta = get_cropland_tsteps_parameters()['rsi_meta']
    features_meta = get_cropland_tsteps_parameters()['features_meta']
    ignore_def_feat = get_cropland_tsteps_parameters()['ignore_def_feat']

    # Setup training chain
    logger.info('Setting up trainingchain ...')
    trainingchain = PixelTrainingChain(
        basedir=basedir,
        collections_dict={'OPTICAL': L2ATraining_collection,
                          'SAR': Sigma0Training_collection,
                          'LABELS': PatchLabelsTraining_collection,
                          'DEM': dem_collection,
                          'METEO': AgERA5Training_collection},
        trainingdb=worldcereal_training_df,
        settings=settings,
        rsi_meta=rsi_meta,
        features_meta=features_meta,
        ignore_def_feat=ignore_def_feat,
        location_ids=['2021_TZA_COPERNICUS-GEOGLAM_POLY_1101950'],
        season=season,
        aez=aez.load())

    trainingdf = trainingchain.get_training_df(sparkcontext=None,
                                               label='LC',
                                               save=True,
                                               max_pixels=3,
                                               debug=True,
                                               format='parquet'
                                               )

    assert trainingdf is not None
