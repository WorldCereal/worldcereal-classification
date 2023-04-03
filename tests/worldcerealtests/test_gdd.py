import numpy as np
from worldcereal.utils.masking import SCL_MASK_VALUES
from worldcereal.fp import (L2AFeaturesProcessor,
                            SARFeaturesProcessor,
                            AgERA5FeaturesProcessor)
from worldcereal.gdd import GDDcomputer


def mean(x):
    return np.mean(x, axis=0, keepdims=True)


def test_compute_gdd(AgERA5Training_collection):

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    gdd = GDDcomputer(AgERA5Training_collection,
                      tbase=10,
                      bounds=bounds,
                      epsg=epsg,
                      start_date='2019-01-01',
                      end_date='2019-08-31')

    # Accumulate GDD without season specification
    accumulated_gdd = gdd.compute_accumulated_gdd()

    assert np.sum(accumulated_gdd.data < 0) == 0

    print(accumulated_gdd)


def test_compute_gdd_season(AgERA5Training_collection):

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    gdd = GDDcomputer(AgERA5Training_collection,
                      tbase=10,
                      bounds=bounds,
                      epsg=epsg,
                      start_date='2018-10-01',
                      end_date='2019-08-31')

    # Accumulate GDD with season specification (winter wheat)
    accumulated_gdd = gdd.compute_accumulated_gdd(season='tc-maize-main')

    assert np.sum(accumulated_gdd.data < 0) == 0

    print(accumulated_gdd)


def test_get_sos_date_outside_aez(AgERA5Training_collection):
    '''This version has a block outside the AEZ
    '''

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631
    aez_id = 22194

    gdd = GDDcomputer(AgERA5Training_collection,
                      tbase=10,
                      bounds=bounds,
                      epsg=epsg,
                      start_date='2018-10-01',
                      end_date='2019-08-31',
                      aez_id=aez_id)

    sos_date = gdd.get_sos_date('tc-maize-main', '2018-10-01')

    assert sos_date == '2019-04-26'


def test_get_sos_date_outside_aezstats(AgERA5Training_collection):
    '''This version has a block inside the AEZ but with a pixel-based
    crop calendar which is BEFORE the AEZ SOSmin
    '''

    bounds = (399960, 7890220, 399980, 7890240)
    epsg = 32720
    aez_id = 20087

    gdd = GDDcomputer(AgERA5Training_collection,
                      tbase=10,
                      bounds=bounds,
                      epsg=epsg,
                      start_date='2018-12-01',
                      end_date='2019-08-31',
                      aez_id=aez_id)

    sos_date = gdd.get_sos_date('tc-maize-main', '2018-10-01')

    # AEZ SOSmin = 2018-12-28
    # pixel-based SOS = 2018-11-12
    # in this case AEZ SOSmin needs to override pixel-based SOSmin
    assert sos_date == '2018-12-28'


def test_L2A_gddnormalization(L2ATraining_collection,
                              AgERA5Training_collection):

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    S2_settings = dict(
        bands=[],
        rsis=["ndvi"],
        composite=dict(
            freq=10,
            window=20,
            mode='median',
            start='2019-01-01',
            end='2019-07-31'),
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            max_invalid_ratio=1),
        normalize_gdd=dict(
            tbase=0,
            tlimit=25,
            gdd_bins=100,
            season='tc-wintercereals'
        )
    )

    gddcomp = GDDcomputer(AgERA5Training_collection,
                          tbase=S2_settings['normalize_gdd']['tbase'],
                          upper_limit=S2_settings['normalize_gdd']['tlimit'],
                          bounds=bounds,
                          epsg=epsg,
                          start_date=S2_settings['composite']['start'],
                          end_date=S2_settings['composite']['end'])
    accumulated_gdd = gddcomp.compute_accumulated_gdd(
        season='tc-wintercereals')
    S2_settings['normalize_gdd']['accumulated_gdd'] = accumulated_gdd

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings)

    # Load original data at 10m
    ts_10m = s2_fp.load_data(10)

    # Invoke preprocessing chain which will do GDD normalization
    ts_10m_gddnormalized = s2_fp.preprocess_data(ts_10m, 10)

    # We're supposed to have 22 normalized GDD timestamps in this setting
    assert len(ts_10m_gddnormalized.timestamps) == 22


def test_sigma0_gddnormalization(Sigma0Training_collection,
                                 AgERA5Training_collection):

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    S1_settings = dict(
        bands=["VV"],
        composite=dict(
            freq=1,
            mode='mean',
            start='2019-02-01',
            end='2019-11-30'
        ),
        normalize_gdd=dict(
            tbase=0,
            tlimit=25,
            gdd_bins=100
        )
    )

    gddcomp = GDDcomputer(AgERA5Training_collection,
                          tbase=S1_settings['normalize_gdd']['tbase'],
                          upper_limit=S1_settings['normalize_gdd']['tlimit'],
                          bounds=bounds,
                          epsg=epsg,
                          start_date=S1_settings['composite']['start'],
                          end_date=S1_settings['composite']['end'])
    accumulated_gdd = gddcomp.compute_accumulated_gdd(season='tc-maize-main')
    S1_settings['normalize_gdd']['accumulated_gdd'] = accumulated_gdd

    s1_fp = SARFeaturesProcessor(Sigma0Training_collection,
                                 settings=S1_settings)

    # Load original data at 20m
    ts_20m = s1_fp.load_data(20)

    # Invoke preprocessing chain which will do GDD normalization
    ts_20m_gddnormalized = s1_fp.preprocess_data(ts_20m, 20)

    # We're supposed to have 33 normalized GDD timestamps in this setting
    assert len(ts_20m_gddnormalized.timestamps) == 33


def test_agera5_gddnormalization(AgERA5Training_collection):

    AgERA5_mean_settings = dict(
        bands=[
            'temperature_mean',
        ],
        rsis=[],
        composite=dict(
            mode='mean',
            freq=10,
            window=None,
            start='2019-02-01',
            end='2019-07-31'
        ),
        normalize_gdd=dict(
            tbase=10,
            tlimit=30,
            gdd_bins=100
        )
    )

    gddcomp = GDDcomputer(
        AgERA5Training_collection,
        tbase=AgERA5_mean_settings['normalize_gdd']['tbase'],
        upper_limit=AgERA5_mean_settings['normalize_gdd']['tlimit'],
        start_date=AgERA5_mean_settings['composite']['start'],
        end_date=AgERA5_mean_settings['composite']['end'])
    accumulated_gdd = gddcomp.compute_accumulated_gdd()
    AgERA5_mean_settings['normalize_gdd']['accumulated_gdd'] = accumulated_gdd

    AgERA5_mean_features_meta = dict(
        mean=dict(
            function=mean,
            names=['mean']
        )
    )

    AgERA5_fp = AgERA5FeaturesProcessor(
        AgERA5Training_collection,
        AgERA5_mean_settings,
        features_meta=AgERA5_mean_features_meta,
        ignore_def_features=True)

    # Load original data at 100m
    ts_100m = AgERA5_fp.load_data(100)

    # Invoke preprocessing chain which will do GDD normalization
    ts_100m_gddnormalized = AgERA5_fp.preprocess_data(ts_100m, 100)

    # We're supposed to have 6 normalized GDD timestamps in this setting
    assert len(ts_100m_gddnormalized.timestamps) == 6
