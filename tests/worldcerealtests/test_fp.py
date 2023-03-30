import numpy as np
import pytest

from worldcereal.fp import (SARFeaturesProcessor,
                            TSSIGMA0FeaturesProcessor,
                            L2AFeaturesProcessor,
                            AgERA5FeaturesProcessor,
                            TSSIGMA0TiledFeaturesProcessor,
                            WorldCerealSARFeaturesProcessor,
                            WorldCerealOpticalFeaturesProcessor,
                            WorldCerealThermalFeaturesProcessor,
                            L8ThermalFeaturesProcessor)
from worldcereal.utils.masking import SCL_MASK_VALUES
from worldcereal.features.settings import get_cropland_tsteps_parameters


def std(x):
    return np.std(x, axis=0)


def mean(x):
    return np.mean(x, axis=0, keepdims=True)


def summation(x):
    return np.sum(x, axis=0, keepdims=True)


def test_L2AFeaturesProcessor_1(L2ATraining_collection):

    # TEST 1: standard features on one band
    S2_settings = dict(
        bands=["B02"],
        composite=dict(
            freq=10,
            window=20,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'),
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            multitemporal=True,
            max_invalid_ratio=1)
    )

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings)
    s2_features = s2_fp.compute_features()

    assert 'B02-p50-10m' in s2_features.names


def test_L2AFeaturesProcessor_2(L2ATraining_collection):

    # TEST 2: custom feature on multiple bands,
    # multiple resolutions

    S2_settings = dict(
        bands=["B11"],
        rsis=["ndvi"],
        composite=dict(
            freq=10,
            window=20,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'),
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            max_invalid_ratio=1)
    )
    S2_features_meta = {
        "std": {"function": std,
                "names": ['std']}
    }

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings,
                                 features_meta=S2_features_meta,
                                 ignore_def_features=True)
    s2_features = s2_fp.compute_features()

    assert len(s2_features.names) == 4
    assert 'B11-std-20m' in s2_features.names
    assert 'ndvi-std-10m' in s2_features.names


def test_L2AFeaturesProcessor_3(L2ATraining_collection):

    # TEST 3: custom feature, only on rsis, with whittaker
    # smoothing on ndvi
    S2_settings = dict(
        bands=[],
        rsis=["ndvi", "ndre1"],
        composite=dict(
            freq=10,
            window=20,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'),
        smooth_ndvi=True,
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            max_invalid_ratio=1)
    )
    S2_features_meta = {
        "std": {"function": std,
                "names": ['std']}
    }

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings,
                                 features_meta=S2_features_meta,
                                 ignore_def_features=True)
    s2_features = s2_fp.compute_features()

    assert len(s2_features.names) == 4
    assert 'ndre1-std-20m' in s2_features.names
    assert 'ndvi-std-10m' in s2_features.names


def test_L2AFeaturesProcessor_4(L2ATraining_collection):

    # TEST 4: Use current full L2A settings from
    # worldcereal
    settings = get_cropland_tsteps_parameters()
    S2_settings = settings['settings']['OPTICAL']
    S2_features_meta = settings['features_meta']['OPTICAL']
    S2_rsi_meta = settings['rsi_meta']['OPTICAL']

    # Start and end needs to be added manually
    S2_settings['composite']['start'] = '2018-08-01'
    S2_settings['composite']['end'] = '2019-11-30'

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings,
                                 features_meta=S2_features_meta,
                                 rsi_meta=S2_rsi_meta
                                 )
    _ = s2_fp.compute_features()


def test_L2AFeaturesProcessor_Mask(L2ATraining_collection):

    S2_settings = dict(
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            max_invalid_ratio=1)
    )

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings)
    mask, obs, _, _ = s2_fp.load_mask()

    # Make sure mask is not alway True or always False,
    # which would mean masking went wrong
    assert not np.any(mask[10].sum(axis=0) == obs)
    assert not np.any(mask[20].sum(axis=0) == obs)
    assert not np.any((~mask[10]).sum(axis=0) == obs)
    assert not np.any((~mask[20]).sum(axis=0) == obs)


def test_L2AFeaturesProcessor_MultiTemporalMask(L2ATraining_collection):

    S2_settings = dict(
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            max_invalid_ratio=1,
            multitemporal=True)
    )

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings)

    # Usual mask loading
    orig_mask, _, _, _ = s2_fp.load_mask()

    mask, _ = s2_fp.load_multitemporal_mask(prior_mask=orig_mask)

    # Make sure mask is not alway True or always False,
    # which would mean masking went wrong
    assert not np.sum(mask[10]) == mask[10].size

    # Make sure shapes of mask and TS match
    assert mask[10].shape == orig_mask[10].shape
    assert mask[20].shape == orig_mask[20].shape


def test_L2AFeaturesProcessor_seasons(L2ATraining_collection):

    # custom feature on multiple bands,
    # multiple resolutions,
    # including seasons detection and pheno features

    S2_settings = dict(
        bands=["B11"],
        rsis=["ndvi"],
        composite=dict(
            freq=10,
            window=20,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'),
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            max_invalid_ratio=1,
            multitemporal=True),
        seasons=dict(
            rsis='evi'
        )
    )

    S2_features_meta = {
        "std": {"function": std,
                "names": ['std']},
        "pheno_mult_season": {},
        "pheno_single_season": {
            "select_season": {
                'mode': 'date',
                'param': ['2019-04-15', 60]}}
    }

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings,
                                 features_meta=S2_features_meta,
                                 ignore_def_features=True)
    s2_features = s2_fp.compute_features()

    assert 'B11-std-20m' in s2_features.names
    assert 'ndvi-std-10m' in s2_features.names
    assert 'evi-lSeasMin-10m' in s2_features.names
    assert 'evi-phenoSOS-10m' in s2_features.names
    assert 'evi-phenoBase-10m' in s2_features.names


def test_L2AFeaturesProcessor_seasons_with_smoothing(L2ATraining_collection):

    # custom feature on multiple bands,
    # multiple resolutions,
    # including seasons detection and pheno features
    # version WITH WHITTAKER SMOOTHING

    S2_settings = dict(
        bands=["B11"],
        rsis=["ndvi"],
        composite=dict(
            freq=10,
            window=20,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'),
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            max_invalid_ratio=1,
            multitemporal=True),
        seasons=dict(
            rsis='evi',
            smooth_rsi=True
        )
    )

    S2_features_meta = {
        "std": {"function": std,
                "names": ['std']},
        "pheno_mult_season": {},
        "pheno_single_season": {
            "select_season": {
                'mode': 'date',
                'param': ['2019-04-15', 60]}}
    }

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings,
                                 features_meta=S2_features_meta,
                                 ignore_def_features=True)
    s2_features = s2_fp.compute_features()

    assert 'B11-std-20m' in s2_features.names
    assert 'ndvi-std-10m' in s2_features.names
    assert 'evi-lSeasMin-10m' in s2_features.names
    assert 'evi-phenoSOS-10m' in s2_features.names
    assert 'evi-phenoBase-10m' in s2_features.names


def test_L2AFeaturesProcessor_Augment(L2ATraining_collection):

    S2_settings = dict(
        mask=dict(
            erode_r=3,
            dilate_r=21,
            mask_values=SCL_MASK_VALUES,
            max_invalid_ratio=1)
    )

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings)
    mask, obs, _, _ = s2_fp.load_mask()

    augmented_mask = s2_fp.augment(mask)

    # Make sure we removed some valid acquisitions
    assert augmented_mask[10].sum() < mask[10].sum()


def test_Sigma0FeaturesProcessor(Sigma0Training_collection,
                                 AgERA5Training_collection):

    precipitation_threshold = 10  # mm

    S1_settings = dict(
        bands=["VV", "VH"],
        rsis=["vh_vv", "rvi"],
        composite=dict(
            freq=1,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'
        ),
        mask=dict(
            precipitation_threshold=precipitation_threshold,
            METEOcol=AgERA5Training_collection
        )
    )

    s1_fp = SARFeaturesProcessor(Sigma0Training_collection,
                                 settings=S1_settings)

    s1_features = s1_fp.compute_features()

    print(s1_features.names)


def test_TSSigma0FeaturesProcessor(TerrascopeSigma0_collection,
                                   AgERA5Training_collection):

    S1_settings = dict(
        bands=["VV", "VH"],
        rsis=["vh_vv", "rvi"],
        composite=dict(
            freq=1,
            mode='median',
            start='2018-08-20',
            end='2018-08-31'
        ),
    )

    s1_fp = TSSIGMA0FeaturesProcessor(TerrascopeSigma0_collection,
                                      settings=S1_settings)

    s1_features = s1_fp.compute_features()

    print(s1_features.names)


def test_TSSigma0TiledFeaturesProcessor(TerrascopeSigma0Tiled_collection,
                                        AgERA5Training_collection):

    S1_settings = dict(
        bands=["VV", "VH"],
        rsis=["vh_vv", "rvi"],
        composite=dict(
            freq=1,
            mode='median',
            start='2020-08-20',
            end='2020-08-31'
        ),
    )

    s1_fp = TSSIGMA0TiledFeaturesProcessor(TerrascopeSigma0Tiled_collection,
                                           settings=S1_settings)

    s1_features = s1_fp.compute_features()

    print(s1_features.names)


def test_WorldCerealS0TiledFeaturesProcessor(
        WorldCerealSigma0Tiled_collection):
    S1_settings = dict(
        bands=["VV", "VH"],
        rsis=["vh_vv", "rvi"],
        composite=dict(
            freq=1,
            mode='median',
            start='2019-08-20',
            end='2019-08-31'
        ),
    )

    s1_fp = WorldCerealSARFeaturesProcessor(
        WorldCerealSigma0Tiled_collection,
        settings=S1_settings)

    s1_features = s1_fp.compute_features()

    print(s1_features.names)


def test_Sigma0FeaturesProcessor_augment(Sigma0Training_collection):

    S1_settings = dict(
        bands=["VV", "VH"],
        rsis=["vh_vv", "rvi"],
        composite=dict(
            freq=1,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'
        )
    )

    s1_fp = SARFeaturesProcessor(Sigma0Training_collection,
                                 settings=S1_settings)

    ts = s1_fp.load_data(20)
    valid_before = np.sum(np.any(np.isfinite(ts.data[0, ...]), axis=(1, 2)))
    ts_augmented = s1_fp.augment(ts)
    valid_after = np.sum(np.any(np.isfinite(ts_augmented.data[0, ...]),
                                axis=(1, 2)))

    assert valid_after < valid_before


def test_WorldCerealOpticalTiledFeaturesProcessor(
        WorldCerealOpticalTiled_collection):

    S2_settings = dict(
        bands=["B11"],
        rsis=["ndvi"],
        composite=dict(
            freq=10,
            window=20,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'),
        mask=dict(
            erode_r=3,
            dilate_r=21,
            max_invalid_ratio=1)
    )
    S2_features_meta = {
        "std": {"function": std,
                "names": ['std']}
    }

    s2_fp = WorldCerealOpticalFeaturesProcessor(
        WorldCerealOpticalTiled_collection,
        S2_settings,
        features_meta=S2_features_meta,
        ignore_def_features=True)
    s2_features = s2_fp.compute_features()

    assert len(s2_features.names) == 4
    assert 'B11-std-20m' in s2_features.names
    assert 'ndvi-std-10m' in s2_features.names


def test_WorldCerealThermalTiledFeaturesProcessor(
        WorldCerealThermalTiled_collection):

    L8_settings = dict(
        bands=["B10"],
        composite=dict(
            freq=10,
            window=20,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'),
        mask=dict(
            erode_r=3,
            dilate_r=21,
            max_invalid_ratio=1)
    )
    L8_features_meta = {
        "std": {"function": std,
                "names": ['std']}
    }

    l8_fp = WorldCerealThermalFeaturesProcessor(
        WorldCerealThermalTiled_collection,
        L8_settings,
        features_meta=L8_features_meta,
        ignore_def_features=True)
    l8_features = l8_fp.compute_features()

    assert len(l8_features.names) == 4
    assert 'B10-std-10m' in l8_features.names
    assert l8_features.data.shape[1] == 6


def test_L8ThermalFeaturesProcessor_1(L8ThermalTraining_collection):

    L8_settings = dict(
        bands=["ST-B10"],
        composite=dict(
            freq=16,
            window=32,
            mode='median',
            start='2018-08-01',
            end='2019-11-30'),
        mask=dict(
            erode_r=3,
            dilate_r=21,
            max_invalid_ratio=1)
    )

    l8_fp = L8ThermalFeaturesProcessor(L8ThermalTraining_collection,
                                       L8_settings)
    _ = l8_fp.compute_features()


def test_AgERA5FeaturesProcessor_mean(AgERA5Training_collection):

    AgERA5_mean_settings = dict(
        bands=[
            'dewpoint_temperature',
            'solar_radiation_flux',
            'temperature_mean',
            'vapour_pressure', 'wind_speed',
        ],
        rsis=[],
        composite=dict(
            mode='mean',
            freq=10,
            window=None,
            start='2018-06-01',
            end='2019-11-30'
        ),
    )

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

    AgERA5_features = AgERA5_fp.compute_features()

    assert AgERA5_features.shape == (5, 64, 64)

    print(AgERA5_features)


def test_AgERA5FeaturesProcessor_sum(AgERA5Training_collection):

    AgERA5_sum_settings = dict(
        bands=[
            'precipitation_flux',
        ],
        rsis=[],
        composite=dict(
            mode='sum',
            freq=10,
            window=None,
            start='2018-06-01',
            end='2019-11-30'
        ),
    )

    AgERA5_sum_features_meta = dict(
        sum=dict(
            function=summation,
            names=['sum']
        )
    )

    AgERA5_fp = AgERA5FeaturesProcessor(
        AgERA5Training_collection,
        AgERA5_sum_settings,
        features_meta=AgERA5_sum_features_meta,
        ignore_def_features=True)

    AgERA5_features = AgERA5_fp.compute_features()

    assert AgERA5_features.shape == (1, 64, 64)

    print(AgERA5_features)


@pytest.mark.xfail
def test_agera5_eto_computation(AgERA5Training_collection,
                                dem_collection):
    '''
    KNOWN TO FAIL CURRENTLY ON JENKINGS BECAUSE GRIDDED
    DEM IS NOT ACCESSIBLE
    '''

    # create necessary settings for features processor
    comp_mode_dict = {'dewpoint_temperature': 'median',
                      'precipitation_flux': 'sum',
                      'solar_radiation_flux': 'median',
                      'temperature_max': 'median',
                      'temperature_mean': 'median',
                      'temperature_min': 'median',
                      'vapour_pressure': 'median',
                      'wind_speed': 'median',
                      'et0': 'sum'}

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631
    demcoll_filt = dem_collection.filter_bounds(bounds, epsg)

    agera5_settings = {
        'bands': ["precipitation_flux"],
        'rsis': ["et0"],
        'composite': {
            'freq': 10,
            'window': 10,
            'mode': comp_mode_dict,
            'start': '2018-06-01',
            'end': '2019-11-30'},
        'demcol': demcoll_filt,
        'bounds': bounds,
        'epsg': epsg
    }

    AgERA5_sum_features_meta = dict(
        sum=dict(
            function=summation,
            names=['sum']
        )
    )

    fp = AgERA5FeaturesProcessor(AgERA5Training_collection,
                                 agera5_settings,
                                 features_meta=AgERA5_sum_features_meta,
                                 ignore_def_features=True)

    # compute features
    features = fp.compute_features()

    assert features.shape == (2, 64, 64)


@pytest.mark.skip
def test_S3AgERA5FeaturesProcessor_mean():

    from worldcereal.collections import AgERA5YearlyCollection

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    coll = AgERA5YearlyCollection.from_path('/data/worldcereal/s3collections/satio_agera5_yearly.csv')  # NOQA
    coll = coll.filter_bounds(bounds, epsg).filter_dates('20200301', '20211130')  # NOQA

    AgERA5_mean_settings = dict(
        bands=[
            'dewpoint_temperature',
            'solar_radiation_flux',
            'temperature_mean',
            'vapour_pressure', 'wind_speed',
        ],
        rsis=[],
        composite=dict(
            mode='mean',
            freq=10,
            window=None,
            start='2020-03-01',
            end='2021-11-30'
        ),
    )

    AgERA5_mean_features_meta = dict(
        mean=dict(
            function=mean,
            names=['mean']
        )
    )

    AgERA5_fp = AgERA5FeaturesProcessor(
        coll,
        AgERA5_mean_settings,
        features_meta=AgERA5_mean_features_meta,
        ignore_def_features=True)

    AgERA5_features = AgERA5_fp.compute_features()

    print(AgERA5_features)
