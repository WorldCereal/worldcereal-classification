try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import numpy as np
import tempfile
import rasterio
from satio import layers
from worldcereal.utils import scalers, aez
from worldcereal.utils.masking import SCL_MASK_VALUES
from worldcereal.utils.scalers import _SCALERANGES
from worldcereal.fp import L2AFeaturesProcessor
from satio.rsindices import RSI_META_S2
from worldcereal.classification.weights import load_refidweights
from worldcereal.utils import (BIOME_RASTERS,
                               get_matching_realm_id,
                               probability_to_binary,
                               probability_to_confidence)
from worldcereal.utils.io import (save_features_geotiff,
                                  load_features_geotiff)


S2_GRID = layers.load('s2grid')


def test_minmaxscaler():

    ft_name = 'SAR-VV-ts-20m'
    minvalue = _SCALERANGES[ft_name]['min']
    maxvalue = _SCALERANGES[ft_name]['max']

    inputvalues = np.array([minvalue, maxvalue])

    scaled = scalers.minmaxscaler(inputvalues,
                                  ft_name=ft_name,
                                  minscaled=0,
                                  maxscaled=1)

    assert np.min(scaled) == 0
    assert np.max(scaled) == 1

    scaled = scalers.minmaxscaler(inputvalues,
                                  ft_name=ft_name,
                                  minscaled=-1,
                                  maxscaled=1)

    assert np.min(scaled) == -1
    assert np.max(scaled) == 1


def test_minmaxunscaler():

    ft_name = 'OPTICAL-ndvi-p10-10m'
    minvalue = _SCALERANGES[ft_name]['min']
    maxvalue = _SCALERANGES[ft_name]['max']

    inputvalues = np.array([0, 1])

    unscaled = scalers.minmaxunscaler(inputvalues,
                                      ft_name=ft_name,
                                      minscaled=0,
                                      maxscaled=1)

    assert np.min(unscaled) == minvalue
    assert np.max(unscaled) == maxvalue

    inputvalues = np.array([-1, 1])

    unscaled = scalers.minmaxunscaler(inputvalues,
                                      ft_name=ft_name,
                                      minscaled=-1,
                                      maxscaled=1)

    assert np.min(unscaled) == minvalue
    assert np.max(unscaled) == maxvalue


def test_save_features(L2ATraining_collection):
    '''Test functionality of feature saving
    '''

    # First compute some features
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
    features = s2_fp.compute_features()

    # Now save them
    with tempfile.TemporaryDirectory(dir='.') as basedir:
        features_file = basedir + '/saved_features.tif'
        save_features_geotiff(features,
                              bounds=(574680, 5621800, 575320, 5622440),
                              epsg=32631,
                              filename=features_file,
                              compress_tag='deflate-uint16-lsb11-z9')

        restored_fts = load_features_geotiff(features_file)

        assert restored_fts.names == features.names

        # Compression has been applied. Compare values
        # with some tolerance
        assert np.allclose(features.data,
                           restored_fts.data,
                           equal_nan=True,
                           rtol=0.05,
                           atol=0.01)


def test_aez():
    import geopandas as gpd
    geodataframe = aez.load()
    assert isinstance(geodataframe, gpd.GeoDataFrame)
    assert geodataframe.size > 200
    assert list(geodataframe.columns.values) == [
        'fid', 'zoneID', 'groupID', 'wwsos_min', 'wwsos_avg', 'wwsos_max',
        'wwsos_var', 'wwsos_sd', 'wweos_min', 'wweos_avg', 'wweos_max',
        'wweos_var', 'wweos_sd', 'm1sos_min', 'm1sos_avg', 'm1sos_max',
        'm1sos_var', 'm1sos_sd', 'm1eos_min', 'm1eos_avg', 'm1eos_max',
        'm1eos_var', 'm1eos_sd', 'm2sos_min', 'm2sos_avg', 'm2sos_max',
        'm2sos_var', 'm2sos_sd', 'm2eos_min', 'm2eos_avg', 'm2eos_max',
        'm2eos_var', 'm2eos_sd', 'trigger_sw', 'L8', 'irr_stats',
        'potapov_cropfraction', 'annual_eos', 'annual_sos', 'geometry']

    ids = aez.ids_from_group(2000)

    assert ids == [2004, 2005, 2006, 2007, 2008, 2154, 2161]
    for id in ids:
        assert aez.group_from_id(id) == 2000


def test_get_realm_id():
    geom = S2_GRID[S2_GRID.tile == '31UFS'].geometry.values[0]
    get_matching_realm_id(geom)


def test_whittakersmoother(L2ATraining_collection):

    S2_settings = dict(
        bands=["B04", "B08"],
        rsis=["evi"],
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
            multitemporal=True
        )
    )

    s2_fp = L2AFeaturesProcessor(L2ATraining_collection,
                                 S2_settings)

    # Load the SCL-based mask
    mask, _, _, _ = s2_fp.load_mask()

    # Load the multitemporal mask
    mask, ts = s2_fp.load_multitemporal_mask(prior_mask=mask)

    ts = s2_fp.load_data(10, timeseries=ts)
    ts_proc = s2_fp.preprocess_data(ts, 10, mask=mask,
                                    composite=False,
                                    interpolate=False)
    rsi = ts_proc.compute_rsis('evi', rsi_meta=RSI_META_S2)

    # RSI 0 means no data
    rsi.data[rsi.data == 0] = np.nan

    # Run whittaker smoother
    _ = s2_fp.smooth_whittaker(rsi)


def test_loadrefidweights():
    load_refidweights()


def test_loadbiomes():
    from worldcereal.resources import biomes

    for _, raster_path in BIOME_RASTERS.items():
        with pkg_resources.open_binary(biomes, raster_path) as src:
            rasterio.open(src)


def test_proba_to_prediction():
    probs = np.array([[0.25, 0.75], [0.1, 0.9],
                      [0.5, 0.5], [0.6, 0.4]])

    # Usual case
    threshold = 0.5
    pred = probability_to_binary(probs, threshold)
    conf = probability_to_confidence(probs)

    assert np.array_equal(pred, [1, 1, 1, 0])
    assert np.allclose(conf, [0.5, 0.8, 0., 0.2])

    # Custom threshold case
    threshold = 0.7
    probs = np.array([[0.3, 0.7], [0.15, 0.85],
                      [0.35, 0.65], [1, 0], [0, 1]])
    pred = probability_to_binary(probs, threshold)
    conf = probability_to_confidence(probs)

    assert np.array_equal(pred, [1, 1, 0, 0, 1])
    assert np.allclose(conf, [0.4, 0.7, 0.3, 1, 1])
