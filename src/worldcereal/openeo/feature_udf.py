# -*- coding: utf-8 -*-
import sys
from typing import Dict

import numpy as np
from openeo.udf import XarrayDataCube
import pandas as pd
from satio.collections import XArrayTrainingCollection
import xarray as xr

from worldcereal.features.settings import (
    get_cropland_features_meta,
    get_default_rsi_meta)
from worldcereal.fp import L2AFeaturesProcessor

sys.path.append('/data/users/Public/driesj/openeo/deps/satio')
sys.path.append('/data/users/Public/driesj/openeo/deps/wc-classification/src')
# sys.path.insert(0,'/data/users/Public/driesj/openeo/deps/tf230')

wheels = ['loguru-0.5.3-py3-none-any.whl',
          'aiocontextvars-0.2.2-py2.py3-none-any.whl', 'contextvars-2.4',
          'immutables-0.14-cp36-cp36m-manylinux1_x86_64.whl',
          'importlib_resources-3.3.0-py2.py3-none-any.whl']
for wheel in wheels:
    sys.path.append('/data/users/Public/driesj/openeo/deps/' + wheel)


classifier_file = ('/tmp/worldcereal_croplandextent_lpis_unet.h5')


features_meta = get_cropland_features_meta()


class L2AFeaturesProcessor10m(L2AFeaturesProcessor):
    L2A_BANDS_10M = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06',
                     'B07', 'B8A', 'B11', 'B12', 'SCL',
                     'sunAzimuthAngles', 'sunZenithAngles',
                     'viewAzimuthMean', 'viewZenithMean']
    L2A_BANDS_DICT_ALL_10M = {10: L2A_BANDS_10M, 20: {'DUMMY'}}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def supported_bands(self):
        return L2AFeaturesProcessor10m.L2A_BANDS_DICT_ALL_10M


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    This UDF computes WorldCereal features using SatIO.
    It works on a spatiotemporal stack for one specific sensor,
    currently Sentinel-2

    @param cube:
    @param context: A context dictionary, has to contain 'satio_settings'
    @return:
    """
    # access the underlying xarray
    inarr = cube.get_array()

    # translate openEO dim name into satio convention
    inarr = inarr.rename({'t': 'timestamp'})
    # satio expects uint16!
    inarr = inarr.astype(np.uint16)

    settings = context["satio_settings"]
    settings['OPTICAL']['composite']['start'] = np.datetime_as_string(
        inarr.coords['timestamp'].values.min(), unit='D')
    settings['OPTICAL']['composite']['end'] = np.datetime_as_string(
        inarr.coords['timestamp'].values.max(), unit='D')

    classify = context["classify"]

    collection = XArrayTrainingCollection(sensor="S2", processing_level="L2A",
                                          df=pd.DataFrame(), array=inarr)

    from satio.rsindices import RSI_META_S2
    default_rsi_meta = RSI_META_S2.copy()
    rsi_meta = get_default_rsi_meta()['OPTICAL']

    # in openEO, all bands are provided in 10m for now
    # so we need to modify satio defaults
    rsi_meta['brightness'] = default_rsi_meta['brightness']
    rsi_meta['brightness']['native_res'] = 10

    if 'sen2agri_temp_feat' in features_meta.get('OPTICAL', {}):
        features_meta['OPTICAL'][
            'sen2agri_temp_feat'][
                'parameters']['time_start'] = settings['OPTICAL'][
                    'composite']['start']

    processor = L2AFeaturesProcessor10m(collection,
                                        settings['OPTICAL'],
                                        rsi_meta=rsi_meta,
                                        features_meta=features_meta['OPTICAL'])
    features = processor.compute_features()

    # Extracted core from worldcereal ClassificationProcessor,
    # to be seen what we need to keep

    if(classify):

        windowsize = 64
        import tensorflow as tf
        #  from worldcereal.classification.models import WorldCerealUNET

        #  unetmodel = WorldCerealUNET(windowsize=64, features= 60)
        #  unetmodel.model.load_weights(classifier_file)
        #  classifier = unetmodel.model
        classifier = tf.keras.models.load_model(classifier_file)

        xdim = features.data.shape[1]
        ydim = features.data.shape[2]

        prediction = np.empty((xdim, ydim))

        # can be avoided by using openEO apply_neighbourhood
        for xStart in range(0, xdim, windowsize):
            for yStart in range(0, ydim, windowsize):
                # We need to check if we're at the end of the master image
                # We have to make sure we have a full subtile
                # so we need to expand such tile and the resulting overlap
                # with previous subtile is not an issue
                if xStart + windowsize > xdim:
                    xStart = xdim - windowsize
                    xEnd = xdim
                else:
                    xEnd = xStart + windowsize
                if yStart + windowsize > ydim:
                    yStart = ydim - windowsize
                    yEnd = ydim
                else:
                    yEnd = yStart + windowsize

                features_patch = features.data[:,
                                               xStart:xEnd,
                                               yStart:yEnd]
                patchprediction = classifier.predict(
                    features_patch.transpose((1, 2, 0)).reshape(
                        (1,
                         windowsize * windowsize,
                         -1))).squeeze().reshape((windowsize, windowsize))

                prediction[xStart:xEnd, yStart:yEnd] = patchprediction

        prediction_xarray = xr.DataArray(
            prediction.astype(np.float32),
            dims=['x', 'y']
        )

        # wrap back to datacube and return
        return XarrayDataCube(prediction_xarray)

    else:
        features_xarray = xr.DataArray(
            features.data.astype(np.float32),
            dims=['bands', 'x', 'y'],
            coords={'bands': features.names}
        )

        # wrap back to datacube and return
        return XarrayDataCube(features_xarray)
