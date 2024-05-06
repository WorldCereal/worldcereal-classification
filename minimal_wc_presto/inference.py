#%% import require libraries
import logging
import numpy as np

import xarray as xr
from openeo.udf import XarrayDataCube

from mvp_wc_presto.world_cereal_inference import PrestoFeatureExtractor, WorldCerealPredictor

#TODO; 
#how do we expect out code the stay stabile when presto changes?

from mvp_wc_presto.dataops import (
    BANDS_GROUPS_IDX,
    NORMED_BANDS,
)
from mvp_wc_presto.presto import Presto


#% Mapping from original band names to Presto names
BAND_MAPPING = {
    "B02": "B2",
    "B03": "B3",
    "B04": "B4",
    "B05": "B5",
    "B06": "B6",
    "B07": "B7",
    "B08": "B8",
    "B8A": "B8A",
    "B11": "B11",
    "B12": "B12",
    "VH": "VH",
    "VV": "VV",
    "precipitation-flux": "total_precipitation",
    "temperature-mean": "temperature_2m",
}

# Index to band groups mapping
IDX_TO_BAND_GROUPS = {
    NORMED_BANDS[idx]: band_group_idx
    for band_group_idx, (_, val) in enumerate(BANDS_GROUPS_IDX.items())
    for idx in val
}

def _setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_presto_features(inarr: xr.DataArray, presto_path: str) -> xr.DataArray:
    """
    Extracts features from input data using Presto.

    Args:
        inarr (xr.DataArray): Input data as xarray DataArray.
        presto_path (str): Path to the pretrained Presto model.

    Returns:
        xr.DataArray: Extracted features as xarray DataArray.
    """
    logger = _setup_logging()
    logger.info("Extracting features using Presto ...")
    presto_model = Presto.load_pretrained(model_path=presto_path, strict=False)
    presto_extractor = PrestoFeatureExtractor(presto_model)
    logger.warning("EPSG is hardcoded to 32631 for the time being!")
    features = presto_extractor.extract_presto_features(inarr, epsg=32631)
    return features


def classify_with_catboost(features: np.ndarray, orig_dims: list, model_path: str) -> xr.DataArray:
    """
    Classifies features using the WorldCereal CatBoost model.

    Args:
        features (np.ndarray): Features to be classified.
        orig_dims (list): Original dimensions of the input data.
        model_path (str): Path to the trained CatBoost model.

    Returns:
        xr.DataArray: Classified data as xarray DataArray.
    """
    logger = _setup_logging()
    logger.info("Predicting class using WorldCereal CatBoost model ...")

    predictor = WorldCerealPredictor()
    predictor.load_model(model_path)
    predictions = predictor.predict(features)
    result_da = predictions.to_xarray().to_array(dim="bands").rename({"lon": "x", "lat": "y"})
    result_da = result_da.transpose(*orig_dims)
    result_da = result_da.squeeze('bands')

    return result_da



def apply_datacube(cube: XarrayDataCube) -> XarrayDataCube:
    logger = _setup_logging() 
    logger.info("Applying datacube...")

    inarr = cube.get_array()

    PRESTO_PATH = './model/presto.pt'
    CATBOOST_PATH = './model/wc_catboost.onnx'  

    orig_dims = list(inarr.dims)
    orig_dims.remove("t")

    features = get_presto_features(inarr, PRESTO_PATH)
    classification = classify_with_catboost(features, orig_dims, CATBOOST_PATH)  # Corrected variable name

    return XarrayDataCube(classification)


#test_inference_catboost_presto()





