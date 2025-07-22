import xarray as xr
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData

# Import the UDF functions directly
from worldcereal.openeo.feature_extractor import apply_udf_data as feature_udf
from worldcereal.openeo.inference import apply_udf_data as inference_udf
from worldcereal.parameters import CropLandParameters, CropTypeParameters
from worldcereal.utils.models import load_model_lut


def create_udf_data(arr: xr.DataArray, parameters: dict, epsg: int = 32631) -> UdfData:
    """Create UdfData object from xarray DataArray and parameters."""
    cube = XarrayDataCube(arr)
    udf_data = UdfData(
        datacube_list=[cube], user_context=parameters, proj={"EPSG": epsg}
    )
    return udf_data


def test_cropland_inference(WorldCerealPreprocessedInputs):
    """Test the local generation of a cropland product using direct UDF calls"""

    print("Get Presto cropland features")

    # Feature extraction parameters for cropland
    feature_params = {
        "ignore_dependencies": True,
        "compile_presto": False,
        "presto_model_url": CropLandParameters().feature_parameters.presto_model_url,
        "rescale_s1": CropLandParameters().feature_parameters.rescale_s1,
        "temporal_prediction": CropLandParameters().feature_parameters.temporal_prediction,
    }

    # Create UDF data for feature extraction
    feature_udf_data = create_udf_data(WorldCerealPreprocessedInputs, feature_params)

    # Run feature extraction UDF
    feature_result = feature_udf(feature_udf_data)
    cropland_features = feature_result.datacube_list[0].get_array()

    print("Running cropland classification inference UDF locally")

    # Classification parameters for cropland
    classifier_params = {
        "ignore_dependencies": True,
        "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
    }

    # Create UDF data for classification
    classifier_udf_data = create_udf_data(cropland_features, classifier_params)

    # Run classification UDF
    classification_result = inference_udf(classifier_udf_data)
    cropland_classification = classification_result.datacube_list[0].get_array()

    assert list(cropland_classification.bands.values) == [
        "classification",
        "probability",
        "probability_other",
        "probability_cropland",
    ]
    assert cropland_classification.sel(bands="classification").values.max() <= 1
    assert cropland_classification.sel(bands="classification").values.min() >= 0
    assert cropland_classification.sel(bands="probability").values.max() <= 100
    assert cropland_classification.sel(bands="probability").values.min() >= 0
    assert cropland_classification.shape == (4, 100, 100)


def test_croptype_inference(WorldCerealPreprocessedInputs):
    """Test the local generation of a croptype product using direct UDF calls"""

    print("Get Presto croptype features")

    # Feature extraction parameters for croptype
    feature_params = {
        "ignore_dependencies": True,
        "compile_presto": False,
        "presto_model_url": CropTypeParameters().feature_parameters.presto_model_url,
        "rescale_s1": False,
        "temporal_prediction": True,
        "target_date": None,  # Use middle timestep
    }

    # Create UDF data for feature extraction
    feature_udf_data = create_udf_data(WorldCerealPreprocessedInputs, feature_params)

    # Run feature extraction UDF
    feature_result = feature_udf(feature_udf_data)
    croptype_features = feature_result.datacube_list[0].get_array()

    print("Running croptype classification inference UDF locally")

    lookup_table = load_model_lut(
        CropTypeParameters().classifier_parameters.classifier_url
    )

    # Classification parameters for croptype
    classifier_params = {
        "ignore_dependencies": True,
        "lookup_table": lookup_table,
        "classifier_url": CropTypeParameters().classifier_parameters.classifier_url,
    }

    # Create UDF data for classification
    classifier_udf_data = create_udf_data(croptype_features, classifier_params)

    # Run classification UDF
    classification_result = inference_udf(classifier_udf_data)
    croptype_classification = classification_result.datacube_list[0].get_array()

    assert list(croptype_classification.bands.values) == [
        "classification",
        "probability",
        "probability_barley",
        "probability_maize",
        "probability_millet_sorghum",
        "probability_other_crop",
        "probability_rapeseed_rape",
        "probability_soy_soybeans",
        "probability_sunflower",
        "probability_wheat",
    ]

    #  First assert below depends on the amount of classes in the model
    assert croptype_classification.sel(bands="classification").values.max() <= 7
    assert croptype_classification.sel(bands="classification").values.min() >= 0
    assert croptype_classification.sel(bands="probability").values.max() <= 100
    assert croptype_classification.sel(bands="probability").values.min() >= 0
    assert croptype_classification.shape == (10, 100, 100)
