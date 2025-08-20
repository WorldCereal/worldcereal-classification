import xarray as xr
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData

# Import the UDF functions directly
from worldcereal.openeo.feature_extractor import apply_udf_data as feature_udf
from worldcereal.openeo.inference import apply_udf_data as inference_udf
from worldcereal.parameters import CropLandParameters, CropTypeParameters


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

    # Initialize CropLandParameters
    cropland_params = CropLandParameters().feature_parameters.model_dump()

    # Create UDF data for feature extraction
    feature_udf_data = create_udf_data(WorldCerealPreprocessedInputs, cropland_params)

    # Run feature extraction UDF
    feature_result = feature_udf(feature_udf_data)
    cropland_features = feature_result.datacube_list[0].get_array()

    print("Running cropland classification inference UDF locally")

    # Get classifier parameters
    classifier_params = CropLandParameters().classifier_parameters.model_dump()

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

    # Initialize CropTypeParameters with custom target_date - that's it!
    croptype_params = CropTypeParameters(target_date=None)

    # Get feature parameters and add any testing overrides
    feature_params = croptype_params.feature_parameters.model_dump()

    # Create UDF data for feature extraction
    feature_udf_data = create_udf_data(WorldCerealPreprocessedInputs, feature_params)

    # Run feature extraction UDF
    feature_result = feature_udf(feature_udf_data)
    croptype_features = feature_result.datacube_list[0].get_array()

    print("Running croptype classification inference UDF locally")

    # Get classifier parameters and add any testing overrides
    classifier_params = croptype_params.classifier_parameters.model_dump()

    # Create UDF data for classification
    classifier_udf_data = create_udf_data(croptype_features, classifier_params)

    # Run classification UDF
    classification_result = inference_udf(classifier_udf_data)
    croptype_classification = classification_result.datacube_list[0].get_array()

    assert list(croptype_classification.bands.values) == [
        "classification",
        "probability",
        "probability_barley",
        "probability_dry_pulses_legumes",
        "probability_fibre_crops",
        "probability_flower_crops",
        "probability_grass_fodder_crops",
        "probability_herb_spice_medicinal_crops",
        "probability_maize",
        "probability_millet",
        "probability_oats",
        "probability_other_oilseed",
        "probability_other_root_tuber",
        "probability_potatoes",
        "probability_rapeseed_rape",
        "probability_rice",
        "probability_rye",
        "probability_sorghum",
        "probability_soy_soybeans",
        "probability_spring_barley",
        "probability_spring_oats",
        "probability_spring_rye",
        "probability_spring_triticale",
        "probability_spring_wheat",
        "probability_sugarbeet",
        "probability_sunflower",
        "probability_triticale",
        "probability_vegetables_fruits",
        "probability_wheat",
    ]

    #  First assert below depends on the amount of classes in the model
    assert croptype_classification.sel(bands="classification").values.max() <= 26
    assert croptype_classification.sel(bands="classification").values.min() >= 0
    assert croptype_classification.sel(bands="probability").values.max() <= 100
    assert croptype_classification.sel(bands="probability").values.min() >= 0
    assert croptype_classification.shape == (29, 100, 100)
