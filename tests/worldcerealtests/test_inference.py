from openeo_gfmap.features.feature_extractor import (
    EPSG_HARMONIZED_NAME,
    apply_feature_extractor_local,
)
from openeo_gfmap.inference.model_inference import apply_model_inference_local

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CropClassifier
from worldcereal.parameters import CropLandParameters, CropTypeParameters
from worldcereal.utils.models import load_model_lut


def test_cropland_inference(WorldCerealPreprocessedInputs):
    """Test the local generation of a cropland product"""

    print("Get Presto cropland features")
    cropland_features = apply_feature_extractor_local(
        PrestoFeatureExtractor,
        WorldCerealPreprocessedInputs,
        parameters={
            EPSG_HARMONIZED_NAME: 32631,
            "ignore_dependencies": True,
            "compile_presto": False,
            "use_valid_date_token": False,
            "presto_model_url": CropLandParameters().feature_parameters.presto_model_url,
        },
    )

    print("Running cropland classification inference UDF locally")

    lookup_table = load_model_lut(
        CropLandParameters().classifier_parameters.classifier_url
    )

    cropland_classification = apply_model_inference_local(
        CropClassifier,
        cropland_features,
        parameters={
            EPSG_HARMONIZED_NAME: 32631,
            "ignore_dependencies": True,
            "lookup_table": lookup_table,
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
        },
    )

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
    """Test the local generation of a croptype product"""

    print("Get Presto croptype features")
    croptype_features = apply_feature_extractor_local(
        PrestoFeatureExtractor,
        WorldCerealPreprocessedInputs,
        parameters={
            EPSG_HARMONIZED_NAME: 32631,
            "ignore_dependencies": True,
            "compile_presto": False,
            "use_valid_date_token": True,
            "presto_model_url": CropTypeParameters().feature_parameters.presto_model_url,
        },
    )

    print("Running croptype classification inference UDF locally")

    lookup_table = load_model_lut(
        CropTypeParameters().classifier_parameters.classifier_url
    )

    croptype_classification = apply_model_inference_local(
        CropClassifier,
        croptype_features,
        parameters={
            EPSG_HARMONIZED_NAME: 32631,
            "ignore_dependencies": True,
            "lookup_table": lookup_table,
            "classifier_url": CropTypeParameters().classifier_parameters.classifier_url,
        },
    )

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
