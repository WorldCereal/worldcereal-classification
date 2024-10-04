from openeo_gfmap.features.feature_extractor import (
    EPSG_HARMONIZED_NAME,
    apply_feature_extractor_local,
)
from openeo_gfmap.inference.model_inference import apply_model_inference_local

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier, CroptypeClassifier
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
        },
    )

    print("Running cropland classification inference UDF locally")

    cropland_classification = apply_model_inference_local(
        CroplandClassifier,
        cropland_features,
        parameters={
            EPSG_HARMONIZED_NAME: 32631,
            "ignore_dependencies": True,
        },
    )

    assert list(cropland_classification.bands.values) == [
        "classification",
        "probability",
    ]
    assert cropland_classification.sel(bands="classification").values.max() <= 1
    assert cropland_classification.sel(bands="classification").values.min() >= 0
    assert cropland_classification.sel(bands="probability").values.max() <= 100
    assert cropland_classification.sel(bands="probability").values.min() >= 0
    assert cropland_classification.shape == (2, 100, 100)


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
        },
    )

    print("Running croptype classification inference UDF locally")

    lookup_table = load_model_lut(CroptypeClassifier.CATBOOST_PATH)

    croptype_classification = apply_model_inference_local(
        CroptypeClassifier,
        croptype_features,
        parameters={
            EPSG_HARMONIZED_NAME: 32631,
            "ignore_dependencies": True,
            "lookup_table": lookup_table,
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
