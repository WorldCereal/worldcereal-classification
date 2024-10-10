from openeo_gfmap.inference.model_inference import (
    EPSG_HARMONIZED_NAME,
    apply_model_inference_local,
)

from worldcereal.openeo.postprocess import PostProcessor
from worldcereal.parameters import CropTypeParameters
from worldcereal.utils.models import load_model_lut


def test_cropland_postprocessing(WorldCerealCroplandClassification):
    """Test the local postprocessing of a cropland product"""

    print("Postprocessing cropland product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroplandClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "is_binary": True,
            "method": "smooth_probabilities",
        },
    )


def test_cropland_postprocessing_majority_vote(WorldCerealCroplandClassification):
    """Test the local postprocessing of a cropland product"""

    print("Postprocessing cropland product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroplandClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "is_binary": True,
            "method": "majority_vote",
            "kernel_size": 7,
            "conf_threshold": 30,
        },
    )


def test_croptype_postprocessing(WorldCerealCroptypeClassification):
    """Test the local postprocessing of a croptype product"""
    lookup_table = load_model_lut(
        CropTypeParameters().classifier_parameters.classifier_url
    )

    print("Postprocessing croptype product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroptypeClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "is_binary": False,
            "lookup_table": lookup_table,
            "method": "smooth_probabilities",
        },
    )


def test_croptype_postprocessing_majority_vote(WorldCerealCroptypeClassification):
    """Test the local postprocessing of a croptype product"""
    lookup_table = load_model_lut(
        CropTypeParameters().classifier_parameters.classifier_url
    )

    print("Postprocessing croptype product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroptypeClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "is_binary": False,
            "lookup_table": lookup_table,
            "method": "majority_vote",
            "kernel_size": 7,
            "conf_threshold": 30,
        },
    )
