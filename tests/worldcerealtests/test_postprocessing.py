import pytest
from openeo_gfmap.inference.model_inference import (
    EPSG_HARMONIZED_NAME,
    apply_model_inference_local,
)

from worldcereal.openeo.postprocess import PostProcessor
from worldcereal.parameters import (
    CropLandParameters,
    CropTypeParameters,
    PostprocessParameters,
)
from worldcereal.utils.models import load_model_lut


def test_cropland_postprocessing(WorldCerealCroplandClassification):
    """Test the local postprocessing of a cropland product"""

    lookup_table = load_model_lut(
        CropLandParameters().classifier_parameters.classifier_url
    )

    print("Postprocessing cropland product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroplandClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "lookup_table": lookup_table,
            "method": "smooth_probabilities",
        },
    )


def test_cropland_postprocessing_majority_vote(WorldCerealCroplandClassification):
    """Test the local postprocessing of a cropland product"""

    lookup_table = load_model_lut(
        CropLandParameters().classifier_parameters.classifier_url
    )

    print("Postprocessing cropland product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroplandClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "lookup_table": lookup_table,
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
            "lookup_table": lookup_table,
            "method": "majority_vote",
            "kernel_size": 7,
            "conf_threshold": 30,
        },
    )


def test_postprocessing_parameters():
    """Test the postprocessing parameters."""

    # This set should work
    params = {
        "enable": True,
        "method": "smooth_probabilities",
        "kernel_size": 5,
        "conf_threshold": 30,
        "save_intermediate": False,
        "keep_class_probs": False,
    }
    PostprocessParameters(**params)

    # This one as well
    params["method"] = "majority_vote"
    PostprocessParameters(**params)

    # This one should fail with invalid kernel size
    params["kernel_size"] = 30
    with pytest.raises(ValueError):
        PostprocessParameters(**params)

    # This one should fail with invalid conf_threshold
    params["kernel_size"] = 5
    params["conf_threshold"] = 101
    with pytest.raises(ValueError):
        PostprocessParameters(**params)

    # This one should fail with invalid method
    params["method"] = "test"
    with pytest.raises(ValueError):
        PostprocessParameters(**params)

    # This one should fail with invalid save_intermediate
    params["enable"] = False
    params["save_intermediate"] = True
    params["method"] = "smooth_probabilities"
    with pytest.raises(ValueError):
        PostprocessParameters(**params)
