import pytest
from openeo_gfmap.inference.model_inference import (
    EPSG_HARMONIZED_NAME,
    apply_model_inference_local,
)

from worldcereal.openeo.postprocess import PostProcessor
from worldcereal.parameters import CropLandParameters, PostprocessParameters


def test_cropland_postprocessing(WorldCerealCroplandClassification):
    """Test the local postprocessing of a cropland product"""

    print("Postprocessing cropland product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroplandClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
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
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
            "method": "majority_vote",
            "kernel_size": 7,
        },
    )


def test_croptype_postprocessing(WorldCerealCroptypeClassification):
    """Test the local postprocessing of a croptype product"""

    print("Postprocessing croptype product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroptypeClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
            "method": "smooth_probabilities",
        },
    )


def test_croptype_postprocessing_majority_vote(WorldCerealCroptypeClassification):
    """Test the local postprocessing of a croptype product"""

    print("Postprocessing croptype product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroptypeClassification,
        parameters={
            "ignore_dependencies": True,
            EPSG_HARMONIZED_NAME: None,
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
            "method": "majority_vote",
            "kernel_size": 7,
        },
    )


def test_postprocessing_parameters():
    """Test the postprocessing parameters."""

    # This set should work
    params = {
        "enable": True,
        "method": "smooth_probabilities",
        "kernel_size": 5,
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
