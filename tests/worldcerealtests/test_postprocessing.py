import pytest
import xarray as xr


from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData

from worldcereal.openeo.postprocess import apply_udf_data as postprocess_udf
from worldcereal.parameters import CropLandParameters, PostprocessParameters


def create_udf_data(arr: xr.DataArray, parameters: dict, epsg: int = 32631) -> UdfData:
    """Create UdfData object from xarray DataArray and parameters."""
    cube = XarrayDataCube(arr)
    udf_data = UdfData(
        datacube_list=[cube], user_context=parameters, proj={"EPSG": epsg}
    )
    return udf_data


def test_cropland_postprocessing(WorldCerealCroplandClassification):
    """Test the local postprocessing of a cropland product"""

    print("Postprocessing cropland product ...")

    parameters={
            "ignore_dependencies": True,
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
            "method": "smooth_probabilities",
        }

    postprocess_udf_data = create_udf_data(
        arr=WorldCerealCroplandClassification,
        parameters=parameters)

    postprocess_result = postprocess_udf(postprocess_udf_data)
    _ = postprocess_result.datacube_list[0].get_array()



def test_cropland_postprocessing_majority_vote(WorldCerealCroplandClassification):
    """Test the local postprocessing of a cropland product"""

    print("Postprocessing cropland product ...")

    parameters={
            "ignore_dependencies": True,
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
            "method": "majority_vote",
            "kernel_size": 7,
        }

    postprocess_udf_data = create_udf_data(
        arr=WorldCerealCroplandClassification,
        parameters=parameters)

    postprocess_result = postprocess_udf(postprocess_udf_data)
    _ = postprocess_result.datacube_list[0].get_array()


def test_croptype_postprocessing(WorldCerealCroptypeClassification):
    """Test the local postprocessing of a croptype product"""

    print("Postprocessing croptype product ...")

    parameters={
            "ignore_dependencies": True,
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
            "method": "smooth_probabilities",
        }

    postprocess_udf_data = create_udf_data(
        arr=WorldCerealCroptypeClassification,
        parameters=parameters)

    postprocess_result = postprocess_udf(postprocess_udf_data)
    _ = postprocess_result.datacube_list[0].get_array()




def test_croptype_postprocessing_majority_vote(WorldCerealCroptypeClassification):
    """Test the local postprocessing of a croptype product"""

    print("Postprocessing croptype product ...")

    parameters={
            "ignore_dependencies": True,
            "classifier_url": CropLandParameters().classifier_parameters.classifier_url,
            "method": "majority_vote",
            "kernel_size": 7,
        }

    postprocess_udf_data = create_udf_data(
        arr=WorldCerealCroptypeClassification,
        parameters=parameters)

    postprocess_result = postprocess_udf(postprocess_udf_data)
    _ = postprocess_result.datacube_list[0].get_array()


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
