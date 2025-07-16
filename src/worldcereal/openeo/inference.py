"""Model inference on Presto feature for binary classication"""

import functools
import sys
import copy

from openeo.udf.udf_data import UdfData
from openeo.udf import XarrayDataCube
from openeo.metadata import CollectionMetadata
import numpy as np

import requests
import xarray as xr

sys.path.append("onnx_deps")
import onnxruntime as ort  # noqa: E402

EPSG_HARMONIZED_NAME = "GEO-EPSG"


@functools.lru_cache(maxsize=6)
def load_and_prepare_model(model_url: str):
    """Function to be used instead the default GFMap load_ort_model function.
    Loads the model, validates it and extracts LUT from the model metadata.


    Parameters
    ----------
    model_url : str
        Public URL to the ONNX classification model.
    """
    # Load the model
    response = requests.get(model_url, timeout=120)
    model = ort.InferenceSession(response.content)

    # Validate the model
    metadata = model.get_modelmeta().custom_metadata_map

    if "class_params" not in metadata:
        raise ValueError("Could not find class names in the model metadata.")

    class_params = eval(metadata["class_params"], {"__builtins__": None}, {})

    if "class_names" not in class_params:
        raise ValueError("Could not find class names in the model metadata.")

    if "class_to_label" not in class_params:
        raise ValueError("Could not find class to labels in the model metadata.")

    # Load model LUT
    lut = dict(zip(class_params["class_names"], class_params["class_to_label"]))
    sorted_lut = {k: v for k, v in sorted(lut.items(), key=lambda item: item[1])}

    return model, sorted_lut


def get_output_labels(lut_sorted: dict) -> list:
    """
    Returns the output labels for the classification.

    LUT needs to be explicitly sorted here as openEO does
    not guarantee the order of a json object being preserved when decoding
    a process graph in the backend.
    """
    class_names = lut_sorted.keys()

    return ["classification", "probability"] + [
        f"probability_{name}" for name in class_names
    ]

def predict(onnx_session: ort.InferenceSession, lut_sorted: dict, features: np.ndarray) -> np.ndarray:
    """
    Predicts labels using the provided features array.

    LUT needs to be explicitly sorted here as openEO does
    not guarantee the order of a json object being preserved when decoding
    a process graph in the backend.
    """

    # Prepare input data for ONNX model
    outputs = onnx_session.run(None, {"features": features})

    # Extract classes as INTs and probability of winning class values
    labels = np.zeros((len(outputs[0]),), dtype=np.uint16)
    probabilities = np.zeros((len(outputs[0]),), dtype=np.uint8)
    for i, (label, prob) in enumerate(zip(outputs[0], outputs[1])):
        labels[i] = lut_sorted[label]
        probabilities[i] = int(round(prob[label] * 100))

    # Extract per class probabilities
    output_probabilities = []
    for output_px in outputs[1]:
        output_probabilities.append(
            [output_px[label] for label in lut_sorted.keys()]
        )

    output_probabilities = (
        (np.array(output_probabilities) * 100).round().astype(np.uint8)
    )

    return np.hstack(
        [labels[:, np.newaxis], probabilities[:, np.newaxis], output_probabilities]
    ).transpose()

def execute(inarr: xr.DataArray, parameters: dict, ) -> xr.DataArray:

    if "classifier_url" not in parameters:
        raise ValueError('Missing required parameter "classifier_url"')
    classifier_url = parameters.get("classifier_url")
    # self.logger.info(f'Loading classifier model from "{classifier_url}"')

    # shape and indices for output ("xy", "bands")
    x_coords, y_coords = inarr.x.values, inarr.y.values
    inarr = inarr.transpose("bands", "x", "y").stack(xy=["x", "y"]).transpose()

    onnx_session, lut_sorted = (
        load_and_prepare_model(classifier_url)
    )

    # Run catboost classification
    # self.logger.info("Catboost classification with input shape: %s", inarr.shape)
    classification = predict(onnx_session=onnx_session, lut_sorted=lut_sorted, features=inarr.values)
    # self.logger.info("Classification done with shape: %s", inarr.shape)

    output_labels = get_output_labels(lut_sorted)

    classification_da = xr.DataArray(
        classification.reshape((len(output_labels), len(x_coords), len(y_coords))),
        dims=["bands", "x", "y"],
        coords={
            "bands": output_labels,
            "x": x_coords,
            "y": y_coords,
        },
    )

    return classification_da

# Below comes the actual UDF part

# Apply the Inference UDF
def apply_udf_data(udf_data: UdfData) -> UdfData:
    """This is the actual openeo UDF that will be executed by the backend."""

    cube = udf_data.datacube_list[0]
    parameters = copy.deepcopy(udf_data.user_context)

    proj = udf_data.proj
    if proj is not None:
        proj = proj.get("EPSG")

    parameters[EPSG_HARMONIZED_NAME] = proj

    arr = cube.get_array().transpose("bands", "y", "x")
    arr = execute(
        inarr=arr,
        parameters=parameters,
    ).transpose("bands", "y", "x")

    cube = XarrayDataCube(arr)

    udf_data.datacube_list = [cube]

    return udf_data


# Change band names, since the target labels are parameterized in the UDF
def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:

    _, lut_sorted = load_and_prepare_model(context["classifier_url"])

    return metadata.rename_labels(dimension="bands", target=get_output_labels(lut_sorted))
