"""Utilities around models for the WorldCereal package."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Union

import onnxruntime as ort
import requests


@lru_cache(maxsize=2)
def load_model_onnx(model_url: Union[str, Path]) -> ort.InferenceSession:
    """Load an ONNX model from a file or URL.

    Parameters
    ----------
    model_url: Union[str, Path]
        path to the ONNX model, either a local path or a public URL.

    Returns
    -------
    ort.InferenceSession
        ONNX model loaded with ONNX runtime.
    """
    # Two minutes timeout to download the model
    if str(model_url).startswith("http"):
        response = requests.get(str(model_url), timeout=120)
        model = response.content
    else:
        model = str(model_url)

    return ort.InferenceSession(model)


def validate_cb_model(model_url: str) -> ort.InferenceSession:
    """Validate a catboost model by loading it and checking if the required
    metadata is present. Checks for the `class_names` and `class_to_labels`
    fields are present in the `class_params` field of the custom metadata of
    the model. By default, the CatBoost module should include those fields
    when exporting a model to ONNX.

    Raises an exception if the model is not valid.

    Parameters
    ----------
    model_url : str
        URL to the ONNX model.

    Returns
    -------
    ort.InferenceSession
        ONNX model loaded with ONNX runtime.
    """
    model = load_model_onnx(model_url=model_url)

    metadata = model.get_modelmeta().custom_metadata_map

    if "class_params" not in metadata:
        raise ValueError("Could not find class names in the model metadata.")

    class_params = json.loads(metadata["class_params"])

    if "class_names" not in class_params:
        raise ValueError("Could not find class names in the model metadata.")

    if "class_to_label" not in class_params:
        raise ValueError("Could not find class to labels in the model metadata.")

    return model


def load_model_lut(model_url: str) -> dict:
    """Load the class names to labels mapping from a CatBoost model.

    Parameters
    ----------
    model_url : str
        URL to the ONNX model.

    Returns
    -------
    dict
        Look-up table with class names and labels.
    """
    model = validate_cb_model(model_url=model_url)
    metadata = model.get_modelmeta().custom_metadata_map
    class_params = json.loads(metadata["class_params"])

    return dict(zip(class_params["class_names"], class_params["class_to_label"]))
