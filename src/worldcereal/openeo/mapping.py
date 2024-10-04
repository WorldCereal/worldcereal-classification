"""Private methods for cropland/croptype mapping. The public functions that
are interracting with the methods here are defined in the `worldcereal.job`
sub-module.
"""

from typing import Optional

from openeo import DataCube
from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from openeo_gfmap.inference.model_inference import apply_model_inference
from openeo_gfmap.preprocessing.scaling import compress_uint8, compress_uint16

from worldcereal.parameters import (
    CropLandParameters,
    CropTypeParameters,
    PostprocessParameters,
)
from worldcereal.utils.models import load_model_lut


def _cropland_map(
    inputs: DataCube,
    cropland_parameters: CropLandParameters,
    postprocess_parameters: PostprocessParameters,
) -> DataCube:
    """Method to produce cropland map from preprocessed inputs, using
    a Presto feature extractor and a CatBoost classifier.

    Parameters
    ----------
    inputs : DataCube
        preprocessed input cube
    cropland_parameters: CropLandParameters
        Parameters for the cropland product inference pipeline
    postprocess_parameters: PostprocessParameters
        Parameters for the postprocessing
    Returns
    -------
    DataCube
        binary labels and probability
    """

    # Run feature computer
    features = apply_feature_extractor(
        feature_extractor_class=cropland_parameters.feature_extractor,
        cube=inputs,
        parameters=cropland_parameters.features_parameters.model_dump(),
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Run model inference on features
    classes = apply_model_inference(
        model_inference_class=cropland_parameters.classifier,
        cube=features,
        parameters=cropland_parameters.classifier_parameters.model_dump(
            exclude=["classifier"]
        ),
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
            {"dimension": "t", "value": "P1D"},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Postprocess
    if postprocess_parameters.enable:
        if postprocess_parameters.save_intermediate:
            classes = classes.save_result(
                format="GTiff", options=dict(filename_prefix="cropland-raw")
            )
        classes = _postprocess(classes, postprocess_parameters, is_binary=True)

    # Cast to uint8
    classes = compress_uint8(classes)

    return classes


def _croptype_map(
    inputs: DataCube,
    croptype_parameters: "CropTypeParameters",
    postprocess_parameters: "PostprocessParameters",
    cropland_mask: DataCube = None,
) -> DataCube:
    """Method to produce croptype map from preprocessed inputs, using
    a Presto feature extractor and a CatBoost classifier.

    Parameters
    ----------
    inputs : DataCube
        preprocessed input cube
    cropland_mask : DataCube, optional
        optional cropland mask, by default None
    lookup_table: dict,
        Mapping of class names to class labels, ordered by model output.
    Returns
    -------
    DataCube
        croptype labels and probability
    """

    # Run feature computer
    features = apply_feature_extractor(
        feature_extractor_class=croptype_parameters.feature_extractor,
        cube=inputs,
        parameters=croptype_parameters.feature_parameters.model_dump(),
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Run model inference on features
    parameters = croptype_parameters.classifier_parameters.model_dump(
        exclude=["classifier"]
    )

    lookup_table = load_model_lut(
        croptype_parameters.classifier_parameters.classifier_url
    )
    parameters.update({"lookup_table": lookup_table})

    classes = apply_model_inference(
        model_inference_class=croptype_parameters.classifier,
        cube=features,
        parameters=parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
            {"dimension": "t", "value": "P1D"},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Mask cropland
    if cropland_mask is not None:
        classes = classes.mask(cropland_mask == 0, replacement=254)

    # Postprocess
    if postprocess_parameters.enable:
        if postprocess_parameters.save_intermediate:
            classes = classes.save_result(
                format="GTiff", options=dict(filename_prefix="croptype-raw")
            )
        # TODO: check how 254 value is treated during postprocessing
        classes = _postprocess(
            classes, postprocess_parameters, is_binary=False, lookup_table=lookup_table
        )

    # Cast to uint16
    classes = compress_uint16(classes)

    return classes


def _postprocess(
    classes: DataCube,
    postprocess_parameters: "PostprocessParameters",
    is_binary: bool = False,
    lookup_table: Optional[dict] = None,
) -> DataCube:
    """Method to postprocess the classes.

    Parameters
    ----------
    classes : DataCube
        classes to postprocess
    postprocess_parameters : PostprocessParameters
        parameter class for postprocessing
    is_binary : bool, optional
        whether the classes are binary, by default False
    lookup_table: dict, optional
        Mapping of class names to class labels, ordered by model output.
        Required if the classes are not binary.
    Returns
    -------
    DataCube
        postprocessed classes
    """
    if not is_binary and lookup_table is None:
        raise ValueError("Lookup table is required for non-binary classes.")

    # Run postprocessing on the raw classification output
    # Note that this uses the `apply_model_inference` method even though
    # it is not truly model inference
    parameters = postprocess_parameters.model_dump(exclude=["postprocessor"])
    parameters.update({"is_binary": is_binary})
    if lookup_table is not None:
        parameters.update({"lookup_table": lookup_table})

    postprocessed_classes = apply_model_inference(
        model_inference_class=postprocess_parameters.postprocessor,
        cube=classes,
        parameters=parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
            {"dimension": "t", "value": "P1D"},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    return postprocessed_classes
