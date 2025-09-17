"""Private methods for cropland/croptype mapping. The public functions that
are interracting with the methods here are defined in the `worldcereal.job`
sub-module.
"""

from pathlib import Path
from typing import Optional

import openeo
from openeo import DataCube
from openeo_gfmap import TemporalContext
from openeo_gfmap.preprocessing.scaling import compress_uint8, compress_uint16

from worldcereal.parameters import (
    CropLandParameters,
    CropTypeParameters,
    EmbeddingsParameters,
    PostprocessParameters,
    WorldCerealProductType,
)


def _cropland_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    cropland_parameters: CropLandParameters,
    postprocess_parameters: PostprocessParameters,
) -> DataCube:
    """Method to produce cropland map from preprocessed inputs, using
    a Presto feature extractor and a CatBoost classifier.

    Parameters
    ----------
    inputs : DataCube
        preprocessed input cube
    temporal_extent : TemporalContext
        temporal extent of the input cube
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
    feature_parameters = cropland_parameters.feature_parameters.model_dump()

    feature_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "feature_extractor.py",
        context=feature_parameters,
    )

    features = inputs.apply_neighborhood(
        process=feature_udf,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Run model inference on features
    inference_parameters = cropland_parameters.classifier_parameters.model_dump()

    inference_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "inference.py",
        context=inference_parameters,
    )

    classes = features.apply_neighborhood(
        process=inference_udf,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
            {"dimension": "t", "value": "P1D"},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Get rid of temporal dimension
    classes = classes.reduce_dimension(dimension="t", reducer="mean")

    # Postprocess
    if postprocess_parameters.enable:
        if postprocess_parameters.save_intermediate:
            classes = classes.save_result(
                format="GTiff",
                options=dict(
                    filename_prefix=f"{WorldCerealProductType.CROPLAND.value}-raw_{temporal_extent.start_date}_{temporal_extent.end_date}"
                ),
            )
        classes = _postprocess(
            classes,
            postprocess_parameters,
            cropland_parameters.classifier_parameters.classifier_url,
        )

    # Cast to uint8
    classes = compress_uint8(classes)

    return classes


def _croptype_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
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
    temporal_extent : TemporalContext
        temporal extent of the input cube
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
    feature_parameters = croptype_parameters.feature_parameters.model_dump()

    feature_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "feature_extractor.py",
        context=feature_parameters,
    )

    features = inputs.apply_neighborhood(
        process=feature_udf,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Run model inference on features
    inference_parameters = croptype_parameters.classifier_parameters.model_dump()

    inference_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "inference.py",
        context=inference_parameters,
    )

    classes = features.apply_neighborhood(
        process=inference_udf,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
            {"dimension": "t", "value": "P1D"},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Get rid of temporal dimension
    classes = classes.reduce_dimension(dimension="t", reducer="mean")

    # Mask cropland
    if cropland_mask is not None:
        classes = classes.mask(cropland_mask == 0, replacement=254)

    # Postprocess
    if postprocess_parameters.enable:
        if postprocess_parameters.save_intermediate:
            classes = classes.save_result(
                format="GTiff",
                options=dict(
                    filename_prefix=f"{WorldCerealProductType.CROPTYPE.value}-raw_{temporal_extent.start_date}_{temporal_extent.end_date}"
                ),
            )
        classes = _postprocess(
            classes,
            postprocess_parameters,
            classifier_url=croptype_parameters.classifier_parameters.classifier_url,
        )

    # Cast to uint16
    classes = compress_uint16(classes)

    return classes


def _embeddings_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    embeddings_parameters: EmbeddingsParameters,
) -> DataCube:
    """Method to produce embeddings map from preprocessed inputs, using
    a Prometheo feature extractor.

    Parameters
    ----------
    inputs : DataCube
        Preprocessed input cube.
    temporal_extent : TemporalContext
        Temporal extent of the input cube.
    embeddings_parameters : EmbeddingsParameters
        Parameters for the embeddings product inference pipeline.

    Returns
    -------
    DataCube
        Embeddings as a DataCube.
    """

    # Run feature computer
    feature_parameters = embeddings_parameters.feature_parameters.model_dump()

    feature_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "feature_extractor.py",
        context=feature_parameters,
    )

    embeddings = inputs.apply_neighborhood(
        process=feature_udf,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Get rid of temporal dimension
    embeddings = embeddings.reduce_dimension(dimension="t", reducer="mean")

    return embeddings


def _postprocess(
    classes: DataCube,
    postprocess_parameters: "PostprocessParameters",
    classifier_url: Optional[str] = None,
) -> DataCube:
    """Method to postprocess the classes.

    Parameters
    ----------
    classes : DataCube
        classes to postprocess
    postprocess_parameters : PostprocessParameters
        parameter class for postprocessing
    lookup_table: dict
        Mapping of class names to class labels, ordered by model output.
    Returns
    -------
    DataCube
        postprocessed classes
    """

    # Run postprocessing on the raw classification output
    parameters = postprocess_parameters.model_dump()
    parameters.update({"classifier_url": classifier_url})

    postprocess_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "postprocess.py", context=parameters
    )

    postprocessed_classes = classes.apply_neighborhood(
        process=postprocess_udf,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    return postprocessed_classes
