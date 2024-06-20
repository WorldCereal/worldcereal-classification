from pathlib import Path
from typing import Union

import openeo
from openeo_gfmap import BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from openeo_gfmap.inference.model_inference import apply_model_inference
from openeo_gfmap.preprocessing.scaling import compress_uint8

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


def generate_map(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    backend_context: BackendContext,
    output_path: Union[Path, str],
    product: str = "cropland",
    format: str = "GTiff",
) -> Path:
    """Main function to generate a WorldCereal product.

    Args:
        spatial_extent (BoundingBoxExtent): spatial extent of the map
        temporal_extent (TemporalContext): temporal range to consider
        backend_context (BackendContext): backend to run the job on
        output_path (Union[Path, str]): output path to download the product to
        product (str, optional): product describer. Defaults to "cropland".
        format (str, optional): Output format. Defaults to "GTiff".

    Raises:
        ValueError: if the product is not supported

    Returns:
        Path: path to output product
    """

    # Connect to openeo
    connection = openeo.connect(
        "https://openeo.creo.vito.be/openeo/"
    ).authenticate_oidc()

    # Preparing the input cube for the inference
    inputs = worldcereal_preprocessed_inputs_gfmap(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    # Run feature computer
    presto_parameters = {
        "rescale_s1": False,  # Will be done in the Presto UDF itself!
    }

    features = apply_feature_extractor(
        feature_extractor_class=PrestoFeatureExtractor,
        cube=inputs,
        parameters=presto_parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    if product == "cropland":
        # initiate default cropland model
        model_inference_class = CroplandClassifier
        model_inference_parameters = {}
    else:
        raise ValueError(f"Product {product} not supported.")

    if format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    classes = apply_model_inference(
        model_inference_class=model_inference_class,
        cube=features,
        parameters=model_inference_parameters,
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

    # Cast to uint8
    classes = compress_uint8(classes)

    classes.execute_batch(
        outputfile=output_path,
        out_format=format,
        job_options={
            "driver-memory": "4g",
            "executor-memoryOverhead": "12g",
            "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
        },
    )

    return output_path
