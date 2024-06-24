from pathlib import Path
from typing import Union

import openeo
from openeo_gfmap import BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from openeo_gfmap.inference.model_inference import apply_model_inference
from openeo_gfmap.preprocessing.scaling import compress_uint8, compress_uint16

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


PRODUCT_SETTINGS = {
    "cropland": {
        "features": {
            "extractor": PrestoFeatureExtractor,
            "parameters": {
                "rescale_s1": False,  # Will be done in the Presto UDF itself!
                "presto_model_url": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt",  # NOQA
            },
        },
        "classification": {
            "classifier": CroplandClassifier,
            "parameters": {
                "classifier_url": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA
            },
        },
    },
    "croptype": {
        "features": {
            "extractor": PrestoFeatureExtractor,
            "parameters": {
                "rescale_s1": False,  # Will be done in the Presto UDF itself!
                "presto_model_url": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test.pt",  # NOQA
            },
        },
        "classification": {
            "classifier": CroplandClassifier,  # TODO: update to croptype classifier
            "parameters": {
                "classifier_url": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test_CROPTYPE9.onnx"  # NOQA
            },
        },
    },
}


def generate_map(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    backend_context: BackendContext,
    output_path: Union[Path, str],
    product: str = "cropland",
    format: str = "GTiff",
):
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

    """

    if product not in PRODUCT_SETTINGS.keys():
        raise ValueError(f"Product {product} not supported.")

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
    features = apply_feature_extractor(
        feature_extractor_class=PRODUCT_SETTINGS[product]["features"]["extractor"],
        cube=inputs,
        parameters=PRODUCT_SETTINGS[product]["features"]["parameters"],
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    if format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    classes = apply_model_inference(
        model_inference_class=PRODUCT_SETTINGS[product]["classification"]["classifier"],
        cube=features,
        parameters=PRODUCT_SETTINGS[product]["classification"]["parameters"],
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
    if product == "cropland":
        classes = compress_uint8(classes)
    else:
        classes = compress_uint16(classes)

    classes.execute_batch(
        outputfile=output_path,
        out_format=format,
        job_options={
            "driver-memory": "4g",
            "executor-memoryOverhead": "6g",
            "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
        },
    )


def collect_inputs(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    backend_context: BackendContext,
    output_path: Union[Path, str],
):
    """Function to retrieve preprocessed inputs that are being
    used in the generation of WorldCereal products.

    Args:
        spatial_extent (BoundingBoxExtent): spatial extent of the map
        temporal_extent (TemporalContext): temporal range to consider
        backend_context (BackendContext): backend to run the job on
        output_path (Union[Path, str]): output path to download the product to

    Raises:
        ValueError: if the product is not supported

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

    inputs.execute_batch(
        outputfile=output_path,
        out_format="NetCDF",
        job_options={"driver-memory": "4g", "executor-memoryOverhead": "4g"},
    )
