"""Executing inference jobs on the OpenEO backend."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import openeo
from openeo_gfmap import BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from openeo_gfmap.inference.model_inference import apply_model_inference
from openeo_gfmap.preprocessing.scaling import compress_uint8, compress_uint16

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier, CroptypeClassifier
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap


class WorldCerealProduct(Enum):
    """Enum to define the different WorldCereal products."""

    CROPLAND = "cropland"
    CROPTYPE = "croptype"


ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"

PRODUCT_SETTINGS = {
    WorldCerealProduct.CROPLAND: {
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
    WorldCerealProduct.CROPTYPE: {
        "features": {
            "extractor": PrestoFeatureExtractor,
            "parameters": {
                "rescale_s1": False,  # Will be done in the Presto UDF itself!
                "presto_model_url": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test.pt",  # NOQA
            },
        },
        "classification": {
            "classifier": CroptypeClassifier,
            "parameters": {
                "classifier_url": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test_CROPTYPE9.onnx"  # NOQA
            },
        },
    },
}


@dataclass
class InferenceResults:
    """Dataclass to store the results of the WorldCereal job.

    Attributes
    ----------
    job_id : str
        Job ID of the finished OpenEO job.
    product_url : str
        Public URL to the product accessible of the resulting OpenEO job.
    output_path : Optional[Path]
        Path to the output file, if it was downloaded locally.
    product : WorldCerealProduct
        Product that was generated.
    """

    job_id: str
    product_url: str
    output_path: Optional[Path]
    product: WorldCerealProduct


def generate_map(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    backend_context: BackendContext,
    output_path: Optional[Union[Path, str]],
    product_type: WorldCerealProduct = WorldCerealProduct.CROPLAND,
    out_format: str = "GTiff",
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

    if product_type not in PRODUCT_SETTINGS.keys():
        raise ValueError(f"Product {product_type.value} not supported.")

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
        feature_extractor_class=PRODUCT_SETTINGS[product_type]["features"]["extractor"],
        cube=inputs,
        parameters=PRODUCT_SETTINGS[product_type]["features"]["parameters"],
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    classes = apply_model_inference(
        model_inference_class=PRODUCT_SETTINGS[product_type]["classification"][
            "classifier"
        ],
        cube=features,
        parameters=PRODUCT_SETTINGS[product_type]["classification"]["parameters"],
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
    if product_type == WorldCerealProduct.CROPLAND:
        classes = compress_uint8(classes)
    else:
        classes = compress_uint16(classes)

    job = classes.execute_batch(
        outputfile=output_path,
        out_format=out_format,
        job_options={
            "driver-memory": "4g",
            "executor-memoryOverhead": "6g",
            "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
        },
    )

    asset = job.get_results().get_assets()[0]

    return InferenceResults(
        job_id=classes.job_id,
        product_url=asset.href,
        output_path=output_path,
        product=product_type,
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
