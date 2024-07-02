"""Executing inference jobs on the OpenEO backend."""

from enum import Enum
from pathlib import Path
from typing import Optional, Union

import openeo
from openeo import DataCube
from openeo_gfmap import BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from openeo_gfmap.inference.model_inference import apply_model_inference
from openeo_gfmap.preprocessing.scaling import compress_uint8, compress_uint16
from pydantic import BaseModel

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier, CroptypeClassifier
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap


class WorldCerealProduct(Enum):
    """Enum to define the different WorldCereal products."""

    CROPLAND = "cropland"
    CROPTYPE = "croptype"


class FeaturesParameters(BaseModel):
    """Parameters for the feature extraction UDFs. Types are enforced by
    Pydantic.

    Attributes
    ----------
    rescale_s1 : bool (default=False)
        Whether to rescale Sentinel-1 bands before feature extraction. Should be
        left to False, as this is done in the Presto UDF itself.
    presto_model_url : str
        Public URL to the Presto model used for feature extraction. The file
        should be a PyTorch serialized model.
    """

    rescale_s1: bool
    presto_model_url: str


class ClassifierParameters(BaseModel):
    """Parameters for the classifier. Types are enforced by Pydantic.

    Attributes
    ----------
    classifier_url : str
        Public URL to the classifier model. Te file should be an ONNX accepting
        a `features` field for input data and returning two output probability
        arrays `true` and `false`.
    """

    classifier_url: str


class CropLandParameters(BaseModel):
    """Parameters for the cropland product inference pipeline. Types are
    enforced by Pydantic.

    Attributes
    ----------
    feature_extractor : PrestoFeatureExtractor
        Feature extractor to use for the inference. This class must be a
        subclass of GFMAP's `PatchFeatureExtractor` and returns float32
        features.
    features_parameters : FeaturesParameters
        Parameters for the feature extraction UDF. Will be serialized into a
        dictionnary and passed in the process graph.
    classifier : CroplandClassifier
        Classifier to use for the inference. This class must be a subclass of
        GFMAP's `ModelInference` and returns predictions/probabilities for
        cropland.
    classifier_parameters : ClassifierParameters
        Parameters for the classifier UDF. Will be serialized into a dictionnary
        and passed in the process graph.
    """

    feature_extractor: PrestoFeatureExtractor = PrestoFeatureExtractor
    features_parameters: FeaturesParameters = FeaturesParameters(
        rescale_s1=False,
        presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt",  # NOQA
    )
    classifier: CroplandClassifier = CroplandClassifier
    classifier_parameters: ClassifierParameters = ClassifierParameters(
        classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA
    )


class CropTypeParameters(BaseModel):
    """Parameters for the croptype product inference pipeline. Types are
    enforced by Pydantic.

    Attributes
    ----------
    feature_extractor : PrestoFeatureExtractor
        Feature extractor to use for the inference. This class must be a
        subclass of GFMAP's `PatchFeatureExtractor` and returns float32
        features.
    features_parameters : FeaturesParameters
        Parameters for the feature extraction UDF. Will be serialized into a
        dictionnary and passed in the process graph.
    classifier : CroptypeClassifier
        Classifier to use for the inference. This class must be a subclass of
        GFMAP's `ModelInference` and returns predictions/probabilities for
        cropland classes.
    classifier_parameters : ClassifierParameters
        Parameters for the classifier UDF. Will be serialized into a dictionnary
        and passed in the process graph.
    """

    feature_extractor: PrestoFeatureExtractor = PrestoFeatureExtractor
    feature_parameters: FeaturesParameters = FeaturesParameters(
        rescale_s1=False,
        presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test.pt",  # NOQA
    )
    classifier: CroptypeClassifier = CroptypeClassifier
    classifier_parameters: ClassifierParameters = ClassifierParameters(
        classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test_CROPTYPE9.onnx"  # NOQA
    )


ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


class InferenceResults(BaseModel):
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
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    out_format: str = "GTiff",
) -> InferenceResults:
    """Main function to generate a WorldCereal product.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    backend_context : BackendContext
        backend to run the job on
    output_path : Optional[Union[Path, str]]
        output path to download the product to
    product_type : WorldCerealProduct, optional
        product describer, by default WorldCerealProduct.CROPLAND
    out_format : str, optional
        Output format, by default "GTiff"

    Returns
    -------
    InferenceResults
        Results of the finished WorldCereal job.

    Raises
    ------
    ValueError
        if the product is not supported
    ValueError
        if the out_format is not supported
    """

    if product_type not in WorldCerealProduct:
        raise ValueError(f"Product {product_type.value} not supported.")

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    # Connect to openeo
    connection = openeo.connect(
        "https://openeo.creo.vito.be/openeo/"
    ).authenticate_oidc()

    # Preparing the input cube for inference
    inputs = worldcereal_preprocessed_inputs_gfmap(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    # Explicit filtering again for bbox because of METEO low
    # resolution causing issues
    inputs = inputs.filter_bbox(dict(spatial_extent))

    # Construct the feature extraction and model inference pipeline
    if product_type == WorldCerealProduct.CROPLAND:
        classes = _cropland_map(inputs, cropland_parameters=cropland_parameters)
    elif product_type == WorldCerealProduct.CROPTYPE:
        # First compute cropland map
        cropland_mask = (
            _cropland_map(inputs, cropland_parameters=cropland_parameters)
            .filter_bands("classification")
            .reduce_dimension(
                dimension="t", reducer="mean"
            )  # Temporary fix to make this work as mask
        )

        classes = _croptype_map(
            inputs,
            croptype_parameters=croptype_parameters,
            cropland_mask=cropland_mask,
        )

    # Submit the job
    job = classes.execute_batch(
        outputfile=output_path,
        out_format=out_format,
        job_options={
            "driver-memory": "4g",
            "executor-memoryOverhead": "4g",
            "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
        },
    )

    asset = job.get_results().get_assets()[0]

    return InferenceResults(
        job_id=job.job_id,
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

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    backend_context : BackendContext
        backend to run the job on
    output_path : Union[Path, str]
        output path to download the product to

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


def _cropland_map(
    inputs: DataCube, cropland_parameters: CropLandParameters
) -> DataCube:
    """Method to produce cropland map from preprocessed inputs, using
    a Presto feature extractor and a CatBoost classifier.

    Parameters
    ----------
    inputs : DataCube
        preprocessed input cube

    Returns
    -------
    DataCube
        binary labels and probability
    """

    # Run feature computer
    features = apply_feature_extractor(
        feature_extractor_class=cropland_parameters.feature_extractor,
        cube=inputs,
        parameters=cropland_parameters.features_parameters.dict(),
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
        parameters=cropland_parameters.classifier_parameters.dict(),
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

    return classes


def _croptype_map(
    inputs: DataCube,
    croptype_parameters: CropTypeParameters,
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

    Returns
    -------
    DataCube
        croptype labels and probability
    """

    # Run feature computer
    features = apply_feature_extractor(
        feature_extractor_class=croptype_parameters.feature_extractor,
        cube=inputs,
        parameters=croptype_parameters.feature_parameters.dict(),
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
        model_inference_class=croptype_parameters.classifier,
        cube=features,
        parameters=croptype_parameters.classifier_parameters.dict(),
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
        classes = classes.mask(cropland_mask == 0, replacement=0)

    # Cast to uint16
    classes = compress_uint16(classes)

    return classes
