"""Executing inference jobs on the OpenEO backend."""

from enum import Enum
from pathlib import Path
from typing import Optional, Type, Union

import openeo
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.features.feature_extractor import PatchFeatureExtractor
from openeo_gfmap.inference.model_inference import ModelInference
from pydantic import BaseModel, Field, ValidationError, model_validator

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier, CroptypeClassifier
from worldcereal.openeo.mapping import _cropland_map, _croptype_map
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
    compile_presto : bool (default=False)
        Whether to compile the Presto encoder for speeding up large-scale inference.
    """

    rescale_s1: bool
    presto_model_url: str
    compile_presto: bool


class ClassifierParameters(BaseModel):
    """Parameters for the classifier. Types are enforced by Pydantic.

    Attributes
    ----------
    classifier_url : str
        Public URL to the classifier model. Te file should be an ONNX accepting
        a `features` field for input data and returning either two output
        probability arrays `true` and `false` in case of cropland mapping, or
        a probability array per-class in case of croptype mapping.
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

    feature_extractor: Type[PatchFeatureExtractor] = Field(
        default=PrestoFeatureExtractor
    )
    features_parameters: FeaturesParameters = FeaturesParameters(
        rescale_s1=False,
        presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/presto.pt",  # NOQA
        compile_presto=False,
    )
    classifier: Type[ModelInference] = Field(default=CroplandClassifier)
    classifier_parameters: ClassifierParameters = ClassifierParameters(
        classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal-minimal-inference/wc_catboost.onnx"  # NOQA
    )

    @model_validator(mode="after")
    def check_udf_types(self):
        """Validates the FeatureExtractor and Classifier classes."""
        if not issubclass(self.feature_extractor, PatchFeatureExtractor):
            raise ValidationError(
                f"Feature extractor must be a subclass of PrestoFeatureExtractor, got {self.feature_extractor}"
            )
        if not issubclass(self.classifier, ModelInference):
            raise ValidationError(
                f"Classifier must be a subclass of ModelInference, got {self.classifier}"
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

    feature_extractor: Type[PatchFeatureExtractor] = Field(
        default=PrestoFeatureExtractor
    )
    feature_parameters: FeaturesParameters = FeaturesParameters(
        rescale_s1=False,
        presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test.pt",  # NOQA
        compile_presto=False,
    )
    classifier: Type[ModelInference] = Field(default=CroptypeClassifier)
    classifier_parameters: ClassifierParameters = ClassifierParameters(
        classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test_CROPTYPE9.onnx"  # NOQA
    )

    @model_validator(mode="after")
    def check_udf_types(self):
        """Validates the FeatureExtractor and Classifier classes."""
        if not issubclass(self.feature_extractor, PatchFeatureExtractor):
            raise ValidationError(
                f"Feature extractor must be a subclass of PrestoFeatureExtractor, got {self.feature_extractor}"
            )
        if not issubclass(self.classifier, ModelInference):
            raise ValidationError(
                f"Classifier must be a subclass of ModelInference, got {self.classifier}"
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
    output_path: Optional[Union[Path, str]],
    product_type: WorldCerealProduct = WorldCerealProduct.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: Optional[CropTypeParameters] = CropTypeParameters(),
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.FED),
    tile_size: Optional[int] = 128,
    job_options: Optional[dict] = None,
) -> InferenceResults:
    """Main function to generate a WorldCereal product.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    output_path : Optional[Union[Path, str]]
        output path to download the product to
    product_type : WorldCerealProduct, optional
        product describer, by default WorldCerealProduct.CROPLAND
    cropland_parameters: CropLandParameters
        Parameters for the cropland product inference pipeline.
    croptype_parameters: Optional[CropTypeParameters]
        Parameters for the croptype product inference pipeline. Only required
        whenever `product_type` is set to `WorldCerealProduct.CROPTYPE`, will be
        ignored otherwise.
    out_format : str, optional
        Output format, by default "GTiff"
    backend_context : BackendContext
        backend to run the job on
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.
    job_options: dict, optional
        Additional job options to pass to the OpenEO backend, by default None

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
        tile_size=tile_size,
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
    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "1g",
        "executor-memoryOverhead": "1g",
        "python-memory": "3g",
        "soft-errors": "true",
        "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    job = classes.execute_batch(
        outputfile=output_path,
        out_format=out_format,
        job_options=JOB_OPTIONS,
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
    tile_size: Optional[int] = None,
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
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default None
        so it uses the OpenEO default setting.
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
        tile_size=tile_size,
    )

    inputs.execute_batch(
        outputfile=output_path,
        out_format="NetCDF",
        job_options={"driver-memory": "4g", "executor-memoryOverhead": "4g"},
    )
