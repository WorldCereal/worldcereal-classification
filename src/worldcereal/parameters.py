from enum import Enum
from typing import Type

from openeo_gfmap.features.feature_extractor import PatchFeatureExtractor
from openeo_gfmap.inference.model_inference import ModelInference
from pydantic import BaseModel, Field, ValidationError, model_validator

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier, CroptypeClassifier
from worldcereal.openeo.postprocess import PostProcessor


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
    feature_parameters : FeaturesParameters
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
    feature_parameters: FeaturesParameters = FeaturesParameters(
        rescale_s1=False,
        presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct_100epochs_30D_random_CROPLAND2_time-token=none_balance=True_augment=True_2017=True.pt",  # NOQA
        use_valid_date_token=False,
        compile_presto=False,
    )
    classifier: Type[ModelInference] = Field(default=CroplandClassifier)
    classifier_parameters: ClassifierParameters = ClassifierParameters(
        classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/downstream/PrestoDownstreamCatBoost_cropland_v001.onnx"  # NOQA
    )

    @model_validator(mode="after")
    def check_udf_types(self):
        """Validates the FeatureExtractor and Classifier classes."""
        if not issubclass(self.feature_extractor, PatchFeatureExtractor):
            raise ValidationError(
                f"Feature extractor must be a subclass of PatchFeatureExtractor, got {self.feature_extractor}"
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
    feature_parameters : FeaturesParameters
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
        use_valid_date_token=True,
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


class PostprocessParameters(BaseModel):
    """Parameters for postprocessing. Types are enforced by Pydantic.

    Attributes
    ----------
    enable: bool (default=True)
        Whether to enable postprocessing.
    save_intermediate: bool (default=False)
        Whether to save intermediate results (before the postprocessing).
        The intermediate results will be saved in the GeoTiff format.
    keep_class_probs: bool (default=False)
        If the per-class probabilities should be outputted in the final product.
    """

    enable: bool = Field(default=True)
    save_intermediate: bool = Field(default=False)
    keep_class_probs: bool = Field(default=False)

    postprocessor: Type[ModelInference] = Field(default=PostProcessor)

    @model_validator(mode="after")
    def check_udf_types(self):
        """Validates the PostProcessor class."""
        if not issubclass(self.postprocessor, ModelInference):
            raise ValidationError(
                f"Postprocessor must be a subclass of PostProcessor, got {self.postprocessor}"
            )
        return self

    @model_validator(mode="after")
    def check_parameters(self):
        """Validates boolean parameters."""
        if not self.enable and self.save_intermediate:
            raise ValidationError(
                "Cannot save intermediate results if postprocessing is disabled."
            )
        return self
