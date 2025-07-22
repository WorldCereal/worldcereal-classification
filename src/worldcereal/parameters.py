from enum import Enum
from typing import Optional

# from typing import Type
# from openeo_gfmap.features.feature_extractor import PatchFeatureExtractor
# from openeo_gfmap.inference.model_inference import ModelInference
from pydantic import BaseModel, Field, ValidationError, model_validator

# from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
# from worldcereal.openeo.postprocess import PostProcessor


class WorldCerealProductType(Enum):
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
    temporal_prediction : bool (default=False)
        Whether to use temporal-explicit predictions. If True, the time dimension
        is preserved in Presto features and a specific timestep is selected later.
        If False, features are pooled across time (non-temporal prediction).
    target_date : str (default=None)
        Target date for temporal-explicit predictions in ISO format (YYYY-MM-DD).
        Only used when temporal_prediction=True. If None, the middle timestep is used.
    """

    rescale_s1: bool
    presto_model_url: str
    compile_presto: bool
    temporal_prediction: bool = Field(default=False)
    target_date: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def check_temporal_parameters(self):
        """Validates temporal prediction parameters."""
        if self.target_date is not None and not self.temporal_prediction:
            raise ValidationError(
                "target_date can only be specified when temporal_prediction=True"
            )

        if self.target_date is not None:
            try:
                from datetime import datetime

                datetime.fromisoformat(self.target_date)
            except ValueError:
                raise ValidationError("target_date must be in ISO format (YYYY-MM-DD)")

        return self


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
    classifier : CropClassifier
        Classifier to use for the inference. This class must be a subclass of
        GFMAP's `ModelInference` and returns predictions/probabilities for
        cropland.
    classifier_parameters : ClassifierParameters
        Parameters for the classifier UDF. Will be serialized into a dictionnary
        and passed in the process graph.
    """

    # feature_extractor: Type[PatchFeatureExtractor] = Field(
    #     default=PrestoFeatureExtractor
    # )
    feature_parameters: FeaturesParameters = FeaturesParameters(
        rescale_s1=False,
        presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-prometheo-testset2023-month-CROPTYPE_INSEASON-augment%3DTrue-balance%3DTrue-timeexplicit%3DFalse-random-masked-from-5-run%3D202505201027_encoder.pt",  # NOQA
        compile_presto=False,
        temporal_prediction=False,
    )
    # classifier: Type[ModelInference] = Field(default=CropClassifier)
    classifier_parameters: ClassifierParameters = ClassifierParameters(
        classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/downstream/PrestoDownstreamCatBoost_cropland_v006-ft-cropland-maxmaskratio05.onnx"  # NOQA
    )

    # @model_validator(mode="after")
    # def check_udf_types(self):
    #     """Validates the FeatureExtractor and Classifier classes."""
    #     if not issubclass(self.feature_extractor, PatchFeatureExtractor):
    #         raise ValidationError(
    #             f"Feature extractor must be a subclass of PatchFeatureExtractor, got {self.feature_extractor}"
    #         )
    # if not issubclass(self.classifier, ModelInference):
    #     raise ValidationError(
    #         f"Classifier must be a subclass of ModelInference, got {self.classifier}"
    #     )


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
    classifier : CropClassifier
        Classifier to use for the inference. This class must be a subclass of
        GFMAP's `ModelInference` and returns predictions/probabilities for
        cropland classes.
    classifier_parameters : ClassifierParameters
        Parameters for the classifier UDF. Will be serialized into a dictionnary
        and passed in the process graph.
    mask_cropland : bool (default=True)
        Whether or not to mask the cropland pixels before running crop type inference.
    save_mask : bool (default=False)
        Whether or not to save the cropland mask as an intermediate result.
    """

    # feature_extractor: Type[PatchFeatureExtractor] = Field(
    #     default=PrestoFeatureExtractor
    # )
    feature_parameters: FeaturesParameters = FeaturesParameters(
        rescale_s1=False,
        presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-prometheo-testset2023-month-CROPTYPE_INSEASON-augment%3DTrue-balance%3DTrue-timeexplicit%3DFalse-random-masked-from-5-run%3D202505201027_encoder.pt",  # NOQA
        compile_presto=False,
        temporal_prediction=True,
        target_date=None,  # By default take the middle date
    )
    # classifier: Type[ModelInference] = Field(default=CropClassifier)
    classifier_parameters: ClassifierParameters = ClassifierParameters(
        classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/downstream/presto-ss-wc-ft-ct_croptype_CROPTYPE0_30D_random_time-token=month_balance=True_augment=True_CROPTYPE9.onnx"
    )
    mask_cropland: bool = Field(default=True)
    save_mask: bool = Field(default=False)

    # @model_validator(mode="after")
    # def check_udf_types(self):
    #     """Validates the FeatureExtractor and Classifier classes."""
    #     if not issubclass(self.feature_extractor, PatchFeatureExtractor):
    #         raise ValidationError(
    #             f"Feature extractor must be a subclass of PrestoFeatureExtractor, got {self.feature_extractor}"
    #         )
    # if not issubclass(self.classifier, ModelInference):
    #     raise ValidationError(
    #         f"Classifier must be a subclass of ModelInference, got {self.classifier}"
    #     )
    # return self

    @model_validator(mode="after")
    def check_mask_parameters(self):
        """Validates the mask-related parameters."""
        if not self.mask_cropland and self.save_mask:
            raise ValidationError("Cannot save mask if mask_cropland is disabled.")
        return self


class PostprocessParameters(BaseModel):
    """Parameters for postprocessing. Types are enforced by Pydantic.

    Attributes
    ----------
    enable: bool (default=True)
        Whether to enable postprocessing.
    method: str (default="smooth_probabilities")
        The method to use for postprocessing. Must be one of ["smooth_probabilities", "majority_vote"]
    kernel_size: int (default=5)
        Used for majority vote postprocessing. Must be an odd number, larger than 1 and smaller than 25.
    save_intermediate: bool (default=False)
        Whether to save intermediate results (before applying the postprocessing).
        The intermediate results will be saved in the GeoTiff format.
    keep_class_probs: bool (default=False)
        If the per-class probabilities should be outputted in the final product.
    """

    enable: bool = Field(default=True)
    method: str = Field(default="smooth_probabilities")
    kernel_size: int = Field(default=5)
    save_intermediate: bool = Field(default=False)
    keep_class_probs: bool = Field(default=False)

    # postprocessor: Type[ModelInference] = Field(default=PostProcessor)

    # @model_validator(mode="after")
    # def check_udf_types(self):
    #     """Validates the PostProcessor class."""
    #     if not issubclass(self.postprocessor, ModelInference):
    #         raise ValidationError(
    #             f"Postprocessor must be a subclass of PostProcessor, got {self.postprocessor}"
    #         )
    #     return self

    @model_validator(mode="after")
    def check_parameters(self):
        """Validates parameters."""
        if not self.enable and self.save_intermediate:
            raise ValueError(
                "Cannot save intermediate results if postprocessing is disabled."
            )

        if self.method not in ["smooth_probabilities", "majority_vote"]:
            raise ValueError(
                f"Method must be one of ['smooth_probabilities', 'majority_vote'], got {self.method}"
            )

        if self.method == "majority_vote":
            if self.kernel_size % 2 == 0:
                raise ValueError(
                    f"Kernel size for majority filtering should be an odd number, got {self.kernel_size}"
                )
            if self.kernel_size > 25:
                raise ValueError(
                    f"Kernel size for majority filtering should be an odd number smaller than 25, got {self.kernel_size}"
                )
            if self.kernel_size < 3:
                raise ValueError(
                    f"Kernel size for majority filtering should be an odd number larger than 1, got {self.kernel_size}"
                )

        return self
