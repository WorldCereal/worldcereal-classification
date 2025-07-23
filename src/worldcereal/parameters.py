from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

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


class BaseParameters(BaseModel):
    """Base class for shared parameter logic."""

    @staticmethod
    def create_feature_parameters(
        rescale_s1: bool,
        presto_model_url: str,
        compile_presto: bool,
        temporal_prediction: bool,
        target_date: Optional[str],
    ):
        return FeaturesParameters(
            rescale_s1=rescale_s1,
            presto_model_url=presto_model_url,
            compile_presto=compile_presto,
            temporal_prediction=temporal_prediction,
            target_date=target_date,
        )

    @staticmethod
    def create_classifier_parameters(classifier_url: str):
        return ClassifierParameters(classifier_url=classifier_url)


class CropLandParameters(BaseParameters):
    """Parameters for the cropland product inference pipeline. Types are
    enforced by Pydantic.

    Attributes
    ----------
    feature_parameters : FeaturesParameters
        Parameters for the feature extraction UDF. Will be serialized into a
        dictionary and passed in the process graph.
    classifier_parameters : ClassifierParameters
        Parameters for the classifier UDF. Will be serialized into a dictionary
        and passed in the process graph.
    """

    feature_parameters: FeaturesParameters = BaseParameters.create_feature_parameters(
        rescale_s1=False,
        presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-prometheo-landcover-month-LANDCOVER10-augment%3DTrue-balance%3DTrue-timeexplicit%3DFalse-run%3D202507170930_encoder.pt",  # NOQA
        compile_presto=False,
        temporal_prediction=False,
        target_date=None,
    )
    classifier_parameters: ClassifierParameters = BaseParameters.create_classifier_parameters(
        classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/downstream/PrestoDownstreamCatBoost_temporary-crops_v001-debug.onnx"  # NOQA
    )


class CropTypeParameters(BaseParameters):
    """Parameters for the croptype product inference pipeline. Types are
    enforced by Pydantic.

    Attributes
    ----------
    feature_parameters : FeaturesParameters
        Parameters for the feature extraction UDF. Will be serialized into a
        dictionary and passed in the process graph.
    classifier_parameters : ClassifierParameters
        Parameters for the classifier UDF. Will be serialized into a dictionary
        and passed in the process graph.
    mask_cropland : bool (default=True)
        Whether or not to mask the cropland pixels before running crop type inference.
    save_mask : bool (default=False)
        Whether or not to save the cropland mask as an intermediate result.
    """

    feature_parameters: FeaturesParameters = Field(
        default_factory=lambda: BaseParameters.create_feature_parameters(
            rescale_s1=False,
            presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-prometheo-landcover-month-CROPTYPE27-augment%3DTrue-balance%3DTrue-timeexplicit%3DTrue-run%3D202507181013_encoder.pt",  # NOQA
            compile_presto=False,
            temporal_prediction=True,
            target_date=None,  # By default take the middle date
        )
    )
    classifier_parameters: ClassifierParameters = Field(
        default_factory=lambda: BaseParameters.create_classifier_parameters(
            classifier_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/downstream/PrestoDownstreamCatBoost_croptype_v001-debug.onnx"
        )
    )
    mask_cropland: bool = Field(default=True)
    save_mask: bool = Field(default=False)

    def __init__(self, target_date: Optional[str] = None, **kwargs):
        feature_params = self.create_feature_parameters(
            rescale_s1=False,
            presto_model_url="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-prometheo-landcover-month-CROPTYPE27-augment%3DTrue-balance%3DTrue-timeexplicit%3DTrue-run%3D202507181013_encoder.pt",  # NOQA
            compile_presto=False,
            temporal_prediction=True,
            target_date=target_date,
        )

        if "feature_parameters" not in kwargs:
            kwargs["feature_parameters"] = feature_params

        super().__init__(**kwargs)

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
