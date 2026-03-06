from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from worldcereal.openeo.parameters import DEFAULT_SEASONAL_MODEL_URL


class WorldCerealProductType(Enum):
    """Enum to define the different WorldCereal products."""

    CROPLAND = "cropland"
    CROPTYPE = "croptype"
    EMBEDDINGS = "embeddings"


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
                datetime.fromisoformat(self.target_date)
            except ValueError:
                raise ValidationError("target_date must be in ISO format (YYYY-MM-DD)")

        return self


class BaseParameters(BaseModel):
    """Base class for shared parameter logic."""

    @staticmethod
    def create_feature_parameters(**kwargs):
        defaults = {
            "rescale_s1": False,
            "presto_model_url": "",
            "compile_presto": False,
            "temporal_prediction": False,
            "target_date": None,
        }
        defaults.update(kwargs)
        return FeaturesParameters(**defaults)


class EmbeddingsParameters(BaseParameters):
    """Parameters for the embeddings product inference pipeline. Types are
    enforced by Pydantic.

    Attributes
    ----------
    feature_parameters : FeaturesParameters
        Parameters for the feature extraction UDF. Will be serialized into a
        dictionary and passed in the process graph.
    """

    @staticmethod
    def _default_feature_parameters() -> FeaturesParameters:
        """Internal helper returning the default feature parameters instance.

        Centralizes the defaults so they are declared only once.
        """
        return BaseParameters.create_feature_parameters(
            rescale_s1=False,
            presto_model_url=DEFAULT_SEASONAL_MODEL_URL,
            compile_presto=False,
            temporal_prediction=False,
            target_date=None,
        )

    feature_parameters: FeaturesParameters = Field(
        # Wrap staticmethod call so pydantic receives a true zero-arg callable
        default_factory=lambda: EmbeddingsParameters._default_feature_parameters()
    )

    def __init__(self, presto_model_url: Optional[str] = None, **kwargs):
        """Allow initialization with a custom Presto model URL without
        duplicating the default argument list.

        Users may still pass an explicit `feature_parameters` to override all
        aspects; in that case `presto_model_url` is ignored.
        """
        if "feature_parameters" not in kwargs and presto_model_url is not None:
            fp = self._default_feature_parameters().model_copy()
            fp.presto_model_url = presto_model_url  # type: ignore[attr-defined]
            kwargs["feature_parameters"] = fp
        super().__init__(**kwargs)
