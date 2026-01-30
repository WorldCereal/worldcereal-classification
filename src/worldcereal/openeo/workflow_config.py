"""Typed helpers for building seasonal workflow configuration blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

SeasonWindow = Tuple[str, str]
SeasonWindowMapping = Mapping[str, SeasonWindow]
PostprocessMapping = Dict[str, Dict[str, Any]]


@dataclass
class ModelSection:
    """Overrides for the seasonal workflow model section."""

    seasonal_model_zip: Optional[str] = None
    landcover_head_zip: Optional[str] = None
    croptype_head_zip: Optional[str] = None
    enable_croptype_head: Optional[bool] = None
    enable_cropland_head: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.seasonal_model_zip is not None:
            data["seasonal_model_zip"] = self.seasonal_model_zip
        if self.landcover_head_zip is not None:
            data["landcover_head_zip"] = self.landcover_head_zip
        if self.croptype_head_zip is not None:
            data["croptype_head_zip"] = self.croptype_head_zip
        if self.enable_croptype_head is not None:
            data["enable_croptype_head"] = self.enable_croptype_head
        if self.enable_cropland_head is not None:
            data["enable_cropland_head"] = self.enable_cropland_head
        return data


@dataclass
class RuntimeSection:
    """Overrides for runtime-specific knobs (device, batch size, cache)."""

    cache_root: Optional[Union[str, Path]] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.cache_root is not None:
            data["cache_root"] = self.cache_root
        if self.batch_size is not None:
            data["batch_size"] = self.batch_size
        if self.device is not None:
            data["device"] = self.device
        return data


@dataclass
class SeasonSection:
    """Season section knobs (windows, gating, shared probabilities)."""

    season_ids: Optional[Sequence[str]] = None
    season_windows: Optional[SeasonWindowMapping] = None
    season_masks: Optional[Sequence[Any]] = None
    composite_frequency: Optional[str] = None
    enforce_cropland_gate: Optional[bool] = None
    export_class_probabilities: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.season_ids is not None:
            data["season_ids"] = list(self.season_ids)
        if self.season_windows is not None:
            data["season_windows"] = dict(self.season_windows)
        if self.season_masks is not None:
            data["season_masks"] = self.season_masks
        if self.composite_frequency is not None:
            data["composite_frequency"] = self.composite_frequency
        if self.enforce_cropland_gate is not None:
            data["enforce_cropland_gate"] = self.enforce_cropland_gate
        if self.export_class_probabilities is not None:
            data["export_class_probabilities"] = self.export_class_probabilities
        return data


def _sanitize_postprocess_options(options: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in options.items():
        if value is None:
            continue
        if key == "enabled":
            cleaned[key] = bool(value)
        elif key == "kernel_size":
            cleaned[key] = int(value)
        else:
            cleaned[key] = value
    return cleaned


@dataclass
class WorldCerealWorkflowConfig:
    """Container that mirrors the seasonal_workflow sections.

    The backend JSON exposes a "season" section consumed by the UDF.
    """

    model: Optional[ModelSection] = None
    runtime: Optional[RuntimeSection] = None
    season: Optional[SeasonSection] = None
    postprocess: Optional[PostprocessMapping] = None

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        sections: Dict[str, Dict[str, Any]] = {}
        if self.model:
            model_dict = self.model.to_dict()
            if model_dict:
                sections["model"] = model_dict
        if self.runtime:
            runtime_dict = self.runtime.to_dict()
            if runtime_dict:
                sections["runtime"] = runtime_dict
        if self.season:
            season_dict = self.season.to_dict()
            if season_dict:
                sections["season"] = season_dict
        if self.postprocess:
            postprocess_dict = {
                product: dict(opts)
                for product, opts in self.postprocess.items()
                if opts
            }
            if postprocess_dict:
                sections["postprocess"] = postprocess_dict
        return sections

    @classmethod
    def builder(cls) -> "WorldCerealWorkflowConfigBuilder":
        return WorldCerealWorkflowConfigBuilder()


@dataclass
class WorldCerealWorkflowConfigBuilder:
    """Fluent helper for composing workflow overrides."""

    model: ModelSection = field(default_factory=ModelSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    season: SeasonSection = field(default_factory=SeasonSection)
    postprocess: PostprocessMapping = field(default_factory=dict)

    def _ensure_copy(self) -> None:
        # dataclass fields are already separate instances via default_factory
        return

    def enable_croptype_head(
        self, enabled: bool = True
    ) -> "WorldCerealWorkflowConfigBuilder":
        self.model.enable_croptype_head = bool(enabled)
        return self

    def disable_croptype_head(self) -> "WorldCerealWorkflowConfigBuilder":
        return self.enable_croptype_head(False)

    def enable_cropland_head(
        self, enabled: bool = True
    ) -> "WorldCerealWorkflowConfigBuilder":
        self.model.enable_cropland_head = bool(enabled)
        return self

    def disable_cropland_head(self) -> "WorldCerealWorkflowConfigBuilder":
        return self.enable_cropland_head(False)

    def export_class_probabilities(
        self, enabled: bool = True
    ) -> "WorldCerealWorkflowConfigBuilder":
        self.season.export_class_probabilities = bool(enabled)
        return self

    def enforce_cropland_gate(
        self, enabled: bool = True
    ) -> "WorldCerealWorkflowConfigBuilder":
        self.season.enforce_cropland_gate = bool(enabled)
        return self

    def season_ids(self, ids: Sequence[str]) -> "WorldCerealWorkflowConfigBuilder":
        self.season.season_ids = list(ids)
        return self

    def season_windows(
        self, windows: SeasonWindowMapping
    ) -> "WorldCerealWorkflowConfigBuilder":
        self.season.season_windows = dict(windows)
        return self

    def season_masks(self, masks: Sequence[Any]) -> "WorldCerealWorkflowConfigBuilder":
        self.season.season_masks = masks
        return self

    def composite_frequency(
        self, frequency: Optional[str]
    ) -> "WorldCerealWorkflowConfigBuilder":
        self.season.composite_frequency = frequency
        return self

    def cache_root(
        self, cache_root: Union[str, Path]
    ) -> "WorldCerealWorkflowConfigBuilder":
        self.runtime.cache_root = cache_root
        return self

    def batch_size(self, batch_size: int) -> "WorldCerealWorkflowConfigBuilder":
        self.runtime.batch_size = batch_size
        return self

    def device(self, device: str) -> "WorldCerealWorkflowConfigBuilder":
        self.runtime.device = device
        return self

    def landcover_head_zip(self, path: str) -> "WorldCerealWorkflowConfigBuilder":
        self.model.landcover_head_zip = path
        return self

    def croptype_head_zip(self, path: str) -> "WorldCerealWorkflowConfigBuilder":
        self.model.croptype_head_zip = path
        return self

    def seasonal_model_zip(self, path: str) -> "WorldCerealWorkflowConfigBuilder":
        self.model.seasonal_model_zip = path
        return self

    def cropland_postprocess(
        self,
        *,
        enabled: Optional[bool] = None,
        method: Optional[str] = None,
        kernel_size: Optional[int] = None,
    ) -> "WorldCerealWorkflowConfigBuilder":
        return self._update_postprocess(
            "cropland",
            enabled=enabled,
            method=method,
            kernel_size=kernel_size,
        )

    def croptype_postprocess(
        self,
        *,
        enabled: Optional[bool] = None,
        method: Optional[str] = None,
        kernel_size: Optional[int] = None,
    ) -> "WorldCerealWorkflowConfigBuilder":
        return self._update_postprocess(
            "croptype",
            enabled=enabled,
            method=method,
            kernel_size=kernel_size,
        )

    def _update_postprocess(
        self, product: str, **options: Optional[Any]
    ) -> "WorldCerealWorkflowConfigBuilder":
        current = dict(self.postprocess.get(product, {}))
        sanitized = _sanitize_postprocess_options(options)
        current.update(sanitized)
        if current:
            self.postprocess[product] = current
        elif product in self.postprocess:
            del self.postprocess[product]
        return self

    def build(self) -> WorldCerealWorkflowConfig:
        model = self.model if self.model.to_dict() else None
        runtime = self.runtime if self.runtime.to_dict() else None
        season = self.season if self.season.to_dict() else None
        postprocess = {
            product: dict(opts) for product, opts in self.postprocess.items() if opts
        }
        postprocess = postprocess or None
        return WorldCerealWorkflowConfig(
            model=model, runtime=runtime, season=season, postprocess=postprocess
        )


__all__ = [
    "ModelSection",
    "RuntimeSection",
    "SeasonSection",
    "WorldCerealWorkflowConfig",
    "WorldCerealWorkflowConfigBuilder",
]
