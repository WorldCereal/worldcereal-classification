"""Central repository for openEO seasonal workflow parameter presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

DEFAULT_SEASONAL_MODEL_URL = "https://s3.waw3-1.cloudferro.com/project_dependencies/worldcereal/WorldCerealPresto-NoSpatialGlobalMLP-month-augment=True-balance=TruePerBin-SpatialBin5.0deg-timeexplicit=True-masking=enabled-ema0.2-clamp=0.2-8.0-run=202605192112.zip"


DEFAULT_SEASONAL_WORKFLOW_PRESET = "phase_ii_multitask"

DEFAULT_POSTPROCESS_SECTION: Dict[str, Dict[str, Any]] = {
    "cropland": {
        "enabled": False,
        "method": "majority_vote",
        "kernel_size": 3,
    },
    "croptype": {
        "enabled": False,
        "method": "majority_vote",
        "kernel_size": 5,
    },
}

SeasonalWorkflowPreset = Dict[str, Any]
SeasonalWorkflowPresets = Dict[str, SeasonalWorkflowPreset]

SEASONAL_WORKFLOW_PRESETS: SeasonalWorkflowPresets = {
    DEFAULT_SEASONAL_WORKFLOW_PRESET: {
        "description": "Phase II multitask seasonal backbone with dual landcover/croptype heads",
        "model": {
            "seasonal_model_zip": DEFAULT_SEASONAL_MODEL_URL,
            "landcover_head_zip": None,
            "croptype_head_zip": None,
            "export_embeddings": False,
        },
        "runtime": {
            "cache_root": None,
            "device": "cpu",
            "batch_size": 1024,
        },
        "season": {
            "mask_cropland": True,
            # Default to GLOBAL_SEASON_IDS at runtime when None.
            "season_ids": None,
            "season_windows": None,
            "season_masks": None,
            "composite_frequency": "month",
        },
        "postprocess": deepcopy(DEFAULT_POSTPROCESS_SECTION),
    }
}
