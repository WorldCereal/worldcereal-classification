"""Central repository for openEO seasonal workflow parameter presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

DEFAULT_SEASONAL_MODEL_URL = (
    "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/worldcereal/"
    "presto-prometheo-dualtask-SeasonalMultiTaskLoss-month-augment=True-"
    "balance=True-timeexplicit=True-masking=enabled-run=202601240103.zip"
)

DEFAULT_SEASONAL_WORKFLOW_PRESET = "phase_ii_multitask"

DEFAULT_POSTPROCESS_SECTION: Dict[str, Dict[str, Any]] = {
    "cropland": {
        "enabled": False,
        "method": "majority_vote",
        "kernel_size": 5,
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
        },
        "runtime": {
            "cache_root": None,
            "device": "cpu",
            "batch_size": 1024,
        },
        "season": {
            "enforce_cropland_gate": True,
            # Default to GLOBAL_SEASON_IDS at runtime when None.
            "season_ids": None,
            "season_windows": None,
            "season_masks": None,
            "composite_frequency": "month",
        },
        "postprocess": deepcopy(DEFAULT_POSTPROCESS_SECTION),
    }
}
