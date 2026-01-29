"""Central repository for openEO seasonal workflow parameter presets."""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_SEASONAL_MODEL_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/"
    "PhaseII/MultiTask/presto-prometheo-dualtask-SeasonalMultiTaskLoss-month-augment%3DTrue-"
    "balance%3DTrue-timeexplicit%3DTrue-masking%3Denabled-run%3D202601240103.zip"
)

DEFAULT_SEASONAL_WORKFLOW_PRESET = "phase_ii_multitask"

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
    }
}
