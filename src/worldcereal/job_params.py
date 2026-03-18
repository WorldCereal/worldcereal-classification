"""Shared parameterization helpers for WorldCereal workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TypedDict, Union

import geopandas as gpd
from openeo_gfmap import Backend, BackendContext, TemporalContext

from worldcereal.job import WorldCerealTask
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.openeo.workflow_config import (
    WorldCerealWorkflowConfig,
    build_config_from_params,
)
from worldcereal.parameters import EmbeddingsParameters, WorldCerealProductType


class WorldCerealJobParams(TypedDict, total=False):
    """Common parameters for WorldCereal job workflows.

    Keys
    ----
    aoi_gdf : gpd.GeoDataFrame
        AOI geometries to process. Must have a CRS.
    output_dir : Path or str
        Output directory used for job tracking and results.
    grid_size : Optional[int]
        Tile size in kilometers. Defaults to 20.
        If None, AOI geometries are used as-is.
    temporal_extent : Optional[TemporalContext]
        Temporal window for all tiles. If None, per-tile dates or crop calendars
        are used when supported by the workflow.
    year : Optional[int]
        Year used to infer crop-calendar windows when temporal_extent is missing.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]]
        Force Sentinel-1 orbit state. If None, the backend selects the best option.
    target_epsg : Optional[int]
        Output EPSG code. If None, local UTM is used.
    compositing_window : Literal["month", "dekad"]
        Temporal compositing window for inputs. Defaults to "month".
    parallel_jobs : int
        Maximum number of concurrent jobs to submit. Defaults to 2.
    randomize_jobs : bool
        Shuffle job order before submission. Defaults to True.
    restart_failed : bool
        Restart failed jobs when reusing a job database. Defaults to False.
    job_options : Optional[Dict[str, Union[str, int, None]]]
        Backend-specific job options.
    poll_sleep : int
        Seconds to wait between status updates. Defaults to 60.
    simplify_logging : bool
        Reduce verbose OpenEO logging. When enabled and no runner is supplied,
        a compact CLI status callback is used. Defaults to True.
    max_retries : int
        Maximum number of job submission retries.
    base_delay : float
        Initial retry delay in seconds.
    max_delay : float
        Maximum retry delay in seconds.
    backend_context : BackendContext
        Backend configuration (CDSE by default).

    Notes
    -----
    Defaults are applied by worldcereal.jobmanager functions when keys are omitted.
    """

    aoi_gdf: gpd.GeoDataFrame
    output_dir: Union[Path, str]
    grid_size: Optional[int]
    temporal_extent: Optional[TemporalContext]
    year: Optional[int]
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]]
    target_epsg: Optional[int]
    compositing_window: Literal["month", "dekad"]
    parallel_jobs: int
    randomize_jobs: bool
    restart_failed: bool
    job_options: Optional[Dict[str, Union[str, int, None]]]
    poll_sleep: int
    simplify_logging: bool
    max_retries: int
    base_delay: float
    max_delay: float
    backend_context: BackendContext


class WorldCerealInputsJobParams(WorldCerealJobParams, total=False):
    """Parameters specific to inputs collection workflows."""


class WorldCerealEmbeddingsJobParams(WorldCerealJobParams, total=False):
    """Parameters specific to embeddings workflows.

    Keys
    ----
    embeddings_parameters : Optional[EmbeddingsParameters]
        Embeddings model configuration. If None, the standard Presto model is used.
    scale_uint16 : bool
        Whether embeddings are scaled to uint16 for storage efficiency. Defaults to True.
    """

    embeddings_parameters: Optional[EmbeddingsParameters]
    scale_uint16: bool


class WorldCerealMapProductionParams(WorldCerealJobParams, total=False):
    """Parameters specific to map production (inference) workflows.

    Keys
    ----
    season_specifications : Optional[Dict[str, TemporalContext]]
        Per-season temporal windows used for inference.
    product_type : Union[WorldCerealProductType, Literal["cropland", "croptype"]]
        Product to generate. Defaults to "cropland".
    seasonal_preset : str
        Seasonal workflow preset name.
        Defaults to standard seasonal WorldCereal configuration.
    seasonal_model_zip : Optional[str]
        Optional override for the seasonal model .zip artifact.
    enable_cropland_head : Optional[bool]
        Override cropland head enablement.
    landcover_head_zip : Optional[str]
        Optional override for the cropland/landcover head artifact.
    enable_croptype_head : Optional[bool]
        Override croptype head enablement.
    croptype_head_zip : Optional[str]
        Optional override for the croptype head artifact.
    enforce_cropland_gate : Optional[bool]
        Mask croptype outputs using cropland predictions.
    export_class_probs : Optional[bool]
        Export per-class probabilities in outputs.
    enable_cropland_postprocess : Optional[bool]
        Enable cropland postprocessing.
    cropland_postprocess_method : Optional[Literal["majority_vote", "smooth_probabilities"]]
        Cropland postprocess method.
    cropland_postprocess_kernel_size : Optional[int]
        Cropland postprocess kernel size.
    enable_croptype_postprocess : Optional[bool]
        Enable croptype postprocessing.
    croptype_postprocess_method : Optional[Literal["majority_vote", "smooth_probabilities"]]
        Croptype postprocess method.
    croptype_postprocess_kernel_size : Optional[int]
        Croptype postprocess kernel size.
    export_embeddings : Optional[bool]
        Whether to export embeddings alongside classifications.
    export_ndvi : Optional[bool]
        Whether to export NDVI alongside classifications.
    merge_classification_products : Optional[bool]
        Whether to merge cropland and croptype outputs into a single product when both heads are enabled
    """

    season_specifications: Optional[Dict[str, TemporalContext]]
    product_type: Union[WorldCerealProductType, Literal["cropland", "croptype"]]
    seasonal_preset: str
    seasonal_model_zip: Optional[str]
    enable_cropland_head: Optional[bool]
    landcover_head_zip: Optional[str]
    enable_croptype_head: Optional[bool]
    croptype_head_zip: Optional[str]
    enforce_cropland_gate: Optional[bool]
    export_class_probs: Optional[bool]
    enable_cropland_postprocess: Optional[bool]
    cropland_postprocess_method: Optional[
        Literal["majority_vote", "smooth_probabilities"]
    ]
    cropland_postprocess_kernel_size: Optional[int]
    enable_croptype_postprocess: Optional[bool]
    croptype_postprocess_method: Optional[
        Literal["majority_vote", "smooth_probabilities"]
    ]
    croptype_postprocess_kernel_size: Optional[int]
    export_embeddings: Optional[bool]
    export_ndvi: Optional[bool]
    merge_classification_products: Optional[bool]


DEFAULT_MAX_RETRIES = 50
DEFAULT_BASE_DELAY = 0.1
DEFAULT_MAX_DELAY = 10.0


def _require_param(params: Dict[str, Any], key: str) -> Any:
    value = params.get(key)
    if value is None:
        raise ValueError(f"'{key}' is required in params.")
    return value


def _with_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _temporal_extent_label(temporal_extent: Optional[TemporalContext]) -> str:
    if temporal_extent is None:
        return "None"
    return f"{temporal_extent.start_date} to {temporal_extent.end_date}"


def _normalize_product_type(
    value: Union[WorldCerealProductType, str],
) -> WorldCerealProductType:
    if isinstance(value, WorldCerealProductType):
        return value
    return WorldCerealProductType(value)


def _season_windows_from_specifications(
    season_specifications: Optional[Dict[str, TemporalContext]],
) -> tuple[Optional[list[str]], Optional[Dict[str, tuple[str, str]]]]:
    if not season_specifications:
        return None, None
    season_ids = list(season_specifications.keys())
    season_windows = {
        season_id: (
            str(season_window.start_date),
            str(season_window.end_date),
        )
        for season_id, season_window in season_specifications.items()
    }
    return season_ids, season_windows


def build_workflow_config_from_params(
    resolved: Dict[str, Any],
) -> WorldCerealWorkflowConfig:
    """Build a workflow config using resolved map production parameters."""
    product_type = _normalize_product_type(resolved["product_type"])
    enable_cropland_head = resolved.get("enable_cropland_head")
    enable_croptype_head = resolved.get("enable_croptype_head")
    enforce_cropland_gate = resolved.get("enforce_cropland_gate")

    if product_type.value == "cropland":
        enable_cropland_head = True
        enable_croptype_head = False
        enforce_cropland_gate = False
    else:
        enable_croptype_head = True

    season_ids, season_windows = _season_windows_from_specifications(
        resolved.get("season_specifications")
    )

    return build_config_from_params(
        enable_cropland_head=enable_cropland_head,
        enable_croptype_head=enable_croptype_head,
        enforce_cropland_gate=enforce_cropland_gate,
        croptype_head_zip=resolved.get("croptype_head_zip"),
        landcover_head_zip=resolved.get("landcover_head_zip"),
        seasonal_model_zip=resolved.get("seasonal_model_zip"),
        export_class_probabilities=resolved.get("export_class_probs"),
        season_ids=season_ids,
        season_windows=season_windows,
        enable_cropland_postprocess=resolved.get("enable_cropland_postprocess"),
        cropland_postprocess_method=resolved.get("cropland_postprocess_method"),
        cropland_postprocess_kernel_size=resolved.get(
            "cropland_postprocess_kernel_size"
        ),
        enable_croptype_postprocess=resolved.get("enable_croptype_postprocess"),
        croptype_postprocess_method=resolved.get("croptype_postprocess_method"),
        croptype_postprocess_kernel_size=resolved.get(
            "croptype_postprocess_kernel_size"
        ),
        export_embeddings=resolved.get("export_embeddings"),
        export_ndvi=resolved.get("export_ndvi"),
        merge_classification_products=resolved.get("merge_classification_products"),
    )


def resolve_job_params(
    task: WorldCerealTask,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply defaults and validate required keys for the selected workflow."""
    aoi_gdf = _require_param(params, "aoi_gdf")
    output_dir = _require_param(params, "output_dir")

    grid_size_default = 20
    randomize_default = True
    restart_default = False
    simplify_logging_default = True
    compositing_window_default = "month"
    parallel_jobs_default = 2
    poll_sleep_default = 60
    backend_context_default = BackendContext(Backend.CDSE)

    resolved: Dict[str, Any] = {
        "aoi_gdf": aoi_gdf,
        "output_dir": output_dir,
        "grid_size": _with_default(params.get("grid_size"), grid_size_default),
        "temporal_extent": params.get("temporal_extent"),
        "year": params.get("year"),
        "s1_orbit_state": params.get("s1_orbit_state"),
        "target_epsg": params.get("target_epsg"),
        "compositing_window": _with_default(
            params.get("compositing_window"), compositing_window_default
        ),
        "parallel_jobs": _with_default(
            params.get("parallel_jobs"), parallel_jobs_default
        ),
        "randomize_jobs": _with_default(
            params.get("randomize_jobs"), randomize_default
        ),
        "restart_failed": _with_default(params.get("restart_failed"), restart_default),
        "job_options": params.get("job_options"),
        "poll_sleep": _with_default(params.get("poll_sleep"), poll_sleep_default),
        "simplify_logging": _with_default(
            params.get("simplify_logging"), simplify_logging_default
        ),
        "max_retries": _with_default(params.get("max_retries"), DEFAULT_MAX_RETRIES),
        "base_delay": _with_default(params.get("base_delay"), DEFAULT_BASE_DELAY),
        "max_delay": _with_default(params.get("max_delay"), DEFAULT_MAX_DELAY),
        "backend_context": _with_default(
            params.get("backend_context"), backend_context_default
        ),
    }

    if task == WorldCerealTask.INPUTS:
        return resolved

    if task == WorldCerealTask.EMBEDDINGS:

        scale_uint16_default = True

        resolved["embeddings_parameters"] = params.get("embeddings_parameters")
        resolved["scale_uint16"] = _with_default(
            params.get("scale_uint16"), scale_uint16_default
        )
        return resolved

    if task == WorldCerealTask.CLASSIFICATION:

        product_type_default = "cropland"
        seasonal_preset_default = DEFAULT_SEASONAL_WORKFLOW_PRESET

        resolved["season_specifications"] = params.get("season_specifications")
        resolved["product_type"] = _with_default(
            params.get("product_type"), product_type_default
        )
        resolved["seasonal_preset"] = _with_default(
            params.get("seasonal_preset"), seasonal_preset_default
        )
        resolved["seasonal_model_zip"] = params.get("seasonal_model_zip")
        resolved["enable_cropland_head"] = params.get("enable_cropland_head")
        resolved["landcover_head_zip"] = params.get("landcover_head_zip")
        resolved["enable_croptype_head"] = params.get("enable_croptype_head")
        resolved["croptype_head_zip"] = params.get("croptype_head_zip")
        resolved["enforce_cropland_gate"] = params.get("enforce_cropland_gate")
        resolved["export_class_probs"] = params.get("export_class_probs")
        resolved["enable_cropland_postprocess"] = params.get(
            "enable_cropland_postprocess"
        )
        resolved["cropland_postprocess_method"] = params.get(
            "cropland_postprocess_method"
        )
        resolved["cropland_postprocess_kernel_size"] = params.get(
            "cropland_postprocess_kernel_size"
        )
        resolved["enable_croptype_postprocess"] = params.get(
            "enable_croptype_postprocess"
        )
        resolved["croptype_postprocess_method"] = params.get(
            "croptype_postprocess_method"
        )
        resolved["croptype_postprocess_kernel_size"] = params.get(
            "croptype_postprocess_kernel_size"
        )
        resolved["export_embeddings"] = params.get("export_embeddings")
        resolved["export_ndvi"] = params.get("export_ndvi")
        resolved["merge_classification_products"] = params.get(
            "merge_classification_products"
        )
        return resolved

    raise ValueError(f"Unsupported task: {task}")


def split_job_params(
    task: WorldCerealTask,
    resolved: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Split resolved params into manager init, run kwargs, and log context."""
    manager_init: Dict[str, Any] = {
        "output_dir": resolved["output_dir"],
        "backend_context": resolved["backend_context"],
        "aoi_gdf": resolved["aoi_gdf"],
        "grid_size": resolved["grid_size"],
        "temporal_extent": resolved["temporal_extent"],
        "year": resolved["year"],
        "poll_sleep": resolved["poll_sleep"],
    }

    job_kwargs: Dict[str, Any] = {
        "restart_failed": resolved["restart_failed"],
        "randomize_jobs": resolved["randomize_jobs"],
        "parallel_jobs": resolved["parallel_jobs"],
        "s1_orbit_state": resolved["s1_orbit_state"],
        "target_epsg": resolved["target_epsg"],
        "compositing_window": resolved["compositing_window"],
        "job_options": resolved["job_options"],
        "max_retries": resolved["max_retries"],
        "base_delay": resolved["base_delay"],
        "max_delay": resolved["max_delay"],
    }

    if task == WorldCerealTask.INPUTS:
        # No task-specific parameters to add for inputs task
        pass

    elif task == WorldCerealTask.EMBEDDINGS:
        resolved_parameters = (
            resolved["embeddings_parameters"] or EmbeddingsParameters()
        )
        job_kwargs["embeddings_parameters"] = resolved_parameters
        job_kwargs["scale_uint16"] = resolved["scale_uint16"]

    elif task == WorldCerealTask.CLASSIFICATION:
        product_type = _normalize_product_type(resolved["product_type"])
        manager_init["season_specifications"] = resolved["season_specifications"]
        manager_init["product_type"] = product_type
        job_kwargs["product_type"] = product_type
        job_kwargs["seasonal_preset"] = resolved["seasonal_preset"]

    else:
        raise ValueError(f"Unsupported task: {task}")

    # Create a log context with all parameters for easy reference in logs and callbacks
    log_context = manager_init.copy()
    log_context.update(job_kwargs)
    # specific modifications for more readable logs
    log_context["output_folder"] = str(log_context["output_dir"])
    log_context["temporal_extent"] = _temporal_extent_label(
        log_context["temporal_extent"]
    )
    log_context["number of AOIs"] = len(log_context["aoi_gdf"])
    del log_context["aoi_gdf"]
    log_context["backend"] = log_context["backend_context"].backend.value
    del log_context["backend_context"]

    return manager_init, job_kwargs, log_context


def build_job_params_from_args(
    task: WorldCerealTask,
    args: argparse.Namespace,
    *,
    aoi_gdf: gpd.GeoDataFrame,
    temporal_extent: Optional[TemporalContext],
    job_options: Optional[Dict[str, Union[str, int, None]]],
    backend_context: Optional[BackendContext] = None,
    embeddings_parameters: Optional[EmbeddingsParameters] = None,
    scale_uint16: Optional[bool] = None,
    season_specifications: Optional[Dict[str, TemporalContext]] = None,
    product_type: Optional[WorldCerealProductType] = None,
    enable_cropland_head: Optional[bool] = None,
    enable_croptype_head: Optional[bool] = None,
    enforce_cropland_gate: Optional[bool] = None,
) -> WorldCerealJobParams:
    """Build workflow params from CLI args for the selected job type."""
    base_params: WorldCerealJobParams = {
        "aoi_gdf": aoi_gdf,
        "output_dir": args.output_folder,
        "grid_size": args.grid_size,
        "temporal_extent": temporal_extent,
        "year": args.year,
        "s1_orbit_state": args.s1_orbit_state,
        "target_epsg": args.target_epsg,
        "compositing_window": args.compositing_window,
        "parallel_jobs": args.parallel_jobs,
        "randomize_jobs": args.randomize_jobs,
        "restart_failed": args.restart_failed,
        "job_options": job_options,
        "poll_sleep": args.poll_sleep,
        "simplify_logging": args.simplify_logging,
        "max_retries": args.max_retries,
        "base_delay": args.base_delay,
        "max_delay": args.max_delay,
        "backend_context": backend_context,
    }

    if task == WorldCerealTask.INPUTS:
        return base_params

    if task == WorldCerealTask.EMBEDDINGS:
        if scale_uint16 is None:
            raise ValueError("scale_uint16 is required for embeddings params.")
        embeddings_params: WorldCerealEmbeddingsJobParams = {
            **base_params,
            "embeddings_parameters": embeddings_parameters,
            "scale_uint16": scale_uint16,
        }
        return embeddings_params

    if task == WorldCerealTask.CLASSIFICATION:
        if product_type is None:
            raise ValueError("product_type is required for map production params.")
        production_params: WorldCerealMapProductionParams = {
            **base_params,
            "season_specifications": season_specifications,
            "product_type": product_type,
            "seasonal_preset": args.seasonal_preset,
            "export_class_probs": args.class_probabilities,
            "seasonal_model_zip": args.seasonal_model_zip,
            "enable_cropland_head": enable_cropland_head,
            "landcover_head_zip": args.landcover_head_zip,
            "enable_croptype_head": enable_croptype_head,
            "croptype_head_zip": args.croptype_head_zip,
            "enforce_cropland_gate": enforce_cropland_gate,
            "enable_cropland_postprocess": args.enable_cropland_postprocess,
            "cropland_postprocess_method": args.cropland_postprocess_method,
            "cropland_postprocess_kernel_size": args.cropland_postprocess_kernel_size,
            "enable_croptype_postprocess": args.enable_croptype_postprocess,
            "croptype_postprocess_method": args.croptype_postprocess_method,
            "croptype_postprocess_kernel_size": args.croptype_postprocess_kernel_size,
            "export_embeddings": args.export_embeddings,
            "export_ndvi": args.export_ndvi,
            "merge_classification_products": args.merge_classification_products,
        }
        return production_params

    raise ValueError(f"Unsupported task: {task}")


__all__ = [
    "WorldCerealJobParams",
    "WorldCerealInputsJobParams",
    "WorldCerealEmbeddingsJobParams",
    "WorldCerealMapProductionParams",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BASE_DELAY",
    "DEFAULT_MAX_DELAY",
    "build_job_params_from_args",
    "build_workflow_config_from_params",
    "resolve_job_params",
    "split_job_params",
]
