"""Unified job manager for WorldCereal batch processing.

Guide
-----
This module coordinates production grid preparation, job tracking, and
batch submission for three tasks: inputs, embeddings, and inference.

Tasks
~~~~~
* inputs: collect preprocessed inputs for each tile.
* embeddings: compute tile embeddings from the input stack.
* inference: run the end-to-end crop mapping workflow.

Temporal and season handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The job database must include temporal extent columns (start_date/end_date)
and, for inference, at least one season specification (captured by season_ids and season_windows attributes).
The season specification can be provided in three ways,
with the following priority:

1) Use tile specific information as present in the production grid.
2) Else use user-provided temporal_extent and season specification for all tiles.
3) Else infer from default crop calendars (requires year).

Job database reuse
~~~~~~~~~~~~~~~~~~
If a job tracking CSV exists, it is reused and the production grid is loaded
from disk instead of being recomputed.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

import geopandas as gpd
import openeo
import pandas as pd
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import CsvJobDatabase, MultiBackendJobManager
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import BACKEND_CONNECTIONS
from shapely import wkt as shapely_wkt

from worldcereal.job import (
    DEFAULT_INFERENCE_JOB_OPTIONS,
    DEFAULT_INPUTS_JOB_OPTIONS,
    WorldCerealTask,
    create_worldcereal_process_graph,
)
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.openeo.workflow_config import (
    WorldCerealWorkflowConfig,
    build_config_from_params,
)
from worldcereal.parameters import EmbeddingsParameters, WorldCerealProductType
from worldcereal.seasons import get_season_dates_for_extent
from worldcereal.utils.production_grid import create_production_grid, ensure_utm_grid

DEFAULT_MAX_RETRIES = 50
DEFAULT_BASE_DELAY = 0.1
DEFAULT_MAX_DELAY = 10.0


def compute_worldcereal_embeddings(
    *,
    aoi_gdf: gpd.GeoDataFrame,
    output_dir: Union[Path, str],
    grid_size: Optional[int] = None,
    temporal_extent: Optional[TemporalContext] = None,
    year: Optional[int] = None,
    embeddings_parameters: Optional[EmbeddingsParameters] = None,
    scale_uint16: bool = True,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    parallel_jobs: int = 2,
    randomize_jobs: bool = False,
    restart_failed: bool = False,
    job_options: Optional[Dict[str, Union[str, int, None]]] = None,
    poll_sleep: int = 60,
    simplify_logging: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    log_fn: Optional[Callable[[str], None]] = None,
    runner: Optional[Callable[..., Any]] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> "WorldCerealJobManager":
    """Run embeddings jobs with optional injected logging/runner hooks.

    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame
        AOI geometries to process. Must have a CRS.
    output_dir : Union[Path, str]
        Output directory used for job tracking and results.
    grid_size : Optional[int], optional
        Tile size in kilometers. If None, the AOI geometries are used as-is.
    temporal_extent : Optional[TemporalContext], optional
        Common temporal window for all tiles. If None, the workflow will fall back
        to per-tile start/end dates or to crop-calendar inference using `year`.
    year : Optional[int], optional
        Year used to infer crop-calendar windows when temporal_extent is missing.
    embeddings_parameters : Optional[EmbeddingsParameters], optional
        Embeddings model configuration. If None, defaults to the standard Presto model.
    scale_uint16 : bool, optional
        Whether embeddings are scaled to uint16 for storage/transfer efficiency.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        Force Sentinel-1 orbit state. If None, the backend selects the best option.
    parallel_jobs : int, optional
        Maximum number of concurrent jobs to submit.
    randomize_jobs : bool, optional
        Shuffle job order before submission.
    restart_failed : bool, optional
        Restart jobs that previously failed in the job database.
    job_options : Optional[Dict[str, Union[str, int, None]]], optional
        Backend-specific job options (driver/executor settings, image name, etc.).
    poll_sleep : int, optional
        Seconds to wait between status updates.
    simplify_logging : bool, optional
        Reduce verbose OpenEO logging. If `runner` is None, a CLI status callback
        is also attached.
    max_retries : int, optional
        Maximum number of job submission retries.
    base_delay : float, optional
        Initial delay (seconds) between retries.
    max_delay : float, optional
        Maximum delay (seconds) between retries.
    backend_context : BackendContext, optional
        Backend configuration (CDSE by default).
    log_fn : Optional[Callable[[str], None]], optional
        Logger function used for progress output. Defaults to loguru `logger.info`.
    runner : Optional[Callable[..., Any]], optional
        Optional runner hook for notebook workflows. When provided, it is called
        as `runner(manager, run_kwargs=..., **runner_kwargs)`.
    runner_kwargs : Optional[Dict[str, Any]], optional
        Extra arguments forwarded to `runner` (e.g., notebook widgets).

    Returns
    -------
    WorldCerealJobManager
        The job manager used to submit and track embeddings jobs.
    """
    if log_fn is None:
        log_fn = logger.info

    log_fn("------------------------------------")
    log_fn("STARTING WORKFLOW: Embeddings computation")
    log_fn("------------------------------------")
    log_fn("----- Workflow configuration -----")

    resolved_parameters = embeddings_parameters or EmbeddingsParameters()
    if temporal_extent is not None:
        temporal_extent_str = (
            f"{temporal_extent.start_date} to {temporal_extent.end_date}"
        )
    else:
        temporal_extent_str = "None"

    params = {
        "output_folder": str(output_dir),
        "number of AOI features": len(aoi_gdf),
        "grid_size": grid_size,
        "temporal_extent": temporal_extent_str,
        "year": year,
        "embeddings_parameters": resolved_parameters,
        "scale_uint16": scale_uint16,
        "s1_orbit_state": s1_orbit_state,
        "parallel_jobs": parallel_jobs,
        "restart_failed": restart_failed,
        "randomize_jobs": randomize_jobs,
        "job_options": job_options,
        "poll_sleep": poll_sleep,
        "simplify_logging": simplify_logging,
        "max_retries": max_retries,
        "base_delay": base_delay,
        "max_delay": max_delay,
    }
    for key, value in params.items():
        log_fn(f"{key}: {value}")
    log_fn("------------------------------------")

    log_fn("Initializing job manager...")
    manager = WorldCerealJobManager(
        output_dir=output_dir,
        task=WorldCerealTask.EMBEDDINGS,
        backend_context=backend_context,
        aoi_gdf=aoi_gdf,
        grid_size=grid_size,
        temporal_extent=temporal_extent,
        year=year,
        poll_sleep=poll_sleep,
    )
    log_fn("Job manager initialized!")

    status_callback = None
    if simplify_logging and runner is None:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )
        status_callback = WorldCerealJobManager.cli_status_callback(
            title="Embeddings job status"
        )
    elif simplify_logging:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )

    log_fn("Starting job submissions...")
    break_msg = (
        "Stopping embeddings computation...\n"
        "Make sure to manually cancel any running jobs in the backend to avoid unnecessary costs!\n"
        "For this, visit the job tracking page in the backend dashboard: https://openeo.dataspace.copernicus.eu/\n"
    )

    run_kwargs = {
        "restart_failed": restart_failed,
        "randomize_jobs": randomize_jobs,
        "parallel_jobs": parallel_jobs,
        "s1_orbit_state": s1_orbit_state,
        "embeddings_parameters": resolved_parameters,
        "scale_uint16": scale_uint16,
        "job_options": job_options,
        "status_callback": status_callback,
        "max_retries": max_retries,
        "base_delay": base_delay,
        "max_delay": max_delay,
    }

    try:
        if runner is not None:
            runner(
                manager,
                run_kwargs=run_kwargs,
                **(runner_kwargs or {}),
            )
        else:
            manager.run_jobs(**run_kwargs)
    except KeyboardInterrupt:
        log_fn(break_msg)
        manager.stop_job_thread()
        log_fn("Embeddings computation has stopped.")
        raise

    log_fn("All done!")
    log_fn(f"Results stored in {output_dir}")
    return manager


def collect_worldcereal_inputs(
    *,
    aoi_gdf: gpd.GeoDataFrame,
    output_dir: Union[Path, str],
    grid_size: Optional[int] = None,
    temporal_extent: Optional[TemporalContext] = None,
    year: Optional[int] = None,
    compositing_window: Literal["month", "dekad"] = "month",
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    parallel_jobs: int = 2,
    randomize_jobs: bool = False,
    restart_failed: bool = False,
    job_options: Optional[Dict[str, Union[str, int, None]]] = None,
    poll_sleep: int = 60,
    simplify_logging: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    log_fn: Optional[Callable[[str], None]] = None,
    runner: Optional[Callable[..., Any]] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> "WorldCerealJobManager":
    """Run inputs jobs with optional injected logging/runner hooks.

    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame
        AOI geometries to process. Must have a CRS.
    output_dir : Union[Path, str]
        Output directory used for job tracking and results.
    grid_size : Optional[int], optional
        Tile size in kilometers. If None, the AOI geometries are used as-is.
    temporal_extent : Optional[TemporalContext], optional
        Common temporal window for all tiles. If None, the workflow will fall back
        to per-tile start/end dates or to crop-calendar inference using `year`.
    year : Optional[int], optional
        Year used to infer crop-calendar windows when temporal_extent is missing.
    compositing_window : Literal["month", "dekad"], optional
        Temporal compositing window for inputs, by default "month".
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        Force Sentinel-1 orbit state. If None, the backend selects the best option.
    parallel_jobs : int, optional
        Maximum number of concurrent jobs to submit.
    randomize_jobs : bool, optional
        Shuffle job order before submission.
    restart_failed : bool, optional
        Restart jobs that previously failed in the job database.
    job_options : Optional[Dict[str, Union[str, int, None]]], optional
        Backend-specific job options (driver/executor settings, image name, etc.).
    poll_sleep : int, optional
        Seconds to wait between status updates.
    simplify_logging : bool, optional
        Reduce verbose OpenEO logging. If `runner` is None, a CLI status callback
        is also attached.
    max_retries : int, optional
        Maximum number of job submission retries.
    base_delay : float, optional
        Initial delay (seconds) between retries.
    max_delay : float, optional
        Maximum delay (seconds) between retries.
    backend_context : BackendContext, optional
        Backend configuration (CDSE by default).
    log_fn : Optional[Callable[[str], None]], optional
        Logger function used for progress output. Defaults to loguru `logger.info`.
    runner : Optional[Callable[..., Any]], optional
        Optional runner hook for notebook workflows. When provided, it is called
        as `runner(manager, run_kwargs=..., **runner_kwargs)`.
    runner_kwargs : Optional[Dict[str, Any]], optional
        Extra arguments forwarded to `runner` (e.g., notebook widgets).

    Returns
    -------
    WorldCerealJobManager
        The job manager used to submit and track inputs jobs.
    """
    if log_fn is None:
        log_fn = logger.info

    log_fn("------------------------------------")
    log_fn("STARTING WORKFLOW: Inputs collection")
    log_fn("------------------------------------")
    log_fn("----- Workflow configuration -----")

    if temporal_extent is not None:
        temporal_extent_str = (
            f"{temporal_extent.start_date} to {temporal_extent.end_date}"
        )
    else:
        temporal_extent_str = "None"

    params = {
        "output_folder": str(output_dir),
        "number of AOI features": len(aoi_gdf),
        "grid_size": grid_size,
        "temporal_extent": temporal_extent_str,
        "year": year,
        "compositing_window": compositing_window,
        "s1_orbit_state": s1_orbit_state,
        "parallel_jobs": parallel_jobs,
        "restart_failed": restart_failed,
        "randomize_jobs": randomize_jobs,
        "job_options": job_options,
        "poll_sleep": poll_sleep,
        "simplify_logging": simplify_logging,
        "max_retries": max_retries,
        "base_delay": base_delay,
        "max_delay": max_delay,
    }
    for key, value in params.items():
        log_fn(f"{key}: {value}")
    log_fn("------------------------------------")

    log_fn("Initializing job manager...")
    manager = WorldCerealJobManager(
        output_dir=output_dir,
        task=WorldCerealTask.INPUTS,
        backend_context=backend_context,
        aoi_gdf=aoi_gdf,
        grid_size=grid_size,
        temporal_extent=temporal_extent,
        year=year,
        poll_sleep=poll_sleep,
    )
    log_fn("Job manager initialized!")

    status_callback = None
    if simplify_logging and runner is None:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )
        status_callback = WorldCerealJobManager.cli_status_callback(
            title="Inputs job status"
        )
    elif simplify_logging:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )

    log_fn("Starting job submissions...")
    break_msg = (
        "Stopping inputs collection...\n"
        "Make sure to manually cancel any running jobs in the backend to avoid unnecessary costs!\n"
        "For this, visit the job tracking page in the backend dashboard: https://openeo.dataspace.copernicus.eu/\n"
    )

    run_kwargs = {
        "restart_failed": restart_failed,
        "randomize_jobs": randomize_jobs,
        "parallel_jobs": parallel_jobs,
        "s1_orbit_state": s1_orbit_state,
        "job_options": job_options,
        "compositing_window": compositing_window,
        "status_callback": status_callback,
        "max_retries": max_retries,
        "base_delay": base_delay,
        "max_delay": max_delay,
    }

    try:
        if runner is not None:
            runner(
                manager,
                run_kwargs=run_kwargs,
                **(runner_kwargs or {}),
            )
        else:
            manager.run_jobs(**run_kwargs)
    except KeyboardInterrupt:
        log_fn(break_msg)
        manager.stop_job_thread()
        log_fn("Inputs collection has stopped.")
        raise

    log_fn("All done!")
    log_fn(f"Results stored in {output_dir}")
    return manager


def run_map_production(
    *,
    aoi_gdf: gpd.GeoDataFrame,
    output_dir: Union[Path, str],
    grid_size: int = 20,
    temporal_extent: Optional[TemporalContext] = None,
    season_specifications: Optional[Dict[str, TemporalContext]] = None,
    year: Optional[int] = None,
    product_type: Union[WorldCerealProductType, Literal["cropland", "croptype"]] = (
        "cropland"
    ),
    seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
    seasonal_model_zip: Optional[str] = None,
    enable_cropland_head: Optional[bool] = None,
    landcover_head_zip: Optional[str] = None,
    enable_croptype_head: Optional[bool] = None,
    croptype_head_zip: Optional[str] = None,
    enforce_cropland_gate: Optional[bool] = None,
    export_class_probs: Optional[bool] = None,
    enable_cropland_postprocess: Optional[bool] = None,
    cropland_postprocess_method: Optional[
        Literal["majority_vote", "smooth_probabilities"]
    ] = None,
    cropland_postprocess_kernel_size: Optional[int] = None,
    enable_croptype_postprocess: Optional[bool] = None,
    croptype_postprocess_method: Optional[
        Literal["majority_vote", "smooth_probabilities"]
    ] = None,
    croptype_postprocess_kernel_size: Optional[int] = None,
    target_epsg: Optional[int] = None,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    restart_failed: bool = True,
    job_options: Optional[dict] = None,
    parallel_jobs: int = 2,
    randomize_jobs: bool = True,
    poll_sleep: int = 60,
    simplify_logging: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    log_fn: Optional[Callable[[str], None]] = None,
    runner: Optional[Callable[..., Any]] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> "WorldCerealJobManager":
    """Run map production with optional injected logging/runner hooks.

    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame
        Areas of interest to process. Must have a CRS.
    output_dir : Union[Path, str]
        The directory where the output files will be saved.
    grid_size : int, optional
        The resolution of the tiles in kilometers, by default 20.
    temporal_extent : Optional[TemporalContext], optional
        Temporal context defining the time range for which to collect input data patches.
        If provided together with `year`, temporal_extent will take precedence and override the year.
    season_specifications : Optional[Dict[str, TemporalContext]], optional
        Per-season temporal windows used for inference, by default None.
    year : Optional[int], optional
        Year used for crop calendar inference when dates are missing, by default None.
    product_type : Union[WorldCerealProductType, Literal["cropland", "croptype"]], optional
        The type of product to produce, by default "cropland".
    seasonal_preset : str, optional
        Name of the seasonal workflow preset to use when building the inference context.
        This determines the default configuration settings of the inference workflow.
        By default we use the "phase_ii_multitask" preset, which corresponds to the Phase II multitask seasonal backbone with dual landcover/croptype heads.
    seasonal_model_zip : Optional[str], optional
        Path to .zip file of a seasonal Presto model to be used for overriding the default seasonal model.
        By default None, which means the seasonal model embedded in the preset will be used.
    enable_cropland_head : Optional[bool], optional
        Override whether the cropland (landcover) head is enabled. If None, defaults to
        preset behavior unless `product_type` forces cropland-only execution.
    landcover_head_zip : Optional[str], optional
        Path to .zip file of a cropland/landcover head artifact to be used for overriding the default head.
        If None, the head embedded in the preset seasonal model will be used.
    enable_croptype_head : Optional[bool], optional
        Override whether the croptype head is enabled. If None, defaults to preset behavior
        unless `product_type` forces cropland-only execution.
    croptype_head_zip : Optional[str], optional
        Path to .zip file of a croptype head artifact to be used for overriding the default head.
        If None, the head embedded in the preset seasonal model will be used.
    enforce_cropland_gate : Optional[bool], optional
        Whether or not to mask the crop type product with the cropland product.
        If None, use the preset default (True).
    export_class_probs : Optional[bool], optional
        Export per-class probabilities in all products.
        If None, use the preset default (True).
    enable_cropland_postprocess : Optional[bool], optional
        Enable cropland postprocessing. If None, use the preset default (False).
    cropland_postprocess_method : Optional[str], optional
        If None and postprocess is enabled, the preset default is used ("majority_vote").
        Available options are "majority_vote" and "smooth_probabilities".
    cropland_postprocess_kernel_size : Optional[int], optional
        Cropland postprocess kernel size override.
        If None and postprocess is enabled, the preset default is used (5).
    enable_croptype_postprocess : Optional[bool], optional
        Enable croptype postprocessing. If None, use the preset default (False).
    croptype_postprocess_method : Optional[str], optional
        Croptype postprocess method override.
        If None and postprocess is enabled, the preset default is used ("majority_vote").
        Available options are "majority_vote" and "smooth_probabilities".
    croptype_postprocess_kernel_size : Optional[int], optional
        Croptype postprocess kernel size override.
        If None and postprocess is enabled, the preset default is used (5).
    target_epsg : Optional[int], optional
        The target EPSG code for the output, by default None.
    backend_context : BackendContext, optional
        The backend context to use for the production.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        The Sentinel-1 orbit state to use for the production.
    restart_failed : bool, optional
        Whether to automatically restart failed jobs, by default True.
    job_options : Optional[dict], optional
        Additional options for the job, by default None.
    parallel_jobs : int, optional
        The number of parallel jobs to run, by default 2.
    randomize_jobs : bool, optional
        Whether to randomize the order of job execution, by default True.
    poll_sleep : int, optional
        The number of seconds to wait between polling the job status, by default 60.
    simplify_logging : bool, optional
        Whether to simplify logging output, by default True.
    max_retries : int, optional
        Maximum number of job submission retries.
    base_delay : float, optional
        Initial retry delay in seconds.
    max_delay : float, optional
        Maximum retry delay in seconds.
    log_fn : Optional[Callable[[str], None]], optional
        Logger function used for progress output. Defaults to loguru `logger.info`.
    runner : Optional[Callable[..., Any]], optional
        Optional runner hook for notebook workflows. When provided, it is called
        as `runner(manager, run_kwargs=..., **runner_kwargs)`.
    runner_kwargs : Optional[Dict[str, Any]], optional
        Extra arguments forwarded to `runner` (e.g., notebook widgets).

    Returns
    -------
    WorldCerealJobManager
        The job manager used to submit and track production jobs.
    """
    if log_fn is None:
        log_fn = logger.info

    if isinstance(product_type, WorldCerealProductType):
        product_type_value = product_type.value
    else:
        product_type_value = product_type

    if product_type_value == "cropland":
        enable_cropland_head = True
        enable_croptype_head = False
        enforce_cropland_gate = False
    else:
        enable_croptype_head = True

    resolved_product_type = WorldCerealProductType(product_type_value)

    if season_specifications is not None and len(season_specifications) > 0:
        season_ids = list(season_specifications.keys())
        season_windows = {
            season_id: (
                str(season_window.start_date),
                str(season_window.end_date),
            )
            for season_id, season_window in season_specifications.items()
        }
    else:
        season_ids = None
        season_windows = None

    workflow_config = build_config_from_params(
        enable_cropland_head=enable_cropland_head,
        enable_croptype_head=enable_croptype_head,
        enforce_cropland_gate=enforce_cropland_gate,
        croptype_head_zip=croptype_head_zip,
        landcover_head_zip=landcover_head_zip,
        seasonal_model_zip=seasonal_model_zip,
        export_class_probabilities=export_class_probs,
        season_ids=season_ids,
        season_windows=season_windows,
        enable_cropland_postprocess=enable_cropland_postprocess,
        cropland_postprocess_method=cropland_postprocess_method,
        cropland_postprocess_kernel_size=cropland_postprocess_kernel_size,
        enable_croptype_postprocess=enable_croptype_postprocess,
        croptype_postprocess_method=croptype_postprocess_method,
        croptype_postprocess_kernel_size=croptype_postprocess_kernel_size,
    )

    log_fn("------------------------------------")
    log_fn("STARTING WORKFLOW: Map production")
    log_fn("------------------------------------")
    log_fn("----- Workflow configuration -----")

    if temporal_extent is not None:
        temporal_extent_str = (
            f"{temporal_extent.start_date} to {temporal_extent.end_date}"
        )
    else:
        temporal_extent_str = "None"

    params = {
        "output_folder": str(output_dir),
        "number of AOI features": len(aoi_gdf),
        "grid_size": grid_size,
        "temporal_extent": temporal_extent_str,
        "season_specifications": season_specifications,
        "year": year,
        "seasonal_preset": seasonal_preset,
        "product_type": resolved_product_type,
        "target_epsg": target_epsg,
        "s1_orbit_state": s1_orbit_state,
        "restart_failed": restart_failed,
        "parallel_jobs": parallel_jobs,
        "randomize_jobs": randomize_jobs,
        "job_options": job_options,
        "poll_sleep": poll_sleep,
        "simplify_logging": simplify_logging,
        "max_retries": max_retries,
        "base_delay": base_delay,
        "max_delay": max_delay,
    }
    for key, value in params.items():
        log_fn(f"{key}: {value}")

    log_fn("Detailed workflow configuration:")
    log_fn(json.dumps(workflow_config.to_dict(), indent=2))
    log_fn("----------------------------------")

    log_fn("Setting up job manager to handle map production...")
    manager = WorldCerealJobManager(
        output_dir=output_dir,
        task=WorldCerealTask.INFERENCE,
        aoi_gdf=aoi_gdf,
        backend_context=backend_context,
        temporal_extent=temporal_extent,
        grid_size=grid_size,
        year=year,
        season_specifications=season_specifications,
        poll_sleep=poll_sleep,
    )

    status_callback = None
    if simplify_logging and runner is None:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )
        status_callback = WorldCerealJobManager.cli_status_callback(
            title="Inference job status"
        )
    elif simplify_logging:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )

    log_fn("Starting map production...")

    break_msg = (
        "Stopping map production...\n"
        "Make sure to manually cancel any running jobs in the backend to avoid unnecessary costs!\n"
        "For this, visit the job tracking page in the backend dashboard: https://openeo.dataspace.copernicus.eu/\n"
    )

    run_kwargs = {
        "restart_failed": restart_failed,
        "randomize_jobs": randomize_jobs,
        "product_type": resolved_product_type,
        "s1_orbit_state": s1_orbit_state,
        "job_options": job_options,
        "target_epsg": target_epsg,
        "parallel_jobs": parallel_jobs,
        "seasonal_preset": seasonal_preset,
        "workflow_config": workflow_config,
        "status_callback": status_callback,
        "max_retries": max_retries,
        "base_delay": base_delay,
        "max_delay": max_delay,
    }

    try:
        if runner is not None:
            runner(
                manager,
                run_kwargs=run_kwargs,
                **(runner_kwargs or {}),
            )
        else:
            manager.run_jobs(**run_kwargs)
    except KeyboardInterrupt:
        log_fn(break_msg)
        manager.stop_job_thread()
        log_fn("Map production has stopped.")
        raise

    log_fn("All done!")
    log_fn(f"Results stored in {output_dir}")

    return manager


class WorldCerealJobManager(MultiBackendJobManager):
    """Unified job manager for inputs, embeddings, and inference tasks."""

    def __init__(
        self,
        output_dir: Union[Path, str],
        task: WorldCerealTask,
        aoi_gdf: gpd.GeoDataFrame,
        backend_context: BackendContext = BackendContext(Backend.CDSE),
        grid_size: Optional[int] = None,
        temporal_extent: Optional[TemporalContext] = None,
        year: Optional[int] = None,
        season_specifications: Optional[Dict[str, TemporalContext]] = None,
        poll_sleep: int = 60,
    ) -> None:
        """Initialize the job manager and prepare or resume the job database."""
        super().__init__(
            root_dir=Path(output_dir),
            poll_sleep=poll_sleep,
        )
        self.task = WorldCerealTask(task)
        self.backend_context = backend_context
        self.output_dir = Path(output_dir)
        # Ensure the output folder exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._aoi_gdf = aoi_gdf
        self._grid_size = grid_size
        self._poll_sleep = poll_sleep

        # Check whether we have information on temporal extent before proceeding
        if temporal_extent is None:
            if not {"start_date", "end_date"}.issubset(aoi_gdf.columns):
                if year is None:
                    raise ValueError(
                        "Provide temporal_extent, year or include 'start_date' and 'end_date' in aoi_gdf."
                    )
        self._temporal_extent = temporal_extent
        self._season_specifications = season_specifications
        self._year = year
        self.prepared_grid: Optional[gpd.GeoDataFrame] = None
        self.job_db: Optional[CsvJobDatabase] = None

        # We check whether a job database already exists
        self.job_db = CsvJobDatabase(path=self._job_db_path())
        if self.job_db.exists():
            logger.info("Job tracking file already exists, resuming from disk.")
            job_df = self.job_db.read()
            self._validate_job_db_df(job_df)
            self.prepared_grid = self._load_production_grid()
        else:
            logger.info(
                "No existing job tracking file found, preparing production grid and creating new job database."
            )
            self.prepared_grid = self._prepare_production_grid(
                aoi_gdf,
                grid_size=grid_size,
            )
            self.job_db = self._create_job_database()

    def add_default_backend(self, parallel_jobs: int = 2) -> None:
        """Register the default backend connection for job submission."""
        backend = self.backend_context.backend
        connection = BACKEND_CONNECTIONS[backend]()
        self.add_backend(
            backend.value, connection=connection, parallel_jobs=parallel_jobs
        )

    def on_job_done(self, job: BatchJob, row: pd.Series) -> None:
        """Dispatch post-processing for completed jobs based on task type."""
        if self.task == WorldCerealTask.INPUTS:
            self._on_job_done_inputs(job, row)
            return
        if self.task == WorldCerealTask.EMBEDDINGS:
            self._on_job_done_embeddings(job, row)
            return
        if self.task == WorldCerealTask.INFERENCE:
            self._on_job_done_inference(job, row)
            return
        raise ValueError(f"Unsupported task: {self.task}")

    def _output_dir_for_row(self, row: pd.Series) -> Path:
        """Return the output directory for a specific tile row."""
        tile_name = row.get("tile_name", "unknown_tile")
        output_dir = self.output_dir / str(tile_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _download_and_tag_assets(self, job: BatchJob, row: pd.Series) -> None:
        """Download job assets and tag filenames with the tile name."""
        output_dir = self._output_dir_for_row(row)
        job_result = job.get_results()
        assets = job_result.get_assets()
        for asset in assets:
            filepath = asset.download(target=output_dir)
            new_filename = f"{filepath.stem}_{row.get('tile_name', 'unknown_tile')}{filepath.suffix}"
            new_filepath = filepath.parent / new_filename
            shutil.move(filepath, new_filepath)

        job_metadata = job.describe()
        result_metadata = job_result.get_metadata()
        job_metadata_path = output_dir / f"job_{job.job_id}.json"
        result_metadata_path = output_dir / f"result_{job.job_id}.json"
        job_metadata_path.write_text(json.dumps(job_metadata), encoding="utf-8")
        result_metadata_path.write_text(json.dumps(result_metadata), encoding="utf-8")

    def _on_job_done_inputs(self, job: BatchJob, row: pd.Series) -> None:
        """Post-processing hook for completed inputs jobs."""
        logger.info(f"Job {job.job_id} completed (inputs)")
        self._download_and_tag_assets(job, row)
        logger.success("Results downloaded and tagged for inputs job")

    def _on_job_done_embeddings(self, job: BatchJob, row: pd.Series) -> None:
        """Post-processing hook for completed embeddings jobs."""
        logger.info(f"Job {job.job_id} completed (embeddings)")
        self._download_and_tag_assets(job, row)
        logger.success("Results downloaded and tagged for embeddings job")

    def _on_job_done_inference(self, job: BatchJob, row: pd.Series) -> None:
        """Post-processing hook for completed inference jobs."""
        logger.info(f"Job {job.job_id} completed (inference)")
        self._download_and_tag_assets(job, row)
        logger.success("Results downloaded and tagged for inference job")

    def _job_db_path(self) -> Path:
        """Return the path of the job tracking CSV."""
        return self.output_dir / "job_tracking.csv"

    def _production_grid_path(self) -> Path:
        """Return the path of the persisted production grid."""
        return self.output_dir / "production_grid.geoparquet"

    def _persist_production_grid(self, gdf: gpd.GeoDataFrame) -> Path:
        """Persist the production grid if it does not already exist."""
        output_path = self._production_grid_path()
        if not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_parquet(output_path, index=False)
        return output_path

    def _load_production_grid(self) -> gpd.GeoDataFrame:
        """Load the persisted production grid from disk."""
        output_path = self._production_grid_path()
        if not output_path.exists():
            raise FileNotFoundError(f"Production grid file not found at {output_path}")
        return gpd.read_parquet(output_path)

    @staticmethod
    def _ensure_required_columns(
        gdf: gpd.GeoDataFrame, required: Sequence[str]
    ) -> None:
        """Validate that required columns exist and contain no missing values."""
        missing = [col for col in required if col not in gdf.columns]
        if missing:
            raise ValueError(
                "The following required attributes are missing in the production grid: "
                + ", ".join(missing)
            )

        null_cols = [col for col in required if gdf[col].isna().any()]
        if null_cols:
            raise ValueError(
                "The following required attributes contain missing values: "
                + ", ".join(null_cols)
            )

    @staticmethod
    def _normalize_date_columns(
        df: pd.DataFrame, columns: Sequence[str]
    ) -> pd.DataFrame:
        """Normalize date columns to YYYY-MM-DD strings."""
        normalized = df.copy()
        for col in columns:
            if col not in normalized.columns:
                continue
            original = normalized[col]
            parsed = pd.to_datetime(original, errors="coerce")
            formatted = parsed.dt.strftime("%Y-%m-%d")
            normalized[col] = formatted.where(original.notna(), None)
        return normalized

    @staticmethod
    def _season_date_columns(df: pd.DataFrame) -> list[str]:
        return [
            col
            for col in df.columns
            if col.startswith("season_start_") or col.startswith("season_end_")
        ]

    def _validate_job_db_df(self, df: pd.DataFrame) -> None:
        """Validate job database columns, types, and season consistency."""
        required = [
            "start_date",
            "end_date",
            "geometry",
            "tile_name",
            "geometry_utm_wkt",
            "epsg_utm",
        ]
        self._ensure_required_columns(df, required)

        # Additional check on tile_name uniqueness
        tile_names = df["tile_name"].astype(str)
        if tile_names.duplicated().any():
            raise ValueError("Job database 'tile_name' values must be unique.")

        # Additional checks on start_date and end_date values
        # Ensure dtype is str, else raise error
        if not pd.api.types.is_string_dtype(df["start_date"]):
            raise ValueError("Job database 'start_date' column must be of string type.")
        if not pd.api.types.is_string_dtype(df["end_date"]):
            raise ValueError("Job database 'end_date' column must be of string type.")
        # Start date should be before end date for all rows
        if (pd.to_datetime(df["start_date"]) > pd.to_datetime(df["end_date"])).any():
            raise ValueError(
                "Job database has start_date values after end_date values."
            )

        # Additional checks on seasonality information for inference task
        if self.task == WorldCerealTask.INFERENCE:
            if "season_ids" not in df.columns or "season_windows" not in df.columns:
                raise ValueError(
                    "Job database must include 'season_ids' and 'season_windows' for inference."
                )
            if df["season_ids"].isna().any():
                raise ValueError("Job database 'season_ids' contains missing values.")
            if df["season_windows"].isna().any():
                raise ValueError(
                    "Job database 'season_windows' contains missing values."
                )

            def _validate_season_windows(value: object) -> None:
                if not isinstance(value, str):
                    raise ValueError(
                        "Job database 'season_windows' must be JSON strings."
                    )
                try:
                    windows = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Job database 'season_windows' contains invalid JSON."
                    ) from exc
                if not isinstance(windows, dict):
                    raise ValueError(
                        "Job database 'season_windows' must be a JSON object."
                    )
                for season_id, window in windows.items():
                    if (
                        not isinstance(window, (list, tuple))
                        or len(window) != 2
                        or window[0] is None
                        or window[1] is None
                    ):
                        raise ValueError(
                            "Job database 'season_windows' must map each season to [start, end]."
                        )
                    start = pd.to_datetime(window[0], errors="raise")
                    end = pd.to_datetime(window[1], errors="raise")
                    if end < start:
                        raise ValueError(
                            f"Season '{season_id}' has end date before start date."
                        )

            df["season_windows"].apply(_validate_season_windows)

    @staticmethod
    def _log_tile_area_range(gdf: gpd.GeoDataFrame) -> None:
        """Log min/max UTM tile areas to help spot problematic tiling."""
        if "geometry_utm_wkt" not in gdf.columns:
            logger.warning(
                "Production grid has no geometry_utm_wkt column; skipping area range log."
            )
            return

        areas = (
            gdf["geometry_utm_wkt"]
            .dropna()
            .apply(lambda value: shapely_wkt.loads(value).area)
        )
        if areas.empty:
            logger.warning("Production grid has no valid UTM geometries to measure.")

        min_km2 = areas.min() / 1_000_000.0
        max_km2 = areas.max() / 1_000_000.0
        logger.info(
            "Production grid tile area range: "
            f"{min_km2:.2f} km^2 to {max_km2:.2f} km^2."
        )

    @staticmethod
    def cli_status_callback(
        title: str = "Job status",
    ) -> Callable[[pd.DataFrame], None]:
        """Build a CLI-friendly status callback for polling job progress."""

        def _callback(status_df: pd.DataFrame) -> None:
            timestamp = time.strftime("%H:%M:%S")
            if status_df.empty:
                logger.info(f"[{timestamp}] {title}: waiting for jobs to start.")
                return
            counts = status_df["status"].value_counts().to_string()
            logger.info(f"[{timestamp}] {title}:\n{counts}")
            logger.info(
                "Detailed job tracking through: https://openeo.dataspace.copernicus.eu/"
            )

        return _callback

    def _prepare_production_grid(
        self,
        aoi_gdf: gpd.GeoDataFrame,
        grid_size: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """Create or normalize the production grid and ensure UTM columns."""

        # Run some basic checks first
        assert (
            "geometry" in aoi_gdf.columns
        ), "The grid file must contain a geometry column."
        assert all(
            aoi_gdf.geometry.type == "Polygon"
        ), "All geometries in the grid file must be of type Polygon."
        if aoi_gdf.crs is None:
            raise ValueError("Input GeoDataFrame must have a CRS.")

        if grid_size is not None:
            # We create a production grid from scratch
            grid = create_production_grid(
                aoi_gdf=aoi_gdf,
                tiling_size_km=grid_size,
                web_mercator_grid=False,
            )
        else:
            grid = aoi_gdf.copy()

        # Ensure date columns are normalized as strings in YYYY-MM-DD format if they exist
        grid = self._normalize_date_columns(
            grid, ["start_date", "end_date", *self._season_date_columns(grid)]
        )

        # Additional checks on tile_name attribute
        if "tile_name" not in grid.columns:
            raise ValueError("Production grid must include a 'tile_name' column.")
        grid = grid.copy()
        grid["tile_name"] = grid["tile_name"].astype(str)
        if grid["tile_name"].isna().any():
            raise ValueError("Production grid has null 'tile_name' values.")
        if grid["tile_name"].duplicated().any():
            raise ValueError("Production grid 'tile_name' values must be unique.")

        # include utm attributes if not already present, as they are required for job execution
        if not {"geometry_utm_wkt", "epsg_utm"}.issubset(grid.columns):
            grid = ensure_utm_grid(
                grid,
                web_mercator_grid=False,
            )

        self._log_tile_area_range(grid)
        # save to disk
        self._persist_production_grid(grid)

        return grid

    def _create_job_database(self) -> CsvJobDatabase:
        """Create and initialize the job database from the production grid."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        job_tracking_path = self._job_db_path()
        job_db = CsvJobDatabase(path=job_tracking_path)

        if self.prepared_grid is None:
            raise ValueError(
                "Production grid is not prepared. Provide aoi_gdf at init."
            )

        logger.info(f"Creating new job tracking file at {job_tracking_path}.")
        production_gdf = self.prepared_grid.copy()

        # Resolve temporal and season specifications for the job database only.
        job_db_df = self._resolve_temporal_and_seasons(production_gdf.copy())
        # Drop redundant season columns if they exist
        season_cols = self._season_date_columns(job_db_df)
        if season_cols:
            job_db_df = job_db_df.drop(columns=season_cols)

        # validate the final job database dataframe before persisting
        self._validate_job_db_df(job_db_df)

        # initialize job database from production grid
        job_db.initialize_from_df(job_db_df)

        # Save the job database
        job_db_df.to_parquet(self.output_dir / "job_db.parquet", index=False)

        return job_db

    @staticmethod
    def _season_ids_from_columns(df: pd.DataFrame) -> list[str]:
        """Return season ids inferred from season_start_/season_end_ columns."""
        start_cols = {
            col[len("season_start_") :]: col
            for col in df.columns
            if col.startswith("season_start_")
        }
        end_cols = {
            col[len("season_end_") :]: col
            for col in df.columns
            if col.startswith("season_end_")
        }
        season_ids = sorted(set(start_cols) | set(end_cols))
        missing = [
            season_id
            for season_id in season_ids
            if season_id not in start_cols or season_id not in end_cols
        ]
        if missing:
            raise ValueError(
                "Season specification is incomplete for ids: " + ", ".join(missing)
            )
        return season_ids

    @staticmethod
    def _apply_season_specifications(
        df: Union[pd.DataFrame, pd.Series],
        season_specifications: Dict[str, TemporalContext],
    ) -> Union[pd.DataFrame, pd.Series]:
        """Attach season_ids and season_windows from specifications."""
        season_ids = sorted(season_specifications.keys())
        season_windows = {
            season_id: (season.start_date, season.end_date)
            for season_id, season in season_specifications.items()
        }
        season_ids_value = ",".join(season_ids)
        season_windows_value = json.dumps(season_windows, sort_keys=True)
        if isinstance(df, pd.Series):
            df["season_ids"] = season_ids_value
            df["season_windows"] = season_windows_value
            return df

        df["season_ids"] = [season_ids_value for _ in range(len(df))]
        df["season_windows"] = [season_windows_value for _ in range(len(df))]
        return df

    @staticmethod
    def _attach_season_metadata_from_columns(df: pd.DataFrame) -> pd.DataFrame:
        season_ids = WorldCerealJobManager._season_ids_from_columns(df)
        if not season_ids:
            return df
        df = df.copy()
        start_cols = {
            season_id: f"season_start_{season_id}" for season_id in season_ids
        }
        end_cols = {season_id: f"season_end_{season_id}" for season_id in season_ids}

        # Check for each row how many complete season specifications we have based on non-null start and end date columns,
        # and attach season_ids and season_windows metadata accordingly
        def _row_season_metadata(row: pd.Series) -> pd.Series:
            available = [
                season_id
                for season_id in season_ids
                if pd.notna(row[start_cols[season_id]])
                and pd.notna(row[end_cols[season_id]])
            ]
            season_windows = {
                season_id: (row[start_cols[season_id]], row[end_cols[season_id]])
                for season_id in available
            }
            return pd.Series(
                {
                    "season_ids": ",".join(available),
                    "season_windows": json.dumps(season_windows, sort_keys=True),
                }
            )

        season_meta = df.apply(_row_season_metadata, axis=1)
        df["season_ids"] = season_meta["season_ids"].values
        df["season_windows"] = season_meta["season_windows"].values
        return df

    def _infer_temporal_and_seasons_from_crop_calendars(
        self, df: pd.DataFrame, get_seasons: bool = True
    ) -> pd.DataFrame:
        """Infer temporal extent and seasons from crop calendars."""

        if self._year is None:
            raise ValueError(
                "Cannot infer seasons from crop calendars without a year specified in the job manager!"
            )
        year = self._year

        df = df.copy()

        for idx, row in df.iterrows():
            extent = self._row_spatial_extent(row)
            # Get the start and end date for each season
            seasons = {}
            for season_idx, season in enumerate(["s1", "s2"]):
                seasons[season] = get_season_dates_for_extent(
                    extent, year, f"tc-{season}"
                )
            # If get_seasons is False, we only infer the overall temporal extent
            if get_seasons:
                row = self._apply_season_specifications(row, seasons)
                df.loc[idx, "season_ids"] = row["season_ids"]
                df.loc[idx, "season_windows"] = row["season_windows"]
            # Infer overall temporal extent from the latest season end
            all_end_dates = [
                pd.to_datetime(season.end_date) for season in seasons.values()
            ]
            proposed_end = max(all_end_dates)
            # snap to end of month
            proposed_end = proposed_end + pd.offsets.MonthEnd(0)
            # proposed start should be 12 months before, on first day of month
            proposed_start = (
                proposed_end - pd.DateOffset(months=12) + pd.offsets.MonthBegin(1)
            )
            df.loc[idx, "end_date"] = proposed_end.strftime("%Y-%m-%d")
            df.loc[idx, "start_date"] = proposed_start.strftime("%Y-%m-%d")

        return df

    def _resolve_temporal_and_seasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve temporal extent and seasons based on grid, user, or inference."""

        # Ensure date columns are normalized as strings in YYYY-MM-DD format if they exist
        resolved = self._normalize_date_columns(
            df, ["start_date", "end_date", *self._season_date_columns(df)]
        )
        has_temporal = {"start_date", "end_date"}.issubset(resolved.columns)
        temporal_complete = False
        if has_temporal:
            start_parsed = pd.to_datetime(resolved["start_date"], errors="coerce")
            end_parsed = pd.to_datetime(resolved["end_date"], errors="coerce")
            temporal_complete = start_parsed.notna().all() and end_parsed.notna().all()
            if not temporal_complete:
                logger.warning(
                    "Temporal extent columns are present but contain invalid dates."
                )
                raise ValueError(
                    "Temporal extent columns 'start_date' and 'end_date' must contain valid date strings in YYYY-MM-DD format."
                )

        # For inputs and embeddings tasks, we require only temporal extent (start and end dates)
        # we prioritize per-tile temporal extent specifications in the grid,
        # then user-provided temporal extent,
        # and finally inference from crop calendars if no other specifications are found
        if self.task in {WorldCerealTask.INPUTS, WorldCerealTask.EMBEDDINGS}:
            if has_temporal and temporal_complete:
                logger.info("Using temporal extent from production grid for all tiles.")
                return resolved
            if self._temporal_extent is not None:
                logger.info("Using user-provided temporal extent for all tiles:")
                logger.info(f"  start_date: {self._temporal_extent.start_date}")
                logger.info(f"  end_date: {self._temporal_extent.end_date}")
                resolved["start_date"] = self._temporal_extent.start_date
                resolved["end_date"] = self._temporal_extent.end_date
                return resolved
            # If no temporal extent is provided, we attempt to infer it from crop calendars
            logger.info("No temporal extent found. Inferring from crop calendars.")
            resolved = self._infer_temporal_and_seasons_from_crop_calendars(
                resolved, get_seasons=False
            )
            return resolved

        # Now we treat the inference task, which requires both temporal extent and season specifications
        # First we check for per-tile specifications in the grid
        season_ids = self._season_ids_from_columns(resolved)
        if temporal_complete and season_ids:
            if len(season_ids) > 2:
                raise ValueError(
                    f"At most two season specifications are supported. {len(season_ids)} found in the grid."
                )
            logger.info(
                f"Using temporal extent and {len(season_ids)} season(s) from production grid."
            )
            return self._attach_season_metadata_from_columns(resolved)

        if has_temporal or season_ids:
            raise ValueError(
                "We found either start_date/end_date or season specifications in the production grid, but not both. "
                "Cannot continue."
            )

        # Now we check for user-provided temporal extent and season specifications
        if self._temporal_extent is not None and self._season_specifications:
            logger.info(
                "Using user-provided temporal extent and seasons for all tiles."
            )
            if len(self._season_specifications) > 2:
                raise ValueError(
                    f"At most two season specifications are supported. {len(self._season_specifications)} provided."
                )
            resolved["start_date"] = self._temporal_extent.start_date
            resolved["end_date"] = self._temporal_extent.end_date
            logger.info(
                f"Start date: {self._temporal_extent.start_date}, End date: {self._temporal_extent.end_date}"
            )
            resolved = self._apply_season_specifications(
                resolved, self._season_specifications
            )
            return resolved

        if self._temporal_extent is not None and not self._season_specifications:
            raise ValueError(
                "Temporal extent is provided without season specifications. Cannot continue."
            )
        if self._temporal_extent is None and self._season_specifications:
            raise ValueError(
                "Season specifications are provided without temporal extent. Cannot continue."
            )

        # If no specifications are found, we attempt to infer both temporal extent and seasons from crop calendars
        logger.info(
            "No temporal extent or season specifications found. Inferring from crop calendars..."
        )
        resolved = self._infer_temporal_and_seasons_from_crop_calendars(
            resolved, get_seasons=True
        )
        return resolved

    @staticmethod
    def _row_temporal_extent(row: pd.Series) -> TemporalContext:
        """Build a TemporalContext from a job database row."""
        return TemporalContext(start_date=row.start_date, end_date=row.end_date)

    @staticmethod
    def _row_spatial_extent(row: pd.Series) -> BoundingBoxExtent:
        """Build a BoundingBoxExtent from a job database row."""
        utm_geom = shapely_wkt.loads(row["geometry_utm_wkt"])
        return BoundingBoxExtent(*utm_geom.bounds, epsg=int(row["epsg_utm"]))

    def _run_jobs_common(
        self,
        *,
        restart_failed: bool,
        randomize_jobs: bool,
        parallel_jobs: int,
        start_job: Callable[..., BatchJob],
        max_retries: int,
        base_delay: float,
        max_delay: float,
        status_callback: Optional[Callable[[pd.DataFrame], None]] = None,
    ) -> CsvJobDatabase:
        """Run jobs with retry, optional status callback, and randomization."""
        if self.job_db is None:
            raise ValueError("Job database was not created during initialization.")
        job_db = self.job_db

        job_df = job_db.read()
        if restart_failed and not job_df.empty:
            job_df.loc[
                job_df["status"].isin(["error", "start_failed"]),
                "status",
            ] = "not_started"
            job_db.persist(job_df)

        if randomize_jobs and not job_df.empty:
            job_df = job_df.sample(frac=1).reset_index(drop=True)
            job_db.persist(job_df)

        attempt = 0
        while True:
            try:
                self.add_default_backend(parallel_jobs=parallel_jobs)
                if status_callback is None:
                    super().run_jobs(
                        start_job=start_job,
                        job_db=job_db,
                    )
                else:
                    self.start_job_thread(start_job=start_job, job_db=job_db)
                    while self._thread and self._thread.is_alive():
                        try:
                            status_df = job_db.read()
                            status_callback(status_df)
                        except pd.errors.EmptyDataError:
                            pass
                        time.sleep(self._poll_sleep)
                break
            except Exception as exc:
                if attempt < max_retries:
                    attempt += 1
                    backoff = min(base_delay * 2**attempt, max_delay)
                    jitter = random.uniform(-0.2 * backoff, 0.2 * backoff)
                    delay = max(0.0, backoff + jitter)
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed: {exc}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"Max retries reached. Last error: {exc}")
                raise

        return job_db

    def _run_inputs_jobs(
        self,
        *,
        restart_failed: bool = False,
        randomize_jobs: bool = False,
        parallel_jobs: int = 2,
        s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
        job_options: Optional[Dict[str, Union[str, int, None]]] = None,
        compositing_window: Literal["month", "dekad"] = "month",
        tile_size: Optional[int] = 128,
        status_callback: Optional[Callable[[pd.DataFrame], None]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ) -> CsvJobDatabase:
        """Run inputs jobs using the unified job manager pipeline."""
        start_job = partial(
            self._create_inputs_job,
            s1_orbit_state=s1_orbit_state,
            job_options=job_options,
            compositing_window=compositing_window,
            tile_size=tile_size,
        )
        return self._run_jobs_common(
            restart_failed=restart_failed,
            randomize_jobs=randomize_jobs,
            parallel_jobs=parallel_jobs,
            start_job=start_job,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            status_callback=status_callback,
        )

    def _run_embeddings_jobs(
        self,
        *,
        restart_failed: bool = False,
        randomize_jobs: bool = False,
        parallel_jobs: int = 2,
        s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
        job_options: Optional[Dict[str, Union[str, int, None]]] = None,
        embeddings_parameters: EmbeddingsParameters = EmbeddingsParameters(),
        scale_uint16: bool = True,
        status_callback: Optional[Callable[[pd.DataFrame], None]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ) -> CsvJobDatabase:
        """Run embeddings jobs using the unified job manager pipeline."""
        start_job = partial(
            self._create_embeddings_job,
            s1_orbit_state=s1_orbit_state,
            job_options=job_options,
            embeddings_parameters=embeddings_parameters,
            scale_uint16=scale_uint16,
        )
        return self._run_jobs_common(
            restart_failed=restart_failed,
            randomize_jobs=randomize_jobs,
            parallel_jobs=parallel_jobs,
            start_job=start_job,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            status_callback=status_callback,
        )

    def _run_inference_jobs(
        self,
        *,
        product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
        target_epsg: Optional[int] = None,
        s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
        job_options: Optional[dict] = None,
        parallel_jobs: int = 2,
        seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
        workflow_config: Optional[WorldCerealWorkflowConfig] = None,
        restart_failed: bool = False,
        randomize_jobs: bool = False,
        status_callback: Optional[Callable[[pd.DataFrame], None]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ) -> CsvJobDatabase:
        """Run inference jobs using the unified job manager pipeline."""
        start_job = partial(
            self._create_inference_job,
            product_type=product_type,
            s1_orbit_state=s1_orbit_state,
            job_options=job_options,
            target_epsg=target_epsg,
            seasonal_preset=seasonal_preset,
            workflow_config=workflow_config,
        )
        return self._run_jobs_common(
            restart_failed=restart_failed,
            randomize_jobs=randomize_jobs,
            parallel_jobs=parallel_jobs,
            start_job=start_job,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            status_callback=status_callback,
        )

    def run_jobs(self, **kwargs: object) -> Optional[CsvJobDatabase]:
        """Dispatch to the task-specific run method."""
        if self.task == WorldCerealTask.INPUTS:
            return self._run_inputs_jobs(**kwargs)
        if self.task == WorldCerealTask.EMBEDDINGS:
            return self._run_embeddings_jobs(**kwargs)
        if self.task == WorldCerealTask.INFERENCE:
            return self._run_inference_jobs(**kwargs)
        raise ValueError(f"Unsupported task: {self.task}")

    def _create_inputs_job(
        self,
        row: pd.Series,
        connection: openeo.Connection,
        provider,
        connection_provider,
        s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
        job_options: Optional[Dict[str, Union[str, int, None]]] = None,
        compositing_window: Literal["month", "dekad"] = "month",
        tile_size: Optional[int] = 128,
    ) -> BatchJob:
        """Create an inputs job for a single tile."""
        temporal_extent = self._row_temporal_extent(row)
        spatial_extent = self._row_spatial_extent(row)

        inputs = create_worldcereal_process_graph(
            task=WorldCerealTask.INPUTS,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            s1_orbit_state=s1_orbit_state,
            target_epsg=int(row["epsg_utm"]),
            compositing_window=compositing_window,
            tile_size=tile_size,
            connection=connection,
        )

        resolved_options = dict(DEFAULT_INPUTS_JOB_OPTIONS)
        if job_options:
            resolved_options.update(job_options)

        return inputs.create_job(
            title=f"WorldCereal collect inputs for {row.tile_name}",
            job_options=resolved_options,
        )

    def _create_embeddings_job(
        self,
        row: pd.Series,
        connection: openeo.Connection,
        provider,
        connection_provider,
        s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
        job_options: Optional[Dict[str, Union[str, int, None]]] = None,
        embeddings_parameters: EmbeddingsParameters = EmbeddingsParameters(),
        scale_uint16: bool = True,
    ) -> BatchJob:
        """Create an embeddings job for a single tile."""
        temporal_extent = self._row_temporal_extent(row)
        spatial_extent = self._row_spatial_extent(row)

        embeddings = create_worldcereal_process_graph(
            task=WorldCerealTask.EMBEDDINGS,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            s1_orbit_state=s1_orbit_state,
            target_epsg=int(row["epsg_utm"]),
            embeddings_parameters=embeddings_parameters,
            scale_uint16=scale_uint16,
            connection=connection,
        )

        resolved_options = dict(DEFAULT_INFERENCE_JOB_OPTIONS)
        if job_options:
            resolved_options.update(job_options)

        return embeddings.create_job(
            title=f"WorldCereal embeddings for {row.tile_name}",
            job_options=resolved_options,
        )

    def _create_inference_job(
        self,
        row: pd.Series,
        connection: openeo.Connection,
        provider,
        connection_provider,
        product_type: WorldCerealProductType = WorldCerealProductType.CROPTYPE,
        s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
        target_epsg: Optional[int] = None,
        job_options: Optional[dict] = None,
        seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
        workflow_config: Optional[WorldCerealWorkflowConfig] = None,
    ) -> BatchJob:
        """Create an inference job for a single tile."""
        temporal_extent = self._row_temporal_extent(row)
        spatial_extent = self._row_spatial_extent(row)

        if target_epsg is None:
            target_epsg = int(row["epsg_utm"])

        inference_result = create_worldcereal_process_graph(
            task=WorldCerealTask.INFERENCE,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            product_type=product_type,
            s1_orbit_state=s1_orbit_state,
            target_epsg=target_epsg,
            connection=connection,
            seasonal_preset=seasonal_preset,
            workflow_config=workflow_config,
            row=row,
        )

        inference_job_options = dict(DEFAULT_INFERENCE_JOB_OPTIONS)
        if job_options is not None:
            inference_job_options.update(job_options)

        return connection.create_job(
            inference_result,
            title=f"WorldCereal [{product_type.value}] job_{row.tile_name}",
            description="Job that performs end-to-end WorldCereal inference",
            additional=inference_job_options,
        )
