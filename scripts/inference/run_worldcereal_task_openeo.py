"""Run WorldCereal openEO tasks using a unified CLI.

This script consolidates inputs collection, embeddings computation, and
classification map production into a single entry point. Use --task to
select which workflow to run.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
from openeo_gfmap import TemporalContext

from worldcereal.job import WorldCerealTask
from worldcereal.job_params import WorldCerealJobParams, build_job_params_from_args
from worldcereal.jobmanager import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    run_worldcereal_task,
)
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.parameters import EmbeddingsParameters, WorldCerealProductType
from worldcereal.utils import parse_job_options_from_args


def _parse_season_specifications(
    specs: Dict[str, Any],
) -> Dict[str, TemporalContext]:
    season_specifications: Dict[str, TemporalContext] = {}
    for season_id, window in specs.items():
        if isinstance(window, list) and len(window) == 2:
            start_date, end_date = window
        elif isinstance(window, dict):
            start_date = window.get("start_date")
            end_date = window.get("end_date")
        else:
            raise ValueError(
                "Season specifications must be [start_date, end_date] or {start_date, end_date}."
            )
        if not start_date or not end_date:
            raise ValueError(
                "Season specifications must include both start_date and end_date."
            )
        season_specifications[season_id] = TemporalContext(
            start_date=str(start_date),
            end_date=str(end_date),
        )
    return season_specifications


def _load_season_specifications(
    args: argparse.Namespace,
) -> Optional[Dict[str, TemporalContext]]:
    if not args.season_specifications_json:
        return None
    raw_specs = json.loads(args.season_specifications_json)
    if not isinstance(raw_specs, dict):
        raise ValueError("Season specifications must be a JSON object.")
    return _parse_season_specifications(raw_specs)


def _task_choices() -> list[str]:
    return [task.value for task in WorldCerealTask]


def _parse_task(value: str) -> WorldCerealTask:
    try:
        return WorldCerealTask(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Unsupported task '{value}'. Choose from: {', '.join(_task_choices())}"
        ) from exc


def _validate_classification_flags(
    args: argparse.Namespace,
) -> tuple[Optional[bool], Optional[bool], Optional[bool]]:
    if args.enable_cropland_head and args.disable_cropland_head:
        raise ValueError(
            "Choose only one of --enable-cropland-head or --disable-cropland-head."
        )
    if args.enable_croptype_head and args.disable_croptype_head:
        raise ValueError(
            "Choose only one of --enable-croptype-head or --disable-croptype-head."
        )
    if args.enforce_cropland_gate and args.disable_cropland_gate:
        raise ValueError(
            "Choose only one of --enforce-cropland-gate or --disable-cropland-gate."
        )

    enable_cropland_head = (
        True
        if args.enable_cropland_head
        else False if args.disable_cropland_head else None
    )
    enable_croptype_head = (
        True
        if args.enable_croptype_head
        else False if args.disable_croptype_head else None
    )
    enforce_cropland_gate = (
        True
        if args.enforce_cropland_gate
        else False if args.disable_cropland_gate else None
    )
    return enable_cropland_head, enable_croptype_head, enforce_cropland_gate


def main(task: WorldCerealTask, params: WorldCerealJobParams) -> None:
    run_worldcereal_task(task, dict(params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="WC - WorldCereal openEO task",
        description="Run a WorldCereal task using the unified job manager",
    )

    parser.add_argument(
        "--task",
        type=_parse_task,
        required=True,
        help=f"WorldCereal task to run. Choices: {', '.join(_task_choices())}.",
    )

    parser.add_argument(
        "--grid_path",
        type=Path,
        required=True,
        help="Path to the grid file (.parquet, .geoparquet, .gpkg, .shp) defining the locations to extract.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="Tile size in kilometers for splitting AOIs. If not specified, the original AOI geometries will be used.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Starting date for processing in 'YYYY-MM-DD' format.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Ending date for processing in 'YYYY-MM-DD' format.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year used for crop calendar inference when dates are missing.",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Path to folder where to save the resulting products.",
    )
    parser.add_argument(
        "--s1_orbit_state",
        type=str,
        choices=["ASCENDING", "DESCENDING"],
        default=None,
        help="Specify the S1 orbit state to use for the jobs.",
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=2,
        help="The maximum number of parallel jobs to run at the same time.",
    )
    parser.add_argument(
        "--restart_failed",
        action="store_true",
        help="Restart the jobs that previously failed.",
    )
    parser.add_argument(
        "--no-randomize-jobs",
        action="store_false",
        dest="randomize_jobs",
        help="Disable randomization of job order before submission.",
    )
    parser.add_argument(
        "--poll_sleep",
        type=int,
        default=60,
        help="Seconds to wait between status updates.",
    )
    parser.add_argument(
        "--no-simplify-logging",
        action="store_false",
        dest="simplify_logging",
        help="Disable compact CLI status callback and openEO log suppression.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max number of job submission retries.",
    )
    parser.add_argument(
        "--base_delay",
        type=float,
        default=DEFAULT_BASE_DELAY,
        help="Initial retry delay in seconds.",
    )
    parser.add_argument(
        "--max_delay",
        type=float,
        default=DEFAULT_MAX_DELAY,
        help="Maximum retry delay in seconds.",
    )

    parser.add_argument(
        "--compositing_window",
        type=str,
        choices=["month", "dekad"],
        default="month",
        help="Temporal compositing window for inputs (month or dekad).",
    )

    parser.add_argument(
        "--presto_model_url",
        type=str,
        default=None,
        help="Optional custom Presto model URL for embeddings.",
    )
    parser.add_argument(
        "--no_scale_uint16",
        action="store_true",
        help="Disable uint16 scaling for embeddings outputs.",
    )

    parser.add_argument(
        "--product",
        type=str,
        choices=["cropland", "croptype"],
        default=None,
        help="Product to generate for classification.",
    )
    parser.add_argument(
        "--class-probabilities",
        action="store_true",
        help="Output per-class probabilities in the resulting product.",
    )
    parser.add_argument(
        "--seasonal-model-zip",
        type=str,
        default=None,
        help="Path to .zip file of a seasonal Presto model override.",
    )
    parser.add_argument(
        "--enable-cropland-head",
        action="store_true",
        help="Force enable the cropland (landcover) head.",
    )
    parser.add_argument(
        "--disable-cropland-head",
        action="store_true",
        help="Force disable the cropland (landcover) head.",
    )
    parser.add_argument(
        "--landcover-head-zip",
        type=str,
        default=None,
        help="Path to .zip file of a cropland/landcover head override.",
    )
    parser.add_argument(
        "--enable-croptype-head",
        action="store_true",
        help="Force enable the croptype head.",
    )
    parser.add_argument(
        "--disable-croptype-head",
        action="store_true",
        help="Force disable the croptype head.",
    )
    parser.add_argument(
        "--croptype-head-zip",
        type=str,
        default=None,
        help="Path to .zip file of a croptype head override.",
    )
    parser.add_argument(
        "--enforce-cropland-gate",
        action="store_true",
        help="Force enable the cropland gate for croptype outputs.",
    )
    parser.add_argument(
        "--disable-cropland-gate",
        action="store_true",
        help="Force disable the cropland gate for croptype outputs.",
    )
    parser.add_argument(
        "--enable-cropland-postprocess",
        action="store_true",
        help="Enable cropland postprocessing.",
    )
    parser.add_argument(
        "--cropland-postprocess-method",
        type=str,
        choices=["majority_vote", "smooth_probabilities"],
        default=None,
        help="Cropland postprocess method override.",
    )
    parser.add_argument(
        "--cropland-postprocess-kernel-size",
        type=int,
        default=None,
        help="Cropland postprocess kernel size override.",
    )
    parser.add_argument(
        "--enable-croptype-postprocess",
        action="store_true",
        help="Enable croptype postprocessing.",
    )
    parser.add_argument(
        "--croptype-postprocess-method",
        type=str,
        choices=["majority_vote", "smooth_probabilities"],
        default=None,
        help="Croptype postprocess method override.",
    )
    parser.add_argument(
        "--croptype-postprocess-kernel-size",
        type=int,
        default=None,
        help="Croptype postprocess kernel size override.",
    )
    parser.add_argument(
        "--seasonal-preset",
        type=str,
        default=DEFAULT_SEASONAL_WORKFLOW_PRESET,
        help="Seasonal workflow preset to use for classification.",
    )
    parser.add_argument(
        "--season-specifications-json",
        type=str,
        default=None,
        help=(
            "JSON string mapping season ids to [start_date, end_date] or {start_date, end_date}."
        ),
    )
    parser.add_argument(
        "--target-epsg",
        type=int,
        default=None,
        help="EPSG code for output reprojection.",
    )

    parser.add_argument(
        "--driver_memory",
        type=str,
        default=None,
        help="Driver memory",
    )
    parser.add_argument(
        "--driver_memoryOverhead",
        type=str,
        default=None,
        help="Driver memory overhead.",
    )
    parser.add_argument(
        "--executor_cores",
        type=int,
        default=None,
        help="Executor cores.",
    )
    parser.add_argument(
        "--executor_memory",
        type=str,
        default=None,
        help="Executor memory.",
    )
    parser.add_argument(
        "--executor_memoryOverhead",
        type=str,
        default=None,
        help="Executor memory overhead.",
    )
    parser.add_argument(
        "--max_executors",
        type=int,
        default=None,
        help="Max executors.",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="openEO image name.",
    )
    parser.add_argument(
        "--organization_id",
        type=int,
        default=None,
        help="Organization id.",
    )

    args = parser.parse_args()
    if not hasattr(args, "simplify_logging"):
        args.simplify_logging = True
    if not hasattr(args, "randomize_jobs"):
        args.randomize_jobs = True

    aoi_gdf = (
        gpd.read_parquet(args.grid_path)
        if args.grid_path.suffix.lower() in [".parquet", ".geoparquet"]
        else gpd.read_file(args.grid_path)
    )

    temporal_extent = None
    if args.start_date and args.end_date:
        temporal_extent = TemporalContext(
            start_date=args.start_date,
            end_date=args.end_date,
        )

    job_options = parse_job_options_from_args(args)

    task = args.task
    embeddings_parameters = None
    scale_uint16 = True
    product_type = None
    enable_cropland_head = None
    enable_croptype_head = None
    enforce_cropland_gate = None
    season_specifications = None

    if task == WorldCerealTask.EMBEDDINGS:
        if args.presto_model_url:
            embeddings_parameters = EmbeddingsParameters(
                presto_model_url=args.presto_model_url
            )
        else:
            embeddings_parameters = EmbeddingsParameters()
        scale_uint16 = not args.no_scale_uint16

    if task == WorldCerealTask.CLASSIFICATION:
        if not args.product:
            raise ValueError("--product is required when --task=classification.")
        product_type = WorldCerealProductType(args.product)
        enable_cropland_head, enable_croptype_head, enforce_cropland_gate = (
            _validate_classification_flags(args)
        )
        season_specifications = _load_season_specifications(args)

    params = build_job_params_from_args(
        task,
        args,
        aoi_gdf=aoi_gdf,
        temporal_extent=temporal_extent,
        season_specifications=season_specifications,
        job_options=job_options,
        product_type=product_type,
        enable_cropland_head=enable_cropland_head,
        enable_croptype_head=enable_croptype_head,
        enforce_cropland_gate=enforce_cropland_gate,
        embeddings_parameters=embeddings_parameters,
        scale_uint16=scale_uint16,
    )

    main(task, params)
