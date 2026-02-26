"""Cropland mapping inference script using the unified job manager.

Usage example:
    python scripts/inference/cropland_croptype_mapping.py \
        --grid_path ./bbox/test.gpkg \
        --grid_size 20 \
        2024-01-01 2024-12-31 cropland ./outputs/maps \
        --season-specifications-json '{"s1": ["2024-03-01", "2024-08-31"], "s2": {"start_date": "2024-09-01", "end_date": "2025-02-28"}}' \
        --parallel-jobs 4
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
from loguru import logger
from openeo_gfmap import TemporalContext
from openeo_gfmap.backend import Backend, BackendContext

from worldcereal.job import WorldCerealTask
from worldcereal.jobmanager import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    WorldCerealJobManager,
)
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.openeo.workflow_config import SeasonSection, WorldCerealWorkflowConfig
from worldcereal.parameters import WorldCerealProductType
from worldcereal.utils import parse_job_options_from_args


def _build_workflow_config(
    export_class_probabilities: bool,
) -> Optional[WorldCerealWorkflowConfig]:
    if not export_class_probabilities:
        return None
    return WorldCerealWorkflowConfig(
        season=SeasonSection(export_class_probabilities=True)
    )


def _parse_season_specifications(
    season_json: Optional[str], season_file: Optional[Path]
) -> Optional[Dict[str, TemporalContext]]:
    if season_json and season_file:
        raise ValueError(
            "Provide only one of --season-specifications-json or --season-specifications-file."
        )
    if not season_json and not season_file:
        return None

    if season_file:
        season_json = season_file.read_text()

    try:
        raw = json.loads(season_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Season specifications must be valid JSON.") from exc

    if not isinstance(raw, dict):
        raise ValueError("Season specifications JSON must be an object.")

    parsed: Dict[str, TemporalContext] = {}
    for season_id, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            start_date, end_date = value
        elif isinstance(value, dict):
            start_date = value.get("start_date")
            end_date = value.get("end_date")
        else:
            raise ValueError(
                "Season specifications values must be [start_date, end_date] or a dict with start_date/end_date."
            )
        if not start_date or not end_date:
            raise ValueError(
                f"Season '{season_id}' must include start_date and end_date."
            )
        parsed[str(season_id)] = TemporalContext(
            start_date=str(start_date), end_date=str(end_date)
        )

    return parsed


def main(
    aoi_gdf: gpd.GeoDataFrame,
    output_folder: Path,
    product: WorldCerealProductType,
    grid_size: int = 20,
    temporal_extent: Optional[TemporalContext] = None,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    target_epsg: Optional[int] = None,
    s1_orbit_state: Optional[str] = None,
    year: Optional[int] = None,
    season_specifications: Optional[Dict[str, TemporalContext]] = None,
    parallel_jobs: int = 2,
    seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
    workflow_config: Optional[WorldCerealWorkflowConfig] = None,
    restart_failed: bool = False,
    randomize_jobs: bool = False,
    job_options: Optional[dict] = None,
    poll_sleep: int = 60,
    simplify_logging: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> None:

    logger.info("------------------------------------")
    logger.info("STARTING WORKFLOW: Crop mapping")
    logger.info("------------------------------------")
    logger.info("----- Workflow configuration -----")

    if temporal_extent is not None:
        temporal_extent_str = (
            f"{temporal_extent.start_date} to {temporal_extent.end_date}"
        )
    else:
        temporal_extent_str = "None"

    params = {
        "number of AOI features": len(aoi_gdf),
        "grid_size": grid_size,
        "output_folder": str(output_folder),
        "product": product,
        "temporal_extent": temporal_extent_str,
        "year": year,
        "season_specifications": season_specifications,
        "seasonal_preset": seasonal_preset,
        "workflow_config": workflow_config,
        "backend_context": backend_context.backend.value,
        "target_epsg": target_epsg,
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
        logger.info(f"{key}: {value}")
    logger.info("----------------------------------")

    logger.info("Initializing job manager...")
    manager = WorldCerealJobManager(
        output_dir=output_folder,
        task=WorldCerealTask.INFERENCE,
        backend_context=backend_context,
        aoi_gdf=aoi_gdf,
        grid_size=grid_size,
        temporal_extent=temporal_extent,
        year=year,
        season_specifications=season_specifications,
        poll_sleep=poll_sleep,
    )
    logger.info("Job manager initialized!")

    status_callback = WorldCerealJobManager.cli_status_callback(
        title="Inference job status"
    )
    if simplify_logging:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )

    logger.info("Starting job submissions...")
    break_msg = (
        "Stopping crop mapping...\n"
        "Make sure to manually cancel any running jobs in the backend to avoid unnecessary costs!\n"
        "For this, visit the job tracking page in the backend dashboard: https://openeo.dataspace.copernicus.eu/\n"
    )

    try:
        manager.run_jobs(
            product_type=product,
            target_epsg=target_epsg,
            s1_orbit_state=s1_orbit_state,
            parallel_jobs=parallel_jobs,
            seasonal_preset=seasonal_preset,
            workflow_config=workflow_config,
            restart_failed=restart_failed,
            job_options=job_options,
            randomize_jobs=randomize_jobs,
            status_callback=status_callback,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    except KeyboardInterrupt:
        logger.info(break_msg)
        manager.stop_job_thread()
        logger.info("Crop mapping has stopped.")
        raise

    logger.success("All done!")
    logger.info(f"Results stored in {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="WC - Crop Mapping Inference",
        description="Crop mapping inference using the WorldCereal job manager",
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
        default=None,
        help="Tile size in kilometers for splitting AOIs. If not specified, the original AOI geometries will be used.",
    )
    parser.add_argument(
        "start_date",
        type=str,
        help="Starting date for processing in 'YYYY-MM-DD' format.",
    )
    parser.add_argument(
        "end_date",
        type=str,
        help="Ending date for processing in 'YYYY-MM-DD' format.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year used for crop calendar inference when dates are missing.",
    )
    parser.add_argument(
        "product",
        type=str,
        choices=["cropland", "croptype"],
        help="Product to generate.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Path to folder where to save the resulting products.",
    )
    parser.add_argument(
        "--class-probabilities",
        action="store_true",
        help="Output per-class probabilities in the resulting product",
    )
    parser.add_argument(
        "--seasonal-preset",
        type=str,
        default=DEFAULT_SEASONAL_WORKFLOW_PRESET,
        help="Seasonal workflow preset to use for inference.",
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
        "--season-specifications-file",
        type=Path,
        default=None,
        help="Path to a JSON file with season specifications.",
    )
    parser.add_argument(
        "--target-epsg",
        type=int,
        default=None,
        help="EPSG code for output reprojection.",
    )
    parser.add_argument(
        "--s1_orbit_state",
        type=str,
        choices=["ASCENDING", "DESCENDING"],
        default=None,
        help="Specify the S1 orbit state to use for the jobs.",
    )
    parser.add_argument(
        "--parallel-jobs",
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
        "--randomize-jobs",
        action="store_true",
        help="Randomize the order of jobs before submitting them.",
    )
    parser.add_argument(
        "--poll_sleep",
        type=int,
        default=60,
        help="Seconds to wait between status updates.",
    )
    parser.add_argument(
        "--simplify_logging",
        action="store_true",
        help=(
            "Suppress openEO job manager logs while keeping the compact status callback."
        ),
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
        "--executor_cores", type=int, default=None, help="Executor cores."
    )
    parser.add_argument(
        "--executor_memory", type=str, default=None, help="Executor memory."
    )
    parser.add_argument(
        "--executor_memoryOverhead",
        type=str,
        default=None,
        help="Executor memory overhead.",
    )
    parser.add_argument(
        "--max_executors", type=int, default=None, help="Max executors."
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="python38",
        help="openEO image name.",
    )
    parser.add_argument(
        "--organization_id", type=int, default=None, help="Organization id."
    )

    args = parser.parse_args()

    aoi_gdf = None
    if args.grid_path.suffix.lower() in [".parquet", ".geoparquet"]:
        aoi_gdf = gpd.read_parquet(args.grid_path)
    else:
        aoi_gdf = gpd.read_file(args.grid_path)

    temporal_extent = None
    if args.start_date and args.end_date:
        temporal_extent = TemporalContext(
            start_date=args.start_date,
            end_date=args.end_date,
        )

    workflow_config = _build_workflow_config(args.class_probabilities)

    try:
        season_specifications = _parse_season_specifications(
            args.season_specifications_json, args.season_specifications_file
        )
    except ValueError as exc:
        parser.error(str(exc))

    job_options = parse_job_options_from_args(args)

    main(
        aoi_gdf=aoi_gdf,
        output_folder=args.output_folder,
        product=WorldCerealProductType(args.product),
        grid_size=args.grid_size,
        temporal_extent=temporal_extent,
        season_specifications=season_specifications,
        year=args.year,
        seasonal_preset=args.seasonal_preset,
        workflow_config=workflow_config,
        backend_context=BackendContext(Backend.CDSE),
        target_epsg=args.target_epsg,
        s1_orbit_state=args.s1_orbit_state,
        parallel_jobs=args.parallel_jobs,
        restart_failed=args.restart_failed,
        randomize_jobs=args.randomize_jobs,
        job_options=job_options,
        poll_sleep=args.poll_sleep,
        simplify_logging=args.simplify_logging,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        max_delay=args.max_delay,
    )
