"""Collect preprocessed inputs for polygon patches.

This script prepares and submits openEO batch jobs that generate the input
stacks used by downstream WorldCereal embedding and inference workflows.
It creates (or reuses) a production grid, determines the temporal extent to be
used for each tile, then submits one job per grid tile.

WORKFLOW DEMO
--------------
A demo of this workflow can be found in worldcereal-classification/notebooks/worldcereal_preprocessed_inputs.ipynb,
which uses the same underlying job manager and processing logic as this script, but with additional visualizations and interactivity.

PROCESSING CHAIN SUMMARY
------------------------
1) Read AOI grid from ``--grid_path``.
2) Optionally tile AOIs into a production grid using ``--grid_size``.
3) Enrich tiles with per-tile UTM geometry and EPSG codes.
4) Determine time range to be extracted per tile (see "Temporal extent options" below).
5) Initialize or resume a job database in ``--output_folder``.
6) Submit openEO jobs (one per tile), track status in the job database and
    download results when done.

REQUIRED PARAMETERS
---------------
* ``--grid_path`` (required): AOI file (.parquet, .geoparquet, .gpkg, .shp).
* ``--output_folder`` (required): output folder for the job DB and results. 

SPATIAL EXTENT OPTIONS
----------------------
* OPTION 1: Use an existing "production-ready" grid file.
    - Provide a grid file with polygons and a unique ``tile_name`` column --> ``--grid_path``.
    - Do not specify any ``--grid_size`` --> the polygons are used as-is. 
    
    Note that we highly recommend your individual grid tiles to be smaller
    than 50x50 km to avoid memory issues and long runtimes.
    In case of larger/irregular AOI's, we recommend OPTION 2.
    
* OPTION 2: Create a production grid from larger AOIs.
    - - Provide a geometry file with polygons and a unique "id" column as --grid_path.
    - Set ``--grid_size`` (km) to tile each of your AOIs into smaller patches. 
        The tiling aligns with Sentinel-2 MGRS tiles and UTM.
        The script adds a "tile_name" column with unique identifiers for each tile
        in the format "{id}_{mgrs_tile}_{tile_index}".

TEMPORAL EXTENT OPTIONS
-----------------------
Choose exactly one of the following approaches:

* OPTION 1: Use tile-specific start/end dates specified in your grid file.
    This requires your grid file to contain pre-defined ``start_date`` and ``end_date`` columns.
    Both str and datetime formats are supported. The script will use these per-tile dates as-is.
    
    Note that in case start/end dates are present in the grid file, OPTIONS 2 and 3 are ignored,
    even if ``--start_date``/``--end_date`` or ``--year`` are provided as input parameters.

* OPTION 2: Use one common start and end date for all tiles.
    This requires specifying ``--start_date`` and ``--end_date`` as input parameters.

* OPTION 3: Ensure temporal extents per tile align with local crop calendars.
    This is the default fallback when OPTIONS 1 and 2 are not applicable.
    We automatically infer for each tile the relevant crop seasons from the global WorldCereal 
    crop calendars based on the tile location and the provided ``--year``.
    Then we derive a 12 month temporal extent per tile that covers the relevant crop seasons.
    
    This approach is recommended to ensure that the extracted time series contain the relevant
    phenological stages of the crops in each tile, which can improve the performance of downstream embedding and inference tasks.
   
NOTES ON JOB DATABASE REUSE
---------------------------
The job database lives in ``--output_folder`` as ``job_tracking.csv`` and is
reused if it exists. To regenerate temporal extent or grid layout, use a new
``--output_folder`` or remove the existing job DB and production grid files.

OPTIONAL PARAMETERS
-------------------
* ``--s1_orbit_state``: Limit Sentinel-1 inputs to ``ASCENDING`` or ``DESCENDING``.
    By default, the workflow automatically determines the most appropriate orbit state per tile
    based on data availability.
    
* ``--compositing_window``: Temporal compositing window (``month`` or ``dekad``).
    Note that WorldCereal models are ALWAYS trained on monthly composites.
    
* ``--parallel_jobs``: Max number of concurrent openEO jobs to submit.
    Note that on CDSE free tier, the maximum number of concurrent batch jobs is liimited to 2.
    
* ``--restart_failed``: Restart jobs with ``error`` or ``start_failed`` status.
    Only relevant if you are reusing an existing job database. By default, failed jobs are not restarted.
    
* ``--randomize_jobs``: Shuffle job order before submission.

* ``--poll_sleep``: Seconds to wait between status updates.

* ``--simplify_logging``: Use a compact CLI status callback and suppress openEO logs.

Retrying options for processing job submission and execution:
    * ``--max_retries``: Max number of submission retries.
    * ``--base_delay``: Initial retry delay in seconds.
    * ``--max_delay``: Maximum retry delay in seconds.

Custom OpenEO job options. 
    Only change these if you know what you are doing and need to deviate from the default CDSE batch job configuration:
    * ``--driver_memory``: Driver memory setting passed as job option.
    * ``--driver_memoryOverhead``: Driver memory overhead passed as job option.
    * ``--executor_cores``: Executor cores passed as job option.
    * ``--executor_memory``: Executor memory passed as job option.
    * ``--executor_memoryOverhead``: Executor memory overhead passed as job option.
    * ``--max_executors``: Max executors passed as job option.
    * ``--image_name``: openEO image name override passed as job option.
    * ``--organization_id``: Organization id passed as job option.

USAGE EXAMPLE
-------------
    python scripts/inference/collect_inputs.py \
            --grid_path ./data/aoi_grid.geojson \
            --grid_size 20 \
            --output_folder ./outputs/inputs \
            --start_date 2024-01-01 \
            --end_date 2024-12-31 \
            
See also run_collect_inputs.sh located in the same folder for an example 
of a bash script that runs this Python script with a specific set of parameters.
"""

import argparse
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import geopandas as gpd
from openeo_gfmap import Backend, BackendContext, TemporalContext

from worldcereal.jobmanager import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    collect_worldcereal_inputs,
)
from worldcereal.utils import parse_job_options_from_args


def main(
    aoi_gdf: gpd.GeoDataFrame,
    output_folder: Path,
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
) -> None:
    """Collect preprocessed inputs for AOI patches.

    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame
        Areas of interest to process.
    output_folder : Path
        The folder where to store the extracted data.
    grid_size : Optional[int]
        Tile size in kilometers for splitting AOIs.
    temporal_extent : TemporalContext
        Temporal extent to use when creating the job database.
    year : Optional[int]
        Year used for crop calendar inference when temporal_extent is missing.
    compositing_window : Literal['month', 'dekad'], optional
        The compositing window to use for the inputs, by default "month".
    s1_orbit_state : Literal['ASCENDING', 'DESCENDING'], optional
        If specified, only Sentinel-1 data from the given orbit state will be used.
    parallel_jobs : int, optional
        The maximum number of parallel jobs to run at the same time, by default 10.
    randomize_jobs : bool, optional
        Randomize the order of jobs before submitting them, by default False.
    restart_failed : bool, optional
        Restart the jobs that previously failed, by default False.
    job_options : dict, optional
        A dictionary of job options to customize the jobs.
    poll_sleep : int, optional
        Seconds to wait between job status updates, by default 60.
    simplify_logging : bool, optional
        Use the CLI status callback and suppress native openEO job manager logs.
    max_retries : int, optional
        Max number of job submission retries, by default DEFAULT_MAX_RETRIES.
    base_delay : float, optional
        Initial retry delay in seconds, by default DEFAULT_BASE_DELAY.
    max_delay : float, optional
        Maximum retry delay in seconds, by default DEFAULT_MAX_DELAY.
    """

    collect_worldcereal_inputs(
        aoi_gdf=aoi_gdf,
        output_dir=output_folder,
        grid_size=grid_size,
        temporal_extent=temporal_extent,
        year=year,
        compositing_window=compositing_window,
        s1_orbit_state=s1_orbit_state,
        parallel_jobs=parallel_jobs,
        randomize_jobs=randomize_jobs,
        restart_failed=restart_failed,
        job_options=job_options,
        poll_sleep=poll_sleep,
        simplify_logging=simplify_logging,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backend_context=BackendContext(Backend.CDSE),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect preprocessed inputs for polygon patches."
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
        "--start_date",
        type=str,
        help="Start date for the extractions in 'YYYY-MM-DD' format",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for the extractions in 'YYYY-MM-DD' format",
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
        help="The folder where to store the extracted data",
    )
    parser.add_argument(
        "--s1_orbit_state",
        type=str,
        choices=["ASCENDING", "DESCENDING"],
        help="Specify the S1 orbit state to use for the jobs.",
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=2,
        help="The maximum number of parallel jobs to run at the same time.",
    )
    parser.add_argument(
        "--compositing_window",
        type=str,
        choices=["month", "dekad"],
        default="month",
        help="The compositing window to use for the inputs.",
    )
    parser.add_argument(
        "--restart_failed",
        action="store_true",
        help="Restart the jobs that previously failed.",
    )
    parser.add_argument(
        "--randomize_jobs",
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
            "Use a compact CLI status callback and suppress openEO job manager logs."
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
    parser.add_argument("--driver_memory", type=str, default=None, help="Driver memory")
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
        default=None,
        help="openEO image name.",
    )
    parser.add_argument(
        "--organization_id", type=int, default=None, help="Organization id."
    )

    args = parser.parse_args()

    aoi_gdf = None
    grid_size = args.grid_size
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

    job_options = parse_job_options_from_args(args)

    main(
        output_folder=args.output_folder,
        aoi_gdf=aoi_gdf,
        grid_size=grid_size,
        temporal_extent=temporal_extent,
        year=args.year,
        s1_orbit_state=args.s1_orbit_state,
        parallel_jobs=args.parallel_jobs,
        restart_failed=args.restart_failed,
        randomize_jobs=args.randomize_jobs,
        job_options=job_options,
        compositing_window=args.compositing_window,
        poll_sleep=args.poll_sleep,
        simplify_logging=args.simplify_logging,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        max_delay=args.max_delay,
    )
