"""Run WorldCereal cropland/croptype map production using the unified job manager.

This script prepares and submits openEO batch jobs that run the end-to-end
WorldCereal inference workflow for each grid tile. It creates (or reuses) a
production grid, resolves temporal and season specifications, then submits one
job per tile and downloads the resulting maps.

WORKFLOW DEMO
--------------
See the notebook workflow in worldcereal-classification/notebooks/worldcereal_custom_croptype.ipynb
for an interactive version that uses the same underlying job manager logic.

PROCESSING CHAIN SUMMARY
------------------------
1) Read AOI grid from ``--grid_path``.
2) Optionally tile AOIs into a production grid using ``--grid_size``.
3) Enrich tiles with per-tile UTM geometry and EPSG codes.
4) Resolve temporal extent and seasonal windows (see "Temporal and seasons").
5) Initialize or resume a job database in ``output_folder``.
6) Submit openEO jobs (one per tile), track status, and download results.

REQUIRED PARAMETERS
-------------------
* ``--grid_path``: AOI file (.parquet, .geoparquet, .gpkg, .shp).
* ``product``: ``cropland`` or ``croptype``.
    -- the cropland workflow only generates a cropland product
    -- the croptype workflow generates both cropland and croptype products.
* ``output_folder``: output folder for the job DB and results.

SPATIAL EXTENT OPTIONS
----------------------
* OPTION 1: Use an existing production grid.
    - Provide a grid file with polygons and a unique ``tile_name`` column.
    - Do not specify ``--grid_size``; polygons are used as-is.

    Note that we highly recommend your individual grid tiles to be smaller
    than 50x50 km to avoid memory issues and long runtimes.
    In case of larger/irregular AOI's, we recommend OPTION 2.
    
* OPTION 2: Create a production grid from larger AOIs.
    - Provide a geometry file with polygons and a unique "id" column as --grid_path.
    - Set ``--grid_size`` (km) to tile each AOI into smaller patches.

TEMPORAL AND SEASONS
--------------------
This workflow requires a temporal extent and season specifications, which can
be resolved from multiple sources.

* OPTION 1: Use tile-specific temporal and season specifications in the grid file.
    - Include "start_date" and "end_date" columns in the grid file for per-tile temporal extents.
    - Include season-specific start and end dates as columns in the grid file.
        (e.g. "season_start_s1", "season_end_s1", "season_start_s2", "season_end_s2")
        You cannot specify more than 2 seasons per tile to be processed.
        In the case of the example above, the two seasons would be "s1" and "s2" with their respective start and end dates.
    Both str and datetime formats are accepted for date columns.
    
    If these columns are present in your grid file, they will take precedence over any global temporal/season specifications.

* OPTION 2: Provide explicit temporal and season specifications to be used for each tile.
    - Provide a global temporal extent using ``--start_date`` and ``--end_date`` arguments.
    - Provide season specifications using ``--season-specifications-json``.
        e.g. '{"s1": ["2024-03-01", "2024-08-31"], "s2": ["2024-09-01", "2025-02-28"]}'

* OPTION 3: Rely on crop calendar inference for temporal and season specifications.
    This is the default fallback when OPTIONS 1 and 2 are not applicable.
    We automatically infer for each tile the two dominant crop seasons from the global WorldCereal 
    crop calendars based on the tile location and the provided ``--year``.
    Then we derive a 12 month temporal extent per tile that covers both seasons and set it as the tile's temporal extent.

MODEL SELECTION
---------------
The workflow uses a seasonal Presto model for inference, with optionally a customized cropland/landcover head and/or crop type head.
By default, the latest available models are used from the WorldCereal model registry, but you can also specify custom model artifacts.

This can be done in two ways:

    * OPTION 1: Use the same models for all tiles of the production grid
        --> just make sure to specify the desired model override parameters
            (--seasonal-model-zip, --landcover-head-zip, --croptype-head-zip) when running the script.
    
    * OPTION 2: Specify different models per tile in the grid file
        --> include columns in the grid file with the URLs to the desired model artifacts for each tile.
            For example, you could include a "seasonal_model_zip" column with the URL to the seasonal model 
            .zip file to use for each tile, and the workflow will use those instead of the default or globally overridden models.
    
OPTIONAL PARAMETERS
-------------------
* ``--target-epsg``: Output reprojection EPSG code (projection of the output maps).

* ``--s1_orbit_state``: Force Sentinel-1 orbit state (ASCENDING/DESCENDING).
    By default, the workflow automatically determines the most appropriate orbit state per tile
    based on data availability.

* ``--seasonal-preset``: Seasonal workflow preset name.
    ONLY CHANGE THIS IF YOU KNOW WHAT YOU ARE DOING.
    This determines the set of default settings for the inference workflow and should
    be parameterized in worldcereal.openeo.parameters.py.

* ``--seasonal-model-zip``: Override seasonal model artifact.
    Should be public URL to a .zip file containing a seasonal Presto model.

* ``--enable/disable-cropland-head``: Override cropland head enablement.
    Determines whether the seasonal model's landcover head will be activated for cropland product generation.
    By default, this follows the settings of the seasonal preset (activated by default).

* ``--enable/disable-croptype-head``: Override croptype head enablement.
    Determines whether the seasonal model's croptype head will be activated for croptype product generation.
    By default, this follows the settings of the seasonal preset
    --> activated for croptype product and deactivated for cropland product.
    
* ``--landcover-head-zip`` / ``--croptype-head-zip``: Override head artifacts.
        Possibility to supply custom head models for the seasonal workflow. 
        Should be public URLs to .zip files containing the respective head models.
        
* ``--enforce/disable-cropland-gate``: Override cropland gate for croptype.
        Determines whether the crop type output will be masked using the cropland product.
        If enabled, non-cropland areas will be masked out in the croptype outputs.
        By default, this follows the settings of the seasonal preset
        --> enabled for croptype product and disabled for cropland product.
        
* ``--class-probabilities``: Export per-class probabilities.
    Whether or not to also export the per-class probabilities in addition to the class predictions.
    This will create additional probability layers in the output products and significantly increase the output file sizes.

Postprocessing options:
    Here you have the possibility to specify post-processing settings for cropland and crop type products separately.
    By default, post-processing is disabled for both products, but it can be enabled and configured using the following parameters:
    
    * ``--enable-cropland-postprocess``: Enable postprocessing for cropland products.
    * ``--cropland-postprocess-method``: Cropland postprocess method override.
        Supported methods are "majority_vote" and "smooth_probabilities".
        The first applies a majority vote filter to the cropland predictions,
        while the second smooths the probability maps, resulting in a less severe cleaning.
    * ``--cropland-postprocess-kernel-size``: Kernel size override for cropland postprocessing methods.
        This determines the size of the neighborhood considered for postprocessing.
         For majority vote, this is the size of the moving window to determine the majority class.
         Not used for smooth_probabilities, which uses a fixed smoothing kernel.
         Default is 5, which corresponds to a 5x5 pixel neighborhood.
         Increasing this will result in stronger smoothing/cleaning.
    * ``--enable-croptype-postprocess``: Enable postprocessing for croptype products.
    * ``--croptype-postprocess-method``: Croptype postprocess method override.
        See higher for supported methods.
    * ``--croptype-postprocess-kernel-size``: Croptype postprocess kernel size override.
        See higher for details.

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
    python scripts/inference/cropland_croptype_mapping.py \
        --grid_path ./bbox/test.gpkg \
        --grid_size 20 \
        --output_folder ./outputs/maps \
        --product croptype \
        --start_date 2024-01-01 \
        --end_date 2024-12-31 \
        --season-specifications-json '{"s1": ["2024-03-01", "2024-08-31"], "s2": {"start_date": "2024-09-01", "end_date": "2025-02-28"}}' \

See also run_cropland_croptype_mapping.sh located in the same folder for an example 
of a bash script that runs this Python script with a specific set of parameters.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import geopandas as gpd
from openeo_gfmap import TemporalContext
from openeo_gfmap.backend import Backend, BackendContext

from worldcereal.jobmanager import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    run_map_production,
)
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.parameters import WorldCerealProductType
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


def main(
    aoi_gdf: gpd.GeoDataFrame,
    output_folder: Path,
    product: WorldCerealProductType,
    grid_size: int = 20,
    temporal_extent: Optional[TemporalContext] = None,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    target_epsg: Optional[int] = None,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    year: Optional[int] = None,
    season_specifications: Optional[Dict[str, TemporalContext]] = None,
    parallel_jobs: int = 2,
    seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
    restart_failed: bool = False,
    randomize_jobs: bool = False,
    job_options: Optional[dict] = None,
    poll_sleep: int = 60,
    simplify_logging: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    export_class_probs: Optional[bool] = None,
    seasonal_model_zip: Optional[str] = None,
    enable_cropland_head: Optional[bool] = None,
    landcover_head_zip: Optional[str] = None,
    enable_croptype_head: Optional[bool] = None,
    croptype_head_zip: Optional[str] = None,
    enforce_cropland_gate: Optional[bool] = None,
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
) -> None:
    run_map_production(
        aoi_gdf=aoi_gdf,
        output_dir=output_folder,
        grid_size=grid_size,
        temporal_extent=temporal_extent,
        season_specifications=season_specifications,
        year=year,
        product_type=product,
        seasonal_preset=seasonal_preset,
        export_class_probs=export_class_probs,
        seasonal_model_zip=seasonal_model_zip,
        enable_cropland_head=enable_cropland_head,
        landcover_head_zip=landcover_head_zip,
        enable_croptype_head=enable_croptype_head,
        croptype_head_zip=croptype_head_zip,
        enforce_cropland_gate=enforce_cropland_gate,
        enable_cropland_postprocess=enable_cropland_postprocess,
        cropland_postprocess_method=cropland_postprocess_method,
        cropland_postprocess_kernel_size=cropland_postprocess_kernel_size,
        enable_croptype_postprocess=enable_croptype_postprocess,
        croptype_postprocess_method=croptype_postprocess_method,
        croptype_postprocess_kernel_size=croptype_postprocess_kernel_size,
        target_epsg=target_epsg,
        backend_context=backend_context,
        s1_orbit_state=s1_orbit_state,
        restart_failed=restart_failed,
        job_options=job_options,
        parallel_jobs=parallel_jobs,
        randomize_jobs=randomize_jobs,
        poll_sleep=poll_sleep,
        simplify_logging=simplify_logging,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )


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

    season_specifications = _load_season_specifications(args)

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
        seasonal_model_zip=args.seasonal_model_zip,
        enable_cropland_head=enable_cropland_head,
        landcover_head_zip=args.landcover_head_zip,
        enable_croptype_head=enable_croptype_head,
        croptype_head_zip=args.croptype_head_zip,
        enforce_cropland_gate=enforce_cropland_gate,
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
        export_class_probs=args.class_probabilities,
        enable_cropland_postprocess=args.enable_cropland_postprocess,
        cropland_postprocess_method=args.cropland_postprocess_method,
        cropland_postprocess_kernel_size=args.cropland_postprocess_kernel_size,
        enable_croptype_postprocess=args.enable_croptype_postprocess,
        croptype_postprocess_method=args.croptype_postprocess_method,
        croptype_postprocess_kernel_size=args.croptype_postprocess_kernel_size,
    )
