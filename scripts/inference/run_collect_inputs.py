import argparse
import json
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import geopandas as gpd
import openeo
import pandas as pd
import shapely
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import CsvJobDatabase, MultiBackendJobManager
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import cdse_connection

from worldcereal.job import create_inputs_process_graph
from worldcereal.utils import parse_job_options_from_args
from worldcereal.utils.production_grid import convert_gdf_to_utm_grid

MAX_RETRIES = 50
BASE_DELAY = 0.1  # initial delay in seconds
MAX_DELAY = 10

REQUIRED_ATTRIBUTES = ["tile_name", "geometry_utm_wkt", "epsg_utm"]

class InferenceJobManager(MultiBackendJobManager):
    def on_job_done(self, job: BatchJob, row: pd.Series) -> None:
        logger.info(f"Job {job.job_id} completed")
        output_dir = self.generate_output_path(row)

        # Get job results
        job_result = job.get_results()

        # Get metadata
        job_metadata = job.describe()
        result_metadata = job_result.get_metadata()
        job_metadata_path = output_dir / f"job_{job.job_id}.json"
        result_metadata_path = output_dir / f"result_{job.job_id}.json"

        # Get the products
        assets = job_result.get_assets()
        for asset in assets:
            filepath = asset.download(target=output_dir)

            # We want to add the tile name to the filename
            new_filename = f"{filepath.stem}_{row.tile_name}.nc"
            new_filepath = filepath.parent / new_filename

            shutil.move(filepath, new_filepath)

        with job_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(job_metadata, f, ensure_ascii=False)
        with result_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(result_metadata, f, ensure_ascii=False)

        logger.success("Job completed")

    def generate_output_path(
        self,
        row: pd.Series,
    ) -> Path:
        tile_name = row.tile_name
        subfolder = self._root_dir / str(tile_name)
        subfolder.mkdir(parents=True, exist_ok=True)
        return subfolder


def create_worldcereal_inputsjob(
    row: pd.Series,
    connection: openeo.Connection,
    provider,
    connection_provider,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    job_options: Optional[dict] = None,
    compositing_window: Literal["month", "dekad"] = "month",
):
    """Function to create a job for collecting preprocessed inputs for WorldCereal.
    Parameters
    ----------
    row : pd.Series
        A row from the job dataframe containing the parameters for the job.
    connection : openeo.Connection
        An active openEO connection.
    provider : str
        The name of the provider to use for the job.
    connection_provider : str
        The name of the provider to use for the connection.
    s1_orbit_state : Literal['ASCENDING', 'DESCENDING'], optional
        If specified, only Sentinel-1 data from the given orbit state will be used.
        If None, it will be automatically determined.
    job_options : dict, optional
        A dictionary of job options to customize the job.
        If None, default options will be used.
    Returns
    -------
    openeo.BatchJob
        The created batch job.
    """
    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)
    bounds = shapely.from_wkt(row.geometry_utm_wkt).bounds
    rounded_bounds = tuple(round(coord / 20) * 20 for coord in bounds)
    spatial_extent = BoundingBoxExtent(*rounded_bounds, epsg=int(row["epsg_utm"]))
    # spatial_extent = BoundingBoxExtent(*row.geometry.bounds, epsg=int(row["epsg"]))

    preprocessed_inputs = create_inputs_process_graph(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        s1_orbit_state=s1_orbit_state,
        target_epsg=int(row["epsg_utm"]),
        compositing_window=compositing_window,
    )

    # If no custom job options are provided, use these defaults
    if not job_options:
        job_options = {
            "driver-memory": "4g",
            "executor-memory": "2g",
            "executor-memoryOverhead": "1g",
            "python-memory": "4g",
            "soft-errors": 0.1,
            "max-executors": 10,
        }

    return preprocessed_inputs.create_job(
        title=f"WorldCereal collect inputs for {row.tile_name}",
        job_options=job_options,
    )


def create_job_dataframe_from_grid(
    grid_path: Path,
    start_date: str,
    end_date: str,
    output_folder: Path,
    tile_name_col: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Function to create a job dataframe from a grid file.
    Parameters
    ----------
    grid_path : Path
        Path to the geo file (any format that GeoPandas read_file() accepts + .(geo)parquet) defining the patches to extract.
        Must contain a geometry column with polygons in valid CRS.
    start_date : str
        Start date for the extractions in 'YYYY-MM-DD' format.
    end_date : str
        End date for the extractions in 'YYYY-MM-DD' format.
    output_folder : Path
        The folder where to store the extracted data
    tile_name_col : str, optional
        Name of the column in the grid file that contains the tile names.
        If None, tiles will be named as "patch_0", "patch_1",  etc.
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with production grid prepared to be transformed into jobs.
    """
    assert grid_path is not None, "grid_path must be provided when overwrite is True."
    assert Path(grid_path).is_file(), f"The grid file {grid_path} does not exist."
    if str(grid_path).endswith("parquet") or str(grid_path).endswith("geoparquet"):
        production_gdf = gpd.read_parquet(grid_path)
    else:
        production_gdf = gpd.read_file(grid_path)

    # Check if all required attributes are present in the production_gdf
    missing_attributes = [
        attr for attr in REQUIRED_ATTRIBUTES if attr not in production_gdf.columns
    ]
    if missing_attributes:
        raise ValueError(
            f"The following required attributes are missing in the production grid: {missing_attributes}"
        )
        
    assert (
        "geometry" in production_gdf.columns
    ), "The grid file must contain a geometry column."
    assert all(
        production_gdf.geometry.type == "Polygon"
    ), "All geometries in the grid file must be of type Polygon."
    assert production_gdf.crs is not None, "The grid file must have a defined CRS."
    assert (
        production_gdf.crs.to_epsg() is not None
    ), "The grid file must have a defined EPSG code."
    assert pd.to_datetime(start_date) < pd.to_datetime(
        end_date
    ), "The start_date must be earlier than the end_date."

    production_gdf["start_date"] = start_date
    production_gdf["end_date"] = end_date
    if not tile_name_col:
        production_gdf["tile_name"] = [f"patch_{i}" for i in range(len(production_gdf))]
    else:
        production_gdf["tile_name"] = production_gdf[tile_name_col]
    # production_gdf["epsg"] = production_gdf.crs.to_epsg()

    return production_gdf


def create_job_database(
    output_folder: Path,
    overwrite: bool = False,
    grid_path: Optional[Path] = None,
    extractions_start_date: Optional[str] = None,
    extractions_end_date: Optional[str] = None,
    tile_name_col: Optional[str] = None,
) -> CsvJobDatabase:
    job_tracking_path = Path(output_folder) / "job_tracking.csv"
    job_db = CsvJobDatabase(path=job_tracking_path)
    if job_db.exists():
        if overwrite:
            if (
                grid_path is None
                or extractions_start_date is None
                or extractions_end_date is None
            ):
                raise ValueError(
                    "grid_path, extractions_start_date and extractions_end_date must be provided when overwriting."
                )
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = job_tracking_path.with_suffix(f".csv.bckp.{ts}")
            job_tracking_path.replace(backup)
            logger.info(f"Backed up old job DB to {backup}")
            logger.info(f"Creating new job tracking file at {job_tracking_path}.")
            production_gdf = create_job_dataframe_from_grid(
                grid_path=grid_path,
                start_date=extractions_start_date,
                end_date=extractions_end_date,
                output_folder=output_folder,
                tile_name_col=tile_name_col,
            )
            job_db.initialize_from_df(production_gdf)
    else:
        if overwrite:
            raise ValueError("Cannot back up a non-existing job database")
        else:
            if (
                grid_path is None
                or extractions_start_date is None
                or extractions_end_date is None
            ):
                raise ValueError(
                    "grid_path, extractions_start_date and extractions_end_date must be provided when creating new job database."
                )
            logger.info(f"Creating new job tracking file at {job_tracking_path}.")
            production_gdf = create_job_dataframe_from_grid(
                grid_path=grid_path,
                start_date=extractions_start_date,
                end_date=extractions_end_date,
                output_folder=output_folder,
                tile_name_col=tile_name_col,
            )
            job_db.initialize_from_df(production_gdf)

    return job_db


def main(
    output_folder: Path,
    overwrite_job_df: bool = False,
    grid_path: Optional[Path] = None,
    extractions_start_date: Optional[str] = None,
    extractions_end_date: Optional[str] = None,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    job_options: Optional[Dict[str, Union[str, int]]] = None,
    compositing_window: Literal["month", "dekad"] = "month",
    tile_name_col: Optional[str] = None,
) -> None:
    """Main function responsible for creating and launching jobs to collect preprocessed inputs.

    Parameters
    ----------
    grid_path : Path
        Path to the geo file (any format that GeoPandas read_file() accepts + .(geo)parquet) defining the patches to extract.
        Must contain a geometry column with polygons in valid CRS.
    extractions_start_date : str
        Start date for the extractions in 'YYYY-MM-DD' format.
    extractions_end_date : str
        End date for the extractions in 'YYYY-MM-DD' format.
    output_folder : Path
        The folder where to store the extracted data
    overwrite_job_df : bool, optional
        Whether to overwrite the existing job tracking dataframe if it exists. Default is False.
    s1_orbit_state : Literal['ASCENDING', 'DESCENDING'], optional
        If specified, only Sentinel-1 data from the given orbit state will be used.
        If None, it will be automatically determined but we want it fixed here.
    parallel_jobs : int, optional
        The maximum number of parallel jobs to run at the same time, by default 10
    restart_failed : bool, optional
        Restart the jobs that previously failed, by default False
    job_options : dict, optional
        A dictionary of job options to customize the jobs.
        If None, default options will be used.
        Recognized keys:
            executor-memory, python-memory, max-executors, image-name, etl_organization_id.
            See worldcereal.utils.argparser.DEFAULT_JOB_OPTIONS for defaults.
    compositing_window : Literal['month', 'dekad'], optional
        The compositing window to use for the inputs, by default "month".

    Returns
    -------
    None
    """

    job_tracking_path = output_folder / "job_tracking.csv"
    if job_tracking_path.is_file() and not overwrite_job_df:
        job_df = pd.read_csv(job_tracking_path)
        if restart_failed:
            logger.info("Resetting failed jobs.")
            job_df.loc[
                job_df["status"].isin(["error", "start_failed"]),
                "status",
            ] = "not_started"
            # Save new job tracking dataframe
            job_df.to_csv(job_tracking_path, index=False)
        job_db = CsvJobDatabase(path=job_tracking_path)
    else:
        # Create a jobdb if it does not exist or if overwrite is True
        job_db = create_job_database(
            output_folder,
            overwrite=overwrite_job_df,
            grid_path=grid_path,
            extractions_start_date=extractions_start_date,
            extractions_end_date=extractions_end_date,
            tile_name_col=tile_name_col,
        )

    # Retry loop starts here
    attempt = 0
    while True:
        try:
            # Setup connection + manager
            connection = cdse_connection()
            logger.info("Setting up the job manager.")
            manager = InferenceJobManager(root_dir=output_folder)
            manager.add_backend(
                "cdse", connection=connection, parallel_jobs=parallel_jobs
            )

            # Kick off all jobs
            manager.run_jobs(
                start_job=partial(
                    create_worldcereal_inputsjob,
                    s1_orbit_state=s1_orbit_state,
                    job_options=job_options,
                    connection=connection,
                    compositing_window=compositing_window,
                ),
                job_db=job_db,
            )
            logger.info("All jobs submitted successfully.")
            break  # success: exit loop

        except Exception as exc:
            if attempt < MAX_RETRIES:
                attempt += 1
                # Exponential backoff with full jitter, capped at MAX_DELAY seconds
                backoff = min(BASE_DELAY * 2**attempt, MAX_DELAY)
                jitter = random.uniform(
                    -0.2 * backoff, 0.2 * backoff
                )  # ±20% of backoff
                delay = max(0, backoff + jitter)
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue
            # Non-retryable or maxed-out
            logger.error(f"Max retries reached. Last error: {exc}")
            raise

    logger.info("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect preprocessed inputs for polygon patches."
    )
    parser.add_argument(
        "--grid_path",
        type=Path,
        help="Path to the grid file (GeoJSON or shapefile) defining the locations to extract.",
    )
    parser.add_argument(
        "--extractions_start_date",
        type=str,
        help="Start date for the extractions in 'YYYY-MM-DD' format",
    )
    parser.add_argument(
        "--extractions_end_date",
        type=str,
        help="End date for the extractions in 'YYYY-MM-DD' format",
    )
    parser.add_argument(
        "--output_folder", type=Path, help="The folder where to store the extracted data"
    )
    parser.add_argument(
        "--overwrite_job_df",
        action="store_true",
        help="Whether to overwrite the existing job tracking dataframe if it exists.",
    )
    parser.add_argument(
        "--s1_orbit_state",
        type=str,
        choices=["ASCENDING", "DESCENDING"],
        help="Specify the S1 orbit state to use for the jobs.",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="1800m",
        help="Memory to allocate for the executor.",
    )
    parser.add_argument(
        "--python_memory",
        type=str,
        default="1900m",
        help="Memory to allocate for the python processes as well as OrfeoToolbox in the executors.",
    )
    parser.add_argument(
        "--max_executors", type=int, default=22, help="Number of executors to run."
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
        "--image_name",
        type=str,
        default=None,
        help="Specific openEO image name to use for the jobs.",
    )
    parser.add_argument(
        "--organization_id",
        type=int,
        default=None,
        help="ID of the organization to use for the job.",
    )
    parser.add_argument(
        "--tile_name_col",
        type=str,
        default=None,
        help="Name of the column in the grid file that contains the tile names. If not provided, tiles will be named as 'patch_0', 'patch_1', etc.",
    )
    parser.add_argument(
        "--compositing_window",
        type=str,
        choices=["month", "dekad"],
        default="month",
        help="The compositing window to use for the inputs.",
    )

    args = parser.parse_args()

    job_options = parse_job_options_from_args(args)

    utm_aware_grid_path = str(args.grid_path).split('.')[0] + "_utm_grid.parquet"
    if not Path(utm_aware_grid_path).is_file():
        convert_gdf_to_utm_grid(
            in_path = Path(args.grid_path),
            out_path = Path(utm_aware_grid_path),
            id_col = args.tile_name_col,
            web_mercator_grid = False
            )

    main(
        grid_path=utm_aware_grid_path,
        extractions_start_date=args.extractions_start_date,
        extractions_end_date=args.extractions_end_date,
        output_folder=args.output_folder,
        overwrite_job_df=args.overwrite_job_df,
        s1_orbit_state=args.s1_orbit_state,
        parallel_jobs=args.parallel_jobs,
        restart_failed=args.restart_failed,
        job_options=job_options,
        tile_name_col=args.tile_name_col,
        compositing_window=args.compositing_window,
    )
