import json
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import openeo
import pandas as pd
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import MultiBackendJobManager
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import cdse_connection

from worldcereal.job import create_inputs_process_graph

MAX_RETRIES = 50
BASE_DELAY = 0.1  # initial delay in seconds
MAX_DELAY = 10


class InferenceJobManager(MultiBackendJobManager):
    def on_job_done(self, job: BatchJob, row):
        logger.info(f"Job {job.job_id} completed")
        output_dir = generate_output_path_inference(self._root_dir, 0, row)

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


def create_worldcereal_inputsjob(
    row: pd.Series,
    connection: openeo.Connection,
    provider,
    connection_provider,
    s1_orbit_state: Literal["ASCENDING", "DESCENDING"] | None,
):
    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)
    spatial_extent = BoundingBoxExtent(*row.geometry.bounds, epsg=int(row["epsg"]))

    preprocessed_inputs = create_inputs_process_graph(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        s1_orbit_state=s1_orbit_state,
        target_epsg=int(row["epsg"]),
    )

    # Submit the job
    job_options = {
        "driver-memory": "4g",
        "executor-memory": "2g",
        "executor-memoryOverhead": "1g",
        # "etl_organization_id": 10523,
        "python-memory": "4g",
        "soft-errors": 0.1,
        "image-name": "python311",
        "max-executors": 10,
    }

    return preprocessed_inputs.create_job(
        title=f"WorldCereal collect inputs for {row.tile_name}",
        job_options=job_options,
    )


def generate_output_path_inference(
    root_folder: Path,
    geometry_index: int,
    row: pd.Series,
    asset_id: Optional[str] = None,
) -> Path:
    """Method to generate the output path for inference jobs.

    Parameters
    ----------
    root_folder : Path
        root folder where the output parquet file will be saved
    geometry_index : int
        For point extractions, only one asset (a geoparquet file) is generated per job.
        Therefore geometry_index is always 0. It has to be included in the function signature
        to be compatible with the GFMapJobManager
    row : pd.Series
        the current job row from the GFMapJobManager
    asset_id : str, optional
        Needed for compatibility with GFMapJobManager but not used.

    Returns
    -------
    Path
        output path for the point extractions parquet file
    """

    tile_name = row.tile_name

    # Create the subfolder to store the output
    subfolder = root_folder / str(tile_name)
    subfolder.mkdir(parents=True, exist_ok=True)

    return subfolder


if __name__ == "__main__":
    # ------------------------
    # Flexible parameters
    output_folder = Path("/vitodata/worldcereal/...")
    parallel_jobs = 20
    randomize_production_grid = (
        True  # If True, it will randomly select tiles from the production grid
    )
    debug = False  # Triggers a selection of tiles
    s1_orbit_state = (
        None  # If None, it will be automatically determined but we want it fixed here.
    )
    start_date = "2024-10-01"
    end_date = "2025-09-30"
    production_grid = "/vitodata/worldcereal/data/INSEASONPOC/france_inseason_poc_productionunits.parquet"
    restart_failed = True  # If True, it will restart failed jobs
    # ------------------------

    job_tracking_csv = output_folder / "job_tracking.csv"

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create a job dataframe if it does not exist
    print(job_tracking_csv)
    if job_tracking_csv.is_file():
        logger.info("Job tracking file already exists, skipping job creation.")
        job_df = pd.read_csv(job_tracking_csv)

        if restart_failed:
            logger.info("Resetting failed jobs.")
            job_df.loc[
                job_df["status"].isin(["error", "start_failed"]),
                "status",
            ] = "not_started"

            # Save new job tracking dataframe
            job_df.to_csv(job_tracking_csv, index=False)

    else:
        logger.info("Job tracking file does not exist, creating new jobs.")

        production_gdf = gpd.read_parquet(production_grid).rename(
            columns={"name": "tile_name"}
        )
        if debug:
            logger.info("Running in debug mode, selecting a subset of tiles.")
            # Select a subset of tiles for debugging
            # This is just an example selection, adjust as needed
            selection = ["E382N290", "E412N272", "E370N226", "E364N274", "E338N288"]
            production_gdf = production_gdf[production_gdf["tile_name"].isin(selection)]

        if randomize_production_grid:
            logger.info("Randomizing the production grid tiles.")
            production_gdf = production_gdf.sample(frac=1).reset_index(drop=True)

        job_df = production_gdf[["tile_name", "geometry"]].copy()
        job_df["start_date"] = start_date
        job_df["end_date"] = end_date

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
                df=job_df,
                start_job=partial(
                    create_worldcereal_inputsjob,
                    s1_orbit_state=s1_orbit_state,
                ),
                job_db=job_tracking_csv,
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
