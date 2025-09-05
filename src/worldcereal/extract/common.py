"""Extraction script."""

import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from pathlib import Path
from openeo_gfmap import Backend

from openeo.extra.job_management import (
        CsvJobDatabase
    )
from openeo import Connection

from worldcereal.extract.job_manager import ExtractionJobManager
from worldcereal.stac.stac_handler import StacHandler
from worldcereal.utils.file_utils import ensure_dir

from worldcereal.extract.dataframe_utils import initialize_job_dataframe
from worldcereal.extract.function_factory import (
    setup_datacube_creation_fn,
    setup_output_path_fn,
    setup_post_job_fn,
)

from worldcereal.extract.point_worldcereal import merge_output_files_point_worldcereal
from worldcereal.extract.utils import pipeline_log
from worldcereal.stac.constants import ExtractionCollection


def run_extractions(
    collection: ExtractionCollection,
    output_folder: Path,
    samples_df_path: Path,
    ref_id: str,
    max_locations_per_job: int = 500,
    job_options: Optional[Dict[str, Union[str, int]]] = None,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    extract_value: int = 1,
    backend=Backend.CDSE,
    connection = Connection,
    write_stac_api: bool = False,
    check_existing_extractions: bool = False,
) -> None:
    """Main function responsible for launching point and patch extractions.

    Parameters
    ----------
    collection : ExtractionCollection
        The collection to extract. Most popular: PATCH_WORLDCEREAL, POINT_WORLDCEREAL
    output_folder : Path
        The folder where to store the extracted data
    samples_df_path : Path
        Path to the input dataframe containing the geometries
        for which extractions need to be done
    ref_id : str
        Official ref_id of the source dataset
    max_locations_per_job : int, optional
        The maximum number of locations to extract per job, by default 500
    job_options : Optional[Dict[str, Union[str, int]]], optional
        Custom job options to set for the extraction, by default None (default options)
        Options that can be set explicitly include:
            - memory : str
                Memory to allocate for the executor, e.g. "1800m"
            - python_memory : str
                Memory to allocate for the python processes as well as OrfeoToolbox in the executors, e.g. "1900m"
            - max_executors : int
                Number of executors to run, e.g. 22
    parallel_jobs : int, optional
        The maximum number of parallel jobs to run at the same time, by default 10
    restart_failed : bool, optional
        Restart the jobs that previously failed, by default False
    extract_value : int, optional
        All samples with an "extract" value equal or larger than this one, will be extracted, by default 1
    backend : openeo_gfmap.Backend, optional
        Cloud backend where to run the extractions, by default Backend.CDSE
    write_stac_api : bool, optional
        Save metadata of extractions to STAC API (requires authentication), by default False
    check_existing_extractions : bool, optional
        Check if the samples already exist in the STAC API and filter them out,
        by default False
    """
    pipeline_log.info("Starting the extractions workflow...")

    # --- Prepare the extraction jobs ---
    ensure_dir(output_folder)
    tracking_df_path = output_folder / "job_tracking.csv"

    # Load or create job dataframe
    pipeline_log.info("Loading or creating job dataframe.")
    job_df = initialize_job_dataframe(
        tracking_df_path,
        samples_df_path,
        collection,
        ref_id,
        max_locations_per_job,
        extract_value,
        backend,
        restart_failed,
        check_existing_extractions,
    )

    # Setup extraction functions
    pipeline_log.info("Setting up extraction functions.")
    datacube_fn = setup_datacube_creation_fn(collection, job_options)

    path_fn = setup_output_path_fn(collection)
    post_job_fn = setup_post_job_fn(collection, extract_value, write_stac_api)

    # Initialize job manager with STAC support if requested
    job_manager = _initialize_job_manager(
        output_folder,
        path_fn,
        post_job_fn,
        connection,
        parallel_jobs,
        write_stac_api=write_stac_api,
        collection_id=f"{ref_id}_extractions",
        collection_description=f"Extractions for {collection.name} with ref_id {ref_id}",
    )

    # --- Run the extraction jobs ---
    _run_extraction_jobs(
        job_manager=job_manager,
        job_df=job_df,
        datacube_fn=datacube_fn,
        tracking_df_path=tracking_df_path,
    )
    

    # --- Merge the extraction jobs (for point extractions) --- #TODO check how this works?
    if collection == ExtractionCollection.POINT_WORLDCEREAL:
        _merge_extraction_jobs(output_folder, ref_id)

    # --- Write STAC collection if enabled ---
    if write_stac_api:
        job_manager.write_stac()
        pipeline_log.info("STAC collection saved successfully.")



def _initialize_job_manager(
    output_folder: Path,
    path_fn: Callable,
    post_job_fn: Callable,
    connection: Connection,
    parallel_jobs: int = 2,
    write_stac_api: bool = False,
    collection_id: Optional[str] = None,
    collection_description: str = "",
) -> ExtractionJobManager:
    """Create and configure the extraction job manager with optional STAC support."""

    job_manager = ExtractionJobManager(
        output_dir=output_folder,
        output_path_generator=path_fn,
        post_job_action=post_job_fn,
        poll_sleep=60,
        stac_handler=StacHandler(output_folder, collection_id, collection_description) if write_stac_api else None,
    )

    job_manager.add_backend(
            "cdse", connection=connection, parallel_jobs=parallel_jobs
        )

    return job_manager


def _run_extraction_jobs(
    job_manager: ExtractionJobManager,
    job_df: pd.DataFrame,
    datacube_fn: Callable,
    tracking_df_path: Path,
) -> None:
    """Execute extraction jobs using the manager."""
    pipeline_log.info("Persisting dataframes.")

    job_db = CsvJobDatabase(path=tracking_df_path)
    job_df = job_manager._normalize_df(job_df)
    job_db.persist(job_df)
    pipeline_log.info("Running the extraction jobs.")
    
    job_manager.run_jobs(df=job_df, job_db=job_db, start_job=datacube_fn)
            
    pipeline_log.info("Extraction jobs completed.")


def _merge_extraction_jobs(output_folder: Path, ref_id: str) -> None:
    """Merge all extraction results into a single partitioned GeoParquet file."""
    pipeline_log.info("Merging extraction jobs into final output.")
    merge_output_files_point_worldcereal(output_folder, ref_id)
    pipeline_log.info("Merging completed successfully.")

