"""Extraction script."""

from pathlib import Path
from typing import Dict, Optional, Union
from pathlib import Path
from openeo_gfmap import Backend

from openeo import Connection
from openeo.extra.job_management import CsvJobDatabase

from worldcereal.extract.job_manager import ExtractionJobManager
from worldcereal.stac.stac_handler import StacHandler
from worldcereal.stac.stac_api_interaction import upload_to_stac_api


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

STAC_ROOT_URL = "https://stac.openeo.vito.be/"

# Collection mapping stays here since it's STAC-specific
PATCH_COLLECTIONS = {
    "PATCH_SENTINEL1": "test_hv_worldcereal_sentinel_1_patch_extractions",
    "PATCH_SENTINEL2": "test_hv_worldcereal_sentinel_2_patch_extractions",
}


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
    """Main function responsible for launching point and patch extractions."""
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
        stac_root_url=STAC_ROOT_URL,  # Add this parameter
        patch_collections=PATCH_COLLECTIONS  # Add this parameter
    )

    # Setup extraction functions
    pipeline_log.info("Setting up extraction functions.")
    datacube_fn = setup_datacube_creation_fn(collection, job_options)
    path_fn = setup_output_path_fn(collection)
    
    # Update post_job_fn if it also needs STAC configuration
    post_job_fn = setup_post_job_fn(
        collection, 
        extract_value, 
        write_stac_api,
    )

    try:
        collection_id = PATCH_COLLECTIONS.get(collection.name)
        collection_description = f"Extractions for {collection_id} with ref_id {ref_id}"
    except Exception as e:
        pipeline_log.error(f"Error retrieving collection info: {e}")


    # Initialize STAC handler if needed
    stac_handler = StacHandler(
        output_dir=output_folder,
        collection_id=collection_id,
        collection_description=collection_description
    )

    # --- Initialize the job manager ---
    job_manager = ExtractionJobManager(
        output_dir=output_folder,
        output_path_generator=path_fn,
        post_job_action=post_job_fn,
        poll_sleep=60,
        stac_handler=stac_handler,  # Use the handler we created
    )

    job_manager.add_backend(
        "cdse", connection=connection, parallel_jobs=parallel_jobs
    )

    # --- Run the extraction jobs ---
    pipeline_log.info("Persisting the job tracking dataframe")
    job_db = CsvJobDatabase(path=tracking_df_path)
    job_df = job_manager._normalize_df(job_df)
    job_db.persist(job_df)

    pipeline_log.info("Running the extraction jobs.")
    job_manager.run_jobs(df=job_df, job_db=job_db, start_job=datacube_fn)
    pipeline_log.info("Extraction jobs completed.")
    
    # --- Merge the extraction jobs (for point extractions) ---
    if collection == ExtractionCollection.POINT_WORLDCEREAL:
        pipeline_log.info("Merging extraction jobs into final output.")
        merge_output_files_point_worldcereal(output_folder, ref_id)
        pipeline_log.info("Merging completed successfully.")

    # --- Handle STAC operations ---
    pipeline_log.info("Writing STAC collection.")
    stac_handler.write_stac()
    pipeline_log.info("STAC collection saved successfully.")

    if write_stac_api:
        job_items = stac_handler.get_all_items()  # Make sure this method exists

        pipeline_log.info("Uploading items to STAC API")
        upload_to_stac_api(job_items, collection_id=collection_id, stac_root_url=STAC_ROOT_URL)
        pipeline_log.info("Upload to STAC API completed successfully.")


