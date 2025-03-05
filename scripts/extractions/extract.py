"""Main script to perform extractions. Each collection has it's specifities and
own functions, but the setup and main thread execution is done here."""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import geopandas as gpd
import pandas as pd
import requests
from openeo.rest import OpenEoApiError, OpenEoApiPlainError, OpenEoRestError
from openeo_gfmap import Backend
from openeo_gfmap.manager.job_manager import GFMAPJobManager

from worldcereal.extract.common import (
    merge_extraction_jobs,
    pipeline_log,
    prepare_extraction_jobs,
)
from worldcereal.stac.constants import ExtractionCollection

# Pushover API endpoint, allowing to send notifications to personal devices.
PUSHOVER_API_ENDPOINT = "https://api.pushover.net/1/messages.json"


def send_notification(message: str, title: str = "OpenEO-GFMAP") -> None:
    """Send a notification to the user's device using the Pushover API.

    The environment needs to have the PUSHOVER_USER_TOKEN and PUSHOVER_APP_TOKEN
    variables setup.
    """
    user_token = os.getenv("PUSHOVER_USER_TOKEN")
    app_token = os.getenv("PUSHOVER_APP_TOKEN")

    if user_token is None or app_token is None:
        pipeline_log.warning("No pushover tokens found, skipping the notification.")
        return

    data = {
        "token": app_token,
        "user": user_token,
        "message": message,
        "title": title,
    }
    response = requests.post(PUSHOVER_API_ENDPOINT, data=data)

    if response.status_code != 200:
        pipeline_log.error("Error sending the notification: %s", response.text)


def manager_main_loop(
    manager: GFMAPJobManager,
    collection: ExtractionCollection,
    job_df: gpd.GeoDataFrame,
    datacube_fn: Callable,
    tracking_df_path: Path,
) -> None:
    """Main loop for the job manager, re-running it whenever an uncatched
    OpenEO exception occurs, and notifying the user through the Pushover API
    whenever the extraction start or an error occurs.
    """
    latest_exception_time = None
    exception_counter = 0

    while True:
        pipeline_log.info("Launching the jobs manager.")
        try:
            send_notification(
                title=f"WorldCereal Extraction {collection.value} - Started",
                message="Extractions have been started.",
            )
            manager.run_jobs(job_df, datacube_fn, tracking_df_path)
            return
        except (
            OpenEoApiPlainError,
            OpenEoApiError,
            OpenEoRestError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        ) as e:
            pipeline_log.exception("An error occurred during the extraction.\n%s", e)
            send_notification(
                title=f"WorldCereal Extraction {collection.value} - OpenEo Exception",
                message=f"An OpenEO/Artifactory error occurred during the extraction.\n{e}",
            )
            if latest_exception_time is None:
                latest_exception_time = pd.Timestamp.now()
                exception_counter += 1
            # 30 minutes between each exception
            elif (datetime.now() - latest_exception_time).seconds < 1800:
                exception_counter += 1
            else:
                latest_exception_time = None
                exception_counter = 0

            if exception_counter >= 3:
                pipeline_log.error(
                    "Too many OpenEO exceptions occurred in a short amount of time, stopping the extraction..."
                )
                send_notification(
                    title=f"WorldCereal Extraction {collection.value} - Failed",
                    message="Too many OpenEO exceptions occurred, stopping the extraction.",
                )
                raise e
        except Exception as e:
            pipeline_log.exception(
                "An unexpected error occurred during the extraction.\n%s", e
            )
            send_notification(
                title=f"WorldCereal Extraction {collection.value} - Failed",
                message=f"An unexpected error occurred during the extraction.\n{e}",
            )
            raise e


def run_extractions(
    collection: ExtractionCollection,
    output_folder: Path,
    samples_df_path: Path,
    max_locations_per_job: int = 500,
    memory: Optional[str] = None,
    python_memory: Optional[str] = None,
    max_executors: Optional[int] = None,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    extract_value: int = 1,
    backend=Backend.CDSE,
    write_stac_api: bool = False,
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
    max_locations_per_job : int, optional
        The maximum number of locations to extract per job, by default 500
    memory : str, optional
        Memory to allocate for the executor.
        If not specified, the default value is used, depending on type of collection.
    python_memory : str, optional
        Memory to allocate for the python processes as well as OrfeoToolbox in the executors,
        If not specified, the default value is used, depending on type of collection.
    max_executors : int, optional
        Number of executors to run.
        If not specified, the default value is used, depending on type of collection.
    parallel_jobs : int, optional
        The maximum number of parallel jobs to run at the same time, by default 10
    restart_failed : bool, optional
        Restart the jobs that previously failed, by default False
    extract_value : int, optional
        All samples with an "extract" value equal or larger than this one, will be extracted, by default 1
    backend : openeo_gfmap.Backend, optional
        cloud backend where to run the extractions, by default Backend.CDSE

    Raises
    ------
    ValueError
        _description_
    """

    # Compile custom job options
    custom_job_options: Dict[str, Union[str, int]] = {}
    if memory:
        custom_job_options["memory"] = memory
    if python_memory:
        custom_job_options["python_memory"] = python_memory
    if max_executors:
        custom_job_options["max_executors"] = max_executors

    # Prepare extraction jobs
    job_manager, job_df, datacube_fn, tracking_df_path = prepare_extraction_jobs(
        collection,
        output_folder,
        samples_df_path,
        max_locations_per_job=max_locations_per_job,
        custom_job_options=custom_job_options,
        parallel_jobs=parallel_jobs,
        restart_failed=restart_failed,
        extract_value=extract_value,
        backend=backend,
        write_stac_api=write_stac_api,
    )

    # Run the extraction jobs with push notifications
    manager_main_loop(job_manager, collection, job_df, datacube_fn, tracking_df_path)

    pipeline_log.info("Extraction completed successfully.")

    # Merge the extractions (for point jobs only)
    merge_extraction_jobs(collection, output_folder, samples_df_path)

    send_notification(
        title=f"WorldCereal Extraction {collection.value} - Completed",
        message="Extractions have been completed successfully.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from a collection")
    parser.add_argument(
        "collection",
        type=ExtractionCollection,
        choices=list(ExtractionCollection),
        help="The collection to extract",
    )
    parser.add_argument(
        "output_folder", type=Path, help="The folder where to store the extracted data"
    )
    parser.add_argument(
        "samples_df_path",
        type=Path,
        help="Path to the samples dataframe with the data to extract",
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=500,
        help="The maximum number of locations to extract per job",
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
        "--extract_value",
        type=int,
        default=1,
        help="The value of the `extract` flag to use in the dataframe.",
    )
    parser.add_argument(
        "--write_stac_api",
        type=bool,
        default=False,
        help="Flag to write S1 and S2 patch extraction results to STAC API or not.",
    )

    args = parser.parse_args()

    run_extractions(
        collection=args.collection,
        output_folder=args.output_folder,
        samples_df_path=args.samples_df_path,
        max_locations_per_job=args.max_locations,
        memory=args.memory,
        python_memory=args.python_memory,
        max_executors=args.max_executors,
        parallel_jobs=args.parallel_jobs,
        restart_failed=args.restart_failed,
        extract_value=args.extract_value,
        backend=Backend.CDSE,
        write_stac_api=args.write_stac_api,
    )
