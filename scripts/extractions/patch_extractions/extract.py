"""Main script to perform extractions. Each collection has it's specifities and
own functions, but the setup and main thread execution is done here."""

import argparse
import os
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from extract_common import generate_output_path, pipeline_log, post_job_action
from extract_meteo import create_datacube_meteo, create_job_dataframe_meteo, meteo_asset
from extract_optical import (
    create_datacube_optical,
    create_job_dataframe_s2,
    sentinel2_asset,
)
from openeo.rest import OpenEoApiError, OpenEoApiPlainError, OpenEoRestError
from openeo_gfmap import Backend
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import load_s2_grid, split_job_s2grid

from extract_sar import (  # isort: skip
    create_datacube_sar,
    create_job_dataframe_s1,
    sentinel1_asset,
)


class ExtractionCollection(Enum):
    """Collections that can be extracted in those scripts."""

    SENTINEL1 = "SENTINEL1"
    SENTINEL2 = "SENTINEL2"
    METEO = "METEO"


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


def load_dataframe(df_path: Path) -> gpd.GeoDataFrame:
    """Load the input dataframe from the given path."""
    pipeline_log.info("Loading input dataframe from %s.", df_path)

    if df_path.name.endswith(".geoparquet"):
        return gpd.read_parquet(df_path)
    else:
        return gpd.read_file(df_path)


def prepare_job_dataframe(
    input_df: gpd.GeoDataFrame,
    collection: ExtractionCollection,
    max_locations: int,
    extract_value: int,
    backend: Backend,
) -> gpd.GeoDataFrame:
    """Prepare the job dataframe to extract the data from the given input
    dataframe."""
    pipeline_log.info("Preparing the job dataframe.")

    # Filter the input dataframe to only keep the locations to extract
    input_df = input_df[input_df["extract"] == extract_value].copy()

    # Split the locations into chunks of max_locations
    split_dfs = []
    pipeline_log.info(
        "Performing splitting by the year...",
    )
    input_df["valid_time"] = pd.to_datetime(input_df.valid_time)
    input_df["year"] = input_df.valid_time.dt.year

    split_dfs_time = [group.reset_index() for _, group in input_df.groupby("year")]
    pipeline_log.info("Performing splitting by s2 grid...")
    for df in split_dfs_time:
        s2_split_df = split_job_s2grid(df, max_points=max_locations)
        split_dfs.extend(s2_split_df)

    pipeline_log.info("Dataframes split to jobs, creating the job dataframe...")
    collection_switch = {
        ExtractionCollection.SENTINEL1: create_job_dataframe_s1,
        ExtractionCollection.SENTINEL2: create_job_dataframe_s2,
        ExtractionCollection.METEO: create_job_dataframe_meteo,
    }

    create_job_dataframe_fn = collection_switch.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    pipeline_log.warning(
        "Subsampling the number of splitted dataframes. Remove this for production."
    )
    split_dfs = split_dfs[:1]

    job_df = create_job_dataframe_fn(backend, split_dfs)
    pipeline_log.info("Job dataframe created with %s jobs.", len(job_df))

    return job_df


def setup_extraction_functions(
    collection: ExtractionCollection,
    memory: str,
    memory_overhead: str,
    max_executors: int,
) -> tuple[callable, callable, callable]:
    """Setup the datacube creation, path generation and post-job action
    functions for the given collection. Returns a tuple of three functions:
    1. The datacube creation function
    2. The output path generation function
    3. The post-job action function
    """

    datacube_creation = {
        ExtractionCollection.SENTINEL1: partial(
            create_datacube_sar,
            executor_memory=memory,
            executor_memory_overhead=memory_overhead,
            max_executors=max_executors,
        ),
        ExtractionCollection.SENTINEL2: partial(
            create_datacube_optical,
            executor_memory=memory,
            executor_memory_overhead=memory_overhead,
            max_executors=max_executors,
        ),
        ExtractionCollection.METEO: partial(
            create_datacube_meteo,
            executor_memory=memory,
            executor_memory_overhead=memory_overhead,
            max_executors=max_executors,
        ),
    }

    datacube_fn = datacube_creation.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    path_fn = partial(generate_output_path, s2_grid=load_s2_grid())

    post_job_actions = {
        ExtractionCollection.SENTINEL1: partial(
            post_job_action,
            description="Sentinel1 GRD raw observations, unprocessed.",
            title="Sentinel-1 GRD",
            spatial_resolution="20m",
        ),
        ExtractionCollection.SENTINEL2: partial(
            post_job_action,
            description="Sentinel2 L2A observations, processed.",
            title="Sentinel-2 L2A",
            spatial_resolution="10m",
        ),
        ExtractionCollection.METEO: partial(
            post_job_action,
            description="Meteo observations",
            title="Meteo observations",
            spatial_resolution="1deg",
        ),
    }

    post_job_fn = post_job_actions.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    return datacube_fn, path_fn, post_job_fn


def manager_main_loop(
    manager: GFMAPJobManager,
    collection: ExtractionCollection,
    job_df: gpd.GeoDataFrame,
    datacube_fn: callable,
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
        except (OpenEoApiPlainError, OpenEoApiError, OpenEoRestError) as e:
            pipeline_log.exception("An error occurred during the extraction.\n%s", e)
            send_notification(
                title=f"WorldCereal Extraction {collection.value} - OpenEo Exception",
                message=f"An OpenEO error occurred during the extraction.\n{e}",
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
        "input_df", type=Path, help="The input dataframe with the data to extract"
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
        "--memory_overhead",
        type=str,
        default="1900m",
        help="Memory overhead to allocate for the executor.",
    )
    parser.add_argument(
        "--max_executors", type=int, default=22, help="Number of executors to run."
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=10,
        help="The maximum number of parrallel jobs to run at the same time.",
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
    args = parser.parse_args()

    # Fetches values and setups hardocded values
    collection = args.collection
    extract_value = args.extract_value
    max_locations_per_job = args.max_locations
    backend = Backend.CDSE

    if not args.output_folder.is_dir():
        raise ValueError(f"Output folder {args.output_folder} does not exist.")

    tracking_df_path = Path(args.output_folder) / "job_tracking.csv"

    # Load the input dataframe and build the job dataframe
    input_df = load_dataframe(args.input_df)

    job_df = prepare_job_dataframe(
        input_df, collection, max_locations_per_job, extract_value, backend
    )

    # Setup the extraction functions
    pipeline_log.info("Setting up the extraction functions.")
    datacube_fn, path_fn, post_job_fn = setup_extraction_functions(
        collection, args.memory, args.memory_overhead, args.max_executors
    )

    # Initialize and setups the job manager
    pipeline_log.info("Initializing the job manager.")

    collection_id = {
        ExtractionCollection.SENTINEL1: "SENTINEL1-EXTRACTION",
        ExtractionCollection.SENTINEL2: "sentinel2-EXTRACTION",
        ExtractionCollection.METEO: "METEO-EXTRACTION",
    }
    collection_description = {
        ExtractionCollection.SENTINEL1: "Sentinel1 GRD data extraction.",
        ExtractionCollection.SENTINEL2: "Sentinel2 L2A data extraction.",
        ExtractionCollection.METEO: "Meteo data extraction.",
    }
    job_manager = GFMAPJobManager(
        output_dir=args.output_folder,
        output_path_generator=path_fn,
        post_job_action=post_job_fn,
        collection_id=collection_id[collection],
        collection_description=collection_description[collection],
        poll_sleep=60,
        n_threads=4,
        restart_failed=args.restart_failed,
    )

    job_manager.add_backend(
        Backend.CDSE.value,
        cdse_connection,
        dynamic_max_jobs=False,
        parallel_jobs=args.parallel_jobs,
    )

    constellation_name = {
        ExtractionCollection.SENTINEL1: "sentinel1",
        ExtractionCollection.SENTINEL2: "sentinel2",
        ExtractionCollection.METEO: "agera5",
    }
    item_assets = {
        ExtractionCollection.SENTINEL1: {"sentinel1": sentinel1_asset},
        ExtractionCollection.SENTINEL2: {"sentinel2": sentinel2_asset},
        ExtractionCollection.METEO: {"agera5": meteo_asset},
    }

    job_manager.setup_stac(
        constellation=constellation_name[collection],
        item_assets=item_assets[collection],
    )

    manager_main_loop(job_manager, collection, job_df, datacube_fn, tracking_df_path)

    pipeline_log.info("Extraction completed successfully.")
    send_notification(
        title=f"WorldCereal Extraction {collection.value} - Completed",
        message="Extractions have been completed successfully.",
    )
