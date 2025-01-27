"""Main script to perform extractions. Each collection has it's specifities and
own functions, but the setup and main thread execution is done here."""

import argparse
import os
import typing
from datetime import datetime
from functools import partial
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from openeo.rest import OpenEoApiError, OpenEoApiPlainError, OpenEoRestError
from openeo_gfmap import Backend
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import load_s2_grid, split_job_s2grid

from worldcereal.extract.common import (
    generate_output_path_patch,
    pipeline_log,
    post_job_action_patch,
)
from worldcereal.extract.patch_meteo import (
    create_job_dataframe_patch_meteo,
    create_job_patch_meteo,
)
from worldcereal.extract.patch_s2 import (
    create_job_dataframe_patch_s2,
    create_job_patch_s2,
)
from worldcereal.extract.point_worldcereal import (
    create_job_dataframe_point_worldcereal,
    create_job_point_worldcereal,
    generate_output_path_point_worldcereal,
    post_job_action_point_worldcereal,
)
from worldcereal.stac.constants import ExtractionCollection

from worldcereal.extract.patch_s1 import (  # isort: skip
    create_job_patch_s1,
    create_job_dataframe_patch_s1,
)


from worldcereal.extract.patch_worldcereal import (  # isort: skip
    create_job_patch_worldcereal,
    create_job_dataframe_patch_worldcereal,
    post_job_action_patch_worldcereal,
    generate_output_path_patch_worldcereal,
)


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
    input_df = input_df[input_df["extract"] >= extract_value].copy()

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
    collection_switch: dict[ExtractionCollection, typing.Callable] = {
        ExtractionCollection.PATCH_SENTINEL1: create_job_dataframe_patch_s1,
        ExtractionCollection.PATCH_SENTINEL2: create_job_dataframe_patch_s2,
        ExtractionCollection.PATCH_METEO: create_job_dataframe_patch_meteo,
        ExtractionCollection.PATCH_WORLDCEREAL: create_job_dataframe_patch_worldcereal,
        ExtractionCollection.POINT_WORLDCEREAL: create_job_dataframe_point_worldcereal,
    }

    create_job_dataframe_fn = collection_switch.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    job_df = create_job_dataframe_fn(backend, split_dfs)
    pipeline_log.info("Job dataframe created with %s jobs.", len(job_df))

    return job_df


def setup_extraction_functions(
    collection: ExtractionCollection,
    extract_value: int,
    memory: typing.Union[str, None],
    python_memory: typing.Union[str, None],
    max_executors: typing.Union[int, None],
) -> tuple[typing.Callable, typing.Callable, typing.Callable]:
    """Setup the datacube creation, path generation and post-job action
    functions for the given collection. Returns a tuple of three functions:
    1. The datacube creation function
    2. The output path generation function
    3. The post-job action function
    """

    datacube_creation = {
        ExtractionCollection.PATCH_SENTINEL1: partial(
            create_job_patch_s1,
            executor_memory=memory if memory is not None else "1800m",
            python_memory=python_memory if python_memory is not None else "1900m",
            max_executors=max_executors if max_executors is not None else 22,
        ),
        ExtractionCollection.PATCH_SENTINEL2: partial(
            create_job_patch_s2,
            executor_memory=memory if memory is not None else "1800m",
            python_memory=python_memory if python_memory is not None else "1900m",
            max_executors=max_executors if max_executors is not None else 22,
        ),
        ExtractionCollection.PATCH_METEO: partial(
            create_job_patch_meteo,
            executor_memory=memory if memory is not None else "1800m",
            python_memory=python_memory if python_memory is not None else "1000m",
            max_executors=max_executors if max_executors is not None else 22,
        ),
        ExtractionCollection.PATCH_WORLDCEREAL: partial(
            create_job_patch_worldcereal,
            executor_memory=memory if memory is not None else "1800m",
            python_memory=python_memory if python_memory is not None else "3000m",
            max_executors=max_executors if max_executors is not None else 22,
        ),
        ExtractionCollection.POINT_WORLDCEREAL: partial(
            create_job_point_worldcereal,
            executor_memory=memory if memory is not None else "1800m",
            python_memory=python_memory if python_memory is not None else "3000m",
            max_executors=max_executors if max_executors is not None else 22,
        ),
    }

    datacube_fn = datacube_creation.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    path_fns = {
        ExtractionCollection.PATCH_SENTINEL1: partial(
            generate_output_path_patch, s2_grid=load_s2_grid()
        ),
        ExtractionCollection.PATCH_SENTINEL2: partial(
            generate_output_path_patch, s2_grid=load_s2_grid()
        ),
        ExtractionCollection.PATCH_METEO: partial(
            generate_output_path_patch, s2_grid=load_s2_grid()
        ),
        ExtractionCollection.PATCH_WORLDCEREAL: partial(
            generate_output_path_patch_worldcereal, s2_grid=load_s2_grid()
        ),
        ExtractionCollection.POINT_WORLDCEREAL: partial(
            generate_output_path_point_worldcereal
        ),
    }

    path_fn = path_fns.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    post_job_actions = {
        ExtractionCollection.PATCH_SENTINEL1: partial(
            post_job_action_patch,
            extract_value=extract_value,
            description="Sentinel1 GRD raw observations, unprocessed.",
            title="Sentinel-1 GRD",
            spatial_resolution="20m",
            s1_orbit_fix=True,
            sensor="Sentinel1",
            write_stac_api=True,
        ),
        ExtractionCollection.PATCH_SENTINEL2: partial(
            post_job_action_patch,
            extract_value=extract_value,
            description="Sentinel2 L2A observations, processed.",
            title="Sentinel-2 L2A",
            spatial_resolution="10m",
            sensor="Sentinel2",
            write_stac_api=True,
        ),
        ExtractionCollection.PATCH_METEO: partial(
            post_job_action_patch,
            extract_value=extract_value,
            description="Meteo observations",
            title="Meteo observations",
            spatial_resolution="1deg",
        ),
        ExtractionCollection.PATCH_WORLDCEREAL: partial(
            post_job_action_patch_worldcereal,
            extract_value=extract_value,
            description="WorldCereal preprocessed inputs",
            title="WorldCereal inputs",
            spatial_resolution="10m",
        ),
        ExtractionCollection.POINT_WORLDCEREAL: partial(
            post_job_action_point_worldcereal,
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
    datacube_fn: typing.Callable,
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
    input_df: Path,
    max_locations_per_job: int = 500,
    memory: str = "1800m",
    python_memory: str = "1900m",
    max_executors: int = 22,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    extract_value: int = 1,
    backend=Backend.CDSE,
) -> None:
    """Main function responsible for launching point and patch extractions.

    Parameters
    ----------
    collection : ExtractionCollection
        The collection to extract. Most popular: PATCH_WORLDCEREAL, POINT_WORLDCEREAL
    output_folder : Path
        The folder where to store the extracted data
    input_df : Path
        Path to the input dataframe containing the geometries
        for which extractions need to be done
    max_locations_per_job : int, optional
        The maximum number of locations to extract per job, by default 500
    memory : str, optional
        Memory to allocate for the executor, by default "1800m"
    python_memory : str, optional
        Memory to allocate for the python processes as well as OrfeoToolbox in the executors,
        by default "1900m"
    max_executors : int, optional
        Number of executors to run, by default 22
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

    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    tracking_df_path = output_folder / "job_tracking.csv"

    # Load the input dataframe and build the job dataframe
    input_df = load_dataframe(input_df)

    job_df = None
    if not tracking_df_path.exists():
        job_df = prepare_job_dataframe(
            input_df, collection, max_locations_per_job, extract_value, backend
        )

    # Setup the extraction functions
    pipeline_log.info("Setting up the extraction functions.")
    datacube_fn, path_fn, post_job_fn = setup_extraction_functions(
        collection, extract_value, memory, python_memory, max_executors
    )

    # Initialize and setups the job manager
    pipeline_log.info("Initializing the job manager.")

    job_manager = GFMAPJobManager(
        output_dir=output_folder,
        output_path_generator=path_fn,
        post_job_action=post_job_fn,
        poll_sleep=60,
        n_threads=4,
        restart_failed=restart_failed,
        stac_enabled=False,
    )

    job_manager.add_backend(
        backend.value,
        cdse_connection,
        parallel_jobs=parallel_jobs,
    )

    manager_main_loop(job_manager, collection, job_df, datacube_fn, tracking_df_path)

    pipeline_log.info("Extraction completed successfully.")
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

    args = parser.parse_args()

    run_extractions(
        collection=args.collection,
        output_folder=args.output_folder,
        input_df=args.input_df,
        max_locations_per_job=args.max_locations,
        memory=args.memory,
        python_memory=args.python_memory,
        max_executors=args.max_executors,
        parallel_jobs=args.parallel_jobs,
        restart_failed=args.restart_failed,
        extract_value=args.extract_value,
        backend=Backend.CDSE,
    )
