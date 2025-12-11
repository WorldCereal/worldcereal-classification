"""Common functions used by extraction scripts."""

import json
import os
import shutil
from datetime import datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, Iterable, List, Optional, Union

import geopandas as gpd
import pandas as pd
import pystac
import pystac_client
import xarray as xr
from openeo_gfmap import Backend
from openeo_gfmap.backend import BACKEND_CONNECTIONS
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import split_job_s2grid
from tabulate import tabulate

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
    merge_output_files_point_worldcereal,
    post_job_action_point_worldcereal,
)
from worldcereal.extract.utils import pipeline_log
from worldcereal.stac.constants import ExtractionCollection
from worldcereal.stac.stac_api_interaction import (
    StacApiInteraction,
    VitoStacApiAuthentication,
)
from worldcereal.utils.retry import retry

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


RETRIES = int(os.environ.get("WORLDCEREAL_EXTRACTION_RETRIES", 5))
DELAY = int(os.environ.get("WORLDCEREAL_EXTRACTION_DELAY", 10))
BACKOFF = int(os.environ.get("WORLDCEREAL_EXTRACTION_BACKOFF", 5))

STAC_ROOT_URL = "https://stac.openeo.vito.be/"


def extraction_job_quality_check(
    job_entry: pd.Series, orfeo_error_threshold: float = 0.3
) -> None:
    """Perform quality checks on an extraction job.

    Parameters
    ----------
    job_entry : pd.Series
        The job entry containing information about the extraction job.
    orfeo_error_threshold : float, optional
        The threshold for the SAR backscatter error ratio, by default 0.3.

    Raises
    ------
    Exception
        Raised if the job has no assets.
    Exception
        Raised if the SAR backscatter error ratio exceeds the threshold.
    """
    conn = BACKEND_CONNECTIONS[Backend[job_entry["backend_name"].upper()]]()
    job = conn.job(job_entry.id)

    # Check if we have any assets resulting from the job
    job_results = job.get_results()
    if len(job_results.get_metadata()["assets"]) == 0:
        raise Exception(f"Job {job_entry.id} has no assets!")

    # Check if SAR backscatter error ratio exceeds the threshold
    if "sar_backscatter_soft_errors" in job.describe()["usage"].keys():
        actual_orfeo_error_rate = job.describe()["usage"][
            "sar_backscatter_soft_errors"
        ]["value"]
        if actual_orfeo_error_rate > orfeo_error_threshold:
            raise Exception(
                f"Job {job_entry.id} had a ORFEO error rate of {actual_orfeo_error_rate}!"
            )

    pipeline_log.debug("Quality checks passed!")


def post_job_action_patch(
    job_items: List[pystac.Item],
    row: pd.Series,
    extract_value: int,
    description: str,
    title: str,
    spatial_resolution: str,
    s1_orbit_fix: bool = False,  # To rename the samples from the S1 orbit
    write_stac_api: bool = False,
    sensor: str = "Sentinel1",
) -> list:
    """From the job items, extract the metadata and save it in a netcdf file."""

    import grp

    # First do some basic quality checks to see if everything went right
    extraction_job_quality_check(row)

    base_gpd = gpd.GeoDataFrame.from_features(json.loads(row.geometry)).set_crs(
        epsg=4326
    )
    if len(base_gpd[base_gpd.extract == extract_value]) != len(job_items):
        pipeline_log.warning(
            "Different amount of geometries in the job output items and the "
            "input geometry. Job items #: %s, Input geometries #: %s",
            len(job_items),
            len(base_gpd[base_gpd.extract == extract_value]),
        )

    extracted_gpd = base_gpd[base_gpd.extract >= extract_value].reset_index(drop=True)
    # In this case we want to burn the metadata in a new file in the same folder as the S2 product
    for item in job_items:
        item_id = item.id.replace(".nc", "").replace("openEO_", "")
        sample_id_column_name = (
            "sample_id" if "sample_id" in extracted_gpd.columns else "sampleID"
        )

        geometry_information = extracted_gpd.loc[
            extracted_gpd[sample_id_column_name] == item_id
        ]

        if len(geometry_information) == 0:
            pipeline_log.warning(
                "No geometry found for the sample_id %s in the input geometry.",
                item_id,
            )
            continue

        if len(geometry_information) > 1:
            pipeline_log.warning(
                "Duplicate geomtries found for the sample_id %s in the input geometry, selecting the first one at index: %s.",
                item_id,
                geometry_information.index[0],
            )

        geometry_information = geometry_information.iloc[0]

        sample_id = geometry_information[sample_id_column_name]
        ref_id = geometry_information.ref_id
        valid_time = geometry_information.valid_time
        h3_l3_cell = geometry_information.h3_l3_cell
        s2_tile = row.s2_tile

        item_asset_path = Path(list(item.assets.values())[0].href)

        # Add some metadata to the result_df netcdf file
        new_attributes = {
            "start_date": row.start_date,
            "end_date": row.end_date,
            "valid_time": valid_time,
            "processing:version": version("openeo_gfmap"),
            "institution": "VITO - ESA WorldCereal",
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "title": title,
            "sample_id": sample_id,
            "ref_id": ref_id,
            "spatial_resolution": spatial_resolution,
            "s2_tile": s2_tile,
            "h3_l3_cell": h3_l3_cell,
            "_FillValue": 65535,  # No data value for uint16
        }

        if s1_orbit_fix:
            new_attributes["sat:orbit_state"] = row.orbit_state
            item.id = item.id.replace(".nc", f"_{row.orbit_state}.nc")

        # Saves the new attributes in the netcdf file
        ds = xr.open_dataset(item_asset_path)

        # Validate spatial dimensions based on resolution
        expected_dim_size = {"10m": 64, "20m": 32}.get(spatial_resolution, None)
        if expected_dim_size is not None:
            actual_x_size = ds.dims.get("x", 0)
            actual_y_size = ds.dims.get("y", 0)

            if actual_x_size != expected_dim_size or actual_y_size != expected_dim_size:
                pipeline_log.warning(
                    "Dimension validation failed for %s: expected %dx%d for %s resolution, got %dx%d",
                    item_asset_path,
                    expected_dim_size,
                    expected_dim_size,
                    spatial_resolution,
                    actual_x_size,
                    actual_y_size,
                )

            pipeline_log.debug(
                "Dimension validation passed for %s: %dx%d matches expected %s resolution",
                item_asset_path,
                actual_x_size,
                actual_y_size,
                spatial_resolution,
            )

        ds = ds.assign_attrs(new_attributes)

        with NamedTemporaryFile(delete=False) as temp_file:
            ds.to_netcdf(temp_file.name)
            shutil.move(temp_file.name, item_asset_path)
            os.chmod(item_asset_path, 0o755)
            gid = grp.getgrnam("vito").gr_gid
            shutil.chown(item_asset_path, group=gid)

        pipeline_log.info(f"Final output file created: {item_asset_path}")

        # Test if the file is not corrupt
        try:
            ds = xr.open_dataset(item_asset_path)
            ds.close()
        except Exception as e:
            pipeline_log.error(
                "The output file %s is corrupt. Error: %s", item_asset_path, e
            )
            raise

        # Update the metadata of the item
        if write_stac_api:
            item.properties.update(new_attributes)

            providers = [{"name": "openEO platform"}]
            item.properties["providers"] = providers

            extension = (
                "https://stac-extensions.github.io/processing/v1.2.0/schema.json"
            )
            item.stac_extensions.extend([extension])

    if write_stac_api:
        username = os.getenv("STAC_API_USERNAME")
        password = os.getenv("STAC_API_PASSWORD")

        if not username or not password:
            error_msg = (
                "STAC API credentials not found. Please set "
                "STAC_API_USERNAME and STAC_API_PASSWORD."
            )
            pipeline_log.error(error_msg)
            raise ValueError(error_msg)

        stac_api_interaction = StacApiInteraction(
            sensor=sensor,
            base_url="https://stac.openeo.vito.be",
            auth=VitoStacApiAuthentication(username=username, password=password),
        )

        pipeline_log.info("Writing the STAC API metadata")
        stac_api_interaction.upload_items_bulk(job_items)
        pipeline_log.info("STAC API metadata written")

    return job_items


def generate_output_path_patch(
    root_folder: Path,
    job_index: int,
    row: pd.Series,
    asset_id: str,
):
    """Generate the output path for the extracted data, from a base path and
    the row information. `root_folder` is assumed to point to the specific
    output folder for a `ref_id`
    """
    # First extract the sample ID from the asset ID
    sample_id = asset_id.replace(".nc", "").replace("openEO_", "")

    if "orbit_state" in row:
        orbit_state = f"_{row.orbit_state}"
    else:
        orbit_state = ""

    s2_tile_id = row.s2_tile
    utm_zone = str(s2_tile_id[0:2])

    subfolder = root_folder / utm_zone / s2_tile_id / sample_id

    return (
        subfolder
        / f"{row.out_prefix}{orbit_state}_{sample_id}_{row.start_date}_{row.end_date}{row.out_extension}"
    )


def load_dataframe(
    df_path: Path,
    extract_value: int = 0,
    check_existing: bool = False,
    collection: Optional[ExtractionCollection] = None,
    ref_id: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load the input dataframe from the given path.
    Optionally filter the dataframe based on the `extract_value` and check
    for existing samples in the STAC API.
    """
    pipeline_log.info("Loading input dataframe from %s.", df_path)

    # Specify a filter for "extract" column. Only extract the samples
    # with a value >= extract_value
    filters = [("extract", ">=", extract_value)]

    pipeline_log.info("Reading the input dataframe with filters: %s", filters)

    if df_path.name.endswith("parquet"):
        df = gpd.read_parquet(df_path, filters=filters)
    else:
        df = gpd.read_file(df_path, filters=filters)

    if check_existing:
        if collection is None:
            pipeline_log.warning(
                "STAC check is only performed for PATCH_SENTINEL1 or PATCH_SENTINEL2 collections, but collection is None."
            )
        elif collection in [
            ExtractionCollection.PATCH_SENTINEL1,
            ExtractionCollection.PATCH_SENTINEL2,
        ]:
            pipeline_log.info(
                "Checking existing samples in STAC API for ref_id %s, collection %s.",
                ref_id,
                collection,
            )
            client = pystac_client.Client.open(STAC_ROOT_URL)
            samples_list: list[str] = []

            if collection == ExtractionCollection.PATCH_SENTINEL1:
                STAC_COLLECTION = "worldcereal_sentinel_1_patch_extractions"
            elif collection == ExtractionCollection.PATCH_SENTINEL2:
                STAC_COLLECTION = "worldcereal_sentinel_2_patch_extractions"
            else:
                raise ValueError(
                    f"Collection {collection} is not supported for STAC check."
                )

            stac_search = client.search(
                collections=[STAC_COLLECTION],
                filter={"op": "=", "args": [{"property": "properties.ref_id"}, ref_id]},
                filter_lang="cql2-json",
                fields={"exclude": ["assets", "links", "geometry", "bbox"]},
            )
            for item in stac_search.items():
                sample_id = item.properties.get("sample_id")
                if sample_id:
                    samples_list.append(sample_id)
            df = df[~df["sample_id"].isin(samples_list)]
            if len(df) > 0:
                pipeline_log.info(
                    "Filtered out %s samples that already exist in STAC API for collection %s.",
                    len(samples_list),
                    collection,
                )
            else:
                pipeline_log.info(
                    "All samples already exist in STAC API for ref_id %s, collection %s. No samples to extract.",
                    ref_id,
                    collection,
                )
                return gpd.GeoDataFrame()
        else:
            pipeline_log.warning(
                "STAC check is only performed for PATCH_SENTINEL1 or PATCH_SENTINEL2 collections. ",
                "Collection %s is not supported. Skipping STAC check.",
                collection,
            )
    return df


def prepare_job_dataframe(
    samples_gdf: gpd.GeoDataFrame,
    collection: ExtractionCollection,
    max_locations: int,
    backend: Backend,
) -> gpd.GeoDataFrame:
    """Prepare the job dataframe to extract the data from the given input
    dataframe."""
    pipeline_log.info("Preparing the job dataframe.")

    # Split the locations into chunks of max_locations
    split_dfs = []
    pipeline_log.info(
        "Performing splitting by the year...",
    )
    samples_gdf["valid_time"] = pd.to_datetime(samples_gdf.valid_time)
    samples_gdf["year"] = samples_gdf.valid_time.dt.year

    split_dfs_time = [
        group.reset_index(drop=True) for _, group in samples_gdf.groupby("year")
    ]
    pipeline_log.info("Performing splitting by s2 grid...")
    for df in split_dfs_time:
        s2_split_df = split_job_s2grid(df, max_points=max_locations)
        split_dfs.extend(s2_split_df)

    pipeline_log.info("Dataframes split to jobs, creating the job dataframe...")
    collection_switch: dict[ExtractionCollection, Callable] = {
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


def _count_by_status(job_status_df, statuses: Iterable[str] = ()) -> dict:
    status_histogram = job_status_df.groupby("status").size().to_dict()
    statuses = set(statuses)
    if statuses:
        status_histogram = {k: v for k, v in status_histogram.items() if k in statuses}
    return status_histogram


def _read_job_tracking_csv(output_folder: Path) -> pd.DataFrame:
    """Read job tracking csv file.

    Parameters
    ----------
    output_folder : Path
        folder where extractions are stored

    Returns
    -------
    pd.DataFrame
        job tracking dataframe

    Raises
    ------
    FileNotFoundError
        if the job status file is not found in the designated folder
    """
    job_status_file = output_folder / "job_tracking.csv"
    if job_status_file.exists():
        job_status_df = pd.read_csv(job_status_file)
    else:
        raise FileNotFoundError(f"Job status file not found at {job_status_file}")
    return job_status_df


def check_job_status(output_folder: Path) -> dict:
    """Check the status of the jobs in the given output folder.

    Parameters
    ----------
    output_folder : Path
        folder where extractions are stored

    Returns
    -------
    dict
        status_histogram
    """

    # Read job tracking csv file
    job_status_df = _read_job_tracking_csv(output_folder)

    # Summarize the status in histogram
    status_histogram = _count_by_status(job_status_df)

    # convert to pandas dataframe
    status_count = pd.DataFrame(status_histogram.items(), columns=["status", "count"])
    status_count = status_count.sort_values(by="count", ascending=False)

    print("-------------------------------------")
    print("Overall jobs status:")
    print(tabulate(status_count, headers="keys", tablefmt="psql", showindex=False))

    return status_histogram


def get_succeeded_job_details(output_folder: Path) -> pd.DataFrame:
    """Get details of succeeded extraction jobs in the given output folder.

    Parameters
    ----------
    output_folder : Path
        folder where extractions are stored
    Returns
    -------
    pd.DataFrame
        details of succeeded jobs
    """

    # Read job tracking csv file
    job_status_df = _read_job_tracking_csv(output_folder)

    # Gather metadata on succeeded jobs
    succeeded_jobs = job_status_df[
        job_status_df["status"].isin(["finished", "postprocessing"])
    ].copy()
    if len(succeeded_jobs) > 0:
        # Derive number of features involved in each job
        nfeatures = []
        for i, row in succeeded_jobs.iterrows():
            nfeatures.append(len(json.loads(row["geometry"])["features"]))
        succeeded_jobs.loc[:, "n_samples"] = nfeatures
        # Gather essential columns
        succeeded_jobs = succeeded_jobs[
            [
                "id",
                "s2_tile",
                "n_samples",
                # "cpu", "memory",
                "duration",
                "costs",
            ]
        ]
        # Convert duration to minutes
        # convert NaN to 0 seconds
        succeeded_jobs["duration"] = succeeded_jobs["duration"].fillna("0s")
        seconds = succeeded_jobs["duration"].str.split("s").str[0].astype(int)
        succeeded_jobs["duration"] = seconds / 60
        succeeded_jobs.rename(columns={"duration": "duration_mins"}, inplace=True)
        if succeeded_jobs["duration_mins"].sum() == 0:
            succeeded_jobs.drop(columns=["duration_mins"], inplace=True)
    else:
        succeeded_jobs = pd.DataFrame()

    # Pretty reporting
    if not succeeded_jobs.empty:
        from tabulate import tabulate

        total_samples = int(succeeded_jobs["n_samples"].sum())
        n_jobs = succeeded_jobs.shape[0]
        # Costs summary (only if all non-null)
        if pd.notnull(succeeded_jobs["costs"]).all():
            total_credits = int(succeeded_jobs["costs"].sum())
            avg_cost = int(succeeded_jobs["costs"].mean())
        else:
            total_credits = 0
            avg_cost = 0

        summary_table = [
            ["Succeeded jobs", n_jobs],
            ["Total samples", f"{total_samples:,}"],
            ["Total credits", total_credits],
            ["Average job cost", avg_cost],
        ]
        print("\n" + "=" * 80)
        print("SUCCEEDED JOBS SUMMARY")
        print("=" * 80)
        print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="grid"))

        # Detailed job table
        detail_cols = [
            c
            for c in ["id", "s2_tile", "n_samples", "duration_mins", "costs"]
            if c in succeeded_jobs.columns
        ]
        detail_df = succeeded_jobs[detail_cols].copy()
        # Sort by n_samples descending then duration
        if "n_samples" in detail_df.columns:
            sort_cols = ["n_samples"] + (
                ["duration_mins"] if "duration_mins" in detail_df.columns else []
            )
            detail_df = detail_df.sort_values(
                by=sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1)
            )

        print("\nDetails per job:")
        print(tabulate(detail_df, headers="keys", tablefmt="psql", showindex=False))
        print("=" * 80 + "\n")
    else:
        print("No succeeded jobs to report.")

    return succeeded_jobs


def setup_extraction_functions(
    collection: ExtractionCollection,
    extract_value: int,
    write_stac_api: bool,
    job_options: Optional[dict] = None,
) -> tuple[Callable, Callable, Callable]:
    """Setup the datacube creation, path generation and post-job action
    functions for the given collection. Returns a tuple of three functions:
    1. The datacube creation function
    2. The output path generation function
    3. The post-job action function
    """

    # Setup the datacube creation function
    datacube_creation = {
        ExtractionCollection.PATCH_SENTINEL1: partial(
            create_job_patch_s1, job_options=job_options
        ),
        ExtractionCollection.PATCH_SENTINEL2: partial(
            create_job_patch_s2, job_options=job_options
        ),
        ExtractionCollection.PATCH_METEO: partial(
            create_job_patch_meteo, job_options=job_options
        ),
        ExtractionCollection.PATCH_WORLDCEREAL: partial(
            create_job_patch_worldcereal, job_options=job_options
        ),
        ExtractionCollection.POINT_WORLDCEREAL: partial(
            create_job_point_worldcereal, job_options=job_options
        ),
    }

    datacube_fn = datacube_creation.get(
        collection,
        lambda: (_ for _ in ()).throw(
            ValueError(f"Collection {collection} not supported.")
        ),
    )

    # Setup the output path generation function
    path_fns = {
        ExtractionCollection.PATCH_SENTINEL1: partial(generate_output_path_patch),
        ExtractionCollection.PATCH_SENTINEL2: partial(generate_output_path_patch),
        ExtractionCollection.PATCH_METEO: partial(generate_output_path_patch),
        ExtractionCollection.PATCH_WORLDCEREAL: partial(
            generate_output_path_patch_worldcereal
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

    # Setup the post-job action function
    post_job_actions = {
        ExtractionCollection.PATCH_SENTINEL1: partial(
            post_job_action_patch,
            extract_value=extract_value,
            description="Sentinel-1 GRD backscatter observations, processed with Orfeo toolbox.",
            title="Sentinel-1 GRD",
            spatial_resolution="20m",
            s1_orbit_fix=True,
            sensor="Sentinel1",
            write_stac_api=write_stac_api,
        ),
        ExtractionCollection.PATCH_SENTINEL2: partial(
            post_job_action_patch,
            extract_value=extract_value,
            description="Sentinel-2 L2A surface reflectance observations.",
            title="Sentinel-2 L2A",
            spatial_resolution="10m",
            sensor="Sentinel2",
            write_stac_api=write_stac_api,
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


def _prepare_extraction_jobs(
    collection: ExtractionCollection,
    output_folder: Path,
    samples_df_path: Path,
    ref_id: str,
    max_locations_per_job: int = 500,
    job_options: Optional[Dict[str, Union[str, int]]] = None,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    extract_value: int = 0,
    backend=Backend.CDSE,
    write_stac_api: bool = False,
    check_existing_extractions: bool = False,
) -> tuple[GFMAPJobManager, pd.DataFrame, Callable, Path]:
    """Function responsible for preparing the extraction jobs:
    splitting jobs, preparing the job manager, setting up the extraction functions.

    Parameters
    ----------
    collection : ExtractionCollection
        The collection to extract
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
        All samples with an "extract" value equal or larger than this one, will be extracted, by default 0
        so all samples are extracted
    backend : _type_, optional
        cloud backend where to run the extractions, by default Backend.CDSE
    write_stac_api : bool, optional
        Save metadata of extractions to STAC API (requires authentication), by default False
    check_existing_extractions : bool, optional
        Check if the samples already exist in the STAC API and filter them out,
        by default False

    Returns
    -------
    tuple[GFMAPJobManager, pd.DataFrame, Callable, Path]
        JobManager, job dataframe, datacube function, job tracking dataframe path
    """

    # Make sure output folder exists
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    # Create path to tracking dataframe
    tracking_df_path = output_folder / "job_tracking.csv"

    # If the tracking dataframe already exists, load it
    if tracking_df_path.exists():
        pipeline_log.info("Loading existing job tracking dataframe.")
        job_df = pd.read_csv(tracking_df_path)

        # Change status "canceled" to "not_started"
        job_df.loc[job_df.status.isin(["canceled"]), "status"] = "not_started"

        # If restart_failed is True, reset some statuses as well.
        # Normally should be handled in GFMap but does not always work
        if restart_failed:
            pipeline_log.info("Resetting failed jobs.")
            job_df.loc[
                job_df["status"].isin(
                    ["error", "postprocessing-error", "start_failed"]
                ),
                "status",
            ] = "not_started"

        # Save new job tracking dataframe
        job_df.to_csv(tracking_df_path, index=False)

        status_histogram = check_job_status(output_folder)
        pipeline_log.info(
            "Job status histogram: %s",
            status_histogram,
        )
    else:
        # Load the input dataframe and build the job dataframe
        samples_gdf = load_dataframe(
            samples_df_path,
            extract_value,
            check_existing=check_existing_extractions,
            collection=collection,
            ref_id=ref_id,
        )
        samples_gdf["ref_id"] = ref_id
        pipeline_log.info("Creating new job tracking dataframe.")
        job_df = prepare_job_dataframe(
            samples_gdf, collection, max_locations_per_job, backend
        )

    # Setup the extraction functions
    pipeline_log.info("Setting up the extraction functions.")
    datacube_fn, path_fn, post_job_fn = setup_extraction_functions(
        collection, extract_value, write_stac_api, job_options
    )

    # Initialize and setup the job manager
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
        BACKEND_CONNECTIONS[backend],
        parallel_jobs=parallel_jobs,
    )

    return job_manager, job_df, datacube_fn, tracking_df_path


@retry(
    exceptions=Exception,
    tries=RETRIES,
    delay=DELAY,
    backoff=BACKOFF,
    logger=pipeline_log,
)
def _run_extraction_jobs(
    job_manager: GFMAPJobManager,
    job_df: pd.DataFrame,
    datacube_fn: Callable,
    tracking_df_path: Path,
) -> None:
    # Run the extraction jobs
    pipeline_log.info("Running the extraction jobs.")
    job_manager.run_jobs(job_df, datacube_fn, tracking_df_path)
    pipeline_log.info("Extraction jobs completed.")


def _merge_extraction_jobs(
    output_folder: Path,
    ref_id: str,
) -> None:
    """Merge all extractions into one partitioned geoparquet file.

    Parameters
    ----------
    output_folder : Path
        Location where extractions are stored.
    ref_id : str
        collection id of the samples
    """

    # Merge the extraction jobs
    pipeline_log.info("Merging the extraction jobs.")
    pipeline_log.info("Merging Geoparquet results...")
    merge_output_files_point_worldcereal(output_folder, ref_id)
    pipeline_log.info("Geoparquet results merged successfully.")


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
        cloud backend where to run the extractions, by default Backend.CDSE
    write_stac_api : bool, optional
        Save metadata of extractions to STAC API (requires authentication), by default False
    check_existing_extractions : bool, optional
        Check if the samples already exist in the STAC API and filter them out,
        by default False

    """
    pipeline_log.info("Starting the extractions workflow...")

    # Prepare the extraction jobs
    job_manager, job_df, datacube_fn, tracking_df_path = _prepare_extraction_jobs(
        collection,
        output_folder,
        samples_df_path,
        ref_id,
        max_locations_per_job=max_locations_per_job,
        job_options=job_options,
        parallel_jobs=parallel_jobs,
        restart_failed=restart_failed,
        extract_value=extract_value,
        backend=backend,
        write_stac_api=write_stac_api,
        check_existing_extractions=check_existing_extractions,
    )

    # Run the extraction jobs
    _run_extraction_jobs(job_manager, job_df, datacube_fn, tracking_df_path)

    # Merge the extraction jobs (for point extractions)
    if collection == ExtractionCollection.POINT_WORLDCEREAL:
        # Merge extractions
        _merge_extraction_jobs(output_folder, ref_id)

    pipeline_log.info("Extractions workflow completed.")
    pipeline_log.info(f"Results stored in folder: {output_folder}.")
