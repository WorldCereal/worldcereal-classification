"""Common functions used by extraction scripts."""

import json
import os
import shutil
from datetime import datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, List, Optional, Union

import geojson
import geopandas as gpd
import pandas as pd
import pystac
import xarray as xr
from openeo_gfmap import Backend
from openeo_gfmap.backend import BACKEND_CONNECTIONS
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import split_job_s2grid

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


RETRIES = int(os.environ.get("WORLDCEREAL_RETRIES", 3))
DELAY = int(os.environ.get("WORLDCEREAL_DELAY", 5))
BACKOFF = int(os.environ.get("WORLDCEREAL_BACKOFF", 1))


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

    extracted_gpd = base_gpd[base_gpd.extract == extract_value].reset_index(drop=True)
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

        ds = ds.assign_attrs(new_attributes)

        with NamedTemporaryFile(delete=False) as temp_file:
            ds.to_netcdf(temp_file.name)
            shutil.move(temp_file.name, item_asset_path)

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
    the row information.
    """
    # First extract the sample ID from the asset ID
    sample_id = asset_id.replace(".nc", "").replace("openEO_", "")

    # Find which index in the FeatureCollection corresponds to the sample_id
    features = geojson.loads(row.geometry)["features"]
    sample_id_to_index = {
        feature.properties.get("sample_id", None): index
        for index, feature in enumerate(features)
    }
    geometry_index = sample_id_to_index.get(sample_id, None)

    ref_id = features[geometry_index].properties["ref_id"]

    if "orbit_state" in row:
        orbit_state = f"_{row.orbit_state}"
    else:
        orbit_state = ""

    s2_tile_id = row.s2_tile
    utm_zone = str(s2_tile_id[0:2])

    subfolder = root_folder / ref_id / utm_zone / s2_tile_id / sample_id

    return (
        subfolder
        / f"{row.out_prefix}{orbit_state}_{sample_id}_{row.start_date}_{row.end_date}{row.out_extension}"
    )


def load_dataframe(df_path: Path) -> gpd.GeoDataFrame:
    """Load the input dataframe from the given path."""
    pipeline_log.info("Loading input dataframe from %s.", df_path)

    if df_path.name.endswith(".geoparquet"):
        return gpd.read_parquet(df_path)
    else:
        return gpd.read_file(df_path)


def prepare_job_dataframe(
    samples_gdf: gpd.GeoDataFrame,
    collection: ExtractionCollection,
    max_locations: int,
    extract_value: int,
    backend: Backend,
) -> gpd.GeoDataFrame:
    """Prepare the job dataframe to extract the data from the given input
    dataframe."""
    pipeline_log.info("Preparing the job dataframe.")

    # Filter the input dataframe to only keep the locations to extract
    samples_gdf = samples_gdf[samples_gdf["extract"] >= extract_value].copy()

    # Split the locations into chunks of max_locations
    split_dfs = []
    pipeline_log.info(
        "Performing splitting by the year...",
    )
    samples_gdf["valid_time"] = pd.to_datetime(samples_gdf.valid_time)
    samples_gdf["year"] = samples_gdf.valid_time.dt.year

    split_dfs_time = [group.reset_index() for _, group in samples_gdf.groupby("year")]
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
            description="Sentinel2 L2A observations, processed.",
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
    max_locations_per_job: int = 500,
    job_options: Optional[Dict[str, Union[str, int]]] = None,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    extract_value: int = 1,
    backend=Backend.CDSE,
    write_stac_api: bool = False,
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
    backend : _type_, optional
        cloud backend where to run the extractions, by default Backend.CDSE
    write_stac_api : bool, optional
        Save metadata of extractions to STAC API (requires authentication), by default False

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
    else:
        # Load the input dataframe and build the job dataframe
        samples_gdf = load_dataframe(samples_df_path)
        pipeline_log.info("Creating new job tracking dataframe.")
        job_df = prepare_job_dataframe(
            samples_gdf, collection, max_locations_per_job, extract_value, backend
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
    return


def _merge_extraction_jobs(
    collection: ExtractionCollection,
    output_folder: Path,
    samples_df_path: Path,
) -> None:

    # Merge the extraction jobs
    pipeline_log.info("Merging the extraction jobs.")

    if collection == ExtractionCollection.POINT_WORLDCEREAL:
        pipeline_log.info("Merging Geoparquet results...")
        ref_id = Path(samples_df_path).stem
        merge_output_files_point_worldcereal(output_folder=output_folder, ref_id=ref_id)
        pipeline_log.info("Geoparquet results merged successfully.")

    return


def run_extractions(
    collection: ExtractionCollection,
    output_folder: Path,
    samples_df_path: Path,
    max_locations_per_job: int = 500,
    job_options: Optional[Dict[str, Union[str, int]]] = None,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    extract_value: int = 1,
    backend=Backend.CDSE,
    write_stac_api: bool = False,
) -> Path:
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

    Returns
    -------
    Path
        Path to the job tracking dataframe
    """
    pipeline_log.info("Starting the extractions workflow...")

    # Prepare the extraction jobs
    job_manager, job_df, datacube_fn, tracking_df_path = _prepare_extraction_jobs(
        collection,
        output_folder,
        samples_df_path,
        max_locations_per_job=max_locations_per_job,
        job_options=job_options,
        parallel_jobs=parallel_jobs,
        restart_failed=restart_failed,
        extract_value=extract_value,
        backend=backend,
        write_stac_api=write_stac_api,
    )

    # Run the extraction jobs
    _run_extraction_jobs(job_manager, job_df, datacube_fn, tracking_df_path)

    # Merge the extraction jobs (for point extractions)
    _merge_extraction_jobs(collection, output_folder, samples_df_path)

    pipeline_log.info("Extractions workflow completed.")

    return tracking_df_path
