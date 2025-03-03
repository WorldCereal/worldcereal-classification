"""Common functions used by extraction scripts."""

import json
import logging
import os
import shutil
from datetime import datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, List, Union

import geojson
import geopandas as gpd
import pandas as pd
import pystac
import requests
import xarray as xr
from openeo_gfmap import Backend
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import load_s2_grid, split_job_s2grid
from shapely import Point

from worldcereal.extract.patch_meteo import (
    create_job_dataframe_patch_meteo,
    create_job_patch_meteo,
)
from worldcereal.extract.patch_s1 import (
    create_job_dataframe_patch_s1,
    create_job_patch_s1,
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
from worldcereal.stac.constants import ExtractionCollection
from worldcereal.stac.stac_api_interaction import (
    StacApiInteraction,
    VitoStacApiAuthentication,
)

from worldcereal.extract.patch_worldcereal import (  # isort: skip
    create_job_dataframe_patch_worldcereal,
    create_job_patch_worldcereal,
    post_job_action_patch_worldcereal,
    generate_output_path_patch_worldcereal,
)


S2GRID = ...

# Logger used for the pipeline
pipeline_log = logging.getLogger("extraction_pipeline")

pipeline_log.setLevel(level=logging.INFO)

stream_handler = logging.StreamHandler()
pipeline_log.addHandler(stream_handler)

formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s:  %(message)s")
stream_handler.setFormatter(formatter)


# Exclude the other loggers from other libraries
class ManagerLoggerFilter(logging.Filter):
    """Filter to only accept the OpenEO-GFMAP manager logs."""

    def filter(self, record):
        return record.name in [pipeline_log.name]


stream_handler.addFilter(ManagerLoggerFilter())


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
    s2_grid: gpd.GeoDataFrame,
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


def buffer_geometry(
    geometries: geojson.FeatureCollection, distance_m: int = 320
) -> gpd.GeoDataFrame:
    """For each geometry of the colleciton, perform a square buffer of 320
    meters on the centroid and return the GeoDataFrame. Before buffering,
    the centroid is clipped to the closest 20m multiplier in order to stay
    aligned with the Sentinel-1 pixel grid.
    """
    gdf = gpd.GeoDataFrame.from_features(geometries).set_crs(epsg=4326)
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)

    # Perform the buffering operation
    gdf["geometry"] = gdf.centroid.apply(
        lambda point: Point(round(point.x / 20.0) * 20.0, round(point.y / 20.0) * 20.0)
    ).buffer(
        distance=distance_m, cap_style=3
    )  # Square buffer

    return gdf


def filter_extract_true(
    geometries: geojson.FeatureCollection, extract_value: int = 1
) -> gpd.GeoDataFrame:
    """Remove all the geometries from the Feature Collection that have the property field `extract` set to `False`"""
    return geojson.FeatureCollection(
        [
            f
            for f in geometries.features
            if f.properties.get("extract", 0) == extract_value
        ]
    )


def upload_geoparquet_artifactory(
    gdf: gpd.GeoDataFrame, name: str, collection: str = ""
) -> str:
    """Upload the given GeoDataFrame to artifactory and return the URL of the
    uploaded file. Necessary as a workaround for Polygon sampling in OpenEO
    using custom CRS.
    """
    # Save the dataframe as geoparquet to upload it to artifactory
    temporary_file = NamedTemporaryFile()
    gdf.to_parquet(temporary_file.name)

    artifactory_username = os.getenv("ARTIFACTORY_USERNAME")
    artifactory_password = os.getenv("ARTIFACTORY_PASSWORD")

    if not artifactory_username or not artifactory_password:
        raise ValueError(
            "Artifactory credentials not found. Please set ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD."
        )

    headers = {"Content-Type": "application/octet-stream"}

    upload_url = f"https://artifactory.vgt.vito.be/artifactory/auxdata-public/gfmap-temp/openeogfmap_dataframe_{collection}{name}.parquet"

    with open(temporary_file.name, "rb") as f:
        response = requests.put(
            upload_url,
            headers=headers,
            data=f,
            auth=(artifactory_username, artifactory_password),
            timeout=180,
        )

    response.raise_for_status()

    return upload_url


def get_job_nb_polygons(row: pd.Series) -> int:
    """Get the number of polygons in the geometry."""
    return len(
        list(
            filter(
                lambda feat: feat.properties.get("extract"),
                geojson.loads(row.geometry)["features"],
            )
        )
    )


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
    memory: Union[str, None],
    python_memory: Union[str, None],
    max_executors: Union[int, None],
    write_stac_api: bool,
) -> tuple[Callable, Callable, Callable]:
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


def prepare_extraction_jobs(
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
    write_stac_api: bool = True,
):

    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    tracking_df_path = output_folder / "job_tracking.csv"

    # Load the input dataframe and build the job dataframe
    input_gdf = load_dataframe(input_df)

    job_df = None
    if not tracking_df_path.exists():
        job_df = prepare_job_dataframe(
            input_gdf, collection, max_locations_per_job, extract_value, backend
        )

    # Setup the extraction functions
    pipeline_log.info("Setting up the extraction functions.")
    datacube_fn, path_fn, post_job_fn = setup_extraction_functions(
        collection, extract_value, memory, python_memory, max_executors, write_stac_api
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

    return job_manager, job_df, datacube_fn, tracking_df_path


def run_extraction_jobs(job_manager, job_df, datacube_fn, tracking_df_path):

    # Run the extraction jobs
    pipeline_log.info("Running the extraction jobs.")
    job_manager.run_jobs(job_df, datacube_fn, tracking_df_path)
    return


def merge_extraction_jobs(
    collection: ExtractionCollection,
    output_folder,
    input_df,
):

    # Merge the extraction jobs
    pipeline_log.info("Merging the extraction jobs.")

    if collection == ExtractionCollection.POINT_WORLDCEREAL:
        pipeline_log.info("Merging Geoparquet results...")
        ref_id = Path(input_df).stem
        merge_output_files_point_worldcereal(output_folder=output_folder, ref_id=ref_id)
        pipeline_log.info("Geoparquet results merged successfully.")


def run_extractions(collection, output_folder, input_df):
    """This will be the main extraction function, chaining all the steps together"""

    job_manager, job_df, datacube_fn, tracking_df_path = prepare_extraction_jobs()

    run_extraction_jobs(job_manager, job_df, datacube_fn, tracking_df_path)

    merge_extraction_jobs(collection, output_folder, input_df)

    return tracking_df_path
