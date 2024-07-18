"""Extract S1 data using OpenEO-GFMAP package."""

import argparse
import json
import logging
import os
import warnings
from datetime import datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
import requests
from openeo_gfmap import (
    Backend,
    BackendContext,
    BoundingBoxExtent,
    FetchType,
    TemporalContext,
)
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import load_s2_grid, split_job_s2grid
from openeo_gfmap.preprocessing.sar import compress_backscatter_uint16
from openeo_gfmap.utils.catalogue import s1_area_per_orbitstate
from openeo_gfmap.utils.netcdf import update_nc_attributes
from shapely.geometry import Point
from tqdm import tqdm

from worldcereal.openeo.preprocessing import raw_datacube_S1

# Define the sentinel 1 asset
sentinel1_asset = pystac.extensions.item_assets.AssetDefinition(
    {
        "gsd": 20,
        "title": "Sentinel1",
        "description": "Sentinel-1 bands",
        "type": "application/x-netcdf",
        "roles": ["data"],
        "proj:shape": [32, 32],
        "raster:bands": [
            {"name": "S1-SIGMA0-VV"},
            {
                "name": "S1-SIGMA0-VH",
            },
        ],
        "cube:variables": {
            "S1-SIGMA0-VV": {"dimesions": ["time", "y", "x"], "type": "data"},
            "S1-SIGMA0-VH": {"dimesions": ["time", "y", "x"], "type": "data"},
        },
        "eo:bands": [
            {
                "name": "S1-SIGMA0-VV",
                "common_name": "VV",
            },
            {
                "name": "S1-SIGMA0-VH",
                "common_name": "VH",
            },
        ],
    }
)

S1_GRD_CATALOGUE_BEGIN_DATE = datetime(2014, 10, 1)

# Logger for this current pipeline
pipeline_log: Optional[logging.Logger] = None

PUSHOVER_API_ENDPOINT = "https://api.pushover.net/1/messages.json"


def send_notification(message: str, title: str = "OpenEO-GFMAP") -> None:
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


def setup_logger(level=logging.INFO) -> None:
    """Setup the logger from the openeo_gfmap package to the assigned level."""
    global pipeline_log
    pipeline_log = logging.getLogger("pipeline_sar")

    pipeline_log.setLevel(level)

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


def filter_extract_true(geometries: geojson.FeatureCollection) -> gpd.GeoDataFrame:
    """Remove all the geometries from the Feature Collection that have the property field `extract` set to `False`"""
    return geojson.FeatureCollection(
        [f for f in geometries.features if f.properties.get("extract", 0) == 1]
    )


def upload_geoparquet_artifactory(gdf: gpd.GeoDataFrame, name: str) -> str:
    """Upload the given GeoDataFrame to artifactory and return the URL of the
    uploaded file. Necessary as a workaround for Polygon sampling in OpenEO
    using custom CRS.
    """
    # Save the dataframe as geoparquet to upload it to artifactory
    temporary_file = NamedTemporaryFile()
    gdf.to_parquet(temporary_file.name)

    artifactory_username = os.getenv("ARTIFACTORY_USERNAME")
    artifactory_password = os.getenv("ARTIFACTORY_PASSWORD")

    headers = {"Content-Type": "application/octet-stream"}

    upload_url = f"https://artifactory.vgt.vito.be/artifactory/auxdata-public/gfmap-temp/openeogfmap_dataframe_{name}.parquet"

    with open(temporary_file.name, "rb") as f:
        response = requests.put(
            upload_url,
            headers=headers,
            data=f,
            auth=(artifactory_username, artifactory_password),
            timeout=180,
        )

    assert (
        response.status_code == 201
    ), f"Error uploading the dataframe to artifactory: {response.text}"

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


def generate_output_path(
    root_folder: Path, geometry_index: int, row: pd.Series, s2_grid: gpd.GeoDataFrame
):
    features = geojson.loads(row.geometry)
    sample_id = features[geometry_index].properties.get("sample_id", None)
    if sample_id is None:
        sample_id = features[geometry_index].properties["sampleID"]
    ref_id = features[geometry_index].properties["ref_id"]

    if "orbit_state" in row:
        orbit_state = f"_{row.orbit_state}"
    else:
        orbit_state = ""

    s2_tile_id = row.s2_tile
    h3_l3_cell = row.h3_l3_cell
    epsg = s2_grid[s2_grid.tile == s2_tile_id].iloc[0].epsg

    subfolder = root_folder / ref_id / h3_l3_cell / sample_id
    return (
        subfolder
        / f"{row.out_prefix}{orbit_state}_{sample_id}_{epsg}_{row.start_date}_{row.end_date}{row.out_extension}"
    )


def create_job_dataframe_s1(
    backend: Backend,
    split_jobs: List[gpd.GeoDataFrame],
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    rows = []
    for job in tqdm(split_jobs):
        # Compute the average in the valid date and make a buffer of 1.5 year around
        min_time = job.valid_time.min()
        max_time = job.valid_time.max()

        # Compute the average in the valid date and make a buffer of 1.5 year around
        # 9 months before and after the valid time
        start_date = (min_time - pd.Timedelta(days=275)).to_pydatetime()
        end_date = (max_time + pd.Timedelta(days=275)).to_pydatetime()

        # Impose limits due to the data availability
        start_date = max(start_date, S1_GRD_CATALOGUE_BEGIN_DATE)
        end_date = min(end_date, datetime.now())

        s2_tile = job.tile.iloc[0]  # Job dataframes are split depending on the
        h3_l3_cell = job.h3_l3_cell.iloc[0]

        # Check wherever the s2_tile is in the grid
        geometry_bbox = job.to_crs(epsg=4326).total_bounds

        area_per_orbit = s1_area_per_orbitstate(
            backend=BackendContext(backend),
            spatial_extent=BoundingBoxExtent(*geometry_bbox),
            temporal_extent=TemporalContext(start_date, end_date),
        )
        descending_area = area_per_orbit["DESCENDING"]["area"]
        ascending_area = area_per_orbit["ASCENDING"]["area"]

        # Convert dates to string format
        start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime(
            "%Y-%m-%d"
        )

        # Set back the valid_time in the geometry as string
        job["valid_time"] = job.valid_time.dt.strftime("%Y-%m-%d")

        variables = {
            "backend_name": backend.value,
            "out_prefix": "S1-SIGMA0-10m",
            "out_extension": ".nc",
            "start_date": start_date,
            "end_date": end_date,
            "s2_tile": s2_tile,
            "h3_l3_cell": h3_l3_cell,
            "geometry": job.to_json(),
        }

        if descending_area > 0:
            variables.update({"orbit_state": "DESCENDING"})
            rows.append(pd.Series(variables))

        if ascending_area > 0:
            variables.update({"orbit_state": "ASCENDING"})
            rows.append(pd.Series(variables))

        if descending_area + ascending_area == 0:
            pipeline_log.warning(
                "No S1 data available for the tile %s in the period %s - %s.",
                s2_tile,
                start_date,
                end_date,
            )

    return pd.DataFrame(rows)


def create_datacube_sar(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    executor_memory: str = "5G",
    executor_memory_overhead: str = "2G",
) -> openeo.BatchJob:
    """Creates an OpenEO BatchJob from the given row information. This job is a
    S1 patch of 32x32 pixels at 20m spatial resolution."""

    # Load the temporal extent
    start_date = row.start_date
    end_date = row.end_date
    temporal_context = TemporalContext(start_date, end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)

    # Filter the geometry to the rows with the extract only flag
    geometry = filter_extract_true(geometry)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Jobs will be run for two orbit direction
    orbit_state = row.orbit_state

    # Performs a buffer of 64 px around the geometry
    geometry_df = buffer_geometry(geometry, distance_m=320)
    spatial_extent_url = upload_geoparquet_artifactory(geometry_df, row.name)

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    # Create the job to extract S2
    cube = raw_datacube_S1(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent_url,
        temporal_extent=temporal_context,
        bands=["S1-SIGMA0-VV", "S1-SIGMA0-VH"],
        fetch_type=FetchType.POLYGON,
        target_resolution=20,
        orbit_direction=orbit_state,
    )
    cube = compress_backscatter_uint16(backend_context, cube)

    # Additional values to generate the BatcJob name
    s2_tile = row.s2_tile
    valid_time = geometry.features[0].properties["valid_time"]

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_polygons = get_job_nb_polygons(row)
    pipeline_log.debug("Number of polygons to extract %s", number_polygons)

    job_options = {
        "executor-memory": executor_memory,
        "executor-memoryOverhead": executor_memory_overhead,
        "soft-errors": "true",
    }
    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_S1_{s2_tile}_{valid_time}_{orbit_state}",
        sample_by_feature=True,
        job_options=job_options,
        feature_id_property="sample_id",
    )


def post_job_action(
    job_items: List[pystac.Item], row: pd.Series, parameters: dict = {}
) -> list:
    base_gpd = gpd.GeoDataFrame.from_features(json.loads(row.geometry)).set_crs(
        epsg=4326
    )
    assert len(base_gpd[base_gpd.extract == 1]) == len(
        job_items
    ), "The number of result paths should be the same as the number of geometries"

    extracted_gpd = base_gpd[base_gpd.extract == 1].reset_index(drop=True)
    # In this case we want to burn the metadata in a new file in the same folder as the S2 product
    for idx, item in enumerate(job_items):
        if "sample_id" in extracted_gpd.columns:
            sample_id = extracted_gpd.iloc[idx].sample_id
        else:
            sample_id = extracted_gpd.iloc[idx].sampleID

        ref_id = extracted_gpd.iloc[idx].ref_id
        valid_time = extracted_gpd.iloc[idx].valid_time
        h3_l3_cell = extracted_gpd.iloc[idx].h3_l3_cell
        s2_tile = row.s2_tile

        item_asset_path = Path(list(item.assets.values())[0].href)

        # Add some metadata to the result_df netcdf file
        new_attributes = {
            "start_date": row.start_date,
            "end_date": row.end_date,
            "valid_time": valid_time,
            "GFMAP_version": version("openeo_gfmap"),
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": parameters.get(
                "description", "Sentinel1 GRD raw observations, unprocessed."
            ),
            "title": parameters.get("title", "Sentinel1 GRD"),
            "sample_id": sample_id,
            "ref_id": ref_id,
            "spatial_resolution": parameters.get("spatial_resolution", "20m"),
            "s2_tile": s2_tile,
            "h3_l3_cell": h3_l3_cell,
        }

        # Saves the new attributes in the netcdf file
        update_nc_attributes(item_asset_path, new_attributes)

    return job_items


if __name__ == "__main__":
    setup_logger()
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")

    parser = argparse.ArgumentParser(
        description="S1 samples extraction with OpenEO-GFMAP package."
    )
    parser.add_argument(
        "output_path", type=Path, help="Path where to save the extraction results."
    )
    parser.add_argument(
        "input_df", type=Path, help="Path to the input dataframe for the training data."
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=500,
        help="Maximum number of locations to extract per job.",
    )
    parser.add_argument(
        "--memory", type=str, default="2G", help="Memory to allocate for the executor."
    )
    parser.add_argument(
        "--memory-overhead",
        type=str,
        default="3G",
        help="Memory overhead to allocate for the executor.",
    )
    parser.add_argument(
        "--restart_failed",
        action="store_true",
        help="Restart the jobs that failed in the previous run.",
    )
    parser.add_argument(
        "--recreate_jobs",
        action="store_true",
        help="Recreates the `job_tracking.csv` file even if it already exists.",
    )

    args = parser.parse_args()

    tracking_df_path = Path(args.output_path) / "job_tracking.csv"

    # Load the input dataframe, and perform dataset splitting using the h3 tile
    # to respect the area of interest. Also filters out the jobs that have
    # no location with the extract=True flag.

    if (not tracking_df_path.exists()) or args.recreate_jobs:
        if not tracking_df_path.exists():
            pipeline_log.info("No tracking dataframe found, creating a new one.")
        else:
            pipeline_log.warning("Removing the old tracking dataframe.")
            tracking_df_path.unlink()
            pipeline_log.warning("Recreating the tracking dataframe.")

        pipeline_log.info("Loading input dataframe from %s.", args.input_df)

        if args.input_df.name.endswith(".geoparquet"):
            input_df = gpd.read_parquet(args.input_df)
        else:
            input_df = gpd.read_file(args.input_df)

        split_dfs = []
        pipeline_log.info(
            "Performing splitting by the year...",
        )
        input_df["valid_time"] = pd.to_datetime(input_df.valid_time)
        input_df["year"] = input_df.valid_time.dt.year

        split_dfs_time = [group.reset_index() for _, group in input_df.groupby("year")]
        pipeline_log.info("Performing splitting by s2 grid...")
        for df in split_dfs_time:
            s2_split_df = split_job_s2grid(df, max_points=args.max_locations)
            split_dfs.extend(s2_split_df)

        # Filter all the datasets withouth any location to extract
        pipeline_log.info("Filtering out the datasets without any location to extract.")
        split_dfs = [df for df in split_dfs if (df.extract == 1).any()]

        pipeline_log.warning(
            "Sub-sampling the job dataframe for testing. Remove this for production."
        )
        split_dfs = split_dfs[:1]

        job_df = create_job_dataframe_s1(Backend.CDSE, split_dfs)
        pipeline_log.info("Created the job dataframe with %s jobs.", len(job_df))
    else:
        pipeline_log.info("Loading the tracking dataframe from %s.", tracking_df_path)
        job_df = None

    # Setup the memory parameters for the job creator.
    create_datacube_sar = partial(
        create_datacube_sar,
        executor_memory=args.memory,
        executor_memory_overhead=args.memory_overhead,
    )

    # Setup the s2 grid for the output path generation function
    generate_output_path = partial(
        generate_output_path,
        s2_grid=load_s2_grid(),
    )

    manager = GFMAPJobManager(
        output_dir=args.output_path,
        output_path_generator=generate_output_path,
        post_job_action=post_job_action,  # No post-job action required for S1
        collection_id="SENTINEL1-EXTRACTION",
        collection_description="Sentinel-1 data extraction example.",
        poll_sleep=60,
        n_threads=4,
        restart_failed=args.restart_failed,
    )

    manager.add_backend(Backend.CDSE.value, cdse_connection, parallel_jobs=20)
    manager.setup_stac(
        constellation="sentinel1", item_assets={"sentinel1": sentinel1_asset}
    )

    pipeline_log.info("Launching the jobs from the manager.")
    manager.run_jobs(job_df, create_datacube_sar, tracking_df_path)
