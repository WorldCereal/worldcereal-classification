"""Extract S1 data using OpenEO-GFMAP package."""
import argparse
import json
import logging
import os
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
import xarray as xr
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.fetching.s1 import build_sentinel1_grd_extractor
from openeo_gfmap.manager import _log
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import split_job_hex
from openeo_gfmap.stac import AUXILIARY
from shapely.geometry import Point

# Logger for this current pipeline
_pipeline_log: Optional[logging.Logger] = None


def _setup_logger(level=logging.INFO) -> None:
    global _pipeline_log
    """Setup the logger from the openeo_gfmap package to the assigned level."""
    _pipeline_log = logging.getLogger("pipeline_sar")

    _pipeline_log.setLevel(level)
    _log.setLevel(level)

    stream_handler = logging.StreamHandler()
    _log.addHandler(stream_handler)
    _pipeline_log.addHandler(stream_handler)

    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s:  %(message)s")
    stream_handler.setFormatter(formatter)

    # Exclude the other loggers from other libraries
    class ManagerLoggerFilter(logging.Filter):
        """Filter to only accept the OpenEO-GFMAP manager logs."""

        def filter(self, record):
            return record.name in [_log.name, _pipeline_log.name]

    stream_handler.addFilter(ManagerLoggerFilter())


def _buffer_geometry(
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


def _filter_extract_true(geometries: geojson.FeatureCollection) -> gpd.GeoDataFrame:
    """Remove all the geometries from the Feature Collection that have the property field `extract` set to `False`"""
    return geojson.FeatureCollection(
        [f for f in geometries.features if f.properties.get("extract", False)]
    )


def _upload_geoparquet_artifactory(gdf: gpd.GeoDataFrame, name: str) -> str:
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


def _get_job_nb_polygons(row: pd.Series) -> int:
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

    s2_tile_id = row.s2_tile
    h3index = row.h3index
    epsg = s2_grid[s2_grid.tile == s2_tile_id].iloc[0].epsg

    subfolder = root_folder / ref_id / h3index / sample_id
    return (
        subfolder
        / f"{row.out_prefix}_{sample_id}_{epsg}_{row.start_date}_{row.end_date}{row.out_extension}"
    )


def create_job_dataframe(
    backend: Backend, split_jobs: List[gpd.GeoDataFrame], prefix: str = "S1-SIGMA0-10m"
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    columns = [
        "backend_name",
        "out_prefix",
        "out_extension",
        "start_date",
        "end_date",
        "s2_tile",
        "h3index",
        "geometry",
    ]
    rows = []
    for job in split_jobs:
        # Compute the average in the valid date and make a buffer of 1.5 year around
        median_time = pd.to_datetime(job.valid_date).mean()
        start_date = median_time - pd.Timedelta(days=275)  # A bit more than 9 months
        end_date = median_time + pd.Timedelta(days=275)  # A bit more than 9 months
        s2_tile = job.tile.iloc[0]  # Job dataframes are split depending on the
        h3index = job.h3index.iloc[0]

        rows.append(
            pd.Series(
                dict(
                    zip(
                        columns,
                        [
                            backend.value,
                            prefix,
                            ".nc",
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                            s2_tile,
                            h3index,
                            job.to_json(),
                        ],
                    )
                )
            )
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
    geometry = _filter_extract_true(geometry)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Performs a buffer of 64 px around the geometry
    geometry_df = _buffer_geometry(geometry)
    spatial_extent_url = _upload_geoparquet_artifactory(geometry_df, row.name)

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    # Create the job to extract S2
    extraction_parameters = {
        "target_resolution": 20,
    }

    # Initialize the fetching utilities in GFMAP to perfrom S1 extraction and
    # backscatter computation.
    extractor = build_sentinel1_grd_extractor(
        backend_context=backend_context,
        bands=["S1-VV", "S1-VH"],
        fetch_type=FetchType.POLYGON,
        **extraction_parameters,
    )

    cube = extractor.get_cube(
        connection=connection,
        spatial_context=spatial_extent_url,
        temporal_context=temporal_context,
    )

    # Additional values to generate the BatcJob name
    h3index = geometry.features[0].properties["h3index"]
    valid_date = geometry.features[0].properties["valid_date"]

    job_options = {
        "executor-memory": executor_memory,
        "executor-memoryOverhead": executor_memory_overhead,
    }
    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_S1_{h3index}_{valid_date}",
        sample_by_feature=True,
        job_options=job_options,
    )


def add_item_asset(related_item: pystac.Item, path: Path):
    asset = AUXILIARY.create_asset(href=path.as_posix())
    related_item.add_asset("auxiliary", asset)


def post_job_action(
    job_items: List[pystac.Item], row: pd.Series, parameters: dict = {}
) -> list:
    base_gpd = gpd.GeoDataFrame.from_features(json.loads(row.geometry)).set_crs(
        epsg=4326
    )
    assert len(base_gpd[base_gpd.extract]) == len(
        job_items
    ), "The number of result paths should be the same as the number of geometries"
    extracted_gpd = base_gpd[base_gpd.extract].reset_index(drop=True)
    # In this case we want to burn the metadata in a new file in the same folder as the S2 product
    for idx, item in enumerate(job_items):
        sample_id = extracted_gpd.iloc[idx].sample_id
        ref_id = extracted_gpd.iloc[idx].ref_id
        valid_date = extracted_gpd.iloc[idx].valid_date
        h3index = extracted_gpd.iloc[idx].h3index

        item_asset_path = Path(list(item.assets.values())[0].href)
        # Read information from the item file (could also read it from the item object metadata)
        result_ds = xr.open_dataset(item_asset_path, chunks="auto")

        # Add some metadata to the result_df netcdf file
        result_ds.attrs.update(
            {
                "start_date": row.start_date,
                "end_date": row.end_date,
                "valid_date": valid_date,
                "GFMAP_version": version("openeo_gfmap"),
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": f"Sentinel1 GRD observations for sample: {sample_id}, unprocessed.",
                "title": f"Sentinel1 GRD - {sample_id}",
                "sample_id": sample_id,
                "ref_id": ref_id,
                "spatial_resolution": "10m",
                "h3index": h3index,
            }
        )
        result_ds.to_netcdf(item_asset_path)

    return job_items


if __name__ == "__main__":
    _setup_logger()

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
        default=50,
        help="Maximum number of locations to extract per job.",
    )
    parser.add_argument(
        "--memory", type=str, default="5G", help="Memory to allocate for the executor."
    )
    parser.add_argument(
        "--memory-overhead",
        type=str,
        default="2G",
        help="Memory overhead to allocate for the executor.",
    )

    args = parser.parse_args()

    tracking_df_path = Path(args.output_path) / "job_tracking.csv"

    # Load the input dataframe, and perform dataset splitting using the h3 tile
    # to respect the area of interest. Also filters out the jobs that have
    # no location with the extract=True flag.
    _pipeline_log.info("Loading input dataframe from %s.", args.input_df)

    input_df = gpd.read_file(args.input_df)

    split_dfs = split_job_hex(input_df, max_points=args.max_locations)
    split_dfs = [df for df in split_dfs if df.extract.any()]

    job_df = create_job_dataframe(Backend.CDSE, split_dfs)

    _pipeline_log.warning(
        "Sub-sampling the job dataframe for testing. Remove this for production."
    )

    # Setup the memory parameters for the job creator.
    create_datacube_sar = partial(
        create_datacube_sar,
        executor_memory=args.memory,
        executor_memory_overhead=args.memory_overhead,
    )

    # Setup the s2 grid for the output path generation function
    generate_output_path = partial(
        generate_output_path,
        s2_grid=gpd.read_file("/data/users/Public/couchard/s2grid_bounds.geojson"),
    )

    manager = GFMAPJobManager(
        output_dir=args.output_path,
        output_path_generator=generate_output_path,
        post_job_action=None,  # No post-job action required for S1
        collection_id="SENTINEL2-EXTRACTION",
        collection_description=("Sentinel-2 and Auxiliary data extraction example."),
        poll_sleep=60,
        n_threads=2,
        post_job_params={},
    )

    manager.add_backend(Backend.CDSE.value, cdse_connection, parallel_jobs=6)

    _pipeline_log.info("Launching the jobs from the manager.")
    manager.run_jobs(job_df, create_datacube_sar, tracking_df_path)
