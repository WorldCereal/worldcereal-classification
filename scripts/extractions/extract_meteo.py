"""Extract AGERA5 (Meteo) data using OpenEO-GFMAP package."""
import argparse
from functools import partial
from pathlib import Path

import geojson
import geopandas as gpd
import openeo
import pandas as pd
from extract_sar import (
    _buffer_geometry,
    _filter_extract_true,
    _pipeline_log,
    _setup_logger,
    _upload_geoparquet_artifactory,
    create_job_dataframe,
    generate_output_path,
)
from openeo_gfmap import Backend, TemporalContext
from openeo_gfmap.backend import vito_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import (
    _append_h3_index,
    _load_s2_grid,
    split_job_s2grid,
)


def create_datacube_meteo(
    row: pd.Series,
    connection: openeo.DataCube,
    provider=None,
    connection_provider=None,
    executor_memory: str = "2G",
    executor_memory_overhead: str = "1G",
) -> gpd.GeoDataFrame:
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
    geometry_df = _buffer_geometry(geometry, distance_m=5)
    spatial_extent_url = _upload_geoparquet_artifactory(geometry_df, row.name)

    bands_to_download = ["temperature-mean"]

    cube = connection.load_collection(
        "AGERA5",
        temporal_extent=[temporal_context.start_date, temporal_context.end_date],
        bands=bands_to_download,
    )
    filter_geometry = connection.load_url(spatial_extent_url, format="parquet")
    cube = cube.filter_spatial(filter_geometry)
    cube.rename_labels(
        dimension="bands",
        target=["AGERA5-temperature-mean"],
        source=["temperature-mean"],
    )

    h3index = geometry.features[0].properties["h3index"]
    valid_time = geometry.features[0].properties["valid_time"]

    job_options = {
        "executor-memory": executor_memory,
        "executor-memoryOverhead": executor_memory_overhead,
    }
    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_AGERA5_{h3index}_{valid_time}",
        sample_by_feature=True,
        job_options=job_options,
    )


if __name__ == "__main__":
    _setup_logger()
    from extract_sar import _pipeline_log

    parser = argparse.ArgumentParser(
        description="AGERA5 samples extraction with OpenEO-GFMAP package."
    )
    parser.add_argument(
        "output_path", type=Path, help="Path where to save the extraction results."
    )
    parser.add_argument(
        "input_df",
        type=str,
        help="Path or URL to the input dataframe for the training data.",
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=5,
        help="Maximum number of locations to extract per job.",
    )
    parser.add_argument(
        "--memory", type=str, default="5G", help="Memory to allocate for the executor."
    )
    parser.add_argument(
        "--memory-overhead",
        type=str,
        default="3G",
        help="Memory overhead to allocate for the executor.",
    )

    args = parser.parse_args()

    tracking_df_path = Path(args.output_path) / "job_tracking.csv"

    # Load the input dataframe
    _pipeline_log.info("Loading input dataframe from %s.", args.input_df)

    input_df = gpd.read_file(args.input_df)
    input_df = _append_h3_index(input_df, grid_resolution=3)

    split_dfs = split_job_s2grid(input_df, max_points=args.max_locations)
    split_dfs = [df for df in split_dfs if df.extract.any()]

    job_df = create_job_dataframe(Backend.TERRASCOPE, split_dfs, prefix="AGERA5")

    _pipeline_log.warning(
        "Sub-sampling the job dataframe for testing. Remove this for production."
    )
    # job_df = job_df.iloc[[0, 2, 3, -6]].reset_index(drop=True)
    job_df = job_df.iloc[[0]].reset_index(drop=True)

    # Setup the memory parameters for the job creator.
    create_datacube_meteo = partial(
        create_datacube_meteo,
        executor_memory=args.memory,
        executor_memory_overhead=args.memory_overhead,
    )

    # Setup the s2 grid for the output path generation function
    generate_output_path = partial(
        generate_output_path,
        s2_grid=_load_s2_grid(),
    )

    manager = GFMAPJobManager(
        output_dir=args.output_path,
        output_path_generator=generate_output_path,
        post_job_action=None,
        collection_id="AGERA5-EXTRACTION",
        collection_description="AGERA5 data extraction example.",
        poll_sleep=60,
        n_threads=2,
        post_job_params={},
    )

    manager.add_backend(Backend.TERRASCOPE.value, vito_connection, parallel_jobs=6)

    _pipeline_log.info("Launching the jobs from the manager.")
    manager.run_jobs(job_df, create_datacube_meteo, tracking_df_path)
