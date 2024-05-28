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
from typing import List, Optional, Union

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
import requests
import xarray as xr
from openeo.processes import ProcessBuilder, array_create
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.fetching.s2 import build_sentinel2_l2a_extractor
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import (
    split_job_s2grid,
)
from openeo_gfmap.preprocessing import (
     median_compositing, 
     linear_interpolation
)
from shapely.geometry import Point

# Logger for this current pipeline
pipeline_log: Optional[logging.Logger] = None


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

def filter_extract_true(geometries: geojson.FeatureCollection) -> geojson.FeatureCollection:
    """Remove all the geometries from the Feature Collection that have the property field `extract` set to `False`"""
    return geojson.FeatureCollection(
        [f for f in geometries.features if f.properties.get("extract", 0) == 1]
    )


def get_job_nb_points(row: pd.Series) -> int:
    """Get the number of points in the geometry."""
    return len(
        list(
            filter(
                lambda feat: feat.properties.get("extract"),
                geojson.loads(row.geometry)["features"],
            )
        )
    )

# TODO: this is an example output_path. Adjust this function to your needs for production.
def generate_output_path(
    root_folder: Path, geometry_index: int, row: pd.Series
):
    features = geojson.loads(row.geometry)
    sample_id = features[geometry_index].properties.get("sample_id", None)
    if sample_id is None:
        sample_id = features[geometry_index].properties["sampleID"]

    s2_tile_id = row.s2_tile
    
    subfolder = root_folder / s2_tile_id 
    return (
        subfolder
        / f"{row.out_prefix}_{sample_id}{row.out_extension}"
    )


def create_job_dataframe(
    backend: Backend, split_jobs: List[gpd.GeoDataFrame]
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    columns = [
        "backend_name",
        "out_extension",
        "start_date",
        "end_date",
        "s2_tile",
        "geometry",
    ]
    rows = []
    for job in split_jobs:
        # Compute the average in the valid date and make a buffer of 1.5 year around
        median_time = pd.to_datetime(job.valid_date).mean()
        start_date = median_time - pd.Timedelta(days=275)  # A bit more than 9 months
        end_date = median_time + pd.Timedelta(days=275)  # A bit more than 9 months
        s2_tile = job.tile.iloc[0] 
        rows.append(
            pd.Series(
                dict(
                    zip(
                        columns,
                        [
                            backend.value,
                            ".parquet",
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                            s2_tile,
                            job.to_json(),
                        ],
                    )
                )
            )
        )

    return pd.DataFrame(rows)

# TODO: this is a temporary function. It will be replaced by worldcereal_preprocessed_inputs_gfmap in preprocessing.py from worldcereal-classification.openeo
def masked_cube(
        connection: openeo.Connection,
        bands: List[str],
        temporal_extent: TemporalContext,
        spatial_extent: Union[geojson.FeatureCollection, dict],
        backend_context: BackendContext,
        fetch_type: FetchType
)-> openeo.DataCube:
    """Create an openeo Datacube with the SCL dilation mask applied to the S2 data."""

    # Extract the SCL collection only and calculate the dilation mask
    scl_cube_properties = {"eo:cloud_cover": lambda val: val <= 95.0}

    scl_cube = connection.load_collection(
        collection_id="SENTINEL2_L2A",
        bands=["SCL"],
        temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
        spatial_extent=dict(spatial_extent) if fetch_type == FetchType.TILE else None,
        properties=scl_cube_properties,
    )

    # Resample to 10m resolution for the SCL layer
    scl_cube = scl_cube.resample_spatial(10)

    # Compute the SCL dilation mask
    scl_dilated_mask = scl_cube.process(
        "to_scl_dilation_mask",
        data=scl_cube,
        scl_band_name="SCL",
        kernel1_size=17,  # 17px dilation on a 10m layer
        kernel2_size=77,  # 77px dilation on a 10m layer
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3,
    ).rename_labels("bands", ["S2-L2A-SCL_DILATED_MASK"])

    # Create the job to extract S2
    extraction_parameters = {
        "target_resolution": 10,  
        "load_collection": {
            "eo:cloud_cover": lambda val: val <= 95.0,
        },
    }

    # Immediately apply the mask 
    extraction_parameters["pre_mask"] = scl_dilated_mask

    extractor = build_sentinel2_l2a_extractor(
        backend_context,
        bands=bands,
        fetch_type=fetch_type,
        **extraction_parameters,
    )

    return extractor.get_cube(connection, spatial_extent, temporal_extent)

def create_datacube(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    executor_memory: str = "5G",
    executor_memory_overhead: str = "2G",
):
    """Creates an OpenEO BatchJob from the given row information.
    """

    # Load the temporal and spatial extent
    temporal_extent = TemporalContext(row.start_date, row.end_date)
    spatial_extent = geojson.loads(row.geometry)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)

    # Filter the geometry to the rows with the extract only flag
    geometry = filter_extract_true(geometry)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    # Select some bands to download (chosen at random at this point)
    bands_to_download = [
        "S2-L2A-B04",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B09",
        "S2-L2A-B11",
        "S2-L2A-B12",
    ]

    fetch_type = FetchType.POINT 

    cube = masked_cube(connection=connection,
                       bands=bands_to_download,
                       temporal_extent=temporal_extent,
                       spatial_extent=spatial_extent,
                       backend_context=backend_context,
                       fetch_type=fetch_type)
    
    # Create monthly median composites
    cube = median_compositing(cube=cube,
                              period="month")
    # Perform linear interpolation
    cube = linear_interpolation(cube)

    # In this case the features will be the average of the bands/NDVI, so just take the average:
    # cube = cube.reduce_dimension(dimension="t", reducer="mean")
    

    def time_to_bands(input_timeseries:ProcessBuilder):
        tsteps = array_create(data=[input_timeseries.array_element(i) for i in range(20)])
        return tsteps

    cube = cube.apply_dimension(dimension='t',
                                target_dimension='bands',
                                process=time_to_bands)

    tstep_labels = [f'{band}_t{i}' for band in bands_to_download for i in range(20)]
    cube = cube.rename_labels('bands', tstep_labels)

    # Finally, create a vector cube based on the Point geometries
    cube = cube.aggregate_spatial(geometries=spatial_extent, reducer="mean")

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_points = get_job_nb_points(row)
    pipeline_log.debug("Number of polygons to extract %s", number_points)

    job_options = {
        "executor-memory": executor_memory,
        "executor-memoryOverhead": executor_memory_overhead,
    }
    return cube.create_job(
        out_format="Parquet",
        title=f"GFMAP_Feature_Extraction_S2_{row.s2_tile}",
        job_options=job_options
    )


if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(
        description="S2 point extractions with OpenEO-GFMAP package."
    )
    parser.add_argument(
        "output_path", type=Path, help="Path where to save the extraction results."
    )
    parser.add_argument(
        "input_df", type=str, help="Path to the input dataframe for the training data."
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=500,
        help="Maximum number of locations to extract per job.",
    )
    parser.add_argument(
        "--memory", type=str, default="3G", help="Memory to allocate for the executor."
    )
    parser.add_argument(
        "--memory-overhead",
        type=str,
        default="1G",
        help="Memory overhead to allocate for the executor.",
    )

    args = parser.parse_args()

    tracking_df_path = Path(args.output_path) / "job_tracking.csv"

    # Load the input dataframe, and perform dataset splitting using the h3 tile
    # to respect the area of interest. Also filters out the jobs that have
    # no location with the extract=True flag.
    pipeline_log.info("Loading input dataframe from %s.", args.input_df)

    input_df = gpd.read_file(args.input_df)

    split_dfs = split_job_s2grid(input_df, max_points=args.max_locations)
    split_dfs = [df for df in split_dfs if df.extract.any()]

    job_df = create_job_dataframe(Backend.CDSE, split_dfs).head(1)  # TODO: remove head

    # Setup the memory parameters for the job creator.
    create_datacube = partial(
        create_datacube,
        executor_memory=args.memory,
        executor_memory_overhead=args.memory_overhead,
    )

    # Setup the s2 grid for the output path generation function
    # generate_output_path = partial(
    #     generate_output_path,
    #     s2_grid=load_s2_grid(),
    # )

    manager = GFMAPJobManager(
        output_dir=args.output_path,
        output_path_generator=generate_output_path,
        post_job_action=None,  
        collection_id="SENTINEL2-POINT-FEATURE-EXTRACTION",
        collection_description="Sentinel-2 basic point feature extraction.",
        poll_sleep=60,
        n_threads=2,
        post_job_params={},
    )

    manager.add_backend(
        Backend.CDSE.value, cdse_connection, parallel_jobs=2
    )

    pipeline_log.info("Launching the jobs from the manager.")
    manager.run_jobs(job_df, create_datacube, tracking_df_path)

