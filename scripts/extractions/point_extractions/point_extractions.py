"""Extract point data using OpenEO-GFMAP package."""

import argparse
import logging
from functools import partial
from pathlib import Path
from typing import List, Optional

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import split_job_s2grid

from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs

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
def generate_output_path(root_folder: Path, geometry_index: int, row: pd.Series):
    features = geojson.loads(row.geometry)
    sample_id = features[geometry_index].properties.get("sample_id", None)
    if sample_id is None:
        sample_id = features[geometry_index].properties["sampleID"]

    s2_tile_id = row.s2_tile

    subfolder = root_folder / s2_tile_id
    return subfolder / f"{sample_id}{row.out_extension}"


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
        median_time = pd.to_datetime(job.valid_time).mean()
        # A bit more than 9 months
        start_date = median_time - pd.Timedelta(days=275)
        # A bit more than 9 months
        end_date = median_time + pd.Timedelta(days=275)
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


def create_datacube(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    executor_memory: str = "3G",
    executor_memory_overhead: str = "5G",
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Load the temporal and spatial extent
    temporal_extent = TemporalContext(row.start_date, row.end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    inputs = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=geometry,
        temporal_extent=temporal_extent,
        fetch_type=FetchType.POINT,
    )

    # Finally, create a vector cube based on the Point geometries
    cube = inputs.aggregate_spatial(geometries=geometry, reducer="mean")

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_points = get_job_nb_points(row)
    pipeline_log.debug("Number of polygons to extract %s", number_points)

    job_options = {
        "executor-memory": executor_memory,
        "executor-memoryOverhead": executor_memory_overhead,
        "soft-error": True,
    }
    return cube.create_job(
        out_format="Parquet",
        title=f"GFMAP_Feature_Extraction_{row.s2_tile}",
        job_options=job_options,
    )


def post_job_action(
    job_items: List[pystac.Item], row: pd.Series, parameters: dict = None
) -> list:
    for idx, item in enumerate(job_items):
        item_asset_path = Path(list(item.assets.values())[0].href)

        gdf = gpd.read_parquet(item_asset_path)

        # Convert the dates to datetime format
        gdf["date"] = pd.to_datetime(gdf["date"])

        # Convert band dtype to uint16 (temporary fix)
        # TODO: remove this step when the issue is fixed on the OpenEO backend
        bands = [
            "S2-L2A-B02",
            "S2-L2A-B03",
            "S2-L2A-B04",
            "S2-L2A-B05",
            "S2-L2A-B06",
            "S2-L2A-B07",
            "S2-L2A-B08",
            "S2-L2A-B11",
            "S2-L2A-B12",
            "S1-SIGMA0-VH",
            "S1-SIGMA0-VV",
            "COP-DEM",
            "AGERA5-PRECIP",
            "AGERA5-TMEAN",
        ]
        gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

        gdf.to_parquet(item_asset_path, index=False)

    return job_items


if __name__ == "__main__":
    setup_logger()

    parser = argparse.ArgumentParser(
        description="S2 point extractions with OpenEO-GFMAP package."
    )
    parser.add_argument(
        "output_path", type=Path, help="Path where to save the extraction results."
    )

    # TODO: get the reference data from the RDM API.
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
        default="5G",
        help="Memory overhead to allocate for the executor.",
    )

    args = parser.parse_args()

    tracking_df_path = Path(args.output_path) / "job_tracking.csv"

    # Load the input dataframe, and perform dataset splitting using the h3 tile
    # to respect the area of interest. Also filters out the jobs that have
    # no location with the extract=True flag.
    pipeline_log.info("Loading input dataframe from %s.", args.input_df)

    input_df = gpd.read_parquet(args.input_df)

    split_dfs = split_job_s2grid(input_df, max_points=args.max_locations)
    split_dfs = [df for df in split_dfs if df.extract.any()]

    job_df = create_job_dataframe(Backend.CDSE, split_dfs).iloc[
        [2]
    ]  # TODO: remove iloc

    # Setup the memory parameters for the job creator.
    create_datacube = partial(
        create_datacube,
        executor_memory=args.memory,
        executor_memory_overhead=args.memory_overhead,
    )

    manager = GFMAPJobManager(
        output_dir=args.output_path,
        output_path_generator=generate_output_path,
        post_job_action=post_job_action,
        collection_id="POINT-FEATURE-EXTRACTION",
        collection_description="Worldcereal point feature extraction.",
        poll_sleep=60,
        n_threads=2,
        post_job_params={},
        restart_failed=True,
    )

    manager.add_backend(Backend.CDSE.value, cdse_connection, parallel_jobs=2)

    pipeline_log.info("Launching the jobs from the manager.")
    manager.run_jobs(job_df, create_datacube, tracking_df_path)
