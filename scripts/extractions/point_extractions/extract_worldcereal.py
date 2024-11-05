"""Extract S1, S2, METEO and DEM point data using OpenEO-GFMAP package."""

from pathlib import Path
from typing import List, Optional

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from tqdm import tqdm

from worldcereal.openeo.extract import get_job_nb_polygons, pipeline_log
from worldcereal.openeo.preprocessing import (
    worldcereal_preprocessed_inputs,
    correct_temporal_context,
)

# from worldcereal.openeo.extract_common import pipeline_log


def generate_output_path_point(root_folder: Path, geometry_index: int, row: pd.Series):
    """
    For point extractions, only one asset (a geoparquet file) is generated per job.
    Therefore geometry_index is always 0.
    It has to be included in the function signature to be compatible with the GFMapJobManager.
    """
    features = geojson.loads(row.geometry)
    ref_id = features[geometry_index].properties["ref_id"]

    s2_tile_id = row.s2_tile

    subfolder = root_folder / ref_id / s2_tile_id

    subfolder.mkdir(parents=True, exist_ok=True)

    # Subfolder is not necessarily unique, so we create numbered folders.
    if not any(subfolder.iterdir()):
        real_subfolder = subfolder / "0"
    else:
        i = 0
        while (subfolder / str(i)).exists():
            i += 1
        real_subfolder = subfolder / str(i)

    return real_subfolder / f"point_extractions{row.out_extension}"


def create_job_dataframe_point(
    backend: Backend, split_jobs: List[gpd.GeoDataFrame]
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    rows = []
    for job in tqdm(split_jobs):
        min_time = job.valid_time.min()
        max_time = job.valid_time.max()
        # 9 months before and after the valid time
        start_date = (min_time - pd.Timedelta(days=275)).to_pydatetime()
        end_date = (max_time + pd.Timedelta(days=275)).to_pydatetime()

        s2_tile = job.tile.iloc[0]
        h3_l3_cell = job.h3_l3_cell.iloc[0]

        # Convert dates to string format
        start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime(
            "%Y-%m-%d"
        )

        # Set back the valid_time in the geometry as string
        job["valid_time"] = job.valid_time.dt.strftime("%Y-%m-%d")

        variables = {
            "backend_name": backend.value,
            "out_prefix": "point-extraction",
            "out_extension": ".geoparquet",
            "start_date": start_date,
            "end_date": end_date,
            "s2_tile": s2_tile,
            "h3_l3_cell": h3_l3_cell,
            "geometry": job.to_json(),
        }

        rows.append(pd.Series(variables))

    return pd.DataFrame(rows)


def create_datacube_point(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    executor_memory: str = "5G",
    python_memory: str = "2G",
    max_executors: int = 22,
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Load the temporal and spatial extent
    temporal_extent = correct_temporal_context(
        TemporalContext(row.start_date, row.end_date)
    )

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
        validate_temporal_context=False,
    )

    # Finally, create a vector cube based on the Point geometries
    cube = inputs.aggregate_spatial(geometries=geometry, reducer="mean")

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_points = get_job_nb_polygons(row)
    if pipeline_log is not None:
        pipeline_log.debug("Number of polygons to extract %s", number_points)

    job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": executor_memory,
        "python-memory": python_memory,
        "executor-cores": "1",
        "max-executors": max_executors,
        "soft-errors": "true",
    }

    return cube.create_job(
        out_format="Parquet",
        title=f"Worldcereal_Point_Extraction_{row.s2_tile}",
        job_options=job_options,
    )


def post_job_action_point(
    job_items: List[pystac.Item], row: pd.Series, parameters: Optional[dict] = None
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
            "elevation",
            "AGERA5-PRECIP",
            "AGERA5-TMEAN",
        ]
        gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

        gdf.to_parquet(item_asset_path, index=False)

    return job_items


# if __name__ == "__main__":
#     setup_logger()

#     parser = argparse.ArgumentParser(
#         description="S2 point extractions with OpenEO-GFMAP package."
#     )
#     parser.add_argument(
#         "output_path", type=Path, help="Path where to save the extraction results."
#     )

#     # TODO: get the reference data from the RDM API.
#     parser.add_argument(
#         "input_df", type=str, help="Path to the input dataframe for the training data."
#     )
#     parser.add_argument(
#         "--max_locations",
#         type=int,
#         default=500,
#         help="Maximum number of locations to extract per job.",
#     )
#     parser.add_argument(
#         "--memory", type=str, default="3G", help="Memory to allocate for the executor."
#     )
#     parser.add_argument(
#         "--memory-overhead",
#         type=str,
#         default="5G",
#         help="Memory overhead to allocate for the executor.",
#     )

#     args = parser.parse_args()

#     tracking_df_path = Path(args.output_path) / "job_tracking.csv"

#     # Load the input dataframe, and perform dataset splitting using the h3 tile
#     # to respect the area of interest. Also filters out the jobs that have
#     # no location with the extract=True flag.
#     if pipeline_log is not None:
#         pipeline_log.info("Loading input dataframe from %s.", args.input_df)

#     input_df = gpd.read_parquet(args.input_df)

#     split_dfs = split_job_s2grid(input_df, max_points=args.max_locations)
#     split_dfs = [df for df in split_dfs if df.extract.any()]

#     job_df = create_job_dataframe(Backend.CDSE, split_dfs).iloc[
#         [2]
#     ]  # TODO: remove iloc

#     # Setup the memory parameters for the job creator.
#     create_datacube = partial(
#         create_datacube,
#         executor_memory=args.memory,
#         executor_memory_overhead=args.memory_overhead,
#     )

#     manager = GFMAPJobManager(
#         output_dir=args.output_path,
#         output_path_generator=generate_output_path,
#         post_job_action=post_job_action,
#         collection_id="POINT-FEATURE-EXTRACTION",
#         collection_description="Worldcereal point feature extraction.",
#         poll_sleep=60,
#         n_threads=2,
#         post_job_params={},
#         restart_failed=True,
#     )

#     manager.add_backend(Backend.CDSE.value, cdse_connection, parallel_jobs=2)

#     if pipeline_log is not None:
#         pipeline_log.info("Launching the jobs from the manager.")
#     manager.run_jobs(job_df, create_datacube, tracking_df_path)
