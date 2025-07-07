"""Extract S1, S2, METEO and DEM point data using OpenEO-GFMAP package."""

import copy
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import duckdb
import geojson
import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
import pystac
from loguru import logger
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from pandas.core.dtypes.dtypes import CategoricalDtype
from tqdm import tqdm

from worldcereal.extract.utils import get_job_nb_polygons, pipeline_log
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs

DEFAULT_JOB_OPTIONS_POINT_WORLDCEREAL = {
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "driver-cores": "1",
    "executor-memory": "1800m",
    "python-memory": "3000m",
    "executor-cores": "1",
    "max-executors": 22,
    "soft-errors": 0.1,
}


REQUIRED_ATTRIBUTES = {
    "feature_index": np.int64,
    "sample_id": str,
    "timestamp": "datetime64[ns]",
    "S2-L2A-B02": np.uint16,
    "S2-L2A-B03": np.uint16,
    "S2-L2A-B04": np.uint16,
    "S2-L2A-B05": np.uint16,
    "S2-L2A-B06": np.uint16,
    "S2-L2A-B07": np.uint16,
    "S2-L2A-B08": np.uint16,
    "S2-L2A-B8A": np.uint16,
    "S2-L2A-B11": np.uint16,
    "S2-L2A-B12": np.uint16,
    "S1-SIGMA0-VH": np.uint16,
    "S1-SIGMA0-VV": np.uint16,
    "slope": np.uint16,
    "elevation": np.uint16,
    "AGERA5-PRECIP": np.uint16,
    "AGERA5-TMEAN": np.uint16,
    "lon": np.float64,
    "lat": np.float64,
    "geometry": "geometry",
    "tile": str,
    "h3_l3_cell": str,
    "start_date": str,
    "end_date": str,
    "year": np.int64,
    "valid_time": str,
    "ewoc_code": np.int64,
    "irrigation_status": np.int64,
    "quality_score_lc": np.int64,
    "quality_score_ct": np.int64,
    "extract": np.int64,
}


def generate_output_path_point_worldcereal(
    root_folder: Path,
    geometry_index: int,
    row: pd.Series,
    asset_id: Optional[str] = None,
) -> Path:
    """Method to generate the output path for the point extractions.

    Parameters
    ----------
    root_folder : Path
        root folder where the output parquet file will be saved
    geometry_index : int
        For point extractions, only one asset (a geoparquet file) is generated per job.
        Therefore geometry_index is always 0. It has to be included in the function signature
        to be compatible with the GFMapJobManager
    row : pd.Series
        the current job row from the GFMapJobManager
    asset_id : str, optional
        Needed for compatibility with GFMapJobManager but not used.

    Returns
    -------
    Path
        output path for the point extractions parquet file
    """

    s2_tile_id = row.s2_tile
    utm_zone = str(s2_tile_id[0:2])

    # Create the subfolder to store the output
    subfolder = root_folder / utm_zone / s2_tile_id
    subfolder.mkdir(parents=True, exist_ok=True)

    # we may have multiple output files per s2_tile_id and need
    # a unique name so we use the job ID
    output_file = f"WORLDCEREAL_{root_folder.name}_{row.start_date}_{row.end_date}_{s2_tile_id}_{row.id}{row.out_extension}"

    return subfolder / output_file


def create_job_dataframe_point_worldcereal(
    backend: Backend, split_jobs: List[gpd.GeoDataFrame]
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containing all the necessary information to run the job."""
    rows = []
    for job in tqdm(split_jobs):
        min_time = job.valid_time.min()
        max_time = job.valid_time.max()

        # 9 months before and after the valid time
        start_date = (min_time - pd.Timedelta(days=275)).to_pydatetime()
        end_date = (max_time + pd.Timedelta(days=275)).to_pydatetime()

        # ensure start date is 1st day of month, end date is last day of month
        start_date = start_date.replace(day=1)
        end_date = end_date.replace(day=1) + pd.offsets.MonthEnd(0)

        s2_tile = job.tile.iloc[0]

        # Convert dates to string format
        start_date, end_date = (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        # Set back the valid_time in the geometry as string
        job["valid_time"] = job.valid_time.dt.strftime("%Y-%m-%d")

        # Add other attributes we want to keep in the result
        job["start_date"] = start_date
        job["end_date"] = end_date
        job["lat"] = job.geometry.y
        job["lon"] = job.geometry.x

        variables = {
            "backend_name": backend.value,
            "out_prefix": "point-extraction",
            "out_extension": ".geoparquet",
            "start_date": start_date,
            "end_date": end_date,
            "s2_tile": s2_tile,
            "geometry": job.to_json(),
        }

        rows.append(pd.Series(variables))

    return pd.DataFrame(rows)


def create_job_point_worldcereal(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    job_options: Optional[Dict[str, Union[str, int]]] = None,
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

    # Try to get s2 tile ID to filter the collection
    if "s2_tile" in row:
        pipeline_log.debug(f"Extracting data for S2 tile {row.s2_tile}")
        s2_tile = row.s2_tile
    else:
        s2_tile = None

    inputs = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=geometry,
        temporal_extent=temporal_extent,
        fetch_type=FetchType.POINT,
        validate_temporal_context=False,
        s2_tile=s2_tile,
    )

    # Finally, create a vector cube based on the Point geometries
    cube = inputs.aggregate_spatial(geometries=geometry, reducer="mean")

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_points = get_job_nb_polygons(row)
    if pipeline_log is not None:
        pipeline_log.debug("Number of polygons to extract %s", number_points)

    # Set job options
    final_job_options = copy.deepcopy(DEFAULT_JOB_OPTIONS_POINT_WORLDCEREAL)
    if job_options:
        final_job_options.update(job_options)

    return cube.create_job(
        out_format="Parquet",
        title=f"Worldcereal_Point_Extraction_{row.s2_tile}",
        job_options=final_job_options,
    )


def post_job_action_point_worldcereal(
    job_items: List[pystac.Item], row: pd.Series, parameters: Optional[dict] = None
) -> list:
    for idx, item in enumerate(job_items):
        item_asset_path = Path(list(item.assets.values())[0].href)

        gdf = gpd.read_parquet(item_asset_path)

        # Convert the dates to datetime format
        gdf["timestamp"] = pd.to_datetime(gdf["date"])
        gdf.drop(columns=["date"], inplace=True)

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
            "S2-L2A-B8A",
            "S2-L2A-B11",
            "S2-L2A-B12",
            "S1-SIGMA0-VH",
            "S1-SIGMA0-VV",
            "elevation",
            "slope",
            "AGERA5-PRECIP",
            "AGERA5-TMEAN",
        ]
        gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

        # Remove samples where S1 and S2 are completely nodata
        cols = [c for c in gdf.columns if "S2" in c or "S1" in c]
        orig_sample_nr = len(gdf["sample_id"].unique())
        nodata_rows = (gdf[cols] == 65535).all(axis=1)
        all_nodata_per_sample = (
            gdf.assign(nodata=nodata_rows).groupby("sample_id")["nodata"].all()
        )
        valid_sample_ids = all_nodata_per_sample[~all_nodata_per_sample].index
        removed_samples = orig_sample_nr - len(valid_sample_ids)
        if removed_samples > 0:
            logger.warning(
                f"Removed {removed_samples} samples with all S1 and S2 bands as nodata."
            )
            gdf = gdf[gdf["sample_id"].isin(valid_sample_ids)]

        # Do some checks and perform corrections
        assert (
            len(gdf["ref_id"].unique()) == 1
        ), f"There are multiple ref_ids in the dataframe: {gdf['ref_id'].unique()}"
        ref_id = gdf["ref_id"][0]
        year = int(ref_id.split("_")[0])
        gdf["year"] = year

        # Make sure we remove the timezone information from the timestamp
        gdf["timestamp"] = gdf["timestamp"].dt.tz_localize(None)

        # Select required attributes and cast to dtypes
        required_attributes = copy.deepcopy(REQUIRED_ATTRIBUTES)
        required_attributes["ref_id"] = CategoricalDtype(
            categories=[ref_id], ordered=False
        )
        gdf = gdf[required_attributes.keys()]
        gdf = gdf.astype(required_attributes)

        gdf.to_parquet(item_asset_path, index=False)

    return job_items


def merge_output_files_point_worldcereal(
    output_folder: Union[str, Path],
    ref_id: str,
) -> None:
    """Merge the output geoparquet files of the point extractions. Partitioned per ref_id

    Parameters
    ----------
    output_folder : Union[str, Path]
        Location where extractions are saved
    ref_id : str
    collection id of the samples

    Raises
    ------
    FileNotFoundError
        If no geoparquet files are found in the output_folder
    """
    output_folder = Path(output_folder)
    merged_path = output_folder.parent / "worldcereal_merged_extractions.parquet"

    # Locate the files to merge and check whether there are any
    filecheck = list(output_folder.glob("**/*.geoparquet"))
    if len(filecheck) == 0:
        raise FileNotFoundError(f"No geoparquet files found in {output_folder}")
    else:
        pipeline_log.info(f"Merging {len(filecheck)} geoparquet files...")
    files_to_merge = str(output_folder / "**" / "*.geoparquet")

    # DuckDB requires the parent directory to exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check if this particular partition is already present in the merged path,
    # and if yes, delete it
    dir_name = merged_path / f"ref_id={ref_id}"
    if dir_name.exists():
        shutil.rmtree(str(dir_name))

    # Merge the files
    con = duckdb.connect()
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    con.execute(
        f"""
    COPY (
        SELECT * FROM read_parquet('{files_to_merge}', filename=false)
    ) TO '{str(merged_path)}' (FORMAT 'parquet', PARTITION_BY ref_id, OVERWRITE_OR_IGNORE, FILENAME_PATTERN '{ref_id}_{{i}}')
"""
    )

    con.close()
