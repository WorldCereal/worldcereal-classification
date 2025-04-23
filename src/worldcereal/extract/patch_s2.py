"""Extract S2 data using OpenEO-GFMAP package."""

import copy
from datetime import datetime
from typing import Dict, List, Optional, Union

import geojson
import geopandas as gpd
import openeo
import pandas as pd
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.manager import _log
from tqdm import tqdm

from worldcereal.openeo.preprocessing import raw_datacube_S2

from worldcereal.extract.utils import (  # isort: skip
    buffer_geometry,  # isort: skip
    get_job_nb_polygons,  # isort: skip
    upload_geoparquet_s3,  # isort: skip
)  # isort: skip


S2_L2A_CATALOGUE_BEGIN_DATE = datetime(2017, 1, 1)


DEFAULT_JOB_OPTIONS_PATCH_S2 = {
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "driver-cores": "1",
    "executor-memory": "1800m",
    "python-memory": "1900m",
    "executor-cores": "1",
    "max-executors": 22,
    "soft-errors": 0.1,
    "gdal-dataset-cache-size": 2,
    "gdal-cachemax": 120,
    "executor-threads-jvm": 1,
}


def create_job_dataframe_patch_s2(
    backend: Backend,
    split_jobs: List[gpd.GeoDataFrame],
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    rows = []
    for job in tqdm(split_jobs):
        # Compute the average in the valid date and make a buffer of 1.5 year around
        min_time = job.valid_time.min()
        max_time = job.valid_time.max()
        # 9 months before and after the valid time
        start_date = (min_time - pd.Timedelta(days=275)).to_pydatetime()
        end_date = (max_time + pd.Timedelta(days=275)).to_pydatetime()

        # Impose limits due to the data availability
        # start_date = max(start_date, S2_L2A_CATALOGUE_BEGIN_DATE)
        # end_date = min(end_date, datetime.now())

        s2_tile = job.tile.iloc[0]
        h3_l3_cell = job.h3_l3_cell.iloc[0]

        # Convert dates to string format
        start_date, end_date = (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        # Set back the valid_time in the geometry as string
        job["valid_time"] = job.valid_time.dt.strftime("%Y-%m-%d")

        variables = {
            "backend_name": backend.value,
            "out_prefix": "S2-L2A-10m",
            "out_extension": ".nc",
            "start_date": start_date,
            "end_date": end_date,
            "s2_tile": s2_tile,
            "h3_l3_cell": h3_l3_cell,
            "geometry": job.to_json(),
        }
        rows.append(pd.Series(variables))

    return pd.DataFrame(rows)


def create_job_patch_s2(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    job_options: Optional[Dict[str, Union[str, int]]] = None,
) -> gpd.GeoDataFrame:
    start_date = row.start_date
    end_date = row.end_date
    temporal_context = TemporalContext(start_date, end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)

    # # Filter the geometry to the rows with the extract only flag
    # geometry = filter_extract_true(geometry)
    # assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Performs a buffer of 64 px around the geometry
    geometry_df = buffer_geometry(geometry, distance_m=320)
    spatial_extent_url = upload_geoparquet_s3(
        provider, geometry_df, row.name, "SENTINEL2"
    )

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    # Get the h3index to use in the tile
    s2_tile = row.s2_tile
    valid_time = geometry.features[0].properties["valid_time"]

    bands_to_download = [
        "S2-L2A-B01",
        "S2-L2A-B02",
        "S2-L2A-B03",
        "S2-L2A-B04",
        "S2-L2A-B05",
        "S2-L2A-B06",
        "S2-L2A-B07",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B09",
        "S2-L2A-B11",
        "S2-L2A-B12",
        "S2-L2A-SCL",
    ]

    cube = raw_datacube_S2(
        connection,
        backend_context,
        temporal_context,
        bands_to_download,
        FetchType.POLYGON,
        spatial_extent=spatial_extent_url,
        filter_tile=s2_tile,
        apply_mask_flag=False,
        additional_masks_flag=True,
    )

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_polygons = get_job_nb_polygons(row)
    _log.debug("Number of polygons to extract %s", number_polygons)

    # Set job options
    final_job_options = copy.deepcopy(DEFAULT_JOB_OPTIONS_PATCH_S2)
    if job_options:
        final_job_options.update(job_options)

    return cube.create_job(
        out_format="NetCDF",
        title=f"Worldcereal_Patch-S2_Extraction_{s2_tile}_{valid_time}",
        sample_by_feature=True,
        job_options=final_job_options,
        feature_id_property="sample_id",
    )
