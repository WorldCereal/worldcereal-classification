"""Extract S2 data using OpenEO-GFMAP package."""

from datetime import datetime
from typing import List

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.manager import _log
from tqdm import tqdm

from worldcereal.openeo.preprocessing import raw_datacube_S2

from extract_common import (  # isort: skip
    buffer_geometry,  # isort: skip
    get_job_nb_polygons,  # isort: skip
    upload_geoparquet_artifactory,  # isort: skip
)  # isort: skip

# Define the sentinel 2 asset
sentinel2_asset = pystac.extensions.item_assets.AssetDefinition(
    {
        "gsd": 10,
        "title": "Sentinel2",
        "description": "Sentinel-2 bands",
        "type": "application/x-netcdf",
        "roles": ["data"],
        "proj:shape": [64, 64],
        "raster:bands": [
            {"name": "S2-L2A-B01"},
            {"name": "S2-L2A-B02"},
            {"name": "S2-L2A-B03"},
            {"name": "S2-L2A-B04"},
            {"name": "S2-L2A-B05"},
            {"name": "S2-L2A-B06"},
            {"name": "S2-L2A-B07"},
            {"name": "S2-L2A-B8A"},
            {"name": "S2-L2A-B08"},
            {"name": "S2-L2A-B11"},
            {"name": "S2-L2A-B12"},
            {"name": "S2-L2A-SCL"},
            {"name": "S2-L2A-SCL_DILATED_MASK"},
            {"name": "S2-L2A-DISTANCE_TO_CLOUD"},
        ],
        "cube:variables": {
            "S2-L2A-B01": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B02": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B03": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B04": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B05": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B06": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B07": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B8A": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B08": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B11": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-B12": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-SCL": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S2-L2A-SCL_DILATED_MASK": {
                "dimensions": ["time", "y", "x"],
                "type": "data",
            },
            "S2-L2A-DISTANCE_TO_CLOUD": {
                "dimensions": ["time", "y", "x"],
                "type": "data",
            },
        },
        "eo:bands": [
            {
                "name": "S2-L2A-B01",
                "common_name": "coastal",
                "center_wavelength": 0.443,
                "full_width_half_max": 0.027,
            },
            {
                "name": "S2-L2A-B02",
                "common_name": "blue",
                "center_wavelength": 0.49,
                "full_width_half_max": 0.098,
            },
            {
                "name": "S2-L2A-B03",
                "common_name": "green",
                "center_wavelength": 0.56,
                "full_width_half_max": 0.045,
            },
            {
                "name": "S2-L2A-B04",
                "common_name": "red",
                "center_wavelength": 0.665,
                "full_width_half_max": 0.038,
            },
            {
                "name": "S2-L2A-B05",
                "common_name": "rededge",
                "center_wavelength": 0.704,
                "full_width_half_max": 0.019,
            },
            {
                "name": "S2-L2A-B06",
                "common_name": "rededge",
                "center_wavelength": 0.74,
                "full_width_half_max": 0.018,
            },
            {
                "name": "S2-L2A-B07",
                "common_name": "rededge",
                "center_wavelength": 0.783,
                "full_width_half_max": 0.028,
            },
            {
                "name": "S2-L2A-B08",
                "common_name": "nir",
                "center_wavelength": 0.842,
                "full_width_half_max": 0.145,
            },
            {
                "name": "S2-L2A-B8A",
                "common_name": "nir08",
                "center_wavelength": 0.865,
                "full_width_half_max": 0.033,
            },
            {
                "name": "S2-L2A-B11",
                "common_name": "swir16",
                "center_wavelength": 1.61,
                "full_width_half_max": 0.143,
            },
            {
                "name": "S2-L2A-B12",
                "common_name": "swir16",
                "center_wavelength": 1.61,
                "full_width_half_max": 0.143,
            },
            {
                "name": "S2-L2A-SCL",
                "common_name": "swir16",
                "center_wavelength": 1.61,
                "full_width_half_max": 0.143,
            },
            {
                "name": "S2-L2A-SCL_DILATED_MASK",
            },
            {
                "name": "S2-L2A-DISTANCE_TO_CLOUD",
            },
        ],
    }
)

S2_L2A_CATALOGUE_BEGIN_DATE = datetime(2017, 1, 1)


def create_job_dataframe_s2(
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

        s2_tile = job.tile.iloc[0]  # Job dataframes are split depending on the
        h3_l3_cell = job.h3_l3_cell.iloc[0]

        # Convert dates to string format
        start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime(
            "%Y-%m-%d"
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


def create_datacube_optical(
    row: pd.Series,
    connection: openeo.DataCube,
    provider=None,
    connection_provider=None,
    executor_memory: str = "5G",
    executor_memory_overhead: str = "2G",
    max_executors: int = 22,
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
    spatial_extent_url = upload_geoparquet_artifactory(geometry_df, row.name)

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
        spatial_extent_url,
        temporal_context,
        bands_to_download,
        FetchType.POLYGON,
        filter_tile=s2_tile,
        apply_mask_flag=False,
        additional_masks_flag=True,
    )

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_polygons = get_job_nb_polygons(row)
    _log.debug("Number of polygons to extract %s", number_polygons)

    job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": executor_memory,
        "executor-memoryOverhead": executor_memory_overhead,
        "executor-cores": "1",
        "max-executors": max_executors,
        "soft-errors": "true",
        "gdal-dataset-cache-size": 2,
        "gdal-cachemax": 120,
        "executor-threads-jvm": 1,
    }

    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_S2_{s2_tile}_{valid_time}",
        sample_by_feature=True,
        job_options=job_options,
        feature_id_property="sample_id",
    )
