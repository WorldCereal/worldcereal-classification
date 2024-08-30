"""Extract S1 data using OpenEO-GFMAP package."""

from datetime import datetime
from typing import List

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
from openeo_gfmap import (
    Backend,
    BackendContext,
    BoundingBoxExtent,
    FetchType,
    TemporalContext,
)
from openeo_gfmap.preprocessing.sar import compress_backscatter_uint16
from openeo_gfmap.utils.catalogue import s1_area_per_orbitstate_vvvh
from tqdm import tqdm

from worldcereal.openeo.preprocessing import raw_datacube_S1

from extract_common import (  # isort: skip
    buffer_geometry,  # isort: skip
    get_job_nb_polygons,  # isort: skip
    pipeline_log,  # isort: skip
    upload_geoparquet_artifactory,  # isort: skip
)

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
            "S1-SIGMA0-VV": {"dimensions": ["time", "y", "x"], "type": "data"},
            "S1-SIGMA0-VH": {"dimensions": ["time", "y", "x"], "type": "data"},
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

        # Convert dates to string format
        start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime(
            "%Y-%m-%d"
        )

        s2_tile = job.tile.iloc[0]  # Job dataframes are split depending on the
        h3_l3_cell = job.h3_l3_cell.iloc[0]

        # Check wherever the s2_tile is in the grid
        geometry_bbox = job.to_crs(epsg=4326).total_bounds
        # Buffer if the geometry is a point
        if geometry_bbox[0] == geometry_bbox[2]:
            geometry_bbox = (
                geometry_bbox[0] - 0.0001,
                geometry_bbox[1],
                geometry_bbox[2] + 0.0001,
                geometry_bbox[3],
            )
        if geometry_bbox[1] == geometry_bbox[3]:
            geometry_bbox = (
                geometry_bbox[0],
                geometry_bbox[1] - 0.0001,
                geometry_bbox[2],
                geometry_bbox[3] + 0.0001,
            )

        area_per_orbit = s1_area_per_orbitstate_vvvh(
            backend=BackendContext(backend),
            spatial_extent=BoundingBoxExtent(*geometry_bbox),
            temporal_extent=TemporalContext(start_date, end_date),
        )
        descending_area = area_per_orbit["DESCENDING"]["area"]
        ascending_area = area_per_orbit["ASCENDING"]["area"]

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
    python_memory: str = "2G",
    max_executors: int = 22,
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
        "python-memory": python_memory,
        "soft-errors": "true",
        "max_executors": max_executors,
    }
    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_S1_{s2_tile}_{valid_time}_{orbit_state}",
        sample_by_feature=True,
        job_options=job_options,
        feature_id_property="sample_id",
    )
