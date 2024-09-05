"""Extract AGERA5 (Meteo) data using OpenEO-GFMAP package."""

from typing import List

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
from openeo_gfmap import Backend, TemporalContext

from extract_common import (  # isort: skip
    buffer_geometry,  # isort: skip
    filter_extract_true,  # isort: skip
    upload_geoparquet_artifactory,  # isort: skip
)  # isort: skip

meteo_asset = pystac.extensions.item_assets.AssetDefinition({})


def create_job_dataframe_meteo(
    backend: Backend, split_jobs: List[gpd.GeoDataFrame]
) -> pd.DataFrame:
    raise NotImplementedError("This function is not implemented yet.")


def create_datacube_meteo(
    row: pd.Series,
    connection: openeo.DataCube,
    provider=None,
    connection_provider=None,
    executor_memory: str = "2G",
    python_memory: str = "1G",
    max_executors: int = 22,
) -> gpd.GeoDataFrame:
    start_date = row.start_date
    end_date = row.end_date
    temporal_context = TemporalContext(start_date, end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)

    # Filter the geometry to the rows with the extract only flag
    geometry = filter_extract_true(geometry)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Performs a buffer of 64 px around the geometry
    geometry_df = buffer_geometry(geometry, distance_m=5)
    spatial_extent_url = upload_geoparquet_artifactory(geometry_df, row.name)

    bands_to_download = ["temperature-mean", "precipitation-flux"]

    cube = connection.load_collection(
        "AGERA5",
        temporal_extent=[temporal_context.start_date, temporal_context.end_date],
        bands=bands_to_download,
    )
    filter_geometry = connection.load_url(spatial_extent_url, format="parquet")
    cube = cube.filter_spatial(filter_geometry)
    cube.rename_labels(
        dimension="bands",
        target=["AGERA5-temperature-mean", "AGERA5-precipitation-flux"],
        source=["temperature-mean", "precipitation-flux"],
    )

    # Rescale to uint16, multiplying by 100 first
    cube = cube * 100
    cube = cube.linear_scale_range(0, 65534, 0, 65534)

    h3index = geometry.features[0].properties["h3index"]
    valid_time = geometry.features[0].properties["valid_time"]

    job_options = {
        "executor-memory": executor_memory,
        "python-memory": python_memory,
        "max-executors": max_executors,
    }
    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_AGERA5_{h3index}_{valid_time}",
        sample_by_feature=True,
        job_options=job_options,
    )
