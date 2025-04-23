"""Extract AGERA5 (Meteo) data using OpenEO-GFMAP package."""

import copy
from typing import Dict, List, Optional, Union

import geojson
import geopandas as gpd
import openeo
import pandas as pd
from openeo_gfmap import Backend, TemporalContext

from worldcereal.extract.utils import (  # isort: skip
    buffer_geometry,  # isort: skip
    filter_extract_true,  # isort: skip
    upload_geoparquet_s3,  # isort: skip
)  # isort: skip


DEFAULT_JOB_OPTIONS_PATCH_METEO = {
    "executor-memory": "1800m",
    "python-memory": "1000m",
    "max-executors": 22,
}


def create_job_dataframe_patch_meteo(
    backend: Backend, split_jobs: List[gpd.GeoDataFrame]
) -> pd.DataFrame:
    raise NotImplementedError("This function is not implemented yet.")


def create_job_patch_meteo(
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

    # Filter the geometry to the rows with the extract only flag
    geometry = filter_extract_true(geometry)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Performs a buffer of 64 px around the geometry
    geometry_df = buffer_geometry(geometry, distance_m=5)
    spatial_extent_url = upload_geoparquet_s3(provider, geometry_df, row.name, "METEO")

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

    # Set job options
    final_job_options = copy.deepcopy(DEFAULT_JOB_OPTIONS_PATCH_METEO)
    if job_options:
        final_job_options.update(job_options)

    return cube.create_job(
        out_format="NetCDF",
        title=f"WorldCereal_Patch-AGERA5_Extraction_{h3index}_{valid_time}",
        sample_by_feature=True,
        job_options=final_job_options,
    )
