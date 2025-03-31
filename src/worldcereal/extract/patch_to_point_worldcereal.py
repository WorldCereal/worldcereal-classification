from typing import Optional

import geopandas as gpd
import openeo
import pandas as pd
import pystac_client
from openeo.processes import eq
from openeo_gfmap import TemporalContext
from openeo_gfmap.preprocessing.compositing import mean_compositing, median_compositing
from openeo_gfmap.preprocessing.sar import (
    compress_backscatter_uint16,
    decompress_backscatter_uint16,
)
from shapely.geometry import MultiPolygon, shape

from worldcereal.rdm_api import RdmInteraction

STAC_ENDPOINT_S1 = (
    "https://stac.openeo.vito.be/collections/worldcereal_sentinel_1_patch_extractions"
)

STAC_ENDPOINT_S2 = (
    "https://stac.openeo.vito.be/collections/worldcereal_sentinel_2_patch_extractions"
)

STAC_ENDPOINT_METEO_TERRASCOPE = (
    "https://stac.openeo.vito.be/collections/agera5_monthly_terrascope"
)


def sample_points_centroid(
    gdf: gpd.GeoDataFrame, epsg: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Sample points from the centroid of the input GeoDataFrame.
    """
    if epsg is not None:
        gdf = gdf.to_crs(epsg=epsg)

    gdf["geometry"] = gdf.centroid

    if epsg is not None:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def get_sample_points_from_rdm(row: pd.Series) -> gpd.GeoDataFrame:
    """
    Get sample points from the random points of the input row.
    """
    # Find all items (i.e. patches) corresponding to the given ref_id and epsg
    stac_query = {
        "ref_id": {"eq": row["ref_id"]},
        "proj:epsg": {"eq": int(row["epsg"])},
    }
    client = pystac_client.Client.open("https://stac.openeo.vito.be/")

    search_s1 = client.search(
        collections=["worldcereal_sentinel_1_patch_extractions"], query=stac_query
    )
    search_s2 = client.search(
        collections=["worldcereal_sentinel_2_patch_extractions"], query=stac_query
    )

    items_s1 = {
        item.properties["sample_id"]: shape(item.geometry).buffer(1e-9)
        for item in search_s1.items()
    }
    items_s2 = {
        item.properties["sample_id"]: shape(item.geometry).buffer(1e-9)
        for item in search_s2.items()
    }

    # Find sample_ids which are present in both STAC collections
    common_sample_ids = set(items_s1.keys()).intersection(set(items_s2.keys()))

    # Items with the same sample_id will also have the same geometry
    polygons = [items_s1[sample_id] for sample_id in common_sample_ids]

    multi_polygon = MultiPolygon(polygons)

    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)

    # From the RDM API, we also want other 'collateral' geometries from different ref_ids.
    gdf = RdmInteraction().get_samples(
        spatial_extent=multi_polygon,
        temporal_extent=temporal_extent,
        include_private=True,
    )

    sampled_gdf = sample_points_centroid(gdf=gdf, epsg=int(row["epsg"]))

    return sampled_gdf


def create_job_patch_to_point_worldcereal(
    row: pd.Series,
    connection: openeo.Connection,
    provider,
    connection_provider,
    executor_memory: str,
    python_memory: str,
    max_executors: int,
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Assume row has the following fields: backend, start_date, end_date, epsg, ref_id and geometry_url

    # TODO: move preprocessing to separate functions 'preprocess_cube_x(cube: openeo.DataCube) -> openeo.DataCube' which will be the same across the different extraction workflows

    point_geometries = connection.load_url(url=row["geometry_url"], format="Parquet")

    stac_property_filter = {
        "ref_id": lambda x: eq(x, row["ref_id"]),
        "proj:epsg": lambda x: eq(x, int(row["epsg"])),
    }

    s1_raw = connection.load_stac(
        url=STAC_ENDPOINT_S1,
        properties=stac_property_filter,
        temporal_extent=[row["start_date"], row["end_date"]],
    )
    s1 = decompress_backscatter_uint16(backend_context=None, cube=s1_raw)
    s1 = mean_compositing(s1, period="month")
    s1 = compress_backscatter_uint16(backend_context=None, cube=s1)

    s2_raw = connection.load_stac(
        url=STAC_ENDPOINT_S2,
        properties=stac_property_filter,
        temporal_extent=[row["start_date"], row["end_date"]],
    )
    s2 = s2_raw.linear_scale_range(0, 65534, 0, 65534)
    s2 = median_compositing(s2, period="month")

    dem_raw = connection.load_collection("COPERNICUS_30", bands=["DEM"])
    dem = dem_raw.min_time()
    dem = dem.linear_scale_range(0, 65534, 0, 65534)
    dem = dem.rename_labels(dimension="bands", target=["elevation"], source=["DEM"])
    dem = dem.resample_cube_spatial(s2, method="bilinear")
    dem = dem.rename_labels(dimension="bands", target=["elevation"])
    # TODO: slope?

    meteo_raw = connection.load_stac(
        url=STAC_ENDPOINT_METEO_TERRASCOPE,
        temporal_extent=[row["start_date"], row["end_date"]],
        bands=["temperature-mean", "precipitation-flux"],
    )

    meteo = meteo_raw.resample_spatial(
        resolution=10.0, projection=int(row["epsg"]), method="bilinear"
    )
    meteo = meteo.rename_labels(
        dimension="bands",
        source=["temperature-mean", "precipitation-flux"],
        target=["AGERA5-TMEAN", "AGERA5-PRECIP"],
    )

    cube = s2.merge_cubes(s1)
    cube = cube.merge_cubes(dem)
    cube = cube.merge_cubes(meteo)

    cube = cube.aggregate_spatial(geometries=point_geometries, reducer="mean")

    return cube.create_job(
        title="Test patch_to_point_worldcereal",
        out_format="Parquet",
    )
