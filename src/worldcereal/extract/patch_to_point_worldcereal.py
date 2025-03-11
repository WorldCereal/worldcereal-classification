from getpass import getpass
from typing import Optional

import geopandas as gpd
import openeo
import pandas as pd
import pystac_client
from openeo.processes import eq
from openeo_gfmap.preprocessing.compositing import median_compositing
from shapely.geometry import MultiPolygon, shape

from worldcereal.extract.utils import upload_geoparquet_artifactory
from worldcereal.rdm_api import RdmInteraction

STAC_ENDPOINT_S1 = (
    "https://stac.openeo.vito.be/collections/worldcereal_sentinel_1_patch_extractions"
)
STAC_ENDPOINT_S2 = (
    "https://stac.openeo.vito.be/collections/worldcereal_sentinel_2_patch_extractions"
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
    query = {
        "ref_id": {"eq": row["ref_id"]},
        "proj:epsg": {"eq": int(row["epsg"])},
        "valid_time": {
            "gte": row["start_date"],
            "lte": row["end_date"],
        },  # Let's discuss temporal filtering together
    }
    client = pystac_client.Client.open("https://stac.openeo.vito.be/")

    search = client.search(
        collections=[
            "worldcereal_sentinel_1_patch_extractions"
        ],  # We base ourselves on S1 patch extractions, could also use S2
        query=query,
    )

    polygons = []

    for item in search.items():
        polygons.append(
            shape(item.geometry).buffer(1e-9)
        )  # Add buffer to avoid TopologyException

    multi_polygon = MultiPolygon(polygons)
    # temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)

    gdf = RdmInteraction().get_samples(
        ref_ids=[
            row["ref_id"].lower()
        ],  # Mismatch between ref_id and RDM API ID --> is this always just lower case?
        spatial_extent=multi_polygon,
        # temporal_extent=temporal_extent,  # TODO: discuss temporal filtering
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
    pass

    # Assume row has the following fields: backend, start_date, end_date, spatial_extent/geometry(?), epsg, ref_id

    # TODO: in a separate issue and PR, we could consider refactoring worldcereal/openeo/preprocessing to separate the S1 and S2 preprocessing from the raw datacube extractions
    # For now I suggest to just get it working

    gdf = get_sample_points_from_rdm(row)
    url = upload_geoparquet_artifactory(
        gdf,
        name=str(row["epsg"]),
        collection=row["ref_id"],
        username="vincent.verelst",
        password=getpass("Enter your password: "),
    )
    point_geometries = connection.load_url(url=url, format="Parquet")

    stac_property_filter = {
        "ref_id": lambda x: eq(x, row["ref_id"]),
        "proj:epsg": lambda x: eq(x, int(row["epsg"])),
        # "valid_time": {"gte": row["start_date"], "lte": row["end_date"]},  # Let's discuss temporal filtering together
    }
    s1_raw = connection.load_stac(
        url="https://stac.openeo.vito.be/collections/worldcereal_sentinel_1_patch_extractions",
        properties=stac_property_filter,
    )
    s1 = median_compositing(s1_raw, period="month")

    s2_raw = connection.load_stac(
        url="https://stac.openeo.vito.be/collections/worldcereal_sentinel_2_patch_extractions",
        properties=stac_property_filter,
    )
    s2 = median_compositing(s2_raw, period="month")

    dem_raw = connection.load_collection("COPERNICUS_30", bands=["DEM"])
    dem = dem_raw.mint_time()
    dem = dem.rename_labels(dimension="bands", target=["elevation"], source=["DEM"])
    dem = dem.resample_cube_spatial(s2, method="bilinear")
    # TODO: slope?

    cube = s1.merge_cubes(s2)
    cube = cube.merge_cubes(dem)

    cube = cube.aggregate_spatial(geometries=point_geometries, reducer="mean")

    return cube.create_job(
        title="Test patch_to_point_worldcereal",
        out_format="Parquet",
    )
