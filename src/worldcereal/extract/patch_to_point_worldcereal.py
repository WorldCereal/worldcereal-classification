import geopandas as gpd 
import openeo
import pandas as pd 
import pystac_client

from openeo_gfmap import TemporalContext
from shapely.geometry import shape, MultiPolygon
from worldcereal.rdm_api import RdmInteraction

STAC_ENDPOINT_S1 = 'https://stac.openeo.vito.be/collections/worldcereal_sentinel_1_patch_extractions'
STAC_ENDPOINT_S2 = 'https://stac.openeo.vito.be/collections/worldcereal_sentinel_2_patch_extractions'

def sample_points_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Sample points from the centroid of the input GeoDataFrame.
    """
    gdf["geometry"] = gdf.centroid
    return gdf

def get_sample_points_from_rdm(row: pd.Series) -> gpd.GeoDataFrame:
    """
    Get sample points from the random points of the input row.
    """
    query = {
    "ref_id": {"eq": row["ref_id"]},
    "proj:epsg": {"eq": int(row["epsg"])},
    "valid_time": {"gte": row["start_date"], "lte": row["end_date"]},  # Let's discuss temporal filtering together
}
    client = pystac_client.Client.open("https://stac.openeo.vito.be/")

    search = client.search(
    collections=["worldcereal_sentinel_1_patch_extractions"],  # We base ourselves on S1 patch extractions, could also use S2
    query=query,
)

    polygons = []

    for item in search.items():  
        polygons.append(shape(item.geometry).buffer(1e-9))  # Add buffer to avoid TopologyException

    multi_polygon = MultiPolygon(polygons)
    temporal_extent = TemporalContext([row.start_date, row.end_date])

    gdf = RdmInteraction().get_samples(
        ref_ids=[row['ref_id']],
        bbox=multi_polygon,  # Currently not possible to filter on MultiPolygon, so first fix that
        temporal_extent=temporal_extent,
        include_private=True
    )
    
    sampled_gdf = sample_points_centroid(gdf)
    
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

    