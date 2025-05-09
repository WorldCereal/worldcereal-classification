from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import openeo
import pandas as pd
import pystac_client
from openeo.processes import ProcessBuilder, eq, if_
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

STAC_ENDPOINT_SLOPE_TERRASCOPE = (
    "https://stac.openeo.vito.be/collections/COPERNICUS30_DEM_SLOPE_TERRASCOPE"
)

# Due to a bug on openEO side (https://github.com/Open-EO/openeo-geopyspark-driver/issues/1153)
# We have to provide here ALL bands in ALPHABETICAL order!
S2_BANDS = [
    "S2-L2A-B01",
    "S2-L2A-B02",
    "S2-L2A-B03",
    "S2-L2A-B04",
    "S2-L2A-B05",
    "S2-L2A-B06",
    "S2-L2A-B07",
    "S2-L2A-B08",
    "S2-L2A-B09",
    "S2-L2A-B11",
    "S2-L2A-B12",
    "S2-L2A-B8A",
    "S2-L2A-DISTANCE-TO-CLOUD",
    "S2-L2A-SCL",
    "S2-L2A-SCL_DILATED_MASK",
]

S2_BANDS_SELECTED = [
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
    "S2-L2A-SCL_DILATED_MASK",
]


def label_points_centroid(
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


def get_label_points(
    row: pd.Series, ground_truth_file: Optional[Union[Path, str]] = None
) -> gpd.GeoDataFrame:
    """
    Retrieve label points for a given row from STAC collections and RDM API.

    Parameters
    ----------
    row : pd.Series
        The row containing ref_id, epsg, start_date, and end_date.
    ground_truth_file : Optional[Union[Path, str]], optional
        The path to the ground truth file. If provided, this file will
        be queried for getting the ground truth. If not, the RDM will
        be used for the query.

    Returns
    -------
    gpd.GeoDataFrame
        The label points as a GeoDataFrame.

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
        ground_truth_file=ground_truth_file,
    )

    sampled_gdf = label_points_centroid(gdf=gdf, epsg=int(row["epsg"]))

    return sampled_gdf


def create_job_patch_to_point_worldcereal(
    row: pd.Series,
    connection: openeo.Connection,
    provider,
    connection_provider,
    executor_memory: str,
    python_memory: str,
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Assume row has the following fields: backend, start_date, end_date, epsg, ref_id and geometry_url

    s1_orbit_state = row.get(
        "orbit_state", "DESCENDING"
    )  # default to DESCENDING, same as for inference workflow

    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)

    # Get preprocessed cube from patch extractions
    cube = worldcereal_preprocessed_inputs_from_patches(
        connection,
        temporal_extent=temporal_extent,
        ref_id=row["ref_id"],
        epsg=int(row["epsg"]),
        s1_orbit_state=s1_orbit_state,
    )

    # Do spatial aggregation
    point_geometries = connection.load_url(url=row["geometry_url"], format="Parquet")
    cube = cube.aggregate_spatial(geometries=point_geometries, reducer="mean")

    job_options = {
        "executor-memory": executor_memory,
        "executor-memoryOverhead": python_memory,
    }

    return cube.create_job(
        title=f"WorldCereal patch-to-point extraction for ref_id: {row['ref_id']} and epsg: {row['epsg']}",
        out_format="Parquet",
        job_options=job_options,
    )


def worldcereal_preprocessed_inputs_from_patches(
    connection,
    temporal_extent,
    ref_id: str,
    epsg: int,
    s1_orbit_state: Optional[str] = None,
):
    # TODO: move preprocessing to separate functions 'preprocess_cube_x(cube: openeo.DataCube) -> openeo.DataCube' which will be the same across the different extraction workflows
    s1_stac_property_filter = {
        "ref_id": lambda x: eq(x, ref_id),
        "proj:epsg": lambda x: eq(x, epsg),
        "sat:orbit_state": lambda x: eq(x, s1_orbit_state),
    }

    s2_stac_property_filter = {
        "ref_id": lambda x: eq(x, ref_id),
        "proj:epsg": lambda x: eq(x, epsg),
    }

    s1_raw = connection.load_stac(
        url=STAC_ENDPOINT_S1,
        properties=s1_stac_property_filter,
        temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
        bands=["S1-SIGMA0-VH", "S1-SIGMA0-VV"],
    )
    s1 = decompress_backscatter_uint16(backend_context=None, cube=s1_raw)
    s1 = mean_compositing(s1, period="month")
    s1 = compress_backscatter_uint16(backend_context=None, cube=s1)

    s2_raw = connection.load_stac(
        url=STAC_ENDPOINT_S2,
        properties=s2_stac_property_filter,
        temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
        bands=S2_BANDS,
    ).filter_bands(S2_BANDS_SELECTED)

    def optimized_mask(input: ProcessBuilder):
        """
        To be used as a callback to apply_dimension on the band dimension.
        It's an optimized way of masking, if the mask is already present in the cube.
        """
        mask_band = input.array_element(label="S2-L2A-SCL_DILATED_MASK")
        return if_(mask_band != 1, input)

    s2 = s2_raw.apply_dimension(dimension="bands", process=optimized_mask)
    s2 = median_compositing(s2, period="month")
    s2 = s2.filter_bands(S2_BANDS_SELECTED[:-1])
    s2 = s2.linear_scale_range(0, 65534, 0, 65534)

    dem_raw = connection.load_collection("COPERNICUS_30", bands=["DEM"])
    dem = dem_raw.min_time()
    dem = dem.rename_labels(dimension="bands", target=["elevation"], source=["DEM"])

    slope = connection.load_stac(
        STAC_ENDPOINT_SLOPE_TERRASCOPE,
        bands=["Slope"],
    ).rename_labels(dimension="bands", target=["slope"])
    # Client fix for CDSE, the openeo client might be unsynchronized with
    # the backend.
    if "t" not in slope.metadata.dimension_names():
        slope.metadata = slope.metadata.add_dimension("t", "2020-01-01", "temporal")
    slope = slope.min_time()

    copernicus = slope.merge_cubes(dem)
    copernicus = copernicus.resample_cube_spatial(s2, method="bilinear")
    copernicus = copernicus.linear_scale_range(0, 65534, 0, 65534)

    meteo_raw = connection.load_stac(
        url=STAC_ENDPOINT_METEO_TERRASCOPE,
        temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
        bands=["temperature-mean", "precipitation-flux"],
    )

    meteo = meteo_raw.resample_spatial(
        resolution=10.0, projection=epsg, method="bilinear"
    )
    meteo = meteo.rename_labels(
        dimension="bands",
        target=["AGERA5-TMEAN", "AGERA5-PRECIP"],
    )

    cube = s2.merge_cubes(s1)
    cube = cube.merge_cubes(meteo)
    cube = cube.merge_cubes(copernicus)

    return cube
