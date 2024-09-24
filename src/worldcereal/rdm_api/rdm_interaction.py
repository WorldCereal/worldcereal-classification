"""Interaction with the WorldCereal RDM API. Used to generate the reference data in geoparquet format for the point extractions."""

from pathlib import Path
from typing import List, Optional, Union

import duckdb
import geopandas as gpd
import requests
from shapely import wkb
from shapely.geometry.base import BaseGeometry

# Define the default columns to be extracted from the RDM API
DEFAULT_COLUMNS = [
    "sample_id",
    "ewoc_code",
    "valid_time",
    "quality_score_lc",
    "quality_score_ct",
]

# RDM API Endpoint
RDM_ENDPOINT = "https://ewoc-rdm-api.iiasa.ac.at"


def _collections_from_rdm(
    geometry: BaseGeometry, temporal_extent: Optional[List[str]] = None
) -> List[str]:
    """Queries the RDM API and finds all intersection collection IDs for a given geometry and temporal extent.

    Parameters
    ----------
    geometry : BaseGeometry
        A user-defined geometry for which all intersecting collection IDs need to be found.
    temporal_extent : Optional[List[str]], optional
        A list of two strings representing the temporal extent, by default None. If None, all available data will be queried.
    Returns
    -------
    List[str]
        A List containing the URLs of all intersection collection IDs.
    """

    bbox = geometry.bounds
    bbox_str = f"Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}"

    val_time = (
        f"&ValidityTime.Start={temporal_extent[0]}T00%3A00%3A00Z&ValidityTime.End={temporal_extent[1]}T00%3A00%3A00Z"
        if temporal_extent
        else ""
    )

    url = f"{RDM_ENDPOINT}/collections/search?{bbox_str}{val_time}"

    response = requests.get(url)
    response_json = response.json()

    col_ids = []
    for col in response_json:
        col_ids.append(col["collectionId"])

    return col_ids


def _get_download_urls(collection_ids: List[str]) -> List[str]:
    """Queries the RDM API and finds all HTTP URLs for the GeoParquet files for each collection ID.

    Parameters
    ----------
    collection_ids : List[str]
        A list of collection IDs.

    Returns
    -------
    List[str]
        A List containing the HTTPs URLs of the GeoParquet files for each collection ID.
    """
    urls = []
    for id in collection_ids:
        url = f"{RDM_ENDPOINT}/collections/{id}/download"
        headers = {
            "accept": "*/*",
        }
        response = requests.get(url, headers=headers)
        urls.append(response.text)

    return urls


def _setup_sql_query(
    urls: List[str],
    geometry: BaseGeometry,
    columns: List[str],
    temporal_extent: Optional[List[str]] = None,
) -> str:
    """Sets up the SQL query for the GeoParquet files.

    Parameters
    ----------
    urls : List[str]
        A list of URLs of the GeoParquet files.
    geometry : BaseGeometry
        A user-defined geometry.
    columns :
        A list of column names to extract.
    temporal_extent : Optional[List[str]], optional
        A list of two strings representing the temporal extent, by default None. If None, all available data will be queried.

    Returns
    -------
    str
        A SQL query for the GeoParquet files.
    """

    combined_query = ""
    columns_str = ", ".join(columns)

    optional_temporal = (
        f"AND valid_time BETWEEN '{temporal_extent[0]}' AND '{temporal_extent[1]}'"
        if temporal_extent
        else ""
    )

    for i, url in enumerate(urls):
        query = f"""
            SELECT {columns_str}, ST_AsWKB(ST_Intersection(geometry, ST_GeomFromText('{str(geometry)}'))) AS wkb_geometry
            FROM read_parquet('{url}')
            WHERE ST_Intersects(geometry, ST_GeomFromText('{str(geometry)}'))
            {optional_temporal}

        """
        if i == 0:
            combined_query = query
        else:
            combined_query += f" UNION ALL {query}"

    return combined_query


def query_ground_truth(
    geometry: BaseGeometry,
    output_path: Union[str, Path],
    temporal_extent: Optional[List[str]] = None,
    columns: List[str] = DEFAULT_COLUMNS,
):
    """Queries the RDM API and generates a GeoParquet file of all intersecting sample IDs.

    Parameters
    ----------
    geometry : BaseGeometry
        A user-defined polygon.
    output_path : Union[str, Path]
        The output path for the GeoParquet file.
    temporal_extent : List[str], optional
        A list of two strings representing the temporal extent, by default None. If None, all available data will be queried.
    columns : List[str], optional
        A list of column names to extract., by default DEFAULT_COLUMNS
    """
    collection_ids = _collections_from_rdm(
        geometry=geometry, temporal_extent=temporal_extent
    )
    urls = _get_download_urls(collection_ids)

    query = _setup_sql_query(
        urls=urls, geometry=geometry, columns=columns, temporal_extent=temporal_extent
    )

    con = duckdb.connect()
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    df = con.execute(query).fetch_df()

    df["geometry"] = df["wkb_geometry"].apply(lambda x: wkb.loads(bytes(x)))
    df.drop(columns=["wkb_geometry"], inplace=True)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf.geometry = (
        gdf.geometry.centroid
    )  # Ensure that the geometry is a point, can be replaced by a more advanced algorithm

    gdf.to_parquet(output_path, index=False)
