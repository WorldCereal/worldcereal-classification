"""Interaction with the WorldCereal RDM API. Used to generate the reference data in geoparquet format for the point extractions."""

from pathlib import Path
from typing import List, Union

import duckdb
import geopandas as gpd
import requests
from shapely import Polygon, wkb

# Define the default columns to be extracted from the RDM API
DEFAULT_COLUMNS = [
    "sample_id",
    "ewoc_code",
]

# RDM API Endpoint
RDM_ENDPOINT = "https://ewoc-rdm-api.iiasa.ac.at"


def _collections_from_rdm(poly: Polygon) -> List[str]:
    """Queries the RDM API and finds all intersection collection IDs for a given polygon.

    Parameters
    ----------
    poly : Polygon
        A user-defined polygon for which all intersection collection IDs need to be found.

    Returns
    -------
    List[str]
        A List containing the URLs of all intersection collection IDs.
    """

    bbox = poly.bounds
    bbox_str = f"Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}"
    url = f"{RDM_ENDPOINT}/collections/search?{bbox_str}"

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
        url = f"{RDM_ENDPOINT}/collections/{id}/sample/download"
        headers = {
            "accept": "*/*",
        }
        response = requests.get(url, headers=headers)
        urls.append(response.text)

    return urls


def _setup_sql_query(urls: List[str], poly: Polygon, columns) -> str:
    """Sets up the SQL query for the GeoParquet files.

    Parameters
    ----------
    urls : List[str]
        A list of URLs of the GeoParquet files.
    poly : Polygon
        A user-defined polygon.
    columns : _type_
        A list of column names to extract.

    Returns
    -------
    str
        A SQL query for the GeoParquet files.
    """

    combined_query = ""
    columns_str = ", ".join(columns)

    for i, url in enumerate(urls):
        query = f"""
            SELECT {columns_str}, ST_AsWKB(geometry) AS wkb_geometry
            FROM read_parquet('{url}')
            WHERE ST_Within(geometry, ST_GeomFromText('{str(poly)}'))

        """
        if i == 0:
            combined_query = query
        else:
            combined_query += f" UNION ALL {query}"

    return combined_query


def query_ground_truth(
    poly: Polygon, output_path: Union[str, Path], columns=DEFAULT_COLUMNS
):
    """Queries the RDM API and generates a GeoParquet file of all intersecting sample IDs.

    Parameters
    ----------
    poly : Polygon
        A user-defined polygon.
    output_path : _type_, optional
        The output path for the GeoParquet file.
    columns : _type_, optional
        A list of column names to extract., by default DEFAULT_COLUMNS
    """
    collection_ids = _collections_from_rdm(poly)
    urls = _get_download_urls(collection_ids)

    query = _setup_sql_query(urls, poly, columns)

    con = duckdb.connect()
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    df = con.execute(query).fetch_df()

    df["geometry"] = df["wkb_geometry"].apply(lambda x: wkb.loads(bytes(x)))
    df.drop(columns=["wkb_geometry"], inplace=True)

    # Convert the pandas DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf.geometry = (
        gdf.geometry.centroid
    )  # Ensure that the geometry is a point, can be replaced by a more advanced algorithm

    gdf.to_parquet(output_path, index=False)
