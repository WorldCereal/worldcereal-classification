"""Interaction with the WorldCereal RDM API. Used to generate the reference data in geoparquet format for the point extractions."""

from typing import Dict, List, Optional

import duckdb
import geopandas as gpd
import requests
from openeo.rest.auth.oidc import (
    OidcClientInfo,
    OidcDeviceAuthenticator,
    OidcProviderInfo,
)
from requests.adapters import HTTPAdapter
from shapely import wkb
from shapely.geometry.base import BaseGeometry
from urllib3.util.retry import Retry


class NoIntersectingCollections(Exception):
    """Raised when no spatiotemporally intersecting collection IDs are found in the RDM."""


class RdmInteraction:
    """Class to interact with the WorldCereal RDM API."""

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

    MAX_RETRIES = 5

    def __init__(self, resilient: bool = True):
        self.headers = None
        self.session = requests.Session()
        if resilient:
            self._make_resilient()

    def _make_resilient(self):
        """Make the session resilient to connection errors."""
        retries = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def authenticate(self):
        """Authenticate the user with the RDM API via device code flow."""
        self.headers = self._get_api_bearer_token()
        return self

    def _get_api_bearer_token(self) -> dict[str, str]:
        """Get API bearer access token via device code flow.

        Returns
        -------
        dict[str, str]
            A Dictionary containing the headers.
        """
        provider_info = OidcProviderInfo(
            issuer="https://sso.terrascope.be/auth/realms/terrascope"
        )

        client_info = OidcClientInfo(
            client_id="worldcereal-rdm",
            provider=provider_info,
        )

        authenticator = OidcDeviceAuthenticator(client_info=client_info)

        tokens = authenticator.get_tokens()

        return {"Authorization": f"Bearer {tokens.access_token}"}

    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for the API requests.
        Returns
        -------
        Dict[str, str]
            A dictionary containing the headers.
        """
        headers = {"accept": "*/*"}
        if self.headers:
            headers.update(self.headers)
        return headers

    def _collections_from_rdm(
        self,
        geometry: BaseGeometry,
        temporal_extent: Optional[List[str]] = None,
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

        url = f"{self.RDM_ENDPOINT}/collections/search?{bbox_str}{val_time}"

        response = self.session.get(url=url, headers=self._get_headers(), timeout=10)

        if response.status_code != 200:
            raise Exception(f"Error fetching collections: {response.text}")

        response_json = response.json()

        col_ids = [col["collectionId"] for col in response_json]

        if not col_ids:
            raise NoIntersectingCollections(
                f"No spatiotemporally intersecting collection IDs found in the RDM for the given geometry: {bbox} and temporal extent: {temporal_extent}."
            )

        return col_ids

    def _get_download_urls(
        self, collection_ids: List[str], user_id: Optional[str] = None
    ) -> List[str]:
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
            url = f"{self.RDM_ENDPOINT}/collections/{id}/download"
            response = self.session.get(url, headers=self._get_headers(), timeout=10)
            if response.status_code != 200:
                raise Exception(
                    f"Failed to get download URL for collection {id}: {response.text}"
                )
            urls.append(response.text)

        return urls

    def _setup_sql_query(
        self,
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
                SELECT {columns_str}, ST_AsWKB(ST_Intersection(ST_MakeValid(geometry), ST_GeomFromText('{str(geometry)}'))) AS wkb_geometry
                FROM read_parquet('{url}')
                WHERE ST_Intersects(ST_MakeValid(geometry), ST_GeomFromText('{str(geometry)}'))
                {optional_temporal}

            """
            if i == 0:
                combined_query = query
            else:
                combined_query += f" UNION ALL {query}"

        return combined_query

    def query_rdm(
        self,
        geometry: BaseGeometry,
        temporal_extent: Optional[List[str]] = None,
        columns: List[str] = DEFAULT_COLUMNS,
    ):
        """Queries the RDM API and generates a GeoParquet file of all intersecting sample IDs.

        Parameters
        ----------
        geometry : BaseGeometry
            A user-defined polygon. CRS should be EPSG:4326.
        temporal_extent : List[str], optional
            A list of two strings representing the temporal extent, by default None. If None, all available data will be queried.
            Dates should be in the format "YYYY-MM-DD".
        columns : List[str], optional
            A list of column names to extract., by default DEFAULT_COLUMNS

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the extracted columns and the geometry.
        """
        collection_ids = self._collections_from_rdm(
            geometry=geometry, temporal_extent=temporal_extent
        )
        urls = self._get_download_urls(collection_ids)

        query = self._setup_sql_query(
            urls=urls,
            geometry=geometry,
            columns=columns,
            temporal_extent=temporal_extent,
        )

        con = duckdb.connect()
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

        df = con.execute(query).fetch_df()

        df["geometry"] = df["wkb_geometry"].apply(lambda x: wkb.loads(bytes(x)))
        df.drop(columns=["wkb_geometry"], inplace=True)

        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        return gdf
