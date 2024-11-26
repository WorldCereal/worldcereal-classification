"""Interaction with the WorldCereal RDM API. Used to generate the reference data in geoparquet format for the point extractions."""

from typing import Dict, List, Optional

import duckdb
import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from openeo.rest.auth.oidc import (
    OidcClientInfo,
    OidcDeviceAuthenticator,
    OidcProviderInfo,
)
from requests.adapters import HTTPAdapter
from shapely import wkb
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from urllib3.util.retry import Retry

from .rdm_collection import RdmCollection


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
        "extract",
        "h3_l3_cell",
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

    def get_collections(
        self,
        geometry: Optional[BaseGeometry] = None,
        temporal_extent: Optional[List[str]] = None,
        include_public: Optional[bool] = True,
        include_private: Optional[bool] = False,
        ewoc_codes: Optional[List[int]] = None,
    ) -> List[RdmCollection]:
        """Queries the RDM API and finds all intersecting collections
        for a given geometry and temporal extent.

        Parameters
        ----------
        geometry : Optional[BaseGeometry], optional
            A user-defined geometry for which all intersecting collection IDs need to be found.
            CRS should be EPSG:4326.
            If None, all available data will be queried., by default None
        temporal_extent : Optional[List[str]], optional
            A list of two strings representing the temporal extent, by default None. If None, all available data will be queried.
        include_public: Optional[bool] = True
            Whether or not to include public collections.
        include_private: Optional[bool] = False
            Whether or not to include private collections.
            If True, the user must be authenticated.
        ewoc_codes: Optional[List[int]] = None
            A list of EWOC codes to filter the collections by.

        Returns
        -------
        List[RdmCollection]
            A List containing the collections matching the criteria.
        """

        # If user requests private collections, they must be authenticated
        if include_private:
            if not self.headers or "Authorization" not in self.headers:
                logger.info("To access private collections, you need to authenticate.")
                self.authenticate()

        # Handle geometry
        bbox = geometry.bounds if geometry is not None else [-180, -90, 180, 90]
        # check if the geometry is valid
        if bbox[0] < -180 or bbox[1] < -90 or bbox[2] > 180 or bbox[3] > 90:
            raise ValueError("Invalid geometry. CRS should be EPSG:4326.")
        bbox_str = f"Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}"

        # Handle temporal extent
        val_time = (
            f"&ValidityTime.Start={temporal_extent[0]}T00%3A00%3A00Z&ValidityTime.End={temporal_extent[1]}T00%3A00%3A00Z"
            if temporal_extent is not None
            else ""
        )

        # Handle EWOC codes
        ewoc_codes_str = (
            "".join([f"&EwocCodes={str(ewoc_code)}" for ewoc_code in ewoc_codes])
            if ewoc_codes is not None
            else ""
        )

        # Construct the URL
        url = f"{self.RDM_ENDPOINT}/collections/search?{bbox_str}{val_time}{ewoc_codes_str}"

        # Process the request
        response = self.session.get(url=url, headers=self._get_headers(), timeout=10)

        if response.status_code != 200:
            raise Exception(f"Error fetching collections: {response.text}")

        collections = [RdmCollection(**col) for col in response.json()]

        # Filter out public and private collections if needed
        if not include_public:
            collections = [col for col in collections if col.access_type != "Public"]

        if not include_private:
            collections = [col for col in collections if col.access_type == "Public"]

        if not collections:
            logger.info("No collections found in the RDM for your search criteria.")

        return collections

    def _get_download_urls(
        self,
        collection_ids: List[str],
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

    def get_crop_counts(
        self,
        collection_ids: List[str],
        ewoc_codes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Counts the number of items matching a specific crop type or types
            for one or multiple collections.

        Parameters
        ----------
        collection_ids : List[str]
            List of collection IDs to check.
        ewoc_codes: Optional[List[int]] = None
            A list of EWOC codes for which crop counts need to be fetched.
            If None, all available crop types will be counted.

        Returns
        -------
        Pandas DataFrame
            Dataframe containing for each collection id the number of items matching the requested crop types.
            If stats are not available for a collection, a warning will be issued.
        """

        # Prepare result
        result = []

        for col_id in collection_ids:
            itemUrl = f"{self.RDM_ENDPOINT}/collections/{col_id}/items/codestats"
            itemsResponse = requests.get(itemUrl)
            res = itemsResponse.json()
            if "ewocStats" in res:
                stats = res["ewocStats"]
                if ewoc_codes is None:
                    # get counts for all crops
                    for stat in stats:
                        result.append(
                            {
                                "collectionId": col_id,
                                "EwocCode": stat["code"],
                                "count": stat["count"],
                            }
                        )
                else:
                    # get counts for specific crops
                    for cropcode in ewoc_codes:
                        count = 0
                        for stat in stats:
                            if stat["code"] == cropcode:
                                count = stat["count"]
                                break
                        result.append(
                            {
                                "collectionId": col_id,
                                "EwocCode": cropcode,
                                "count": count,
                            }
                        )
            else:
                # crop counts not available
                logger.warning(f"No crop counts available for collection {col_id}")

        result_df = pd.DataFrame(result)

        if len(result) > 0:
            # Pivot the DataFrame to have collections in rows and crop types in columns
            result_df = result_df.pivot(
                index="collectionId", columns="EwocCode", values="count"
            ).fillna(0)

        return result_df

    def _setup_sql_query(
        self,
        urls: List[str],
        geometry: BaseGeometry,
        columns: List[str],
        temporal_extent: Optional[List[str]] = None,
        ewoc_codes: Optional[List[int]] = None,
        subset: Optional[bool] = False,
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
        ewoc_codes: Optional[List[int]] = None
            A list of EWOC codes to filter the samples by.
        subset : Optional[bool], optional
            If True, only download a subset of the samples (for which extract attribute ==1)
            If False, extract all samples.
            Default is False.

        Returns
        -------
        str
            A SQL query for the GeoParquet files.
        """

        combined_query = ""
        columns_str = ", ".join(columns)

        optional_temporal = (
            f"AND valid_time BETWEEN '{temporal_extent[0]}' AND '{temporal_extent[1]}'"
            if temporal_extent is not None
            else ""
        )

        optional_ewoc_codes = (
            f"AND ewoc_code IN ({', '.join([str(code) for code in ewoc_codes])})"
            if ewoc_codes is not None
            else ""
        )

        optional_subset = "AND extract > 0" if subset else ""

        for i, url in enumerate(urls):
            collection_id = str(url).split("/")[-2]
            query = f"""
                SELECT {columns_str}, ST_AsWKB(ST_Intersection(ST_MakeValid(geometry), ST_GeomFromText('{str(geometry)}'))) AS wkb_geometry, '{collection_id}' AS collection_id
                FROM read_parquet('{url}')
                WHERE ST_Intersects(ST_MakeValid(geometry), ST_GeomFromText('{str(geometry)}'))
                {optional_temporal}
                {optional_ewoc_codes}
                {optional_subset}

            """
            if i == 0:
                combined_query = query
            else:
                combined_query += f" UNION ALL {query}"

        return combined_query

    def download_samples(
        self,
        collection_ids: Optional[List[str]] = None,
        columns: List[str] = DEFAULT_COLUMNS,
        subset: Optional[bool] = False,
        geometry: Optional[BaseGeometry] = None,
        temporal_extent: Optional[List[str]] = None,
        ewoc_codes: Optional[List[int]] = None,
        include_public: Optional[bool] = True,
        include_private: Optional[bool] = False,
    ) -> gpd.GeoDataFrame:
        """Queries the RDM API and generates a GeoPandas GeoDataframe of all samples meeting the search criteria.

        Parameters
        ----------
        collection_ids : Optional(List[str]), optional
            List of collection IDs to download samples from.
            If not specified, all collections matching the search criteria defined by
            the other input parameters will be queried.
        columns : List[str], optional
            A list of column names to extract., by default DEFAULT_COLUMNS
        subset : Optional[bool], optional
            If True, only download a subset of the samples (for which extract attribute ==1)
            If False, extract all samples.
            Default is False.
        geometry : Optional[BaseGeometry], optional
            A user-defined geometry for which all intersecting collections need to be found.
            CRS should be EPSG:4326.
            If None, all available data will be queried., by default None
        temporal_extent : List[str], optional
            A list of two strings representing the temporal extent, by default None.
            If None, all available data will be queried.
            Dates should be in the format "YYYY-MM-DD".
        ewoc_codes: Optional[List[int]] = None
            If specified, only samples with the specified EWOC codes will be extracted.
        include_public: Optional[bool] = True
            Whether or not to include public collections.
        include_private: Optional[bool] = False
            Whether or not to include private collections.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the extracted samples.
            For each sample, the collection ID is automatically included.
        """

        # Determine which collections need to be queried if they are not specified
        if not collection_ids:
            collections = self.get_collections(
                geometry=geometry,
                temporal_extent=temporal_extent,
                ewoc_codes=ewoc_codes,
                include_public=include_public,
                include_private=include_private,
            )
            if not collections:
                logger.warning(
                    "No collections found in the RDM for your search criteria."
                )
                return gpd.GeoDataFrame()
            collection_ids = [col.id for col in collections]

        logger.info(f"Querying {len(collection_ids)} collections...")

        # For each collection, get the download URL
        urls = self._get_download_urls(collection_ids)

        # Ensure we have a valid geometry
        if not geometry:
            bbox = [-180, -90, 180, 90]
            # convert bbox to shapely geometry
            geometry = Polygon(
                [
                    (bbox[0], bbox[1]),  # Bottom-left corner
                    (bbox[0], bbox[3]),  # Top-left corner
                    (bbox[2], bbox[3]),  # Top-right corner
                    (bbox[2], bbox[1]),  # Bottom-right corner
                    (bbox[0], bbox[1]),
                ]
            )

        # Set up the SQL query
        query = self._setup_sql_query(
            urls=urls,
            geometry=geometry,
            columns=columns,
            temporal_extent=temporal_extent,
            ewoc_codes=ewoc_codes,
            subset=subset,
        )

        # Execute the query
        con = duckdb.connect()
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

        df = con.execute(query).fetch_df()

        # Convert the WKB geometry to a Shapely geometry
        df["geometry"] = df["wkb_geometry"].apply(lambda x: wkb.loads(bytes(x)))
        df.drop(columns=["wkb_geometry"], inplace=True)

        # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        return gdf
