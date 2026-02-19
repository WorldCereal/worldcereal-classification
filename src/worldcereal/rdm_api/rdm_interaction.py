"""Interaction with the WorldCereal RDM API. Used to generate the reference data in geoparquet format for the point extractions."""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import duckdb
import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from openeo.rest.auth.oidc import (
    OidcClientInfo,
    OidcDeviceAuthenticator,
    OidcDeviceCodePollTimeout,
    OidcException,
    OidcProviderInfo,
    VerificationInfo,
    clip,
    create_timer,
)
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from requests.adapters import HTTPAdapter
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from urllib3.util.retry import Retry

from worldcereal.rdm_api.rdm_collection import RdmCollection
from worldcereal.utils.legend import ewoc_code_to_label

# Define the default columns to be extracted from the RDM API
RDM_DEFAULT_COLUMNS = [
    "sample_id",
    "ewoc_code",
    "valid_time",
    "irrigation_status",
    "quality_score_lc",
    "quality_score_ct",
    "extract",
    "h3_l3_cell",
    "ref_id",
    "geometry",
]


class NoIntersectingCollections(Exception):
    """Raised when no spatiotemporally intersecting collection IDs are found in the RDM."""


class RdmInteraction:
    """Class to interact with the WorldCereal RDM API."""

    # RDM API Endpoint
    RDM_ENDPOINT = "https://ewoc-rdm-api.iiasa.ac.at"

    MAX_RETRIES = 5

    def __init__(self, resilient: bool = True):
        self.headers: Optional[dict[str, str]] = None
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

    def authenticate(
        self,
        display_callback: Optional[Callable[[VerificationInfo], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        """Authenticate the user with the RDM API via device code flow."""
        self.headers = self._get_api_bearer_token(
            display_callback=display_callback, progress_callback=progress_callback
        )
        return self

    def _get_api_bearer_token(
        self,
        display_callback: Optional[Callable[[VerificationInfo], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, str]:
        """Get API bearer access token via device code flow.

        Returns
        -------
        dict[str, str]
            A Dictionary containing the headers.
        """
        # Set up the OIDC provider and client information
        provider_info = OidcProviderInfo(
            issuer="https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
        )

        client_info = OidcClientInfo(
            client_id="openeo-worldcereal-rdm",
            provider=provider_info,
        )

        authenticator = OidcDeviceAuthenticator(client_info=client_info)

        if display_callback is None:
            tokens = authenticator.get_tokens()
            return {"Authorization": f"Bearer {tokens.access_token}"}

        tokens = self._get_tokens_with_device_flow(
            authenticator=authenticator,
            display_callback=display_callback,
            progress_callback=progress_callback,
        )

        return {"Authorization": f"Bearer {tokens.access_token}"}

    def _get_tokens_with_device_flow(
        self,
        authenticator: OidcDeviceAuthenticator,
        display_callback: Callable[[VerificationInfo], None],
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        """Run device flow with custom UI callbacks to keep output inside widgets."""
        verification_info = authenticator._get_verification_info()
        display_callback(verification_info)

        token_endpoint = authenticator._provider_config["token_endpoint"]
        post_data = {
            "client_id": authenticator.client_id,
            "device_code": verification_info.device_code,
            "grant_type": authenticator.grant_type,
        }
        if authenticator._pkce:
            post_data["code_verifier"] = authenticator._pkce.code_verifier
        else:
            post_data["client_secret"] = authenticator.client_secret

        poll_interval = verification_info.interval
        elapsed = create_timer()
        next_poll = elapsed() + poll_interval
        sleep = clip(authenticator._max_poll_time / 100, min=1, max=5)

        while elapsed() <= authenticator._max_poll_time:
            time.sleep(sleep)

            if elapsed() >= next_poll:
                if progress_callback:
                    progress_callback("Polling authorization status...")
                try:
                    resp = authenticator._requests.post(
                        url=token_endpoint, data=post_data, timeout=5
                    )
                except requests.exceptions.RequestException as exc:
                    raise OidcException(
                        f"Failed to retrieve access token at {token_endpoint!r}: {exc!r}"
                    ) from exc

                if resp.status_code == 200:
                    if progress_callback:
                        progress_callback("Authorized successfully.")
                    return authenticator._get_access_token_result(data=resp.json())

                try:
                    error = resp.json()["error"]
                except Exception:
                    error = "unknown"

                if error == "authorization_pending":
                    if progress_callback:
                        progress_callback("Authorization pending...")
                elif error == "slow_down":
                    if progress_callback:
                        progress_callback("Slowing down...")
                    poll_interval += 5
                else:
                    raise OidcException(
                        f"Failed to retrieve access token at {token_endpoint!r}: {resp.status_code} {resp.reason!r} {resp.text!r}"
                    )

                next_poll = elapsed() + poll_interval

        if progress_callback:
            progress_callback("Timed out while waiting for authorization.")
        raise OidcDeviceCodePollTimeout(
            f"Timeout ({authenticator._max_poll_time:.1f}s) while polling for access token."
        )

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
        spatial_extent: Optional[
            Union[BoundingBoxExtent, Polygon, MultiPolygon]
        ] = None,
        temporal_extent: Optional[TemporalContext] = None,
        include_public: Optional[bool] = True,
        include_private: Optional[bool] = False,
        ewoc_codes: Optional[List[int]] = None,
    ) -> List[RdmCollection]:
        """Queries the RDM API and finds all intersecting collections
        for a given geometry and temporal extent.

        Parameters
        ----------
        spatial_extent : Optional[Union[BoundingBoxExtent, Polygon, MultiPolygon]], optional
            A user-defined bounding box, or shapely Polygon or MultiPolygon for which all intersecting collections need to be found.
            CRS should be EPSG:4326.
            If None, all available data will be queried, by default None
        temporal_extent : Optional[TemporalContext], optional
            Temporal extent, defined by start and end date, by default None.
            If None, all available data will be queried.
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
        if isinstance(spatial_extent, Polygon) or isinstance(
            spatial_extent, MultiPolygon
        ):
            spatial_extent = spatial_extent.bounds
            spatial_extent = BoundingBoxExtent(
                west=spatial_extent[0],
                south=spatial_extent[1],
                east=spatial_extent[2],
                north=spatial_extent[3],
                epsg=4326,
            )

        if spatial_extent is None:
            spatial_extent = BoundingBoxExtent(
                west=-180, south=-90, east=180, north=90, epsg=4326
            )
        # check if the geometry is valid
        self.assert_valid_spatial_extent(spatial_extent)

        bbox_str = f"Bbox={spatial_extent.west}&Bbox={spatial_extent.south}&Bbox={spatial_extent.east}&Bbox={spatial_extent.north}"

        # Handle temporal extent
        val_time = (
            f"&ValidityTime.Start={temporal_extent.start_date}T00%3A00%3A00Z&ValidityTime.End={temporal_extent.end_date}T00%3A00%3A00Z"
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
        ref_ids: List[str],
        subset: bool = False,
    ) -> List[str]:
        """Queries the RDM API and finds all HTTP URLs for the GeoParquet files for each ref ID.

        Parameters
        ----------
        ref_ids : List[str]
            A list of collection IDs.

        Returns
        -------
        List[str]
            A List containing the HTTPs URLs of the GeoParquet files for each ref ID.
        """
        urls = []

        additional_part = "/sample" if subset else ""

        for id in ref_ids:
            url = f"{self.RDM_ENDPOINT}/collections/{id}{additional_part}/download"
            response = self.session.get(url, headers=self._get_headers(), timeout=30)
            if response.status_code != 200:
                raise Exception(
                    f"Failed to get download URL for collection {id}: {response.text}"
                )
            urls.append(response.text)

        return urls

    def get_crop_counts(
        self,
        ref_ids: List[str],
        ewoc_codes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Counts the number of items matching a specific crop type or types
            for one or multiple collections.

        Parameters
        ----------
        ref_ids : List[str]
            List of collections to check (identified by their collection IDs).
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

        for ref_id in ref_ids:
            itemUrl = f"{self.RDM_ENDPOINT}/collections/{ref_id}/items/codestats"
            itemsResponse = requests.get(itemUrl)
            res = itemsResponse.json()
            if "ewocStats" in res:
                stats = res["ewocStats"]
                if ewoc_codes is None:
                    # get counts for all crops
                    for stat in stats:
                        result.append(
                            {
                                "ref_id": ref_id,
                                "ewoc_code": stat["code"],
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
                                "ref_id": ref_id,
                                "ewoc_code": cropcode,
                                "count": count,
                            }
                        )
            else:
                # crop counts not available
                logger.warning(f"No crop counts available for collection {ref_id}")

        result_df = pd.DataFrame(result)

        if len(result) > 0:
            # Correctly format the dataframe
            result_df = result_df.pivot(
                index="ref_id", columns="ewoc_code", values="count"
            ).fillna(0)
            # Ensure all columns are integers
            result_df = result_df.astype(int)
            # add crop labels from legend and pivot again
            result_df = result_df.T
            result_df.reset_index(inplace=True)
            result_df["Label"] = ewoc_code_to_label(result_df["ewoc_code"].values)
            result_df.sort_values(by="ewoc_code", inplace=True)
            result_df.set_index(["ewoc_code", "Label"], inplace=True)

        else:
            result_df = pd.DataFrame(columns=ref_ids)

        return result_df

    def _setup_sql_query(
        self,
        urls: List[str],
        spatial_extent: Union[BoundingBoxExtent, Polygon, MultiPolygon],
        columns: List[str],
        temporal_extent: Optional[TemporalContext] = None,
        ewoc_codes: Optional[List[int]] = None,
        subset: Optional[bool] = False,
        min_quality_lc: int = 0,
        min_quality_ct: int = 0,
    ) -> str:
        """Sets up the SQL query for the GeoParquet files.

        Parameters
        ----------
        urls : List[str]
            A list of URLs of the GeoParquet files.
        spatial_extent : Union[BoundingBoxExtent, Polygon, MultiPolygon]
            A user-defined bounding box, or shapely Polygon or MultiPolygon.
        columns :
            A list of column names to extract.
        temporal_extent : Optional[TemporalContext], optional
            Temporal extent, defined by start and end date, by default None.
            If None, all available data will be queried.
        ewoc_codes: Optional[List[int]] = None
            A list of EWOC codes to filter the samples by.
        subset : Optional[bool], optional
            If True, only download a subset of the samples (for which extract attribute ==1)
            If False, extract all samples.
            Default is False.
        min_quality_lc: int = 0
            Minimum quality score for land cover [0-100].
        min_quality_ct: int = 0
            Minimum quality score for crop type [0-100].

        Returns
        -------
        str
            A SQL query for the GeoParquet files.
        """

        # initialize query
        combined_query = ""

        # compile list of columns to request
        # ref_id is not part of the parquet files, so should be ignored here
        columns_str = ", ".join([c for c in columns if c != "ref_id"])

        optional_temporal = (
            f"AND valid_time BETWEEN '{temporal_extent.start_date}' AND '{temporal_extent.end_date}'"
            if temporal_extent is not None
            else ""
        )

        optional_ewoc_codes = (
            f"AND ewoc_code IN ({', '.join([str(code) for code in ewoc_codes])})"
            if ewoc_codes is not None
            else ""
        )

        optional_subset = "AND extract > 0" if subset else ""

        optional_quality_lc = (
            f"AND quality_score_lc >= {min_quality_lc}" if min_quality_lc > 0 else ""
        )

        optional_quality_ct = (
            f"AND quality_score_ct >= {min_quality_ct}" if min_quality_ct > 0 else ""
        )

        self.assert_valid_spatial_extent(spatial_extent)

        # Create a shapely polygon from the bounding box
        if isinstance(spatial_extent, BoundingBoxExtent):
            geometry = Polygon(
                [
                    (spatial_extent.west, spatial_extent.south),
                    (spatial_extent.east, spatial_extent.south),
                    (spatial_extent.east, spatial_extent.north),
                    (spatial_extent.west, spatial_extent.north),
                    (spatial_extent.west, spatial_extent.south),
                ]
            )
        else:
            geometry = spatial_extent

        # Inward buffer of the patches
        buffer = -20
        logger.info(f"Buffering patch extents by {buffer} meters")
        gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        gdf_utm["geometry"] = gdf_utm.buffer(
            buffer, cap_style=3
        )  # cap_style=3 makes square corners
        gdf = gdf_utm.to_crs("EPSG:4326")
        geometry = gdf.geometry.values[0]

        for i, url in enumerate(urls):
            ref_id = str(url).split("/")[-2]
            query = f"""
                SELECT {columns_str}, ST_AsWKB(ST_Intersection(ST_MakeValid(ST_Simplify(geometry, 0.000001)), ST_Simplify(ST_GeomFromText('{str(geometry)}'), 0.000001))) AS wkb_geometry, '{ref_id}' AS ref_id
                FROM read_parquet('{url}')
                WHERE ST_Intersects(ST_MakeValid(ST_Simplify(geometry, 0.000001)), ST_Simplify(ST_GeomFromText('{str(geometry)}'), 0.000001))
                {optional_temporal}
                {optional_ewoc_codes}
                {optional_subset}
                {optional_quality_lc}
                {optional_quality_ct}
            """
            if i == 0:
                combined_query = query
            else:
                combined_query += f" UNION ALL {query}"

        return combined_query

    def get_samples(
        self,
        ref_ids: Optional[List[str]] = None,
        columns: List[str] = RDM_DEFAULT_COLUMNS,
        subset: Optional[bool] = False,
        spatial_extent: Optional[
            Union[BoundingBoxExtent, Polygon, MultiPolygon]
        ] = None,
        temporal_extent: Optional[TemporalContext] = None,
        ewoc_codes: Optional[List[int]] = None,
        include_public: Optional[bool] = True,
        include_private: Optional[bool] = False,
        min_quality_lc: int = 0,
        min_quality_ct: int = 0,
        ground_truth_file: Optional[Union[Path, str]] = None,
    ) -> gpd.GeoDataFrame:
        """Queries the RDM API and generates a GeoPandas GeoDataframe of all samples meeting the search criteria.

        Parameters
        ----------
        ref_ids : Optional(List[str]), optional
            List of collection IDs to download samples from.
            If not specified, all collections matching the search criteria defined by
            the other input parameters will be queried.
        columns : List[str], optional
            A list of column names to extract., by default DEFAULT_COLUMNS
        subset : Optional[bool], optional
            If True, only download a subset of the samples (for which extract attribute ==1)
            If False, extract all samples.
            Default is False.
        spatial_extent : Optional[Union[BoundingBoxExtent, Polygon, MultiPolygon]], optional
            A user-defined bounding box or shapely Polygon or MultiPolygon for which all intersecting samples need to be found.
            CRS should be EPSG:4326.
            If None, all available data will be queried, by default None
        temporal_extent : TemporalContext, optional
            Temporal extent, defined by start and end date, by default None.
            If None, all available data will be queried.
        ewoc_codes: Optional[List[int]] = None
            If specified, only samples with the specified EWOC codes will be extracted.
        include_public: Optional[bool] = True
            Whether or not to include public collections.
        include_private: Optional[bool] = False
            Whether or not to include private collections.
        min_quality_lc: int = 0
            Minimum quality score for land cover [0-100].
        min_quality_ct: int = 0
            Minimum quality score for crop type [0-100].
        ground_truth_file: Optional[Union[Path, str]] = None
            Optional path to a ground truth file. If provided, the query to the
            RDM is bypassed and the ground truth file is used instead.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing the extracted samples.
        """

        if ground_truth_file is None:
            # Determine which collections need to be queried if they are not specified
            if not ref_ids:
                collections = self.get_collections(
                    spatial_extent=spatial_extent,
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
                ref_ids = [col.id for col in collections]

            logger.info(f"Querying {len(ref_ids)} collections...")

            # For each collection, get the download URL
            urls = self._get_download_urls(ref_ids)
        else:
            logger.info(f"Querying ground truth from: {ground_truth_file}")
            urls = [str(ground_truth_file)]

        # Ensure we have a valid spatial_extent
        if spatial_extent is None:
            spatial_extent = BoundingBoxExtent(
                west=-180, south=-90, east=180, north=90, epsg=4326
            )

        # Set up the SQL query
        query = self._setup_sql_query(
            urls=urls,
            spatial_extent=spatial_extent,
            columns=columns,
            temporal_extent=temporal_extent,
            ewoc_codes=ewoc_codes,
            subset=subset,
            min_quality_lc=min_quality_lc,
            min_quality_ct=min_quality_ct,
        )

        # Execute the query
        con = duckdb.connect()
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        con.execute("SET TimeZone='UTC';")

        df = con.execute(query).fetch_df()

        # Convert the WKB geometry to a Shapely geometry
        df["geometry"] = df["wkb_geometry"].apply(lambda x: wkb.loads(bytes(x)))
        df.drop(columns=["wkb_geometry"], inplace=True)
        # Make sure df contains only requested columns
        df = df[columns]

        # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        return gdf

    def get_collection_metadata(self, ref_id: str) -> dict:
        """Get metadata for a collection.

        Parameters
        ----------
        ref_id : str
            The collection ID.

        Raises
        ------
        Exception
            If the request fails.

        Returns
        -------
        dict
            A dictionary containing the metadata for the collection.
        """
        url = f"{self.RDM_ENDPOINT}/collections/{ref_id}/metadata/items"
        response = self.session.get(url, headers=self._get_headers(), timeout=10)
        if response.status_code != 200:
            raise Exception(f"Error fetching collection metadata: {response.text}")

        # convert to nice dictionary
        metadata_dict = {}
        for item in response.json():
            metadata_dict[item["name"]] = item["value"]

        return metadata_dict

    def get_collection_stats(
        self, ref_id: str, stats_type: str = "crop_type"
    ) -> pd.DataFrame:
        """Extract crop statistics from the metadata of a collection.

        Parameters:
        ----------
            ref_id : str
                The collection ID.
            stats_type (str): Type of statistics to extract. Default is "crop_type".
                Possible values are: "crop_type", "irrigation" and "land_cover".

        Returns:
        -------
            pd.DataFrame: DataFrame containing crop statistics.

        Raises:
        -------
            ValueError: If no statistics are found for the collection or for the specified type.
        """

        # Get the metadata
        metadata = self.get_collection_metadata(ref_id)

        # Get the crop statistics from the metadata
        stats = metadata.get("codeStats", None)

        if stats is None:
            raise ValueError("No statistics found for this collection.")
        else:
            stats = json.loads(stats)

        # Extract the desired statistics
        if stats_type == "crop_type":
            field = "EwocStats"
        elif stats_type == "irrigation":
            field = "IrrStats"
        elif stats_type == "land_cover":
            field = "LcStats"
        else:
            raise ValueError(
                "Invalid statistics type, please select one of the following: land_cover, crop_type or irrigation."
            )

        stats = stats.get(field, None)

        if stats is None:
            raise ValueError(f"No {stats_type} statistics found for this collection.")

        # Create a DataFrame from the crop statistics
        df = pd.DataFrame(stats)
        df = df.set_index("Code")

        # add labels column
        labels = ewoc_code_to_label(df.index.values)
        df["Label"] = labels

        return df

    def download_collection_metadata(self, ref_id: str, dst_path: str) -> str:
        """Download metadata for a specific collection as xlsx file.

        Parameters
        ----------
        ref_id : str
            The collection ID.
        dst_path : str
            The folder name where the metadata file should be saved.

        Returns
        -------
        str
            The path to the downloaded metadata file.
        """

        url = f"{self.RDM_ENDPOINT}/collections/{ref_id}/metadata/download"
        response = self.session.get(url, headers=self._get_headers(), timeout=10)

        if response.status_code != 200:
            raise Exception(f"Error downloading collection metadata: {response.text}")

        outfile = Path(dst_path) / f"{ref_id}_metadata.xlsx"
        Path(dst_path).mkdir(parents=True, exist_ok=True)
        with open(outfile, "wb") as f:
            f.write(response.content)

        logger.info(f"Metadata for collection {ref_id} downloaded to {dst_path}")

        return str(outfile)

    def download_collection_geoparquet(
        self, ref_id: str, dst_path: str, subset: bool = False
    ) -> str:
        """Download the features (samples) from a specific collection
            as geoparquet file.

        Parameters
        ----------
        ref_id : str
            The collection ID.
        dst_path : str
            The folder name where the GeoParquet file should be saved.
        subset : bool
            If True, only download a subset of the full collection.
            Defaults to False.

        Returns
        -------
        str
            The path to the downloaded GeoParquet file.

        Raises
        ------
        Exception
            If the request fails.
        """

        url = self._get_download_urls([ref_id], subset=subset)[0]

        # Download the file directly
        part2 = "_samples" if subset else ""
        filename = f"{ref_id}{part2}.parquet"
        outfile = Path(dst_path) / filename
        Path(dst_path).mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=(5, 120))
        if response.status_code != 200:
            raise Exception(
                f"Error downloading samples for collection {ref_id}: {response.text}"
            )
        with open(outfile, "wb") as f:
            f.write(response.content)

        logger.info(f"Samples for collection {ref_id} downloaded to {outfile}")

        return str(outfile)

    def download_collection_harmonization_info(self, ref_id: str, dst_path: str) -> str:
        """Download the harmonization information for a specific collection
            as a PDF file. Only works for public collections!

        Parameters
        ----------
        ref_id : str
            The collection ID.
        dst_path : str
            The folder name where the PDF file should be saved.

        Returns
        -------
        str
            The path to the downloaded PDF file.

        Raises
        ------
        Exception
            If the request fails.
        """

        # Get the metadata
        metadata = self.get_collection_metadata(ref_id)

        # Get the correct link from the metadata
        download_link = metadata["CuratedDataSet:Harmonization:Pdf"]

        # Download the file directly
        filename = download_link.split("/")[-1]
        outfile = Path(dst_path) / filename
        Path(dst_path).mkdir(parents=True, exist_ok=True)
        response = requests.get(download_link)
        if response.status_code != 200:
            raise Exception(
                f"Error downloading harmonization PDF for collection {ref_id}: {response.text}"
            )
        with open(outfile, "wb") as f:
            f.write(response.content)

        logger.info(
            f"Harmonization PDF for collection {ref_id} downloaded to {outfile}"
        )

        return str(outfile)

    def assert_valid_spatial_extent(
        self, spatial_extent: Union[BoundingBoxExtent, Polygon, MultiPolygon]
    ) -> None:
        """Validate that the given spatial extent is in EPSG:4326 and is either a BoundingBoxExtent or shapely Polygon or MultiPolygon.

        Parameters
        ----------
        spatial_extent : Union[BoundingBoxExtent, Polygon, MultiPolygon]
            The spatial_extent to check.


        Raises
        ------
        ValueError
            If the spatial_extent is not in EPSG:4326 or either a BoundingBoxExtent or shapely Polygon or MultiPolygon
        """

        if isinstance(spatial_extent, BaseGeometry):
            if not isinstance(spatial_extent, Polygon) and not isinstance(
                spatial_extent, MultiPolygon
            ):
                raise ValueError(
                    "Spatial extent should be either a BoundingBoxExtent or a shapely.geometry.Polygon or shapely.geometry.MultiPolygon."
                )

            spatial_extent = spatial_extent.bounds
            spatial_extent = BoundingBoxExtent(
                west=spatial_extent[0],
                south=spatial_extent[1],
                east=spatial_extent[2],
                north=spatial_extent[3],
                epsg=4326,
            )

        elif not isinstance(spatial_extent, BoundingBoxExtent):
            raise ValueError(
                "Spatial extent should be either a BoundingBoxExtent or a shapely.geometry.Polygon or shapely.geometry.MultiPolygon."
            )

        if (
            spatial_extent.west < -180
            or spatial_extent.south < -90
            or spatial_extent.east > 180
            or spatial_extent.north > 90
        ):
            raise ValueError("Invalid spatial_extent. CRS should be EPSG:4326.")
