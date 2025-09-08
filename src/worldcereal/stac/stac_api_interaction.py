from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional

import pystac
import os
import requests
from openeo.rest.auth.oidc import (
    OidcClientInfo,
    OidcProviderInfo,
    OidcResourceOwnerPasswordAuthenticator,
)
from requests.auth import AuthBase
from worldcereal.extract.utils import pipeline_log


class VitoStacApiAuthentication(AuthBase):
    """Class that handles authentication for the VITO STAC API. https://stac.openeo.vito.be/"""

    def __init__(self, **kwargs):
        self.username = kwargs.get("username")
        self.password = kwargs.get("password")

    def __call__(self, request):
        request.headers["Authorization"] = self.get_access_token()
        return request

    def get_access_token(self) -> str:
        """Get API bearer access token via password flow.

        Returns
        -------
        str
            A string containing the bearer access token.
        """
        provider_info = OidcProviderInfo(
            issuer="https://sso.terrascope.be/auth/realms/terrascope"
        )

        client_info = OidcClientInfo(
            client_id="terracatalogueclient",
            provider=provider_info,
        )

        if self.username and self.password:
            authenticator = OidcResourceOwnerPasswordAuthenticator(
                client_info=client_info, username=self.username, password=self.password
            )
        else:
            raise ValueError(
                "Credentials are required to obtain an access token. Please set STAC_API_USERNAME and STAC_API_PASSWORD environment variables."
            )

        tokens = authenticator.get_tokens()

        return f"Bearer {tokens.access_token}"

class StacApiInteraction:
    """
    Handles interaction with a STAC API root and a specific collection ID.
    Use stac_root to point to the STAC API root (e.g. "https://.../stac")
    and collection_id for the collection you want to operate on.
    """

    def __init__(
        self,
        stac_root: str,
        collection_id: Optional[str],
        auth: AuthBase,
        bulk_size: int = 500,
    ):
        # normalize root URL (no trailing slash)
        self.base_url = stac_root.rstrip("/")
        self.collection_id = collection_id
        self.auth = auth
        self.bulk_size = bulk_size

    def _join_url(self, url_path: str) -> str:
        # safe join ensuring exactly one slash between parts
        return self.base_url + "/" + url_path.lstrip("/")

    def exists(self) -> bool:
        """
        If self.collection_id is provided, check GET /collections/{collection_id}.
        If collection_id is None, check that the STAC root is reachable (GET root).
        """
        # check root reachable
        root_resp = requests.get(self.base_url, auth=self.auth)
        if root_resp.status_code != requests.codes.ok:
            # root unreachable -> treat as non-existent / error
            return False

        if not self.collection_id:
            # root exists and no collection specified
            return True

        coll_resp = requests.get(
            self._join_url(f"collections/{self.collection_id}"), auth=self.auth
        )
        return coll_resp.status_code == requests.codes.ok

    def create_collection(self, description: Optional[str] = None):
        spatial_extent = pystac.SpatialExtent([[-180, -90, 180, 90]])
        temporal_extent = pystac.TemporalExtent([[None, None]])
        extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)
        collection = pystac.Collection(
            id=self.collection_id,
            description=description or f"Collection {self.collection_id}",
            extent=extent,
        )
        collection.validate()
        coll_dict = collection.to_dict()
        default_auth = {
            "_auth": {
                "read": ["anonymous"],
                "write": ["stac-openeo-admin", "stac-openeo-editor"],
            }
        }
        coll_dict.update(default_auth)
        response = requests.post(
            self._join_url("collections"), auth=self.auth, json=coll_dict
        )
        expected_status = [
            requests.status_codes.codes.ok,
            requests.status_codes.codes.created,
            requests.status_codes.codes.accepted,
        ]
        self._check_response_status(response, expected_status)
        return response

    def add_item(self, item: pystac.Item):
        if not self.exists():
            self.create_collection()
        self._prepare_item(item)
        url_path = f"collections/{self.collection_id}/items"
        response = requests.post(self._join_url(url_path), auth=self.auth, json=item.to_dict())
        expected_status = [
            requests.status_codes.codes.ok,
            requests.status_codes.codes.created,
            requests.status_codes.codes.accepted,
        ]
        self._check_response_status(response, expected_status)
        return response

    def _prepare_item(self, item: pystac.Item):
        if self.collection_id:
            item.collection_id = self.collection_id
            if not item.get_links(pystac.RelType.COLLECTION):
                item.add_link(
                    pystac.Link(rel=pystac.RelType.COLLECTION, target=item.collection_id)
                )

    def _ingest_bulk(self, items: Iterable[pystac.Item]) -> dict:
        if not all(i.collection_id == self.collection_id for i in items):
            raise Exception("All collection IDs should be identical for bulk ingests")
        url_path = f"collections/{self.collection_id}/bulk_items"
        data = {
            "method": "upsert",
            "items": {item.id: item.to_dict() for item in items},
        }
        response = requests.post(url=self._join_url(url_path), auth=self.auth, json=data)
        expected_status = [
            requests.status_codes.codes.ok,
            requests.status_codes.codes.created,
            requests.status_codes.codes.accepted,
        ]
        self._check_response_status(response, expected_status)
        return response.json()

    def upload_items_bulk(self, items: Iterable[pystac.Item]) -> None:
        if not self.exists():
            self.create_collection()

        chunk = []
        futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for item in items:
                self._prepare_item(item)
                chunk.append(item)
                if len(chunk) >= self.bulk_size:
                    futures.append(executor.submit(self._ingest_bulk, chunk.copy()))
                    chunk = []
            # submit any final chunk
            if chunk:
                futures.append(executor.submit(self._ingest_bulk, chunk.copy()))

            # wait for and surface exceptions if any
            for fut in as_completed(futures):
                _res = fut.result()  # will raise if underlying call raised

    def _check_response_status(self, response: requests.Response, expected_status_codes: list[int]):
        if response.status_code not in expected_status_codes:
            message = (
                f"Expecting HTTP status to be any of {expected_status_codes} "
                + f"but received {response.status_code} - {response.reason}, request method={response.request.method}\n"
                + f"response body:\n{response.text}"
            )
            raise Exception(message)

    def get_collection_id(self) -> Optional[str]:
        return self.collection_id


def upload_to_stac_api(items: List[pystac.Item], collection_id: str, stac_root_url: str) -> None:
    """Debug version of STAC API upload with detailed logging."""
    pipeline_log.info(f"Preparing to upload {len(items)} items to STAC API")
    username = os.getenv("STAC_API_USERNAME")
    password = os.getenv("STAC_API_PASSWORD")
    if not username or not password:
        error_msg = "STAC API credentials not found."
        pipeline_log.error(error_msg)
        raise ValueError(error_msg)

    stac_api_interaction = StacApiInteraction(
        stac_root=stac_root_url,
        collection_id=collection_id,
        auth=VitoStacApiAuthentication(username=username, password=password),
    )

    pipeline_log.info(f"Checking if collection '{collection_id}' exists...")
    collection_exists_before = stac_api_interaction.exists()
    pipeline_log.info(f"Collection exists before upload: {collection_exists_before}")

    if not collection_exists_before:
        pipeline_log.info("Collection doesn't exist, attempting to create it...")
        try:
            stac_api_interaction.create_collection()
            pipeline_log.info("Collection creation attempted")
            collection_exists_after = stac_api_interaction.exists()
            pipeline_log.info(f"Collection exists after creation attempt: {collection_exists_after}")
        except Exception as e:
            pipeline_log.error(f"Collection creation failed: {e}")
            raise

    pipeline_log.info(f"Starting bulk upload of {len(items)} items...")
    try:
        stac_api_interaction.upload_items_bulk(items)
        pipeline_log.info("Bulk upload completed")
    except Exception as e:
        pipeline_log.error(f"Bulk upload failed: {e}")
        raise