import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import pystac
import pystac_client
import requests
from openeo.rest.auth.oidc import (
    OidcAuthCodePkceAuthenticator,
    OidcClientInfo,
    OidcProviderInfo,
    OidcResourceOwnerPasswordAuthenticator,
)
from requests.auth import AuthBase


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
            authenticator = OidcAuthCodePkceAuthenticator(client_info=client_info)

        tokens = authenticator.get_tokens()

        return f"Bearer {tokens.access_token}"


class StacApiInteraction:
    """Class that handles the interaction with a STAC API."""

    def __init__(
        self, sensor: str, base_url: str, auth: AuthBase, bulk_size: int = 500
    ):
        if sensor not in ["Sentinel1", "Sentinel2"]:
            raise ValueError(
                f"Invalid sensor '{sensor}'. Allowed values are 'Sentinel1' and 'Sentinel2'."
            )
        self.sensor = sensor
        self.base_url = base_url
        self.collection_id = f"worldcereal_{sensor.lower()}_patch_extractions"

        self.auth = auth

        self.client = pystac_client.Client.open(base_url)

        self.bulk_size = bulk_size

    def exists(self) -> bool:
        return (
            len(
                [
                    c.id
                    for c in self.client.get_collections()
                    if c.id == self.collection_id
                ]
            )
            > 0
        )

    def _join_url(self, url_path: str) -> str:
        return str(self.base_url + "/" + url_path)

    def create_collection(self):
        spatial_extent = pystac.SpatialExtent([[-180, -90, 180, 90]])
        temporal_extent = pystac.TemporalExtent([[None, None]])
        extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)

        collection = pystac.Collection(
            id=self.collection_id,
            description=f"WorldCereal Patch Extractions for {self.sensor}",
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
        response = requests.post(
            self._join_url(url_path), auth=self.auth, json=item.to_dict()
        )

        expected_status = [
            requests.status_codes.codes.ok,
            requests.status_codes.codes.created,
            requests.status_codes.codes.accepted,
        ]

        self._check_response_status(response, expected_status)

        return response

    def _prepare_item(self, item: pystac.Item):
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
        response = requests.post(self._join_url(url_path), auth=self.auth, json=data)

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

                if len(chunk) == self.bulk_size:
                    futures.append(executor.submit(self._ingest_bulk, chunk.copy()))
                    chunk = []

            if chunk:
                self._ingest_bulk(chunk)

            for _ in concurrent.futures.as_completed(futures):
                continue

    def _check_response_status(
        self, response: requests.Response, expected_status_codes: list[int]
    ):
        if response.status_code not in expected_status_codes:
            message = (
                f"Expecting HTTP status to be any of {expected_status_codes} "
                + f"but received {response.status_code} - {response.reason}, request method={response.request.method}\n"
                + f"response body:\n{response.text}"
            )

            raise Exception(message)

    def get_collection_id(self) -> str:
        return self.collection_id
