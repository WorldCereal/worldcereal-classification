"""STAC-specific utility functions."""

import os
from typing import List
import pystac
import pystac_client

from worldcereal.extract.utils import pipeline_log
from worldcereal.stac.stac_api_interaction import StacApiInteraction, VitoStacApiAuthentication

STAC_ROOT_URL = "https://stac.openeo.vito.be/"

# Collection mapping stays here since it's STAC-specific
PATCH_COLLECTIONS = {
    "PATCH_SENTINEL1": "hv_test_worldcereal_sentinel_1_patch_extractions",
    "PATCH_SENTINEL2": "hv_test_worldcereal_sentinel_2_patch_extractions",
}

def get_collection_id(collection_name: str) -> str:
    """Get STAC collection ID from collection name."""
    return PATCH_COLLECTIONS.get(collection_name)

def fetch_existing_sample_ids(collection_id: str, ref_id: str) -> List[str]:
    """
    Fetch the IDs of existing samples in the specified collection and reference ID.
    Pure STAC operation - no DataFrame knowledge.
    """
    client = pystac_client.Client.open(STAC_ROOT_URL)
    search = client.search(
        collections=[collection_id],
        filter={"op": "=", "args": [{"property": "properties.ref_id"}, ref_id]},
        filter_lang="cql2-json",
        fields={"exclude": ["assets", "links", "geometry", "bbox"]},
    )
    return [item.properties.get("sample_id") for item in search.items() if item.properties.get("sample_id")]

def update_stac_item_metadata(item: pystac.Item, new_attributes: dict) -> None:
    """Update STAC item metadata with new attributes."""
    pipeline_log.info(f"Updating STAC item metadata for {item.id}")
    item.properties.update(new_attributes)
    item.properties["providers"] = [{"name": "openEO platform"}]
    extension = "https://stac-extensions.github.io/processing/v1.2.0/schema.json"
    item.stac_extensions.extend([extension])

def upload_to_stac_api(job_items: List[pystac.Item], sensor: str) -> None:
    """Upload items to STAC API."""
    pipeline_log.info("Preparing to upload items to STAC API")
    username = os.getenv("STAC_API_USERNAME")
    password = os.getenv("STAC_API_PASSWORD")

    if not username or not password:
        error_msg = "STAC API credentials not found. Please set STAC_API_USERNAME and STAC_API_PASSWORD."
        pipeline_log.error(error_msg)
        raise ValueError(error_msg)

    stac_api_interaction = StacApiInteraction(
        sensor=sensor,
        base_url="https://stac.openeo.vito.be",
        auth=VitoStacApiAuthentication(username=username, password=password),
    )

    pipeline_log.info("Writing the STAC API metadata")
    stac_api_interaction.upload_items_bulk(job_items)
    pipeline_log.info("STAC API metadata written")