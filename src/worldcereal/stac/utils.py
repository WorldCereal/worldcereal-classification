"""STAC-specific utility functions."""
from typing import List
import pystac
import pystac_client

from worldcereal.extract.utils import pipeline_log


def fetch_existing_sample_ids(collection_id: str, ref_id: str, stac_root_url: str) -> List[str]:
    """
    Fetch the IDs of existing samples in the specified collection and reference ID.
    Pure STAC operation - no DataFrame knowledge.
    """
    client = pystac_client.Client.open(stac_root_url)
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

