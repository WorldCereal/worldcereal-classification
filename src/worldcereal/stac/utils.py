import os
import pystac
from typing import List
from worldcereal.extract.utils import pipeline_log
from worldcereal.stac.stac_api_interaction import StacApiInteraction, VitoStacApiAuthentication

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