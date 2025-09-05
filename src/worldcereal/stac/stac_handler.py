from pathlib import Path
from typing import Optional
import pystac
import threading
import pickle
from worldcereal.extract.utils import pipeline_log

class StacHandler:
    def __init__(self, output_dir: Path, collection_id: str, collection_description: str = ""):
        self.output_dir = output_dir
        self.collection_id = collection_id
        self.collection_description = collection_description
        self._catalogue_cache = output_dir / "catalogue_cache.bin"
        self.lock = threading.Lock()
        self._root_collection = self._initialize_stac()

    def _load_stac(self) -> Optional[pystac.Collection]:
        if self._catalogue_cache.exists():
            with open(self._catalogue_cache, "rb") as f:
                return pickle.load(f)
        return None

    def _create_stac(self) -> pystac.Collection:
        if not self.collection_id:
            raise ValueError("collection_id must be set to create a STAC collection.")
        return pystac.Collection(
            id=self.collection_id,
            description=self.collection_description,
            extent=None,
        )

    def _initialize_stac(self) -> pystac.Collection:
        collection = self._load_stac()
        if not collection:
            collection = self._create_stac()
        return collection

    def _persist_stac(self):
        if self._root_collection:
            with open(self._catalogue_cache, "wb") as f:
                pickle.dump(self._root_collection, f)

    def update_stac(self, job_id: str, items: list[pystac.Item]):
        """Update STAC catalog with new items"""
        if not self._root_collection:
            return
        with self.lock:
            existing_ids = {item.id for item in self._root_collection.get_all_items()}
            new_items = [item for item in items if item.id not in existing_ids]
            self._root_collection.add_items(new_items)
            self._persist_stac()
            pipeline_log.info("Updated STAC catalog with %s new items from job %s", len(new_items), job_id)

    def write_stac(self):
        """Write the final STAC collection to output_dir/stac/collection.json."""
        if not self._root_collection:
            return
        stac_dir = self.output_dir / "stac"
        stac_dir.mkdir(parents=True, exist_ok=True)
        self._root_collection.update_extent_from_items()
        self._root_collection.set_self_href(str(stac_dir / "collection.json"))
        self._root_collection.normalize_hrefs(str(stac_dir))
        self._root_collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
        pipeline_log.info("STAC collection saved to %s", stac_dir)

   