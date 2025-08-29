from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
from openeo.extra.job_management import MultiBackendJobManager
from openeo.rest import BatchJob
import logging
import pystac
import threading
import pickle
from pystac import CatalogType

_log = logging.getLogger(__name__)


class ExtractionJobManager(MultiBackendJobManager):
    """A simplified job manager that handles job execution, post-processing, and optional STAC integration."""

    def __init__(
        self,
        output_dir: Path,
        output_path_generator: Callable,
        post_job_action: Optional[Callable] = None,
        poll_sleep: int = 5,
        stac_enabled: bool = False,
        collection_id: Optional[str] = None,
        collection_description: Optional[str] = "",
        stac_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        super().__init__(poll_sleep=poll_sleep, **kwargs)

        self._output_dir = output_dir
        self._output_path_gen = output_path_generator
        self._post_job_action = post_job_action

        # STAC support
        self.stac_enabled = stac_enabled
        self.collection_id = collection_id
        self.collection_description = collection_description
        self.stac_path = stac_path
        self.lock = threading.Lock()
        self._catalogue_cache = self._output_dir / "catalogue_cache.bin"

        self._root_collection = None
        if self.stac_enabled:
            self._root_collection = self._initialize_stac()

        _log.info("Extraction job manager initialized with STAC enabled=%s", stac_enabled)

    # -------------------------
    # STAC helper methods
    # -------------------------
    #TODO validate the whole STAC stuff
    def _load_stac(self) -> Optional[pystac.Collection]:
        if self._catalogue_cache.exists():
            _log.info("Loading STAC from cache: %s", self._catalogue_cache)
            with open(self._catalogue_cache, "rb") as f:
                return pickle.load(f)
        elif self.stac_path and Path(self.stac_path).exists():
            _log.info("Loading STAC from provided path: %s", self.stac_path)
            return pystac.read_file(str(self.stac_path))
        return None

    def _create_stac(self) -> pystac.Collection:
        if not self.collection_id:
            raise ValueError("collection_id must be set to create a STAC collection.")
        collection = pystac.Collection(
            id=self.collection_id,
            description=self.collection_description,
            extent=None,
        )
        return collection

    def _initialize_stac(self) -> pystac.Collection:
        collection = self._load_stac()
        if not collection:
            _log.info("Creating a new STAC collection.")
            collection = self._create_stac()
        return collection

    def _persist_stac(self):
        if self._root_collection:
            _log.info("Persisting STAC collection to cache: %s", self._catalogue_cache)
            with open(self._catalogue_cache, "wb") as f:
                pickle.dump(self._root_collection, f)

    def _update_stac(self, job_id: str, items: list[pystac.Item]):
        if not self._root_collection:
            return
        with self.lock:
            existing_ids = {item.id for item in self._root_collection.get_all_items()}
            new_items = [item for item in items if item.id not in existing_ids]
            self._root_collection.add_items(new_items)
            _log.info("Added %d items to STAC collection from job %s", len(new_items), job_id)
            self._persist_stac()

    def write_stac(self):
        """Write the final STAC collection to output_dir/stac/collection.json."""
        if not self._root_collection:
            return
        stac_dir = self._output_dir / "stac"
        stac_dir.mkdir(parents=True, exist_ok=True)
        self._root_collection.set_self_href(str(stac_dir / "collection.json"))
        self._root_collection.normalize_hrefs(str(stac_dir))
        self._root_collection.save(catalog_type=CatalogType.SELF_CONTAINED)
        _log.info("STAC collection saved to %s", stac_dir / "collection.json")

    # -------------------------
    # Job handling
    # -------------------------
    def on_job_done(self, job: BatchJob, row: pd.Series):
        _log.info("Job %s finished, processing...", job.job_id)
        try:
            #job_products = self._download_job_results(job, row)
            #job_metadata = job.get_results().get_metadata()
            #job_items = self._process_job_items(job, job_products, job_metadata)

            # Call user-defined post-job action
            #if self._post_job_action:
            #    _log.debug("Running post_job_action for job %s", job.job_id)
            #    job_items = self._post_job_action(job_items, row)

            # STAC integration
            #if self.stac_enabled:
            #    stac_items = []
            #    for item_info in job_items:
            #        # Convert each processed job item into a minimal pystac.Item
            #        item_id = f"{job.job_id}_{item_info['asset_id']}"
            #        item = pystac.Item(
            #            id=item_id,
            #            geometry=None,
            #            bbox=None,
            #            datetime=None,
            #            properties={}
            #        )
            #        item.add_asset(
            #            key=item_info['asset_id'],
            #            asset=pystac.Asset(href=str(item_info['path']))
            #        )
            #        stac_items.append(item)
            #
            #    self._update_stac(job.job_id, stac_items)

            _log.info("Job %s processed successfully.", job.job_id)

        except Exception as e:
            _log.exception("Error processing job %s: %s", job.job_id, e)
            raise

    def _download_job_results(self, job: BatchJob, row: pd.Series) -> dict:
        job_products = {}
        job_results = job.get_results()
        for idx, asset_id in enumerate([a.name for a in job_results.get_assets()]):
            asset = job_results.get_asset(asset_id)
            output_path = self._output_path_gen(self._output_dir, idx, row, asset_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            asset.download(output_path)
            job_products[asset_id] = {"path": output_path, "asset": asset}
        return job_products

    def _process_job_items(self, job: BatchJob, job_products: dict, job_metadata: dict) -> list:
        items = []
        for asset_id, info in job_products.items():
            items.append({
                "job_id": job.job_id,
                "asset_id": asset_id,
                "path": info["path"],
                "asset": info["asset"],
                "metadata": job_metadata
            })
        return items