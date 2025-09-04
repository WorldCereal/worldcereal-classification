from openeo.extra.job_management import (
        MultiBackendJobManager
    )
from pathlib import Path
from typing import Callable, Optional, Union
import threading
import pystac
import pickle
import pandas as pd
from openeo.rest.job import BatchJob
from worldcereal.extract.utils import pipeline_log



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


    # -------------------------
    # STAC helper methods
    # -------------------------
    #TODO validate the whole STAC stuff
    def _load_stac(self) -> Optional[pystac.Collection]:
        if self._catalogue_cache.exists():
            with open(self._catalogue_cache, "rb") as f:
                return pickle.load(f)
        elif self.stac_path and Path(self.stac_path).exists():
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
            collection = self._create_stac()
        return collection

    def _persist_stac(self):
        if self._root_collection:
            with open(self._catalogue_cache, "wb") as f:
                pickle.dump(self._root_collection, f)

    def _update_stac(self, job_id: str, items: list[pystac.Item]):
        if not self._root_collection:
            return
        with self.lock:
            existing_ids = {item.id for item in self._root_collection.get_all_items()}
            new_items = [item for item in items if item.id not in existing_ids]
            self._root_collection.add_items(new_items)
            self._persist_stac()

    def write_stac(self):
        """Write the final STAC collection to output_dir/stac/collection.json."""
        if not self._root_collection:
            return
        stac_dir = self._output_dir / "stac"
        stac_dir.mkdir(parents=True, exist_ok=True)
        self._root_collection.set_self_href(str(stac_dir / "collection.json"))
        self._root_collection.normalize_hrefs(str(stac_dir))
        self._root_collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    # -------------------------
    # Job handling
    # -------------------------
    def on_job_done(self, job: BatchJob, row: pd.Series):
        try:
            job_products = self._download_job_results(job, row)
            job_metadata = job.get_results().get_metadata()
            job_items = self._process_job_items(job, job_products, job_metadata)

            # Call user-defined post-job action
            if self._post_job_action:
                pipeline_log.info("Running post_job_action for job %s", job.job_id)
                job_items = self._post_job_action(job_items,
                                                  row,
                                                  extract_value=1) #TODO figure out how to pass this value correctly

            # STAC integration
            if self.stac_enabled:
                pipeline_log.info("Running post_job_action for job %s", job.job_id)
                stac_items = []
                for item_info in job_items:
                    # Convert each processed job item into a minimal pystac.Item
                    item_id = f"{job.job_id}_{item_info['asset_id']}"
                    item = pystac.Item(
                        id=item_id,
                        geometry=None,
                        bbox=None,
                        datetime=None,
                        properties={}
                    )
                    item.add_asset(
                        key=item_info['asset_id'],
                        asset=pystac.Asset(href=str(item_info['path']))
                    )
                    stac_items.append(item)

            #TODO enable
            # self._update_stac(job.job_id, stac_items)

            pipeline_log.info("Job %s processed successfully.", job.job_id)

        except Exception as e:
            pipeline_log.warning("Error processing job %s: %s", job.job_id, e)
            raise

    def _download_job_results(self, job: BatchJob, row: pd.Series) -> dict:
        job_products = {}
        job_results = job.get_results()
        for idx, asset_id in enumerate([a.name for a in job_results.get_assets()]):
            asset = job_results.get_asset(asset_id)
            output_path = self._output_path_gen(self._output_dir, row, asset_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            asset.download(output_path)
            job_products[asset_id] = {"path": output_path, "asset": asset}
        return job_products

    def _process_job_items(self, job: BatchJob, job_products: dict, job_metadata: dict) -> list:
        """Convert job results to proper PySTAC Items."""
        job_collection = pystac.Collection.from_dict(job_metadata)
        job_items = []

        for item_metadata in job_collection.get_all_items():
            try:
                item = pystac.read_file(item_metadata.get_self_href())
                
                # Get the asset name from the item
                asset_name = list(item.assets.values())[0].title
                
                # Use the asset name directly as the key (without job ID prefix)
                asset_path = job_products[asset_name]["path"]
                
                # Update asset href to local downloaded path
                for asset in item.assets.values():
                    asset.href = str(asset_path)
                
                job_items.append(item)
                pipeline_log.info("Processed item %s from job %s", item.id, job.job_id)
                
            except Exception as e:
                pipeline_log.exception(
                    "Failed to process item %s from job %s: %s",
                    item_metadata.id, job.job_id, e
                )
                raise e
        
        return job_items