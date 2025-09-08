from typing import Optional, Callable, Union
from pathlib import Path
import pandas as pd
from openeo.rest.job import BatchJob
from openeo.extra.job_management import MultiBackendJobManager
from worldcereal.extract.utils import pipeline_log
import pystac
from worldcereal.stac.stac_handler import StacHandler  # Import from separate module

class ExtractionJobManager(MultiBackendJobManager):
    """A simplified job manager that handles job execution, post-processing, and optional STAC integration."""

    def __init__(
        self,
        output_dir: Path,
        output_path_generator: Callable,
        post_job_action: Optional[Callable] = None,
        poll_sleep: int = 5,
        stac_handler: Optional[StacHandler] = None,  # Accept StacHandler instance
        **kwargs
    ):
        super().__init__(poll_sleep=poll_sleep, **kwargs)
        self._output_dir = output_dir
        self._output_path_gen = output_path_generator
        self._post_job_action = post_job_action
        self.stac_handler = stac_handler  # Store the handler instance

    def on_job_done(self, job: BatchJob, row: pd.Series):
        try:
            job_products = self._download_job_results(job, row)
            job_metadata = job.get_results().get_metadata()
            
            # Process job items
            #TODO check output; collection seems to be wrong
            job_items = self._process_job_items(job, job_products, job_metadata)
            
            # Apply post-job action if provided
            if self._post_job_action:
                job_items = self._post_job_action(job_items, row, extract_value=1)
            
            # Update STAC catalog if handler is available
            if self.stac_handler:
                self.stac_handler.update_stac(job.job_id, job_items)

            pipeline_log.info("Job %s processed successfully.", job.job_id)

        except Exception as e:
            pipeline_log.error("Error processing job %s: %s", job.job_id, e)
            raise

    def _download_job_results(self, job: BatchJob, row: pd.Series) -> dict:
        job_products = {}
        job_results = job.get_results()
        for asset_id in [a.name for a in job_results.get_assets()]:
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
                
                # Update asset href to local downloaded path
                asset_path = job_products[asset_name]["path"]
                for asset in item.assets.values():
                    asset.href = str(asset_path)
                
                job_items.append(item)
                pipeline_log.info("Processed item %s from job %s", item.id, job.job_id)
                
            except Exception as e:
                pipeline_log.exception(
                    "Failed to process item %s from job %s: %s",
                    item_metadata.id, job.job_id, e
                )
                raise
        
        return job_items