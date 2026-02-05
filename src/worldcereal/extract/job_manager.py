import json
from pathlib import Path
from typing import Callable, Union

import openeo
import pandas as pd
import pystac
from openeo.extra.job_management import MultiBackendJobManager

from worldcereal.extract.utils import pipeline_log


class ExtractionJobManager(MultiBackendJobManager):
    def __init__(
        self,
        poll_sleep: int,
        root_dir: Union[Path, str],
        output_path_generator: Callable,
        post_job_action: Callable,
    ):
        super().__init__(poll_sleep=poll_sleep, root_dir=root_dir)
        self.output_path_generator = output_path_generator
        self.post_job_action = post_job_action

    def _download_job_products(self, job: openeo.BatchJob, row: pd.Series) -> dict:
        job_products = {}
        job_results = job.get_results()
        asset_ids = [a.name for a in job_results.get_assets()]
        for idx, asset_id in enumerate(asset_ids):
            try:
                asset = job_results.get_asset(asset_id)
                pipeline_log.debug(
                    "Generating output path for asset %s from job %s...",
                    asset_id,
                    job.job_id,
                )
                output_path = self.output_path_generator(
                    self._root_dir, idx, row, asset_id
                )
                # Make the output path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                asset.download(output_path)
                # Add to the list of downloaded products
                job_products[f"{job.job_id}_{asset_id}"] = [output_path]
                pipeline_log.debug(
                    "Downloaded %s from job %s -> %s",
                    asset_id,
                    job.job_id,
                    output_path,
                )
            except Exception as e:
                pipeline_log.exception(
                    "Error downloading asset %s from job %s:\n%s",
                    asset_id,
                    job.job_id,
                    e,
                )
                raise e
        return job_products

    def _process_stac_items(self, job: openeo.BatchJob, job_products: dict) -> list:
        collection = pystac.Collection.from_dict(job.get_results().get_metadata())
        job_items = []
        for item_metadata in collection.get_all_items():
            try:
                item = pystac.read_file(item_metadata.get_self_href())
                asset_name = list(item.assets.values())[0].title
                asset_path = job_products[f"{job.job_id}_{asset_name}"][0]

                assert len(item.assets.values()) == 1, (
                    "Each item should only contain one asset"
                )
                for asset in item.assets.values():
                    asset.href = str(
                        asset_path
                    )  # Update the asset href to the output location set by the output_path_generator

                # Add the item to the the current job items.
                job_items.append(item)
                pipeline_log.info("Parsed item %s from job %s", item.id, job.job_id)
            except Exception as e:
                pipeline_log.exception(
                    "Error failed to add item %s from job %s to STAC collection:\n%s",
                    item.id,
                    job.job_id,
                    e,
                )
        return job_items

    def on_job_done(self, job: openeo.BatchJob, row: pd.Series):
        """Method called when a job finishes successfully.
        Parameters
        ----------
        job: BatchJob
            The job that finished successfully.
        row: pd.Series
            The row in the dataframe that contains the job relative information.
        """
        pipeline_log.debug("Downloading products for job %s...", job.job_id)
        job_products = self._download_job_products(job, row)
        pipeline_log.debug("Finished downloading products for job %s.", job.job_id)

        pipeline_log.debug("Processing STAC items for job %s...", job.job_id)
        job_items = self._process_stac_items(job, job_products)
        pipeline_log.debug("Finished processing STAC items for job %s.", job.job_id)

        pipeline_log.debug("Calling post job action for job %s...", job.job_id)
        job_items = self.post_job_action(job_items, row)
        pipeline_log.debug("Finished post job action for job %s.", job.job_id)

    def on_job_error(self, job: openeo.BatchJob, row: pd.Series):
        """Method called when a job finishes with an error.

        Parameters
        ----------
        job: BatchJob
            The job that finished with an error.
        row: pd.Series
            The row in the dataframe that contains the job relative information.
        """
        try:
            logs = job.logs()
        except Exception as e:  # pylint: disable=broad-exception-caught
            pipeline_log.exception(
                "Error getting logs in `on_job_error` for job %s:\n%s", job.job_id, e
            )
            logs = []

        error_logs = [log for log in logs if log.level.lower() == "error"]

        job_metadata = job.describe_job()
        title = job_metadata["title"]
        job_id = job_metadata["id"]

        output_log_path = Path(self._root_dir) / "failed_jobs" / f"{title}_{job_id}.log"
        output_log_path.parent.mkdir(parents=True, exist_ok=True)

        if len(error_logs) > 0:
            output_log_path.write_text(json.dumps(error_logs, indent=2))
        else:
            output_log_path.write_text(
                f"Couldn't find any error logs. Please check the error manually on job ID: {job.job_id}."
            )
