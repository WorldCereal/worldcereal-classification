import argparse
import json
from functools import partial
from pathlib import Path
from typing import List, Union

import geopandas as gpd
import openeo
import pandas as pd
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import CsvJobDatabase, MultiBackendJobManager

from worldcereal.extract.patch_to_point_worldcereal import (
    create_job_dataframe_patch_to_point_worldcereal,
    create_job_patch_to_point_worldcereal,
    generate_output_path_patch_to_point_worldcereal,
    post_job_action_point_worldcereal,
)


class PatchToPointJobManager(MultiBackendJobManager):
    def on_job_done(self, job: BatchJob, row):
        logger.info(f"Job {job.job_id} completed")
        output_file = generate_output_path_patch_to_point_worldcereal(
            self._root_dir, 0, row
        )
        job.get_results().download_file(target=output_file, name="timeseries.parquet")

        job_metadata = job.describe()
        metadata_path = output_file.parent / f"job_{job.job_id}.json"
        self.ensure_job_dir_exists(job.job_id)

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(job_metadata, f, ensure_ascii=False)

        post_job_action_point_worldcereal(output_file)
        logger.success("Job completed")


def merge_individual_parquet_files(
    parquet_files: List[Union[Path, str]],
) -> gpd.GeoDataFrame:
    """
    Merge individual parquet files into a single GeoDataFrame.

    Parameters
    ----------
    parquet_files : list of Union[Path, str]
        List of paths to individual parquet files.

    Returns
    -------
    gpd.GeoDataFrame
        Merged GeoDataFrame containing data from all the parquet files.

    Raises
    ------
    ValueError
        If more than 25% of the rows have missing attributes.
    """

    seen_ids: set[str] = set()
    gdfs = []

    # Iterate over each parquet file manually to remove any
    # duplicate sample_ids across files
    for file in parquet_files:
        gdf = gpd.read_parquet(file)
        # Keep only rows with unseen sample_ids
        gdf_filtered = gdf[~gdf["sample_id"].isin(seen_ids)].copy()

        # Update the seen set
        seen_ids.update(gdf_filtered["sample_id"])

        gdfs.append(gdf_filtered)

    # Concatenate after all filtering is done
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    # Check for missing attributes based on one columns
    missing_attrs = gdf["start_date"].isnull()
    if missing_attrs.sum() > 0.25 * len(gdf):
        error_msg = (
            r"More than 25% of the rows have missing attributes. "
            "Please check extractions! No merged parquet will be generated."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif missing_attrs.any():
        logger.warning(
            f"Missing attributes in {missing_attrs.sum()} rows. "
            "This may indicate an issue during extraction. Rows are removed"
        )
        gdf = gdf[~missing_attrs]

    # Make sure we remove the timezone information from the timestamp
    gdf["timestamp"] = gdf["timestamp"].dt.tz_localize(None)

    return gdf


def main(
    connection,
    ref_id,
    ground_truth_file,
    root_folder,
    period="month",
    restart_failed=False,
):
    """
    Main function to orchestrate patch-to-point extractions.

    Parameters
    ----------
    connection : openeo.Connection
        OpenEO connection object.
    ref_id : str
        Reference ID for the extraction.
    ground_truth_file : str
        Path to the ground truth file.
    root_folder : Path
        Root folder for storing extraction outputs.
    period : str, optional
        Period for extractions, either 'month' or 'dekad'. Default is 'month'.
    restart_failed : bool, optional
        Whether to restart failed jobs. Default is False.

    Returns
    -------
    None
    """

    assert period in ["month", "dekad"], "Period must be either 'month' or 'dekad'."

    # Ref_id output folder
    output_folder = (
        root_folder / ref_id if period == "month" else root_folder / f"{ref_id}_10D"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    job_tracking_path = output_folder / "job_tracking.csv"
    job_db = CsvJobDatabase(path=job_tracking_path)

    if job_db.exists():
        logger.info(f"Job tracking file found at {job_tracking_path}.")
        if restart_failed:
            logger.info("Resetting failed jobs.")
            job_df = job_db.read()
            job_df.loc[
                job_df["status"].isin(["error", "postprocessing-error"]), "status"
            ] = "not_started"
            job_db.persist(job_df)

    if not job_db.exists():
        logger.info("Job tracking file does not exist, creating new jobs.")
        job_df = create_job_dataframe_patch_to_point_worldcereal(
            ref_id, ground_truth_file
        )
        job_db.initialize_from_df(job_df)

    manager = PatchToPointJobManager(root_dir=output_folder)
    manager.add_backend("terrascope", connection=connection, parallel_jobs=1)
    manager.run_jobs(
        start_job=partial(create_job_patch_to_point_worldcereal, period=period),
        job_db=job_db,
    )

    # Merge all subparquets
    logger.info("Merging individual files ...")
    parquet_files = list(output_folder.rglob("*.geoparquet"))
    if len(parquet_files) != len(job_df):
        raise ValueError(
            f"Number of parquet files ({len(parquet_files)}) does not match number of jobs ({len(job_df)})"
        )
    logger.info(f"Found {len(parquet_files)} parquet files.")
    merged_gdf = merge_individual_parquet_files(parquet_files)
    merged_dir = (
        root_folder / "MERGED_PARQUETS"
        if period == "month"
        else root_folder / "MERGED_PARQUETS_10D"
    )
    merged_file = merged_dir / f"{ref_id}.geoparquet"
    merged_gdf.to_parquet(merged_file, index=False)
    logger.info(f"Merged parquet file saved to: {merged_file}")

    logger.success("ref_id fully done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run patch-to-point extractions for WorldCereal."
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        required=True,
        help="Root folder for storing extraction outputs.",
    )
    parser.add_argument(
        "--period",
        type=str,
        choices=["month", "dekad"],
        default="month",
        help="Period for extractions, either 'month' or 'dekad'. Default is 'month'.",
    )
    parser.add_argument(
        "--ref-ids",
        type=str,
        nargs="+",
        required=True,
        help="List of ref_ids to process.",
    )
    parser.add_argument(
        "--restart-failed",
        action="store_true",
        help="Restart failed jobs if the job tracking file exists.",
    )

    args = parser.parse_args()

    root_folder = Path(args.root_folder)
    period = args.period
    ref_ids = args.ref_ids
    restart_failed = args.restart_failed

    logger.info("Starting patch to point extractions ...")
    logger.info(f"Root folder: {root_folder}")
    logger.info(f"Period: {period}")

    connection = openeo.connect("openeo.vito.be").authenticate_oidc()

    for ref_id in ref_ids:
        logger.info(f"Processing ref_id: {ref_id}")
        ground_truth_file = (
            f"/vitodata/worldcereal/data/RDM/{ref_id}/harmonized/{ref_id}.geoparquet"
        )

        main(connection, ref_id, ground_truth_file, root_folder, period, restart_failed)

    logger.success("All done!")
