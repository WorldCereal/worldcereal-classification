import argparse
import json
from functools import partial
from pathlib import Path
from typing import List, Literal, Union

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
from worldcereal.utils import parse_job_options_from_args


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
    parquet_files: Union[List[Path], List[str]],
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
    connection: openeo.Connection,
    ref_id: str,
    ground_truth_file: str,
    root_folder: Path,
    job_options: dict[str, Union[str, int, None]],
    period: str = "month",
    restart_failed: bool = False,
    only_flagged_samples: bool = False,
    parallel_jobs: int = 1,
    optical_mask_method: Literal[
        "mask_scl_dilation", "satio", "mask_scl_raw_values"
    ] = "mask_scl_dilation",
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
    job_options : dict
        openEO job options. May contain None values (they will fall back to defaults).
    period : str, optional
        Period for extractions, either 'month' or 'dekad'. Default is 'month'.
    restart_failed : bool, optional
        Whether to restart failed jobs. Default is False.
    only_flagged_samples : bool, optional
        If True, only samples with extract flag >0 will be retrieved, no collateral samples.
        (This is useful for very large and dense datasets like USDA).
    parallel_jobs : int, optional
        Number of local parallel jobs to run concurrently. Default is 1.
    optical_mask_method : Literal["mask_scl_dilation", "satio", "mask_scl_raw_values"], optional
        Method to use for optical masking. Default is 'mask_scl_dilation' which uses the default precomputed mask with large erosion/dilation radius.
        This method is the fastest.
        'satio' allows to configure custom erode/dilation radius but can make the whole process a lot slower as the mask is computed on-the-fly.
        A proxy between speed and quality is 'mask_scl_raw_values' which uses the raw SCL values for masking without erosion/dilation.
        This option is available for patch-to-point only.

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
            ref_id, ground_truth_file, only_flagged_samples
        )
        job_db.initialize_from_df(job_df)

    manager = PatchToPointJobManager(root_dir=output_folder)
    manager.add_backend(
        "terrascope", connection=connection, parallel_jobs=parallel_jobs
    )
    manager.run_jobs(
        start_job=partial(
            create_job_patch_to_point_worldcereal,
            period=period,
            job_options=job_options,
            optical_mask_method=optical_mask_method,
        ),
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
    merged_gdf = (
        merged_gdf.drop_duplicates()
    )  # Ensure no duplicates in case of double extractions
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
        "--optical-mask-method",
        type=str,
        choices=["mask_scl_dilation", "satio", "mask_scl_raw_values"],
        default="mask_scl_dilation",
        help="Method to use for optical masking. Default is 'mask_scl_dilation' which uses the default precomputed mask with large erosion/dilation radius. "
        "This method is the fastest. 'satio' allows to configure custom erode/dilation radius but can make the whole process a lot slower as the mask is computed on-the-fly. "
        "A proxy between speed and quality is 'mask_scl_raw_values' which uses the raw SCL values for masking without erosion/dilation. This option is available for patch-to-point only.",
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
    parser.add_argument(
        "--only-flagged-samples",
        action="store_true",
        help="If True, only samples with extract flag >0 will be retrieved, no collateral samples.",
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=None,
        help="Local parallel jobs.",
    )
    parser.add_argument("--driver_memory", type=str, default=None, help="Driver memory")
    parser.add_argument(
        "--driver_memoryOverhead",
        type=str,
        default=None,
        help="Driver memory overhead.",
    )
    parser.add_argument(
        "--executor_cores", type=int, default=None, help="Executor cores."
    )
    parser.add_argument(
        "--executor_memory", type=str, default=None, help="Executor memory."
    )
    parser.add_argument(
        "--executor_memoryOverhead",
        type=str,
        default=None,
        help="Executor memory overhead.",
    )
    parser.add_argument(
        "--max_executors", type=int, default=None, help="Max executors."
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="python38",
        help="openEO image name.",  # Use python 3.8 by default, until patch-to-point works on 3.11 https://github.com/eu-cdse/openeo-cdse-infra/issues/738
    )
    parser.add_argument(
        "--organization_id", type=int, default=None, help="Organization id."
    )

    args = parser.parse_args()

    root_folder = Path(args.root_folder)
    period = args.period
    optical_mask_method = args.optical_mask_method
    ref_ids = args.ref_ids
    restart_failed = args.restart_failed
    only_flagged_samples = args.only_flagged_samples
    parallel_jobs = args.parallel_jobs or 1
    job_options = parse_job_options_from_args(args)

    logger.info("Starting patch to point extractions ...")
    logger.info(f"Root folder: {root_folder}")
    logger.info(f"Period: {period}")

    connection = openeo.connect("openeo.vito.be").authenticate_oidc()

    for ref_id in ref_ids:
        logger.info(f"Processing ref_id: {ref_id}")
        ground_truth_file = (
            f"/vitodata/worldcereal/data/RDM/{ref_id}/harmonized/{ref_id}.geoparquet"
        )

        main(
            connection=connection,
            ref_id=ref_id,
            ground_truth_file=ground_truth_file,
            root_folder=root_folder,
            period=period,
            restart_failed=restart_failed,
            only_flagged_samples=only_flagged_samples,
            parallel_jobs=parallel_jobs,
            job_options=job_options,
            optical_mask_method=optical_mask_method,
        )

    logger.success("All done!")
