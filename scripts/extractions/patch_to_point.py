import argparse
import json
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import openeo
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from geopandas.io.arrow import _geopandas_to_arrow
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


def _prepare_temp_output_path(output_path: Path) -> Path:
    """
    Create a deterministic temporary file next to the target to enable atomic rename.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    return temp_path


def _strip_index_columns(schema: pa.Schema) -> pa.Schema:
    """
    Remove pandas index helper columns from an Arrow schema.
    """
    drop_indices: list[int] = []
    drop_names: list[str] = []
    for idx, name in enumerate(schema.names):
        if name.startswith("__index_level_"):
            drop_indices.append(idx)
            drop_names.append(name)

    if not drop_indices:
        return schema

    for idx in sorted(drop_indices, reverse=True):
        schema = schema.remove(idx)

    metadata = schema.metadata
    if metadata and b"pandas" in metadata:
        pandas_meta = json.loads(metadata[b"pandas"].decode("utf-8"))
        pandas_meta["columns"] = [
            column
            for column in pandas_meta.get("columns", [])
            if column.get("name") not in drop_names
        ]
        pandas_meta["index_columns"] = [
            column
            for column in pandas_meta.get("index_columns", [])
            if column not in drop_names
        ]
        metadata = dict(metadata)
        metadata[b"pandas"] = json.dumps(pandas_meta).encode("utf-8")
        schema = schema.with_metadata(metadata)

    return schema


def merge_individual_parquet_files(
    parquet_files: Union[List[Path], List[str]], output_path: Path
) -> dict[str, int]:
    """
    Merge individual parquet files into a single geoparquet without loading the entire
    dataset in memory.

    Parameters
    ----------
    parquet_files : list of Union[Path, str]
        List of paths to individual parquet files.
    output_path : pathlib.Path
        Target path where the merged geoparquet should be written.

    Returns
    -------
    dict[str, int]
        Basic statistics about the merge process.

    Raises
    ------
    ValueError
        If more than 25% of the rows have missing attributes.
    KeyError
        If required columns are missing.
    """

    if not parquet_files:
        raise ValueError("No parquet files provided for merging.")

    parquet_paths = [Path(p) for p in parquet_files]
    parquet_paths.sort()

    schema_source = pq.ParquetFile(str(parquet_paths[0])).schema_arrow
    schema_source = _strip_index_columns(schema_source)
    schema_names = list(schema_source.names)

    temp_output = _prepare_temp_output_path(output_path)

    seen_ids: set = set()
    total_rows = 0
    missing_rows = 0
    written_rows = 0
    duplicated_rows = 0

    writer: Optional[pq.ParquetWriter] = None

    try:
        for file_path in parquet_paths:
            gdf = gpd.read_parquet(file_path)
            if gdf.empty:
                continue

            if "sample_id" not in gdf.columns:
                raise KeyError(f"'sample_id' column missing in {file_path}")
            if "start_date" not in gdf.columns:
                raise KeyError(f"'start_date' column missing in {file_path}")

            # Ensure we operate on the same column order as the schema template.
            missing_columns = [col for col in schema_names if col not in gdf.columns]
            for column in missing_columns:
                gdf[column] = pd.NA
            extra_columns = [col for col in gdf.columns if col not in schema_names]
            if extra_columns:
                gdf = gdf.drop(columns=extra_columns)
            gdf = gdf[schema_names]

            mask = ~gdf["sample_id"].isin(seen_ids)
            if not mask.any():
                duplicated_rows += len(gdf)
                continue

            gdf_filtered = gdf.loc[mask].copy()
            duplicated_rows += len(gdf) - len(gdf_filtered)

            # Track new ids and counts before removing rows with missing attributes.
            seen_ids.update(gdf_filtered["sample_id"].dropna().tolist())
            total_rows += len(gdf_filtered)

            missing_mask = gdf_filtered["start_date"].isnull()
            missing_rows += int(missing_mask.sum())
            if missing_mask.any():
                gdf_filtered = gdf_filtered.loc[~missing_mask]

            if gdf_filtered.empty:
                continue

            if "timestamp" in gdf_filtered.columns:
                gdf_filtered["timestamp"] = pd.to_datetime(
                    gdf_filtered["timestamp"], errors="coerce"
                ).dt.tz_localize(None)

            gdf_filtered = gdf_filtered.drop_duplicates()
            if gdf_filtered.empty:
                continue

            gdf_filtered = gdf_filtered.reset_index(drop=True)

            # Preserve GeoParquet metadata by relying on GeoPandas' arrow conversion helper.
            arrow_table = _geopandas_to_arrow(gdf_filtered, index=None)
            if arrow_table.schema != schema_source:
                arrow_table = arrow_table.cast(schema_source, safe=False)

            if writer is None:
                writer = pq.ParquetWriter(temp_output, schema_source)

            writer.write_table(arrow_table)
            written_rows += arrow_table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if total_rows:
        missing_ratio = missing_rows / total_rows
    else:
        missing_ratio = 0.0

    if missing_ratio > 0.25:
        if temp_output.exists():
            temp_output.unlink()
        error_msg = (
            r"More than 25% of the rows have missing attributes. "
            "Please check extractions! No merged parquet will be generated."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif missing_rows:
        logger.warning(
            f"Missing attributes in {missing_rows} rows. "
            "This may indicate an issue during extraction. Rows are removed"
        )

    if writer is None:
        # No rows survived the filters; persist an empty geoparquet with the same schema.
        empty_table = schema_source.empty_table()
        pq.write_table(empty_table, temp_output)

    temp_output.replace(output_path)

    return {
        "total_rows": total_rows,
        "missing_rows": missing_rows,
        "written_rows": written_rows,
        "duplicate_sample_rows": duplicated_rows,
    }


def _normalize_h3_cell(value: Union[str, float, int, None]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    normalized = str(value).strip()
    return normalized or None


def _discover_completed_jobs(output_folder: Path) -> dict[int, set[Optional[str]]]:
    """
    Inspect existing parquet outputs to identify completed EPSG/H3 jobs.
    """
    completed: dict[int, set[Optional[str]]] = {}
    if not output_folder.exists():
        return completed

    for parquet_file in output_folder.rglob("*.geoparquet"):
        try:
            epsg = int(parquet_file.parent.name)
        except ValueError:
            continue

        target_set = completed.setdefault(epsg, set())

        try:
            df = pd.read_parquet(parquet_file, columns=["h3_l3_cell"])
        except (FileNotFoundError, ValueError, KeyError, OSError):
            target_set.add(None)
            continue

        column = df.get("h3_l3_cell")
        if column is None:
            target_set.add(None)
            continue

        unique_values = {_normalize_h3_cell(v) for v in column.unique()}
        if not unique_values or unique_values == {None}:
            target_set.add(None)
        else:
            target_set.update(unique_values)

    return completed


def _filter_completed_jobs(
    job_df: pd.DataFrame, completed_jobs: dict[int, set[Optional[str]]]
) -> tuple[pd.DataFrame, dict[tuple[int, Optional[str]], int]]:
    """
    Remove rows that are already covered by existing parquet outputs.
    """
    if job_df.empty or not completed_jobs:
        return job_df, {}

    keep_mask: list[bool] = []
    skipped: dict[tuple[int, Optional[str]], int] = {}

    for _, row in job_df.iterrows():
        epsg = int(row["epsg"])
        completed_for_epsg = completed_jobs.get(epsg, set())
        h3_value = _normalize_h3_cell(row.get("h3l3_cell", None))

        if h3_value in completed_for_epsg:
            keep_mask.append(False)
            key = (epsg, h3_value)
            skipped[key] = skipped.get(key, 0) + 1
        else:
            keep_mask.append(True)

    filtered_df = job_df.loc[keep_mask].reset_index(drop=True)
    return filtered_df, skipped


def main(
    connection: openeo.Connection,
    ref_id: str,
    ground_truth_file: str,
    root_folder: Path,
    job_options: dict[str, Union[str, int, None]],
    period: str = "month",
    restart_failed: bool = False,
    only_flagged_samples: bool = False,
    max_samples_per_job: Optional[int] = None,
    parallel_jobs: int = 1,
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
    max_samples_per_job : int, optional
        Maximum number of samples allowed per job before splitting by H3 L3 cell identifiers.
        If None, additional splitting is disabled.
    parallel_jobs : int, optional
        Number of local parallel jobs to run concurrently. Default is 1.

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
    completed_jobs = _discover_completed_jobs(output_folder)

    if job_db.exists():
        logger.info(f"Job tracking file found at {job_tracking_path}.")
        job_df = job_db.read()
        if restart_failed:
            logger.info("Resetting failed jobs.")
            job_df.loc[
                job_df["status"].isin(["error", "postprocessing-error"]), "status"
            ] = "not_started"
            job_db.persist(job_df)
        tracked_keys = {
            (int(row["epsg"]), _normalize_h3_cell(row.get("h3l3_cell", None)))
            for _, row in job_df.iterrows()
        }
        missing_tracked_outputs = sum(
            1
            for epsg_value, h3_values in completed_jobs.items()
            for h3_value in h3_values
            if (epsg_value, h3_value) not in tracked_keys
        )
        if missing_tracked_outputs:
            logger.info(
                f"Detected {missing_tracked_outputs} parquet output(s) already present on disk but missing from tracking."
            )

    if not job_db.exists():
        logger.info("Job tracking file does not exist, creating new jobs.")
        job_df = create_job_dataframe_patch_to_point_worldcereal(
            ref_id,
            ground_truth_file,
            only_flagged_samples,
            max_samples_per_job,
        )
        job_df, skipped_jobs = _filter_completed_jobs(job_df, completed_jobs)
        if skipped_jobs:
            total_skipped = sum(skipped_jobs.values())
            logger.info(
                f"Detected {total_skipped} already processed job(s); skipping their rerun."
            )
            for (epsg_value, h3_value), count in sorted(skipped_jobs.items()):
                if h3_value is None:
                    logger.info(f"  - EPSG {epsg_value}: {count} job(s) already completed.")
                else:
                    logger.info(
                        f"  - EPSG {epsg_value}, H3 {h3_value}: {count} job(s) already completed."
                    )
        if job_df.empty:
            logger.info(
                "All jobs appear to be completed already; creating empty tracking file."
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
        ),
        job_db=job_db,
    )

    # Merge all subparquets
    logger.info("Merging individual files ...")
    parquet_files = list(output_folder.rglob("*.geoparquet"))
    job_df = job_db.read()
    completed_statuses = {"finished"}
    completed_jobs_df = job_df[job_df["status"].isin(completed_statuses)]
    incomplete_jobs = job_df[~job_df["status"].isin(completed_statuses)]
    if not incomplete_jobs.empty:
        statuses = ", ".join(sorted({str(status) for status in incomplete_jobs["status"].unique()}))
        raise ValueError(
            f"{len(incomplete_jobs)} job(s) still have non-completed status ({statuses}); aborting merge."
        )
    expected_output_count = len(completed_jobs_df)
    if len(parquet_files) < expected_output_count:
        raise ValueError(
            "Found {files} parquet file(s) but {expected} job(s) are marked finished; outputs missing.".format(
                files=len(parquet_files), expected=expected_output_count
            )
        )
    extra_files = len(parquet_files) - expected_output_count
    if extra_files > 0:
        logger.info(
            f"Found {extra_files} additional parquet file(s) beyond tracked finished jobs; merging everything."
        )
    logger.info(f"Found {len(parquet_files)} parquet files.")
    merged_dir = (
        root_folder / "MERGED_PARQUETS"
        if period == "month"
        else root_folder / "MERGED_PARQUETS_10D"
    )
    merged_file = merged_dir / f"{ref_id}.geoparquet"
    merge_stats = merge_individual_parquet_files(parquet_files, merged_file)
    if merge_stats["duplicate_sample_rows"]:
        logger.info(
            f"Skipped {merge_stats['duplicate_sample_rows']} row(s) with duplicate sample_id values."
        )
    logger.info(
        f"Merged parquet file saved to: {merged_file} "
        f"({merge_stats['written_rows']} rows written)."
    )

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
    parser.add_argument(
        "--only-flagged-samples",
        action="store_true",
        help="If True, only samples with extract flag >0 will be retrieved, no collateral samples.",
    )
    parser.add_argument(
        "--max_samples_per_job",
        type=int,
        default=None,
        help="Maximum number of samples allowed per job before splitting by H3 L3 cells.",
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
        "--image_name", type=str, default="python38", help="openEO image name."  # Use python 3.8 by default, until patch-to-point works on 3.11 https://github.com/eu-cdse/openeo-cdse-infra/issues/738
    )
    parser.add_argument(
        "--organization_id", type=int, default=None, help="Organization id."
    )

    args = parser.parse_args()

    root_folder = Path(args.root_folder)
    period = args.period
    ref_ids = args.ref_ids
    restart_failed = args.restart_failed
    only_flagged_samples = args.only_flagged_samples
    max_samples_per_job = args.max_samples_per_job
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
            max_samples_per_job=max_samples_per_job,
            parallel_jobs=parallel_jobs,
            job_options=job_options,
        )

    logger.success("All done!")
