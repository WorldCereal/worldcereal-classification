"""Quality check functions for extraction jobs."""

from pathlib import Path
import xarray as xr
import pandas as pd

from worldcereal.extract.utils import pipeline_log
from openeo_gfmap import Backend
from openeo_gfmap.backend import BACKEND_CONNECTIONS

def extraction_job_quality_check(job_entry: pd.Series, orfeo_error_threshold: float = 0.3) -> None:
    """Perform quality checks on an extraction job."""
    
    conn = BACKEND_CONNECTIONS[Backend[job_entry["backend_name"].upper()]]()
    job = conn.job(job_entry.id)

    # Check if we have any assets resulting from the job
    job_results = job.get_results()
    if len(job_results.get_metadata()["assets"]) == 0:
        raise Exception(f"Job {job_entry.id} has no assets!")

    # Check if SAR backscatter error ratio exceeds the threshold
    if "sar_backscatter_soft_errors" in job.describe()["usage"].keys():
        actual_orfeo_error_rate = job.describe()["usage"]["sar_backscatter_soft_errors"]["value"]
        if actual_orfeo_error_rate > orfeo_error_threshold:
            raise Exception(f"Job {job_entry.id} had a ORFEO error rate of {actual_orfeo_error_rate}!")

    pipeline_log.debug("Quality checks passed!")

def validate_dataset_dimensions(ds: xr.Dataset, item_asset_path: Path, spatial_resolution: str) -> None:
    """Validate dataset dimensions match expected resolution."""
    pipeline_log.info(f"Validating dataset dimensions for {item_asset_path}")
    expected_dim_sizes = {"10m": 64, "20m": 32}
    expected_dim_size = expected_dim_sizes.get(spatial_resolution)
    
    if expected_dim_size is None:
        return
    
    actual_x_size = ds.dims.get("x", 0)
    actual_y_size = ds.dims.get("y", 0)

    if actual_x_size != expected_dim_size or actual_y_size != expected_dim_size:
        pipeline_log.error(
            "Dimension validation failed for %s: expected %dx%d for %s resolution, got %dx%d",
            item_asset_path,
            expected_dim_size,
            expected_dim_size,
            spatial_resolution,
            actual_x_size,
            actual_y_size,
        )
        raise ValueError(
            f"Invalid dimensions for {spatial_resolution} resolution: "
            f"expected {expected_dim_size}x{expected_dim_size}, got {actual_x_size}x{actual_y_size}"
        )

    pipeline_log.debug("Dimension validation passed")

def verify_file_integrity(item_asset_path: Path) -> None:
    """Verify that the output file is not corrupt."""
    pipeline_log.info(f"Verifying file integrity for {item_asset_path}")
    try:
        with xr.open_dataset(item_asset_path) as src:
            src.load()
    except Exception as e:
        pipeline_log.error("The output file %s is corrupt. Error: %s", item_asset_path, e)
        raise

