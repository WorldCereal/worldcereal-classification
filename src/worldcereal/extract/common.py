"""Common functions used by extraction scripts."""

import json
import os
import shutil
from datetime import datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, Iterable, List, Optional, Union

import geopandas as gpd
import pandas as pd
import pystac
import pystac_client
import xarray as xr
import grp

#TODO move this dependence off

from openeo_gfmap import Backend
from openeo_gfmap.backend import BACKEND_CONNECTIONS

from openeo_gfmap.manager.job_splitters import split_job_s2grid

from worldcereal.extract.jobmanager import ExtractionJobManager  

from worldcereal.extract.patch_meteo import (
    create_job_dataframe_patch_meteo,
    create_job_patch_meteo,
)
from worldcereal.extract.patch_s2 import (
    create_job_dataframe_patch_s2,
    create_job_patch_s2,
)
from worldcereal.extract.point_worldcereal import (
    create_job_dataframe_point_worldcereal,
    create_job_point_worldcereal,
    generate_output_path_point_worldcereal,
    merge_output_files_point_worldcereal,
    post_job_action_point_worldcereal,
)
from worldcereal.extract.utils import pipeline_log
from worldcereal.stac.constants import ExtractionCollection
from worldcereal.stac.stac_api_interaction import (
    StacApiInteraction,
    VitoStacApiAuthentication,
)

from worldcereal.extract.patch_s1 import (  # isort: skip
    create_job_patch_s1,
    create_job_dataframe_patch_s1,
)

from worldcereal.extract.patch_worldcereal import (  # isort: skip
    create_job_patch_worldcereal,
    create_job_dataframe_patch_worldcereal,
    post_job_action_patch_worldcereal,
    generate_output_path_patch_worldcereal,
)


STAC_ROOT_URL = "https://stac.openeo.vito.be/"

PATCH_COLLECTIONS = {
    "PATCH_SENTINEL1": "worldcereal_sentinel_1_patch_extractions",
    "PATCH_SENTINEL2": "worldcereal_sentinel_2_patch_extractions",
}

# ----------------------------
# Utility helpers
# ----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def safe_chown_group(path: Path, group_name: str = "vito") -> None:
    try:
        gid = grp.getgrnam(group_name).gr_gid
    except KeyError:
        pipeline_log.debug("Group '%s' not found; skipping chown for %s", group_name, path)
        return
    try:
        shutil.chown(path, group=gid)
    except PermissionError:
        pipeline_log.debug("No permission to chown %s to group %s", path, group_name)
    except Exception as e:
        pipeline_log.warning("Unexpected error changing group for %s: %s", path, e)

def is_parquet_path(path: Path) -> bool:
    return path.name.lower().endswith((".parquet", ".parq"))

# ----------------------------
# Job status helpers
# ----------------------------
def _count_by_status(job_status_df: pd.DataFrame, statuses: Iterable[str] = ()) -> dict:
    """Count the number of jobs by their status."""
    status_histogram = job_status_df.groupby("status").size().to_dict()
    if statuses:
        status_histogram = {k: v for k, v in status_histogram.items() if k in set(statuses)}
    return status_histogram

def _read_job_tracking_csv(output_folder: Path) -> pd.DataFrame:
    """Read the job tracking CSV file."""
    job_status_file = output_folder / "job_tracking.csv"
    if job_status_file.exists():
        return pd.read_csv(job_status_file)
    raise FileNotFoundError(f"Job status file not found at {job_status_file}")

def check_job_status(output_folder: Path) -> dict:
    """Check the status of jobs in the specified output folder."""
    job_status_df = _read_job_tracking_csv(output_folder)
    status_histogram = _count_by_status(job_status_df)
    status_count = pd.DataFrame(status_histogram.items(), columns=["status", "count"]).sort_values("count", ascending=False)
    pipeline_log.info("\nOverall jobs status:\n%s", status_count.to_string(index=False))
    return status_histogram

# ----------------------------
# Successful Job Statistics
# ----------------------------
def get_succeeded_job_details(output_folder: Path) -> pd.DataFrame:
    """
    Get details of succeeded jobs from the job tracking CSV file.
    """
    job_status_df = _read_job_tracking_csv(output_folder)
    succeeded_jobs = _filter_succeeded_jobs(job_status_df)
    if succeeded_jobs.empty:
        return pd.DataFrame()
    succeeded_jobs = _derive_job_metrics(succeeded_jobs)
    _print_job_summary(succeeded_jobs)
    return succeeded_jobs

def _filter_succeeded_jobs(job_status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the job status DataFrame to only include succeeded jobs.
    """
    return job_status_df[job_status_df["status"].isin(["finished", "postprocessing"])].copy()

def _derive_job_metrics(succeeded_jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional metrics from the succeeded jobs DataFrame.
    """
    succeeded_jobs["n_samples"] = succeeded_jobs["geometry"].apply(lambda g: len(json.loads(g)["features"]))
    columns = ["id", "s2_tile", "n_samples", "duration", "costs"]
    succeeded_jobs = succeeded_jobs[columns]
    succeeded_jobs["duration"] = succeeded_jobs["duration"].fillna("0s")
    seconds = succeeded_jobs["duration"].str.rstrip("s").astype(int)
    succeeded_jobs["duration_mins"] = seconds / 60
    if succeeded_jobs["duration_mins"].sum() == 0:
        succeeded_jobs.drop(columns=["duration_mins"], inplace=True)
    return succeeded_jobs

def _print_job_summary(succeeded_jobs: pd.DataFrame) -> None:
    """
    Print a summary of the succeeded jobs.
    """
    pipeline_log.info("-" * 37)
    if pd.notnull(succeeded_jobs["costs"]).all():
        total_credits = int(succeeded_jobs["costs"].sum())
        avg_credits = int(succeeded_jobs["costs"].mean())
        pipeline_log.info("Total credits: %s, Avg cost: %s over %s jobs", total_credits, avg_credits, len(succeeded_jobs))
    pipeline_log.info("Number of samples extracted: %s", succeeded_jobs["n_samples"].sum())
    pipeline_log.info("Succeeded job details:\n%s", succeeded_jobs.to_string(index=False))
    pipeline_log.info("-" * 37)


# ----------------------------
# Quality check
# ----------------------------
def extraction_job_quality_check(
    job_entry: pd.Series, orfeo_error_threshold: float = 0.3
) -> None:
    """Perform quality checks on an extraction job.

    Parameters
    ----------
    job_entry : pd.Series
        The job entry containing information about the extraction job.
    orfeo_error_threshold : float, optional
        The threshold for the SAR backscatter error ratio, by default 0.3.

    Raises
    ------
    Exception
        Raised if the job has no assets.
    Exception
        Raised if the SAR backscatter error ratio exceeds the threshold.
    """
    conn = BACKEND_CONNECTIONS[Backend[job_entry["backend_name"].upper()]]()
    job = conn.job(job_entry.id)

    # Check if we have any assets resulting from the job
    job_results = job.get_results()
    if len(job_results.get_metadata()["assets"]) == 0:
        raise Exception(f"Job {job_entry.id} has no assets!")

    # Check if SAR backscatter error ratio exceeds the threshold
    if "sar_backscatter_soft_errors" in job.describe()["usage"].keys():
        actual_orfeo_error_rate = job.describe()["usage"][
            "sar_backscatter_soft_errors"
        ]["value"]
        if actual_orfeo_error_rate > orfeo_error_threshold:
            raise Exception(
                f"Job {job_entry.id} had a ORFEO error rate of {actual_orfeo_error_rate}!"
            )

    pipeline_log.debug("Quality checks passed!")



# ----------------------------
# Dataframe loading and filtering
# ----------------------------
def load_dataframe(df_path: Path, extract_value: int = 0, check_existing: bool = False,
                   collection: Optional[str] = None, ref_id: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load a GeoDataFrame from a file, applying filters as necessary.
    """
    pipeline_log.info("Loading input dataframe from %s", df_path)
    df = _read_filtered_dataframe(df_path, extract_value)
    if check_existing:
        df = _filter_existing_samples(df, collection, ref_id)
    return df

def _read_filtered_dataframe(df_path: Path, extract_value: int) -> gpd.GeoDataFrame:
    """
    Read a GeoDataFrame from a file, applying filters as necessary.
    """
    filters = [("extract", ">=", extract_value)]
    pipeline_log.info("Reading dataframe with filters: %s", filters)
    if df_path.suffix == ".parquet":
        return gpd.read_parquet(df_path, filters=filters)
    return gpd.read_file(df_path, filters=filters)

def _filter_existing_samples(df: gpd.GeoDataFrame, collection: Optional[str], ref_id: Optional[str]) -> gpd.GeoDataFrame:
    """
    Filter the GeoDataFrame to only include samples that do not already exist in the specified collection.
    """
    if not collection or collection not in PATCH_COLLECTIONS or not ref_id:
        pipeline_log.warning("STAC check skipped: unsupported collection or missing ref_id")
        return df
    
    collection_id = PATCH_COLLECTIONS[collection]
    existing_ids = _fetch_existing_sample_ids(collection_id, ref_id)
    if not existing_ids:
        return df
    
    filtered_df = df[~df["sample_id"].isin(existing_ids)]
    pipeline_log.info("Filtered out %s existing samples for %s", len(df)-len(filtered_df), collection)
    return filtered_df

def _fetch_existing_sample_ids(collection_id: str, ref_id: str) -> list[str]:
    """
    Fetch the IDs of existing samples in the specified collection and reference ID.
    """
    client = pystac_client.Client.open(STAC_ROOT_URL)
    search = client.search(
        collections=[collection_id],
        filter={"op": "=", "args": [{"property": "properties.ref_id"}, ref_id]},
        filter_lang="cql2-json",
        fields={"exclude": ["assets", "links", "geometry", "bbox"]},
    )
    return [item.properties.get("sample_id") for item in search.items() if item.properties.get("sample_id")]

# ----------------------------
# Job dataframe preparation
# ----------------------------
def prepare_job_dataframe(samples_gdf: gpd.GeoDataFrame, collection: ExtractionCollection, max_locations: int) -> pd.DataFrame:
    """Prepare a job dataframe by splitting the samples and creating jobs for the specified collection."""
    pipeline_log.info("Preparing the job dataframe")
    split_dfs = _split_samples(samples_gdf, max_locations)
    job_df = _create_jobs_for_collection(collection, split_dfs)
    pipeline_log.info("Job dataframe created with %s jobs", len(job_df))
    return job_df

def _split_samples(samples_gdf: gpd.GeoDataFrame, max_locations: int) -> list[gpd.GeoDataFrame]:
    """
    Split the samples GeoDataFrame into smaller chunks based on the max_locations parameter.
    """
    samples_gdf["valid_time"] = pd.to_datetime(samples_gdf.valid_time)
    samples_gdf["year"] = samples_gdf.valid_time.dt.year
    split_by_time = [group.reset_index(drop=True) for _, group in samples_gdf.groupby("year")]
    split_dfs = []
    for df in split_by_time:
        split_dfs.extend(split_job_s2grid(df, max_points=max_locations))
    return split_dfs

def _create_jobs_for_collection(collection: ExtractionCollection, split_dfs: list[gpd.GeoDataFrame]) -> pd.DataFrame:
    """
    Create jobs for the specified collection from the split GeoDataFrames.
    """
    dataframe_creators: dict[ExtractionCollection, Callable] = {
        ExtractionCollection.PATCH_SENTINEL1: create_job_dataframe_patch_s1,
        ExtractionCollection.PATCH_SENTINEL2: create_job_dataframe_patch_s2,
        ExtractionCollection.PATCH_METEO: create_job_dataframe_patch_meteo,
        ExtractionCollection.PATCH_WORLDCEREAL: create_job_dataframe_patch_worldcereal,
        ExtractionCollection.POINT_WORLDCEREAL: create_job_dataframe_point_worldcereal,
    }
    if collection not in dataframe_creators:
        raise ValueError(f"Collection {collection} not supported")
    return dataframe_creators[collection](split_dfs)


# ----------------------------
# Post job action for patches
# ----------------------------
def _extract_geometry_information(extracted_gpd: gpd.GeoDataFrame, item_id: str) -> Optional[pd.Series]:
    """Extract geometry information for a given item ID."""
    sample_id_column_name = "sample_id" if "sample_id" in extracted_gpd.columns else "sampleID"
    
    geometry_information = extracted_gpd.loc[extracted_gpd[sample_id_column_name] == item_id]
    
    if len(geometry_information) == 0:
        pipeline_log.warning("No geometry found for the sample_id %s in the input geometry.", item_id)
        return None
    
    if len(geometry_information) > 1:
        pipeline_log.warning(
            "Duplicate geometries found for the sample_id %s in the input geometry, selecting the first one at index: %s.",
            item_id,
            geometry_information.index[0],
        )
    
    return geometry_information.iloc[0]

def _create_new_attributes(row: pd.Series, geometry_info: pd.Series, 
                          description: str, title: str, spatial_resolution: str) -> dict:
    """Create new attributes for the netCDF file."""
    attributes = {
        "start_date": row.start_date,
        "end_date": row.end_date,
        "valid_time": geometry_info.valid_time,
        "processing:version": version("openeo_gfmap"),
        "institution": "VITO - ESA WorldCereal",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": description,
        "title": title,
        "sample_id": geometry_info.sample_id if hasattr(geometry_info, 'sample_id') else geometry_info.sampleID,
        "ref_id": geometry_info.ref_id,
        "spatial_resolution": spatial_resolution,
        "s2_tile": row.s2_tile,
        "h3_l3_cell": geometry_info.h3_l3_cell,
        "_FillValue": 65535,  # No data value for uint16
    }
    
    if hasattr(row, 'orbit_state'):
        attributes["sat:orbit_state"] = row.orbit_state
    
    return attributes

def _validate_dataset_dimensions(ds: xr.Dataset, item_asset_path: Path, spatial_resolution: str) -> None:
    """Validate dataset dimensions match expected resolution."""
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

    pipeline_log.debug(
        "Dimension validation passed for %s: %dx%d matches expected %s resolution",
        item_asset_path,
        actual_x_size,
        actual_y_size,
        spatial_resolution,
    )

def _save_dataset_with_attributes(ds: xr.Dataset, item_asset_path: Path) -> None:
    """Save dataset with new attributes and proper permissions."""
    with NamedTemporaryFile(delete=False) as temp_file:
        ds.to_netcdf(temp_file.name)
        shutil.move(temp_file.name, item_asset_path)
        os.chmod(item_asset_path, 0o755)
        safe_chown_group(item_asset_path)

def _verify_file_integrity(item_asset_path: Path) -> None:
    """Verify that the output file is not corrupt."""
    try:
        ds = xr.open_dataset(item_asset_path)
        ds.close()
    except Exception as e:
        pipeline_log.error("The output file %s is corrupt. Error: %s", item_asset_path, e)
        raise

def _update_stac_item_metadata(item: pystac.Item, new_attributes: dict) -> None:
    """Update STAC item metadata with new attributes."""
    item.properties.update(new_attributes)
    item.properties["providers"] = [{"name": "openEO platform"}]
    extension = "https://stac-extensions.github.io/processing/v1.2.0/schema.json"
    item.stac_extensions.extend([extension])

def _upload_to_stac_api(job_items: List[pystac.Item], sensor: str) -> None:
    """Upload items to STAC API."""
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

def _process_single_item(item: pystac.Item, extracted_gpd: gpd.GeoDataFrame, row: pd.Series,
                        description: str, title: str, spatial_resolution: str,
                        s1_orbit_fix: bool, write_stac_api: bool) -> None:
    """Process a single STAC item."""
    item_id = item.id.replace(".nc", "").replace("openEO_", "")
    
    geometry_info = _extract_geometry_information(extracted_gpd, item_id)
    if geometry_info is None:
        return

    # Apply orbit state fix if needed
    if s1_orbit_fix:
        item.id = item.id.replace(".nc", f"_{row.orbit_state}.nc")

    item_asset_path = Path(list(item.assets.values())[0].href)
    
    # Create and apply new attributes
    new_attributes = _create_new_attributes(row, geometry_info, description, title, spatial_resolution)
    ds = xr.open_dataset(item_asset_path)
    
    # Validate dimensions
    _validate_dataset_dimensions(ds, item_asset_path, spatial_resolution)
    
    # Apply attributes and save
    ds = ds.assign_attrs(new_attributes)
    _save_dataset_with_attributes(ds, item_asset_path)
    
    pipeline_log.info(f"Final output file created: {item_asset_path}")
    
    # Verify file integrity
    _verify_file_integrity(item_asset_path)
    
    # Update STAC metadata if needed
    if write_stac_api:
        _update_stac_item_metadata(item, new_attributes)

def post_job_action_patch(
    job_items: List[pystac.Item],
    row: pd.Series,
    extract_value: int,
    description: str,
    title: str,
    spatial_resolution: str,
    s1_orbit_fix: bool = False,
    write_stac_api: bool = False,
    sensor: str = "Sentinel1",
) -> list:
    """Process job items after extraction to add metadata and validate results."""
    
    # Perform quality checks
    extraction_job_quality_check(row)
    
    # Load and validate geometries
    base_gpd = gpd.GeoDataFrame.from_features(json.loads(row.geometry)).set_crs(epsg=4326)
    
    if len(base_gpd[base_gpd.extract == extract_value]) != len(job_items):
        pipeline_log.warning(
            "Different amount of geometries in the job output items and the input geometry. "
            "Job items #: %s, Input geometries #: %s",
            len(job_items),
            len(base_gpd[base_gpd.extract == extract_value]),
        )
    
    extracted_gpd = base_gpd[base_gpd.extract >= extract_value].reset_index(drop=True)
    
    # Process each item
    for item in job_items:
        _process_single_item(
            item, extracted_gpd, row, description, title, 
            spatial_resolution, s1_orbit_fix, write_stac_api
        )
    
    # Upload to STAC API if requested
    if write_stac_api:
        _upload_to_stac_api(job_items, sensor)
    
    return job_items


def generate_output_path_patch(
    root_folder: Path,
    row: pd.Series,
    asset_id: str,
) -> Path:
    """Generate the output path for extracted patch data."""
    
    # Extract sample ID from asset ID
    sample_id = asset_id.replace(".nc", "").replace("openEO_", "")
    
    # Handle orbit state if present
    orbit_state = f"_{row.orbit_state}" if hasattr(row, 'orbit_state') and row.orbit_state else ""
    
    # Parse S2 tile information
    s2_tile_id = row.s2_tile
    utm_zone = str(s2_tile_id[0:2])
    
    # Build directory structure
    subfolder = root_folder / utm_zone / s2_tile_id / sample_id
    
    # Generate filename
    filename = f"{row.out_prefix}{orbit_state}_{sample_id}_{row.start_date}_{row.end_date}{row.out_extension}"
    
    return subfolder / filename


# ----------------------------
# Extraction function dispatch
# ----------------------------
def _get_dispatch_fn(mapping: dict, collection: ExtractionCollection, collection_name: str) -> Callable:
    """Get a function from a mapping based on the collection, raising an error if not found."""
    if collection not in mapping:
        raise ValueError(f"{collection_name} collection {collection} not supported")
    return mapping[collection]

def setup_datacube_creation_fn(collection: ExtractionCollection, job_options: Optional[dict] = None) -> Callable:
    """
    Set up the datacube creation function for the specified collection.
    """
    datacube_creation = {
        ExtractionCollection.PATCH_SENTINEL1: partial(create_job_patch_s1, job_options=job_options),
        ExtractionCollection.PATCH_SENTINEL2: partial(create_job_patch_s2, job_options=job_options),
        ExtractionCollection.PATCH_METEO: partial(create_job_patch_meteo, job_options=job_options),
        ExtractionCollection.PATCH_WORLDCEREAL: partial(create_job_patch_worldcereal, job_options=job_options),
        ExtractionCollection.POINT_WORLDCEREAL: partial(create_job_point_worldcereal, job_options=job_options),
    }
    return _get_dispatch_fn(datacube_creation, collection, "Datacube creation")

def setup_output_path_fn(collection: ExtractionCollection) -> Callable:
    """
    Set up the output path generation function for the specified collection.
    """
    path_fns = {
        ExtractionCollection.PATCH_SENTINEL1: partial(generate_output_path_patch),
        ExtractionCollection.PATCH_SENTINEL2: partial(generate_output_path_patch),
        ExtractionCollection.PATCH_METEO: partial(generate_output_path_patch),
        ExtractionCollection.PATCH_WORLDCEREAL: partial(generate_output_path_patch_worldcereal),
        ExtractionCollection.POINT_WORLDCEREAL: partial(generate_output_path_point_worldcereal),
    }
    return _get_dispatch_fn(path_fns, collection, "Output path generation")

def setup_post_job_fn(collection: ExtractionCollection, extract_value: int, write_stac_api: bool) -> Callable:
    """
    Setup post-job function for collection.
    """
    post_job_actions = {
        ExtractionCollection.PATCH_SENTINEL1: partial(
            post_job_action_patch,
            extract_value=extract_value,
            description="Sentinel-1 GRD backscatter observations, processed with Orfeo toolbox.",
            title="Sentinel-1 GRD",
            spatial_resolution="20m",
            s1_orbit_fix=True,
            sensor="Sentinel1",
            write_stac_api=write_stac_api,
        ),
        ExtractionCollection.PATCH_SENTINEL2: partial(
            post_job_action_patch,
            extract_value=extract_value,
            description="Sentinel-2 L2A surface reflectance observations.",
            title="Sentinel-2 L2A",
            spatial_resolution="10m",
            sensor="Sentinel2",
            write_stac_api=write_stac_api,
        ),
        ExtractionCollection.PATCH_METEO: partial(
            post_job_action_patch,
            extract_value=extract_value,
            description="Meteo observations",
            title="Meteo observations",
            spatial_resolution="1deg",
        ),
        ExtractionCollection.PATCH_WORLDCEREAL: partial(
            post_job_action_patch_worldcereal,
            extract_value=extract_value,
        ),
        ExtractionCollection.POINT_WORLDCEREAL: partial(
            post_job_action_point_worldcereal,
        ),
    }
    return _get_dispatch_fn(post_job_actions, collection, "Post-job action")

# ----------------------------
# Extraction job setup & orchestration
# ----------------------------

def run_extractions(
    collection: ExtractionCollection,
    output_folder: Path,
    samples_df_path: Path,
    ref_id: str,
    max_locations_per_job: int = 500,
    job_options: Optional[Dict[str, Union[str, int]]] = None,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    extract_value: int = 1,
    backend=Backend.CDSE,
    write_stac_api: bool = False,
    check_existing_extractions: bool = False,
) -> None:
    """Main function responsible for launching point and patch extractions.

    Parameters
    ----------
    collection : ExtractionCollection
        The collection to extract. Most popular: PATCH_WORLDCEREAL, POINT_WORLDCEREAL
    output_folder : Path
        The folder where to store the extracted data
    samples_df_path : Path
        Path to the input dataframe containing the geometries
        for which extractions need to be done
    ref_id : str
        Official ref_id of the source dataset
    max_locations_per_job : int, optional
        The maximum number of locations to extract per job, by default 500
    job_options : Optional[Dict[str, Union[str, int]]], optional
        Custom job options to set for the extraction, by default None (default options)
        Options that can be set explicitly include:
            - memory : str
                Memory to allocate for the executor, e.g. "1800m"
            - python_memory : str
                Memory to allocate for the python processes as well as OrfeoToolbox in the executors, e.g. "1900m"
            - max_executors : int
                Number of executors to run, e.g. 22
    parallel_jobs : int, optional
        The maximum number of parallel jobs to run at the same time, by default 10
    restart_failed : bool, optional
        Restart the jobs that previously failed, by default False
    extract_value : int, optional
        All samples with an "extract" value equal or larger than this one, will be extracted, by default 1
    backend : openeo_gfmap.Backend, optional
        Cloud backend where to run the extractions, by default Backend.CDSE
    write_stac_api : bool, optional
        Save metadata of extractions to STAC API (requires authentication), by default False
    check_existing_extractions : bool, optional
        Check if the samples already exist in the STAC API and filter them out,
        by default False
    """
    pipeline_log.info("Starting the extractions workflow...")

    # --- Prepare the extraction jobs ---
    ensure_dir(output_folder)
    tracking_df_path = output_folder / "job_tracking.csv"

    # Load or create job dataframe
    job_df = _load_or_create_job_dataframe(
        tracking_df_path,
        samples_df_path,
        collection,
        ref_id,
        max_locations_per_job,
        extract_value,
        backend,
        restart_failed,
        check_existing_extractions,
    )

    # Setup extraction functions
    datacube_fn = setup_datacube_creation_fn(collection, job_options)
    path_fn = setup_output_path_fn(collection)
    post_job_fn = setup_post_job_fn(collection, extract_value, write_stac_api)

    # Initialize job manager with STAC support if requested
    job_manager = _initialize_job_manager(
        output_folder,
        path_fn,
        post_job_fn,
        backend,
        parallel_jobs,
        stac_enabled=write_stac_api,
        collection_id=f"{ref_id}_extractions",
        collection_description=f"Extractions for {collection.name} with ref_id {ref_id}",
    )

    # --- Run the extraction jobs ---
    _run_extraction_jobs(
        job_manager=job_manager,
        job_df=job_df,
        datacube_fn=datacube_fn,
        tracking_df_path=tracking_df_path,
    )

    # --- Write STAC collection if enabled ---
    if write_stac_api:
        job_manager.write_stac()
        pipeline_log.info("STAC collection saved successfully.")


def _load_or_create_job_dataframe(
    tracking_df_path: Path,
    samples_df_path: Path,
    collection: ExtractionCollection,
    ref_id: str,
    max_locations_per_job: int,
    extract_value: int,
    backend: Backend,
    restart_failed: bool,
    check_existing_extractions: bool,
) -> pd.DataFrame:
    """Load an existing job tracking dataframe or create a new one."""
    if tracking_df_path.exists():
        pipeline_log.info("Loading existing job tracking dataframe.")
        job_df = pd.read_csv(tracking_df_path)
        # Reset canceled and failed jobs if needed
        job_df.loc[job_df.status.isin(["canceled"]), "status"] = "not_started"
        if restart_failed:
            job_df.loc[
                job_df.status.isin(["error", "postprocessing-error", "start_failed"]),
                "status",
            ] = "not_started"
        job_df.to_csv(tracking_df_path, index=False)
        pipeline_log.info("Job status histogram: %s", check_job_status(tracking_df_path.parent))
        return job_df

    # Create new job dataframe
    samples_gdf = load_dataframe(
        samples_df_path,
        extract_value,
        check_existing=check_existing_extractions,
        collection=collection,
        ref_id=ref_id,
    )
    samples_gdf["ref_id"] = ref_id
    pipeline_log.info("Creating new job tracking dataframe.")
    job_df = prepare_job_dataframe(samples_gdf, collection, max_locations_per_job, backend)
    job_df.to_csv(tracking_df_path, index=False)
    return job_df


def _initialize_job_manager(
    output_folder: Path,
    path_fn: Callable,
    post_job_fn: Callable,
    backend: Backend,
    parallel_jobs: int = 2,
    stac_enabled: bool = False,
    collection_id: Optional[str] = None,
    collection_description: str = "",
) -> ExtractionJobManager:
    """Create and configure the extraction job manager with optional STAC support."""

    job_manager = ExtractionJobManager(
        output_dir=output_folder,
        output_path_generator=path_fn,
        post_job_action=post_job_fn,
        poll_sleep=60,
        stac_enabled=stac_enabled,
        collection_id=collection_id,
        collection_description=collection_description,
    )

    job_manager.add_backend(
        backend.value,
        BACKEND_CONNECTIONS[backend],
        parallel_jobs=parallel_jobs,
    )

    return job_manager


def _run_extraction_jobs(
    job_manager: ExtractionJobManager,
    job_df: pd.DataFrame,
    datacube_fn: Callable,
    tracking_df_path: Path,
) -> None:
    """Execute extraction jobs using the manager."""
    pipeline_log.info("Running the extraction jobs.")
    job_manager.run_jobs(job_df, datacube_fn, tracking_df_path)
    pipeline_log.info("Extraction jobs completed.")


def _merge_extraction_jobs(output_folder: Path, ref_id: str) -> None:
    """Merge all extraction results into a single partitioned GeoParquet file."""
    pipeline_log.info("Merging extraction jobs into final output.")
    merge_output_files_point_worldcereal(output_folder, ref_id)
    pipeline_log.info("Merging completed successfully.")

