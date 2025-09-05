from pathlib import Path
from typing import Optional, Callable
import geopandas as gpd
import pandas as pd
import pystac_client

from worldcereal.extract.utils import pipeline_log
from openeo_gfmap.manager.job_splitters import split_job_s2grid
from openeo_gfmap import Backend

from worldcereal.stac.constants import ExtractionCollection

from worldcereal.extract.patch_meteo import (
    create_job_dataframe_patch_meteo,
)
from worldcereal.extract.patch_s2 import (
    create_job_dataframe_patch_s2,
)
from worldcereal.extract.point_worldcereal import (
    create_job_dataframe_point_worldcereal,
)
from worldcereal.extract.patch_s1 import (  # isort: skip
    create_job_dataframe_patch_s1,
)
from worldcereal.extract.patch_worldcereal import (  # isort: skip
    create_job_dataframe_patch_worldcereal)

from worldcereal.extract.utils import pipeline_log

PATCH_COLLECTIONS = {
    "PATCH_SENTINEL1": "hv_test_worldcereal_sentinel_1_patch_extractions",
    "PATCH_SENTINEL2": "hv_test_worldcereal_sentinel_2_patch_extractions",
}
STAC_ROOT_URL = "https://stac.openeo.vito.be/"


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
    if df_path.suffix.lower() in [".parquet", ".geoparquet"]:
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


def prepare_job_dataframe(samples_gdf: gpd.GeoDataFrame, collection: ExtractionCollection, max_locations: int, backend: Backend) -> pd.DataFrame:
    """Prepare a job dataframe by splitting the samples and creating jobs for the specified collection."""
    pipeline_log.info("Preparing the job dataframe")
    split_dfs = _split_samples(samples_gdf, max_locations)
    job_df = _create_jobs_for_collection(collection, split_dfs, backend)
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

def _create_jobs_for_collection(collection: ExtractionCollection, split_dfs: list[gpd.GeoDataFrame], backend: Backend) -> pd.DataFrame:
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
    return dataframe_creators[collection](backend, split_dfs)

def load_or_create_job_dataframe(
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
