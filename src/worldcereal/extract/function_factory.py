

from functools import partial
import pandas as pd
import pystac
import xarray as xr
import json
import geopandas as gpd
from typing import List
from pathlib import Path
from worldcereal.extract.utils import pipeline_log
from typing import Callable, Optional
from worldcereal.stac.constants import ExtractionCollection
from worldcereal.extract.quality_checks import extraction_job_quality_check, validate_dataset_dimensions, verify_file_integrity
from worldcereal.stac.utils import update_stac_item_metadata
from worldcereal.extract.utils import extract_geometry_information
from worldcereal.extract.attribute_processing import create_new_attributes, save_dataset_with_attributes

from worldcereal.extract.patch_meteo import (
    create_job_patch_meteo,
)
from worldcereal.extract.patch_s1 import (
    create_job_patch_s1,
)
from worldcereal.extract.patch_s2 import (
    create_job_patch_s2,
)
from worldcereal.extract.patch_worldcereal import (
    create_job_patch_worldcereal,
    post_job_action_patch_worldcereal,
    generate_output_path_patch_worldcereal,
)
from worldcereal.extract.point_worldcereal import (
    create_job_point_worldcereal,
    post_job_action_point_worldcereal,
    generate_output_path_point_worldcereal,
    
)

# Helper function to get function from mapping with error handling
def _get_fn(mapping: dict, collection: ExtractionCollection, collection_name: str) -> Callable:
    """Get a function from a mapping based on the collection, raising an error if not found."""
    if collection not in mapping:
        raise ValueError(f"{collection_name} collection {collection} not supported")
    return mapping[collection]

#TODO consider moving
# Function for path generation for patch data
def generate_output_path_patch(
    root_folder: Path,
    row: pd.Series,
    asset_id: str,
) -> Path:
    """Generate the output path for extracted patch data."""
    pipeline_log.info(f"Generating output path for asset_id: {asset_id}")
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
    return _get_fn(path_fns, collection, "Output path generation")


# Function for path generation for worldcereal patch datacubes
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
    return _get_fn(datacube_creation, collection, "Datacube creation")


# FUnction for post-job action setup
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
    return _get_fn(post_job_actions, collection, "Post-job action")

def post_job_action_patch(
    job_items: List[pystac.Item],
    row: pd.Series,
    extract_value: int,
    description: str,
    title: str,
    spatial_resolution: str,
    s1_orbit_fix: bool = False,
    write_stac_api: bool = False,
    sensor: str = "Sentinel2", #TODO this is terrible and needs to be removed
) -> list:
    """Process job items after extraction to add metadata and validate results."""
    pipeline_log.info(f"Post-processing {len(job_items)} job items")
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
    
    return job_items

def _process_single_item(item: pystac.Item, extracted_gpd: gpd.GeoDataFrame, row: pd.Series,
                        description: str, title: str, spatial_resolution: str,
                        s1_orbit_fix: bool, write_stac_api: bool) -> None:
    """Process a single STAC item."""
    pipeline_log.info(f"Processing STAC item: {item}")
    item_id = item.id.replace(".nc", "").replace("openEO_", "")
    
    geometry_info = extract_geometry_information(extracted_gpd, item_id)
    if geometry_info is None:
        return

    # Apply orbit state fix if needed
    if s1_orbit_fix:
        item.id = item.id.replace(".nc", f"_{row.orbit_state}.nc")

    item_asset_path = Path(list(item.assets.values())[0].href)
    pipeline_log.info(f"Item asset path: {item_asset_path}")
    
    # Create and apply new attributes
    new_attributes = create_new_attributes(row, geometry_info, description, title, spatial_resolution)
    with xr.open_dataset(item_asset_path) as src:
        ds = src.load() 
    
    # Validate dimensions
    validate_dataset_dimensions(ds, item_asset_path, spatial_resolution)
    
    # Apply attributes and save
    ds = ds.assign_attrs(new_attributes)
    save_dataset_with_attributes(ds, item_asset_path)
    
    pipeline_log.info(f"Final output file created: {item_asset_path}")
    
    # Verify file integrity
    verify_file_integrity(item_asset_path)
    
    # Update STAC metadata if needed
    if write_stac_api:
        update_stac_item_metadata(item, new_attributes)

