"""Data processing utilities for extraction results."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime
from typing import Optional
import pandas as pd
import geopandas as gpd
import xarray as xr
from importlib.metadata import version

from worldcereal.extract.utils import pipeline_log
from worldcereal.utils.file_utils import set_file_permissions



def create_new_attributes(row: pd.Series, geometry_info: pd.Series, 
                         description: str, title: str, spatial_resolution: str) -> dict:
    """Create new attributes for the netCDF file."""
    pipeline_log.info(f"Creating new attributes for {title}")
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

def save_dataset_with_attributes(ds: xr.Dataset, item_asset_path: Path) -> None:
    """Save dataset with proper file handling and permissions."""
    item_asset_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary file
    with NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        pipeline_log.info(f"Writing dataset to temporary file: {tmp_path}")
        ds.to_netcdf(tmp_path)  # write safely to temp file

    # Move temp file to final destination
    pipeline_log.info(f"Moving temporary file to final destination: {item_asset_path}")
    tmp_path.replace(item_asset_path)
    set_file_permissions(item_asset_path)