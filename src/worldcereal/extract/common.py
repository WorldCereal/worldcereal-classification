"""Common functions used by extraction scripts."""

import json
import logging
import os
import shutil
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import geojson
import geopandas as gpd
import pandas as pd
import pystac
import requests
import xarray as xr
from shapely import Point

from worldcereal.stac.stac_api_interaction import (
    StacApiInteraction,
    VitoStacApiAuthentication,
)

# Logger used for the pipeline
pipeline_log = logging.getLogger("extraction_pipeline")

pipeline_log.setLevel(level=logging.INFO)

stream_handler = logging.StreamHandler()
pipeline_log.addHandler(stream_handler)

formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s:  %(message)s")
stream_handler.setFormatter(formatter)


# Exclude the other loggers from other libraries
class ManagerLoggerFilter(logging.Filter):
    """Filter to only accept the OpenEO-GFMAP manager logs."""

    def filter(self, record):
        return record.name in [pipeline_log.name]


stream_handler.addFilter(ManagerLoggerFilter())


def post_job_action_patch(
    job_items: List[pystac.Item],
    row: pd.Series,
    extract_value: int,
    description: str,
    title: str,
    spatial_resolution: str,
    s1_orbit_fix: bool = False,  # To rename the samples from the S1 orbit
    write_stac_api: bool = False,
    sensor: str = "Sentinel1",
) -> list:
    """From the job items, extract the metadata and save it in a netcdf file."""
    base_gpd = gpd.GeoDataFrame.from_features(json.loads(row.geometry)).set_crs(
        epsg=4326
    )
    if len(base_gpd[base_gpd.extract == extract_value]) != len(job_items):
        pipeline_log.warning(
            "Different amount of geometries in the job output items and the "
            "input geometry. Job items #: %s, Input geometries #: %s",
            len(job_items),
            len(base_gpd[base_gpd.extract == extract_value]),
        )

    extracted_gpd = base_gpd[base_gpd.extract == extract_value].reset_index(drop=True)
    # In this case we want to burn the metadata in a new file in the same folder as the S2 product
    for item in job_items:
        item_id = item.id.replace(".nc", "").replace("openEO_", "")
        sample_id_column_name = (
            "sample_id" if "sample_id" in extracted_gpd.columns else "sampleID"
        )

        geometry_information = extracted_gpd.loc[
            extracted_gpd[sample_id_column_name] == item_id
        ]

        if len(geometry_information) == 0:
            pipeline_log.warning(
                "No geometry found for the sample_id %s in the input geometry.",
                item_id,
            )
            continue

        if len(geometry_information) > 1:
            pipeline_log.warning(
                "Duplicate geomtries found for the sample_id %s in the input geometry, selecting the first one at index: %s.",
                item_id,
                geometry_information.index[0],
            )

        geometry_information = geometry_information.iloc[0]

        sample_id = geometry_information[sample_id_column_name]
        ref_id = geometry_information.ref_id
        valid_time = geometry_information.valid_time
        h3_l3_cell = geometry_information.h3_l3_cell
        s2_tile = row.s2_tile

        item_asset_path = Path(list(item.assets.values())[0].href)

        # Add some metadata to the result_df netcdf file
        new_attributes = {
            "start_date": row.start_date,
            "end_date": row.end_date,
            "valid_time": valid_time,
            "processing:version": version("openeo_gfmap"),
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "title": title,
            "sample_id": sample_id,
            "ref_id": ref_id,
            "spatial_resolution": spatial_resolution,
            "s2_tile": s2_tile,
            "h3_l3_cell": h3_l3_cell,
            "_FillValue": 65535,  # No data value for uint16
        }

        if s1_orbit_fix:
            new_attributes["sat:orbit_state"] = row.orbit_state
            item.id = item.id.replace(".nc", f"_{row.orbit_state}.nc")

        # Saves the new attributes in the netcdf file
        ds = xr.open_dataset(item_asset_path)

        ds = ds.assign_attrs(new_attributes)

        with NamedTemporaryFile(delete=False) as temp_file:
            ds.to_netcdf(temp_file.name)
            shutil.move(temp_file.name, item_asset_path)

    if write_stac_api:
        username = os.getenv("STAC_API_USERNAME")
        password = os.getenv("STAC_API_PASSWORD")

        stac_api_interaction = StacApiInteraction(
            sensor=sensor,
            base_url="https://stac.openeo.vito.be",
            auth=VitoStacApiAuthentication(username=username, password=password),
        )

        pipeline_log.info("Writing the STAC API metadata")
        stac_api_interaction.upload_items_bulk(job_items)
        pipeline_log.info("STAC API metadata written")

    return job_items


def generate_output_path_patch(
    root_folder: Path,
    job_index: int,
    row: pd.Series,
    asset_id: str,
    s2_grid: gpd.GeoDataFrame,
):
    """Generate the output path for the extracted data, from a base path and
    the row information.
    """
    # First extract the sample ID from the asset ID
    sample_id = asset_id.replace(".nc", "").replace("openEO_", "")

    # Find which index in the FeatureCollection corresponds to the sample_id
    features = geojson.loads(row.geometry)["features"]
    sample_id_to_index = {
        feature.properties.get("sample_id", None): index
        for index, feature in enumerate(features)
    }
    geometry_index = sample_id_to_index.get(sample_id, None)

    ref_id = features[geometry_index].properties["ref_id"]

    if "orbit_state" in row:
        orbit_state = f"_{row.orbit_state}"
    else:
        orbit_state = ""

    s2_tile_id = row.s2_tile
    utm_zone = str(s2_tile_id[0:2])
    epsg = s2_grid[s2_grid.tile == s2_tile_id].iloc[0].epsg

    subfolder = root_folder / ref_id / utm_zone / s2_tile_id / sample_id

    return (
        subfolder
        / f"{row.out_prefix}{orbit_state}_{sample_id}_{epsg}_{row.start_date}_{row.end_date}{row.out_extension}"
    )


def buffer_geometry(
    geometries: geojson.FeatureCollection, distance_m: int = 320
) -> gpd.GeoDataFrame:
    """For each geometry of the colleciton, perform a square buffer of 320
    meters on the centroid and return the GeoDataFrame. Before buffering,
    the centroid is clipped to the closest 20m multiplier in order to stay
    aligned with the Sentinel-1 pixel grid.
    """
    gdf = gpd.GeoDataFrame.from_features(geometries).set_crs(epsg=4326)
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)

    # Perform the buffering operation
    gdf["geometry"] = gdf.centroid.apply(
        lambda point: Point(round(point.x / 20.0) * 20.0, round(point.y / 20.0) * 20.0)
    ).buffer(
        distance=distance_m, cap_style=3
    )  # Square buffer

    return gdf


def filter_extract_true(
    geometries: geojson.FeatureCollection, extract_value: int = 1
) -> gpd.GeoDataFrame:
    """Remove all the geometries from the Feature Collection that have the property field `extract` set to `False`"""
    return geojson.FeatureCollection(
        [
            f
            for f in geometries.features
            if f.properties.get("extract", 0) == extract_value
        ]
    )


def upload_geoparquet_artifactory(
    gdf: gpd.GeoDataFrame, name: str, collection: str = ""
) -> str:
    """Upload the given GeoDataFrame to artifactory and return the URL of the
    uploaded file. Necessary as a workaround for Polygon sampling in OpenEO
    using custom CRS.
    """
    # Save the dataframe as geoparquet to upload it to artifactory
    temporary_file = NamedTemporaryFile()
    gdf.to_parquet(temporary_file.name)

    artifactory_username = os.getenv("ARTIFACTORY_USERNAME")
    artifactory_password = os.getenv("ARTIFACTORY_PASSWORD")

    if not artifactory_username or not artifactory_password:
        raise ValueError(
            "Artifactory credentials not found. Please set ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD."
        )

    headers = {"Content-Type": "application/octet-stream"}

    upload_url = f"https://artifactory.vgt.vito.be/artifactory/auxdata-public/gfmap-temp/openeogfmap_dataframe_{collection}{name}.parquet"

    with open(temporary_file.name, "rb") as f:
        response = requests.put(
            upload_url,
            headers=headers,
            data=f,
            auth=(artifactory_username, artifactory_password),
            timeout=180,
        )

    response.raise_for_status()

    return upload_url


def get_job_nb_polygons(row: pd.Series) -> int:
    """Get the number of polygons in the geometry."""
    return len(
        list(
            filter(
                lambda feat: feat.properties.get("extract"),
                geojson.loads(row.geometry)["features"],
            )
        )
    )
