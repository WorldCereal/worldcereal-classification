import logging
import os
from io import BytesIO

import geojson
import geopandas as gpd
import pandas as pd
import requests
from openeo_gfmap.manager.job_splitters import load_s2_grid
from shapely import Point

from worldcereal.utils.upload import OpenEOArtifactHelper

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


S2_GRID = load_s2_grid()


def buffer_geometry(
    geometries: geojson.FeatureCollection, distance_m: int = 320
) -> gpd.GeoDataFrame:
    """For each geometry of the collection, perform a square buffer of 320
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


def upload_geoparquet_s3(backend, gdf: gpd.GeoDataFrame, name: str, collection: str = "") -> str:
    """Upload the given GeoDataFrame to S3 and return a presigned URL."""
    # Write GeoDataFrame to an in-memory buffer
    buffer = BytesIO()
    gdf.to_parquet(buffer)
    buffer.seek(0)

    targetpath = f"openeogfmap_dataframe_{collection}_{name}.parquet"

    artifact_helper = OpenEOArtifactHelper.from_openeo_backend(backend)
    normal_s3_uri = artifact_helper.upload_file(targetpath, buffer)
    presigned_uri = artifact_helper.get_presigned_url(normal_s3_uri)

    return presigned_uri


def upload_geoparquet_artifactory(gdf: gpd.GeoDataFrame, name: str, collection: str = "") -> str:
    """Upload the given GeoDataFrame to Artifactory and return the URL."""
    # Write GeoDataFrame to an in-memory buffer
    buffer = BytesIO()
    gdf.to_parquet(buffer)
    buffer.seek(0)

    artifactory_username = os.getenv("ARTIFACTORY_USERNAME")
    artifactory_password = os.getenv("ARTIFACTORY_PASSWORD")

    if not artifactory_username or not artifactory_password:
        raise ValueError(
            "Artifactory credentials not found. Please set ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD."
        )

    headers = {"Content-Type": "application/octet-stream"}
    upload_url = (
        f"https://artifactory.vgt.vito.be/artifactory/auxdata-public/"
        f"gfmap-temp/openeogfmap_dataframe_{collection}_{name}.parquet"
    )

    response = requests.put(
        upload_url,
        headers=headers,
        data=buffer,
        auth=(artifactory_username, artifactory_password),
        timeout=180,
    )
    response.raise_for_status()

    verify_response = requests.get(upload_url, auth=(artifactory_username, artifactory_password), timeout=60)
    if verify_response.status_code != 200 or len(verify_response.content) == 0:
        raise RuntimeError(f"Upload may have failed: file not found or empty at {upload_url}")

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

