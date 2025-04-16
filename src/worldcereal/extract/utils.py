"""Common utilities used by extraction scripts."""

import logging
from tempfile import NamedTemporaryFile
from typing import Optional

import geojson
import geopandas as gpd
import openeo
import pandas as pd
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


def upload_geoparquet_s3(
    conn: openeo.Connection, gdf: gpd.GeoDataFrame, name: str, collection: str = ""
) -> str:
    """Upload the given GeoDataFrame to s3 and return the URL of the
    uploaded file. Necessary as a workaround for Polygon sampling in OpenEO
    using custom CRS.
    """
    # Save the dataframe as geoparquet to upload it to artifactory
    temporary_file = NamedTemporaryFile()
    gdf.to_parquet(temporary_file.name)

    targetpath = f"openeogfmap_dataframe_{collection}_{name}.parquet"

    artifact_helper = OpenEOArtifactHelper.from_openeo_connection(conn)
    normal_s3_uri = artifact_helper.upload_file(targetpath, temporary_file.name)
    presigned_uri = artifact_helper.get_presigned_url(normal_s3_uri)

    return presigned_uri


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
