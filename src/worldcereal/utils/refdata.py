import importlib.resources
import json
from typing import Dict

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import Polygon

from worldcereal.data import croptype_mappings


def get_class_mappings() -> Dict:
    """Method to get the WorldCereal class mappings for downstream task.

    Returns
    -------
    Dict
        the resulting dictionary with the class mappings
    """
    with importlib.resources.open_text(croptype_mappings, "croptype_classes.json") as f:  # type: ignore
        CLASS_MAPPINGS = json.load(f)

    return CLASS_MAPPINGS


def query_public_extractions(
    bbox_poly: Polygon, buffer: int = 250000, filter_cropland: bool = True
) -> pd.DataFrame:
    """Function that queries the WorldCereal global database of pre-extracted input
    data for a given area.

    Parameters
    ----------
    bbox_poly : Polygon
        bounding box of the area to make the query for. Expected to be in WGS84 coordinates.
    buffer : int, optional
        buffer (in meters) to apply to the requested area, by default 250000
    filter_cropland : bool, optional
        limit the query to samples on cropland only, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extractions matching the request.
    """

    logger.info(f"Applying a buffer of {buffer/1000} km to the selected area ...")

    bbox_poly = (
        gpd.GeoSeries(bbox_poly, crs="EPSG:4326")
        .to_crs(epsg=3785)
        .buffer(buffer, cap_style="square", join_style="mitre")
        .to_crs(epsg=4326)[0]
    )

    xmin, ymin, xmax, ymax = bbox_poly.bounds
    twisted_bbox_poly = Polygon(
        [(ymin, xmin), (ymin, xmax), (ymax, xmax), (ymax, xmin)]
    )
    h3_cells_lst = []  # type: ignore
    res = 5
    while len(h3_cells_lst) == 0:
        h3_cells_lst = list(h3.polyfill(twisted_bbox_poly.__geo_interface__, res))
        res += 1
    if res > 5:
        h3_cells_lst = tuple(np.unique([h3.h3_to_parent(xx, 5) for xx in h3_cells_lst]))  # type: ignore

    db = duckdb.connect()
    db.sql("INSTALL spatial")
    db.load_extension("spatial")

    parquet_path = "s3://geoparquet/worldcereal_extractions_phase1/*/*.parquet"

    # only querying the croptype data here
    logger.info(
        "Querying WorldCereal global extractions database (this can take a while) ..."
    )
    if filter_cropland:
        query = f"""
        set s3_endpoint='s3.waw3-1.cloudferro.com';
        set enable_progress_bar=false;
        select *
        from read_parquet('{parquet_path}', hive_partitioning = 1) original_data
        where original_data.h3_l5_cell in {h3_cells_lst}
        and original_data.LANDCOVER_LABEL = 11
        and original_data.CROPTYPE_LABEL not in (0, 991, 7900, 9900, 9998, 1910, 1900, 1920, 1000, 11, 9910, 6212, 7920, 9520, 3400, 3900, 4390, 4000, 4300)
        """
    else:
        query = f"""
            set s3_endpoint='s3.waw3-1.cloudferro.com';
            set enable_progress_bar=false;
            select *
            from read_parquet('{parquet_path}', hive_partitioning = 1) original_data
            where original_data.h3_l5_cell in {h3_cells_lst}
        """

    public_df_raw = db.sql(query).df()

    # Process the parquet into the format we need for training
    processed_public_df = process_parquet(public_df_raw)

    return processed_public_df


def process_parquet(public_df_raw: pd.DataFrame) -> pd.DataFrame:
    """Method to transform the raw parquet data into a format that can be used for
    training. Includes pivoting of the dataframe and mapping of the crop types.

    Parameters
    ----------
    public_df_raw : pd.DataFrame
        Input raw flattened dataframe from the global database.

    Returns
    -------
    pd.DataFrame
        processed dataframe with the necessary columns for training.
    """
    from presto.utils import process_parquet as process_parquet_for_presto

    logger.info("Processing selected samples ...")
    public_df = process_parquet_for_presto(public_df_raw)
    public_df = map_croptypes(public_df)
    logger.info(
        f"Extracted and processed {public_df.shape[0]} samples from global database."
    )

    return public_df


def map_croptypes(
    df: pd.DataFrame, downstream_classes: str = "CROPTYPE9"
) -> pd.DataFrame:
    """Helper function to map croptypes to a specific legend.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe containing the croptype labels
    downstream_classes : str, optional
        column name with the labels, by default "CROPTYPE9"

    Returns
    -------
    pd.DataFrame
        mapped crop types
    """
    with importlib.resources.open_text(croptype_mappings, "wc2eurocrops_map.csv") as f:  # type: ignore
        wc2ewoc_map = pd.read_csv(f)

    wc2ewoc_map["ewoc_code"] = wc2ewoc_map["ewoc_code"].str.replace("-", "").astype(int)

    with importlib.resources.open_text(croptype_mappings, "eurocrops_map_wcr_edition.csv") as f:  # type: ignore
        ewoc_map = pd.read_csv(f)

    ewoc_map = ewoc_map[ewoc_map["ewoc_code"].notna()]
    ewoc_map["ewoc_code"] = ewoc_map["ewoc_code"].str.replace("-", "").astype(int)
    ewoc_map = ewoc_map.apply(lambda x: x[: x.last_valid_index()].ffill(), axis=1)
    ewoc_map.set_index("ewoc_code", inplace=True)

    df["CROPTYPE_LABEL"] = df["CROPTYPE_LABEL"].replace(0, np.nan)
    df["CROPTYPE_LABEL"] = df["CROPTYPE_LABEL"].fillna(df["LANDCOVER_LABEL"])

    df["ewoc_code"] = df["CROPTYPE_LABEL"].map(
        wc2ewoc_map.set_index("croptype")["ewoc_code"]
    )
    df["landcover_name"] = df["ewoc_code"].map(ewoc_map["landcover_name"])
    df["cropgroup_name"] = df["ewoc_code"].map(ewoc_map["cropgroup_name"])
    df["croptype_name"] = df["ewoc_code"].map(ewoc_map["croptype_name"])

    df["downstream_class"] = df["ewoc_code"].map(
        {int(k): v for k, v in get_class_mappings()[downstream_classes].items()}
    )

    return df


def _check_geom(row):
    try:
        result = row["geometry"].contains(row["centroid"])
    except Exception:
        result = False
    return result


def _to_points(df):
    """Convert reference dataset to points."""

    # if geometry type is point, return df
    if df["geometry"].geom_type[0] == "Point":
        return df
    else:
        # convert polygons to points
        df["centroid"] = df["geometry"].centroid
        # check whether centroid is in the original geometry
        df["centroid_in"] = df.apply(lambda x: _check_geom(x), axis=1)
        df = df[df["centroid_in"]]
        df.drop(columns=["geometry", "centroid_in"], inplace=True)
        df.rename(columns={"centroid": "geometry"}, inplace=True)
        return df
