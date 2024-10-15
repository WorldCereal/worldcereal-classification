import importlib.resources
import json
from typing import Dict

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
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

    logger.info(f"Applying a buffer of {int(buffer/1000)} km to the selected area ...")

    bbox_poly = (
        gpd.GeoSeries(bbox_poly, crs="EPSG:4326")
        .to_crs(epsg=3785)
        .buffer(buffer, cap_style="square", join_style="mitre")
        .to_crs(epsg=4326)[0]
    )

    xmin, ymin, xmax, ymax = bbox_poly.bounds
    bbox_poly = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    db = duckdb.connect()
    db.sql("INSTALL spatial")
    db.load_extension("spatial")

    metadata_s3_path = "s3://geoparquet/ref_id_extent.parquet"

    query_metadata = f"""
    SET s3_endpoint='s3.waw3-1.cloudferro.com';
    SET enable_progress_bar=false;
    SELECT distinct ref_id
    FROM read_parquet('{metadata_s3_path}') metadata
    WHERE ST_Intersects(ST_GeomFromText(str_geom), ST_GeomFromText('{str(bbox_poly)}'))
    """
    ref_ids_lst = db.sql(query_metadata).df()["ref_id"].values

    if len(ref_ids_lst) == 0:
        logger.error(
            "No datasets found in WorldCereal global extractions database that intersect with the selected area."
        )
        raise ValueError()

    logger.info(
        f"Found {len(ref_ids_lst)} datasets in WorldCereal global extractions database that intersect with the selected area."
    )

    # only querying the croptype data here
    logger.info(
        "Querying WorldCereal global extractions database (this can take a while) ..."
    )

    all_extractions_url = "https://s3.waw3-1.cloudferro.com/swift/v1/geoparquet/"
    f = requests.get(all_extractions_url)
    all_dataset_names = f.text.split("\n")
    matching_dataset_names = [
        xx
        for xx in all_dataset_names
        if xx.endswith(".parquet")
        and xx.startswith("worldcereal_extractions_phase1")
        and any([yy in xx for yy in ref_ids_lst])
    ]
    base_s3_path = "s3://geoparquet/"
    s3_urls_lst = [f"{base_s3_path}{xx}" for xx in matching_dataset_names]

    main_query = "SET s3_endpoint='s3.waw3-1.cloudferro.com';"
    if filter_cropland:
        for i, url in enumerate(s3_urls_lst):
            query = f"""
SELECT *
FROM read_parquet('{url}')
WHERE ST_Intersects(ST_MakeValid(ST_GeomFromText(geometry)), ST_GeomFromText('{str(bbox_poly)}'))
AND LANDCOVER_LABEL = 11
AND CROPTYPE_LABEL not in (0, 991, 7900, 9900, 9998, 1910, 1900, 1920, 1000, 11, 9910, 6212, 7920, 9520, 3400, 3900, 4390, 4000, 4300)
"""
            if i == 0:
                main_query += query
            else:
                main_query += f"UNION ALL {query}"
    else:
        for i, url in enumerate(s3_urls_lst):
            query = f"""
SELECT *
FROM read_parquet('{url}')
WHERE ST_Intersects(ST_MakeValid(ST_GeomFromText(geometry)), ST_GeomFromText('{str(bbox_poly)}'))
"""
            if i == 0:
                main_query += query
            else:
                main_query += f"UNION ALL {query}"

    public_df_raw = db.sql(main_query).df()

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

    # type: ignore
    with importlib.resources.open_text(
        croptype_mappings, "eurocrops_map_wcr_edition.csv"
    ) as f:
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
    df["label_level1"] = df["ewoc_code"].map(ewoc_map["cropland_name"])
    df["label_level2"] = df["ewoc_code"].map(ewoc_map["landcover_name"])
    df["label_level3"] = df["ewoc_code"].map(ewoc_map["croptype_name"])

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
