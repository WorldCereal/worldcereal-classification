import importlib.resources
import json
from typing import Dict

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from loguru import logger
from openeo_gfmap import TemporalContext
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
    bbox_poly: Polygon,
    buffer: int = 250000,
    filter_cropland: bool = True,
    processing_period: TemporalContext = None,
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
    processing_period : TemporalContext, optional
        user-defined temporal extent to align the samples with, by default None,
        which means that 12-month processing window will be aligned around each sample's original valid_date.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extractions matching the request.
    """
    from IPython.display import Markdown

    nodata_helper_message = f"""
### What to do?
1. **Increase the buffer size**: Try increasing the buffer size by passing the `buffer` parameter to the `query_public_extractions` function (to a reasonable extent).
    *Current setting is: {buffer} m².*
2. **Consult the WorldCereal Reference Data Module portal**: Assess data density in the selected region by visiting the [WorldCereal Reference Data Module portal](https://ewoc-rdm-ui.iiasa.ac.at/map).
3. **Pick another area**: Consult RDM portal (see above) to find areas with more data density.
4. **Contribute data**: Collect some data and contribute to our global database! 🌍🌾 [Learn how to contribute here.](https://worldcereal.github.io/worldcereal-documentation/rdm/upload.html)
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
            "No datasets found in the WorldCereal global extractions database that intersect with the selected area."
        )
        Markdown(nodata_helper_message)
        raise ValueError(
            "No datasets found in the WorldCereal global extractions database that intersect with the selected area."
        )

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

    if public_df_raw.empty:
        logger.error(
            f"No samples from the WorldCereal global extractions database fall into the selected area with buffer {int(buffer/1000)}km2."
        )
        Markdown(nodata_helper_message)
        raise ValueError(
            "No samples from the WorldCereal global extractions database fall into the selected area."
        )
    if public_df_raw["CROPTYPE_LABEL"].nunique() == 1:
        logger.error(
            f"Queried data contains only one class: {public_df_raw['croptype_name'].unique()[0]}. Cannot train a model with only one class."
        )
        Markdown(nodata_helper_message)
        raise ValueError(
            "Queried data contains only one class. Cannot train a model with only one class."
        )

    # Process the parquet into the format we need for training
    processed_public_df = process_parquet(public_df_raw, processing_period)

    return processed_public_df


def month_diff(month1: int, month2: int) -> int:
    """This function computes the difference between `month1` and `month2`
    assuming that `month1` is in the past relative to `month2`.
    The difference is calculated such that it falls within the range of 0 to 12 months.

    Parameters
    ----------
    month1 : int
        The reference month (1-12).
    month2 : int
        The month to compare against (1-12).

    Returns
    -------
    int
        The difference between `month1` and `month2`.
    """

    return month2 - month1 if month2 >= month1 else 12 - month1 + month2


def get_best_valid_date(row: pd.Series):
    """Determine the best valid date for a given row based on forward and backward shifts.
    This function checks if shifting the valid date forward or backward by a specified number of months
    will fit within the existing extraction dates. It returns the new valid date based on the shifts or
    NaN if neither shift is possible.

    Parameters
    ----------
    row : pd.Series
        A row from raw flattened dataframe from the global database that contains the following columns:
        - "sample_id" (str): The unique sample identifier.
        - "valid_date" (pd.Timestamp): The original valid date.
        - "valid_month_shift_forward" (int): Number of months to shift forward.
        - "valid_month_shift_backward" (int): Number of months to shift backward.
        - "start_date" (pd.Timestamp): The start date of the extraction period.
        - "end_date" (pd.Timestamp): The end date of the extraction period.

    Returns
    -------
    pd.Datetime
        shifted valid date
    """

    from presto.dataops import MIN_EDGE_BUFFER, NUM_TIMESTEPS

    # check if shift forward will fit into existing extractions
    # allow buffer of MIN_EDGE_BUFFER months at the start and end of the extraction period
    temp_end_date = row["valid_date"] + pd.DateOffset(
        months=row["valid_month_shift_forward"] + NUM_TIMESTEPS // 2 - MIN_EDGE_BUFFER
    )
    temp_start_date = temp_end_date - pd.DateOffset(months=NUM_TIMESTEPS)
    if (temp_end_date <= row["end_date"]) & (temp_start_date >= row["start_date"]):
        shift_forward_ok = True
    else:
        shift_forward_ok = False

    # check if shift backward will fit into existing extractions
    # allow buffer of MIN_EDGE_BUFFER months at the start and end of the extraction period
    temp_start_date = row["valid_date"] - pd.DateOffset(
        months=row["valid_month_shift_backward"] + NUM_TIMESTEPS // 2 - MIN_EDGE_BUFFER
    )
    temp_end_date = temp_start_date + pd.DateOffset(months=NUM_TIMESTEPS)
    if (temp_end_date <= row["end_date"]) & (temp_start_date >= row["start_date"]):
        shift_backward_ok = True
    else:
        shift_backward_ok = False

    if (not shift_forward_ok) & (not shift_backward_ok):
        return np.nan

    if shift_forward_ok & (not shift_backward_ok):
        return row["valid_date"] + pd.DateOffset(
            months=row["valid_month_shift_forward"]
        )

    if (not shift_forward_ok) & shift_backward_ok:
        return row["valid_date"] - pd.DateOffset(
            months=row["valid_month_shift_backward"]
        )

    if shift_forward_ok & shift_backward_ok:
        # if shift backward is not too much bigger than shift forward, choose backward
        if (
            row["valid_month_shift_backward"] - row["valid_month_shift_forward"]
        ) <= MIN_EDGE_BUFFER:
            return row["valid_date"] - pd.DateOffset(
                months=row["valid_month_shift_backward"]
            )
        else:
            return row["valid_date"] + pd.DateOffset(
                months=row["valid_month_shift_forward"]
            )


def process_parquet(
    public_df_raw: pd.DataFrame, processing_period: TemporalContext = None
) -> pd.DataFrame:
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

    if processing_period is not None:
        logger.info("Aligning the samples with the user-defined temporal extent ...")

        # get the middle of the user-defined temporal extent
        start_date, end_date = processing_period.to_datetime()
        processing_period_middle_ts = start_date + pd.DateOffset(months=6)
        processing_period_middle_month = processing_period_middle_ts.month

        # get a lighter subset with only the necessary columns
        sample_dates = (
            public_df_raw[["sample_id", "start_date", "end_date", "valid_date"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # save the true valid_date for later
        true_valid_date_map = sample_dates.set_index("sample_id")["valid_date"]

        # calculate the shifts and assign new valid date
        sample_dates["true_valid_date_month"] = public_df_raw["valid_date"].dt.month
        sample_dates["proposed_valid_date_month"] = processing_period_middle_month
        sample_dates["valid_month_shift_forward"] = sample_dates.apply(
            lambda xx: month_diff(
                xx["proposed_valid_date_month"], xx["true_valid_date_month"]
            ),
            axis=1,
        )
        sample_dates["valid_month_shift_backward"] = sample_dates.apply(
            lambda xx: month_diff(
                xx["true_valid_date_month"], xx["proposed_valid_date_month"]
            ),
            axis=1,
        )
        sample_dates["proposed_valid_date"] = sample_dates.apply(
            lambda xx: get_best_valid_date(xx), axis=1
        )

        # remove invalid samples
        invalid_samples = sample_dates.loc[
            sample_dates["proposed_valid_date"].isna(), "sample_id"
        ].values
        public_df_raw = public_df_raw[~public_df_raw["sample_id"].isin(invalid_samples)]
        public_df_raw["valid_date"] = public_df_raw["sample_id"].map(
            sample_dates.set_index("sample_id")["proposed_valid_date"]
        )
        if public_df_raw.empty:
            error_msg = "None of the samples matched the proposed temporal extent. Please select a different temporal extent."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.warning(
                f"Removed {invalid_samples.shape[0]} samples that do not fit into selected temporal extent."
            )

    public_df = process_parquet_for_presto(public_df_raw)

    if processing_period is not None:
        # put back the true valid_date
        public_df["valid_date"] = public_df.index.map(true_valid_date_map)
        public_df["valid_date"] = public_df["valid_date"].astype(str)

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
