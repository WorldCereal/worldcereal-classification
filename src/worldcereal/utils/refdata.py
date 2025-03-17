import importlib.resources
import json
from typing import Dict, List, Literal, Optional, Union

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from loguru import logger
from openeo_gfmap import TemporalContext
from shapely import wkt
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
        which means that 12-month processing window will be aligned around each sample's original valid_time.

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
2. **Consult the WorldCereal Reference Data Module portal**: Assess data density in the selected region by visiting the [WorldCereal Reference Data Module portal](https://rdm.esa-worldcereal.org/map).
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

    # metadata_s3_path = "s3://geoparquet/ref_id_extent.parquet"

    # query_metadata = f"""
    # SET s3_endpoint='s3.waw3-1.cloudferro.com';
    # SET enable_progress_bar=false;
    # SELECT distinct ref_id
    # FROM read_parquet('{metadata_s3_path}') metadata
    # WHERE ST_Intersects(ST_GeomFromText(str_geom), ST_GeomFromText('{str(bbox_poly)}'))
    # """
    # ref_ids_lst = db.sql(query_metadata).df()["ref_id"].values

    # if len(ref_ids_lst) == 0:
    #     logger.error(
    #         "No datasets found in the WorldCereal global extractions database that intersect with the selected area."
    #     )
    #     Markdown(nodata_helper_message)
    #     raise ValueError(
    #         "No datasets found in the WorldCereal global extractions database that intersect with the selected area."
    #     )

    # logger.info(
    #     f"Found {len(ref_ids_lst)} datasets in WorldCereal global extractions database that intersect with the selected area."
    # )

    logger.info(
        "Querying WorldCereal global extractions database (this can take a while) ..."
    )

    query_ref_ids = """
SET s3_endpoint='s3.waw3-1.cloudferro.com';
SELECT distinct ref_id
FROM read_parquet('s3://geoparquet/worldcereal_public_extractions.parquet/**/*.parquet')
    """

    ref_ids_lst = db.sql(query_ref_ids).df()["ref_id"].values

    all_extractions_url = "https://s3.waw3-1.cloudferro.com/swift/v1/geoparquet/"
    f = requests.get(all_extractions_url)
    all_dataset_names = f.text.split("\n")
    matching_dataset_names = [
        xx
        for xx in all_dataset_names
        if xx.endswith(".parquet")
        and xx.startswith("worldcereal_public_extractions")
        and any([yy in xx for yy in ref_ids_lst])
    ]
    base_s3_path = "s3://geoparquet/"
    s3_urls_lst = [f"{base_s3_path}{xx}" for xx in matching_dataset_names]

    main_query = "SET s3_endpoint='s3.waw3-1.cloudferro.com';"

    # worldcereal provides two types of presto models: for binary crop/nocrop task and for multiclass croptype task
    # multiclass models are trained on temporary cropland samples only, thus when user wants to do croptype classification
    # we need to filter out non-cropland samples, since croptype model will not be able to predict them correctly;
    # temporary_cropland is defined based on WorldCereal legend
    # https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal//legend/WorldCereal_LC_CT_legend_latest.csv
    # and constitutes of all classes that start with 11-..., except fallow classes (11-15-...).
    if filter_cropland:
        cropland_filter_query_part = """
AND ewoc_code < 1115000000
AND ewoc_code > 1100000000
"""
    else:
        cropland_filter_query_part = ""

    for i, url in enumerate(s3_urls_lst):
        query = f"""
SELECT *, ST_AsText(ST_MakeValid(geometry)) AS geom_text
FROM read_parquet('{url}')
WHERE ST_Intersects(ST_MakeValid(geometry), ST_GeomFromText('{str(bbox_poly)}'))
{cropland_filter_query_part}
"""
        if i == 0:
            main_query += query
        else:
            main_query += f"UNION ALL {query}"

    public_df_raw = db.sql(main_query).df()
    public_df_raw["geometry"] = public_df_raw["geom_text"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) else None
    )
    # Convert to a GeoDataFrame
    public_df_raw = gpd.GeoDataFrame(public_df_raw, geometry="geometry")

    if public_df_raw.empty:
        logger.error(
            f"No samples from the WorldCereal global extractions database fall into the selected area with buffer {int(buffer/1000)}km2."
        )
        Markdown(nodata_helper_message)
        raise ValueError(
            "No samples from the WorldCereal global extractions database fall into the selected area."
        )
    if public_df_raw["ewoc_code"].nunique() == 1:
        logger.error(
            f"Queried data contains only one class: {public_df_raw['ewoc_code'].unique()[0]}. Cannot train a model with only one class."
        )
        Markdown(nodata_helper_message)
        raise ValueError(
            "Queried data contains only one class. Cannot train a model with only one class."
        )
    # add filename column for compatibility with private extractions; make it copy of ref_id for now
    public_df_raw["filename"] = public_df_raw["ref_id"]

    return public_df_raw


def query_private_extractions(
    private_collection_paths: Union[str, List[str]],
    processing_period: TemporalContext = None,
    bbox_poly: Optional[Polygon] = None,
    filter_cropland: bool = True,
) -> pd.DataFrame:

    if isinstance(private_collection_paths, str):
        private_collection_paths = [private_collection_paths]

    if bbox_poly is not None:
        bbox_poly = gpd.GeoSeries(bbox_poly, crs="EPSG:4326")[0]
        xmin, ymin, xmax, ymax = bbox_poly.bounds
        bbox_poly = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        spatial_query_part = (
            f"WHERE ST_Intersects(geometry, ST_GeomFromText('{str(bbox_poly)}'))"
        )
    else:
        spatial_query_part = ""

    db = duckdb.connect()
    db.sql("INSTALL spatial")
    db.load_extension("spatial")

    # worldcereal provides two types of presto models: for binary crop/nocrop task and for multiclass croptype task
    # multiclass models are trained on temporary cropland samples only, thus when user wants to do croptype classification
    # we need to filter out non-cropland samples, since croptype model will not be able to predict them correctly;
    # temporary_cropland is defined based on WorldCereal legend
    # https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal//legend/WorldCereal_LC_CT_legend_latest.csv
    # and constitutes of all classes that start with 11-..., except fallow classes (11-15-...).
    if filter_cropland:
        cropland_filter_query_part = """
AND ewoc_code < 1115000000
AND ewoc_code > 1100000000
"""
    else:
        cropland_filter_query_part = ""

    main_query = ""
    for i, tpath in enumerate(private_collection_paths):
        query = f"""
SELECT *, ST_AsText(ST_MakeValid(geometry)) AS geom_text 
FROM read_parquet('{tpath}') 
{spatial_query_part}
{cropland_filter_query_part}
"""
        if i == 0:
            main_query += query
        else:
            main_query += f"UNION ALL {query}"

    private_df_raw = db.sql(main_query).df()
    private_df_raw["geometry"] = private_df_raw["geom_text"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) else None
    )
    # Convert to a GeoDataFrame
    private_df_raw = gpd.GeoDataFrame(private_df_raw, geometry="geometry")

    if private_df_raw.empty:
        logger.error(
            f"No samples detected in the private collections: {private_collection_paths}."
        )
        raise ValueError("No samples detected in the private collections.")

    return private_df_raw


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

    return (month2 - month1) % 12


def get_best_valid_time(row: pd.Series):
    """Determine the best valid date for a given row based on forward and backward shifts.
    This function checks if shifting the valid date forward or backward by a specified number of months
    will fit within the existing extraction dates. It returns the new valid date based on the shifts or
    NaN if neither shift is possible.

    Parameters
    ----------
    row : pd.Series
        A row from raw flattened dataframe from the global database that contains the following columns:
        - "sample_id" (str): The unique sample identifier.
        - "valid_time" (pd.Timestamp): The original valid date.
        - "valid_month_shift_forward" (int): Number of months to shift forward.
        - "valid_month_shift_backward" (int): Number of months to shift backward.
        - "start_date" (pd.Timestamp): The start date of the extraction period.
        - "end_date" (pd.Timestamp): The end date of the extraction period.

    Returns
    -------
    pd.Datetime
        shifted valid date
    """

    from worldcereal.utils.timeseries import MIN_EDGE_BUFFER, NUM_TIMESTEPS

    def is_within_period(proposed_date, start_date, end_date):
        return (proposed_date - pd.DateOffset(months=MIN_EDGE_BUFFER) >= start_date) & (
            proposed_date + pd.DateOffset(months=MIN_EDGE_BUFFER) <= end_date
        )

    def check_shift(proposed_date, valid_time, start_date, end_date):
        proposed_start_date = proposed_date - pd.DateOffset(
            months=(NUM_TIMESTEPS // 2 - 1)
        )
        proposed_end_date = proposed_date + pd.DateOffset(months=(NUM_TIMESTEPS // 2))
        return (
            is_within_period(proposed_date, start_date, end_date)
            & (valid_time >= proposed_start_date)
            & (valid_time <= proposed_end_date)
        )

    valid_time = row["valid_time"]
    start_date = row["start_date"]
    end_date = row["end_date"]

    proposed_valid_time_fwd = valid_time + pd.DateOffset(
        months=row["valid_month_shift_forward"]
    )
    proposed_valid_time_bwd = valid_time - pd.DateOffset(
        months=row["valid_month_shift_backward"]
    )

    shift_forward_ok = check_shift(
        proposed_valid_time_fwd, valid_time, start_date, end_date
    )
    shift_backward_ok = check_shift(
        proposed_valid_time_bwd, valid_time, start_date, end_date
    )

    if not shift_forward_ok and not shift_backward_ok:
        return np.nan
    if shift_forward_ok and not shift_backward_ok:
        return proposed_valid_time_fwd
    if not shift_forward_ok and shift_backward_ok:
        return proposed_valid_time_bwd
    if shift_forward_ok and shift_backward_ok:
        return (
            proposed_valid_time_bwd
            if (row["valid_month_shift_backward"] - row["valid_month_shift_forward"])
            <= MIN_EDGE_BUFFER
            else proposed_valid_time_fwd
        )


def process_extractions_df(
    df_raw: Union[pd.DataFrame, gpd.GeoDataFrame],
    processing_period: TemporalContext = None,
    freq: Literal["MS", "10D"] = "MS",
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Method to transform the raw parquet data into a format that can be used for
    training. Includes pivoting of the dataframe and mapping of the crop types.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Input raw flattened dataframe from the global database.
    processing_period: TemporalContext, optional
        User-defined temporal extent to align the samples with, by default None,
        which means that 12-month processing window will be aligned around each sample's original valid_time.
        If provided, the processing window will be aligned with the middle of the user-defined temporal extent, according to the
        following principles:
        - the original valid_time of the sample should remain within the processing window
        - the center of the user-defined temporal extent should be not closer than MIN_EDGE_BUFFER (by default 2 months)
          to the start or end of the extraction period
    freq : str, optional
        Frequency of the time series, by default "MS". Provided frequency alias should be compatible with pandas.
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        Currently only MS and 10D are supported.
    """
    from worldcereal.utils.timeseries import process_parquet

    logger.info("Processing selected samples ...")

    # make sure the valid_time, start and end dates are datetime objects
    for date_col in ["valid_time", "start_date", "end_date"]:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])
        df_raw[date_col] = df_raw[date_col].dt.tz_localize(df_raw["timestamp"].dt.tz)

    if processing_period is not None:
        logger.info("Aligning the samples with the user-defined temporal extent ...")

        # get the middle of the user-defined temporal extent
        start_date, end_date = processing_period.to_datetime()

        # sanity check to make sure freq is not something we still don't support in Presto
        if freq not in ["MS", "10D"]:
            raise ValueError(
                f"Unsupported frequency alias: {freq}. Please use 'MS' or '10D'."
            )

        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        middle_index = len(date_range) // 2 - 1
        processing_period_middle_ts = date_range[middle_index]
        processing_period_middle_month = processing_period_middle_ts.month

        # get a lighter subset with only the necessary columns
        sample_dates = (
            df_raw[["sample_id", "start_date", "end_date", "valid_time"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # save the true valid_time for later
        true_valid_time_map = sample_dates.set_index("sample_id")["valid_time"]

        # calculate the shifts and assign new valid date
        sample_dates["true_valid_time_month"] = df_raw["valid_time"].dt.month
        sample_dates["proposed_valid_time_month"] = processing_period_middle_month
        sample_dates["valid_month_shift_backward"] = sample_dates.apply(
            lambda xx: month_diff(
                xx["proposed_valid_time_month"], xx["true_valid_time_month"]
            ),
            axis=1,
        )
        sample_dates["valid_month_shift_forward"] = sample_dates.apply(
            lambda xx: month_diff(
                xx["true_valid_time_month"], xx["proposed_valid_time_month"]
            ),
            axis=1,
        )
        sample_dates["proposed_valid_time"] = sample_dates.apply(
            lambda xx: get_best_valid_time(xx), axis=1
        )

        # remove invalid samples
        invalid_samples = sample_dates.loc[
            sample_dates["proposed_valid_time"].isna(), "sample_id"
        ].values
        df_raw = df_raw[~df_raw["sample_id"].isin(invalid_samples)]

        # put the proposed valid_time back into the main dataframe
        df_raw.loc[:, "valid_time"] = df_raw["sample_id"].map(
            sample_dates.set_index("sample_id")["proposed_valid_time"]
        )
        if df_raw.empty:
            error_msg = "None of the samples matched the proposed temporal extent. Please select a different temporal extent."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.warning(
                f"Removed {invalid_samples.shape[0]} samples that do not fit into selected temporal extent."
            )

    df_processed = process_parquet(
        df_raw, freq=freq, use_valid_time=True, required_min_timesteps=None
    )

    if processing_period is not None:
        # put back the true valid_time
        df_processed["valid_time"] = df_processed.index.map(true_valid_time_map)
        df_processed["valid_time"] = df_processed["valid_time"].astype(str)

    if "ewoc_code" not in df_processed.columns:
        df_processed = map_croptypes(df_processed)
    logger.info(
        f"Extracted and processed {df_processed.shape[0]} samples from global database."
    )

    return df_processed


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
