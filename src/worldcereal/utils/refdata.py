import glob
import importlib.resources
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union, cast

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from loguru import logger
from openeo_gfmap import TemporalContext
from prometheo.utils import DEFAULT_SEED
from shapely import wkt
from shapely.geometry import Polygon

from worldcereal.data import croptype_mappings
from worldcereal.train.datasets import MIN_EDGE_BUFFER

DATA_DIR = Path(__file__).parent.parent / "data"


def get_class_mappings() -> Dict:
    """Method to get the WorldCereal class mappings for downstream task.

    Returns
    -------
    Dict
        the resulting dictionary with the class mappings
    """
    resource = importlib.resources.files(croptype_mappings) / "class_mappings.json"  # type: ignore[attr-defined]
    CLASS_MAPPINGS = json.loads(resource.read_text(encoding="utf-8"))

    return CLASS_MAPPINGS


def get_legend() -> pd.DataFrame:
    """Method to get the latest version of the WorldCereal legend.

    Returns
    -------
    pd.DataFrame
        the latest parsed version of the WorldCereal legend
    """

    artifactory_base_url = (
        "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/"
    )
    crop_legend_url = (
        artifactory_base_url + "legend/WorldCereal_LC_CT_legend_latest.csv"
    )

    crop_legend = pd.read_csv(crop_legend_url, header=0, sep=";")
    crop_legend["ewoc_code"] = crop_legend["ewoc_code"].str.replace("-", "").astype(int)
    crop_legend = crop_legend.ffill(axis=1)

    return crop_legend


def map_classes(
    df: pd.DataFrame,
    finetune_classes="CROPTYPE0",
    class_mappings: Dict[str, Dict[str, str]] = get_class_mappings(),
    filter_classes=[0, 1000000000],
) -> pd.DataFrame:
    """
    Maps the original classes in a DataFrame to fine-tuning classes based on predefined mappings.

    This function takes a DataFrame containing 'ewoc_code' column and maps these codes to new classes
    defined in `class_mappings` dictionary under the specified fine-tuning class set. It also
    creates a 'balancing_class' column based on the WorldCereal crop legend (dynamically loaded)
    for potential class balancing.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing an 'ewoc_code' column with original class codes.
        finetune_classes (str):
            The set of fine-tuning classes to use from CLASS_MAPPINGS.
            This should be one of the keys in CLASS_MAPPINGS.
            Most popular maps: "LANDCOVER14", "CROPTYPE9", "CROPTYPE0", "CROPLAND2".
            Defaults to "CROPLAND2".
        class_mappings (dict, optional):
            Dictionary containing the mapping of original class codes to new class labels.
        filter_classes (list, optional):
            List of class codes to exclude from the dataset.
            Defaults to [0, 1000000000].

    Returns:
        pd.DataFrame: The processed DataFrame with added columns:
            - 'finetune_class': The mapped class labels
            - 'balancing_class': Class labels for dataset balancing

    Notes:
        - Removes classes that are not present in the CLASS_MAPPINGS dictionary
    """

    df = df.loc[~df["ewoc_code"].isin(filter_classes)].copy()
    legend = get_legend().set_index("ewoc_code")

    # Check if all classes are present in the mapping dictionary
    existing_codes_list = set(df["ewoc_code"].astype(int).astype(str).unique())
    # Compute codes that are missing in the mapping dictionary
    missing_codes = existing_codes_list - set(class_mappings[finetune_classes])
    if missing_codes:
        missing_classes_count = {}
        for code in list(missing_codes):
            if int(code) not in legend.index:
                logger.warning(
                    f"Class code `{code}` is not present in the WorldCereal legend, and will be skipped!"
                )
                continue
            class_name = legend.loc[int(code)]["label_full"]
            class_count = (df["ewoc_code"] == int(code)).sum()
            missing_classes_count[f"{code} - {class_name}"] = class_count

        if len(missing_classes_count) > 0:
            logger.warning(
                f"Some classes are missing in the mapping dictionary and thus will be removed: {missing_classes_count}. "
                f"A total of {(df.ewoc_code.astype(str).isin(missing_codes)).sum()} samples will be removed from the dataframe."
            )
        df = df.loc[~df["ewoc_code"].astype(int).astype(str).isin(missing_codes)].copy()

    df.loc[:, "finetune_class"] = df["ewoc_code"].map(
        {int(k): v for k, v in class_mappings[finetune_classes].items()}
    )

    # Will be used for balancing if the flag is set
    df.loc[:, "balancing_class"] = df["ewoc_code"].map(legend["sampling_label"])

    return df


def query_public_extractions(
    bbox_poly: Polygon,
    buffer: int = 250000,
    filter_cropland: bool = True,
    crop_types: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Query the WorldCereal global extractions database for reference data within a specified area.

    This function retrieves training samples from the WorldCereal public extractions database
    that intersect with the provided polygon (with optional buffer). The function connects to
    an S3 bucket containing WorldCereal reference data and performs spatial queries to find
    relevant samples.

    Parameters
    ----------
    bbox_poly : Polygon
        A shapely Polygon object defining the area of interest in EPSG:4326 (WGS84)
    buffer : int, default=250000
        Buffer distance in meters to expand the search area around the input polygon.
        The buffer is applied in EPSG:3785 (Web Mercator) projection.
    filter_cropland : bool, default=True
        If True, filter results to include only temporary cropland samples (WorldCereal
        classes with codes 11-... except fallow classes 11-15-...). This step is needed
        when preparing data for croptype classification models.
    crop_types : Optional[List[int]], optional
        List of crop types to filter on, by default None
        If None, all crop types are included.

    Returns
    -------
    pd.DataFrame
        A GeoDataFrame containing reference data points that intersect with the area of interest.
        Each row represents a single sample with its geometry and associated attributes.

    """

    logger.info(
        f"Applying a buffer of {int(buffer / 1000)} km to the selected area ..."
    )

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

    metadata_s3_path = "s3://geoparquet/worldcereal_public_extractions_extent.parquet"

    query_metadata = f"""
    SET s3_endpoint='s3.waw3-1.cloudferro.com';
    SET enable_progress_bar=false;
    SET TimeZone = 'UTC';
    SELECT distinct ref_id
    FROM read_parquet('{metadata_s3_path}') metadata
    WHERE ST_Intersects(geometry, ST_GeomFromText('{str(bbox_poly)}'))
    """
    ref_ids_lst = db.sql(query_metadata).df()["ref_id"].values

    if len(ref_ids_lst) == 0:
        logger.warning(
            "No datasets found in the WorldCereal global extractions database that intersect with the selected area."
        )
        return pd.DataFrame()

    logger.info(
        f"Found {len(ref_ids_lst)} datasets in WorldCereal global extractions database that intersect with the selected area."
    )

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

    if crop_types is not None:
        ct_list_str = ",".join([str(x) for x in crop_types])
        cropland_filter_query_part += f"""
AND ewoc_code IN ({ct_list_str})
"""

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
        logger.warning(
            f"No samples from the WorldCereal global extractions database fall into the selected area with buffer {int(buffer / 1000)} km²."
        )
        return pd.DataFrame()

    # add filename column for compatibility with private extractions; make it copy of ref_id for now
    public_df_raw["filename"] = public_df_raw["ref_id"]

    return public_df_raw


def query_private_extractions(
    merged_private_parquet_path: str,
    bbox_poly: Optional[Polygon] = None,
    filter_cropland: bool = True,
    buffer: int = 250000,
    crop_types: Optional[list[int]] = None,
    ref_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Query and filter private extraction data stored in parquet files.

    This function reads parquet files from a specified path, optionally filters them based
    on spatial and cropland criteria, and returns the results as a GeoPandas DataFrame.

    Parameters
    ----------
    merged_private_parquet_path : str
        Path to the directory containing private parquet files, which will be searched recursively.
    bbox_poly : Optional[Polygon], default=None
        Optional bounding box polygon to spatially filter the data. If provided, only data
        intersecting with this polygon (after buffering) will be returned.
    filter_cropland : bool, default=True
        Whether to filter for temporary cropland samples (WorldCereal codes 1100000000-1115000000,
        excluding fallow classes). Should be True when using data for croptype classification.
    buffer : int, default=250000
        Buffer distance in meters to apply to the bounding box polygon when spatial filtering.
    crop_types : Optional[List[int]], optional
            List of crop types to filter on, by default None
            If None, all crop types are included.
    ref_ids : Optional[List[str]], optional
        List of reference IDs to filter on, by default None
        If None, all reference IDs are included.

    Returns
    -------
    pd.DataFrame
        A GeoPandas DataFrame containing the filtered private extraction data with valid geometry objects.

    Notes
    -----
    - The function uses DuckDB with spatial extension for efficient spatial querying.
    - Temporary cropland is defined based on WorldCereal legend codes starting with 11-...,
      except fallow classes (11-15-...).
    """

    private_collection_paths = glob.glob(
        f"{merged_private_parquet_path}/**/*.parquet",
        recursive=True,
    )

    if ref_ids is not None:
        private_collection_paths = [
            p for p in private_collection_paths if Path(p).stem in ref_ids
        ]
    if len(private_collection_paths) == 0:
        logger.warning("No private collections found.")

    logger.info(f"Checking {len(private_collection_paths)} datasets...")

    if bbox_poly is not None:
        bbox_poly = (
            gpd.GeoSeries(bbox_poly, crs="EPSG:4326")
            .to_crs(epsg=3785)
            .buffer(buffer, cap_style="square", join_style="mitre")
            .to_crs(epsg=4326)[0]
        )

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
        prefix = "WHERE" if bbox_poly is None else "AND"
        cropland_filter_query_part = f"""
{prefix} ewoc_code < 1115000000
AND ewoc_code > 1100000000
"""
    else:
        cropland_filter_query_part = ""

    if crop_types is not None:
        if (bbox_poly is None) and (not filter_cropland):
            prefix = "WHERE"
        else:
            prefix = "AND"
        ct_list_str = ",".join([str(x) for x in crop_types])
        cropland_filter_query_part += f"""
{prefix} ewoc_code IN ({ct_list_str})
"""

    main_query = "SET TimeZone = 'UTC';\n"
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
        logger.warning(
            f"No intersecting samples detected in the private collections: {private_collection_paths}."
        )

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


def is_within_period(proposed_date, start_date, end_date, buffer):
    return (proposed_date - pd.DateOffset(months=buffer) >= start_date) & (
        proposed_date + pd.DateOffset(months=buffer) <= end_date
    )


def check_shift(proposed_date, valid_time, start_date, end_date, buffer, num_timesteps):
    proposed_start_date = proposed_date - pd.DateOffset(months=(num_timesteps // 2 - 1))
    proposed_end_date = proposed_date + pd.DateOffset(months=(num_timesteps // 2))
    return (
        # checks that the middle of the proposed period is within the available extractions
        is_within_period(proposed_date, start_date, end_date, MIN_EDGE_BUFFER)
        # checks that the proposed period does not fall too far outside the available extractions
        & (proposed_start_date + pd.DateOffset(months=MIN_EDGE_BUFFER) >= start_date)
        & (proposed_end_date - pd.DateOffset(months=MIN_EDGE_BUFFER) <= end_date)
        # checks that true valid_time is not too close to the edges of the proposed period
        & ((valid_time - pd.DateOffset(months=buffer)) >= proposed_start_date)
        & ((valid_time + pd.DateOffset(months=buffer)) <= proposed_end_date)
    )


def get_best_valid_time(
    row: pd.Series, valid_time_buffer: int, num_timesteps: int
) -> Union[pd.Timestamp, float]:
    """
    Determines the best valid time for a given row of data based on specified shift constraints.

    This function evaluates potential valid times by shifting the original valid time
    forward or backward according to the values in 'valid_month_shift_forward' and
    'valid_month_shift_backward' fields. It ensures the shifted time remains within
    the period defined by 'start_date' and 'end_date' with sufficient buffer.

    Parameters
    ----------
    row : pd.Series
        A pandas Series containing the following fields:
        - valid_time: The original valid time
        - start_date: The start date of the allowed period
        - end_date: The end date of the allowed period
        - valid_month_shift_forward: Number of months to shift forward
        - valid_month_shift_backward: Number of months to shift backward
    valid_time_buffer : int
        Temporal buffer in months, determining how close we allow the true valid_time of the sample to be to the edge of the proposed processing period.

    num_timesteps : int
        The number of timesteps accepted by the model.
        This is used to define the middle of the user-defined period.

    Returns
    -------
    datetime or np.nan
        The best valid time after applying shifts, or np.nan if no valid time can be found.
        If both forward and backward shifts are valid, the choice depends on the relative
        magnitude of the shifts compared to buffer.
    """

    # Run a check on provided valid_time_buffer
    if valid_time_buffer > num_timesteps // 2:
        logger.warning(
            f"The provided valid_time_buffer ({valid_time_buffer} months) is larger than half the number of timesteps in the processing period. "
            f"Reducing valid_time_buffer to half the number of timesteps: {num_timesteps // 2} months"
        )
        valid_time_buffer = num_timesteps // 2
    if valid_time_buffer < 0:
        raise ValueError(
            f"The provided valid_time_buffer ({valid_time_buffer} months) must be a non-negative integer."
        )

    # Extract necessary fields from the row
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
        proposed_valid_time_fwd,
        valid_time,
        start_date,
        end_date,
        valid_time_buffer,
        num_timesteps,
    )
    shift_backward_ok = check_shift(
        proposed_valid_time_bwd,
        valid_time,
        start_date,
        end_date,
        valid_time_buffer,
        num_timesteps,
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
    return np.nan


def process_extractions_df(
    df_raw: Union[pd.DataFrame, gpd.GeoDataFrame],
    processing_period: TemporalContext = None,
    freq: Literal["month", "dekad"] = "month",
    valid_time_buffer: int = MIN_EDGE_BUFFER,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Process a dataframe of extracted samples to align with a specified temporal context and frequency.

    This function processes a dataframe of raw samples, typically extracted from a reference database,
    by ensuring datetime consistency, aligning samples with a specified temporal extent (if provided),
    and applying additional time-series processing.

    Parameters
    ----------
    df_raw : Union[pd.DataFrame, gpd.GeoDataFrame]
        Raw dataframe or geodataframe containing extracted samples with at least 'valid_time'
        and 'timestamp' columns.
    processing_period : TemporalContext, optional
        User-defined temporal context to align samples with. If provided, samples that do not fit
        within this temporal extent will be removed. Default is None (no temporal filtering).
    freq : Literal["month", "dekad"], default "month"
        Frequency alias for time series processing. Currently only "month" and "dekad" are supported.
    valid_time_buffer : int, default MIN_EDGE_BUFFER (2)
        Buffer in months to apply when aligning available extractions with user-defined temporal extent.
        Determines how close we allow the true valid_time of the sample to be to the edge of the processing period

    Returns
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]
        Processed dataframe with time series aligned to the specified frequency and temporal context.
        Will include 'ewoc_code' column mapping crop types if not already present.

    Raises
    ------
    ValueError
        If an unsupported frequency alias is provided or if no samples match the proposed temporal extent.

    Notes
    -----
    The function preserves the original 'valid_time' even when samples are aligned to a new temporal context.
    """

    from worldcereal.utils.timeseries import (
        DataFrameValidator,
        TimeSeriesProcessor,
        process_parquet,
    )

    logger.info("Processing selected samples ...")

    # check for essential attributes
    required_columns = [
        "valid_time",
        "timestamp",
        "sample_id",
    ]
    for col in required_columns:
        if col not in df_raw.columns:
            error_msg = f"Missing required column: {col}. Please check the input data."
            logger.error(error_msg)
            raise ValueError(error_msg)

    # make sure the timestamp and valid_time are datetime objects with no timezone
    df_raw = DataFrameValidator.validate_and_fix_dt_cols(df_raw)
    df_raw = cast(Union[pd.DataFrame, gpd.GeoDataFrame], df_raw)

    if processing_period is not None:
        logger.info("Aligning the samples with the user-defined temporal extent ...")

        # get the middle of the user-defined temporal extent
        start_date, end_date = processing_period.to_datetime()

        # define num_timesteps based on frequency
        if freq == "month":
            num_timesteps = 12
        elif freq == "dekad":
            num_timesteps = 36
        else:
            raise ValueError(
                f"Unsupported frequency alias: {freq}. Please use 'month' or 'dekad'."
            )

        date_range = TimeSeriesProcessor.get_expected_dates(start_date, end_date, freq)

        middle_index = len(date_range) // 2 - 1
        processing_period_middle_ts = date_range[middle_index]
        processing_period_middle_month = processing_period_middle_ts.month

        # # calculate the start and end dates of available extractions per sample
        df_raw["start_date"] = df_raw.groupby("sample_id")["timestamp"].transform("min")
        df_raw["end_date"] = df_raw.groupby("sample_id")["timestamp"].transform("max")

        # get a lighter subset with only the necessary columns
        sample_dates = (
            df_raw[["sample_id", "start_date", "end_date", "valid_time"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # save the true valid_time for later
        true_valid_time_map = sample_dates.set_index("sample_id")["valid_time"]

        # calculate the shifts and assign new valid date
        sample_dates["true_valid_time_month"] = sample_dates["valid_time"].dt.month
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
            get_best_valid_time,
            axis=1,
            valid_time_buffer=valid_time_buffer,
            num_timesteps=num_timesteps,
        )

        # remove invalid samples
        invalid_samples = sample_dates.loc[
            sample_dates["proposed_valid_time"].isna(), "sample_id"
        ].values
        df_raw = df_raw.loc[~df_raw["sample_id"].isin(invalid_samples)].copy()

        if df_raw.empty:
            error_msg = "None of the samples matched the proposed temporal extent. Please select a different temporal extent."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            if invalid_samples.shape[0] > 0:
                logger.warning(
                    f"Removed {invalid_samples.shape[0]} samples that do not fit into selected temporal extent."
                )

        # put the proposed valid_time back into the main dataframe
        df_raw.loc[:, "valid_time"] = df_raw["sample_id"].map(
            sample_dates.set_index("sample_id")["proposed_valid_time"]
        )

    df_processed = process_parquet(
        df_raw,
        freq=freq,
        use_valid_time=True,
        max_timesteps_trim="auto",
    )

    if processing_period is not None:
        # put back the true valid_time
        df_processed["valid_time"] = df_processed.index.map(true_valid_time_map)
        df_processed["valid_time"] = df_processed["valid_time"].dt.strftime("%Y-%m-%d")

    logger.info(
        f"Extracted and processed {df_processed.shape[0]} samples from global database."
    )

    return df_processed


def join_with_world_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    filename = (
        "world-administrative-boundaries/world-administrative-boundaries.geoparquet"
    )
    world_df = gpd.read_parquet(DATA_DIR / filename)
    world_df = world_df.drop(columns=["status", "color_code", "iso_3166_1_"])

    gdataframe = gpd.GeoDataFrame(
        data=dataframe,
        geometry=gpd.GeoSeries.from_xy(x=dataframe.lon, y=dataframe.lat),
        crs="EPSG:4326",
    )
    # project to non geographic CRS, otherwise geopandas gives a warning
    joined = gpd.sjoin_nearest(
        gdataframe.to_crs("EPSG:3857"), world_df.to_crs("EPSG:3857"), how="left"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    if joined.isna().any(axis=1).any():
        logger.warning("Some coordinates couldn't be matched to a country")
    return joined.to_crs("EPSG:4326")


def split_df(
    df: pd.DataFrame,
    val_sample_ids: Optional[List[str]] = None,
    val_countries_iso3: Optional[List[str]] = None,
    val_years: Optional[List[int]] = None,
    val_size: Optional[float] = None,
    train_only_samples: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_size is not None:
        assert (
            (val_countries_iso3 is None)
            and (val_years is None)
            and (val_sample_ids is None)
        )
        val, train = np.split(
            df.sample(frac=1, random_state=DEFAULT_SEED), [int(val_size * len(df))]
        )
        logger.info(f"Using {len(train)} train and {len(val)} val samples")
        return pd.DataFrame(train), pd.DataFrame(val)
    if val_sample_ids is not None:
        assert (val_countries_iso3 is None) and (val_years is None)
        is_val = df.sample_id.isin(val_sample_ids)
        is_train = ~df.sample_id.isin(val_sample_ids)
    elif val_countries_iso3 is not None:
        assert (val_sample_ids is None) and (val_years is None)
        df = join_with_world_df(df)
        for country in val_countries_iso3:
            assert df.iso3.str.contains(
                country
            ).any(), f"Tried removing {country} but it is not in the dataframe"
        if train_only_samples is not None:
            is_val = df.iso3.isin(val_countries_iso3) & ~df.sample_id.isin(
                train_only_samples
            )
        else:
            is_val = df.iso3.isin(val_countries_iso3)
        is_train = ~df.iso3.isin(val_countries_iso3)
    elif val_years is not None:
        df["end_date_ts"] = pd.to_datetime(df.end_date)
        if train_only_samples is not None:
            is_val = df.end_date_ts.dt.year.isin(val_years) & ~df.sample_id.isin(
                train_only_samples
            )
        else:
            is_val = df.end_date_ts.dt.year.isin(val_years)
        is_train = ~df.end_date_ts.dt.year.isin(val_years)

    logger.info(
        f"Using {len(is_val) - sum(is_val)} train and {sum(is_val)} val samples"
    )

    return df[is_train], df[is_val]


def _check_geom(row):
    try:
        result = row["geometry"].contains(row["centroid"])
    except Exception:
        result = False
    return result


def gdf_to_points(gdf):
    """Convert reference dataset to points.
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        input geodataframe containing reference data samples
    Returns
    -------
    gpd.GeoDataFrame
        geodataframe in which polygons have been converted to points
    """

    # reproject to projected system
    crs_ori = gdf.crs
    gdf = gdf.to_crs(epsg=3857)
    # convert polygons to points
    gdf["centroid"] = gdf["geometry"].centroid
    # check whether centroid is in the original geometry
    n_original = gdf.shape[0]
    gdf["centroid_in"] = gdf.apply(lambda x: _check_geom(x), axis=1)
    gdf = gdf[gdf["centroid_in"]]
    n_remaining = gdf.shape[0]
    if n_remaining < n_original:
        logger.warning(
            f"Removed {n_original - n_remaining} polygons that do not contain their centroid."
        )
    gdf.drop(columns=["geometry", "centroid_in"], inplace=True)
    gdf.rename(columns={"centroid": "geometry"}, inplace=True)
    # reproject to original system
    gdf = gdf.to_crs(crs_ori)

    return gdf
