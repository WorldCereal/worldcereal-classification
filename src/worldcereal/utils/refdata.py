import calendar
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
from worldcereal.train import MIN_EDGE_BUFFER

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
        missing_classes_count = pd.DataFrame(
            index=[int(code) for code in missing_codes], columns=["name", "count"]
        )
        for code in list(missing_codes):
            if int(code) not in legend.index:
                logger.warning(
                    f"Class code `{code}` is not present in the WorldCereal legend, and will be skipped!"
                )
                continue
            class_name = legend.loc[int(code)]["label_full"]
            class_count = (df["ewoc_code"] == int(code)).sum()
            missing_classes_count.loc[int(code)] = [class_name, class_count]

        if len(missing_classes_count) > 0:
            missing_classes_count = (
                missing_classes_count.sort_values(by="count", ascending=False)
                .reset_index(names=["ewoc_code"])
                .set_index("name")
            )
            missing_nr = df.ewoc_code.astype(str).isin(missing_codes).sum()
            missing_perc = round(100 * missing_nr / len(df), 2)
            logger.warning(
                f"Some classes are missing in the mapping dictionary and thus will be removed: \n\n {missing_classes_count.to_string()}\n\n"
                f"A total of {missing_nr} samples ({missing_perc}%) will be removed from the dataframe."
            )
        df = df.loc[~df["ewoc_code"].astype(int).astype(str).isin(missing_codes)].copy()

    df.loc[:, "finetune_class"] = df["ewoc_code"].map(
        {int(k): v for k, v in class_mappings[finetune_classes].items()}
    )

    # Will be used for balancing if the flag is set
    df.loc[:, "balancing_class"] = df["ewoc_code"].map(legend["sampling_label"])

    return df


def query_public_extractions(
    bbox_poly: Optional[Polygon] = None,
    buffer: int = 250000,
    filter_cropland: bool = True,
    crop_types: Optional[list[int]] = None,
    query_collateral_samples: bool = True,
    ref_ids: Optional[list[str]] = None,
) -> gpd.GeoDataFrame:
    """
    Query the WorldCereal public extractions database for reference data within a specified area and/or for specific datasets.

    This function retrieves training samples from the WorldCereal public extractions database
    using one of three modes:
    1. Spatial discovery: Query by spatial intersection with a polygon (bbox_poly only)
    2. Dataset-specific: Query specific datasets directly (ref_ids only)
    3. Combined filtering: Query specific datasets within a spatial area (both bbox_poly and ref_ids)

    The function connects to an S3 bucket containing WorldCereal reference data and performs efficient queries.

    Note: You must specify at least one of 'bbox_poly' or 'ref_ids'.

    Parameters
    ----------
    bbox_poly : Optional[Polygon], default=None
        A shapely Polygon object defining the area of interest in EPSG:4326 (WGS84).
        Can be used alone for spatial discovery, or combined with ref_ids for spatial filtering
        of specific datasets. If None, no spatial filtering is applied.
    buffer : int, default=250000
        Buffer distance in meters to expand the search area around the input polygon.
        The buffer is applied in EPSG:3785 (Web Mercator) projection.
        Only used when bbox_poly is provided.
    filter_cropland : bool, default=True
        If True, filter results to include only temporary cropland samples (WorldCereal
        classes with codes 11-... except fallow classes 11-15-...). This step is needed
        when preparing data for croptype classification models.
    crop_types : Optional[List[int]], optional
        List of crop types to filter on, by default None
        If None, all crop types are included.
    query_collateral_samples : bool, default=False
        Whether to include collateral samples in the query.
        Collateral samples are those samples that were not specifically marked for extraction,
        but fell into the vicinity of such samples during the extraction process. While using
        collateral samples will result in significant increase in the amount of samples available for training,
        it will also shift the distribution of the classes.
    ref_ids : Optional[List[str]], default=None
        List of specific reference dataset IDs to filter on. Can be used alone for dataset-specific
        queries, or combined with bbox_poly to spatially filter specific datasets. If provided,
        only data from these datasets will be queried.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing reference data points that intersect with the area of interest
        or from the specified datasets. Each row represents a single sample with its geometry
        and associated attributes. CRS of the geometry column is EPSG:4326 (WGS84).

    Raises
    ------
    ValueError
        If neither bbox_poly nor ref_ids is specified.

    """

    # Validate that at least one of bbox_poly or ref_ids is provided
    if bbox_poly is None and ref_ids is None:
        raise ValueError(
            "You must specify either 'bbox_poly' (spatial area of interest) OR 'ref_ids' (specific datasets) OR both. "
            "Cannot proceed without knowing what data to query."
        )

    # Apply buffering when spatial filtering is requested (bbox_poly provided)
    if bbox_poly is not None:
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

    # Determine which ref_ids to query
    if ref_ids is not None and bbox_poly is None:
        # Case 1: Only ref_ids provided - use them directly
        ref_ids_lst = ref_ids
        logger.info(f"Querying {len(ref_ids_lst)} specific datasets: {ref_ids_lst}")
    elif ref_ids is None and bbox_poly is not None:
        # Case 2: Only spatial area provided - discover intersecting datasets
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
                "No datasets found in the WorldCereal public extractions database that intersect with the selected area."
            )
            return gpd.GeoDataFrame()

        logger.info(
            f"Found {len(ref_ids_lst)} datasets in WorldCereal public extractions database that intersect with the selected area."
        )
    else:
        # Case 3: Both ref_ids and bbox_poly provided - filter ref_ids by spatial intersection
        # At this point we know ref_ids is not None due to the conditional logic above
        assert ref_ids is not None
        logger.info(
            f"Filtering {len(ref_ids)} specific datasets by spatial intersection..."
        )

        # Create a filter for the specific ref_ids
        ref_ids_filter = "', '".join(ref_ids)
        query_metadata = f"""
        SET s3_endpoint='s3.waw3-1.cloudferro.com';
        SET enable_progress_bar=false;
        SET TimeZone = 'UTC';
        SELECT distinct ref_id
        FROM read_parquet('{metadata_s3_path}') metadata
        WHERE ST_Intersects(geometry, ST_GeomFromText('{str(bbox_poly)}'))
        AND ref_id IN ('{ref_ids_filter}')
        """
        ref_ids_lst = db.sql(query_metadata).df()["ref_id"].values

        if len(ref_ids_lst) == 0:
            logger.warning(
                f"None of the specified datasets ({ref_ids}) intersect with the selected area."
            )
            return gpd.GeoDataFrame()

    logger.info("Querying extractions...")

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

    if query_collateral_samples:
        collateral_query_part = "AND extract >= 0"
    else:
        collateral_query_part = "AND extract >= 1"

    if crop_types is not None:
        ct_list_str = ",".join([str(x) for x in crop_types])
        cropland_filter_query_part += f"""
AND ewoc_code IN ({ct_list_str})
"""

    for i, url in enumerate(s3_urls_lst):
        # Apply spatial filtering when bbox_poly is provided
        if bbox_poly is not None:
            spatial_condition = f"WHERE ST_Intersects(ST_MakeValid(geometry), ST_GeomFromText('{str(bbox_poly)}'))"
            # Use AND for subsequent conditions
            cropland_condition = cropland_filter_query_part
            collateral_condition = collateral_query_part
        else:
            spatial_condition = ""
            # Convert first AND to WHERE if no spatial condition
            cropland_condition = (
                cropland_filter_query_part.replace("AND ", "WHERE ", 1)
                if cropland_filter_query_part
                else ""
            )
            collateral_condition = collateral_query_part.replace(
                "AND ", "WHERE " if not cropland_condition else "AND ", 1
            )

        query = f"""
SELECT *, ST_AsText(ST_MakeValid(geometry)) AS geom_text
FROM read_parquet('{url}')
{spatial_condition}
{cropland_condition}
{collateral_condition}
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
    public_df_raw = gpd.GeoDataFrame(
        public_df_raw, geometry="geometry", crs="EPSG:4326"
    )

    if public_df_raw.empty:
        logger.warning("No samples found matching your search criteria.")
        return gpd.GeoDataFrame()

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
) -> gpd.GeoDataFrame:
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
    gpd.GeoDataFrame
        A GeoPandas DataFrame containing the filtered private extraction data with valid geometry objects.
        The CRS of the geometry column is EPSG:4326 (WGS84).

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
        return gpd.GeoDataFrame()

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
    private_df_raw = gpd.GeoDataFrame(
        private_df_raw, geometry="geometry", crs="EPSG:4326"
    )

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


STEPS_PER_YEAR = {"month": 12, "dekad": 36}


def _step_in_year(ts: pd.Timestamp, freq: str) -> int:
    ts = pd.Timestamp(ts)
    if freq == "month":
        return ts.month - 1
    if freq == "dekad":
        dekad = 0 if ts.day <= 10 else 1 if ts.day <= 20 else 2
        return (ts.month - 1) * 3 + dekad
    raise ValueError(f"Unsupported freq: {freq}")


def _freq_step_index(ts: pd.Timestamp, freq: str) -> int:
    ts = pd.Timestamp(ts)
    if freq == "month":
        return ts.year * 12 + (ts.month - 1)
    if freq == "dekad":
        dekad = 0 if ts.day <= 10 else 1 if ts.day <= 20 else 2
        return ts.year * 36 + (ts.month - 1) * 3 + dekad
    raise ValueError(f"Unsupported freq: {freq}")


def _index_to_ts(idx: int, freq: str) -> pd.Timestamp:
    if freq == "month":
        year, month0 = divmod(idx, 12)
        return pd.Timestamp(year, month0 + 1, 1)
    if freq == "dekad":
        year, rem = divmod(idx, 36)
        month0, dekad = divmod(rem, 3)
        day = 1 if dekad == 0 else 11 if dekad == 1 else 21
        return pd.Timestamp(year, month0 + 1, day)
    raise ValueError(f"Unsupported freq: {freq}")


def is_within_period(proposed_step, start_step, end_step, buffer_steps):
    return (proposed_step - buffer_steps >= start_step) & (
        proposed_step + buffer_steps <= end_step
    )


def check_shift(
    proposed_step: int,
    valid_step: int,
    start_step: int,
    end_step: int,
    buffer_steps: int,
    num_timesteps: int,
    edge_steps: int,
) -> bool:
    proposed_start_step = proposed_step - (num_timesteps // 2 - 1)
    proposed_end_step = proposed_step + (num_timesteps // 2)
    return (
        # checks that the middle of the proposed period is within the available extractions
        is_within_period(proposed_step, start_step, end_step, edge_steps)
        # checks that the proposed period does not fall too far outside the available extractions
        & (proposed_start_step + edge_steps >= start_step)
        & (proposed_end_step - edge_steps <= end_step)
        # checks that true valid_time is not too close to the edges of the proposed period
        & ((valid_step - buffer_steps) >= proposed_start_step)
        & ((valid_step + buffer_steps) <= proposed_end_step)
    )


def get_best_valid_time(
    row: pd.Series, valid_time_buffer: int, num_timesteps: int, freq: str
) -> Union[pd.Timestamp, float]:
    """
    Determines the best valid time for a given row of data based on specified shift constraints.

    This function evaluates potential valid times by shifting the original valid time
    forward or backward according to the values in 'valid_step_shift_forward' and
    'valid_step_shift_backward' fields. It ensures the shifted time remains within
    the period defined by 'start_date' and 'end_date' with sufficient buffer.

    Parameters
    ----------
    row : pd.Series
        A pandas Series containing the following fields:
        - valid_time: The original valid time
        - start_date: The start date of the allowed period
        - end_date: The end date of the allowed period
        - valid_step_shift_forward: Number of timesteps to shift forward
        - valid_step_shift_backward: Number of timesteps to shift backward
    valid_time_buffer : int
        Temporal buffer in months, determining how close we allow the true valid_time of the
        sample to be to the edge of the proposed processing period. For dekads, this is
        converted to timesteps (months * 3).

    num_timesteps : int
        The number of timesteps accepted by the model.
        This is used to define the middle of the user-defined period.

    freq : str
        Frequency alias for time series processing. Currently only "month" and "dekad"
        are supported.

    Returns
    -------
    datetime or np.nan
        The best valid time after applying shifts, or np.nan if no valid time can be found.
        If both forward and backward shifts are valid, the choice depends on the relative
        magnitude of the shifts compared to buffer.
    """

    # Run a check on provided valid_time_buffer
    if valid_time_buffer < 0:
        raise ValueError(
            f"The provided valid_time_buffer ({valid_time_buffer} months) must be a non-negative integer."
        )
    buffer_multiplier = 1 if freq == "month" else 3
    buffer_steps = valid_time_buffer * buffer_multiplier
    edge_steps = MIN_EDGE_BUFFER * buffer_multiplier
    if buffer_steps > num_timesteps // 2:
        logger.warning(
            f"The provided valid_time_buffer ({valid_time_buffer} months) is larger than half the number of timesteps in the processing period. "
            f"Reducing valid_time_buffer to half the number of timesteps: {num_timesteps // 2} timesteps"
        )
        buffer_steps = num_timesteps // 2

    # Extract necessary fields from the row
    valid_time = row["valid_time"]
    start_date = row["start_date"]
    end_date = row["end_date"]

    valid_step = _freq_step_index(valid_time, freq)
    start_step = _freq_step_index(start_date, freq)
    end_step = _freq_step_index(end_date, freq)
    proposed_step_fwd = valid_step + int(row["valid_step_shift_forward"])
    proposed_step_bwd = valid_step - int(row["valid_step_shift_backward"])

    shift_forward_ok = check_shift(
        proposed_step_fwd,
        valid_step,
        start_step,
        end_step,
        buffer_steps,
        num_timesteps,
        edge_steps,
    )
    shift_backward_ok = check_shift(
        proposed_step_bwd,
        valid_step,
        start_step,
        end_step,
        buffer_steps,
        num_timesteps,
        edge_steps,
    )

    if not shift_forward_ok and not shift_backward_ok:
        return np.nan
    if shift_forward_ok and not shift_backward_ok:
        return _index_to_ts(proposed_step_fwd, freq)
    if not shift_forward_ok and shift_backward_ok:
        return _index_to_ts(proposed_step_bwd, freq)
    if shift_forward_ok and shift_backward_ok:
        return (
            _index_to_ts(proposed_step_bwd, freq)
            if (row["valid_step_shift_backward"] - row["valid_step_shift_forward"])
            <= edge_steps
            else _index_to_ts(proposed_step_fwd, freq)
        )
    return np.nan


def _season_window_membership(
    timestamp: Optional[pd.Timestamp],
    *,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
    year_offset: int,
) -> bool:
    if timestamp is None or pd.isna(timestamp):
        return False

    ts = pd.Timestamp(timestamp)

    def _coerce(year: int, month: int, day: int) -> pd.Timestamp:
        safe_day = min(day, calendar.monthrange(year, month)[1])
        return pd.Timestamp(year=year, month=month, day=safe_day)

    if year_offset not in (0, 1):
        raise ValueError(
            "season_window only supports spans up to 12 months (year_offset must be 0 or 1)."
        )

    if year_offset == 0:
        start_year = ts.year
    else:
        ts_tuple = (ts.month, ts.day)
        end_tuple = (end_month, end_day)
        start_year = ts.year if ts_tuple > end_tuple else ts.year - 1

    start_dt = _coerce(start_year, start_month, start_day)
    end_dt = _coerce(start_year + year_offset, end_month, end_day)
    # Inclusive on both boundaries: samples on start_date and end_date are kept
    return start_dt <= ts <= end_dt


def process_extractions_df(
    df_raw: Union[pd.DataFrame, gpd.GeoDataFrame],
    processing_period: TemporalContext = None,
    freq: Literal["month", "dekad"] = "month",
    valid_time_buffer: int = 0,
    *,
    season_window: Optional[TemporalContext] = None,
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
    valid_time_buffer : int, default 0
        Buffer in months to apply when aligning available extractions with user-defined temporal extent.
        Determines how close we allow the true valid_time of the sample to be to the edge of the processing period.
        For dekads, this is converted to timesteps (months * 3).

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

    from worldcereal.utils.legend import ewoc_code_to_label
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
        processing_period_middle_step = _step_in_year(processing_period_middle_ts, freq)

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
        steps_per_year = STEPS_PER_YEAR[freq]
        sample_dates["valid_step_in_year"] = sample_dates["valid_time"].apply(
            lambda ts: _step_in_year(ts, freq)
        )
        sample_dates["valid_step_shift_backward"] = (
            sample_dates["valid_step_in_year"] - processing_period_middle_step
        ) % steps_per_year
        sample_dates["valid_step_shift_forward"] = (
            processing_period_middle_step - sample_dates["valid_step_in_year"]
        ) % steps_per_year
        sample_dates["proposed_valid_time"] = sample_dates.apply(
            get_best_valid_time,
            axis=1,
            valid_time_buffer=valid_time_buffer,
            num_timesteps=num_timesteps,
            freq=freq,
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

    # When processing_period is specified, trim to exact length (12 months or 36 dekads)
    # This ensures all samples have consistent available_timesteps for downstream processing
    max_trim = "auto" if processing_period is not None else None

    df_processed = process_parquet(
        df_raw,
        freq=freq,
        use_valid_time=True,
        max_timesteps_trim=max_trim,
    )

    if processing_period is not None:
        # put back the true valid_time
        df_processed["valid_time"] = df_processed.index.map(true_valid_time_map)
        df_processed["valid_time"] = df_processed["valid_time"].dt.strftime("%Y-%m-%d")

    if season_window is not None:
        season_start, season_end = season_window.to_datetime()
        if season_end < season_start:
            raise ValueError(
                "season_window end date must be greater than or equal to the start date."
            )
        year_offset = season_end.year - season_start.year
        if year_offset < 0 or year_offset > 1:
            raise ValueError("season_window may span at most 12 consecutive months.")

        valid_times = pd.to_datetime(df_processed["valid_time"], errors="coerce")
        missing_valid = valid_times.isna()
        if missing_valid.any():
            logger.warning(
                f"Dropping {int(missing_valid.sum())} samples without a valid_time while enforcing the manual season window."
            )

        in_window = valid_times.apply(
            lambda ts: _season_window_membership(
                ts,
                start_month=season_start.month,
                start_day=season_start.day,
                end_month=season_end.month,
                end_day=season_end.day,
                year_offset=year_offset,
            )
        )
        in_window &= ~missing_valid

        dropped = int((~in_window).sum())
        if dropped:
            logger.warning(
                f"Discarded {dropped} samples outside the season window "
                f"{season_start.strftime('%b %d')} -> {season_end.strftime('%b %d')}.",
            )
        df_processed = df_processed.loc[in_window].copy()
        if df_processed.empty:
            raise ValueError(
                "No samples remain inside the selected growing-season window. Try widening the window or revisiting your reference data selection."
            )

    # Enrich resulting dataframe with full and sampling string labels
    df_processed["label_full"] = ewoc_code_to_label(
        df_processed["ewoc_code"], label_type="full"
    )
    df_processed["sampling_label"] = ewoc_code_to_label(
        df_processed["ewoc_code"], label_type="sampling"
    )

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