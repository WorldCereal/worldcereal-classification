from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

STATIC_FEATURES = ["elevation", "slope", "lat", "lon"]
REQUIRED_COLUMNS = ["sample_id", "timestamp"] + STATIC_FEATURES
NODATAVALUE = 65535
MIN_EDGE_BUFFER = 2
NUM_TIMESTEPS = 12

BAND_MAPPINGS = {
    "10m": ["OPTICAL-B02", "OPTICAL-B03", "OPTICAL-B04", "OPTICAL-B08"],
    "20m": [
        "SAR-VH",
        "SAR-VV",
        "OPTICAL-B05",
        "OPTICAL-B06",
        "OPTICAL-B07",
        "OPTICAL-B11",
        "OPTICAL-B12",
        "OPTICAL-B8A",
    ],
    "100m": ["AGERA5-TMEAN", "AGERA5-PRECIP"],
}

FEATURE_COLUMNS = BAND_MAPPINGS["10m"] + BAND_MAPPINGS["20m"] + BAND_MAPPINGS["100m"]

COLUMN_RENAMES: Dict[str, str] = {
    "S1-SIGMA0-VV": "SAR-VV",
    "S1-SIGMA0-VH": "SAR-VH",
    "S2-L2A-B02": "OPTICAL-B02",
    "S2-L2A-B03": "OPTICAL-B03",
    "S2-L2A-B04": "OPTICAL-B04",
    "S2-L2A-B05": "OPTICAL-B05",
    "S2-L2A-B06": "OPTICAL-B06",
    "S2-L2A-B07": "OPTICAL-B07",
    "S2-L2A-B08": "OPTICAL-B08",
    "S2-L2A-B11": "OPTICAL-B11",
    "S2-L2A-B12": "OPTICAL-B12",
    "AGERA5-precipitation-flux": "AGERA5-PRECIP",
    "AGERA5-temperature-mean": "AGERA5-TMEAN",
    # since the openEO output has the attribute "valid_time",
    # # we need the following line for compatibility with earlier datasets
    "valid_date": "valid_time",
}

# Expected distances between observations for different frequencies, in days
EXPECTED_DISTANCES = {"month": 31, "dekad": 10}


class DataFrameValidator:
    @staticmethod
    def validate_required_columns(df_long: pd.DataFrame) -> None:
        missing_columns = [
            col for col in REQUIRED_COLUMNS if col not in df_long.columns
        ]
        if missing_columns:
            raise AttributeError(
                f"DataFrame must contain the following columns: {missing_columns}"
            )

    @staticmethod
    def validate_timestamps(df_long: pd.DataFrame, freq: str = "month") -> None:
        """
        Validate that timestamps in DataFrame adhere to specific frequency requirements.

        Parameters
        ----------
        df_long : pd.DataFrame
            DataFrame containing a 'timestamp' column with datetime objects.
        freq : str, default='month'
            Frequency to validate against. Currently supported values are:
            - 'month': requires timestamps at the beginning of each month
            - 'dekad': requires timestamps on the 1st, 11th, or 21st of each month

        Raises
        ------
        ValueError
            If any timestamp does not conform to the specified frequency pattern.
        NotImplementedError
            If the specified frequency is not supported.
        """

        if freq == "month":
            if not df_long["timestamp"].dt.is_month_start.all():
                bad_dates = df_long[~df_long["timestamp"].dt.is_month_start][
                    "timestamp"
                ].unique()
                raise ValueError(
                    f"All monthly timestamps must be at month start. Found: {bad_dates}"
                )
        elif freq == "dekad":
            if not df_long["timestamp"].dt.day.isin([1, 11, 21]).all():
                raise ValueError(
                    "All dekad timestamps must be at the 1st, 11th, or 21st of the month"
                )
        else:
            raise NotImplementedError(f"Frequency {freq} not supported")

    @staticmethod
    def check_faulty_samples(
        df_wide: pd.DataFrame, min_edge_buffer: int
    ) -> pd.DataFrame:
        """
        Filter out faulty samples from a DataFrame based on timestamp validations.

        This function evaluates if a valid center point can be established for each sample
        within the given edge buffer constraints.

        Parameters
        ----------
        df_wide : pd.DataFrame
            Input DataFrame containing at least the columns 'available_timesteps', 'valid_position'.
            - 'available_timesteps': The total number of timesteps available for a sample
            - 'valid_position': The position index that is considered valid

        min_edge_buffer : int
            Minimum required buffer from the edge of the time series. This ensures that
            windows created around valid positions maintain a minimum distance from the edges.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with faulty samples removed.

        Notes
        -----
        The function logs a warning message indicating how many faulty samples were dropped.
        """
        min_center_point = np.maximum(
            df_wide["available_timesteps"] // 2,
            df_wide["valid_position"]
            + min_edge_buffer
            - df_wide["available_timesteps"] // 2,
        )
        max_center_point = np.minimum(
            df_wide["available_timesteps"] - df_wide["available_timesteps"] // 2,
            df_wide["valid_position"]
            - min_edge_buffer
            + df_wide["available_timesteps"] // 2,
        )

        faulty_samples = min_center_point > max_center_point
        if faulty_samples.sum() > 0:
            logger.warning(f"Dropping {faulty_samples.sum()} faulty sample(s).")

        return df_wide[~faulty_samples]

    @staticmethod
    def check_min_timesteps(
        df_wide: pd.DataFrame, required_min_timesteps: int
    ) -> pd.DataFrame:
        """
        Filter out samples from a DataFrame that don't have the minimum required number of timesteps.

        Parameters
        ----------
        df_wide : pd.DataFrame
            DataFrame containing a column 'available_timesteps' that indicates the number of
            available timesteps for each sample.
        required_min_timesteps : int
            The minimum number of timesteps required for each sample to be kept.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only samples with at least the required minimum number of timesteps.

        Raises
        ------
        ValueError
            If all samples have fewer timesteps than required, resulting in an empty DataFrame.

        Notes
        -----
        The function logs a warning message indicating the number of samples dropped due to
        insufficient timesteps.
        """
        samples_with_too_few_ts = (
            df_wide["available_timesteps"] < required_min_timesteps
        )
        if samples_with_too_few_ts.sum() > 0:
            logger.warning(
                f"Dropping {samples_with_too_few_ts.sum()} sample(s) with \
number of available timesteps less than {required_min_timesteps}."
            )
        df_wide = df_wide[~samples_with_too_few_ts]
        if len(df_wide) == 0:
            raise ValueError(
                f"Left with an empty DataFrame! \
All samples have fewer timesteps than required ({required_min_timesteps})."
            )
        else:
            return df_wide

    @staticmethod
    def check_median_distance(df_long: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Validates if the temporal frequency of observations in the dataframe matches the expected frequency.

        This function computes the median time distance between observations for each sample and
        compares it with the expected distance for the given frequency. Samples with median distances
        that deviate significantly from the expected distance are removed from the dataset.

        Parameters
        ----------
        df_long : pd.DataFrame
            A long-format DataFrame containing at least 'sample_id' and 'timestamp' columns.
            The timestamp column should be in datetime format.

        freq : str
            The expected frequency of observations. Must be one of:
            - 'month': Monthly observations
            - 'dekad': Dekadal observations (10-day periods)

        Returns
        -------
        pd.DataFrame
            The input DataFrame with samples removed if their observation frequency
            doesn't match the expected frequency.

        Raises
        ------
        NotImplementedError
            If the specified frequency is not supported.
        ValueError
            If all samples are removed due to frequency mismatch.

        Notes
        -----
        The function allows for a tolerance of ±2 days from the expected distance.
        """

        ts_subset_df = (
            df_long[["sample_id", "timestamp"]]
            .drop_duplicates()
            .sort_values(by=["sample_id", "timestamp"])
        )
        ts_subset_df["timestamp_diff_days"] = (
            ts_subset_df.groupby("sample_id")["timestamp"].diff().dt.days.abs()
        )
        median_distance = (
            ts_subset_df.groupby("sample_id")["timestamp_diff_days"]
            .median()
            .fillna(0)
            .astype(int)
        )

        if freq == "month":
            expected_distance = EXPECTED_DISTANCES["month"]
            samples_with_mismatching_distance = median_distance[
                (median_distance < expected_distance - 2)
                | (median_distance > expected_distance + 2)
            ]
        elif freq == "dekad":
            expected_distance = EXPECTED_DISTANCES["dekad"]
            samples_with_mismatching_distance = median_distance[
                (median_distance < expected_distance - 2)
                | (median_distance > expected_distance + 2)
            ]
        else:
            raise NotImplementedError(f"Frequency {freq} not supported")

        if len(samples_with_mismatching_distance) > 0:
            logger.warning(
                f"Found {len(samples_with_mismatching_distance)} samples with median distance \
between observations not corresponding to {freq}. \
Removing them from the dataset."
            )
            df_long = df_long[
                ~df_long["sample_id"].isin(samples_with_mismatching_distance.index)
            ]
            if len(df_long) == 0:
                raise ValueError(
                    f"Left with an empty DataFrame! All samples have median distance between \
observations not corresponding to {freq}."
                )
        else:
            logger.info(
                f"Expected observations frequency: {freq}; \
Median observed distance between observations: {median_distance.unique()} days"
            )

        return df_long


class TimeSeriesProcessor:
    @staticmethod
    def calculate_valid_position(df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the valid position for each sample in the dataframe based on the minimum absolute
        time difference between 'valid_time' and 'timestamp'.

        This function adds two new columns to the input dataframe:
        1. 'valid_time_ts_diff_days': The absolute difference in days between 'valid_time' and 'timestamp'
        2. 'valid_position': Maps each sample_id to the timestamp_ind where the time difference is minimal

        Parameters
        ----------
        df_long : pd.DataFrame
            Input dataframe containing at least columns: 'sample_id', 'timestamp', 'valid_time', 'timestamp_ind'

        Returns
        -------
        pd.DataFrame
            The input dataframe with added 'valid_time_ts_diff_days' and 'valid_position' columns
        """

        df_long["valid_time_ts_diff_days"] = (
            df_long["valid_time"] - df_long["timestamp"]
        ).dt.days.abs()
        valid_position = (
            df_long.set_index("timestamp_ind")
            .groupby("sample_id")["valid_time_ts_diff_days"]
            .idxmin()
        )
        df_long["valid_position"] = df_long["sample_id"].map(valid_position)
        return df_long

    @staticmethod
    def get_expected_dates(start_date, end_date, freq):
        if freq == "month":
            return pd.date_range(start=start_date, end=end_date, freq="MS")
        elif freq == "dekad":
            return pd.DatetimeIndex(
                np.unique(
                    [
                        _dekad_startdate_from_date(xx)
                        for xx in _dekad_timestamps(start_date, end_date)
                    ]
                )
            )
        else:
            raise NotImplementedError(f"Frequency {freq} not supported")

    @staticmethod
    def fill_missing_dates(
        df_long: pd.DataFrame, freq: str, index_columns: List[str]
    ) -> pd.DataFrame:
        """
        Fill missing dates in a DataFrame with NODATAVALUE for feature columns.

        This function identifies samples with missing observations based on the expected
        frequency (monthly or dekadal) and fills in the gaps with NODATAVALUE.

        Parameters
        ----------
        df_long : pd.DataFrame
            Input DataFrame in long format containing time series data.
            Must have 'start_date', 'end_date', 'timestamp', and 'sample_id' columns.
        freq : str
            Frequency of the time series data. Supported values are:
            - 'month': Monthly data with timestamps at the start of each month
            - 'dekad': Dekadal data (10-day periods)
        index_columns : List[str]
            List of column names that uniquely identify each sample and should be
            preserved when filling missing dates.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same structure as the input but with rows added for
            any missing timestamps, where feature columns are filled with NODATAVALUE.

        Notes
        -----
        - If all samples already have the expected number of observations, the original
          DataFrame is returned unchanged.
        - The function preserves all metadata columns specified in index_columns.
        - Expected timestamps are determined based on the start_date and end_date for each sample.

        Raises
        ------
        NotImplementedError
            If a frequency other than 'month' or 'dekad' is provided.
        """

        def fill_sample(sample_df):
            expected_dates = TimeSeriesProcessor.get_expected_dates(
                sample_df["start_date"].iloc[0], sample_df["end_date"].iloc[0], freq
            )
            missing_dates = expected_dates.difference(sample_df["timestamp"])
            if not missing_dates.empty:
                static_cols = sample_df.iloc[0][index_columns].to_dict()
                for date in missing_dates:
                    new_row = {**static_cols, "timestamp": date}
                    for col in FEATURE_COLUMNS:
                        new_row[col] = NODATAVALUE
                    sample_df.loc[-1] = new_row
                    sample_df.reset_index(drop=True, inplace=True)
            return sample_df

        unique_date_pairs = df_long[["start_date", "end_date"]].drop_duplicates()
        unique_date_pairs["expected_n_observations"] = unique_date_pairs.apply(
            lambda xx: len(
                TimeSeriesProcessor.get_expected_dates(
                    xx["start_date"], xx["end_date"], freq
                )
            ),
            axis=1,
        )
        unique_date_pairs.set_index(["start_date", "end_date"], inplace=True)

        expected_observations_s = df_long.groupby(
            ["sample_id", "start_date", "end_date"]
        )[["sample_id", "start_date", "end_date", "timestamp"]].apply(
            lambda xx: xx["timestamp"].nunique()
        )
        expected_observations_s.name = "actual_n_observations"
        expected_observations_df = expected_observations_s.reset_index()
        expected_observations_df.set_index(["start_date", "end_date"], inplace=True)

        expected_observations_df["expected_n_observations"] = (
            expected_observations_df.index.map(
                unique_date_pairs["expected_n_observations"]
            )
        )
        expected_observations_df.reset_index(drop=False, inplace=True)

        samples_to_fill = expected_observations_df[
            expected_observations_df["actual_n_observations"]
            != expected_observations_df["expected_n_observations"]
        ]["sample_id"].unique()

        if samples_to_fill.size == 0:
            logger.info("All samples have the expected number of observations.")
            return df_long
        else:
            logger.warning(
                f"{len(samples_to_fill)} samples have missing observations. \
Filling them with NODATAVALUE."
            )
            df_subset = df_long[df_long["sample_id"].isin(samples_to_fill)]
            df_long = df_long[~df_long["sample_id"].isin(samples_to_fill)]
            df_subset = (
                df_subset.groupby("sample_id")[
                    [*index_columns, *FEATURE_COLUMNS, "timestamp"]
                ]
                .apply(fill_sample)
                .reset_index(drop=True)
            )
            return pd.concat([df_long, df_subset], ignore_index=True)

    @staticmethod
    def add_dummy_timestamps(
        df_long: pd.DataFrame, min_edge_buffer: int, freq: str
    ) -> pd.DataFrame:
        """
        Add dummy timestamps to a time series DataFrame to ensure minimum buffer before and after valid observations.

        This function adds dummy timestamps with NODATAVALUE for features to ensure there are at least
        `min_edge_buffer` timestamps before the first valid observation and after the last valid observation
        for each sample. Samples with valid times outside the start and end dates will be removed.

        Parameters
        ----------
        df_long : pd.DataFrame
            Longitudinal DataFrame containing time series data with columns including 'sample_id',
            'timestamp', 'valid_position', 'timestamp_ind', and feature columns.
        min_edge_buffer : int
            Minimum number of timestamps required as buffer before the first valid observation
            and after the last valid observation.
        freq : str
            Frequency of time series data, either 'month' or 'dekad' (10-day period).

        Returns
        -------
        pd.DataFrame
            DataFrame with added dummy timestamps where needed to satisfy the minimum edge buffer
            requirement. The returned DataFrame has recalculated timestamp indices, start/end dates,
            and valid positions.

        Notes
        -----
        - The function will remove samples where the valid time is before the start date or after the end date.
        - For each sample requiring buffer, dummy rows with NODATAVALUE for features are added.
        - The function recalculates temporal indices and positions after adding dummy timestamps.
        """

        def create_dummy_rows(samples_to_add, n_ts_to_add, direction, freq):
            dummy_df = df_long[
                df_long["sample_id"].isin(samples_to_add)
                & (
                    df_long["timestamp_ind"]
                    == (0 if direction == "before" else df_long["timestamp_ind"].max())
                )
            ].copy()

            if freq == "month":
                offset = pd.DateOffset(
                    months=n_ts_to_add * (1 if direction == "after" else -1)
                )
                dummy_df["timestamp"] += offset
            elif freq == "dekad":
                offset = pd.DateOffset(
                    days=n_ts_to_add * (10 if direction == "after" else -10)
                )
                dummy_df["timestamp"] = dummy_df["timestamp"] + offset
                dummy_df["timestamp"] = dummy_df["timestamp"].apply(
                    _dekad_startdate_from_date
                )

            # dummy_df["timestamp"] += offset
            dummy_df[FEATURE_COLUMNS] = NODATAVALUE
            return dummy_df

        latest_obs_position = df_long.groupby("sample_id")[
            ["valid_position", "timestamp_ind", "valid_position_diff"]
        ].max()

        samples_after_end_date = latest_obs_position[
            latest_obs_position["valid_position"] > latest_obs_position["timestamp_ind"]
        ].index.tolist()
        samples_before_start_date = latest_obs_position[
            latest_obs_position["valid_position"] < 0
        ].index.tolist()

        if (len(samples_after_end_date) > 0) or (len(samples_before_start_date) > 0):
            logger.warning(
                f"Removing {len(samples_after_end_date)} samples with valid_time \
after the end_date and {len(samples_before_start_date)} samples with valid_time \
before the start_date"
            )
            df_long = df_long[
                ~df_long["sample_id"].isin(
                    samples_before_start_date + samples_after_end_date
                )
            ]

        intermediate_dummy_df = pd.concat(
            [
                create_dummy_rows(
                    latest_obs_position[
                        (min_edge_buffer - latest_obs_position["valid_position"])
                        >= -n_ts_to_add
                    ].index,
                    n_ts_to_add,
                    "before",
                    freq,
                )
                for n_ts_to_add in range(1, min_edge_buffer)
            ]
            + [
                create_dummy_rows(
                    latest_obs_position[
                        (min_edge_buffer - latest_obs_position["valid_position_diff"])
                        >= n_ts_to_add
                    ].index,
                    n_ts_to_add,
                    "after",
                    freq,
                )
                for n_ts_to_add in range(1, min_edge_buffer)
            ]
        )

        if not intermediate_dummy_df.empty:
            logger.warning(
                f"Added {intermediate_dummy_df['timestamp'].nunique()} dummy timestamp(s) \
for {intermediate_dummy_df['sample_id'].nunique()} samples to fill in the found gaps."
            )

        df_long = pd.concat([df_long, intermediate_dummy_df])

        # re-initilize all dates and positions with respect to potentially added new timestamps
        df_long["timestamp_ind"] = (
            df_long.groupby("sample_id")["timestamp"].rank().astype(int) - 1
        )
        df_long["start_date"] = df_long.groupby("sample_id")["timestamp"].transform(
            "min"
        )
        df_long["end_date"] = df_long.groupby("sample_id")["timestamp"].transform("max")
        df_long = TimeSeriesProcessor.calculate_valid_position(df_long)

        return df_long


class ColumnProcessor:
    """
    ColumnProcessor contains static methods that standardize column names and structure,
    handle missing data, and perform validation on DataFrame columns.

    Methods:
        rename_columns: Rename DataFrame columns according to predefined mappings.
        add_band_suffix: Add band-specific suffixes to column names in wide-format DataFrames.
        construct_index: Identify and return columns to be used as an index in the DataFrame.
        check_feature_columns: Validate that all required feature columns are present.
        check_sar_columns: Validate and correct SAR data to prevent invalid values.
    """

    @staticmethod
    def rename_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        return df_long.rename(columns=COLUMN_RENAMES)

    @staticmethod
    def add_band_suffix(df_wide: pd.DataFrame) -> pd.DataFrame:
        df_wide.columns = [
            f"{xx[0]}-ts{xx[1]}" if isinstance(xx[1], int) else xx[0]
            for xx in df_wide.columns.to_flat_index()
        ]  # type: ignore
        df_wide.columns = [
            f"{xx}-10m" if any(band in xx for band in BAND_MAPPINGS["10m"]) else xx
            for xx in df_wide.columns
        ]  # type: ignore
        df_wide.columns = [
            f"{xx}-20m" if any(band in xx for band in BAND_MAPPINGS["20m"]) else xx
            for xx in df_wide.columns
        ]  # type: ignore
        df_wide.columns = [
            (f"{xx}-100m" if any(band in xx for band in BAND_MAPPINGS["100m"]) else xx)
            for xx in df_wide.columns
        ]  # type: ignore
        return df_wide

    @staticmethod
    def construct_index(df_long: pd.DataFrame) -> List[str]:
        # for index columns we need to include all columns that are not feature columns
        index_columns = [col for col in df_long.columns if col not in FEATURE_COLUMNS]
        index_columns.remove("timestamp")
        return index_columns

    @staticmethod
    def check_feature_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        # check that all feature columns are present in the DataFrame
        # or initialize them with NODATAVALUE
        missing_features = [
            col for col in FEATURE_COLUMNS if col not in df_long.columns
        ]
        if len(missing_features) > 0:
            df_long[missing_features] = NODATAVALUE
            logger.warning(
                f"The following features are missing and are filled \
with NODATAVALUE: {missing_features}"
            )
        return df_long

    @staticmethod
    def check_sar_columns(df_long: pd.DataFrame) -> pd.DataFrame:
        # SAR cannot equal 0.0 since we take the log of it
        # TO DO: need to check the behavior of presto itself in this case
        sar_cols = ["SAR-VV", "SAR-VH"]
        faulty_sar_observations = (df_long[sar_cols] == 0.0).sum().sum()
        if faulty_sar_observations > 0:
            affected_samples = df_long[(df_long[sar_cols] == 0.0).any(axis=1)][
                "sample_id"
            ].nunique()
            logger.warning(
                f"Found {faulty_sar_observations} SAR observation(s) \
equal to 0 across {affected_samples} sample(s). \
Replacing them with NODATAVALUE."
            )
            df_long[sar_cols] = df_long[sar_cols].replace(0.0, NODATAVALUE)

        return df_long


def _dekad_timestamps(begin, end):
    """Creates a temporal sequence on a dekadal basis.
    Returns end date for each dekad.
    Based on: https://pytesmo.readthedocs.io/en/7.1/_modules/pytesmo/timedate/dekad.html  # NOQA

    Parameters
    ----------
    begin : datetime
        Datetime index start date.
    end : datetime, optional
        Datetime index end date, set to current date if None.

    Returns
    -------
    dtindex : pandas.DatetimeIndex
        Dekadal datetime index.
    """

    import calendar

    daterange = generate_month_sequence(begin, end)

    dates = []

    for i, dat in enumerate(daterange):
        year, month = int(str(dat)[:4]), int(str(dat)[5:7])
        lday = calendar.monthrange(year, month)[1]
        if i == 0 and begin.day > 1:
            if begin.day < 11:
                if daterange.size == 1:
                    if end.day < 11:
                        dekads = [10]
                    elif end.day >= 11 and end.day < 21:
                        dekads = [10, 20]
                    else:
                        dekads = [10, 20, lday]
                else:
                    dekads = [10, 20, lday]
            elif begin.day >= 11 and begin.day < 21:
                if daterange.size == 1:
                    if end.day < 21:
                        dekads = [20]
                    else:
                        dekads = [20, lday]
                else:
                    dekads = [20, lday]
            else:
                dekads = [lday]
        elif i == (len(daterange) - 1) and end.day < 21:
            if end.day < 11:
                dekads = [10]
            else:
                dekads = [10, 20]
        else:
            dekads = [10, 20, lday]

        for j in dekads:
            dates.append(datetime(year, month, j))

    return dates


def _dekad_startdate_from_date(dt_in):
    """
    dekadal startdate that a date falls in
    Based on: https://pytesmo.readthedocs.io/en/7.1/_modules/pytesmo/timedate/dekad.html  # NOQA

    Parameters
    ----------
    run_dt: datetime.datetime

    Returns
    -------
    startdate: datetime.datetime
        startdate of dekad
    """
    if dt_in.day <= 10:
        startdate = datetime(dt_in.year, dt_in.month, 1, 0, 0, 0)
    if dt_in.day >= 11 and dt_in.day <= 20:
        startdate = datetime(dt_in.year, dt_in.month, 11, 0, 0, 0)
    if dt_in.day >= 21:
        startdate = datetime(dt_in.year, dt_in.month, 21, 0, 0, 0)
    return startdate


def generate_month_sequence(start_date: datetime, end_date: datetime) -> np.ndarray:
    """Helper function to generate a sequence of months between start_date and end_date.
    This is much faster than using a pd.date_range().

    Parameters
    ----------
    start_date : datetime
        start of the sequence
    end_date : datetime
        end of the sequence

    Returns
    -------
    array contaning the sequence of months

    """
    start = np.datetime64(start_date, "M")  # Truncate to month start
    end = np.datetime64(end_date, "M")  # Truncate to month start
    timestamps = np.arange(start, end + 1, dtype="datetime64[M]")

    return timestamps


def process_parquet(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    freq: Literal["month", "dekad", "MS", "10D"] = "month",
    use_valid_time: bool = True,
    required_min_timesteps: Optional[int] = None,
    min_edge_buffer: int = 2,  # only used if valid_time is used
    return_after_fill: bool = False,  # added for debugging purposes
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Process a DataFrame or GeoDataFrame with time series data by filling missing timestamps,
    transforming to wide format, and validating the data.

    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        Input DataFrame or GeoDataFrame containing time series data.
        Must include 'sample_id' and 'timestamp' columns.
    freq : Literal["month", "dekad", "MS", "10D"], default="month"
        Frequency of the time series data.
        "MS" is an alias for "month" and "10D" is an alias for "dekad".
    use_valid_time : bool, default=True
        Whether to calculate and use valid time positions in the time series.
    required_min_timesteps : Optional[int], default=None
        Minimum number of timesteps required for each sample.
        Samples with fewer timesteps will be filtered out.
    min_edge_buffer : int, default=2
        Minimum number of timesteps to include as buffer at the edges
        when calculating valid positions. Only used if use_valid_time is True.
    return_after_fill : bool, default=False
        If True, returns the DataFrame after filling missing dates
        but before pivoting. Used for debugging purposes.

    Returns
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]
        Processed DataFrame in wide format with features spread across columns
        by timestamp index. Index is set to 'sample_id'.

    Raises
    ------
    ValueError
        If the input DataFrame is empty or if the resulting DataFrame after processing is empty.

    Notes
    -----
    The function performs several operations including:
    - Validating required columns and timestamps
    - Checking and processing feature columns
    - Filling in missing dates according to the specified frequency
    - Converting from long to wide format
    - Adding suffixes to band names for clarity
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty!")

    if freq == "MS":
        freq = "month"
    if freq == "10D":
        freq = "dekad"

    # Validate input
    validator = DataFrameValidator()
    validator.validate_required_columns(df)
    validator.validate_timestamps(df, freq)
    df = validator.check_median_distance(df, freq)

    # Process columns
    df = (
        df.pipe(ColumnProcessor.rename_columns)
        .pipe(ColumnProcessor.check_feature_columns)
        .pipe(ColumnProcessor.check_sar_columns)
    )
    index_columns = ColumnProcessor.construct_index(df)

    # Assign start_date and end_date as the minimum and maximum available timestamp
    df["start_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].min())
    df["end_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].max())
    index_columns.extend(["start_date", "end_date"])
    index_columns = list(set(index_columns))

    # Process time series
    processor = TimeSeriesProcessor()
    df = processor.fill_missing_dates(df, freq, index_columns)
    if return_after_fill:
        return df

    # Initialize timestep_ind
    df["timestamp_ind"] = df.groupby("sample_id")["timestamp"].rank().astype(int) - 1

    if use_valid_time:
        df = processor.calculate_valid_position(df)
        index_columns.append("valid_position")
        df["valid_position_diff"] = df["timestamp_ind"] - df["valid_position"]
        df = processor.add_dummy_timestamps(df, min_edge_buffer, freq)

    df["available_timesteps"] = df["sample_id"].map(
        df.groupby("sample_id")["timestamp"].nunique().astype(int)
    )
    index_columns.append("available_timesteps")
    index_columns = list(set(index_columns))

    # Transform to wide format
    df_pivot = df.pivot(
        index=index_columns,
        columns="timestamp_ind",
        values=FEATURE_COLUMNS,
    )

    df_pivot = df_pivot.fillna(NODATAVALUE)
    if df_pivot.empty:
        raise ValueError("Left with an empty DataFrame!")

    df_pivot.reset_index(inplace=True)
    df_pivot = ColumnProcessor.add_band_suffix(df_pivot)

    if use_valid_time:
        df_pivot["year"] = df_pivot["valid_time"].dt.year
        df_pivot["valid_time"] = df_pivot["valid_time"].dt.date.astype(str)
        df_pivot = validator.check_faulty_samples(df_pivot, min_edge_buffer)

    if required_min_timesteps:
        df_pivot = validator.check_min_timesteps(df_pivot, required_min_timesteps)

    df_pivot["start_date"] = df_pivot["start_date"].dt.date.astype(str)
    df_pivot["end_date"] = df_pivot["end_date"].dt.date.astype(str)
    df_pivot = df_pivot.set_index("sample_id")

    return df_pivot
