from datetime import datetime
from typing import Dict, List, Literal, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from prometheo.predictors import NODATAVALUE

from worldcereal.train.datasets import MIN_EDGE_BUFFER

STATIC_FEATURES = ["elevation", "slope", "lat", "lon"]
REQUIRED_COLUMNS = ["sample_id", "timestamp"] + STATIC_FEATURES

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
    "100m": ["METEO-temperature_mean", "METEO-precipitation_flux"],
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
    "S2-L2A-B8A": "OPTICAL-B8A",
    "S2-L2A-B11": "OPTICAL-B11",
    "S2-L2A-B12": "OPTICAL-B12",
    "AGERA5-PRECIP": "METEO-precipitation_flux",
    "AGERA5-TMEAN": "METEO-temperature_mean",
    "slope": "DEM-slo-20m",
    "elevation": "DEM-alt-20m",
}

# Expected distances between observations for different frequencies, in days
EXPECTED_DISTANCES = {"month": 31, "dekad": 10}


class DataFrameValidator:
    @staticmethod
    def validate_and_fix_dt_cols(df_long: pd.DataFrame) -> None:
        # make sure the timestamp and valid_time are datetime objects with no timezone
        df_long["timestamp"] = pd.to_datetime(df_long["timestamp"])
        df_long["timestamp"] = df_long["timestamp"].dt.tz_localize(None)
        df_long["timestamp"] = df_long["timestamp"].dt.floor("D")  # trim to date only
        if "valid_time" in df_long.columns:
            df_long["valid_time"] = pd.to_datetime(df_long["valid_time"])
            df_long["valid_time"] = df_long["valid_time"].dt.tz_localize(None)
        return df_long

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

        validtime_outside_range = (
            (df_wide["valid_position"] < 0)
            | (df_wide["valid_position"] >= df_wide["available_timesteps"])
            | (df_wide["valid_time"] > df_wide["end_date"])
            | (df_wide["valid_time"] < df_wide["start_date"])
        )
        faulty_samples = (
            min_center_point > max_center_point
        ) & ~validtime_outside_range

        # best effort to identify the dataset being processed, purely for logging
        ref_id = "_".join(df_wide["sample_id"].iloc[0].split("_")[:-1])

        if validtime_outside_range.sum() > 0:
            logger.warning(
                f"{ref_id}: Dropping {validtime_outside_range.sum()} samples with valid_time outside the expected range. \n"
                f"Reason: Valid time must be within the range of available timesteps. Samples with the following dates are affected:\n"
                f"{df_wide[validtime_outside_range][['start_date', 'valid_time', 'end_date']].drop_duplicates().to_string(index=False)}"
            )

        if faulty_samples.sum() > 0:
            logger.warning(
                f"{ref_id}: Dropping {faulty_samples.sum()} faulty sample(s). \n"
                f"Reason: Could not establish a valid center point with the given \n"
                f"min_edge_buffer of {min_edge_buffer} for the following date ranges:\n"
                f"{df_wide[faulty_samples][['start_date', 'valid_time', 'end_date']].drop_duplicates().to_string(index=False)}"
            )

        df_wide = df_wide[~(faulty_samples | validtime_outside_range)]

        return df_wide

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

        # best effort to identify the dataset being processed, purely for logging
        ref_id = "_".join(df_wide["sample_id"].iloc[0].split("_")[:-1])
        if samples_with_too_few_ts.sum() > 0:
            logger.warning(
                f"{ref_id}: Dropping {samples_with_too_few_ts.sum()} sample(s) with \
number of available timesteps less than {required_min_timesteps}."
            )
        df_wide = df_wide[~samples_with_too_few_ts]
        if len(df_wide) == 0:
            raise ValueError(
                f"{ref_id}: Left with an empty DataFrame! \
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

        # best effort to identify the dataset being processed, purely for logging
        ref_id = "_".join(df_long["sample_id"].iloc[0].split("_")[:-1])
        if len(samples_with_mismatching_distance) > 0:
            logger.warning(
                f"{ref_id}: Found {len(samples_with_mismatching_distance)} samples with median distance \
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
                f"{ref_id}: Expected observations frequency: {freq}; \
Median observed distance between observations: {median_distance.unique()} days"
            )

        return df_long


class TimeSeriesProcessor:
    """
    TimeSeriesProcessor contains methods for common time series data processing tasks,
    particularly focused on handling temporal observations with varying frequencies
    and managing missing data points.

    Methods
    calculate_valid_position(df_long)
        Calculate the valid position for each sample based on minimum absolute time difference
        between valid time and timestamp.

    get_expected_dates(start_date, end_date, freq)
        Generate a sequence of expected timestamps between start and end dates based on
        the specified frequency.

    fill_missing_dates(df_long, freq, index_columns)
        Fill missing dates in a DataFrame with NODATAVALUE for feature columns based
        on expected frequency.

    check_vt_closeness(df_long, min_edge_buffer, freq)
        Check valid_time closeness to the edges of the time series and remove samples
        that do not satisfy the minimum edge buffer requirement.

    The class is designed to work with pandas DataFrames containing time series data
    with specific expected columns including 'sample_id', 'timestamp', and temporal
    indicators.

    Currently supports 'month' and 'dekad' (10-day periods) frequencies.
    """

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

        # best effort to identify the dataset being processed, purely for logging
        ref_id = "_".join(df_long["sample_id"].iloc[0].split("_")[:-1])

        if samples_to_fill.size == 0:
            logger.info(
                f"{ref_id}: All samples have the expected number of observations."
            )
            return df_long
        else:
            logger.warning(
                f"{ref_id}: {len(samples_to_fill)} samples have missing observations. \
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
    def check_vt_closeness(
        df_long: pd.DataFrame, min_edge_buffer: int, freq: str
    ) -> pd.DataFrame:
        """
        Check valid_time closeness to the edges of the time series.
        Essential for downstream processing at the Dataset level.
        Samples that fail this check will be removed.

        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format DataFrame containing time series data with columns including 'sample_id',
            'timestamp', 'valid_position', 'timestamp_ind', and feature columns.
        min_edge_buffer : int
            Minimum number of timestamps required as buffer before the first valid observation
            and after the last valid observation.
            Must be consistent with what is used at the Dataset level.
        freq : str
            Frequency of time series data, either 'month' or 'dekad' (10-day period).

        Returns
        -------
        pd.DataFrame
            DataFrame with only those samples that satisfy the minimum edge buffer
            requirement.
        """

        if df_long.empty:
            return df_long

        # Summarise per sample to evaluate distance from the valid time to window edges.
        summary = (
            df_long.groupby("sample_id")
            .agg(
                start_date=("start_date", "first"),
                end_date=("end_date", "first"),
                valid_time=("valid_time", "first"),
                first_timestamp_ind=("timestamp_ind", "min"),
                last_timestamp_ind=("timestamp_ind", "max"),
                valid_position=("valid_position", "first"),
            )
            .copy()
        )

        summary["distance_to_start"] = (
            summary["valid_position"] - summary["first_timestamp_ind"]
        )
        summary["distance_to_end"] = (
            summary["last_timestamp_ind"] - summary["valid_position"]
        )

        faulty_end = summary[summary["distance_to_end"] < min_edge_buffer]

        # best effort to identify the dataset being processed, purely for logging
        ref_id = "_".join(df_long["sample_id"].iloc[0].split("_")[:-1])
        if not faulty_end.empty:
            logger.warning(
                f"{ref_id}: Dropping {len(faulty_end)} samples with valid_time too close to the end of the time series. \n"
                f"Reason: Minimum edge buffer of {min_edge_buffer} not satisfied. Samples with the following date ranges are affected:\n"
                f"{faulty_end[['start_date', 'valid_time', 'end_date']].drop_duplicates().to_string(index=False)}"
            )

        remaining_summary = summary.drop(index=faulty_end.index, errors="ignore")
        faulty_start = remaining_summary[
            remaining_summary["distance_to_start"] < min_edge_buffer
        ]
        if not faulty_start.empty:
            logger.warning(
                f"{ref_id}: Dropping {len(faulty_start)} samples with valid_time too close to the start of the time series. \n"
                f"Reason: Minimum edge buffer of {min_edge_buffer} not satisfied. Samples with the following date ranges are affected:\n"
                f"{faulty_start[['start_date', 'valid_time', 'end_date']].drop_duplicates().to_string(index=False)}"
            )

        to_drop = faulty_end.index.union(faulty_start.index)
        if to_drop.empty:
            logger.info(
                f"{ref_id}: All samples' valid_time satisfy the min_edge_buffer requirement."
            )
            return df_long

        return df_long[~df_long["sample_id"].isin(to_drop)]


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
        # best effort to identify the dataset being processed, purely for logging
        ref_id = "_".join(df_long["sample_id"].iloc[0].split("_")[:-1])
        if len(missing_features) > 0:
            df_long[missing_features] = NODATAVALUE
            logger.warning(
                f"{ref_id}: The following features are missing and are filled \
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
            # best effort to identify the dataset being processed, purely for logging
            ref_id = "_".join(df_long["sample_id"].iloc[0].split("_")[:-1])
            affected_samples = df_long[(df_long[sar_cols] == 0.0).any(axis=1)][
                "sample_id"
            ].nunique()
            logger.warning(
                f"{ref_id}: Found {faulty_sar_observations} SAR observation(s) \
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


def _trim_timesteps(
    df: pd.DataFrame,
    max_timesteps_trim: Union[int, None, tuple[str, str]],
    required_min_timesteps: int,
    min_edge_buffer: int,
    use_valid_time: bool,
    freq: str,
) -> pd.DataFrame:
    """Trim per-sample timesteps to reduce width prior to pivot.

    Centering strategy:
      - If use_valid_time: center window on valid_position.
      - Else: center window on midpoint of series (floor(available/2)).

    After trimming, timestamp_ind, start/end_date, and (if applicable) valid_position
    + valid_position_diff are recomputed.
    """
    import gc

    if max_timesteps_trim is None:
        return df

    # Resolve 'auto'
    if isinstance(max_timesteps_trim, str):
        if max_timesteps_trim.lower() == "auto":
            max_timesteps_trim = required_min_timesteps + 2 * min_edge_buffer
        else:
            raise ValueError(
                "Unsupported string for max_timesteps_trim. Use 'auto' or provide an int."
            )
    
    if isinstance(max_timesteps_trim, tuple):  # type: ignore
        # check that tuple elements are valid dates
        trim_start, trim_end = max_timesteps_trim  # type: ignore
        try:
            trim_start = pd.to_datetime(trim_start)
            trim_end = pd.to_datetime(trim_end)
        except Exception as e:
            raise ValueError("If max_timesteps_trim is a tuple, it must contain valid date strings.") from e
        if trim_start >= trim_end:
            raise ValueError("In max_timesteps_trim tuple, start date must be before end date.")
        # filter df to only include timestamps within the specified range
        df = df[(df["timestamp"] >= trim_start) & (df["timestamp"] <= trim_end)]
        # After filtering, we need to ensure that all samples still meet the required minimum timesteps
        sample_counts = df["sample_id"].value_counts()
        samples_too_few = sample_counts[sample_counts < required_min_timesteps].index
        if len(samples_too_few) > 0 and len(samples_too_few) < len(sample_counts):
            logger.warning(
                f"Dropping {len(samples_too_few)} samples that have fewer than the required minimum timesteps ({required_min_timesteps}) after applying max_timesteps_trim date range."
            )
            df = df[~df["sample_id"].isin(samples_too_few)]
        elif len(samples_too_few) == len(sample_counts):
            raise ValueError(
                f"All samples have fewer than the required minimum timesteps ({required_min_timesteps}) after applying max_timesteps_trim date range {trim_start.strftime('%Y-%m-%d')} - {trim_end.strftime('%Y-%m-%d')}. Check your date range."
            )
        else:
            logger.info(
                f"Applied max_timesteps_trim date range {trim_start.strftime('%Y-%m-%d')} - {trim_end.strftime('%Y-%m-%d')}. All remaining samples meet the required minimum timesteps ({required_min_timesteps})."
            )
        # Recompute basics
        df["timestamp_ind"] = df.groupby("sample_id")["timestamp"].rank().astype(int) - 1
        df["start_date"] = df.groupby("sample_id")["timestamp"].transform("min")
        df["end_date"] = df.groupby("sample_id")["timestamp"].transform("max")
        if use_valid_time:
            df = TimeSeriesProcessor.calculate_valid_position(df)
            df["valid_position_diff"] = df["timestamp_ind"] - df["valid_position"]
        return df

    if not isinstance(max_timesteps_trim, int):  # type: ignore
        raise TypeError("max_timesteps_trim must be int, 'auto', or None")
    
    if max_timesteps_trim < required_min_timesteps:
        raise ValueError(
            f"max_timesteps_trim ({max_timesteps_trim}) cannot be smaller than required minimum timesteps ({required_min_timesteps})."
        )

    def _trim_sample(g: pd.DataFrame) -> pd.DataFrame:
        total_ts = g["timestamp_ind"].nunique()
        if total_ts <= max_timesteps_trim:  # nothing to trim
            return []
        if use_valid_time:
            center = int(g["valid_position"].iloc[0])
        else:
            center = total_ts // 2
        half = max_timesteps_trim // 2
        left = center - half
        right = left + max_timesteps_trim - 1
        if left < 0:
            right += -left
            left = 0
        if right >= total_ts:
            shift = right - (total_ts - 1)
            right = total_ts - 1
            left = max(0, left - shift)
        window_size = right - left + 1
        if window_size > max_timesteps_trim:
            # remove excess from side furthest from center
            if right - center > center - left:
                right -= (window_size - max_timesteps_trim)
            else:
                left += (window_size - max_timesteps_trim)
        index_to_drop = g.index[~((g["timestamp_ind"] >= left) & (g["timestamp_ind"] <= right))].to_list()

        return index_to_drop

    before_unique = df["timestamp_ind"].nunique()

    ind_cols = ["sample_id", "timestamp_ind"]
    if use_valid_time:
        ind_cols.append("valid_position")
        
    inds_to_drop = df[ind_cols].groupby("sample_id", group_keys=False).apply(_trim_sample).values
    inds_to_drop = np.concatenate(inds_to_drop)
    df.drop(index=inds_to_drop, inplace=True)
    gc.collect()

    # Recompute basics
    df["timestamp_ind"] = df.groupby("sample_id")["timestamp"].rank().astype(int) - 1
    df["start_date"] = df.groupby("sample_id")["timestamp"].transform("min")
    df["end_date"] = df.groupby("sample_id")["timestamp"].transform("max")
    if use_valid_time:
        df = TimeSeriesProcessor.calculate_valid_position(df)
        df["valid_position_diff"] = df["timestamp_ind"] - df["valid_position"]

    after_unique = df["timestamp_ind"].nunique()
    if after_unique < before_unique:
        logger.info(
            f"Trimmed timesteps (global unique indices {before_unique} -> {after_unique}) using max_timesteps_trim={max_timesteps_trim}."
        )
    return df


def process_parquet(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    freq: Literal["month", "dekad"] = "month",
    use_valid_time: bool = True,
    min_edge_buffer: int = MIN_EDGE_BUFFER,  # only used if valid_time is used
    return_after_fill: bool = False,  # added for debugging purposes
    max_timesteps_trim: Union[int, str, None] = None,  # optionally trim width before pivot
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Process a DataFrame or GeoDataFrame with time series data by filling missing timestamps,
    transforming to wide format, and validating the data.

    Parameters
    ----------
    df : Union[pd.DataFrame, gpd.GeoDataFrame]
        Input DataFrame or GeoDataFrame containing time series data.
        Must include 'sample_id' and 'timestamp' columns.
    freq : Literal["month", "dekad"], default="month"
        Frequency of the time series data.
    use_valid_time : bool, default=True
        Whether to calculate and use valid time positions in the time series.
    min_edge_buffer : int, default = MIN_EDGE_BUFFER
        Minimum number of timesteps to include as buffer at the edges
        when calculating valid positions. Only used if use_valid_time is True.
    return_after_fill : bool, default=False
        If True, returns the DataFrame after filling missing dates
        but before pivoting. Used for debugging purposes.
    max_timesteps_trim : Union[int, str, None], default=None
        Optional maximum number of timesteps to retain (after dummy timestamp handling) per sample
        prior to pivoting. When provided, each sample's observations are trimmed to a centered
        window around the valid position (if use_valid_time=True) or around the series midpoint
        (if use_valid_time=False). Accepted values:
          - None: (default) No trimming; preserves current behaviour.
          - 'auto': Uses required_min_timesteps + 2 * min_edge_buffer.
          - int: Explicit maximum; must be >= required_min_timesteps.
        After trimming, timestamp indices, start/end dates, and (if applicable) valid_position
        and derived relative positions are recomputed. This reduces memory footprint for global
        models where very long sequences are unnecessary.

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

    # Determine required minimum timesteps based on frequency
    if freq == "dekad":
        required_min_timesteps = 36
    if freq == "month":
        required_min_timesteps = 12

    # `feature_index` is an openEO spefic column we should remove to avoid
    # it being treated as unique values which is not true after merging
    # multiple parquet files
    if "feature_index" in df.columns:
        df = df.drop("feature_index", axis=1)

    # Validate input
    validator = DataFrameValidator()
    validator.validate_required_columns(df)
    validator.validate_and_fix_dt_cols(df)
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

    # Perform check on number of unique values for each index column
    nsamples = df["sample_id"].nunique()
    to_drop = []
    for col in index_columns:
        if df[col].nunique() > nsamples:
            # best effort to identify the dataset being processed, purely for logging
            ref_id = "_".join(df["sample_id"].iloc[0].split("_")[:-1])
            df = df.drop(col, axis=1)
            to_drop.append(col)
            logger.warning(
                f"{ref_id}: Column {col} has more unique values than samples. This may cause issues, column has been dropped!"
            )
    index_columns = [col for col in index_columns if col not in to_drop]

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
        df = processor.check_vt_closeness(df, min_edge_buffer, freq)

    if max_timesteps_trim is not None:
        logger.info(
            f"Trimming to max_timesteps_trim={max_timesteps_trim} per sample prior to pivot."
        )
        df = _trim_timesteps(
            df=df,
            max_timesteps_trim=max_timesteps_trim,
            required_min_timesteps=required_min_timesteps,
            min_edge_buffer=min_edge_buffer,
            use_valid_time=use_valid_time,
            freq=freq,
        )

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
        df_pivot["valid_time"] = df_pivot["valid_time"].dt.strftime("%Y-%m-%d")
        df_pivot = validator.check_faulty_samples(df_pivot, min_edge_buffer)

    df_pivot = validator.check_min_timesteps(df_pivot, required_min_timesteps)

    df_pivot["start_date"] = df_pivot["start_date"].dt.strftime("%Y-%m-%d")
    df_pivot["end_date"] = df_pivot["end_date"].dt.strftime("%Y-%m-%d")

    df_pivot = df_pivot.set_index("sample_id")

    return df_pivot
