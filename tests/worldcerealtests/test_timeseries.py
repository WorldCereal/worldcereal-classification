from unittest import TestCase

import numpy as np
import pandas as pd

from worldcereal.utils.timeseries import (
    FEATURE_COLUMNS,
    _dekad_startdate_from_date,
    _dekad_timestamps,
    process_parquet,
)

MIN_EDGE_BUFFER = 2
NODATAVALUE = 65535


class TestProcessParquet(TestCase):
    def setUp(self):
        self.allowed_freqs = ["month", "dekad"]
        # Sample DataFrame setup with monthly timestamps
        self.start_date = pd.to_datetime("2020-10-01")
        self.end_date = pd.to_datetime("2022-06-01")
        dekad_range = np.array(
            [
                _dekad_startdate_from_date(t)
                for t in _dekad_timestamps(self.start_date, self.end_date)
            ]
        )
        month_range = pd.date_range(start=self.start_date, end=self.end_date, freq="MS")
        self.n_dekads = len(dekad_range)
        self.n_months = len(month_range)
        self.min_timesteps_month = 12
        self.min_timesteps_dekad = 36

        # normal case
        sample_1_data_month = {
            "sample_id": ["sample_1"] * self.n_months,
            "timestamp": month_range,
            "start_date": [self.start_date] * self.n_months,
            "valid_time": [self.start_date + pd.DateOffset(months=9)] * self.n_months,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "slope": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_months),
            "AGERA5-PRECIP": np.random.randint(100, size=self.n_months),
            "AGERA5-TMEAN": np.random.randint(30, size=self.n_months),
            "CROPTYPE_LABEL": [1200] * self.n_months,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_months,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_months,
        }

        # valid_time close to start_date
        sample_2_data_month = {
            "sample_id": ["sample_2"] * self.n_months,
            "timestamp": month_range,
            "start_date": [self.start_date] * self.n_months,
            "valid_time": [
                self.start_date + pd.DateOffset(months=(MIN_EDGE_BUFFER // 2))
            ]
            * self.n_months,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "slope": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_months),
            "AGERA5-PRECIP": np.random.randint(100, size=self.n_months),
            "AGERA5-TMEAN": np.random.randint(30, size=self.n_months),
            "CROPTYPE_LABEL": [1310] * self.n_months,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_months,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_months,
        }

        # valid_time close to end_date
        sample_3_data_month = {
            "sample_id": ["sample_3"] * self.n_months,
            "timestamp": month_range,
            "start_date": [self.start_date] * self.n_months,
            "valid_time": [self.end_date - pd.DateOffset(months=(MIN_EDGE_BUFFER // 2))]
            * self.n_months,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "slope": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_months),
            "AGERA5-PRECIP": np.random.randint(100, size=self.n_months),
            "AGERA5-TMEAN": np.random.randint(30, size=self.n_months),
            "CROPTYPE_LABEL": [1104] * self.n_months,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_months,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_months,
        }

        # valid_time outside range of extractions
        sample_4_data_month = {
            "sample_id": ["sample_4"] * self.n_months,
            "timestamp": month_range,
            "start_date": [self.start_date] * self.n_months,
            "valid_time": [self.end_date + pd.DateOffset(months=(MIN_EDGE_BUFFER + 1))]
            * self.n_months,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "slope": [np.random.randint(1000, size=1)[0]] * self.n_months,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_months),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_months),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_months),
            "AGERA5-PRECIP": np.random.randint(100, size=self.n_months),
            "AGERA5-TMEAN": np.random.randint(30, size=self.n_months),
            "CROPTYPE_LABEL": [1102] * self.n_months,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_months,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_months,
        }

        # valid_date too close to end_date, with little room to maneuver
        sample_5_data_month = {
            "sample_id": ["sample_5"] * self.min_timesteps_month,
            "timestamp": pd.date_range(start=self.start_date, end=(self.start_date+pd.DateOffset(months=self.min_timesteps_month-1)), freq="MS"),
            "start_date": [self.start_date] * self.min_timesteps_month,
            "valid_time": [self.start_date+pd.DateOffset(months=self.min_timesteps_month-1)]
            * self.min_timesteps_month,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.min_timesteps_month,
            "slope": [np.random.randint(1000, size=1)[0]] * self.min_timesteps_month,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.min_timesteps_month),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B02": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B03": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B04": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B05": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B06": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B07": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B08": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B11": np.random.randint(1000, size=self.min_timesteps_month),
            "S2-L2A-B12": np.random.randint(1000, size=self.min_timesteps_month),
            "AGERA5-PRECIP": np.random.randint(100, size=self.min_timesteps_month),
            "AGERA5-TMEAN": np.random.randint(30, size=self.min_timesteps_month),
            "CROPTYPE_LABEL": [1102] * self.min_timesteps_month,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.min_timesteps_month,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.min_timesteps_month,
        }

        # not enough timesteps
        sample_6_data_month = {
            "sample_id": ["sample_6"] * (self.min_timesteps_month-1),
            "timestamp": pd.date_range(start=self.start_date, end=(self.start_date+pd.DateOffset(months=self.min_timesteps_month-2)), freq="MS"),
            "start_date": [self.start_date] * (self.min_timesteps_month-1),
            "valid_time": [self.start_date + pd.DateOffset(months=9)]
            * (self.min_timesteps_month-1),
            "elevation": [np.random.randint(1000, size=1)[0]] * (self.min_timesteps_month-1),
            "slope": [np.random.randint(1000, size=1)[0]] * (self.min_timesteps_month-1),
            "S1-SIGMA0-VV": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S1-SIGMA0-VH": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B02": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B03": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B04": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B05": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B06": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B07": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B08": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B11": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "S2-L2A-B12": np.random.randint(1000, size=(self.min_timesteps_month-1)),
            "AGERA5-PRECIP": np.random.randint(100, size=(self.min_timesteps_month-1)),
            "AGERA5-TMEAN": np.random.randint(30, size=(self.min_timesteps_month-1)),
            "CROPTYPE_LABEL": [1102] * (self.min_timesteps_month-1),
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * (self.min_timesteps_month-1),
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * (self.min_timesteps_month-1),
        }

        self.df_month = pd.concat(
            (
                pd.DataFrame(sample_1_data_month),
                pd.DataFrame(sample_2_data_month),
                pd.DataFrame(sample_3_data_month),
                pd.DataFrame(sample_4_data_month),
                pd.DataFrame(sample_5_data_month),
                pd.DataFrame(sample_6_data_month),
            )
        )
        self.df_month = self.df_month.fillna(NODATAVALUE).reset_index(drop=True)

        # Sample DataFrame setup with dekadal timestamps
        sample_1_data_dekad = {
            "sample_id": ["sample_1"] * self.n_dekads,
            "timestamp": dekad_range,
            "start_date": [self.start_date] * self.n_dekads,
            "valid_time": [pd.to_datetime("2021-07-05")] * self.n_dekads,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.n_dekads,
            "slope": [np.random.randint(1000, size=1)[0]] * self.n_dekads,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_dekads),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_dekads),
            "AGERA5-PRECIP": np.random.randint(100, size=self.n_dekads),
            "AGERA5-TMEAN": np.random.randint(30, size=self.n_dekads),
            "CROPTYPE_LABEL": [1200] * self.n_dekads,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_dekads,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_dekads,
        }

        # valid_time close to start_date
        sample_2_data_dekad = {
            "sample_id": ["sample_2"] * self.n_dekads,
            "timestamp": dekad_range,
            "start_date": [self.start_date] * self.n_dekads,
            "valid_time": [
                _dekad_startdate_from_date(
                    pd.to_datetime(self.start_date)
                    + pd.DateOffset(days=10 * (MIN_EDGE_BUFFER // 2))
                )
            ]
            * self.n_dekads,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.n_dekads,
            "slope": [np.random.randint(1000, size=1)[0]] * self.n_dekads,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_dekads),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_dekads),
            "AGERA5-PRECIP": np.random.randint(100, size=self.n_dekads),
            "AGERA5-TMEAN": np.random.randint(30, size=self.n_dekads),
            "CROPTYPE_LABEL": [1310] * self.n_dekads,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_dekads,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_dekads,
        }

        # valid_time close to end_date
        sample_3_data_dekad = {
            "sample_id": ["sample_3"] * self.n_dekads,
            "timestamp": dekad_range,
            "start_date": [self.start_date] * self.n_dekads,
            "valid_time": [
                _dekad_startdate_from_date(
                    pd.to_datetime(self.end_date)
                    - pd.DateOffset(days=10 * (MIN_EDGE_BUFFER // 2))
                )
            ]
            * self.n_dekads,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.n_dekads,
            "slope": [np.random.randint(1000, size=1)[0]] * self.n_dekads,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_dekads),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_dekads),
            "AGERA5-PRECIP": np.random.randint(100, size=self.n_dekads),
            "AGERA5-TMEAN": np.random.randint(30, size=self.n_dekads),
            "CROPTYPE_LABEL": [1104] * self.n_dekads,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_dekads,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_dekads,
        }

        # valid_time outside range of extractions
        sample_4_data_dekad = {
            "sample_id": ["sample_4"] * self.n_dekads,
            "timestamp": dekad_range,
            "start_date": [self.start_date] * self.n_dekads,
            "valid_time": [
                _dekad_startdate_from_date(
                    pd.to_datetime(self.end_date)
                    + pd.DateOffset(days=10 * (MIN_EDGE_BUFFER + 1))
                )
            ]
            * self.n_dekads,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.n_dekads,
            "slope": [np.random.randint(1000, size=1)[0]] * self.n_dekads,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.n_dekads),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B02": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B03": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B04": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B05": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B06": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B07": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B08": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B11": np.random.randint(1000, size=self.n_dekads),
            "S2-L2A-B12": np.random.randint(1000, size=self.n_dekads),
            "AGERA5-PRECIP": np.random.randint(100, size=self.n_dekads),
            "AGERA5-TMEAN": np.random.randint(30, size=self.n_dekads),
            "CROPTYPE_LABEL": [1102] * self.n_dekads,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.n_dekads,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.n_dekads,
        }

        # valid_date too close to end_date, with little room to maneuver
        sample_5_data_dekad = {
            "sample_id": ["sample_5"] * self.min_timesteps_dekad,
            "timestamp": [
                _dekad_startdate_from_date(t)
                for t in _dekad_timestamps(self.start_date, (self.start_date+pd.DateOffset(days=10*self.min_timesteps_dekad)))
            ],
            "start_date": [self.start_date] * self.min_timesteps_dekad,
            "valid_time": [
                _dekad_startdate_from_date(
                    pd.to_datetime(self.end_date)
                    + pd.DateOffset(days=10 * (MIN_EDGE_BUFFER + 1))
                )
            ]
            * self.min_timesteps_dekad,
            "elevation": [np.random.randint(1000, size=1)[0]] * self.min_timesteps_dekad,
            "slope": [np.random.randint(1000, size=1)[0]] * self.min_timesteps_dekad,
            "S1-SIGMA0-VV": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S1-SIGMA0-VH": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B02": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B03": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B04": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B05": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B06": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B07": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B08": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B11": np.random.randint(1000, size=self.min_timesteps_dekad),
            "S2-L2A-B12": np.random.randint(1000, size=self.min_timesteps_dekad),
            "AGERA5-PRECIP": np.random.randint(100, size=self.min_timesteps_dekad),
            "AGERA5-TMEAN": np.random.randint(30, size=self.min_timesteps_dekad),
            "CROPTYPE_LABEL": [1102] * self.min_timesteps_dekad,
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * self.min_timesteps_dekad,
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * self.min_timesteps_dekad,
        }

        # not enough timesteps; for dekad, we need to subtract at least 3 steps,
        # since months with incomplete dekads are still rounded up to full months
        sample_6_data_dekad = {
            "sample_id": ["sample_6"] * (self.min_timesteps_dekad-3),
            "timestamp": [
                _dekad_startdate_from_date(t)
                for t in _dekad_timestamps(self.start_date, (self.start_date+pd.DateOffset(days=10*(self.min_timesteps_dekad-3))))
            ],
            "start_date": [self.start_date] * (self.min_timesteps_dekad-3),
            "valid_time": [
                _dekad_startdate_from_date(
                    pd.to_datetime(self.start_date)
                    + pd.DateOffset(days=10 * (MIN_EDGE_BUFFER // 2))
                )
            ]
            * (self.min_timesteps_dekad-3),
            "elevation": [np.random.randint(1000, size=1)[0]] * (self.min_timesteps_dekad-3),
            "slope": [np.random.randint(1000, size=1)[0]] * (self.min_timesteps_dekad-3),
            "S1-SIGMA0-VV": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S1-SIGMA0-VH": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B02": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B03": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B04": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B05": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B06": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B07": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B08": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B11": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "S2-L2A-B12": np.random.randint(1000, size=(self.min_timesteps_dekad-3)),
            "AGERA5-PRECIP": np.random.randint(100, size=(self.min_timesteps_dekad-3)),
            "AGERA5-TMEAN": np.random.randint(30, size=(self.min_timesteps_dekad-3)),
            "CROPTYPE_LABEL": [1102] * (self.min_timesteps_dekad-3),
            "lat": [np.random.uniform(-90, 90, size=1)[0]] * (self.min_timesteps_dekad-3),
            "lon": [np.random.uniform(-180, 180, size=1)[0]] * (self.min_timesteps_dekad-3),
        }

        self.df_dekad = pd.concat(
            (
                pd.DataFrame(sample_1_data_dekad),
                pd.DataFrame(sample_2_data_dekad),
                pd.DataFrame(sample_3_data_dekad),
                pd.DataFrame(sample_4_data_dekad),
                pd.DataFrame(sample_5_data_dekad),
                pd.DataFrame(sample_6_data_dekad),
            )
        )
        self.df_dekad = self.df_dekad.fillna(NODATAVALUE).reset_index(drop=True)

    def test_process_parquet_valid_input(self):
        for freq in self.allowed_freqs:
            result = process_parquet(
                (
                    self.df_month
                    if freq == "month"
                    else self.df_dekad if freq == "dekad" else None
                ),
                freq=freq,
                use_valid_time=True,
                min_edge_buffer=MIN_EDGE_BUFFER,
            )

            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)
            self.assertIn("OPTICAL-B02-ts0-10m", result.columns)
            self.assertIn("SAR-VV-ts0-20m", result.columns)
            self.assertIn("METEO-temperature_mean-ts0-100m", result.columns)

    def test_process_parquet_missing_timestamps(self):
        for freq in self.allowed_freqs:
            df = (
                self.df_month
                if freq == "month"
                else self.df_dekad if freq == "dekad" else None
            )

            # Remove n timestamps to create missing timestamps scenario
            # Make sure not to remove first or last timestamp for each sample
            n = 20
            df["start_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].min())
            df["end_date"] = df["sample_id"].map(df.groupby(["sample_id"])["timestamp"].max())
            rows_to_remove = df[
                (df["timestamp"] != df["start_date"])
                & (df["timestamp"] != df["end_date"])
                & (df["timestamp"] != df["valid_time"])
                & (df["timestamp"] != df["valid_time"])
            ].sample(n)

            df_missing = df.drop(rows_to_remove.index)
            result = process_parquet(
                df_missing,
                freq=freq,
                use_valid_time=True,
                min_edge_buffer=MIN_EDGE_BUFFER,
                return_after_fill=True,
            )
            self.assertIsInstance(result, pd.DataFrame)
            for ii, trow in rows_to_remove.iterrows():
                row_to_check = result[
                    (result["sample_id"] == trow["sample_id"])
                    & (result["timestamp"] == trow["timestamp"])
                ]
                self.assertFalse(row_to_check.empty)
                self.assertTrue(
                    (row_to_check[FEATURE_COLUMNS] == NODATAVALUE).values.all()
                )

    def test_process_parquet_validtime_outside_range(self):
        for freq in self.allowed_freqs:
            df = (
                self.df_month
                if freq == "month"
                else self.df_dekad if freq == "dekad" else None
            )
            result = process_parquet(
                df,
                freq=freq,
                use_valid_time=True,
                min_edge_buffer=MIN_EDGE_BUFFER,
            )
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse("sample_4" in result.index.unique())
    
    def test_process_parquet_wild_timestamps(self):
        # TODO: Implement test when wild timestamp(s) are injected into the dataframe
        for freq in self.allowed_freqs:
            test_df = (
                self.df_month
                if freq == "month"
                else self.df_dekad if freq == "dekad" else None
            )

            # Add n wild timestamp(s) to the dataframe
            # Make sure not to add any of the existing timestamps
            n_samples = 3
            bag_of_wild_timestamps_inside = pd.date_range(
                start=self.start_date, end=self.end_date, freq="D"
            )
            bag_of_wild_timestamps_inside = [
                xx
                for xx in bag_of_wild_timestamps_inside
                if xx not in test_df["timestamp"].unique()
            ]
            wild_timestamps_inside = np.random.choice(
                bag_of_wild_timestamps_inside, n_samples
            )

            dummy_df = test_df.sample(n_samples)
            dummy_df["timestamp"] = wild_timestamps_inside

            test_df = pd.concat([test_df, dummy_df], ignore_index=True)

            with self.assertRaises(ValueError):
                process_parquet(
                    test_df,
                    freq=freq,
                    use_valid_time=True,
                    min_edge_buffer=MIN_EDGE_BUFFER,
                )

    def test_process_parquet_valid_time_close_to_start(self):
        for freq in self.allowed_freqs:
            result = process_parquet(
                (
                    self.df_month
                    if freq == "month"
                    else self.df_dekad if freq == "dekad" else None
                ),
                freq=freq,
                use_valid_time=True,
                min_edge_buffer=MIN_EDGE_BUFFER,
            )
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)

            if freq == "month":
                expected_start_date = pd.to_datetime(self.start_date) + pd.DateOffset(
                    months=-(MIN_EDGE_BUFFER // 2 + 1)
                )
                expected_available_timesteps = self.n_months + MIN_EDGE_BUFFER
            elif freq == "dekad":
                expected_start_date = _dekad_startdate_from_date(
                    pd.to_datetime(self.start_date)
                    + pd.DateOffset(days=-10 * (MIN_EDGE_BUFFER // 2 + 1))
                )
                expected_available_timesteps = self.n_dekads + MIN_EDGE_BUFFER
            
            obtained_start_date = pd.to_datetime(result.loc["sample_2", "start_date"])
            obtained_available_timesteps = result.loc["sample_2", "available_timesteps"]

            self.assertTrue(obtained_start_date == expected_start_date)
            self.assertTrue(
                obtained_available_timesteps == expected_available_timesteps
            )

    def test_process_parquet_valid_time_close_to_end(self):
        for freq in self.allowed_freqs:
            result = process_parquet(
                (
                    self.df_month
                    if freq == "month"
                    else self.df_dekad if freq == "dekad" else None
                ),
                freq=freq,
                use_valid_time=True,
                min_edge_buffer=MIN_EDGE_BUFFER,
            )
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)

            if freq == "month":
                expected_last_timestep = pd.to_datetime(self.end_date) + pd.DateOffset(
                    months=(MIN_EDGE_BUFFER // 2)
                )
                expected_available_timesteps = self.n_months + MIN_EDGE_BUFFER - 1
            elif freq == "dekad":
                expected_last_timestep = _dekad_startdate_from_date(
                    pd.to_datetime(self.end_date)
                    + pd.DateOffset(days=10 * (MIN_EDGE_BUFFER // 2))
                )
                expected_available_timesteps = self.n_dekads + MIN_EDGE_BUFFER - 1

            obtained_last_timestep = pd.to_datetime(result.loc["sample_3", "end_date"])
            obtained_available_timesteps = result.loc["sample_3", "available_timesteps"]

            self.assertTrue(obtained_last_timestep == expected_last_timestep)
            self.assertTrue(
                obtained_available_timesteps == expected_available_timesteps
            )

    def test_process_parquet_invalid_input(self):
        for freq in self.allowed_freqs:
            result = process_parquet(
                (
                    self.df_month
                    if freq == "month"
                    else self.df_dekad if freq == "dekad" else None
                ),
                freq=freq,
                use_valid_time=True,
                min_edge_buffer=MIN_EDGE_BUFFER,
            )
            self.assertFalse("sample_4" in result.index.unique())

    def test_empty_dataframe(self):
        """Test processing empty dataframe"""
        # No need to test all frequencies
        empty_df = pd.DataFrame(columns=self.df_month.columns)
        with self.assertRaises(ValueError):
            process_parquet(empty_df)

    def test_invalid_timestamp_freq(self):
        """Test invalid timestamp frequency"""
        # No need to test all frequencies
        with self.assertRaises(NotImplementedError):
            process_parquet(self.df_month, freq="weekly")

    def test_all_zero_sar_values(self):
        """Test handling of all zero SAR values"""
        # No need to test all frequencies
        test_df = self.df_month.copy()
        test_df["S1-SIGMA0-VV"] = 0.0
        test_df["S1-SIGMA0-VH"] = 0.0

        result = process_parquet(test_df)
        self.assertTrue((result["SAR-VV-ts0-20m"] == NODATAVALUE).all())
        self.assertTrue((result["SAR-VH-ts0-20m"] == NODATAVALUE).all())

    def test_missing_required_columns(self):
        """Test missing required columns"""
        # No need to test all frequencies
        test_df = self.df_month.drop(columns=["lat"])
        with self.assertRaises(AttributeError):
            process_parquet(test_df)

    def test_minimum_timesteps_requirement(self):
        """Test minimum timesteps requirement"""
        for freq in self.allowed_freqs:
            result = process_parquet(
                (
                    self.df_month
                    if freq == "month"
                    else self.df_dekad if freq == "dekad" else None
                ),
                freq=freq,
                min_edge_buffer=MIN_EDGE_BUFFER,
            )
            self.assertFalse("sample_6" in result.index.unique())

    def test_valid_position_calculation(self):
        """Test valid position calculation"""
        for freq in self.allowed_freqs:
            result = process_parquet(
                (
                    self.df_month
                    if freq == "month"
                    else self.df_dekad if freq == "dekad" else None
                ),
                freq=freq,
                use_valid_time=True,
                min_edge_buffer=MIN_EDGE_BUFFER,
            )
            self.assertIn("valid_position", result.columns)
            self.assertTrue((result["valid_position"] >= 0).all())
            # additional assert that all values are consecutive?

    def test_feature_columns_initialization(self):
        """Test initialization of missing feature columns"""
        # No need to test all frequencies
        test_df = self.df_month.drop(columns=["S2-L2A-B08"])
        result = process_parquet(test_df)
        self.assertIn("OPTICAL-B08-ts0-10m", result.columns)
        self.assertTrue((result["OPTICAL-B08-ts0-10m"] == NODATAVALUE).all())

    def test_band_suffix_addition(self):
        """Test correct band suffix addition"""
        # No need to test all frequencies
        result = process_parquet(self.df_dekad, freq="dekad")

        # Check 10m bands
        self.assertTrue(any(col.endswith("10m") for col in result.columns))
        # Check 20m bands
        self.assertTrue(any(col.endswith("20m") for col in result.columns))
        # Check 100m bands
        self.assertTrue(any(col.endswith("100m") for col in result.columns))

    def test_non_month_start_timestamps(self):
        """Test handling of non-month-start timestamps"""
        test_df = self.df_month.copy()
        test_df["timestamp"] = test_df["timestamp"] + pd.Timedelta(days=15)

        with self.assertRaises(ValueError):
            process_parquet(test_df)

    def test_date_conversions(self):
        """Test date format conversions"""
        result = process_parquet(self.df_month)

        # Check date string formats
        self.assertTrue(all(isinstance(d, str) for d in result["start_date"]))
        self.assertTrue(all(isinstance(d, str) for d in result["end_date"]))
        self.assertTrue(all(isinstance(d, str) for d in result["valid_time"]))

    def test_datetime_handling(self):
        """Test handling of datetime objects in timestamp columns"""
        test_df = self.df_month.copy()
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"]).dt.tz_localize('UTC')
        test_df["valid_time"] = pd.to_datetime(test_df["valid_time"]).dt.strftime('%Y-%m-%d')

        result = process_parquet(test_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)