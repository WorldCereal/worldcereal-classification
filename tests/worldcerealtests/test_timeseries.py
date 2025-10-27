from unittest import TestCase

import numpy as np
import pandas as pd
from prometheo.predictors import NODATAVALUE

from worldcereal.train.datasets import MIN_EDGE_BUFFER
from worldcereal.utils.timeseries import (
    FEATURE_COLUMNS,
    _dekad_startdate_from_date,
    _dekad_timestamps,
    process_parquet,
)


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
            "timestamp": pd.date_range(
                start=self.start_date,
                end=(
                    self.start_date + pd.DateOffset(months=self.min_timesteps_month - 1)
                ),
                freq="MS",
            ),
            "start_date": [self.start_date] * self.min_timesteps_month,
            "valid_time": [
                self.start_date + pd.DateOffset(months=self.min_timesteps_month - 1)
            ]
            * self.min_timesteps_month,
            "elevation": [np.random.randint(1000, size=1)[0]]
            * self.min_timesteps_month,
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
            "sample_id": ["sample_6"] * (self.min_timesteps_month - 1),
            "timestamp": pd.date_range(
                start=self.start_date,
                end=(
                    self.start_date + pd.DateOffset(months=self.min_timesteps_month - 2)
                ),
                freq="MS",
            ),
            "start_date": [self.start_date] * (self.min_timesteps_month - 1),
            "valid_time": [self.start_date + pd.DateOffset(months=9)]
            * (self.min_timesteps_month - 1),
            "elevation": [np.random.randint(1000, size=1)[0]]
            * (self.min_timesteps_month - 1),
            "slope": [np.random.randint(1000, size=1)[0]]
            * (self.min_timesteps_month - 1),
            "S1-SIGMA0-VV": np.random.randint(
                1000, size=(self.min_timesteps_month - 1)
            ),
            "S1-SIGMA0-VH": np.random.randint(
                1000, size=(self.min_timesteps_month - 1)
            ),
            "S2-L2A-B02": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "S2-L2A-B03": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "S2-L2A-B04": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "S2-L2A-B05": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "S2-L2A-B06": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "S2-L2A-B07": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "S2-L2A-B08": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "S2-L2A-B11": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "S2-L2A-B12": np.random.randint(1000, size=(self.min_timesteps_month - 1)),
            "AGERA5-PRECIP": np.random.randint(
                100, size=(self.min_timesteps_month - 1)
            ),
            "AGERA5-TMEAN": np.random.randint(30, size=(self.min_timesteps_month - 1)),
            "CROPTYPE_LABEL": [1102] * (self.min_timesteps_month - 1),
            "lat": [np.random.uniform(-90, 90, size=1)[0]]
            * (self.min_timesteps_month - 1),
            "lon": [np.random.uniform(-180, 180, size=1)[0]]
            * (self.min_timesteps_month - 1),
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
                for t in _dekad_timestamps(
                    self.start_date,
                    (
                        self.start_date
                        + pd.DateOffset(days=10 * self.min_timesteps_dekad)
                    ),
                )
            ],
            "start_date": [self.start_date] * self.min_timesteps_dekad,
            "valid_time": [
                _dekad_startdate_from_date(
                    pd.to_datetime(self.end_date)
                    + pd.DateOffset(days=10 * (MIN_EDGE_BUFFER + 1))
                )
            ]
            * self.min_timesteps_dekad,
            "elevation": [np.random.randint(1000, size=1)[0]]
            * self.min_timesteps_dekad,
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
            "sample_id": ["sample_6"] * (self.min_timesteps_dekad - 3),
            "timestamp": [
                _dekad_startdate_from_date(t)
                for t in _dekad_timestamps(
                    self.start_date,
                    (
                        self.start_date
                        + pd.DateOffset(days=10 * (self.min_timesteps_dekad - 3))
                    ),
                )
            ],
            "start_date": [self.start_date] * (self.min_timesteps_dekad - 3),
            "valid_time": [
                _dekad_startdate_from_date(
                    pd.to_datetime(self.start_date)
                    + pd.DateOffset(days=10 * (MIN_EDGE_BUFFER // 2))
                )
            ]
            * (self.min_timesteps_dekad - 3),
            "elevation": [np.random.randint(1000, size=1)[0]]
            * (self.min_timesteps_dekad - 3),
            "slope": [np.random.randint(1000, size=1)[0]]
            * (self.min_timesteps_dekad - 3),
            "S1-SIGMA0-VV": np.random.randint(
                1000, size=(self.min_timesteps_dekad - 3)
            ),
            "S1-SIGMA0-VH": np.random.randint(
                1000, size=(self.min_timesteps_dekad - 3)
            ),
            "S2-L2A-B02": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "S2-L2A-B03": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "S2-L2A-B04": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "S2-L2A-B05": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "S2-L2A-B06": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "S2-L2A-B07": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "S2-L2A-B08": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "S2-L2A-B11": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "S2-L2A-B12": np.random.randint(1000, size=(self.min_timesteps_dekad - 3)),
            "AGERA5-PRECIP": np.random.randint(
                100, size=(self.min_timesteps_dekad - 3)
            ),
            "AGERA5-TMEAN": np.random.randint(30, size=(self.min_timesteps_dekad - 3)),
            "CROPTYPE_LABEL": [1102] * (self.min_timesteps_dekad - 3),
            "lat": [np.random.uniform(-90, 90, size=1)[0]]
            * (self.min_timesteps_dekad - 3),
            "lon": [np.random.uniform(-180, 180, size=1)[0]]
            * (self.min_timesteps_dekad - 3),
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

            # Remove a single eligible timestamp per sample to create missing timestamps scenario
            # without forcing the sample to be dropped by the median-distance check.
            df["start_date"] = df["sample_id"].map(
                df.groupby(["sample_id"])["timestamp"].min()
            )
            df["end_date"] = df["sample_id"].map(
                df.groupby(["sample_id"])["timestamp"].max()
            )

            rows_to_remove_idx = []
            for sample_id, sample_df in df.groupby("sample_id"):
                eligible_idx = sample_df[
                    (sample_df["timestamp"] != sample_df["start_date"])
                    & (sample_df["timestamp"] != sample_df["end_date"])
                    & (sample_df["timestamp"] != sample_df["valid_time"])
                ].index
                if not eligible_idx.empty:
                    rows_to_remove_idx.append(np.random.choice(eligible_idx))

            rows_to_remove = df.loc[rows_to_remove_idx]

            df_missing = df.drop(rows_to_remove_idx)
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

            self.assertNotIn("sample_2", result.index.unique())

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

            self.assertNotIn("sample_3", result.index.unique())

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

    def test_max_timesteps_trim_auto(self):
        """Test automatic trimming reduces width while keeping required minimum timesteps."""
        # choose dekad to have larger initial potential width
        result_no_trim = process_parquet(
            self.df_dekad, freq="dekad", use_valid_time=True
        )
        # expect many timestep columns (feature ts indices). Count SAR-VV entries as proxy
        wide_cols_no_trim = [
            c for c in result_no_trim.columns if c.startswith("SAR-VV-ts")
        ]  # after suffixing
        n_no_trim = len(wide_cols_no_trim)

        result_trim = process_parquet(
            self.df_dekad,
            freq="dekad",
            use_valid_time=True,
            max_timesteps_trim="auto",
        )
        wide_cols_trim = [c for c in result_trim.columns if c.startswith("SAR-VV-ts")]
        n_trim = len(wide_cols_trim)

        # auto should trim when original > required_min_timesteps + 2 * MIN_EDGE_BUFFER
        self.assertTrue(n_trim <= n_no_trim)
        self.assertTrue(n_trim >= self.min_timesteps_dekad)

        # valid_position must remain within [0, available_timesteps)
        self.assertTrue((result_trim["valid_position"] >= 0).all())
        self.assertTrue(
            (result_trim["valid_position"] < result_trim["available_timesteps"]).all()
        )

    def test_max_timesteps_trim_explicit(self):
        """Test explicit trimming window produces expected available_timesteps upper bound."""
        target_max = 20
        result_trim = process_parquet(
            self.df_month,
            freq="month",
            use_valid_time=True,
            max_timesteps_trim=target_max,
        )
        self.assertTrue((result_trim["available_timesteps"] <= target_max).all())
        # Still meets required minimum
        self.assertTrue(
            (result_trim["available_timesteps"] >= self.min_timesteps_month).all()
        )

    def test_max_timesteps_trim_date_range(self):
        """Test trimming within a specific date range."""
        trim_start = pd.Timestamp("2021-01-01")
        trim_end = pd.Timestamp("2022-01-31")
        result_trim = process_parquet(
            self.df_month,
            freq="month",
            use_valid_time=True,
            max_timesteps_trim=(trim_start, trim_end),
        )
        self.assertTrue(
            (result_trim["available_timesteps"] >= self.min_timesteps_month).all()
        )
        self.assertTrue(
            (result_trim["available_timesteps"] <= 13).all()
        )  # Jan 2021 to Jan 2022 inclusive = 13 months
        self.assertTrue((result_trim["start_date"] >= "2021-01-01").all())
        self.assertTrue((result_trim["end_date"] <= "2022-01-31").all())

    def test_max_timesteps_trim_no_valid_time(self):
        """Trimming should also work when valid_time logic disabled (center on midpoint)."""
        target_max = self.min_timesteps_month
        result_trim = process_parquet(
            self.df_month,
            freq="month",
            use_valid_time=False,
            max_timesteps_trim=target_max,
        )
        self.assertTrue((result_trim["available_timesteps"] <= target_max).all())
        # required minimum still enforced
        self.assertTrue(
            (result_trim["available_timesteps"] >= self.min_timesteps_month).all()
        )

    def test_max_timesteps_trim_errors(self):
        """Test invalid inputs for max_timesteps_trim raise errors."""
        with self.assertRaises(ValueError):
            process_parquet(
                self.df_month, freq="month", use_valid_time=True, max_timesteps_trim=5
            )  # below required 12
        with self.assertRaises(ValueError):
            process_parquet(
                self.df_month,
                freq="month",
                use_valid_time=True,
                max_timesteps_trim="invalid",
            )

    def test_datetime_handling(self):
        """Test handling of datetime objects in timestamp columns"""
        test_df = self.df_month.copy()
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"]).dt.tz_localize(
            "UTC"
        )
        test_df["valid_time"] = pd.to_datetime(test_df["valid_time"]).dt.strftime(
            "%Y-%m-%d"
        )

        result = process_parquet(test_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_duplicate_rows_handling(self):
        """Test handling of duplicate rows"""
        test_df = pd.concat([self.df_month, self.df_month.sample(5)], ignore_index=True)

        result = process_parquet(test_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        # Check that duplicates are removed
        self.assertEqual(result.index.nunique(), result.shape[0])

    def test_trimming_tuple_with_valid_time(self):
        """Test trimming with a tuple range when valid_time is used"""

        test_df = self.df_month.copy()
        test_df.loc[test_df["sample_id"] == "sample_1", "valid_time"] = pd.Timestamp(
            "2021-06-01"
        )
        test_df.loc[test_df["sample_id"] == "sample_3", "valid_time"] = pd.Timestamp(
            "2021-11-01"
        )
        test_df.loc[test_df["sample_id"] == "sample_2", "valid_time"] = pd.Timestamp(
            "2022-06-01"
        )

        trim_start = pd.Timestamp("2021-01-01")
        trim_end = pd.Timestamp("2022-01-01")

        result_trim = process_parquet(
            test_df,
            freq="month",
            use_valid_time=True,
            max_timesteps_trim=(trim_start, trim_end),
        )
        self.assertTrue(
            (result_trim["available_timesteps"] >= self.min_timesteps_month).all()
        )
        self.assertTrue((result_trim["start_date"] >= "2021-01-01").all())
        self.assertTrue((result_trim["end_date"] <= str(trim_end)).all())
        self.assertIn("sample_1", result_trim.index.unique())
        self.assertIn("sample_3", result_trim.index.unique())
        self.assertNotIn("sample_2", result_trim.index.unique())
