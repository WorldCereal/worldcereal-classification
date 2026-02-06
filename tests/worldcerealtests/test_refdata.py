import unittest

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from worldcereal.utils.refdata import (
    STEPS_PER_YEAR,
    _step_in_year,
    get_best_valid_time,
    get_class_mappings,
    map_classes,
    query_public_extractions,
)

LANDCOVER_KEY = "LANDCOVER10"
CROPTYPE_KEY = "CROPTYPE27"


def test_query_public_extractions():
    """Unittest for querying public extractions."""

    # Define small polygon
    poly = Polygon.from_bounds(*(4.535, 51.050719, 4.600936, 51.098176))

    # Query extractions
    gdf = query_public_extractions(poly, buffer=100)

    # Check if dataframe has samples
    assert not gdf.empty
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_string() == "EPSG:4326"


def test_get_best_valid_time():
    from worldcereal.utils.timeseries import MIN_EDGE_BUFFER

    NUM_TIMESTEPS = 12

    def process_test_case(test_case: pd.Series) -> pd.DataFrame:
        test_case_res = []
        steps_per_year = STEPS_PER_YEAR["month"]
        valid_step_in_year = _step_in_year(test_case["valid_time"], "month")
        for processing_period_middle_month in range(1, 13):
            processing_period_middle_ts = pd.Timestamp(
                test_case["valid_time"].year, processing_period_middle_month, 1
            )
            processing_period_middle_step = _step_in_year(
                processing_period_middle_ts, "month"
            )
            test_case["valid_step_shift_backward"] = (
                valid_step_in_year - processing_period_middle_step
            ) % steps_per_year
            test_case["valid_step_shift_forward"] = (
                processing_period_middle_step - valid_step_in_year
            ) % steps_per_year
            proposed_valid_time = get_best_valid_time(
                test_case,
                valid_time_buffer=MIN_EDGE_BUFFER,
                num_timesteps=NUM_TIMESTEPS,
                freq="month",
            )
            test_case_res.append([processing_period_middle_month, proposed_valid_time])
        return pd.DataFrame(
            test_case_res, columns=["proposed_valid_month", "resulting_valid_time"]
        )

    test_case1 = pd.Series(
        {
            "start_date": pd.to_datetime("2019-01-01"),
            "end_date": pd.to_datetime("2019-12-01"),
            "valid_time": pd.to_datetime("2019-06-01"),
        }
    )
    test_case2 = pd.Series(
        {
            "start_date": pd.to_datetime("2019-01-01"),
            "end_date": pd.to_datetime("2019-12-01"),
            "valid_time": pd.to_datetime("2019-10-01"),
        }
    )
    test_case3 = pd.Series(
        {
            "start_date": pd.to_datetime("2019-01-01"),
            "end_date": pd.to_datetime("2019-12-01"),
            "valid_time": pd.to_datetime("2019-03-01"),
        }
    )

    # Process test cases
    test_case1_res = process_test_case(test_case1)
    test_case2_res = process_test_case(test_case2)
    test_case3_res = process_test_case(test_case3)

    # Asserts are valid for default MIN_EDGE_BUFFER and NUM_TIMESTEPS values
    # Assertions for test case 1
    assert (
        test_case1_res[
            test_case1_res["proposed_valid_month"].isin([1, 2, 3, 9, 10, 11, 12])
        ]["resulting_valid_time"]
        .isna()
        .all()
    )
    assert (
        test_case1_res[test_case1_res["proposed_valid_month"].isin(range(4, 9))][
            "resulting_valid_time"
        ]
        .notna()
        .all()
    )

    # Assertions for test case 2
    assert (
        test_case2_res[~test_case2_res["proposed_valid_month"].isin([6, 7, 8])][
            "resulting_valid_time"
        ]
        .isna()
        .all()
    )
    assert (
        test_case2_res[test_case2_res["proposed_valid_month"].isin([6, 7, 8])][
            "resulting_valid_time"
        ]
        .notna()
        .all()
    )

    # Assertions for test case 3
    assert (
        test_case3_res[~test_case3_res["proposed_valid_month"].isin([4, 5, 6])][
            "resulting_valid_time"
        ]
        .isna()
        .all()
    )
    assert (
        test_case3_res[test_case3_res["proposed_valid_month"].isin([4, 5, 6])][
            "resulting_valid_time"
        ]
        .notna()
        .all()
    )


def _compute_best_valid_time(
    valid_time: str,
    start_date: str,
    end_date: str,
    processing_period_middle_ts: pd.Timestamp,
    freq: str,
    valid_time_buffer: int,
    num_timesteps: int,
) -> pd.Timestamp:
    steps_per_year = STEPS_PER_YEAR[freq]
    valid_time_ts = pd.to_datetime(valid_time)
    row = pd.Series(
        {
            "valid_time": valid_time_ts,
            "start_date": pd.to_datetime(start_date),
            "end_date": pd.to_datetime(end_date),
        }
    )
    valid_step_in_year = _step_in_year(valid_time_ts, freq)
    processing_period_middle_step = _step_in_year(processing_period_middle_ts, freq)
    row["valid_step_shift_backward"] = (
        valid_step_in_year - processing_period_middle_step
    ) % steps_per_year
    row["valid_step_shift_forward"] = (
        processing_period_middle_step - valid_step_in_year
    ) % steps_per_year
    return get_best_valid_time(
        row,
        valid_time_buffer=valid_time_buffer,
        num_timesteps=num_timesteps,
        freq=freq,
    )


def test_get_best_valid_time_month_end_not_dropped():
    num_timesteps = 12
    valid_time_buffer = 0
    start_date = "2021-03-01"
    end_date = "2022-08-01"
    processing_period_middle_ts = pd.Timestamp("2022-01-01")

    valid_times = [
        "2022-01-16",
        "2022-01-17",
        "2022-01-18",
        "2022-01-15",
        "2022-01-01",
        "2022-01-04",
        "2022-01-05",
        "2022-01-31",
    ]
    for valid_time in valid_times:
        result = _compute_best_valid_time(
            valid_time=valid_time,
            start_date=start_date,
            end_date=end_date,
            processing_period_middle_ts=processing_period_middle_ts,
            freq="month",
            valid_time_buffer=valid_time_buffer,
            num_timesteps=num_timesteps,
        )
        assert pd.notna(result)


def test_get_best_valid_time_dekad_not_dropped():
    num_timesteps = 36
    valid_time_buffer = 0
    start_date = "2021-03-01"
    end_date = "2022-08-01"
    processing_period_middle_ts = pd.Timestamp("2022-01-21")

    result = _compute_best_valid_time(
        valid_time="2022-01-31",
        start_date=start_date,
        end_date=end_date,
        processing_period_middle_ts=processing_period_middle_ts,
        freq="dekad",
        valid_time_buffer=valid_time_buffer,
        num_timesteps=num_timesteps,
    )
    assert pd.notna(result)


def test_get_best_valid_time_month_should_drop():
    num_timesteps = 12
    valid_time_buffer = 0
    start_date = "2021-03-01"
    end_date = "2021-10-01"
    processing_period_middle_ts = pd.Timestamp("2021-07-01")

    result = _compute_best_valid_time(
        valid_time="2021-01-15",
        start_date=start_date,
        end_date=end_date,
        processing_period_middle_ts=processing_period_middle_ts,
        freq="month",
        valid_time_buffer=valid_time_buffer,
        num_timesteps=num_timesteps,
    )
    assert pd.isna(result)


def test_get_best_valid_time_month_leap_day_not_dropped():
    num_timesteps = 12
    valid_time_buffer = 0
    start_date = "2019-05-01"
    end_date = "2020-12-01"
    processing_period_middle_ts = pd.Timestamp("2020-02-01")

    result = _compute_best_valid_time(
        valid_time="2020-02-29",
        start_date=start_date,
        end_date=end_date,
        processing_period_middle_ts=processing_period_middle_ts,
        freq="month",
        valid_time_buffer=valid_time_buffer,
        num_timesteps=num_timesteps,
    )
    assert pd.notna(result)


def test_get_best_valid_time_dekad_all_dekads():
    num_timesteps = 36
    valid_time_buffer = 0
    start_date = "2020-01-01"
    end_date = "2022-12-21"
    cases = [
        ("2021-07-05", pd.Timestamp("2021-07-01")),
        ("2021-07-15", pd.Timestamp("2021-07-11")),
        ("2021-07-25", pd.Timestamp("2021-07-21")),
    ]

    for valid_time, processing_period_middle_ts in cases:
        result = _compute_best_valid_time(
            valid_time=valid_time,
            start_date=start_date,
            end_date=end_date,
            processing_period_middle_ts=processing_period_middle_ts,
            freq="dekad",
            valid_time_buffer=valid_time_buffer,
            num_timesteps=num_timesteps,
        )
        assert pd.notna(result)


def test_get_best_valid_time_dekad_should_drop():
    num_timesteps = 36
    valid_time_buffer = 0
    start_date = "2021-01-01"
    end_date = "2021-12-01"
    processing_period_middle_ts = pd.Timestamp("2021-11-21")

    result = _compute_best_valid_time(
        valid_time="2021-07-15",
        start_date=start_date,
        end_date=end_date,
        processing_period_middle_ts=processing_period_middle_ts,
        freq="dekad",
        valid_time_buffer=valid_time_buffer,
        num_timesteps=num_timesteps,
    )
    assert pd.isna(result)

def test_map_classes(WorldCerealExtractionsDF):
    # Tests the automatic download of latest legend and default
    # class mapping for prometheo finetuning
    from worldcereal.utils.refdata import map_classes

    map_classes(WorldCerealExtractionsDF, finetune_classes=CROPTYPE_KEY)


class TestMapClasses(unittest.TestCase):
    def setUp(self):
        """Set up test data for map_classes tests."""
        self.num_samples = 5

        # Create a simple dataframe with real ewoc_codes from the class mappings
        self.df = pd.DataFrame(
            {
                "ewoc_code": [
                    1101060000,
                    1101010000,
                    1106000032,
                    2000000000,
                    6000000000,
                    0,
                    1000000000,
                ],  # Last two should be filtered out
                "lat": [45.1, 45.2, 45.3, 45.4, 45.5, 45.6, 45.7],
                "lon": [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7],
            }
        )

    def test_map_classes_binary(self):
        """Test mapping classes in binary classification scenario."""
        # Use the real CROPLAND2 mapping which has binary classes (temporary_crops/not_temporary_crops)
        result_df = map_classes(self.df, finetune_classes=LANDCOVER_KEY)

        # Check that filter classes were removed
        self.assertEqual(len(result_df), 5)

        # Check finetune_class mapping
        self.assertEqual(result_df.iloc[0]["finetune_class"], "temporary_crops")
        self.assertEqual(result_df.iloc[1]["finetune_class"], "temporary_crops")
        self.assertEqual(result_df.iloc[2]["finetune_class"], "temporary_crops")
        self.assertEqual(result_df.iloc[3]["finetune_class"], "grasslands")
        self.assertEqual(result_df.iloc[4]["finetune_class"], "built_up")

        # No one-hot columns should be created for binary case
        self.assertFalse(
            any(
                col.startswith("temporary_crops")
                for col in result_df.columns
                if col != "finetune_class"
            )
        )

        # Check balancing_class mapping (using actual CROP_LEGEND values)
        self.assertTrue("balancing_class" in result_df.columns)

    def test_map_classes_multiclass(self):
        """Test mapping classes in multiclass scenario."""
        result_df = map_classes(self.df, finetune_classes=CROPTYPE_KEY)

        # Check that filter classes were removed
        self.assertEqual(len(result_df), 3)

        # Check that classes match the expected ones
        for i, ewoc_code in enumerate([1101060000, 1101010000, 1106000032]):
            if str(ewoc_code) in get_class_mappings()[CROPTYPE_KEY]:
                expected_class = get_class_mappings()[CROPTYPE_KEY][str(ewoc_code)]
                self.assertEqual(result_df.iloc[i]["finetune_class"], expected_class)

    def test_map_classes_missing_codes(self):
        """Test handling of missing codes in mapping dictionary."""
        # Add an ewoc_code that doesn't exist in the real CLASS_MAPPINGS
        df_with_missing = self.df.copy()
        missing_code = 9999
        df_with_missing.loc[len(df_with_missing)] = [missing_code, 45.8, 5.8]

        # Process the dataframe and check what happens
        result_df = map_classes(df_with_missing, finetune_classes=CROPTYPE_KEY)

        # Check that rows with missing codes were removed
        self.assertTrue(missing_code not in result_df["ewoc_code"].values)
