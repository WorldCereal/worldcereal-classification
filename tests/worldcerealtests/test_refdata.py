import unittest

import pandas as pd
from shapely.geometry import Polygon

from worldcereal.utils.refdata import (
    get_best_valid_time,
    get_class_mappings,
    map_classes,
    month_diff,
    query_public_extractions,
)

LANDCOVER_KEY = "LANDCOVER10"
CROPTYPE_KEY = "CROPTYPE27"


def test_query_public_extractions():
    """Unittest for querying public extractions."""

    # Define small polygon
    poly = Polygon.from_bounds(*(4.535, 51.050719, 4.600936, 51.098176))

    # Query extractions
    df = query_public_extractions(poly, buffer=100)

    # Check if dataframe has samples
    assert not df.empty


def test_get_best_valid_time():
    def process_test_case(test_case: pd.Series) -> pd.DataFrame:
        test_case_res = []
        for processing_period_middle_month in range(1, 13):
            test_case["true_valid_time_month"] = test_case["valid_time"].month
            test_case["proposed_valid_time_month"] = processing_period_middle_month
            test_case["valid_month_shift_backward"] = month_diff(
                test_case["proposed_valid_time_month"],
                test_case["true_valid_time_month"],
            )
            test_case["valid_month_shift_forward"] = month_diff(
                test_case["true_valid_time_month"],
                test_case["proposed_valid_time_month"],
            )
            proposed_valid_time = get_best_valid_time(test_case)
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
        test_case1_res[test_case1_res["proposed_valid_month"].isin([1, 2, 11, 12])][
            "resulting_valid_time"
        ]
        .isna()
        .all()
    )
    assert (
        test_case1_res[test_case1_res["proposed_valid_month"].isin(range(3, 11))][
            "resulting_valid_time"
        ]
        .notna()
        .all()
    )

    # Assertions for test case 2
    assert (
        test_case2_res[test_case2_res["proposed_valid_month"].isin([1, 2, 3, 11, 12])][
            "resulting_valid_time"
        ]
        .isna()
        .all()
    )
    assert (
        test_case2_res[test_case2_res["proposed_valid_month"].isin(range(4, 11))][
            "resulting_valid_time"
        ]
        .notna()
        .all()
    )

    # Assertions for test case 3
    assert (
        test_case3_res[
            test_case3_res["proposed_valid_month"].isin([1, 2, 9, 10, 11, 12])
        ]["resulting_valid_time"]
        .isna()
        .all()
    )
    assert (
        test_case3_res[test_case3_res["proposed_valid_month"].isin(range(3, 9))][
            "resulting_valid_time"
        ]
        .notna()
        .all()
    )


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
