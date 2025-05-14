import pandas as pd
from shapely.geometry import Polygon

from worldcereal.utils.refdata import (
    get_best_valid_time,
    month_diff,
    query_public_extractions,
)


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

    map_classes(WorldCerealExtractionsDF)
