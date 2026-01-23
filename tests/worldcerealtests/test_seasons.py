import datetime

import numpy as np
import pandas as pd
from openeo_gfmap import BoundingBoxExtent

from worldcereal.seasons import (
    circular_median_day_of_year,
    doy_from_tiff,
    doy_to_date_after,
    get_processing_dates_for_extent,
)


def test_doy_from_tiff():
    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    doy_data = doy_from_tiff("tc-s1", "SOS", bounds, epsg, resolution=10000)

    assert doy_data.size == 1

    doy_data = int(np.asarray(doy_data).squeeze())

    assert doy_data != 0


def test_doy_to_date_after():
    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    doy_data = doy_from_tiff("tc-s2", "SOS", bounds, epsg, resolution=10000)
    doy_data = int(np.asarray(doy_data).squeeze())

    after_date = datetime.datetime(2019, 1, 1)
    doy_date = doy_to_date_after(doy_data, after_date)

    assert pd.to_datetime(doy_date) >= after_date

    after_date = datetime.datetime(2019, 8, 1)
    doy_date = doy_to_date_after(doy_data, after_date)

    assert pd.to_datetime(doy_date) >= after_date


def test_get_processing_dates_for_extent():
    # Test to check if we can infer processing dates for default season
    # tc-annual
    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631
    year = 2021
    extent = BoundingBoxExtent(*bounds, epsg)

    temporal_context = get_processing_dates_for_extent(extent, year)
    start_date = temporal_context.start_date
    end_date = temporal_context.end_date

    assert pd.to_datetime(end_date).year == year
    assert pd.to_datetime(end_date) - pd.to_datetime(start_date) == pd.Timedelta(
        days=364
    )


def test_compute_median_doy():
    # Test to check if we can compute median day of year
    # if array crosses the calendar year
    doy_array = np.array([360, 362, 365, 1, 3, 5, 7])
    assert circular_median_day_of_year(doy_array) == 1

    # Other tests
    assert circular_median_day_of_year([1, 2, 3, 4, 5]) == 3
    assert circular_median_day_of_year([320, 330, 340, 360]) == 330
    assert circular_median_day_of_year([320, 330, 340, 360, 10]) == 340
