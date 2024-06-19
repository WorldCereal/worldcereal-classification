import datetime

import pandas as pd

from worldcereal import BoundingBoxExtent
from worldcereal.seasons import (
    doy_from_tiff,
    doy_to_date_after,
    get_processing_dates_for_extent,
)


def test_doy_from_tiff():
    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    doy_data = doy_from_tiff("tc-s1", "SOS", bounds, epsg, resolution=10000)

    assert doy_data.size == 1

    doy_data = int(doy_data)

    assert doy_data != 0


def test_doy_to_date_after():
    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    doy_data = doy_from_tiff("tc-s2", "SOS", bounds, epsg, resolution=10000)

    after_date = datetime.datetime(2019, 1, 1)
    doy_date = doy_to_date_after(int(doy_data), after_date)

    assert pd.to_datetime(doy_date) >= after_date

    after_date = datetime.datetime(2019, 8, 1)
    doy_date = doy_to_date_after(int(doy_data), after_date)

    assert pd.to_datetime(doy_date) >= after_date


def test_get_processing_dates_for_extent():
    # Test to check if we can infer processing dates for default season
    # tc-annual
    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631
    year = 2021
    extent = BoundingBoxExtent(*bounds, epsg)

    start_date, end_date = get_processing_dates_for_extent(extent, year)

    assert pd.to_datetime(end_date).year == year
    assert pd.to_datetime(end_date) - pd.to_datetime(start_date) == pd.Timedelta(
        days=364
    )
