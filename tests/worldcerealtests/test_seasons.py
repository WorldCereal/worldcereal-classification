import datetime
import pandas as pd
from worldcereal.seasons import (doy_from_tiff,
                                 doy_to_date_after,
                                 infer_season_dates)


def test_doy_from_tiff():

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    doy_data = doy_from_tiff('winter', 'SOS', bounds, epsg,
                             resolution=10000)

    assert doy_data.size == 1

    doy_data = int(doy_data)

    assert doy_data != 0


def test_doy_to_date_after():

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    doy_data = doy_from_tiff('tc-maize-main', 'SOS', bounds, epsg,
                             resolution=10000)

    after_date = datetime.datetime(2019, 1, 1)
    doy_date = doy_to_date_after(int(doy_data), after_date)

    assert pd.to_datetime(doy_date) >= after_date

    after_date = datetime.datetime(2019, 8, 1)
    doy_date = doy_to_date_after(int(doy_data), after_date)

    assert pd.to_datetime(doy_date) >= after_date


def test_infer_season_dates_maize1():

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    closest_date = '2019-06-01'

    start_date, end_date = infer_season_dates('tc-maize-main',
                                              bounds, epsg,
                                              closest_date)

    assert pd.to_datetime(start_date) <= pd.to_datetime(closest_date)
    assert pd.to_datetime(end_date) >= pd.to_datetime(closest_date)


def test_infer_season_dates_annual():

    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    closest_date = '2019-06-01'

    start_date, end_date = infer_season_dates('tc-annual',
                                              bounds, epsg,
                                              closest_date)

    assert pd.to_datetime(start_date) <= pd.to_datetime(closest_date)
    assert pd.to_datetime(end_date) >= pd.to_datetime(closest_date)
