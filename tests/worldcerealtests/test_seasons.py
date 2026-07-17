import numpy as np

from worldcereal.seasons import (
    circular_median_day_of_year,
    doy_from_tiff,
)


def test_doy_from_tiff():
    bounds = (574680, 5621800, 575320, 5622440)
    epsg = 32631

    doy_data = doy_from_tiff("tc-s1", "SOS", bounds, epsg, resolution=10000)

    assert doy_data.size == 1

    doy_data = int(doy_data.item())

    assert doy_data != 0


def test_compute_median_doy():
    # Test to check if we can compute median day of year
    # if array crosses the calendar year
    doy_array = np.array([360, 362, 365, 1, 3, 5, 7])
    assert circular_median_day_of_year(doy_array) == 1

    # Other tests
    assert circular_median_day_of_year([1, 2, 3, 4, 5]) == 3
    assert circular_median_day_of_year([320, 330, 340, 360]) == 330
    assert circular_median_day_of_year([320, 330, 340, 360, 10]) == 340
