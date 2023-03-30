import pytest
from worldcereal.collections import AgERA5YearlyCollection


@pytest.mark.skip
def test_timeout():

    meteopath = '/data/worldcereal/s3collections/satio_agera5_yearly.csv'
    AgERA5coll = AgERA5YearlyCollection.from_path(meteopath)

    epsg = 32631
    bounds = (574680, 5621800, 575320, 5622440)

    coll = AgERA5coll.filter_bounds(bounds, epsg)
    coll = coll.filter_dates(
        '2021-06-01',
        '2021-08-31'
    )
    ts = coll.load_timeseries('temperature_mean',
                              resolution=100)


if __name__ == '__main__':
    test_timeout()
