import numpy as np
import xarray as xr

from worldcereal.openeo.feature_extractor import compute_slope, evaluate_resolution


def test_slope_computation():
    test_elevation = np.array(
        [[10, 20, 30, 30], [10, 20, 20, 20], [65535, 20, 20, 20]], dtype=np.uint16
    )

    array = xr.DataArray(
        test_elevation[None, :, :],
        dims=["bands", "y", "x"],
        coords={
            "bands": ["elevation"],
            "x": [
                71.2302216215233,
                71.23031145305171,
                71.23040128458014,
                71.23040128458014,
            ],
            "y": [25.084450211061935, 25.08436885206669, 25.084287493017356],
        },
    )

    # In the UDF no_data is set to 65535
    resolution = evaluate_resolution(inarr=array, epsg=4326)
    slope = compute_slope(
        array, resolution
    ).values  # pylint: disable=protected-access

    assert slope[0, -1, 0] == 65535
    assert resolution == 10
