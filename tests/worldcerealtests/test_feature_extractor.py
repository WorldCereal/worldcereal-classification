import numpy as np
import xarray as xr

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor


def test_dem_computation():
    test_elevation = np.array(
        [[10, 20, 30], [10, 20, 20], [65535, 20, 20]], dtype=np.uint16
    )

    array = xr.DataArray(
        test_elevation[None, :, :],
        dims=["bands", "y", "x"],
        coords={"bands": ["elevation"], "y": [0, 1, 2], "x": [0, 1, 2]},
    )

    extractor = PrestoFeatureExtractor()

    # In the UDF no_data is set to 65535
    slope = extractor._compute_slope(array).values  # pylint: disable=protected-access

    assert slope[0, -1, 0] == 65535
