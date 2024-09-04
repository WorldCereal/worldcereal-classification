import os
from pathlib import Path

import pytest
import xarray as xr


def get_test_resource(relative_path):
    dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return dir / "testresources" / relative_path


@pytest.fixture
def WorldCerealPreprocessedInputs():
    filepath = get_test_resource("worldcereal_preprocessed_inputs.nc")
    arr = (
        xr.open_dataset(filepath)
        .to_array(dim="bands")
        .drop_sel(bands="crs")
        .astype("uint16")
    )
    return arr
