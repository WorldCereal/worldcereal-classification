import os
from pathlib import Path

import geojson
import pandas as pd
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


@pytest.fixture
def SpatialExtent():
    filepath = get_test_resource("spatial_extent.json")
    with open(filepath, "r") as f:
        return geojson.load(f)


@pytest.fixture
def WorldCerealCroplandClassification():
    filepath = get_test_resource("worldcereal_cropland_classification.nc")
    arr = xr.open_dataarray(filepath).astype("uint16")
    return arr


@pytest.fixture
def WorldCerealCroptypeClassification():
    filepath = get_test_resource("worldcereal_croptype_classification.nc")
    arr = xr.open_dataarray(filepath).astype("uint16")
    return arr


@pytest.fixture
def WorldCerealExtractionsDF():
    filepath = get_test_resource("test_public_extractions.parquet")
    return pd.read_parquet(filepath)


@pytest.fixture
def WorldCerealPrivateExtractionsPath():
    filepath = get_test_resource("worldcereal_private_extractions_dummy.parquet")
    return filepath
