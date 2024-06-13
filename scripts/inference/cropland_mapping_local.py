"""Perform cropland mapping inference using a local execution of presto.

Make sure you test this script on the Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file installed with it's dependencies.
"""

from pathlib import Path

import requests
import xarray as xr
from openeo_gfmap.features.feature_extractor import (
    EPSG_HARMONIZED_NAME,
    apply_feature_extractor_local,
)

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor

TEST_FILE_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/presto/localtestdata/local_presto_inputs.nc"
TEST_FILE_PATH = Path.cwd() / "presto_test_inputs.nc"


if __name__ == "__main__":
    if not TEST_FILE_PATH.exists():
        print("Downloading test input data...")
        # Download the test input data
        with requests.get(TEST_FILE_URL, stream=True, timeout=180) as response:
            response.raise_for_status()
            with open(TEST_FILE_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    print("Loading array in-memory...")
    arr = (
        xr.open_dataset(TEST_FILE_PATH)
        .to_array(dim="bands")
        .drop_sel(bands="crs")
        .astype("uint16")
    )

    print("Running presto UDF locally")
    features = apply_feature_extractor_local(
        PrestoFeatureExtractor,
        arr,
        parameters={EPSG_HARMONIZED_NAME: 32631, "ignore_dependencies": True},
    )

    features.to_netcdf(Path.cwd() / "presto_test_features.nc")
