"""Perform cropland and croptype mapping inference using local execution of UDFs.

Make sure you test this script on Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file and its dependencies.

This script tests both cropland and croptype mapping workflows by calling
the UDF functions directly without running batch jobs on OpenEO.
"""

from pathlib import Path
from typing import Optional

import requests
import xarray as xr

from worldcereal.openeo.inference import run_single_workflow
from worldcereal.parameters import CropLandParameters, CropTypeParameters

TEST_FILE_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/presto/localtestdata/local_presto_inputs.nc"
TEST_FILE_PATH = Path.cwd() / "presto_test_inputs.nc"


def reconstruct_dataset(arr: xr.DataArray, ds: xr.Dataset) -> xr.Dataset:
    """Reconstruct CRS attributes."""
    crs_attrs = ds["crs"].attrs
    x = ds.coords.get("x", None)
    y = ds.coords.get("y", None)

    # Build dataset with bands as separate variables
    new_ds = arr.assign_coords(bands=arr.bands.astype(str)).to_dataset(dim="bands")

    # Reset the coordinates
    new_ds = new_ds.assign_coords(x=x)
    new_ds["x"].attrs.setdefault("standard_name", "projection_x_coordinate")
    new_ds["x"].attrs.setdefault("units", "m")

    new_ds = new_ds.assign_coords(y=y)
    new_ds["y"].attrs.setdefault("standard_name", "projection_y_coordinate")
    new_ds["y"].attrs.setdefault("units", "m")

    # Assign CRS attributes to all data variables
    crs_name = "spatial_ref"
    new_ds[crs_name] = xr.DataArray(0, attrs=crs_attrs)

    for v in new_ds.data_vars:
        new_ds[v].attrs["grid_mapping"] = crs_name

    return new_ds


def run_cropland_mapping(
    arr: xr.DataArray, epsg: int = 32631
) -> tuple[xr.DataArray, xr.DataArray]:
    """Run cropland mapping pipeline: feature extraction + classification."""
    print("Running cropland feature extraction UDF...")

    # Initialize CropLandParameters - simple and clean
    cropland_params = CropLandParameters().model_dump()
    cropland_params["feature_parameters"].update({"ignore_dependencies": True})
    cropland_params["classifier_parameters"].update({"ignore_dependencies": True})

    # Run classification UDF
    classification = run_single_workflow(arr, epsg, parameters=cropland_params)

    print(f"Classification completed with shape: {classification.shape}")
    print(f"Classification bands: {list(classification.bands.values)}")

    return classification


def run_croptype_mapping(
    arr: xr.DataArray, epsg: int = 32631, target_date: Optional[str] = None
) -> tuple[xr.DataArray, xr.DataArray]:
    """Run croptype mapping pipeline: feature extraction + classification.

    Parameters
    ----------
    arr : xr.DataArray
        Input satellite data array
    target_date : str, optional
        Target date in ISO format (YYYY-MM-DD). If None, uses middle timestep.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Tuple of (features, classification) arrays
    """
    print("Running croptype feature extraction UDF...")

    # Initialize CropTypeParameters with target_date - that's it!
    croptype_params = CropTypeParameters(target_date=target_date).model_dump()
    croptype_params["feature_parameters"].update({"ignore_dependencies": True})
    croptype_params["classifier_parameters"].update({"ignore_dependencies": True})

    print("Running croptype classification UDF...")

    # Run classification UDF
    classification = run_single_workflow(arr, epsg, parameters=croptype_params)

    print(f"Classification completed with shape: {classification.shape}")
    print(f"Classification bands: {list(classification.bands.values)}")

    return classification


if __name__ == "__main__":
    # Download test data if not exists
    if not TEST_FILE_PATH.exists():
        print("Downloading test input data...")
        with requests.get(TEST_FILE_URL, stream=True, timeout=180) as response:
            response.raise_for_status()
            with open(TEST_FILE_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    print("Loading array in-memory...")
    ds = xr.open_dataset(TEST_FILE_PATH)
    crs_attrs = ds["crs"].attrs
    arr = ds.drop_vars("crs").astype("uint16").to_array(dim="bands")

    print(f"Input array shape: {arr.shape}")
    print(f"Input bands: {list(arr.bands.values)}")
    print(f"Time steps: {len(arr.t)}")

    # Run cropland mapping
    print("\n" + "=" * 50)
    print("CROPLAND MAPPING WORKFLOW")
    print("=" * 50)

    try:
        cropland_classification = run_cropland_mapping(arr)

        # Save results
        cropland_classification_ds = reconstruct_dataset(
            arr=cropland_classification, ds=ds
        )
        cropland_classification_ds.to_netcdf(
            Path.cwd() / "test_classification_cropland.nc"
        )

        print("Cropland mapping completed successfully!")
        print("Classification saved to: test_classification_cropland.nc")

    except Exception as e:
        print(f"Error in cropland mapping: {e}")

    # Run croptype mapping
    print("\n" + "=" * 50)
    print("CROPTYPE MAPPING WORKFLOW")
    print("=" * 50)

    try:
        # Example 1: Default behavior (middle timestep)
        print(
            "\n1. Running croptype mapping with default target_date (middle timestep)..."
        )
        croptype_classification = run_croptype_mapping(arr, target_date=None)

        # Save results
        croptype_classification_ds = reconstruct_dataset(
            arr=croptype_classification, ds=ds
        )
        croptype_classification_ds.to_netcdf(
            Path.cwd() / "test_classification_croptype_default.nc"
        )

        print("Croptype mapping (default) completed successfully!")
        print("Classification saved to: test_classification_croptype_default.nc")

        # Example 2: Custom target date
        print("\n2. Running croptype mapping with custom target_date...")
        # You can specify any date within your temporal range
        custom_target_date = "2021-07-15"  # Example date - adjust based on your data
        croptype_features_custom, croptype_classification_custom = run_croptype_mapping(
            arr, target_date=custom_target_date
        )

        # Save results with different names
        croptype_classification_custom_ds = reconstruct_dataset(
            arr=croptype_classification_custom, ds=ds
        )
        croptype_classification_custom_ds.to_netcdf(
            Path.cwd() / "test_classification_croptype_custom.nc"
        )

        print(
            f"Croptype mapping (target_date={custom_target_date}) completed successfully!"
        )
        print("Classification saved to: test_classification_croptype_custom.nc")

    except Exception as e:
        print(f"Error in croptype mapping: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("This script demonstrates how to use CropTypeParameters with target_date:")
    print("1. target_date=None (default): Uses middle timestep from temporal data")
    print("2. target_date='YYYY-MM-DD': Uses specific date for temporal prediction")
    print("3. All other parameters use defaults from CropTypeParameters")
    print("4. Easy to override any parameter using get_feature_parameters_dict()")
    print("=" * 60)
