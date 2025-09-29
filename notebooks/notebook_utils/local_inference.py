"""Perform cropland and croptype mapping inference using local execution of UDFs.

Make sure you test this script on Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file and its dependencies.

This script tests both cropland and croptype mapping workflows by calling
the UDF functions directly without running batch jobs on OpenEO.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import xarray as xr
from dateutil.parser import parse
from loguru import logger

from worldcereal.openeo.feature_extractor import extract_presto_embeddings
from worldcereal.openeo.inference import apply_inference
from worldcereal.parameters import CropLandParameters, CropTypeParameters


def subset_ds_temporally(ds, season):
    """
    Subsets a dataset temporally based on a given season.
    This function extracts a subset of the dataset `ds` that matches the
    temporal context defined by the `season`. The `season` is expected to
    provide a start and end date, and the function ensures that the subset
    spans a complete 12-month window, even if the season wraps over the
    end of the year.
    Parameters:
    ----------
    ds : xarray.Dataset
        The input dataset containing a time dimension `t` to be subset.
    season : TemporalContext
        An object containing `start_date` and `end_date` as strings,
        representing the start and end of the season.
    Returns:
    -------
    xarray.Dataset
        A subset of the input dataset `ds` that matches the 12-month
        temporal window defined by the `season`.
    Raises:
    ------
    ValueError
        If no matching 12-month window is found in the dataset for the
        given season pattern.
    Notes:
    -----
    - The function assumes that the `t` dimension in the dataset is
      convertible to a pandas DatetimeIndex.
    - If the season wraps over the end of the year (e.g., starts in
      December and ends in February), the function handles this case
      by constructing the appropriate month sequence.
    """

    # Parse season (already a TemporalContext with string dates)
    start_dt = parse(season.start_date)
    end_dt = parse(season.end_date)

    # Does the season wrap over year end?
    wrap = (end_dt.month, end_dt.day) <= (start_dt.month, start_dt.day)

    # Month sequence for one full season (12 months)
    months = (
        (list(range(start_dt.month, 13)) + list(range(1, end_dt.month + 1)))
        if wrap
        else list(range(start_dt.month, end_dt.month + 1))
    )

    t_index = ds.t.to_index()

    # Find the first year in ds that can provide the complete 12‑month window
    selected = None
    for y in sorted(set(t_index.year)):
        if wrap:
            expected = [
                pd.Timestamp(datetime(y if m >= start_dt.month else y + 1, m, 1))
                for m in months
            ]
        else:
            expected = [pd.Timestamp(datetime(y, m, 1)) for m in months]
        if all(ts in t_index for ts in expected):
            selected = ds.sel(t=expected)
            break

    if selected is None:
        raise ValueError("No matching 12-month window in ds for the season pattern.")

    return selected


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
    """Run cropland mapping pipeline: embedding extraction + classification."""
    logger.info("Running cropland embedding extraction UDF...")

    # Initialize CropLandParameters - simple and clean
    cropland_params = CropLandParameters()

    # Get feature parameters and add any testing overrides
    feature_params = cropland_params.feature_parameters.model_dump()
    feature_params.update({"ignore_dependencies": True})

    # Run feature extraction UDF
    embeddings = extract_presto_embeddings(
        inarr=arr, parameters=feature_params, epsg=epsg
    )

    logger.info(f"Embeddings extracted with shape: {embeddings.shape}")

    logger.info("Running cropland classification UDF...")

    # Get classifier parameters and add any testing overrides
    classifier_params = cropland_params.classifier_parameters.model_dump()
    classifier_params.update(
        {
            "ignore_dependencies": True,
        }
    )

    # Run classification UDF
    classification = apply_inference(inarr=embeddings, parameters=classifier_params)

    logger.info(f"Classification completed with shape: {classification.shape}")

    return embeddings, classification


def run_croptype_mapping(
    arr: xr.DataArray,
    epsg: int = 32631,
    target_date: Optional[str] = None,
    classifier_url: Optional[str] = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Run croptype mapping pipeline: feature extraction + classification.

    Parameters
    ----------
    arr : xr.DataArray
        Input satellite data array
    target_date : str, optional
        Target date in ISO format (YYYY-MM-DD). If None, uses middle timestep.
    classifier_url : str, optional
        URL to download the croptype classification model from. If None, uses
        the default model provided by WorldCereal.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Tuple of (embeddings, classification) arrays
    """
    logger.info("Running croptype feature extraction UDF...")

    # Initialize CropTypeParameters with target_date - that's it!
    croptype_params = CropTypeParameters(target_date=target_date)

    # Get feature parameters and add any testing overrides
    feature_params = croptype_params.feature_parameters.model_dump()
    feature_params.update({"ignore_dependencies": True})

    # Run feature extraction UDF
    embeddings = extract_presto_embeddings(
        inarr=arr, parameters=feature_params, epsg=epsg
    )

    logger.info(f"Embeddings extracted with shape: {embeddings.shape}")

    logger.info("Running croptype classification UDF...")

    # Get classifier parameters and add any testing overrides
    classifier_params = croptype_params.classifier_parameters.model_dump()
    classifier_params.update(
        {
            "ignore_dependencies": True,
        }
    )
    if classifier_url is not None:
        logger.info(f"Setting custom classifier URL: {classifier_url}")
        classifier_params.update({"classifier_url": classifier_url})

    # Run classification UDF
    classification = apply_inference(inarr=embeddings, parameters=classifier_params)

    logger.info(f"Classification completed with shape: {classification.shape}")

    return embeddings, classification
