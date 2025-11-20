"""Perform cropland and croptype mapping inference using local execution of UDFs.

Make sure you test this script on Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file and its dependencies.

This script tests both cropland and croptype mapping workflows by calling
the UDF functions directly without running batch jobs on OpenEO.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr
from dateutil.parser import parse
from loguru import logger
from prometheo.predictors import NODATAVALUE
from pyproj import CRS

# from worldcereal.openeo.classifier import apply_inference
from worldcereal.openeo.inference import run_single_workflow
from worldcereal.parameters import CropLandParameters, CropTypeParameters


def subset_ds_temporally(
    ds, season, allow_partial: bool = False, nodata_value: int = NODATAVALUE
):
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
    allow_partial : bool, optional
        If False (default), behaves strictly and raises ValueError when no
        complete 12‑month window is found. If True, returns a 12‑month
        window for the earliest candidate year, filling missing timestamps
        with nodata_value and logging a warning.
    nodata_value : int, optional
        Value used to fill missing timestamps when allow_partial is True.
        Default is NODATAVALUE defined by Prometheo (65535).

    Returns:
    -------
    xarray.Dataset
        A subset of the input dataset `ds` that matches the 12-month
        temporal window defined by the `season`. If allow_partial is True
        and a full window is not available, missing months are inserted
        and filled with nodata_value.

    Raises:
    ------
    ValueError
        If no matching 12-month window is found and allow_partial=False.
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
        if not allow_partial:
            raise ValueError(
                "No matching 12-month window in ds for the season pattern."
            )
        # Partial mode: use earliest year to build expected sequence and fill gaps
        y = min(t_index.year)
        if wrap:
            expected = [
                pd.Timestamp(datetime(y if m >= start_dt.month else y + 1, m, 1))
                for m in months
            ]
        else:
            expected = [pd.Timestamp(datetime(y, m, 1)) for m in months]
        present = [ts for ts in expected if ts in t_index]
        missing = [ts for ts in expected if ts not in t_index]
        logger.warning(
            f"Partial temporal subset: missing {len(missing)} months "
            f"({', '.join([ts.strftime('%Y-%m') for ts in missing])}); "
            f"filling with nodata_value={nodata_value}."
        )
        # Reindex will insert missing timestamps with nodata_value
        selected = ds.sel(t=present).reindex(t=expected, fill_value=nodata_value)

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


def run_cropland_croptype_mapping(
    ds: xr.Dataset,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
):
    """Run full croptype mapping inference workflow on local files.
    Parameters
    ----------
    input_patches : dict[str,Path]
        Dictionary of input patch names and their corresponding file paths.
    outdir : Path
        Output directory to save results.
    model_url : str, optional
        URL to download the croptype classification model from. If None, uses
        the default model provided by WorldCereal.
    mask_croptype_with_cropland : bool, optional
        Whether to mask croptype classification with cropland classification.
        Defaults to True.
    custom_landcover_presto_url : str, optional
        URL to download the cropland feature extraction model from. If None, uses
        the default model provided by WorldCereal.
    custom_landcover_classifier_url : str, optional
        URL to download the cropland classification model from. If None, uses
        the default model provided by WorldCereal.
    Returns
    -------
    dict[str, dict[str, Path]]
        Dictionary with patch names as keys and another dictionary as values,
        containing paths to cropland and croptype classification GeoTIFFs.
    """

    logger.info("Running combined cropland/croptype mapping workflow ...")

    # Initialize results dict
    results = {}

    # Optional cropland classification
    if cropland_parameters is not None and croptype_parameters.mask_cropland:
        cropland_classification = run_cropland_mapping(ds, cropland_parameters)

        # Reconstruct dataset with CRS attributes
        cropland_classification = reconstruct_dataset(cropland_classification, ds)
        results["cropland"] = cropland_classification

        cropland_mask = cropland_classification.sel(bands="classification")
    else:
        cropland_mask = None

    # Croptype classification
    croptype_classification = run_croptype_mapping(
        ds, croptype_parameters, mask=cropland_mask
    )

    # Reconstruct dataset with CRS attributes
    croptype_classification = reconstruct_dataset(croptype_classification, ds)
    results["croptype"] = croptype_classification

    # # Save cropland classification to GeoTIFF
    # cropland_path = outdir / name / "cropland_classification.tif"
    # cropland_path.parent.mkdir(exist_ok=True)
    # classification_to_geotiff(cropland_classification, epsg, cropland_path)
    # output_paths[name]["cropland"] = cropland_path

    # # save crop type map to GeoTIFF
    # croptype_path = outdir / name / "croptype_classification.tif"
    # croptype_path.parent.mkdir(exist_ok=True)
    # classification_to_geotiff(
    #     classification=croptype_classification, epsg=epsg, out_path=croptype_path

    return results


def run_cropland_mapping(
    ds: xr.Dataset,
    parameters: CropLandParameters = CropLandParameters(),
) -> xr.Dataset:
    """Run cropland mapping pipeline: embedding extraction + classification.
    parameters
    ----------
    ds : xr.Dataset
        Input satellite dataset
    parameters : CropLandParameters, optional
        Parameters for the cropland mapping pipeline. Default is CropLandParameters().

    Returns
    -------
    xr.Dataset
        Cropland classification dataset
    """
    logger.info("Running cropland mapping workflow...")

    # Get CRS and convert to xarray DataArray
    epsg = CRS.from_wkt(ds.crs.attrs["spatial_ref"]).to_epsg()
    arr = ds.drop_vars("crs").fillna(NODATAVALUE).astype("uint16").to_array(dim="bands")

    # Convert parameters to dict and add local run overrides
    cropland_params = parameters.model_dump()
    cropland_params["feature_parameters"].update({"ignore_dependencies": True})
    cropland_params["classifier_parameters"].update({"ignore_dependencies": True})

    # Run classification UDF
    classification = run_single_workflow(arr, epsg, parameters=cropland_params)

    # Reconstruct dataset with CRS attributes
    classification = reconstruct_dataset(classification, ds)

    return classification


def run_croptype_mapping(
    ds: xr.Dataset,
    parameters: CropTypeParameters = CropTypeParameters(),
    mask: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """Run croptype mapping pipeline: embedding extraction + classification.

    Parameters
    ----------
    ds : xr.Dataset
        Input satellite dataset
    parameters : CropTypeParameters, optional
        Parameters for the croptype mapping pipeline. Default is CropTypeParameters().
        URL to download the croptype classification model from. If None, uses
        the default model provided by WorldCereal.
    mask : xr.DataArray, optional
        Optional cropland mask to apply during classification. Pixels with mask value
        of 0 will be set to 254 (non-cropland) in the output classification.

    Returns
    -------
    xr.Dataset
        Croptype classification dataset
    """
    logger.info("Running croptype mapping workflow...")

    # Get CRS and convert to xarray DataArray
    epsg = CRS.from_wkt(ds.crs.attrs["spatial_ref"]).to_epsg()
    arr = ds.drop_vars("crs").fillna(NODATAVALUE).astype("uint16").to_array(dim="bands")

    # Convert parameters to dict and add local run overrides
    croptype_params = parameters.model_dump()
    croptype_params["feature_parameters"].update({"ignore_dependencies": True})
    croptype_params["classifier_parameters"].update({"ignore_dependencies": True})

    # Run classification UDF
    classification = run_single_workflow(
        arr, epsg, parameters=croptype_params, mask=mask
    )

    # Reconstruct dataset with CRS attributes
    classification = reconstruct_dataset(classification, ds)

    return classification


def classification_to_geotiff(
    classification: xr.DataArray,
    epsg: int,
    out_path: Path,
) -> None:
    """Save classification DataArray as GeoTIFF.
    Parameters
    ----------
    classification : xr.DataArray
        Classification DataArray to save.
    epsg : int
        EPSG code for the CRS.
    out_path : Path
        Output path for the GeoTIFF file."""

    # ignore import error for rioxarray if not used
    import rioxarray  # noqa: F401

    logger.info(f"Saving classification to GeoTIFF at: {out_path}")

    classification.rio.set_crs(f"epsg:{epsg}", inplace=True)
    classification.rio.to_raster(out_path)
