"""Perform cropland and croptype mapping inference using local execution of UDFs.

Make sure you test this script on Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file and its dependencies.

This script tests both cropland and croptype mapping workflows by calling
the UDF functions directly without running batch jobs on OpenEO.
"""

from typing import Optional
from pathlib import Path

import xarray as xr
from loguru import logger

from worldcereal.openeo.feature_extractor import extract_presto_embeddings
from worldcereal.openeo.inference import apply_inference
from worldcereal.parameters import CropLandParameters, CropTypeParameters


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

    classification.rio.set_crs(f"epsg:{epsg}", inplace=True)
    classification.rio.to_raster(out_path)
