"""Perform cropland and croptype mapping inference using local execution of UDFs.

Make sure you test this script on Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file and its dependencies.

This script tests both cropland and croptype mapping workflows by calling
the UDF functions directly without running batch jobs on OpenEO.
"""

from typing import Optional
from pathlib import Path

from pyproj import CRS
import xarray as xr
from loguru import logger

from worldcereal.openeo.feature_extractor import extract_presto_embeddings
from worldcereal.openeo.inference import apply_inference
from worldcereal.parameters import CropLandParameters, CropTypeParameters


def run_full_croptype_inference_workflow(
    input_patches: dict[str, Path],
    outdir: Path,
    model_url: Optional[str] = None,
    mask_croptype_with_cropland: bool = True,
    custom_landcover_presto_url: Optional[str] = None,
    custom_landcover_classifier_url: Optional[str] = None,
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

    # Create dictionnary with final output paths
    output_paths = {}

    npatches = len(input_patches)
    i = 0
    for name, in_path in input_patches.items():
        i += 1
        logger.info(f"Processing patch {name} ({i}/{npatches})")
        output_paths[name] = {}

        # Open the file
        ds = xr.open_dataset(in_path)
        # Get the EPSG code and convert to xarray DataArray
        epsg = CRS.from_wkt(ds.crs.attrs["spatial_ref"]).to_epsg()
        arr = ds.drop_vars("crs").fillna(65535).astype("uint16").to_array(dim="bands")

        if mask_croptype_with_cropland:
            logger.info("Generating cropland mask...")
            # Run cropland mapping
            landcover_embeddings, cropland_classification = run_cropland_mapping(
                arr,
                epsg=epsg,
                custom_presto_url=custom_landcover_presto_url,
                classifier_url=custom_landcover_classifier_url,
            )
            # Save cropland classification to GeoTIFF
            cropland_path = outdir / name / "cropland_classification.tif"
            cropland_path.parent.mkdir(exist_ok=True)
            classification_to_geotiff(cropland_classification, epsg, cropland_path)
            output_paths[name]["cropland"] = cropland_path

        # Run croptype mapping
        logger.info("Generating crop type map...")
        croptype_embeddings, croptype_classification = run_croptype_mapping(
            arr, epsg=epsg, classifier_url=model_url
        )

        if mask_croptype_with_cropland:
            logger.info("Masking crop type map with cropland mask...")
            # Set all croptype_classification pixel values to 254 where cropland_classification 'classification' band == 0
            mask = cropland_classification.sel(bands="classification") == 0
            croptype_classification = croptype_classification.where(~mask, 254)

        # save crop type map to GeoTIFF
        croptype_path = outdir / name / "croptype_classification.tif"
        croptype_path.parent.mkdir(exist_ok=True)
        classification_to_geotiff(
            classification=croptype_classification, epsg=epsg, out_path=croptype_path
        )
        output_paths[name]["croptype"] = croptype_path

    return output_paths


def run_cropland_mapping(
    arr: xr.DataArray,
    epsg: int = 32631,
    custom_presto_url: Optional[str] = None,
    classifier_url: Optional[str] = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Run cropland mapping pipeline: embedding extraction + classification.
    parameters
    ----------
    arr : xr.DataArray
        Input satellite data array
    epsg : int, optional
        EPSG code for the input data. Default is 32631.
    custom_presto_url : str, optional
        URL to download the cropland feature extraction model from. If None, uses
        the default model provided by WorldCereal.
    classifier_url : str, optional
        URL to download the cropland classification model from. If None, uses
        the default model provided by WorldCereal.
    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Tuple of (embeddings, classification) arrays
    """
    logger.info("Running cropland embedding extraction UDF...")

    # Initialize CropLandParameters - simple and clean
    cropland_params = CropLandParameters()

    # Get feature parameters and add any testing overrides
    feature_params = cropland_params.feature_parameters.model_dump()
    feature_params.update({"ignore_dependencies": True})
    # add custom presto URL
    if custom_presto_url is not None:
        feature_params.update({"presto_model_url": custom_presto_url})
        logger.info(f"Custom Presto URL set to: {feature_params['presto_model_url']}")

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
    # Add custom classifier URL if provided
    if classifier_url is not None:
        classifier_params.update({"classifier_url": classifier_url})
        logger.info(
            f"Custom classifier URL set to: {classifier_params['classifier_url']}"
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
