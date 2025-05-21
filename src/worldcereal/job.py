"""Executing inference jobs on the OpenEO backend."""

import json
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import openeo
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import BACKEND_CONNECTIONS
from pydantic import BaseModel
from typing_extensions import TypedDict

from worldcereal.openeo.mapping import _cropland_map, _croptype_map
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs
from worldcereal.parameters import (
    CropLandParameters,
    CropTypeParameters,
    PostprocessParameters,
    WorldCerealProductType,
)
from worldcereal.utils.models import load_model_lut

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


class WorldCerealProduct(TypedDict):
    """Dataclass representing a WorldCereal inference product.

    Attributes
    ----------
    url: str
        URL to the product.
    type: WorldCerealProductType
        Type of the product. Either cropland or croptype.
    temporal_extent: TemporalContext
        Period of time for which the product has been generated.
    path: Optional[Path]
        Path to the downloaded product.
    lut: Optional[Dict]
        Look-up table for the product.

    """

    url: str
    type: WorldCerealProductType
    temporal_extent: TemporalContext
    path: Optional[Path]
    lut: Optional[Dict]


class InferenceResults(BaseModel):
    """Dataclass to store the results of the WorldCereal job.

    Attributes
    ----------
    job_id : str
        Job ID of the finished OpenEO job.
    products: Dict[str, WorldCerealProduct]
        Dictionary with the different products.
    metadata: Optional[Path]
        Path to metadata file, if it was downloaded locally.
    """

    job_id: str
    products: Dict[str, WorldCerealProduct]
    metadata: Optional[Path]


def create_inference_process_graph(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    postprocess_parameters: PostprocessParameters = PostprocessParameters(),
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
) -> openeo.DataCube:
    """Wrapper function that creates the inference openEO process graph.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    product_type : WorldCerealProductType, optional
        product describer, by default WorldCerealProductType.CROPLAND
    cropland_parameters: CropLandParameters
        Parameters for the cropland product inference pipeline.
    croptype_parameters: Optional[CropTypeParameters]
        Parameters for the croptype product inference pipeline. Only required
        whenever `product_type` is set to `WorldCerealProductType.CROPTYPE`,
        will be ignored otherwise.
    postprocess_parameters: PostprocessParameters
        Parameters for the postprocessing pipeline. By default disabled.
    out_format : str, optional
        Output format, by default "GTiff"
    backend_context : BackendContext
        backend to run the job on, by default CDSE.
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.

    Returns
    -------
    openeo.DataCube
        DataCube object representing the inference process graph.
        This object can be used to execute the job on the OpenEO backend.
        The result will be a DataCube with the classification results.

    Raises
    ------
    ValueError
        if the product is not supported
    ValueError
        if the out_format is not supported
    """
    if product_type not in WorldCerealProductType:
        raise ValueError(f"Product {product_type.value} not supported.")

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    # Make a connection to the OpenEO backend
    connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Preparing the input cube for inference
    inputs = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
        # disable_meteo=True,
    )

    # Spatial filtering
    inputs = inputs.filter_bbox(dict(spatial_extent))

    # Construct the feature extraction and model inference pipeline
    if product_type == WorldCerealProductType.CROPLAND:
        classes = _cropland_map(
            inputs,
            temporal_extent,
            cropland_parameters=cropland_parameters,
            postprocess_parameters=postprocess_parameters,
        )

    elif product_type == WorldCerealProductType.CROPTYPE:
        if not isinstance(croptype_parameters, CropTypeParameters):
            raise ValueError(
                f"Please provide a valid `croptype_parameters` parameter."
                f" Received: {croptype_parameters}"
            )
        # First compute cropland map
        cropland_mask = _cropland_map(
            inputs,
            temporal_extent,
            cropland_parameters=cropland_parameters,
            postprocess_parameters=postprocess_parameters,
        )

        # Save final mask if required
        if croptype_parameters.save_mask:
            cropland_mask = cropland_mask.save_result(
                format="GTiff",
                options=dict(
                    filename_prefix=f"{WorldCerealProductType.CROPLAND.value}_{temporal_extent.start_date}_{temporal_extent.end_date}",
                ),
            )

        # To use it as a mask, we need to filter out the classification band
        # Use the generic 'process' to avoid client-side errors on missing metadata
        cropland_mask = cropland_mask.process(
            process_id="filter_bands",
            arguments=dict(
                data=cropland_mask,
                bands=["classification"],
            ),
        )

        # Generate crop type map
        classes = _croptype_map(
            inputs,
            temporal_extent,
            croptype_parameters=croptype_parameters,
            cropland_mask=cropland_mask,
            postprocess_parameters=postprocess_parameters,
        )

    # Save the final result
    classes = classes.save_result(
        format=out_format,
        options=dict(
            filename_prefix=f"{product_type.value}_{temporal_extent.start_date}_{temporal_extent.end_date}",
        ),
    )

    return classes


def generate_map(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    output_dir: Optional[Union[Path, str]] = None,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    postprocess_parameters: PostprocessParameters = PostprocessParameters(),
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    job_options: Optional[dict] = None,
) -> InferenceResults:
    """Main function to generate a WorldCereal product.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    output_dir : Optional[Union[Path, str]]
        path to directory where products should be downloaded to
    product_type : WorldCerealProductType, optional
        product describer, by default WorldCerealProductType.CROPLAND
    cropland_parameters: CropLandParameters
        Parameters for the cropland product inference pipeline.
    croptype_parameters: Optional[CropTypeParameters]
        Parameters for the croptype product inference pipeline. Only required
        whenever `product_type` is set to `WorldCerealProductType.CROPTYPE`,
        will be ignored otherwise.
    postprocess_parameters: PostprocessParameters
        Parameters for the postprocessing pipeline. By default disabled.
    out_format : str, optional
        Output format, by default "GTiff"
    backend_context : BackendContext
        backend to run the job on, by default CDSE.
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.
    job_options: dict, optional
        Additional job options to pass to the OpenEO backend, by default None

    Returns
    -------
    InferenceResults
        Results of the finished WorldCereal job.

    Raises
    ------
    ValueError
        if the product is not supported
    ValueError
        if the out_format is not supported
    """

    classes = create_inference_process_graph(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        product_type=product_type,
        cropland_parameters=cropland_parameters,
        croptype_parameters=croptype_parameters,
        postprocess_parameters=postprocess_parameters,
        out_format=out_format,
        backend_context=backend_context,
        tile_size=tile_size,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Submit the job
    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "2g",
        "executor-memoryOverhead": "1g",
        "python-memory": "3g",
        "soft-errors": 0.1,
        "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    # Execute the job
    job = classes.execute_batch(
        job_options=JOB_OPTIONS,
        title=f"WorldCereal [{product_type.value}] job",
        description="Job that performs end-to-end WorldCereal inference",
    )

    # Get look-up tables
    luts = {}
    luts[WorldCerealProductType.CROPLAND.value] = load_model_lut(
        cropland_parameters.classifier_parameters.classifier_url
    )
    if product_type == WorldCerealProductType.CROPTYPE:
        luts[WorldCerealProductType.CROPTYPE.value] = load_model_lut(
            croptype_parameters.classifier_parameters.classifier_url
        )

    # Get job results
    job_result = job.get_results()

    # Get the products
    assets = job_result.get_assets()
    products = {}
    for asset in assets:
        asset_name = asset.name.split(".")[0].split("_")[0]
        asset_type = asset_name.split("-")[0]
        asset_type = getattr(WorldCerealProductType, asset_type.upper())
        if output_dir is not None:
            filepath = asset.download(target=output_dir)
        else:
            filepath = None
        products[asset_name] = {
            "url": asset.href,
            "type": asset_type,
            "temporal_extent": temporal_extent,
            "path": filepath,
            "lut": luts[asset_type.value],
        }

    # Download job metadata if output path is provided
    if output_dir is not None:
        metadata_file = output_dir / "job-results.json"
        metadata_file.write_text(json.dumps(job_result.get_metadata()))
    else:
        metadata_file = None

    # Compile InferenceResults and return
    return InferenceResults(
        job_id=job.job_id, products=products, metadata=metadata_file
    )


def collect_inputs(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    output_path: Union[Path, str],
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    job_options: Optional[dict] = None,
    compositing_window: Literal["month", "dekad"] = "month",
):
    """Function to retrieve preprocessed inputs that are being
    used in the generation of WorldCereal products.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    output_path : Union[Path, str]
        output path to download the product to
    backend_context : BackendContext
        backend to run the job on, by default CDSE
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128
        so it uses the OpenEO default setting.
    job_options: dict, optional
        Additional job options to pass to the OpenEO backend, by default None
    compositing_window: Literal["month", "dekad"]
        Compositing window to use for the data loading in OpenEO, by default
        "month".
    """

    # Make a connection to the OpenEO backend
    connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Preparing the input cube for the inference
    inputs = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
        validate_temporal_context=False,
        compositing_window=compositing_window,
    )

    # Spatial filtering
    inputs = inputs.filter_bbox(dict(spatial_extent))

    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "1g",
        "executor-memoryOverhead": "1g",
        "python-memory": "3g",
        "soft-errors": 0.1,
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    inputs.execute_batch(
        outputfile=output_path,
        out_format="NetCDF",
        title="WorldCereal [collect_inputs] job",
        description="Job that collects inputs for WorldCereal inference",
        job_options=JOB_OPTIONS,
    )
