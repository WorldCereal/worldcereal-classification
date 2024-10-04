"""Executing inference jobs on the OpenEO backend."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

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

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


class WorldCerealProduct(TypedDict):
    """Dataclass representing a WorldCereal inference product.

    Attributes
    ----------
    url: str
        URL to the product.
    type: WorldCerealProductType
        Type of the product. Either cropland or croptype.
    path: Optional[Path]
        Path to the downloaded product.
    """

    url: str
    type: WorldCerealProductType
    path: Optional[Path]


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


def generate_map(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    output_path: Optional[Union[Path, str]],
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: Optional[CropTypeParameters] = CropTypeParameters(),
    postprocess_parameters: PostprocessParameters = PostprocessParameters(enable=False),
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
    output_path : Optional[Union[Path, str]]
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

    if product_type not in WorldCerealProductType:
        raise ValueError(f"Product {product_type.value} not supported.")

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    if backend_context.backend == Backend.CDSE:
        connection = openeo.connect(
            "https://openeo.creo.vito.be/openeo/"
        ).authenticate_oidc()
    else:
        # Make a connection to the OpenEO backend
        connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Preparing the input cube for inference
    inputs = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
    )

    # Explicit filtering again for bbox because of METEO low
    # resolution causing issues
    inputs = inputs.filter_bbox(dict(spatial_extent))

    # Construct the feature extraction and model inference pipeline
    if product_type == WorldCerealProductType.CROPLAND:
        classes = _cropland_map(
            inputs,
            cropland_parameters=cropland_parameters,
            postprocess_parameters=postprocess_parameters,
        )
    elif product_type == WorldCerealProductType.CROPTYPE:
        if not isinstance(croptype_parameters, CropTypeParameters):
            raise ValueError(
                f"Please provide a valid `croptype_parameters` parameter."
                f" Received: {croptype_parameters}"
            )
        # First compute cropland map and save as additional output
        cropland_mask = (
            _cropland_map(
                inputs,
                cropland_parameters=cropland_parameters,
                postprocess_parameters=postprocess_parameters,
            )
            .filter_bands("classification")
            .reduce_dimension(
                dimension="t", reducer="mean"
            )  # Temporary fix to make this work as mask
        ).save_result(format="GTiff", options=dict(filename_prefix="cropland-mask"))

        classes = _croptype_map(
            inputs,
            croptype_parameters=croptype_parameters,
            cropland_mask=cropland_mask,
            postprocess_parameters=postprocess_parameters,
        )

    # Submit the job
    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "1g",
        "executor-memoryOverhead": "1g",
        "python-memory": "3g",
        "soft-errors": "true",
        "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    # Compile filename of final product
    proc_level = "raw" if postprocess_parameters.enable is False else "cleaned"
    filename = f"{product_type.value}-{proc_level}"

    # Execute the job
    job = classes.execute_batch(
        out_format=out_format,
        job_options=JOB_OPTIONS,
        title="WorldCereal [generate_map] job",
        description="Job that performs end-to-end WorldCereal inference",
        filename_prefix=filename,
    )

    # Get job results
    job_result = job.get_results()

    # Get the products
    assets = job_result.get_assets()
    products = {}
    for asset in assets:
        name = asset.name.split("_")[0]
        prod_type = (
            WorldCerealProductType.CROPLAND
            if name.split("-")[0] == "cropland"
            else WorldCerealProductType.CROPTYPE
        )
        if output_path is not None:
            filepath = asset.download(target=Path(output_path))
        else:
            filepath = None
        products[name] = {
            "url": asset.href,
            "type": prod_type,
            "path": filepath,
        }

    # Download job metadata if output path is provided
    if output_path is not None:
        metadata_file = Path(output_path) / "job-results.json"
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
    )

    inputs.execute_batch(
        outputfile=output_path,
        out_format="NetCDF",
        title="WorldCereal [collect_inputs] job",
        description="Job that collects inputs for WorldCereal inference",
        job_options={
            "driver-memory": "4g",
            "executor-memory": "1g",
            "executor-memoryOverhead": "1g",
            "python-memory": "2g",
            "soft-errors": "true",
        },
    )
