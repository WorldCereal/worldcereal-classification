"""Executing inference jobs on the OpenEO backend."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import openeo
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import BACKEND_CONNECTIONS
from pydantic import BaseModel
from pyproj import Transformer
from typing_extensions import TypedDict

from worldcereal.openeo.mapping import _cropland_map, _croptype_map
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs
from worldcereal.parameters import (
    CropLandParameters,
    CropTypeParameters,
    PostprocessParameters,
    WorldCereal2021ProductType,
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

    if product_type not in WorldCerealProductType:
        raise ValueError(f"Product {product_type.value} not supported.")

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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
        # disable_meteo=True,
    )

    # Explicit filtering again for bbox because of METEO low
    # resolution causing issues
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
        cropland_mask = cropland_mask.filter_bands("classification")

        # Generate crop type map
        classes = _croptype_map(
            inputs,
            temporal_extent,
            croptype_parameters=croptype_parameters,
            cropland_mask=cropland_mask,
            postprocess_parameters=postprocess_parameters,
        )

    # Submit the job
    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "2g",
        "executor-memoryOverhead": "1g",
        "python-memory": "3g",
        "soft-errors": "true",
        "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    # Execute the job
    job = classes.execute_batch(
        out_format=out_format,
        job_options=JOB_OPTIONS,
        title=f"WorldCereal [{product_type.value}] job",
        description="Job that performs end-to-end WorldCereal inference",
        filename_prefix=f"{product_type.value}_{temporal_extent.start_date}_{temporal_extent.end_date}",
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

    JOB_OPTIONS = {
        "driver-memory": "4g",
        "executor-memory": "1g",
        "executor-memoryOverhead": "1g",
        "python-memory": "2g",
        "soft-errors": "true",
    }
    if job_options is not None:
        JOB_OPTIONS.update(job_options)

    inputs.execute_batch(
        outputfile=output_path,
        out_format="NetCDF",
        title="WorldCereal [collect_inputs] job",
        description="Job that collects inputs for WorldCereal inference",
        job_options=job_options,
    )


def download_official_product(
    spatial_extent: BoundingBoxExtent,
    product_type: str,
    output_dir: Path,
    season: Optional[str] = None,
    product_collection: str = "2021",
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.FED),
    job_options: Optional[dict] = None,
) -> List[Path]:
    """Download an official WorldCereal product.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        desired spatial extent of the product
    product_type : str
        type of the product to download
    output_dir : Path
        directory to save the downloaded product
    season : Optional[str], optional
        season of the product, by default None (meaning all seasons)
    product_collection : str, optional
        collection of the product, by default "2021"
    out_format : str, optional
        output format, by default "GTiff"
    backend_context : BackendContext
        backend to run the job on, by default FED
    job_options: dict, optional
        Additional job options to pass to the OpenEO backend, by default None

    Returns
    -------
    List[Path]
        Paths to downloaded products

    Raises
    ------
    ValueError
        if the product collection or type is not supported
    """

    # Check product collection and product type are valid
    if product_collection == "2021":
        if product_type not in WorldCereal2021ProductType.__members__:
            raise ValueError(f"Product {product_type} not supported.")
    else:
        raise ValueError(f"Product collection {product_collection} not supported.")

    # Establish connection to the OpenEO backend
    connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # get lat, lon centroid of extent
    transformer = Transformer.from_crs(
        f"EPSG:{spatial_extent.epsg}", "EPSG:4326", always_xy=True
    )
    minx, miny = transformer.transform(spatial_extent.west, spatial_extent.south)
    maxx, maxy = transformer.transform(spatial_extent.east, spatial_extent.north)
    lat = (maxy + miny) / 2
    lon = (maxx + minx) / 2
    location = f"lat={lat:.2f}, lon={lon:.2f}"

    # Set properties and define job title
    if season is not None:
        properties = [openeo.collection_property("productGroupId") == season]
        job_title = f"WorldCereal_Download_{product_collection}_{product_type}_{season}_{location}"
    else:
        properties = []
        job_title = (
            f"WorldCereal_Download_{product_collection}_{product_type}_{location}"
        )

    # Load the collection
    products = connection.load_collection(
        f"ESA_WORLDCEREAL_{product_type}",
        spatial_extent=dict(spatial_extent),
        properties=properties,
    )

    # In case season is specified, reduce temporal dimension
    if season is not None:
        products = products.reduce_dimension(dimension="t", reducer="mean")
        filename_prefix = f"{product_type}_{season}"
    elif product_type.lower() == WorldCereal2021ProductType.TEMPORARYCROPS.value:
        # in case of temporarycrops product, reduce temporal dimension
        products = products.reduce_dimension(dimension="t", reducer="mean")
        filename_prefix = product_type
    else:
        # possibly multiple seasons, no need to reduce temporal dimension
        filename_prefix = product_type

    # Execute the job
    job = products.execute_batch(
        title=job_title,
        out_format=out_format,
        job_options=job_options,
        filename_prefix=filename_prefix,
    )

    # Download the results and return paths
    return job.get_results().download_files(target=output_dir)
