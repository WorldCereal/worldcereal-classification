"""Executing inference jobs on the OpenEO backend.

Possible entry points for inference in this module:
- `generate_map`: This function is used to generate a map for a single patch.
    It creates one OpenEO job and processes the inference for the specified
    spatial and temporal extent.
- `collect_inputs`: This function is used to collect preprocessed inputs
    without performing inference. It retrieves the required data for further
    processing or analysis.
- `run_largescale_inference`: This function utilizes a job manager to
    orchestrate and execute multiple inference jobs automatically, enabling
    efficient large-scale processing.
- `setup_inference_job_manager`: This function prepares the job manager
    and job database for large-scale inference jobs. It sets up the necessary
    infrastructure to manage and track jobs in a notebook environment.
    Used in the WorldCereal demo notebooks.

"""

import json
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

import geopandas as gpd
import openeo
import pandas as pd
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import CsvJobDatabase, MultiBackendJobManager
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import BACKEND_CONNECTIONS
from pydantic import BaseModel
from typing_extensions import TypedDict

from worldcereal.openeo.mapping import _cropland_map, _croptype_map, _embeddings_map
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs
from worldcereal.parameters import (
    CropLandParameters,
    CropTypeParameters,
    EmbeddingsParameters,
    WorldCerealProductType,
)
from worldcereal.utils.models import load_model_lut

ONNX_DEPS_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_deps_python311.zip"
FEATURE_DEPS_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/torch_deps_python311.zip"
INFERENCE_JOB_OPTIONS = {
    "driver-memory": "4g",
    "executor-memory": "2g",
    "executor-memoryOverhead": "3g",
    "max-executors": 20,
    "python-memory": "disable",
    "soft-errors": 0.1,
    "image-name": "python311",
    "udf-dependency-archives": [
        f"{ONNX_DEPS_URL}#onnx_deps",
        f"{FEATURE_DEPS_URL}#feature_deps",
    ],
}


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


class InferenceJobManager(MultiBackendJobManager):
    """A job manager for executing large-scale WorldCereal inference jobs on the OpenEO backend.
    Based on official MultiBackendJobManager with extension of how results are downloaded
    and named.
    """

    @classmethod
    def generate_output_path_inference(
        cls,
        root_folder: Path,
        geometry_index: int,
        row: pd.Series,
        asset_id: Optional[str] = None,
    ) -> Path:
        """Method to generate the output path for inference jobs.

        Parameters
        ----------
        root_folder : Path
            root folder where the output parquet file will be saved
        geometry_index : int
            For point extractions, only one asset (a geoparquet file) is generated per job.
            Therefore geometry_index is always 0. It has to be included in the function signature
            to be compatible with the GFMapJobManager
        row : pd.Series
            the current job row from the GFMapJobManager
        asset_id : str, optional
            Needed for compatibility with GFMapJobManager but not used.

        Returns
        -------
        Path
            output path for the point extractions parquet file
        """

        tile_name = row.tile_name

        # Create the subfolder to store the output
        subfolder = root_folder / str(tile_name)
        subfolder.mkdir(parents=True, exist_ok=True)

        return subfolder

    def on_job_done(self, job: BatchJob, row):
        logger.info(f"Job {job.job_id} completed")
        output_dir = self.generate_output_path_inference(self._root_dir, 0, row)

        # Get job results
        job_result = job.get_results()

        # Get the products
        assets = job_result.get_assets()
        for asset in assets:
            asset_name = asset.name.split(".")[0].split("_")[0]
            asset_type = asset_name.split("-")[0]
            asset_type = getattr(WorldCerealProductType, asset_type.upper())
            filepath = asset.download(target=output_dir)

            # We want to add the tile name to the filename
            new_filepath = filepath.parent / f"{filepath.stem}_{row.tile_name}.tif"
            shutil.move(filepath, new_filepath)

        job_metadata = job.describe()
        result_metadata = job_result.get_metadata()
        job_metadata_path = output_dir / f"job_{job.job_id}.json"
        result_metadata_path = output_dir / f"result_{job.job_id}.json"

        with job_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(job_metadata, f, ensure_ascii=False)
        with result_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(result_metadata, f, ensure_ascii=False)

        # post_job_action(output_file)
        logger.success("Job completed")


def create_inference_process_graph(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    connection: Optional[openeo.Connection] = None,
) -> List[openeo.DataCube]:
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
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]]
        Sentinel-1 orbit state to use for the inference. If not provided,
        the orbit state will be dynamically determined based on the spatial extent.
    out_format : str, optional
        Output format, by default "GTiff"
    backend_context : BackendContext
        backend to run the job on, by default CDSE.
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.
    target_epsg: Optional[int] = None
        EPSG code to use for the output products. If not provided, the
        default EPSG will be used.
    connection: Optional[openeo.Connection] = None,
        Optional OpenEO connection to use. If not provided, a new connection
        will be created based on the backend_context.

    Returns
    -------
    List[openeo.DataCube]
        A list with one or more result objects or a list of DataCube objects, representing the inference
        process graph. This object can be used to execute the job on the OpenEO backend.
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
    if connection is None:
        connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Preparing the input cube for inference
    inputs = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
        # disable_meteo=True,
    )

    # Spatial filtering
    inputs = inputs.filter_bbox(dict(spatial_extent))

    # Construct the feature extraction and model inference pipeline
    if product_type == WorldCerealProductType.CROPLAND:
        results = _cropland_map(
            inputs,
            temporal_extent,
            cropland_parameters=cropland_parameters,
        )

    elif product_type == WorldCerealProductType.CROPTYPE:
        if not isinstance(croptype_parameters, CropTypeParameters):
            raise ValueError(
                f"Please provide a valid `croptype_parameters` parameter."
                f" Received: {croptype_parameters}"
            )

        # Generate crop type map with optional cropland masking
        results = _croptype_map(
            inputs,
            temporal_extent,
            cropland_parameters=cropland_parameters,
            croptype_parameters=croptype_parameters,
        )

    return results


def create_embeddings_process_graph(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    embeddings_parameters: EmbeddingsParameters = EmbeddingsParameters(),
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    scale_uint16: bool = True,
) -> openeo.DataCube:
    """Create an OpenEO process graph for generating embeddings.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        Spatial extent of the map.
    temporal_extent : TemporalContext
        Temporal range to consider.
    embeddings_parameters : EmbeddingsParameters, optional
        Parameters for the embeddings product inference pipeline, by default EmbeddingsParameters().
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        Sentinel-1 orbit state to use for the inference. If not provided, the orbit state will be dynamically determined based on the spatial extent, by default None.
    out_format : str, optional
        Output format, by default "GTiff".
    backend_context : BackendContext, optional
        Backend to run the job on, by default BackendContext(Backend.CDSE).
    tile_size : Optional[int], optional
        Tile size to use for the data loading in OpenEO, by default 128.
    target_epsg : Optional[int], optional
        EPSG code to use for the output products. If not provided, the default EPSG will be used.
    scale_uint16 : bool, optional
        Whether to scale the embeddings to uint16 for memory optimization, by default True.

    Returns
    -------
    openeo.DataCube
        DataCube object representing the embeddings process graph. This object can be used to execute the job on the OpenEO backend. The result will be a DataCube with the embeddings.

    Raises
    ------
    ValueError
        If the output format is not supported.
    """

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
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
        # disable_meteo=True,
    )

    # Spatial filtering
    inputs = inputs.filter_bbox(dict(spatial_extent))

    embeddings = _embeddings_map(
        inputs,
        temporal_extent,
        embeddings_parameters=embeddings_parameters,
        scale_uint16=scale_uint16,
    )

    # Save the final result
    embeddings = embeddings.save_result(
        format=out_format,
        options=dict(
            filename_prefix=f"WorldCereal_Embeddings_{temporal_extent.start_date}_{temporal_extent.end_date}",
        ),
    )

    return embeddings


def create_inputs_process_graph(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: str = "NetCDF",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    compositing_window: Literal["month", "dekad"] = "month",
) -> openeo.DataCube:
    """Wrapper function that creates the inputs openEO process graph.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]]
        Sentinel-1 orbit state to use for the inference. If not provided,
        the orbit state will be dynamically determined based on the spatial extent.
    out_format : str, optional
        Output format, by default "NetCDF"
    backend_context : BackendContext
        backend to run the job on, by default CDSE.
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.
    target_epsg: Optional[int] = None
        EPSG code to use for the output products. If not provided, the
        default EPSG will be used.
    compositing_window: Literal["month", "dekad"]
        Compositing window to use for the data loading in OpenEO, by default
        "month".

    Returns
    -------
    openeo.DataCube
        DataCube object representing the inputs process graph.
        This object can be used to execute the job on the OpenEO backend.
        The result will be a DataCube with the preprocessed inputs.

    Raises
    ------
    ValueError
        if the out_format is not supported
    """

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
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
        compositing_window=compositing_window,
        # disable_meteo=True,
    )

    # Spatial filtering
    inputs = inputs.filter_bbox(dict(spatial_extent))

    # Save the final result
    inputs = inputs.save_result(
        format=out_format,
        options=dict(
            filename_prefix=f"preprocessed-inputs_{temporal_extent.start_date}_{temporal_extent.end_date}",
        ),
    )

    return inputs


def create_inference_job(
    row: pd.Series,
    connection: openeo.Connection,
    provider: str,
    connection_provider: str,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPTYPE,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    target_epsg: Optional[int] = None,
    job_options: Optional[dict] = None,
) -> BatchJob:
    """Create an OpenEO batch job for WorldCereal inference.

    Parameters
    ----------
    row : pd.Series
        _description_
        Contains at least the following fields:
        - start_date: str, start date of the temporal extent
        - end_date: str, end date of the temporal extent
        - geometry: shapely.geometry, geometry of the spatial extent
        - tile_name: str, name of the tile
        - epsg: int, EPSG code of the spatial extent
        - bounds_epsg: str representation of tuple,
                        bounds of the spatial extent in CRS as
                        specified by epsg attribute
    connection : openeo.Connection
        openEO connection to the backend
    provider : str
        unused but required for compatibility with MultiBackendJobManager
    connection_provider : str
        unused but required for compatibility with MultiBackendJobManager6
    product_type : WorldCerealProductType, optional
        Type of the WorldCereal product to generate, by default WorldCerealProductType.CROPTYPE
    croptype_parameters :  Optional[CropTypeParameters], optional
        Parameters for the croptype product inference pipeline. Only required
        whenever `product_type` is set to `WorldCerealProductType.CROPTYPE`,
        will be ignored otherwise, by default None
    cropland_parameters : Optional[CropLandParameters], optional
        Parameters for the cropland product inference pipeline, by default None
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        Sentinel-1 orbit state to use for the inference. If not provided, the
        best orbit will be dynamically derived from the catalogue.
    target_epsg : Optional[int], optional
        EPSG code to reproject the data to. If not provided, the data will be
        left in the original epsg as mentioned in the row.
    job_options : Optional[dict], optional
        Additional job options to pass to the OpenEO backend, by default None

    Returns
    -------
    BatchJob
        Batch job created on openEO backend.
    """

    # Get temporal and spatial extents from the row
    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)
    epsg = int(row.epsg)
    bounds = eval(row.bounds_epsg)
    spatial_extent = BoundingBoxExtent(
        west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3], epsg=epsg
    )

    if target_epsg is None:
        # If no target EPSG is provided, use the EPSG from the row
        target_epsg = epsg

    # Update default job options with the provided ones
    inference_job_options = deepcopy(INFERENCE_JOB_OPTIONS)
    if job_options is not None:
        inference_job_options.update(job_options)

    inference_result = create_inference_process_graph(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        product_type=product_type,
        croptype_parameters=croptype_parameters,
        cropland_parameters=cropland_parameters,
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
        connection=connection,
    )

    # Submit the job
    return connection.create_job(
        inference_result,
        title=f"WorldCereal [{product_type.value}] job_{row.tile_name}",
        description="Job that performs end-to-end WorldCereal inference",
        additional=inference_job_options,  # TODO: once openeo-python-client supports job_options, use that
    )


def generate_map(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    output_dir: Optional[Union[Path, str]] = None,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    job_options: Optional[dict] = None,
    target_epsg: Optional[int] = None,
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
    out_format : str, optional
        Output format, by default "GTiff"
    backend_context : BackendContext
        backend to run the job on, by default CDSE.
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.
    job_options: dict, optional
        Additional job options to pass to the OpenEO backend, by default None
    target_epsg: Optional[int] = None
        EPSG code to use for the output products. If not provided, the
        default EPSG will be used.

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

    # Get a connection to the OpenEO backend
    connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Create the process graph
    results = create_inference_process_graph(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        product_type=product_type,
        cropland_parameters=cropland_parameters,
        croptype_parameters=croptype_parameters,
        out_format=out_format,
        backend_context=backend_context,
        tile_size=tile_size,
        target_epsg=target_epsg,
        connection=connection,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Submit the job
    inference_job_options = deepcopy(INFERENCE_JOB_OPTIONS)
    if job_options is not None:
        inference_job_options.update(job_options)

    # Execute the job
    job = connection.create_job(
        results,
        additional=inference_job_options,  # TODO: once openeo-python-client supports job_options, use that
        title=f"WorldCereal [{product_type.value}] job",
        description="Job that performs end-to-end WorldCereal inference",
    ).start_and_wait()

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


def run_largescale_inference(
    production_grid: Union[Path, gpd.GeoDataFrame],
    output_dir: Union[Path, str],
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    target_epsg: Optional[int] = None,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    job_options: Optional[dict] = None,
    parallel_jobs: int = 2,
):
    """
    Run large-scale inference jobs on the OpenEO backend.
    This function orchestrates the execution of large-scale inference jobs
    using a production grid (either a Parquet file or a GeoDataFrame) and specified parameters.
    It manages job creation, tracking, and execution on the OpenEO backend.

    Parameters
    ----------
    production_grid : Union[Path, gpd.GeoDataFrame]
        Path to the production grid file in Parquet format or a GeoDataFrame.
        The grid must contain the required attributes: 'start_date', 'end_date',
        'geometry', 'tile_name', 'epsg' and 'bounds_epsg'.
    output_dir : Union[Path, str]
        Directory where output files and job tracking information will be stored.
    product_type : WorldCerealProductType
        Type of product to generate. Defaults to WorldCerealProductType.CROPLAND.
    cropland_parameters : CropLandParameters
        Parameters for cropland inference.
    croptype_parameters : CropTypeParameters
        Parameters for crop type inference.
    backend_context : BackendContext
        Context for the backend to use. Defaults to BackendContext(Backend.CDSE).
    target_epsg : Optional[int]
        EPSG code for the target coordinate reference system.
        If None, no reprojection will be performed.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]]
        Sentinel-1 orbit state to use ('ASCENDING' or 'DESCENDING')
        If None, no specific orbit state is enforced.
    job_options : Optional[dict]
        Additional options for configuring the inference jobs. Defaults to None.
    parallel_jobs : int
        Number of parallel jobs to manage on the backend. Defaults to 2. Note that load
        balancing does not guarantee that all jobs will run in parallel.

    Returns
    -------
    None
    """

    job_manager, job_db, start_job = setup_inference_job_manager(
        production_grid=production_grid,
        output_dir=output_dir,
        product_type=product_type,
        cropland_parameters=cropland_parameters,
        croptype_parameters=croptype_parameters,
        backend_context=backend_context,
        target_epsg=target_epsg,
        s1_orbit_state=s1_orbit_state,
        job_options=job_options,
        parallel_jobs=parallel_jobs,
    )

    job_df = job_db.df
    job_tracking_csv = job_db.path

    # Run the jobs
    job_manager.run_jobs(
        df=job_df,
        start_job=start_job,
        job_db=job_tracking_csv,
    )

    logger.info("Job manager finished.")


def setup_inference_job_manager(
    production_grid: Union[Path, gpd.GeoDataFrame],
    output_dir: Union[Path, str],
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    target_epsg: Optional[int] = None,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    job_options: Optional[dict] = None,
    parallel_jobs: int = 2,
) -> tuple[InferenceJobManager, CsvJobDatabase, Callable]:
    """
    Prepare large-scale inference jobs on the OpenEO backend.
    This function sets up the job manager, creates job tracking information,
    and defines the job creation function for WorldCereal inference jobs.

    Parameters
    ----------
    production_grid : Union[Path, gpd.GeoDataFrame]
        Path to the production grid file in Parquet format or a GeoDataFrame.
        The grid must contain the required attributes: 'start_date', 'end_date',
        'geometry', 'tile_name', 'epsg' and 'bounds_epsg'.
    output_dir : Union[Path, str]
        Directory where output files and job tracking information will be stored.
    product_type : WorldCerealProductType
        Type of product to generate. Defaults to WorldCerealProductType.CROPLAND.
    cropland_parameters : CropLandParameters
        Parameters for cropland inference.
    croptype_parameters : CropTypeParameters
        Parameters for crop type inference.
    backend_context : BackendContext
        Context for the backend to use. Defaults to BackendContext(Backend.CDSE).
    target_epsg : Optional[int]
        EPSG code for the target coordinate reference system.
        If None, no reprojection will be performed.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]]
        Sentinel-1 orbit state to use ('ASCENDING' or 'DESCENDING')
        If None, no specific orbit state is enforced.
    job_options : Optional[dict]
        Additional options for configuring the inference jobs. Defaults to None.
    parallel_jobs : int
        Number of parallel jobs to manage on the backend. Defaults to 2. Note that load
        balancing does not guarantee that all jobs will run in parallel.

    Returns
    -------
    tuple[InferenceJobManager, CsvJobDatabase, callable]
        A tuple containing:
        - InferenceJobManager: The job manager for handling inference jobs.
        - CsvJobDatabase: The job database for tracking job information.
        - callable: A function to create individual inference jobs.

    Raises
    -------
    AssertionError:
        If the production grid does not contain the required attributes.
    """

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make a connection to the OpenEO backend
    backend = backend_context.backend
    connection = BACKEND_CONNECTIONS[backend]()

    # Setup the job manager
    logger.info("Setting up the job manager.")
    manager = InferenceJobManager(root_dir=output_dir)
    manager.add_backend(
        backend.value, connection=connection, parallel_jobs=parallel_jobs
    )

    # Configure job tracking CSV file
    job_tracking_csv = output_dir / "job_tracking.csv"

    job_db = CsvJobDatabase(path=job_tracking_csv)
    if not job_db.exists():
        logger.info("Job tracking file does not exist, creating new jobs.")

        if isinstance(production_grid, Path):
            production_gdf = gpd.read_parquet(production_grid)
        elif isinstance(production_grid, gpd.GeoDataFrame):
            production_gdf = production_grid
        else:
            raise ValueError("production_grid must be a Path or a GeoDataFrame.")

        REQUIRED_ATTRIBUTES = [
            "start_date",
            "end_date",
            "geometry",
            "tile_name",
            "epsg",
            "bounds_epsg",
        ]
        for attr in REQUIRED_ATTRIBUTES:
            assert (
                attr in production_gdf.columns
            ), f"The production grid must contain a '{attr}' column."

        job_df = production_gdf[REQUIRED_ATTRIBUTES].copy()

        df = manager._normalize_df(job_df)
        # Save the job tracking DataFrame to the job database
        job_db.persist(df)

    else:
        logger.info("Job tracking file already exists, skipping job creation.")

    # Define the job creation function
    start_job = partial(
        create_inference_job,
        product_type=product_type,
        cropland_parameters=cropland_parameters,
        croptype_parameters=croptype_parameters,
        s1_orbit_state=s1_orbit_state,
        job_options=job_options,
        target_epsg=target_epsg,
    )

    # Check if there are jobs to run
    if job_db.df.empty:
        logger.warning("No jobs to run. The job tracking CSV is empty.")
        raise ValueError(
            "No jobs to run. The job tracking CSV is empty. "
            "Please check the production grid and ensure it contains valid data."
        )

    return manager, job_db, start_job
