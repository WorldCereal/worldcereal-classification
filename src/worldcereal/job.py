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

"""

import json
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import geopandas as gpd
import openeo
import pandas as pd
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import MultiBackendJobManager
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

ONNX_DEPS_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_deps_python311.zip"
FEATURE_DEPS_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/torch_deps_python311.zip"
INFERENCE_JOB_OPTIONS = {
    "driver-memory": "4g",
    "executor-memory": "2g",
    "executor-memoryOverhead": "1g",
    "python-memory": "3g",
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
    postprocess_parameters: PostprocessParameters = PostprocessParameters(),
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
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
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
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


def create_inference_job(
    row: pd.Series,
    connection: openeo.Connection,
    provider: str,
    connection_provider: str,
    epsg: Optional[int] = 4326,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPTYPE,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    postprocess_parameters: PostprocessParameters = PostprocessParameters(),
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
    connection : openeo.Connection
        openEO connection to the backend
    provider : str
        unused but required for compatibility with MultiBackendJobManager
    connection_provider : str
        unused but required for compatibility with MultiBackendJobManager
    epsg : int, optional
        EPSG code for the spatial extent of the job, by default 4326
    product_type : WorldCerealProductType, optional
        Type of the WorldCereal product to generate, by default WorldCerealProductType.CROPTYPE
    croptype_parameters :  Optional[CropTypeParameters], optional
        Parameters for the croptype product inference pipeline. Only required
        whenever `product_type` is set to `WorldCerealProductType.CROPTYPE`,
        will be ignored otherwise, by default None
    cropland_parameters : Optional[CropLandParameters], optional
        Parameters for the cropland product inference pipeline, by default None
    postprocess_parameters : Optional[PostprocessParameters], optional
        Parameters for the postprocessing pipeline. By default disabled.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        Sentinel-1 orbit state to use for the inference. If not provided, the
        best orbit will be dynamically derived from the catalogue.
    target_epsg : Optional[int], optional
        EPSG code to reproject the data to. If not provided, the data will be
        left in the original coordinate reference system (UTM).
    job_options : Optional[dict], optional
        Additional job options to pass to the OpenEO backend, by default None

    Returns
    -------
    BatchJob
        Batch job created on openEO backend.
    """

    # Get temporal and spatial extents from the row
    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)
    spatial_extent = BoundingBoxExtent(*row.geometry.bounds, epsg=epsg)

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
        postprocess_parameters=postprocess_parameters,
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
    )

    # Submit the job
    return inference_result.create_job(
        title=f"WorldCereal [{product_type.value}] job",
        description="Job that performs end-to-end WorldCereal inference",
        job_options=inference_job_options,
    )


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
    inference_job_options = deepcopy(INFERENCE_JOB_OPTIONS)
    if job_options is not None:
        inference_job_options.update(job_options)

    # Execute the job
    job = classes.execute_batch(
        job_options=inference_job_options,
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


def run_largescale_inference(
    production_grid: Union[Path, gpd.GeoDataFrame],
    output_dir: Union[Path, str],
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    postprocess_parameters: PostprocessParameters = PostprocessParameters(),
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
        'geometry', and 'tile_name'.
    output_dir : Union[Path, str]
        Directory where output files and job tracking information will be stored.
    product_type : WorldCerealProductType
        Type of product to generate. Defaults to WorldCerealProductType.CROPLAND.
    cropland_parameters : CropLandParameters
        Parameters for cropland inference.
    croptype_parameters : CropTypeParameters
        Parameters for crop type inference.
    postprocess_parameters : PostprocessParameters
        Parameters for postprocessing the inference results.
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

    Raises
    -------
    AssertionError:
        If the production grid does not contain the required attributes.
    """

    # Setup output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Configure job tracking CSV file
    job_tracking_csv = output_dir / "job_tracking.csv"

    if job_tracking_csv.is_file():
        logger.info("Job tracking file already exists, skipping job creation.")
        job_df = pd.read_csv(job_tracking_csv)
    else:
        logger.info("Job tracking file does not exist, creating new jobs.")

        if isinstance(production_grid, Path):
            production_gdf = gpd.read_parquet(production_grid)
        elif isinstance(production_grid, gpd.GeoDataFrame):
            production_gdf = production_grid
        else:
            raise ValueError("production_grid must be a Path or a GeoDataFrame.")
        if target_epsg is not None:
            production_gdf = production_gdf.to_crs(epsg=target_epsg)

        REQUIRED_ATTRIBUTES = ["start_date", "end_date", "geometry", "tile_name"]
        for attr in REQUIRED_ATTRIBUTES:
            assert (
                attr in production_gdf.columns
            ), f"The production grid must contain a '{attr}' column."

        job_df = production_gdf[REQUIRED_ATTRIBUTES].copy()

    # Make a connection to the OpenEO backend
    connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Setup the job manager
    logger.info("Setting up the job manager.")
    manager = InferenceJobManager(root_dir=output_dir)
    manager.add_backend("cdse", connection=connection, parallel_jobs=parallel_jobs)

    # Run the jobs
    manager.run_jobs(
        df=job_df,
        start_job=partial(
            create_inference_job,
            epsg=target_epsg,
            product_type=product_type,
            cropland_parameters=cropland_parameters,
            croptype_parameters=croptype_parameters,
            postprocess_parameters=postprocess_parameters,
            s1_orbit_state=s1_orbit_state,
            target_epsg=target_epsg,
            job_options=job_options,
        ),
        job_db=job_tracking_csv,
    )

    logger.info("Job manager finished.")
