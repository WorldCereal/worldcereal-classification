import glob
from pathlib import Path
from typing import Dict, Literal, Optional

import geopandas as gpd
import rasterio
from ipywidgets import Output
from openeo_gfmap import Backend, BackendContext, TemporalContext
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from worldcereal.jobmanager import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    WorldCerealJobManager,
    run_map_production,
)
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET

from .job_manager import notebook_logger, run_notebook_job_manager


def run_map_production_notebook(
    aoi_gdf: gpd.GeoDataFrame,
    output_folder: Path,
    grid_size: int = 20,
    temporal_extent: Optional[TemporalContext] = None,
    season_specifications: Optional[Dict[str, TemporalContext]] = None,
    year: Optional[int] = None,
    product_type: Literal["cropland", "croptype"] = "cropland",
    seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
    seasonal_model_zip: Optional[str] = None,
    enable_cropland_head: Optional[bool] = None,
    landcover_head_zip: Optional[str] = None,
    enable_croptype_head: Optional[bool] = None,
    croptype_head_zip: Optional[str] = None,
    enforce_cropland_gate: Optional[bool] = None,
    export_class_probs: Optional[bool] = None,
    enable_cropland_postprocess: Optional[bool] = None,
    cropland_postprocess_method: Optional[
        Literal["majority_vote", "smooth_probabilities"]
    ] = None,
    cropland_postprocess_kernel_size: Optional[int] = None,
    enable_croptype_postprocess: Optional[bool] = None,
    croptype_postprocess_method: Optional[
        Literal["majority_vote", "smooth_probabilities"]
    ] = None,
    croptype_postprocess_kernel_size: Optional[int] = None,
    target_epsg: Optional[int] = None,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    restart_failed: bool = True,
    job_options: Optional[dict] = None,
    parallel_jobs: int = 2,
    randomize_jobs: bool = True,
    plot_out: Optional[Output] = None,
    log_out: Optional[Output] = None,
    display_outputs: bool = True,
    poll_sleep: int = 60,
    simplify_logging: bool = True,
) -> WorldCerealJobManager:
    """Run a WorldCereal map production for the given AOI and temporal extent.
    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame
        Areas of interest to process. Must have a CRS.
    output_folder : Path
        The directory where the output files will be saved.
    grid_size : int, optional
        The resolution of the tiles in kilometers, by default 20.
    temporal_extent : Optional[TemporalContext], optional
        Temporal context defining the time range for which to collect input data patches.
        If provided together with `year`, temporal_extent will take precedence and override the year.
    season_specifications : Optional[Dict[str, TemporalContext]], optional
        Per-season temporal windows used for inference, by default None.
    year : Optional[int], optional
        Year used for crop calendar inference when dates are missing, by default None.
    product_type : WorldCerealProductType, optional
        The type of product to produce, by default WorldCerealProductType.CROPLAND
    seasonal_preset : str, optional
        Name of the seasonal workflow preset to use when building the inference context.
        This determines the default configuration settings of the inference workflow.
        By default we use the "phase_ii_multitask" preset, which corresponds to the Phase II multitask seasonal backbone with dual landcover/croptype heads.
    seasonal_model_zip : Optional[str], optional
        Path to .zip file of a seasonal Presto model to be used for overriding the default seasonal model.
        By default None, which means the seasonal model embedded in the preset will be used.
    enable_cropland_head : Optional[bool], optional
        Override whether the cropland (landcover) head is enabled. If None, defaults to
        preset behavior unless `product_type` forces cropland-only execution.
    landcover_head_zip : Optional[str], optional
        Path to .zip file of a cropland/landcover head artifact to be used for overriding the default head.
        If None, the head embedded in the preset seasonal model will be used.
    enable_croptype_head : Optional[bool], optional
        Override whether the croptype head is enabled. If None, defaults to preset behavior
        unless `product_type` forces cropland-only execution.
    croptype_head_zip : Optional[str], optional
        Path to .zip file of a croptype head artifact to be used for overriding the default head.
        If None, the head embedded in the preset seasonal model will be used.
    enforce_cropland_gate : Optional[bool], optional
        Whether or not to mask the crop type product with the cropland product.
        If None, use the preset default (True).
    export_class_probs : Optional[bool], optional
        Export per-class probabilities in all products.
        If None, use the preset default (True).
    enable_cropland_postprocess : Optional[bool], optional
        Enable cropland postprocessing. If None, use the preset default (False).
    cropland_postprocess_method : Optional[str], optional
        If None and postprocess is enabled, the preset default is used ("majority_vote").
        Available options are "majority_vote" and "smooth_probabilities".
    cropland_postprocess_kernel_size : Optional[int], optional
        Cropland postprocess kernel size override.
        If None and postprocess is enabled, the preset default is used (5).
    enable_croptype_postprocess : Optional[bool], optional
        Enable croptype postprocessing. If None, use the preset default (False).
    croptype_postprocess_method : Optional[str], optional
        Croptype postprocess method override.
        If None and postprocess is enabled, the preset default is used ("majority_vote").
        Available options are "majority_vote" and "smooth_probabilities".
    croptype_postprocess_kernel_size : Optional[int], optional
        Croptype postprocess kernel size override.
        If None and postprocess is enabled, the preset default is used (5).
    target_epsg : Optional[int], optional
        The target EPSG code for the output, by default None.
        If None, the output will be in the CRS as defined by the tiling grid (local UTM projection).
    backend_context : BackendContext, optional
        The backend context to use for the production, by default BackendContext(Backend.CDSE)
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        The Sentinel-1 orbit state to use for the production, by default None.
        If None, the  orbit state will be automatically determined.
    restart_failed : bool, optional
        Whether to automatically restart failed jobs, by default True.
    job_options : Optional[dict], optional
        Additional options for the job, by default None.
        If None, default options will be used.
    parallel_jobs : int, optional
        The number of parallel jobs to run, by default 2.
    randomize_jobs : bool, optional
        Whether to randomize the order of job execution, by default True.
    plot_out : Optional[Output], optional
        An optional ipywidgets Output widget to use for plotting the job status.
        If None, a new Output widget will be created, by default None.
    log_out : Optional[Output], optional
        An optional ipywidgets Output widget to use for logging the job status.
        If None, a new Output widget will be created, by default None.
    display_outputs : bool, optional
        Whether to display the output widgets in the notebook, by default True.
    poll_sleep : int, optional
        The number of seconds to wait between polling the job status, by default 60
    simplify_logging : bool, optional
        Whether to simplify logging output for better readability in the notebook, by default True.
        If True, openeo job manager logs will be simplified to show only essential information and errors, instead of the full verbose logs.

    Returns
    -------
    WorldCerealJobManager
        The job manager instance that was used to run the production.
        You can use this instance to inspect job statuses and load results after the production has finished.
    """

    # Set up logging and plotting outputs for the notebook
    plot_out, log_out, _log = notebook_logger(
        plot_out=plot_out,
        log_out=log_out,
        display_outputs=display_outputs,
    )

    if display_outputs:
        _log("Job status will be displayed in the output widget below.")
        _log(
            "For individual job tracking, we refer to: https://openeo.dataspace.copernicus.eu/"
        )

    return run_map_production(
        aoi_gdf=aoi_gdf,
        output_dir=output_folder,
        grid_size=grid_size,
        temporal_extent=temporal_extent,
        season_specifications=season_specifications,
        year=year,
        product_type=product_type,
        seasonal_preset=seasonal_preset,
        seasonal_model_zip=seasonal_model_zip,
        enable_cropland_head=enable_cropland_head,
        landcover_head_zip=landcover_head_zip,
        enable_croptype_head=enable_croptype_head,
        croptype_head_zip=croptype_head_zip,
        enforce_cropland_gate=enforce_cropland_gate,
        export_class_probs=export_class_probs,
        enable_cropland_postprocess=enable_cropland_postprocess,
        cropland_postprocess_method=cropland_postprocess_method,
        cropland_postprocess_kernel_size=cropland_postprocess_kernel_size,
        enable_croptype_postprocess=enable_croptype_postprocess,
        croptype_postprocess_method=croptype_postprocess_method,
        croptype_postprocess_kernel_size=croptype_postprocess_kernel_size,
        target_epsg=target_epsg,
        backend_context=backend_context,
        s1_orbit_state=s1_orbit_state,
        restart_failed=restart_failed,
        job_options=job_options,
        parallel_jobs=parallel_jobs,
        randomize_jobs=randomize_jobs,
        poll_sleep=poll_sleep,
        simplify_logging=simplify_logging,
        max_retries=DEFAULT_MAX_RETRIES,
        base_delay=DEFAULT_BASE_DELAY,
        max_delay=DEFAULT_MAX_DELAY,
        log_fn=_log,
        runner=run_notebook_job_manager,
        runner_kwargs={
            "plot_out": plot_out,
            "log_out": log_out,
            "display_outputs": display_outputs,
            "status_title": "Inference job status",
        },
    )


def merge_maps(outdir: Path) -> dict[str, Path]:
    """Merge all product maps in the output directory into .tif files.

    Returns a mapping of product name -> merged output path.
    """

    if not outdir.exists():
        raise FileNotFoundError(f"Output directory {outdir} does not exist.")

    # Find all .tif files in the output directory
    tifs = glob.glob(str(outdir / "*" / "*.tif"))
    if len(tifs) == 0:
        raise FileNotFoundError("No tif files found in the output directory to merge.")

    product_groups: dict[str, list[str]] = {}
    for tif in tifs:
        product = Path(tif).name.split("_")[0]
        product_groups.setdefault(product, []).append(tif)

    merged_outputs: dict[str, Path] = {}

    def _merge_tifs(product_name: str, product_tifs: list[str]) -> Path:
        reprojected_tifs = []
        with rasterio.Env(CPL_LOG="ERROR"):
            for tif in product_tifs:
                # reproject to EPSG:3857 if not already in that CRS
                with rasterio.open(tif) as src:
                    dst_crs = "EPSG:3857"
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds
                    )

                    kwargs = src.meta.copy()
                    kwargs.update(
                        {
                            "crs": dst_crs,
                            "transform": transform,
                            "width": width,
                            "height": height,
                        }
                    )

                    memfile = MemoryFile()
                    with memfile.open(**kwargs) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest,
                            )
                        dst.descriptions = src.descriptions
                    reprojected_tifs.append(memfile.open())

            # Merge all reprojected rasters
            mosaic, out_trans = merge(reprojected_tifs)

            # Use metadata from one of the input files and update
            out_meta = reprojected_tifs[0].meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw",
                }
            )

            # Write to output
            outfile = outdir / f"{product_name}_merged.tif"
            with rasterio.open(outfile, "w", **out_meta) as dest:
                dest.write(mosaic)
                # Preserve band descriptions (if any)
                for idx, desc in enumerate(reprojected_tifs[0].descriptions, start=1):
                    if desc:
                        dest.set_band_description(idx, desc)

        return outfile

    for product_name, product_tifs in product_groups.items():
        merged_outputs[product_name] = _merge_tifs(product_name, product_tifs)

    return merged_outputs
