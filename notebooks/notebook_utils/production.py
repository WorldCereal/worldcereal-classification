import glob
import json
import logging
from pathlib import Path
from typing import Dict, Literal, Optional

import geopandas as gpd
import rasterio
from ipywidgets import Output
from openeo_gfmap import Backend, BackendContext, TemporalContext
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from worldcereal.job import WorldCerealTask
from worldcereal.jobmanager import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    WorldCerealJobManager,
)
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.openeo.workflow_config import WorldCerealWorkflowConfig
from worldcereal.parameters import WorldCerealProductType

from .job_manager import notebook_logger, run_notebook_job_manager


def run_map_production(
    aoi_gdf: gpd.GeoDataFrame,
    output_folder: Path,
    grid_size: int = 20,
    temporal_extent: Optional[TemporalContext] = None,
    season_specifications: Optional[Dict[str, TemporalContext]] = None,
    year: Optional[int] = None,
    seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
    workflow_config: Optional[WorldCerealWorkflowConfig] = None,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
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
    seasonal_preset : str, optional
        Name of the seasonal workflow preset to use when building the inference context.
    workflow_config : Optional[WorldCerealWorkflowConfig], optional
        Structured overrides applied on top of the preset to tweak model/runtime/season settings.
    product_type : WorldCerealProductType, optional
        The type of product to produce, by default WorldCerealProductType.CROPLAND
    target_epsg : Optional[int], optional
        The target EPSG code for the output, by default None.
        If None, the output will be in the CRS as defined by the tiling grid.
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

    _log("------------------------------------")
    _log("STARTING WORKFLOW: Map production")
    _log("------------------------------------")
    _log("----- Workflow configuration -----")

    if temporal_extent is not None:
        temporal_extent_str = (
            f"{temporal_extent.start_date} to {temporal_extent.end_date}"
        )
    else:
        temporal_extent_str = "None"

    params = {
        "output_folder": str(output_folder),
        "number of AOI features": len(aoi_gdf),
        "grid_size": grid_size,
        "temporal_extent": temporal_extent_str,
        "season_specifications": season_specifications,
        "year": year,
        "seasonal_preset": seasonal_preset,
        "product_type": product_type,
        "target_epsg": target_epsg,
        "s1_orbit_state": s1_orbit_state,
        "restart_failed": restart_failed,
        "parallel_jobs": parallel_jobs,
        "randomize_jobs": randomize_jobs,
        "job_options": job_options,
        "poll_sleep": poll_sleep,
        "simplify_logging": simplify_logging,
    }
    for key, value in params.items():
        _log(f"{key}: {value}")

    _log("Detailed workflow configuration:")
    _log(json.dumps(workflow_config.to_dict(), indent=2))
    _log("----------------------------------")

    # Prepare the job manager and job database
    _log("Setting up job manager to handle map production...")
    job_manager = WorldCerealJobManager(
        output_dir=output_folder,
        task=WorldCerealTask.INFERENCE,
        aoi_gdf=aoi_gdf,
        backend_context=backend_context,
        temporal_extent=temporal_extent,
        grid_size=grid_size,
        year=year,
        season_specifications=season_specifications,
        poll_sleep=poll_sleep,
    )

    if simplify_logging:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )

    # Start a threaded job manager with optional live status updates
    _log("Starting map production...")
    if display_outputs:
        _log("Job status will be displayed in the output widget below.")
        _log(
            "For individual job tracking, we refer to: https://openeo.dataspace.copernicus.eu/"
        )

    break_msg = (
        "Stopping map production...\n"
        "Make sure to manually cancel any running jobs in the backend to avoid unnecessary costs!\n"
        "For this, visit the job tracking page in the backend dashboard: https://openeo.dataspace.copernicus.eu/\n"
    )

    try:
        run_notebook_job_manager(
            job_manager,
            run_kwargs={
                "restart_failed": restart_failed,
                "randomize_jobs": randomize_jobs,
                "product_type": product_type,
                "s1_orbit_state": s1_orbit_state,
                "job_options": job_options,
                "target_epsg": target_epsg,
                "parallel_jobs": parallel_jobs,
                "seasonal_preset": seasonal_preset,
                "workflow_config": workflow_config,
                "max_retries": DEFAULT_MAX_RETRIES,
                "base_delay": DEFAULT_BASE_DELAY,
                "max_delay": DEFAULT_MAX_DELAY,
            },
            plot_out=plot_out,
            log_out=log_out,
            display_outputs=display_outputs,
            status_title="Inference job status",
        )
    except KeyboardInterrupt:
        _log(break_msg)
        job_manager.stop_job_thread()
        _log("Map production has stopped.")
        raise

    _log("All done!")
    _log(f"Results stored in {output_folder}")

    return job_manager


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
