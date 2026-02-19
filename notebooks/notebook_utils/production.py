import copy
import glob
import json
import sys
import time
from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd
import pandas as pd
import plotly.express as px
import rasterio
from IPython.display import display
from ipywidgets import Output
from loguru import logger
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from pandas.errors import EmptyDataError
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box

from worldcereal.job import setup_inference_job_manager
from worldcereal.utils.production_grid import create_production_grid
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.openeo.workflow_config import WorldCerealWorkflowConfig
from worldcereal.parameters import WorldCerealProductType

# Define the color mapping for job statuses
JOB_STATUS_COLORS = {
    "not_started": "grey",
    "created": "gold",
    "queued": "lightsteelblue",
    "queued_for_start": "lightsteelblue",
    "running": "navy",
    "finished": "lime",
    "error": "darkred",
    "skipped": "darkorange",
    "start_failed": "red",
    None: "black",  # Default color for any undefined status
}


def plot_job_status(status_df, color_dict, center, zoom=10):
    """Plot the job status on a map using Plotly Express.

    Parameters
    ----------
    status_df : pd.DataFrame
        DataFrame containing job status information with a 'geometry' column in WKT format.
        The 'status' column should contain the job status values.
        Each row should represent a job with its status and geometry.
    color_dict : dict
        Dictionary mapping job statuses to colors for visualization.
        Keys should be the job status values, and values should be the corresponding colors.
        If a status is not in the dictionary, it will default to the color for `None`.
    center : dict
        Dictionary with 'lat' and 'lon' keys to specify the center of the map.
        Example: {'lat': 52.5, 'lon': 13.4}
        This will be used to center the map view.
    zoom : int, optional
        Plot zoom level, by default 12

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly figure object containing the choropleth map of job statuses.
    """
    status_plot = copy.deepcopy(status_df)
    status_plot.crs = "EPSG:4326"
    status_plot["color"] = (
        status_plot["status"].map(color_dict).fillna(color_dict[None])
    )

    # Convert the entire GeoDataFrame to a FeatureCollection
    geojson = status_plot.set_index("tile_name").__geo_interface__

    fig = px.choropleth_mapbox(
        status_plot,
        geojson=geojson,
        locations="tile_name",
        color="status",
        color_discrete_map=color_dict,
        mapbox_style="carto-positron",
        center=center,
        zoom=zoom,
        title="Job Status Overview",
    )
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    return fig


def run_map_production(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    output_dir: Path,
    tile_resolution: int = 20,
    tiling_crs: Optional[str] = None,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    target_epsg: Optional[int] = None,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    job_options: Optional[dict] = None,
    parallel_jobs: int = 2,
    seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
    workflow_config: Optional[WorldCerealWorkflowConfig] = None,
    stop_event=None,
    plot_out: Optional[Output] = None,
    log_out: Optional[Output] = None,
    display_outputs: bool = True,
    poll_sleep: int = 60,
) -> pd.DataFrame:
    """Run a WorldCereal map production for the given spatial and temporal extent.
    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        The spatial extent for the production grid.
        Must be in WGS84 (EPSG:4326) coordinate reference system.
    temporal_extent : TemporalContext
        The temporal extent for the production grid.
        Must have 'start_date' and 'end_date' attributes.
    output_dir : Path
        The directory where the output files will be saved.
    tile_resolution : int, optional
        The resolution of the tiles in kilometers, by default 20.
    tiling_crs : Optional[str], optional
        The coordinate reference system to use for tiling.
        If None, the best local UTM CRS will be derived.
    product_type : WorldCerealProductType, optional
        The type of product to produce, by default WorldCerealProductType.CROPLAND
    backend_context : BackendContext, optional
        The backend context to use for the production, by default BackendContext(Backend.CDSE)
    target_epsg : Optional[int], optional
        The target EPSG code for the output, by default None.
        If None, the output will be in the CRS as defined by the tiling grid.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        The Sentinel-1 orbit state to use for the production, by default None.
        If None, the default orbit state will be used.
    job_options : Optional[dict], optional
        Additional options for the job, by default None.
        If None, default options will be used.
    parallel_jobs : int, optional
        The number of parallel jobs to run, by default 2.
    seasonal_preset : str, optional
        Name of the seasonal workflow preset to use when building the inference context.
    workflow_config : Optional[WorldCerealWorkflowConfig], optional
        Structured overrides applied on top of the preset to tweak model/runtime/season settings.
    stop_event : Optional[threading.Event], optional
        An optional threading event to stop the job manager gracefully, by default None.
    plot_out : Optional[Output], optional
        An optional ipywidgets Output widget to use for plotting the job status.
        If None, a new Output widget will be created, by default None.
    log_out : Optional[Output], optional
        An optional ipywidgets Output widget to use for logging the job status.
        If None, a new Output widget will be created, by default None.
    display_outputs : bool, optional
        Whether to display the output widgets in the notebook, by default True.
    poll_sleep : int, optional
        The number of seconds to wait between polling the job status, by default 60.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the final job status.
    Raises
    ------
    ValueError
        If the spatial extent is not in WGS84 (EPSG:4326) coordinate reference system.
    """
    # Configure logger of job manager to output to stdout
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

    # Create output widgets (if not provided) and track if we created them
    plot_out_created = plot_out is None
    log_out_created = log_out is None

    if plot_out is None:
        plot_out = Output()
    if log_out is None:
        log_out = Output()

    # Only display widgets if we created them internally
    if display_outputs and (plot_out_created or log_out_created):
        if log_out_created:
            display(log_out)
        if plot_out_created:
            display(plot_out)

    # Helper function to log messages either to widget or logger
    def _log(message: str):
        if display_outputs:
            with log_out:
                print(message)
        else:
            print(message)

    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True, parents=True)
    _log(f"Output directory set to: {output_dir.resolve()}")

    # Some additional logs about the processing parameters
    _log(
        f"Full processing period: {temporal_extent.start_date} to {temporal_extent.end_date}"
    )

    _log("Detailed processing parameters:")
    _log(json.dumps(workflow_config.to_dict(), indent=2))
    _log("----------------------------------")

    # Create production grid according to tile resolution and spatial extent
    grid_file = output_dir / "production_grid.gpkg"
    if grid_file.exists():
        _log(f"Loading existing production grid from {grid_file}")
        production_grid = gpd.read_file(grid_file)
    else:
        _log(f"Creating new production grid with resolution {tile_resolution} km.")
        production_grid = create_production_grid(
            spatial_extent,
            temporal_extent,
            resolution=tile_resolution,
            tiling_crs=tiling_crs,
        )
        production_grid.to_file(grid_file, driver="GPKG")

    # Prepare the job manager and job database
    _log("Setting up job manager to handle map production...")
    job_manager, job_db, start_job = setup_inference_job_manager(
        production_grid,
        output_dir,
        product_type=product_type,
        backend_context=backend_context,
        target_epsg=target_epsg,
        s1_orbit_state=s1_orbit_state,
        job_options=job_options,
        parallel_jobs=parallel_jobs,
        seasonal_preset=seasonal_preset,
        workflow_config=workflow_config,
    )

    # Start a threaded job manager
    _log("Starting map production...")
    if display_outputs:
        _log("Job status will be displayed in the output widget below.")
        _log(
            "For individual job tracking, we refer to: https://openeo.dataspace.copernicus.eu/"
        )
    job_manager.start_job_thread(start_job=start_job, job_db=job_db)

    # Compute center of total bounding box (used for map centering)
    minx, miny, maxx, maxy = production_grid.total_bounds
    center = {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2}

    # Initialize status_df to None
    status_df = None

    break_msg = (
        "Stopping map production...\n"
        "Make sure to manually cancel any running jobs in the backend to avoid unnecessary costs!\n"
        "For this, visit the job tracking page in the backend dashboard: https://openeo.dataspace.copernicus.eu/\n"
    )

    try:
        # Monitor while the job manager thread is alive
        while job_manager._thread and job_manager._thread.is_alive():
            # Check if external stop event was triggered
            if stop_event is not None and stop_event.is_set():
                _log(break_msg)
                job_manager.stop_job_thread()
                break

            try:
                status_df = job_db.read()

                # Update plot and job status table with current status
                if display_outputs:
                    fig = plot_job_status(
                        status_df=status_df, color_dict=JOB_STATUS_COLORS, center=center
                    )
                    status_msg = f"[{time.strftime('%H:%M:%S')}] Job status:\n{status_df['status'].value_counts().to_string()}\n"
                    with plot_out:
                        plot_out.clear_output(wait=True)
                        fig.show(renderer="notebook")
                        print(status_msg)

                time.sleep(poll_sleep)

            except (EmptyDataError, pd.errors.EmptyDataError):
                _log(
                    f"[{time.strftime('%H:%M:%S')}] Job database is empty, waiting for jobs to start..."
                )
                time.sleep(10)
                continue

    except KeyboardInterrupt:
        _log(break_msg)
        job_manager.stop_job_thread()

    _log("Map production has stopped.")

    # Get final status
    try:
        status_df = job_db.read()
    except Exception as e:
        _log(f"Warning: Could not read final job status: {e}")
        if status_df is None:
            status_df = pd.DataFrame()  # Return empty DataFrame if we never got data

    return status_df


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


def bbox_extent_to_gdf(extent: BoundingBoxExtent, outfile: Path) -> gpd.GeoDataFrame:
    """Save drawn bounding box to a geodataframe and save it."""

    # create output directory if it does not exist
    outfile.parent.mkdir(exist_ok=True, parents=True)

    # convert to a GeoDataFrame
    bbox_geom = box(
        extent.west,
        extent.south,
        extent.east,
        extent.north,
    )
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=extent.epsg)
    bbox_gdf.to_file(outfile, driver="GPKG")


def gdf_to_bbox_extent(gdf: Union[gpd.GeoDataFrame, Path]) -> BoundingBoxExtent:
    """Convert a GeoDataFrame with a single geometry to a BoundingBoxExtent.
    Parameters
    ----------
    gdf : gpd.GeoDataFrame or Path
        A GeoDataFrame with a single geometry or a file path to a GeoDataFrame.
        The GeoDataFrame must have a defined CRS and contain exactly one geometry.
    Returns
    -------
    BoundingBoxExtent
        An instance of BoundingBoxExtent with the bounds of the geometry and its EPSG code.
    Raises
    ------
    TypeError
        If the input is not a GeoDataFrame or a file path to a GeoDataFrame.
    ValueError
        If the GeoDataFrame does not have a defined CRS or contains more than one geometry.
    """

    if isinstance(gdf, Path):
        gdf = gpd.read_file(gdf)
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            "Input must be a GeoDataFrame or a file path to a GeoDataFrame."
        )
    if len(gdf) != 1:
        raise ValueError("GeoDataFrame must contain exactly one geometry.")

    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a defined CRS.")

    geom = gdf.geometry.iloc[0]
    if not geom.is_valid:
        raise ValueError("Geometry is not valid.")

    bounds = geom.bounds
    return BoundingBoxExtent(
        west=bounds[0],
        south=bounds[1],
        east=bounds[2],
        north=bounds[3],
        epsg=gdf.crs.to_epsg(),
    )
