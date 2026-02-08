import copy
import glob
import time
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
from IPython.display import display
from ipywidgets import Output
from loguru import logger
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely import wkt
from shapely.geometry import box

from worldcereal.job import setup_inference_job_manager
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.openeo.workflow_config import WorldCerealWorkflowConfig
from worldcereal.parameters import WorldCerealProductType

# Define the color mapping for job statuses
JOB_STATUS_COLORS = {
    "not_started": "grey",
    "created": "gold",
    "queued": "lightsteelblue",
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
    status_plot["geometry"] = status_plot["geometry"].apply(wkt.loads)
    status_plot = gpd.GeoDataFrame(status_plot, geometry="geometry", crs="EPSG:4326")
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


def lon_to_utm_zone(lon: float) -> int:
    return int((lon + 180) / 6) + 1


def utm_zone_bounds(zone: int):
    min_lon = (zone - 1) * 6 - 180
    max_lon = min_lon + 6
    return min_lon, max_lon


def split_bbox_by_utm_and_hemisphere(west, south, east, north) -> list:
    """
    Splits a bounding box into UTM zones and hemispheres.
    Coordinates must be in WGS84 (EPSG:4326) format.

    Returns
    -------
    list
        A list of dictionaries, each containing the bounding box for a UTM zone and hemisphere.
        Each dictionary has keys: 'west', 'south', 'east', 'north', 'crs', 'zone', 'hemisphere'.
    """
    zone_start = lon_to_utm_zone(west)
    zone_end = lon_to_utm_zone(east)
    hemi_splits = []

    if south < 0 and north > 0:
        lat_splits = [("S", south, 0), ("N", 0, north)]
    else:
        hemisphere = "N" if south >= 0 else "S"
        lat_splits = [(hemisphere, south, north)]

    for zone in range(zone_start, zone_end + 1):
        zmin_lon, zmax_lon = utm_zone_bounds(zone)
        lon_min_clipped = max(west, zmin_lon)
        lon_max_clipped = min(east, zmax_lon)

        if lon_min_clipped >= lon_max_clipped:
            continue

        for hemisphere, hemi_min_lat, hemi_max_lat in lat_splits:
            hemi_min_clipped = max(south, hemi_min_lat)
            hemi_max_clipped = min(north, hemi_max_lat)

            if hemi_min_clipped < hemi_max_clipped:
                hemi_splits.append(
                    {
                        "west": lon_min_clipped,
                        "south": hemi_min_clipped,
                        "east": lon_max_clipped,
                        "north": hemi_max_clipped,
                        "crs": "EPSG:4326",
                        "zone": zone,
                        "hemisphere": hemisphere,
                    }
                )

    return hemi_splits


def create_tiling_grid(
    bbox: dict,
    basename: str = "tile",
    output_crs: str = "EPSG:4326",
    grid_size_m: float = 20000,
    tiling_crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Create a grid of square tiles over a bounding box (with CRS).
    Tiles in `tiling_crs`, output in `output_crs`.

    Parameters
    ----------
    bbox : dict
        Dict with 'west', 'south', 'east', 'north', 'crs' keys.
    basename : str
        Base name for the tile identifiers.
    output_crs : str
        Output CRS for the geometry column in the grid (e.g., WGS84 for OpenEO).
    grid_size_m : float
        Side length of each square tile in meters.
    tiling_crs : Optional[str]
        CRS to use for tiling. If None (default), the best local UTM CRS is derived

    Returns
    -------
    GeoDataFrame
        Grid tiles as polygons, with 'tile_name', 'epsg', and 'bounds_epsg' columns.
    """
    # Validate input bbox
    if not {"west", "south", "east", "north", "crs"}.issubset(bbox):
        raise ValueError("bbox must include 'west', 'south', 'east', 'north', 'crs'.")

    # Build bounding box geometry and project to tiling CRS
    bbox_geom = box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=bbox["crs"])
    if tiling_crs is None:
        # Derive best local UTM CRS for tiling
        crs = bbox_gdf.estimate_utm_crs()
        epsg = int(crs.to_epsg())
        tiling_crs = f"EPSG:{epsg}"
    bbox_gdf = bbox_gdf.to_crs(tiling_crs)

    # Create grid tiles
    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    x_coords = np.arange(minx, maxx, grid_size_m)
    y_coords = np.arange(miny, maxy, grid_size_m)
    coordinates = [
        (x, y, min(x + grid_size_m, maxx), min(y + grid_size_m, maxy))
        for x in x_coords
        for y in y_coords
    ]
    # convert coordinates to string for writing to GeoDataFrame
    bounds_tiling = [repr(coords) for coords in coordinates]

    # Create geometries for each tile
    geometries_tiling = [box(*coords) for coords in coordinates]

    # Create GeoDataFrame with the geometries
    grid = gpd.GeoDataFrame(geometry=geometries_tiling, crs=tiling_crs).to_crs(
        output_crs
    )
    grid["tile_name"] = [f"{basename}_{i}" for i in range(len(grid))]
    grid["epsg"] = (
        int(tiling_crs.split(":")[1]) if tiling_crs.startswith("EPSG:") else None
    )
    grid["bounds_epsg"] = bounds_tiling

    return grid


def create_production_grid(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    resolution: int = 20,
    tiling_crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Create a production grid for the given extent.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        The extent for which to create the production grid.
        Must be in WGS84 (EPSG:4326) coordinate reference system.
    temporal_extent : TemporalContext
        The temporal extent for the production grid.
        Must have 'start_date' and 'end_date' attributes.
        These will be added to the grid as metadata.
    resolution : int, optional
        The resolution of the grid in meters, by default 20.
    tiling_crs : Optional[str], optional
        The coordinate reference system to use for tiling.
        If None, the best local UTM CRS will be derived.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the production grid.
    """

    # TODO: make sure the tiling grid is in line with the Sentinel 20 m grid

    # Ensure the spatial extent is in WGS84 (EPSG:4326)
    if not spatial_extent.epsg == 4326:
        logger.info(
            '"Spatial extent is not in WGS84 (EPSG:4326). Reprojecting to WGS84.")'
        )
        # Convert to geometry and reproject to WGS84
        bbox_geom = box(
            spatial_extent.west,
            spatial_extent.south,
            spatial_extent.east,
            spatial_extent.north,
        )
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=spatial_extent.epsg)
        bbox_gdf = bbox_gdf.to_crs("EPSG:4326")
        spatial_extent = BoundingBoxExtent(
            west=bbox_gdf.total_bounds[0],
            south=bbox_gdf.total_bounds[1],
            east=bbox_gdf.total_bounds[2],
            north=bbox_gdf.total_bounds[3],
            epsg=4326,
        )

    # We first split the bounding box by UTM zones
    bbox_splits = split_bbox_by_utm_and_hemisphere(
        spatial_extent.west,
        spatial_extent.south,
        spatial_extent.east,
        spatial_extent.north,
    )

    logger.info(f"Splitted bounding box into {len(bbox_splits)} UTM zone splits.")
    grid_dfs = []
    for split in bbox_splits:
        # Splitting bbox into smaller tiles, specified by the resolution
        tile_name = f"tile_{split['zone']}{split['hemisphere']}"
        grid_dfs.append(
            create_tiling_grid(
                split,
                basename=tile_name,
                grid_size_m=resolution * 1000,
                tiling_crs=tiling_crs,
            )
        )

    # Concatenate all the grids into a single GeoDataFrame
    grid = gpd.GeoDataFrame(pd.concat(grid_dfs, ignore_index=True))
    # Add metadata columns
    grid["start_date"] = temporal_extent.start_date
    grid["end_date"] = temporal_extent.end_date

    logger.info(f"Created production grid with {len(grid)} tiles.")

    return grid


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

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the final job status.
    Raises
    ------
    ValueError
        If the spatial extent is not in WGS84 (EPSG:4326) coordinate reference system.
    """

    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create production grid according to tile resolution and spatial extent
    grid_file = output_dir / "production_grid.gpkg"
    if grid_file.exists():
        logger.info(f"Loading existing production grid from {grid_file}")
        production_grid = gpd.read_file(grid_file)
    else:
        logger.info("Creating new production grid.")
        production_grid = create_production_grid(
            spatial_extent,
            temporal_extent,
            resolution=tile_resolution,
            tiling_crs=tiling_crs,
        )
        production_grid.to_file(grid_file, driver="GPKG")

    # Prepare the job manager and job database
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
    job_manager.start_job_thread(start_job=start_job, job_db=job_db)

    # Compute center of total bounding box (used for map centering)
    minx, miny, maxx, maxy = production_grid.total_bounds
    center = {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2}

    # Create output widgets (if not provided)
    if plot_out is None:
        plot_out = Output()
    if log_out is None:
        log_out = Output()
    if display_outputs:
        display(plot_out)
        display(log_out)

    # Now run an update every 60 seconds
    poll_sleep = 60
    while not job_manager._stop_thread:
        if stop_event and stop_event.is_set():
            with log_out:
                print("Stop event triggered. Exiting loop.")
            job_manager.stop_job_thread()
            break

        try:
            status_df = pd.read_csv(job_db.path)
            fig = plot_job_status(
                status_df=status_df, color_dict=JOB_STATUS_COLORS, center=center
            )
            with plot_out:
                plot_out.clear_output(wait=True)
                fig.show(renderer="notebook")
            with log_out:
                log_out.clear_output(wait=True)
                print(
                    f"[{time.strftime('%H:%M:%S')}] Checked job status:\n"
                    f"{status_df['status'].value_counts().to_string()}\n"
                )

            # Check if all jobs are done
            if (
                status_df["status"]
                .isin(["not_started", "created", "queued", "running"])
                .sum()
                == 0
            ):
                job_manager.stop_job_thread()
                break

            time.sleep(poll_sleep)  # Wait before the next update

        except Exception as e:
            with log_out:
                print(f"Error occurred: {e}")
            job_manager.stop_job_thread()
            raise

    with log_out:
        print("Job manager stopped or finished.")

    # Get final status
    status_df = pd.read_csv(job_db.path)

    return status_df


def run_production_wrapper(queue, stop_event, args, kwargs):
    try:
        result = run_map_production(*args, stop_event=stop_event, **kwargs)
        queue.put(("done", result))
    except Exception as e:
        queue.put(("error", e))


def start_production_process(args, kwargs):
    queue = Queue()
    stop_event = Event()
    proc = Process(
        target=run_production_wrapper, args=(queue, stop_event, args, kwargs)
    )
    proc.start()
    return proc, queue, stop_event


def monitor_production_process(proc, queue, stop_event):
    try:
        while proc.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("⚠️ KeyboardInterrupt: requesting shutdown...")
        stop_event.set()
        proc.terminate()
        proc.join()
        print("✅ Process forcibly terminated.")
        return

    if not queue.empty():
        status, result = queue.get()
        if status == "done":
            print("✅ Processing completed successfully.")
            return result
        elif status == "error":
            print("❌ Job failed with exception:")
            raise result
    else:
        print("⚠️ No result returned. The process may have been killed or crashed.")


def merge_maps(outdir: Path, product="croptype") -> Path:
    """Merge all maps in the output directory into a single .tif file."""

    if not outdir.exists():
        raise FileNotFoundError(f"Output directory {outdir} does not exist.")

    # Find all .tif files in the output directory
    tifs = glob.glob(str(outdir / "*" / f"{product}_*.tif"))
    if len(tifs) == 0:
        raise FileNotFoundError(
            "No tif files found in the output directory matching your product."
        )

    reprojected_tifs = []

    with rasterio.Env(CPL_LOG="ERROR"):
        for tif in tifs:
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
        outfile = outdir / f"{product}_merged.tif"
        with rasterio.open(outfile, "w", **out_meta) as dest:
            dest.write(mosaic)
            # Preserve band descriptions (if any)
            for idx, desc in enumerate(reprojected_tifs[0].descriptions, start=1):
                if desc:
                    dest.set_band_description(idx, desc)

    return outfile


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
