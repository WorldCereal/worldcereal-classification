import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from ipywidgets import Output
from loguru import logger
from openeo_gfmap import Backend, BackendContext, TemporalContext
from tabulate import tabulate

from worldcereal.job import WorldCerealTask
from worldcereal.jobmanager import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    WorldCerealJobManager,
)

from .extractions import (
    NODATAVALUE,
    WORLDCEREAL_BANDS,
    _apply_band_scaling,
)
from .job_manager import (
    fetch_results_from_outdir,
    notebook_logger,
    run_notebook_job_manager,
)


def collect_worldcereal_inputs_patches(
    aoi_gdf: gpd.GeoDataFrame,
    output_folder: Path,
    grid_size: int = 20,
    temporal_extent: Optional[TemporalContext] = None,
    year: Optional[int] = None,
    parallel_jobs: int = 2,
    randomize_jobs: bool = False,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    restart_failed: bool = True,
    job_options: Optional[Dict[str, Union[str, int, None]]] = None,
    plot_out: Optional[Output] = None,
    log_out: Optional[Output] = None,
    display_outputs: bool = True,
    poll_sleep: int = 60,
    simplify_logging: bool = True,
) -> WorldCerealJobManager:
    """Collect WorldCereal input data patches using WorldCerealJobManager.

    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the AOI geometries for which to collect input data patches.
    output_folder : Path
        Path to the folder where the collected input data patches will be stored.
    grid_size : int, optional
        Grid size in kilometers for tiling the AOI during job processing. Default is 20 km.
    temporal_extent : Optional[TemporalContext], optional
        Temporal context defining the time range for which to collect input data patches.
        If provided together with `year`, temporal_extent will take precedence and override the year.
    year : Optional[int], optional
        Specific year to collect input data patches for.
        If provided, it will be used only if temporal_extent is not provided.
    parallel_jobs : int, optional
        Number of parallel jobs to run. Default is 2.
    randomize_jobs : bool, optional
        Whether to randomize the order of job submissions. Default is False.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        If specified, only collect input data patches for Sentinel-1 data from the given orbit state.
        If not specified, the best orbit state will be automatically selected for each tile.
    restart_failed : bool, optional
        Whether to automatically restart failed jobs. Default is True.
    job_options : Optional[Dict[str, Union[str, int, None]]], optional
        Additional job options to pass to the backend. Keys and values depend on the backend used.
    plot_out : Optional[Output], optional
        Optional ipywidgets Output widget for plotting job status and results.
        If None, a new Output widget will be created for plotting. Default is None.
    log_out : Optional[Output], optional
        Optional ipywidgets Output widget for logging job progress and information.
        If None, a new Output widget will be created for logging. Default is None.
    display_outputs : bool, optional
        Whether to display the log and plot Output widgets in the notebook. Default is True.
    poll_sleep : int, optional
        Time in seconds to wait between polling job status. Default is 60.
    simplify_logging : bool, optional
        Whether to simplify logging output by reducing OpenEO job management messages.
        Default is True.

    Returns
    -------
    WorldCerealJobManager
        The job manager instance that was used to run the input data collection.
        Can be used to further inspect the job status and results.
    """

    # Set up logging and plotting outputs for the notebook
    plot_out, log_out, _log = notebook_logger(
        plot_out=plot_out,
        log_out=log_out,
        display_outputs=display_outputs,
    )

    _log("------------------------------------")
    _log("STARTING WORKFLOW: Input data collection")
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
        "year": year,
        "s1_orbit_state": s1_orbit_state,
        "parallel_jobs": parallel_jobs,
        "restart_failed": restart_failed,
        "randomize_jobs": randomize_jobs,
        "job_options": job_options,
        "poll_sleep": poll_sleep,
        "simplify_logging": simplify_logging,
    }
    for key, value in params.items():
        _log(f"{key}: {value}")
    _log("----------------------------------")

    _log("Initializing job manager...")
    manager = WorldCerealJobManager(
        output_dir=output_folder,
        task=WorldCerealTask.INPUTS,
        backend_context=BackendContext(Backend.CDSE),
        aoi_gdf=aoi_gdf,
        grid_size=grid_size,
        temporal_extent=temporal_extent,
        year=year,
        poll_sleep=poll_sleep,
    )

    if simplify_logging:
        logging.getLogger("openeo").setLevel(logging.WARNING)
        logging.getLogger("openeo.extra.job_management._manager").setLevel(
            logging.WARNING
        )

    _log("Starting job submissions...")
    break_msg = (
        "Stopping input data collection...\n"
        "Make sure to manually cancel any running jobs in the backend to avoid unnecessary costs!\n"
        "For this, visit the job tracking page in the backend dashboard: https://openeo.dataspace.copernicus.eu/\n"
    )

    try:
        run_notebook_job_manager(
            manager,
            run_kwargs={
                "restart_failed": restart_failed,
                "randomize_jobs": randomize_jobs,
                "parallel_jobs": parallel_jobs,
                "s1_orbit_state": s1_orbit_state,
                "job_options": job_options,
                "max_retries": DEFAULT_MAX_RETRIES,
                "base_delay": DEFAULT_BASE_DELAY,
                "max_delay": DEFAULT_MAX_DELAY,
            },
            plot_out=plot_out,
            log_out=log_out,
            display_outputs=display_outputs,
            status_title="Inputs job status",
        )
    except KeyboardInterrupt:
        _log(break_msg)
        manager.stop_job_thread()
        _log("Input data collection has stopped.")
        raise

    _log("All done!")
    _log(f"Results stored in {output_folder}")

    return manager


# ---------------------------------------------------------------------------
# Utility functions to inspect locally downloaded preprocessed input NetCDFs
# ---------------------------------------------------------------------------


def fetch_inputs_results_from_outdir(
    outdir: Path,
) -> list[Path]:
    """Fetch local file paths of successfully completed input data jobs from the given output directory.

    Parameters
    ----------
    outdir : Path
        The output directory to fetch results from.

    Returns
    -------
    list[Path]
        A list of paths to the files matching the pattern for successfully completed input data jobs.
    """

    expected_file_pattern = "preprocessed-inputs_*.nc"

    return fetch_results_from_outdir(outdir, expected_file_pattern)


def get_band_statistics_netcdf(ds: xr.Dataset) -> pd.DataFrame:
    """Compute scaled statistics for supported WorldCereal bands in an xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing band variables; 'crs' is ignored if present.

    Returns
    -------
    pd.DataFrame
        Index: band name. Columns: %_nodata, min, max, mean, std.

    Notes
    -----
    Nodata defined by NODATAVALUE (65535) or non-finite values. Unsupported bands skipped.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray.Dataset")

    stats = {}
    skipped = []

    for bandname, da in ds.data_vars.items():
        if bandname == "crs":
            continue
        if not any(bandname in v for v in WORLDCEREAL_BANDS.values()):
            skipped.append(bandname)
            continue

        arr = da.values
        flat = arr.ravel()
        total = flat.size
        if total == 0:
            continue

        nodata_mask = (flat == NODATAVALUE) | (~np.isfinite(flat))
        valid = flat[~nodata_mask]
        nodata_pct = (nodata_mask.sum() / total * 100.0) if total else 0.0

        if valid.size == 0:
            stats[bandname] = {
                "%_nodata": f"{nodata_pct:.2f}",
                "min": "N/A",
                "max": "N/A",
                "mean": "N/A",
                "std": "N/A",
            }
            continue

        scaled = _apply_band_scaling(valid.astype(np.float32), bandname)
        stats[bandname] = {
            "%_nodata": f"{nodata_pct:.2f}",
            "min": f"{scaled.min():.4f}",
            "max": f"{scaled.max():.4f}",
            "mean": f"{scaled.mean():.4f}",
            "std": f"{scaled.std():.4f}",
        }

    if not stats:
        raise ValueError("No supported bands with data found.")

    df = pd.DataFrame(stats).T
    print("-------------------------------------")
    print("Band statistics (NetCDF):")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=True))
    if skipped:
        logger.warning(f"Skipped unsupported bands: {', '.join(skipped)}")
    return df


def visualize_timeseries_netcdf(
    ds: xr.Dataset,
    band: str = "NDVI",
    npixels: int = 5,
    pixel_coords: Optional[list[tuple]] = None,
    time_dim: str = "t",
    y_dim: str = "y",
    x_dim: str = "x",
    random_seed: Optional[int] = None,
    outfile: Optional[Path] = None,
    nodata: int = NODATAVALUE,
) -> None:
    """Visualize time series for random or specified pixels in a preprocessed inputs Dataset.

    If band == 'NDVI', it is computed from S2-L2A-B08 (NIR) and S2-L2A-B04 (RED).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    band : str, default 'NDVI'
        Band to plot or 'NDVI'.
    npixels : int, default 5
        Number of random pixels if pixel_coords not given.
    pixel_coords : list[tuple], optional
        Explicit (row, col) coordinates.
    time_dim, y_dim, x_dim : str
        Dimension names.
    random_seed : int, optional
        Seed for reproducible sampling.
    outfile : Path, optional
        File path to save figure.
    nodata : int
        Nodata sentinel value.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray.Dataset")

    for dim in (time_dim, y_dim, x_dim):
        if dim not in ds.dims:
            raise KeyError(f"Dimension '{dim}' not in dataset.")

    if band == "NDVI":
        for needed in ["S2-L2A-B08", "S2-L2A-B04"]:
            if needed not in ds.data_vars:
                raise KeyError(
                    f"Band '{needed}' required for NDVI but missing from dataset"
                )
        nir = (
            ds["S2-L2A-B08"]
            .astype("float32")
            .where(lambda v: (v != nodata) & np.isfinite(v))
        )
        red = (
            ds["S2-L2A-B04"]
            .astype("float32")
            .where(lambda v: (v != nodata) & np.isfinite(v))
        )
        data_da = (nir - red) / (nir + red)
        ylabel = "NDVI"
    else:
        if band not in ds.data_vars:
            raise KeyError(f"Band '{band}' not found in dataset")
        data_da = (
            ds[band].astype("float32").where(lambda v: (v != nodata) & np.isfinite(v))
        )
        # Apply scaling if supported
        try:
            vals = data_da.values
            mask_valid = (vals != nodata) & np.isfinite(vals)
            scaled = vals.copy()
            scaled[mask_valid] = _apply_band_scaling(scaled[mask_valid], band)
            data_da = data_da.copy(data=scaled)
        except ValueError:
            pass
        ylabel = band

    ny, nx = data_da.sizes[y_dim], data_da.sizes[x_dim]
    if pixel_coords is None:
        rng = np.random.default_rng(random_seed)
        finite_mask = np.isfinite(data_da).any(dim=time_dim).values
        valid_idx = np.argwhere(finite_mask)
        if valid_idx.size == 0:
            raise ValueError("No valid pixels found for plotting.")
        if npixels > len(valid_idx):
            logger.warning(
                f"Requested {npixels} pixels; only {len(valid_idx)} valid. Using all valid."
            )
            npixels = len(valid_idx)
        sel = rng.choice(len(valid_idx), size=npixels, replace=False)
        pixel_coords = [tuple(valid_idx[i]) for i in sel]
    else:
        for r, c in pixel_coords:
            if not (0 <= r < ny and 0 <= c < nx):
                raise IndexError(f"Pixel ({r},{c}) out of bounds ({ny},{nx}).")

    fig, ax = plt.subplots(figsize=(12, 6))
    time_values = pd.to_datetime(data_da[time_dim].values)
    for r, c in pixel_coords:
        ts = data_da.isel({y_dim: r, x_dim: c}).values
        ax.plot(time_values, ts, marker="o", linestyle="-", label=f"(row={r}, col={c})")

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=90)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.subplots_adjust(right=0.75)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if outfile is not None:
        plt.savefig(outfile)
    return
