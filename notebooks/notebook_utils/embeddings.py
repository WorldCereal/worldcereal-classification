import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from ipywidgets import Output
from openeo_gfmap import Backend, BackendContext, TemporalContext
from sklearn.decomposition import IncrementalPCA

from worldcereal.job import WorldCerealTask
from worldcereal.jobmanager import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    WorldCerealJobManager,
)
from worldcereal.parameters import EmbeddingsParameters

from .job_manager import (
    fetch_results_from_outdir,
    notebook_logger,
    run_notebook_job_manager,
)


def collect_worldcereal_embeddings(
    aoi_gdf: gpd.GeoDataFrame,
    output_folder: Path,
    temporal_extent: Optional[TemporalContext] = None,
    year: Optional[int] = None,
    embeddings_parameters: Optional[EmbeddingsParameters] = None,
    scale_uint16: bool = True,
    grid_size: int = 20,
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
    """Collect WorldCereal embeddings using WorldCerealJobManager.

    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the AOI geometries for which to collect embeddings.
    output_folder : Path
        Path to the folder where the collected embeddings will be stored.
    temporal_extent : Optional[TemporalContext], optional
        Temporal context defining the time range for which to collect embeddings.
        If provided together with `year`, temporal_extent will take precedence and override the year.
    year : Optional[int], optional
        Specific year to collect embeddings for.
        If provided, it will be used only if temporal_extent is not provided.
    embeddings_parameters : Optional[EmbeddingsParameters], optional
        Parameters for configuring the embeddings computation.
        By default the globally pre-trained Presto model is used.
    scale_uint16 : bool, optional
        Whether to apply the empirically determined scaling for Presto-based WorldCereal embeddings
        when reading the results. Reduces memory usage while still allowing to recover the original float values.
        Default is True.
    grid_size : int, optional
        Grid size in kilometers for tiling the AOI during job processing. Default is 20 km.
    parallel_jobs : int, optional
        Number of parallel jobs to run. Default is 2.
    randomize_jobs : bool, optional
        Whether to randomize the order of job submissions. Default is False.
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        If specified, only collect embeddings for Sentinel-1 data from the given orbit state.
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
        The job manager instance that was used to run the embedding collection.
        Can be used to further inspect the job status and results.
    """

    # Set up logging and plotting outputs for the notebook
    plot_out, log_out, _log = notebook_logger(
        plot_out=plot_out,
        log_out=log_out,
        display_outputs=display_outputs,
    )

    # Ensure embeddings parameters are initialized
    if embeddings_parameters is None:
        embeddings_parameters = EmbeddingsParameters()

    _log("------------------------------------")
    _log("STARTING WORKFLOW: Embeddings computation")
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
        "embeddings_parameters": embeddings_parameters,
        "scale_uint16": scale_uint16,
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
        task=WorldCerealTask.EMBEDDINGS,
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
        "Stopping embeddings computation...\n"
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
                "embeddings_parameters": embeddings_parameters,
                "scale_uint16": scale_uint16,
                "job_options": job_options,
                "max_retries": DEFAULT_MAX_RETRIES,
                "base_delay": DEFAULT_BASE_DELAY,
                "max_delay": DEFAULT_MAX_DELAY,
            },
            plot_out=plot_out,
            log_out=log_out,
            display_outputs=display_outputs,
            status_title="Embeddings job status",
        )
    except KeyboardInterrupt:
        _log(break_msg)
        manager.stop_job_thread()
        _log("Embeddings computation has stopped.")
        raise

    _log("All done!")
    _log(f"Results stored in {output_folder}")

    return manager


# ---------------------------------------------------------------------------
# Utility functions to inspect locally downloaded embeddings tif files
# ---------------------------------------------------------------------------


def fetch_embeddings_results_from_outdir(
    outdir: Path,
) -> list[Path]:
    """Fetch local file paths of successfully completed embeddings jobs from a specified output directory.

    Parameters
    ----------
    outdir : Path
        The output directory to fetch results from.

    Returns
    -------
    list[Path]
        A list of paths to the files matching the pattern for successfully completed embeddings jobs.
    """

    expected_file_pattern = "WorldCereal_Embeddings_*.tif"

    return fetch_results_from_outdir(outdir, expected_file_pattern)


def read_embeddings_raster(infile: Path, scale_uint16: bool) -> tuple[
    np.ndarray,
    rasterio.profiles.Profile,
]:
    """Read an embeddings raster file and return the data and metadata.

    Parameters
    ----------
    infile : Path
        Path to the embeddings raster file.

    Returns
    -------
    tuple[np.ndarray, rasterio.profiles.Profile]
        A tuple containing the embeddings data as a numpy array and the raster metadata profile.
    """

    with rasterio.open(infile) as ds:
        data = ds.read()  # shape: (bands, height, width)
        profile = ds.profile

    # Go back to original scale if needed
    if scale_uint16:
        # Empirically determined scaling factors for Presto-based WorldCereal embeddings
        offset = -6
        scale = 0.0002
        data = data.astype(np.float32) * scale + offset

    return data, profile


def show_embeddings_histogram(
    data: np.ndarray,
    bands_to_plot: Optional[int] = 6,
    bins: int = 50,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Plot histograms of the embeddings values for each band.

    Parameters
    ----------
    data : np.ndarray
        The embeddings data array with shape (bands, height, width).
    bands_to_plot : Optional[int], optional
        Number of bands to plot. If None, all bands will be plotted, by default 6.
    bins : int, optional
        Number of bins to use in the histogram, by default 50.
    figsize : tuple[int, int], optional
        Size of the figure for plotting, by default (12, 6).
    """

    num_bands = data.shape[0]
    bands_to_plot = (
        min(num_bands, bands_to_plot) if bands_to_plot is not None else num_bands
    )

    fig, axes = plt.subplots(2, (bands_to_plot + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    for i in range(bands_to_plot):
        arr = data[i].ravel()
        sample = (
            arr if arr.size < 300_000 else np.random.choice(arr, 300_000, replace=False)
        )
        axes[i].hist(sample, bins=bins, color="steelblue", alpha=0.8)
        axes[i].set_title(f"Embedding dimension {i+1}")
    plt.tight_layout()
    plt.show()


def run_pca_on_embeddings(
    data: np.ndarray,
    n_components: int = 3,
    fit_sample_max: int = 400_000,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Incremental PCA on the embeddings data in a memory-efficient way by processing in chunks.

    Parameters
    ----------
    data : np.ndarray
        The embeddings data array with shape (bands, height, width).
    n_components : int, optional
        Number of principal components to compute, by default 3.
    fit_sample_max : int, optional
        Maximum number of pixels to use for fitting the IncrementalPCA.
        If the total number of pixels exceeds this, a random subset will be used.
        Default is 400,000.
    chunk_size : int, optional
        Number of pixels to process in each chunk during both fitting and transformation.
        Default is 50,000.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - comp_img: The PCA-transformed image data with shape (height, width, n_components).
        - comp_min: The minimum value of each PCA component across the entire image.
        - comp_max: The maximum value of each PCA component across the entire image.
    """

    # Preparations
    bands, h, w = data.shape
    flat = data.reshape(bands, -1).T  # (pixels, bands)
    n_pixels = flat.shape[0]
    idx_all = np.arange(n_pixels)

    # Select subset for fitting (random) to speed up & reduce RAM
    if n_pixels > fit_sample_max:
        fit_idx = np.random.choice(n_pixels, fit_sample_max, replace=False)
    else:
        fit_idx = idx_all

    # initiate PCA
    ipca = IncrementalPCA(n_components=n_components)

    # Pass 1: partial_fit on chunks of the fit subset
    for start in range(0, fit_idx.size, chunk_size):
        end = start + chunk_size
        batch_idx = fit_idx[start:end]
        ipca.partial_fit(flat[batch_idx])

    print("Explained variance ratio (incremental):", ipca.explained_variance_ratio_)

    # Pass 2: transform entire image in chunks (streaming) and track per-component min/max
    components_full = np.empty((n_pixels, ipca.n_components), dtype=np.float32)
    comp_min = np.full(ipca.n_components, np.inf, dtype=np.float64)
    comp_max = np.full(ipca.n_components, -np.inf, dtype=np.float64)

    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        transformed = ipca.transform(flat[start:end])
        components_full[start:end] = transformed
        # Update running min/max
        comp_min = np.minimum(comp_min, transformed.min(axis=0))
        comp_max = np.maximum(comp_max, transformed.max(axis=0))

    # Reshape to (h, w, n_components)
    comp_img = components_full.reshape(h, w, ipca.n_components)

    # Normalize using collected min/max (avoid second full pass)
    rng = comp_max - comp_min
    rng[rng == 0] = 1
    comp_norm = (comp_img - comp_min) / rng

    plt.figure(figsize=(10, 10))
    plt.imshow(comp_norm[..., :3])
    plt.title("Incremental PCA Components 1-3 (as RGB)")
    plt.axis("off")
    plt.show()

    print("Final component ranges:", list(zip(comp_min, comp_max)))

    return comp_img, comp_min, comp_max


def plot_embeddings_as_rgb(
    data: np.ndarray,
) -> None:
    """Plot random embeddings as RGB image.

    Parameters
    ----------
    data : np.ndarray
        The embeddings data array with shape (bands, height, width).
    """

    pseudo_indices = (0, 1, 2) if data.shape[0] >= 3 else None
    if pseudo_indices:
        pseudo = data[list(pseudo_indices), :, :].astype(float)
        # Normalize each selected band individually
        for i in range(3):
            b = pseudo[i]
            pseudo[i] = (b - b.min()) / (b.max() - b.min() + 1e-9)
        rgb = np.transpose(pseudo, (1, 2, 0))
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.title(f"Pseudo RGB Bands {pseudo_indices}")
        plt.axis("off")
        plt.show()
    else:
        print("Not enough bands for pseudo-RGB composite.")
