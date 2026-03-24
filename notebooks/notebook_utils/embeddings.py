from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.decomposition import IncrementalPCA

from .job_manager import fetch_results_from_outdir

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
