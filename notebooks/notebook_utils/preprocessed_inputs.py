from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from tabulate import tabulate

from .extractions import (
    NODATAVALUE,
    WORLDCEREAL_BANDS,
    _apply_band_scaling,
)
from .job_manager import fetch_results_from_outdir

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
