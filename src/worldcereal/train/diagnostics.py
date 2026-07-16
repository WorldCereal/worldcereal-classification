"""Plot per-bin per-class weight grids matching what the sampler uses.

Visualizes the per-sample weights produced by ``SeasonalTaskBatchSampler``
(its ``_lc_probs`` / ``_ct_probs`` attributes) as a per-(bin, class)
heatmap on a geographic map. Because those weights already encode every
dispatch decision the sampler made — class-balancing scope, spatial
density factor, smoothing, fallbacks, clipping — the plot reflects
exactly what the model trains on. NaN cells are absent class-bin
combinations (no samples), highlighted in a distinct color.

`dump_per_bin_class_grids` is the public entry point: given a training
dataframe and the matching per-sample weights, it writes three
artifacts per pool to *output_dir*:
  - per_bin_class_weights_{pool}.png   (one panel per class)
  - per_bin_sample_counts_{pool}.png   (single map of samples per bin)
  - per_bin_class_summary_{pool}.csv   (per-class summary stats)
"""

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
    _MAP_PROJ = ccrs.PlateCarree()
except ImportError:
    _HAS_CARTOPY = False
    _MAP_PROJ = None


def _draw_basemap(ax: plt.Axes, extent: Tuple[float, float, float, float]) -> None:
    if _HAS_CARTOPY:
        ax.set_extent(extent, crs=_MAP_PROJ)
        ax.add_feature(cfeature.OCEAN, facecolor="#dfe7ef", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f7f5ef", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888")
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.4, alpha=0.5, linestyle=":",
            color="#666",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 6}
        gl.ylabel_style = {"size": 6}
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)


def _plain_log_formatter() -> mticker.FuncFormatter:
    return mticker.FuncFormatter(lambda x, _: f"{x:g}")


def _make_per_class_grid_plot(
    grids: Dict[str, np.ndarray],
    classes: List[str],
    lat_min: int, lon_min: int, n_lat: int, n_lon: int,
    bin_size: float,
    extent: Tuple[float, float, float, float],
    output_path: Path,
    title: str,
) -> None:
    n_classes = len(classes)
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols

    subplot_kw = {"projection": _MAP_PROJ} if _HAS_CARTOPY else {}
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        subplot_kw=subplot_kw,
    )
    axes_flat = np.atleast_2d(axes).flatten()

    # Shared log color norm across panels
    all_vals = np.concatenate(
        [g[~np.isnan(g)] for g in grids.values() if not np.all(np.isnan(g))]
    )
    vmin = max(float(all_vals.min()), 1e-3)
    vmax = float(all_vals.max())
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["viridis"].with_extremes(bad="#cccccc")  # NaN -> grey

    # Bin edges in lat/lon
    lat_edges = (np.arange(n_lat + 1) + lat_min) * bin_size - 90.0
    lon_edges = (np.arange(n_lon + 1) + lon_min) * bin_size - 180.0

    last_mesh = None
    for i, cls in enumerate(classes):
        ax = axes_flat[i]
        _draw_basemap(ax, extent)
        grid = grids[cls]
        masked = np.ma.masked_invalid(grid)
        kwargs = dict(norm=norm, cmap=cmap, shading="flat")
        if _HAS_CARTOPY:
            kwargs["transform"] = _MAP_PROJ
        last_mesh = ax.pcolormesh(lon_edges, lat_edges, masked, **kwargs)
        n_present = int(np.sum(~np.isnan(grid)))
        n_total_pixels = grid.size
        ax.set_title(
            f"{cls} ({n_present}/{n_total_pixels} bins with weight)",
            fontsize=9,
        )

    # Hide unused subplots
    for j in range(n_classes, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_mesh, cax=cbar_ax)
    cbar.set_label("per-bin balanced class weight", fontsize=9)
    cbar.ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20)
    )
    cbar.ax.yaxis.set_major_formatter(_plain_log_formatter())
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.annotate(
        f"max = {vmax:.2g}\nmin = {vmin:.2g}\ngrey = NaN (sparse / absent)",
        xy=(1.05, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=7, color="#444",
    )

    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _make_bin_count_plot(
    bin_sample_counts: Dict[Tuple[int, int], int],
    lat_min: int, lon_min: int, n_lat: int, n_lon: int,
    bin_size: float,
    extent: Tuple[float, float, float, float],
    output_path: Path,
    title: str,
    min_samples_per_bin: int,
) -> None:
    """Single-panel map of total samples per bin (sparsity overview)."""
    grid = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    for (li, lo), n in bin_sample_counts.items():
        gi = li - lat_min
        gj = lo - lon_min
        if 0 <= gi < n_lat and 0 <= gj < n_lon:
            grid[gi, gj] = n

    subplot_kw = {"projection": _MAP_PROJ} if _HAS_CARTOPY else {}
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw=subplot_kw)
    _draw_basemap(ax, extent)

    masked = np.ma.masked_invalid(grid)
    norm = mcolors.LogNorm(
        vmin=max(1, np.nanmin(grid)), vmax=float(np.nanmax(grid))
    )
    cmap = plt.colormaps["plasma"].with_extremes(bad="#cccccc")
    lat_edges = (np.arange(n_lat + 1) + lat_min) * bin_size - 90.0
    lon_edges = (np.arange(n_lon + 1) + lon_min) * bin_size - 180.0
    kwargs = dict(norm=norm, cmap=cmap, shading="flat")
    if _HAS_CARTOPY:
        kwargs["transform"] = _MAP_PROJ
    mesh = ax.pcolormesh(lon_edges, lat_edges, masked, **kwargs)

    fig.suptitle(title, fontsize=12, y=0.97)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("samples per bin", fontsize=9)
    cbar.ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20)
    )
    cbar.ax.yaxis.set_major_formatter(_plain_log_formatter())
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.axhline(min_samples_per_bin, color="red", linestyle="--", linewidth=0.8)
    cbar.ax.annotate(
        f"red line = min_samples_per_bin ({min_samples_per_bin})\n"
        f"grey = empty bins",
        xy=(1.05, 1.02), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=7, color="#444",
    )

    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _compute_per_class_grids(
    df: pd.DataFrame,
    label_col: str,
    bin_size: float,
    per_sample_weights: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[int, int], int], int, int, int, int]:
    """Reshape sampler per-sample weights into per-(bin, class) grids.

    Cell value = mean weight of all samples of that class in that bin.
    For unsmoothed sampler configurations, every sample in a (bin, class)
    cell has the identical weight, so mean = exact value. Under
    ``smoothing="bilinear"``, weights vary continuously by lat/lon
    within a bin and the mean is a representative summary.
    NaN cell = no samples of that class in that bin.
    """
    str_labels = df[label_col].astype(str).to_numpy()
    lat_bin = np.floor((df["lat"].to_numpy() + 90.0) / bin_size).astype(np.int64)
    lon_bin = np.floor((df["lon"].to_numpy() + 180.0) / bin_size).astype(np.int64)

    helper_df = pd.DataFrame(
        {
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "label": str_labels,
            "weight": per_sample_weights,
        }
    )
    bin_sample_counts: Dict[Tuple[int, int], int] = (
        helper_df.groupby(["lat_bin", "lon_bin"]).size().to_dict()
    )
    cell_weights = (
        helper_df.groupby(["lat_bin", "lon_bin", "label"])["weight"]
        .mean()
        .to_dict()
    )

    classes = sorted(set(str_labels))
    lat_min = int(lat_bin.min())
    lat_max = int(lat_bin.max())
    lon_min = int(lon_bin.min())
    lon_max = int(lon_bin.max())
    n_lat = lat_max - lat_min + 1
    n_lon = lon_max - lon_min + 1

    grids: Dict[str, np.ndarray] = {
        cls: np.full((n_lat, n_lon), np.nan, dtype=np.float64) for cls in classes
    }
    for (li, lo, cls), w in cell_weights.items():
        grids[cls][li - lat_min, lo - lon_min] = w

    return grids, bin_sample_counts, lat_min, lon_min, n_lat, n_lon


def dump_per_bin_class_grids(
    df: pd.DataFrame,
    per_sample_weights: np.ndarray,
    output_dir: Path,
    *,
    bin_size: float,
    min_samples_per_bin: int,
    pool: Literal["lc", "ct"],
) -> None:
    """Visualize the per-sample sampler weights as a per-(bin, class) grid
    and write the 3 inspection artifacts (PNG x2 + CSV) for one pool to
    *output_dir*.

    *per_sample_weights* must come from the constructed sampler — typically
    ``SeasonalTaskBatchSampler._lc_probs`` for ``pool="lc"`` or
    ``SeasonalTaskBatchSampler._ct_probs`` for ``pool="ct"``. They already
    encode every dispatch decision (scope, density, smoothing, fallbacks,
    clipping), so the plot reflects exactly what the model trains on.

    *min_samples_per_bin* is used only as the threshold annotation on the
    sample-count map; the weights themselves already incorporate any
    sparse-bin behavior the sampler chose.

    For ``pool="ct"``, samples without a valid (non-null, non-"ignore")
    croptype label are dropped before binning, mirroring the CT pool used
    by `SeasonalTaskBatchSampler`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if pool == "ct":
        df = df[
            df["croptype_label"].notna()
            & (df["croptype_label"].astype(str) != "ignore")
        ].reset_index(drop=True)
        label_col = "croptype_label"
    else:
        label_col = "landcover_label"

    if len(per_sample_weights) != len(df):
        raise ValueError(
            f"per_sample_weights length {len(per_sample_weights)} does not "
            f"match {pool.upper()} pool size {len(df)}"
        )

    # Sampler stores `_lc_probs` / `_ct_probs` sum-normalized to 1
    # (each value ~ 1/N). Rescale to mean=1 so the plotted weights match
    # the same units as `get_class_weights`-style logs and the LogNorm
    # has reasonable dynamic range.
    per_sample_weights = (
        np.asarray(per_sample_weights, dtype=np.float64) * len(per_sample_weights)
    )

    grids, bin_counts, lat_min, lon_min, n_lat, n_lon = _compute_per_class_grids(
        df, label_col, bin_size, per_sample_weights
    )
    classes = sorted(grids.keys())

    pad = 2.0
    extent = (
        float(df["lon"].min() - pad), float(df["lon"].max() + pad),
        float(df["lat"].min() - pad), float(df["lat"].max() + pad),
    )

    _make_per_class_grid_plot(
        grids, classes, lat_min, lon_min, n_lat, n_lon, bin_size, extent,
        output_dir / f"per_bin_class_weights_{pool}.png",
        f"Sampler weights per (bin, class) — {pool.upper()} pool "
        f"(bin={bin_size}°)",
    )

    _make_bin_count_plot(
        bin_counts, lat_min, lon_min, n_lat, n_lon, bin_size, extent,
        output_dir / f"per_bin_sample_counts_{pool}.png",
        f"Samples per bin ({pool.upper()} pool, bin={bin_size}°)",
        min_samples_per_bin,
    )

    rows = []
    for cls in classes:
        grid = grids[cls]
        n_bins_with_weight = int(np.sum(~np.isnan(grid)))
        rows.append({
            "class": cls,
            "n_samples_total": int((df[label_col].astype(str) == cls).sum()),
            "n_bins_with_weight": n_bins_with_weight,
            "weight_min": float(np.nanmin(grid)) if n_bins_with_weight else np.nan,
            "weight_max": float(np.nanmax(grid)) if n_bins_with_weight else np.nan,
            "weight_median": (
                float(np.nanmedian(grid)) if n_bins_with_weight else np.nan
            ),
        })
    pd.DataFrame(rows).to_csv(
        output_dir / f"per_bin_class_summary_{pool}.csv", index=False
    )
