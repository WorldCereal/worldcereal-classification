"""
grid_utils.py
=============
Shared utilities for the WorldCereal production-grid enrichment pipeline.

Covers:
  - Grid FGB reading helpers
  - Generic per-EPSG WarpedVRT raster-summation engine
  - Potapov cropland proportion (25 m binary raster → fraction)
  - IIASA hybrid cropland fraction (500 m percent raster → fraction)
  - ESA WorldCover per-class proportions (10 m, run on Terrascope)
  - WorldCover props expansion (JSON → per-class float columns + majority LC)
  - Parquet merge helper
  - Grid-based season computation (worldcereal date-based approach)
  - Season overlap metrics (date-based, for 50 km grid)
  - DOY / date conversion utility

Usage (from notebook in the same repo):
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "misc"))
    from grid_utils import (
        list_unique_epsgs, collect_and_merge_epsg_parts,
        run_potapov_pipeline, run_iiasa_pipeline,
        run_worldcover_pipeline, expand_wc_props,
        get_seasons_for_grid_cell, compute_seasons_for_grid,
        compute_season_overlap, doy_to_date,
    )
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio import warp
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import pyogrio  # type: ignore
    HAS_PYOGRIO = True
except Exception:
    HAS_PYOGRIO = False

try:
    import dask  # type: ignore
    from dask.distributed import Client, LocalCluster  # type: ignore
    HAS_DASK = True
except Exception:
    HAS_DASK = False

try:
    import pyarrow.parquet as pq  # type: ignore
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

# ── Constants ─────────────────────────────────────────────────────────────────
GRID_COLS = ["tile_block_id", "tile", "epsg", "antimeridian", "xmin", "ymin", "xmax", "ymax"]
MAX_PIXELS_PER_TILE_WARP = 300_000_000
MAX_PIXELS_PER_BLOCK_READ = 50_000_000


# =============================================================================
# Grid readers
# =============================================================================

def read_grid_epsg_subset(fgb_path: str, epsg: int) -> pd.DataFrame:
    """Return rows for a single EPSG zone from the FlatGeobuf grid."""
    where = f"epsg = {int(epsg)}"
    if HAS_PYOGRIO:
        return pyogrio.read_dataframe(fgb_path, where=where, columns=GRID_COLS, read_geometry=False)
    import geopandas as gpd
    gdf = gpd.read_file(fgb_path)
    gdf = gdf.loc[gdf["epsg"].astype(int) == int(epsg), GRID_COLS].copy()
    if "geometry" in gdf.columns:
        gdf = gdf.drop(columns=["geometry"])
    return pd.DataFrame(gdf)


def list_unique_epsgs(fgb_path: str) -> List[int]:
    """List all unique EPSG codes present in the grid FlatGeobuf."""
    if HAS_PYOGRIO:
        df = pyogrio.read_dataframe(fgb_path, columns=["epsg"], read_geometry=False)
    else:
        import geopandas as gpd
        df = gpd.read_file(fgb_path, columns=["epsg"])
    return sorted(pd.Series(df["epsg"]).dropna().astype(int).unique().tolist())


# =============================================================================
# Parquet helpers
# =============================================================================

def merge_parquet_parts(parts: List[str], out_path: str, value_col: str) -> None:
    """Concatenate per-EPSG parquet files into one output file."""
    if not parts:
        raise ValueError("No part files to merge")
    cols = ["tile_block_id", value_col]
    if HAS_PYARROW:
        writer = None
        for p in parts:
            tbl = pq.read_table(p, columns=cols)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
        if writer is not None:
            writer.close()
    else:
        dfs = [pd.read_parquet(p)[cols] for p in parts]
        pd.concat(dfs, ignore_index=True).to_parquet(out_path, index=False)


def collect_and_merge_epsg_parts(out_dir: str, value_col: str, final_parquet: str) -> pd.DataFrame:
    """Read all epsg_*.parquet files from out_dir and merge them into one DataFrame (and save)."""
    files = sorted(Path(out_dir).glob("epsg_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(final_parquet, index=False)
    print(f"Merged {len(files)} files → {len(merged):,} rows → {final_parquet}")
    return merged


# =============================================================================
# Generic per-EPSG WarpedVRT processing engine
# =============================================================================

def _process_epsg_generic(
    epsg: int,
    fgb_path: str,
    raster_path: str,
    out_dir: str,
    value_col: str,
    compute_fn,           # callable(arr, src_nodata) -> float
    resolution_m: int,
    fail_on_all_empty: bool = True,
    limit_tiles: Optional[int] = None,
) -> str:
    """
    Core per-EPSG processing loop shared by Potapov and IIASA pipelines.

    For each EPSG zone in the grid:
      1. Build a tile-level WarpedVRT of `raster_path` in that UTM zone.
      2. For each grid block inside the tile, read the windowed array and
         call `compute_fn(arr, src_nodata)` to get a scalar summary.
      3. Write results to ``out_dir/epsg_<epsg>.parquet``.

    Returns the path to the output parquet.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"epsg_{int(epsg)}.parquet")
    log_path = os.path.join(out_dir, f"log_epsg_{int(epsg)}.txt")

    if os.path.exists(out_path):
        print(f"[SKIP] EPSG {epsg} already done: {out_path}")
        return out_path

    df = read_grid_epsg_subset(fgb_path, epsg)
    if df.empty:
        pd.DataFrame({"tile_block_id": [], value_col: []}).to_parquet(out_path, index=False)
        return out_path

    anti = df["antimeridian"].fillna(False).astype(bool).to_numpy()
    df_work = df.loc[~anti].copy()
    df_anti = df.loc[anti, ["tile_block_id"]].copy()

    logs = [
        f"EPSG={epsg} rows_total={len(df)} rows_work={len(df_work)} rows_antimeridian={len(df_anti)}",
        f"resolution_m={resolution_m} raster={raster_path}",
    ]

    results_id: List[str] = []
    results_val: List[float] = []

    if df_work.empty:
        for tbid in df["tile_block_id"].astype(str):
            results_id.append(tbid)
            results_val.append(0.0)
    else:
        df_work["tile"] = df_work["tile"].astype(str)
        n_tiles = n_blocks = n_nonempty = n_win_bad = n_read_err = 0

        with rasterio.open(raster_path) as src:
            src_crs = src.crs
            src_nodata = src.nodata

            tile_keys = sorted(df_work["tile"].unique())
            if limit_tiles is not None:
                tile_keys = tile_keys[:int(limit_tiles)]

            from tqdm import tqdm
            for tile_key in tqdm(tile_keys, desc=f"EPSG {epsg}"):
                g = df_work.loc[df_work["tile"] == tile_key].reset_index(drop=True)
                if g.empty:
                    continue
                n_tiles += 1

                minx = float(g["xmin"].min()); miny = float(g["ymin"].min())
                maxx = float(g["xmax"].max()); maxy = float(g["ymax"].max())
                if (maxx - minx) <= 0 or (maxy - miny) <= 0:
                    continue

                out_w = max(1, int(round((maxx - minx) / resolution_m)))
                out_h = max(1, int(round((maxy - miny) / resolution_m)))
                if out_w * out_h > MAX_PIXELS_PER_TILE_WARP:
                    logs.append(f"[SKIP_TILE] {tile_key}: canvas too large")
                    for tbid in g["tile_block_id"].astype(str):
                        results_id.append(tbid); results_val.append(0.0)
                    continue

                snapped_maxx = minx + out_w * resolution_m
                snapped_maxy = miny + out_h * resolution_m
                dst_transform = rasterio.transform.Affine(
                    resolution_m, 0.0, minx, 0.0, -resolution_m, snapped_maxy
                )

                try:
                    src_minx, src_miny, src_maxx, src_maxy = warp.transform_bounds(
                        f"EPSG:{epsg}", src_crs, minx, miny, snapped_maxx, snapped_maxy, densify_pts=21
                    )
                except Exception as e:
                    logs.append(f"[SKIP_TILE] {tile_key}: transform_bounds failed: {e}")
                    for tbid in g["tile_block_id"].astype(str):
                        results_id.append(tbid); results_val.append(0.0)
                    continue

                vrt_opts = dict(
                    crs=f"EPSG:{epsg}", transform=dst_transform,
                    width=out_w, height=out_h, resampling=Resampling.nearest,
                    src_bounds=(src_minx, src_miny, src_maxx, src_maxy),
                    src_nodata=src_nodata, nodata=src_nodata,
                )

                try:
                    with WarpedVRT(src, **vrt_opts) as tvrt:
                        tvrt_bounds = tvrt.bounds
                        for r in g.itertuples(index=False):
                            n_blocks += 1
                            tbid = str(r.tile_block_id)
                            ixmin = max(float(r.xmin), tvrt_bounds.left)
                            iymin = max(float(r.ymin), tvrt_bounds.bottom)
                            ixmax = min(float(r.xmax), tvrt_bounds.right)
                            iymax = min(float(r.ymax), tvrt_bounds.top)

                            if not (ixmin < ixmax and iymin < iymax):
                                n_win_bad += 1
                                results_id.append(tbid); results_val.append(0.0)
                                continue
                            try:
                                w = from_bounds(ixmin, iymin, ixmax, iymax, transform=tvrt.transform)
                                w = w.round_offsets().round_lengths()
                                if w.width <= 0 or w.height <= 0 or w.width * w.height > MAX_PIXELS_PER_BLOCK_READ:
                                    n_win_bad += 1
                                    results_id.append(tbid); results_val.append(0.0)
                                    continue
                                arr = tvrt.read(1, window=w, boundless=False, out_dtype="uint8")
                                val = compute_fn(arr, src_nodata)
                                if val > 0.0:
                                    n_nonempty += 1
                                results_id.append(tbid); results_val.append(val)
                            except Exception:
                                n_read_err += 1
                                results_id.append(tbid); results_val.append(0.0)
                except Exception as e:
                    logs.append(f"[SKIP_TILE] {tile_key}: WarpedVRT failed: {e}")
                    for tbid in g["tile_block_id"].astype(str):
                        results_id.append(tbid); results_val.append(0.0)

        logs += [
            f"tiles={n_tiles} blocks={n_blocks} nonempty={n_nonempty}",
            f"win_bad={n_win_bad} read_err={n_read_err}",
            f"src_crs={src_crs} src_nodata={src_nodata}",
        ]
        if fail_on_all_empty and n_blocks > 0 and n_nonempty == 0:
            print(f"[WARN] EPSG {epsg}: all blocks empty — check {log_path}")

    # antimeridian blocks → 0
    for tbid in df_anti["tile_block_id"].astype(str):
        results_id.append(tbid); results_val.append(0.0)

    with open(log_path, "w") as f:
        f.write("\n".join(logs) + "\n")

    pd.DataFrame({"tile_block_id": results_id, value_col: results_val}).to_parquet(out_path, index=False)
    return out_path


# =============================================================================
# Potapov cropland proportion  (binary 0/1 at ~25 m)
# =============================================================================

def _potapov_compute(arr: np.ndarray, src_nodata: Optional[float]) -> float:
    """Fraction of pixels == 1 (cropland present), excluding nodata."""
    flat = arr.ravel()
    if flat.size == 0:
        return 0.0
    if src_nodata is not None and not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
        flat = flat[flat != src_nodata]
    if flat.size == 0:
        return 0.0
    return float(np.count_nonzero(flat == 1) / flat.size)


def process_potapov_epsg(
    epsg: int,
    fgb_path: str,
    potapov_vrt_path: str,
    out_dir: str,
    resolution_m: int = 25,
    limit_tiles: Optional[int] = None,
) -> str:
    """Compute potapov_prop for one EPSG zone and write to parquet."""
    return _process_epsg_generic(
        epsg, fgb_path, potapov_vrt_path, out_dir,
        value_col="potapov_prop",
        compute_fn=_potapov_compute,
        resolution_m=resolution_m,
        limit_tiles=limit_tiles,
    )


def run_potapov_pipeline(
    fgb_path: str,
    potapov_vrt_path: str,
    out_dir: str,
    resolution_m: int = 25,
    n_workers: int = 5,
) -> List[str]:
    """
    Run Potapov proportion computation across all EPSG zones.
    Uses Dask if available, otherwise sequential.
    """
    epsgs = list_unique_epsgs(fgb_path)
    existing = {int(p.stem.split("_")[1]) for p in Path(out_dir).glob("epsg_*.parquet")}
    epsgs = [e for e in epsgs if e not in existing]
    print(f"[Potapov] EPSGs to process: {len(epsgs)}")

    if HAS_DASK and epsgs:
        cluster = LocalCluster(processes=True, threads_per_worker=1,
                               n_workers=n_workers, memory_limit="5.25GB")
        client = Client(cluster)
        tasks = [dask.delayed(process_potapov_epsg)(e, fgb_path, potapov_vrt_path, out_dir, resolution_m)
                 for e in epsgs]
        parts = list(dask.compute(*tasks))
        client.shutdown()
    else:
        parts = [process_potapov_epsg(e, fgb_path, potapov_vrt_path, out_dir, resolution_m)
                 for e in epsgs]
    return parts


# =============================================================================
# IIASA hybrid cropland fraction  (percent 0–100 at 500 m)
# =============================================================================

def _iiasa_compute(arr: np.ndarray, src_nodata: Optional[float]) -> float:
    """Mean cropland fraction in [0, 1] from a 0–100 percent raster, excluding nodata."""
    flat = arr.ravel()
    if flat.size == 0:
        return 0.0
    if src_nodata is not None and not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
        flat = flat[flat != src_nodata]
    if flat.size == 0:
        return 0.0
    flat = np.clip(flat.astype(np.float32, copy=False), 0.0, 100.0)
    return float(np.mean(flat) / 100.0)


def process_iiasa_epsg(
    epsg: int,
    fgb_path: str,
    iiasa_tif_path: str,
    out_dir: str,
    resolution_m: int = 500,
    limit_tiles: Optional[int] = None,
) -> str:
    """Compute iiasa_hybrid_props for one EPSG zone and write to parquet."""
    return _process_epsg_generic(
        epsg, fgb_path, iiasa_tif_path, out_dir,
        value_col="iiasa_hybrid_props",
        compute_fn=_iiasa_compute,
        resolution_m=resolution_m,
        limit_tiles=limit_tiles,
    )


def run_iiasa_pipeline(
    fgb_path: str,
    iiasa_tif_path: str,
    out_dir: str,
    resolution_m: int = 500,
    n_workers: int = 5,
) -> List[str]:
    """
    Run IIASA proportion computation across all EPSG zones.
    Uses Dask if available, otherwise sequential.
    """
    epsgs = list_unique_epsgs(fgb_path)
    existing = {int(p.stem.split("_")[1]) for p in Path(out_dir).glob("epsg_*.parquet")}
    epsgs = [e for e in epsgs if e not in existing]
    print(f"[IIASA] EPSGs to process: {len(epsgs)}")

    if HAS_DASK and epsgs:
        cluster = LocalCluster(processes=True, threads_per_worker=1,
                               n_workers=n_workers, memory_limit="5.25GB")
        client = Client(cluster)
        tasks = [dask.delayed(process_iiasa_epsg)(e, fgb_path, iiasa_tif_path, out_dir, resolution_m)
                 for e in epsgs]
        parts = list(dask.compute(*tasks))
        client.shutdown()
    else:
        parts = [process_iiasa_epsg(e, fgb_path, iiasa_tif_path, out_dir, resolution_m)
                 for e in epsgs]
    return parts


# =============================================================================
# ESA WorldCover per-class proportions  (10 m, runs on Terrascope)
# =============================================================================

# WorldCover class codes and their names
WC_CLASSES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], dtype=np.int16)
WC_LEGEND: dict = {
    0:   "background",
    10:  "tree_cover",
    20:  "shrubland",
    30:  "grassland",
    40:  "cropland",
    50:  "built_up",
    60:  "bare_sparse_vegetation",
    70:  "snow_ice",
    80:  "permanent_water_bodies",
    90:  "herbaceous_wetland",
    95:  "mangroves",
    100: "moss_lichen",
}

# Default WorldCover data root on Terrascope / VITO HPC
_WC_ROOT_DEFAULT = "/data/MTDA/WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/MAP"


def _ensure_worldcover_vrt(vrt_path: str, root: str) -> str:
    """
    Build (once) a global VRT over all WorldCover *_Map.tif tiles under `root`.
    Uses gdalbuildvrt; falls back to osgeo.gdal if not on PATH.
    """
    if os.path.exists(vrt_path) and os.path.getsize(vrt_path) > 0:
        return vrt_path

    pattern = os.path.join(root, "**", "*_Map.tif")
    tifs = sorted(glob.glob(pattern, recursive=True))
    if not tifs:
        raise FileNotFoundError(f"No '*_Map.tif' found under: {root}")

    lst = vrt_path + ".filelist.txt"
    with open(lst, "w") as fh:
        for p in tifs:
            fh.write(p + "\n")

    cmd = ["gdalbuildvrt", "-overwrite", "-input_file_list", lst,
           "-srcnodata", "0", "-vrtnodata", "0", vrt_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(vrt_path) and os.path.getsize(vrt_path) > 0:
            return vrt_path
    except Exception:
        pass

    try:
        from osgeo import gdal  # type: ignore
        opts = gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0)
        ds = gdal.BuildVRT(vrt_path, tifs, options=opts)
        if ds is None:
            raise RuntimeError("gdal.BuildVRT returned None")
        ds = None
        return vrt_path
    except Exception as e:
        raise RuntimeError(
            "Could not build WorldCover VRT. Need gdalbuildvrt or osgeo.gdal.\n"
            f"Error: {e}"
        )


def _wc_bincount(arr: np.ndarray) -> np.ndarray:
    """Count pixels per WorldCover class; returns array aligned to WC_CLASSES."""
    flat = arr.ravel().astype(np.int32)
    flat = flat[flat != 0]          # 0 is nodata / background
    if flat.size == 0:
        return np.zeros(len(WC_CLASSES), dtype=np.uint32)
    bc = np.bincount(flat, minlength=256)
    return bc[WC_CLASSES].astype(np.uint32)


def _counts_to_props_json(counts: np.ndarray) -> str:
    """Convert per-class counts to a compact JSON proportion string."""
    total = int(counts.sum())
    if total <= 0:
        return "{}"
    nz = np.nonzero(counts > 0)[0]
    d = {int(WC_CLASSES[i]): round(float(counts[i] / total), 6) for i in nz}
    return json.dumps(d, separators=(",", ":"), sort_keys=True)


def _process_worldcover_epsg(
    epsg: int,
    fgb_path: str,
    wc_vrt_path: str,
    out_dir: str,
    resolution_m: int = 10,
    limit_tiles: Optional[int] = None,
) -> str:
    """
    Per-EPSG WorldCover processing (incremental — skips completed rows).
    Stores `esa_wc_props` as a JSON string of {class_code: proportion} per tile_block_id.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"epsg_{int(epsg)}.parquet")

    if os.path.exists(out_path):
        print(f"[SKIP] EPSG {epsg} already done: {out_path}")
        return out_path

    df = read_grid_epsg_subset(fgb_path, epsg)
    if df.empty:
        pd.DataFrame({"tile_block_id": [], "esa_wc_props": []}).to_parquet(out_path, index=False)
        return out_path

    anti    = df["antimeridian"].fillna(False).astype(bool).to_numpy()
    df_work = df.loc[~anti].copy()
    df_anti = df.loc[anti, ["tile_block_id"]].copy()

    results_id:  List[str] = []
    results_val: List[str] = []

    if not df_work.empty:
        df_work["tile"] = df_work["tile"].astype(str)
        tile_keys = sorted(df_work["tile"].unique())
        if limit_tiles is not None:
            tile_keys = tile_keys[:int(limit_tiles)]

        with rasterio.open(wc_vrt_path) as src:
            src_crs = src.crs
            from tqdm import tqdm
            for tile_key in tqdm(tile_keys, desc=f"WC EPSG {epsg}"):
                g = df_work.loc[df_work["tile"] == tile_key].reset_index(drop=True)
                if g.empty:
                    continue

                minx = float(g["xmin"].min()); miny = float(g["ymin"].min())
                maxx = float(g["xmax"].max()); maxy = float(g["ymax"].max())
                if (maxx - minx) <= 0 or (maxy - miny) <= 0:
                    continue

                out_w = max(1, int(round((maxx - minx) / resolution_m)))
                out_h = max(1, int(round((maxy - miny) / resolution_m)))
                if out_w * out_h > MAX_PIXELS_PER_TILE_WARP:
                    for tbid in g["tile_block_id"].astype(str):
                        results_id.append(tbid); results_val.append("{}")
                    continue

                snapped_maxx = minx + out_w * resolution_m
                snapped_maxy = miny + out_h * resolution_m
                dst_transform = rasterio.transform.Affine(
                    resolution_m, 0.0, minx, 0.0, -resolution_m, snapped_maxy
                )

                try:
                    bx, by, bX, bY = warp.transform_bounds(
                        f"EPSG:{epsg}", src_crs, minx, miny, snapped_maxx, snapped_maxy, densify_pts=21
                    )
                except Exception:
                    for tbid in g["tile_block_id"].astype(str):
                        results_id.append(tbid); results_val.append("{}")
                    continue

                vrt_opts = dict(
                    crs=f"EPSG:{epsg}", transform=dst_transform,
                    width=out_w, height=out_h, resampling=Resampling.nearest,
                    src_bounds=(bx, by, bX, bY), src_nodata=0, nodata=0,
                )
                try:
                    with WarpedVRT(src, **vrt_opts) as tvrt:
                        bounds = tvrt.bounds
                        for r in g.itertuples(index=False):
                            tbid = str(r.tile_block_id)
                            ixmin = max(float(r.xmin), bounds.left)
                            iymin = max(float(r.ymin), bounds.bottom)
                            ixmax = min(float(r.xmax), bounds.right)
                            iymax = min(float(r.ymax), bounds.top)
                            if not (ixmin < ixmax and iymin < iymax):
                                results_id.append(tbid); results_val.append("{}")
                                continue
                            try:
                                w = from_bounds(ixmin, iymin, ixmax, iymax, transform=tvrt.transform)
                                w = w.round_offsets().round_lengths()
                                if w.width <= 0 or w.height <= 0 or w.width * w.height > MAX_PIXELS_PER_BLOCK_READ:
                                    results_id.append(tbid); results_val.append("{}")
                                    continue
                                arr = tvrt.read(1, window=w, boundless=False, out_dtype="uint8")
                                props = _counts_to_props_json(_wc_bincount(arr))
                                results_id.append(tbid); results_val.append(props)
                            except Exception:
                                results_id.append(tbid); results_val.append("{}")
                except Exception:
                    for tbid in g["tile_block_id"].astype(str):
                        results_id.append(tbid); results_val.append("{}")

    # antimeridian → empty props
    for tbid in df_anti["tile_block_id"].astype(str):
        results_id.append(tbid); results_val.append("{}")

    pd.DataFrame({"tile_block_id": results_id, "esa_wc_props": results_val}).to_parquet(out_path, index=False)
    print(f"[WC] EPSG {epsg}: wrote {len(results_id)} rows → {out_path}")
    return out_path


def run_worldcover_pipeline(
    fgb_path: str,
    wc_vrt_path: str,
    out_dir: str,
    wc_root: str = _WC_ROOT_DEFAULT,
    resolution_m: int = 10,
    n_workers: int = 3,
) -> List[str]:
    """
    Run ESA WorldCover per-class proportion computation across all EPSG zones.

    ⚠  Intended to run on Terrascope / VITO HPC where the WorldCover tiles are
       available at ``wc_root``.  The VRT at ``wc_vrt_path`` is built once
       automatically if it does not exist yet.

    Args:
        fgb_path    : Path to the grid FlatGeobuf (``blocks_global_v13.fgb``).
        wc_vrt_path : Path where the global VRT will be written / read from.
        out_dir     : Directory for per-EPSG ``epsg_*.parquet`` outputs.
        wc_root     : Root directory of WorldCover *_Map.tif tiles.
        resolution_m: Working resolution in metres (default 10).
        n_workers   : Number of Dask workers (used only when Dask is available).

    Returns:
        List of per-EPSG parquet paths.
    """
    wc_vrt = _ensure_worldcover_vrt(wc_vrt_path, wc_root)
    print(f"[WC] Using VRT: {wc_vrt}")

    epsgs    = list_unique_epsgs(fgb_path)
    existing = {int(p.stem.split("_")[1]) for p in Path(out_dir).glob("epsg_*.parquet")}
    epsgs    = [e for e in epsgs if e not in existing]
    print(f"[WC] EPSGs to process: {len(epsgs)}")

    if HAS_DASK and epsgs:
        cluster = LocalCluster(processes=True, threads_per_worker=1,
                               n_workers=n_workers, memory_limit="8GB")
        client  = Client(cluster)
        tasks   = [dask.delayed(_process_worldcover_epsg)(e, fgb_path, wc_vrt, out_dir, resolution_m)
                   for e in epsgs]
        parts   = list(dask.compute(*tasks))
        client.shutdown()
    else:
        parts = [_process_worldcover_epsg(e, fgb_path, wc_vrt, out_dir, resolution_m) for e in epsgs]
    return parts


def expand_wc_props(
    df: pd.DataFrame,
    props_col: str = "esa_wc_props",
    exclude_background: bool = True,
) -> pd.DataFrame:
    """
    Expand the ``esa_wc_props`` JSON column into per-class float columns
    (``wc_<name>``) and add ``wc_majority_lc`` (name of the dominant class,
    excluding background by default).

    Args:
        df                : DataFrame containing ``props_col``.
        props_col         : Name of the JSON props column.
        exclude_background: If True, background (class 0) is ignored when
                            computing the majority landcover.

    Returns:
        Copy of ``df`` with added columns; original props column is kept.
    """
    def _parse(x) -> dict:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return {}
        if isinstance(x, dict):
            return {int(k): float(v) for k, v in x.items()}
        s = str(x).strip()
        if not s or s == "{}":
            return {}
        try:
            return {int(k): float(v) for k, v in json.loads(s).items()}
        except Exception:
            try:
                return {int(k): float(v) for k, v in json.loads(s.replace("'", '"')).items()}
            except Exception:
                return {}

    def _majority(props: dict) -> str:
        candidates = {k: v for k, v in props.items() if k != 0} if exclude_background else props
        if not candidates:
            candidates = props  # fall back to including background
        if not candidates:
            return WC_LEGEND.get(0, "background")
        best = max(candidates.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return WC_LEGEND.get(best, str(best))

    out = df.copy()
    parsed = out[props_col].map(_parse)

    for code, name in WC_LEGEND.items():
        out[f"wc_{name}"] = parsed.map(lambda d, c=code: float(d.get(c, 0.0))).astype(np.float32)

    out["wc_majority_lc"] = parsed.map(_majority).astype("category")
    return out


# =============================================================================
# Season computation  (worldcereal date-based, per grid cell)
# =============================================================================

def get_seasons_for_grid_cell(row, year: int = 2024, season: str = "tc-annual") -> dict:
    """
    Compute start/end dates for a single grid cell using worldcereal season lookup.

    Args:
        row  : DataFrame row with xmin, ymin, xmax, ymax, epsg.
        year : Reference year for season computation.
        season: Season identifier ("tc-annual", "tc-s1", "tc-s2").

    Returns:
        dict with keys season_start, season_end, season_error.
    """
    try:
        from openeo_gfmap import BoundingBoxExtent
        from worldcereal.seasons import get_season_dates_for_extent

        extent = BoundingBoxExtent(
            west=float(row["xmin"]), south=float(row["ymin"]),
            east=float(row["xmax"]), north=float(row["ymax"]),
            epsg=int(row["epsg"]),
        )
        ctx = get_season_dates_for_extent(extent=extent, year=year, season=season,
                                          max_seasonality_difference=60)
        return {"season_start": ctx.start_date, "season_end": ctx.end_date, "season_error": None}
    except Exception as e:
        return {"season_start": None, "season_end": None, "season_error": str(e)}


def compute_seasons_for_grid(
    grid_data,
    year: int = 2024,
    season: str = "tc-annual",
):
    """
    Append season_start, season_end, season_error, season_los columns to grid_data.

    Args:
        grid_data  : GeoDataFrame with grid cells.
        year       : Reference year.
        season     : Season identifier.

    Returns:
        Copy of grid_data with added season columns.
    """
    import warnings
    result = grid_data.copy()

    print(f"  Computing {season} seasons for {len(grid_data):,} grid cells...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        season_info = grid_data.apply(
            lambda row: get_seasons_for_grid_cell(row, year=year, season=season),
            axis=1,
        )
    season_df = pd.DataFrame(season_info.tolist(), index=grid_data.index)
    result["season_start"] = season_df["season_start"]
    result["season_end"] = season_df["season_end"]
    result["season_error"] = season_df["season_error"]

    result["season_los"] = (
        pd.to_datetime(result["season_end"]) - pd.to_datetime(result["season_start"])
    ).dt.days
    return result


# =============================================================================
# Season overlap metrics  (date-based, for the 50 km grid combined dataframe)
# =============================================================================

def compute_season_overlap(row) -> dict:
    """
    Compute overlap metrics between S1 and S2 seasons for a single grid row.

    Returns a dict with:
      overlap_days, overlap_fraction_s1, overlap_fraction_s2,
      overlap_fraction_max, seasons_distinct, overlap_error.
    """
    _nan = {
        "overlap_days": np.nan, "overlap_fraction_s1": np.nan,
        "overlap_fraction_s2": np.nan, "overlap_fraction_max": np.nan,
        "seasons_distinct": np.nan, "overlap_error": None,
    }
    try:
        s1_start = pd.to_datetime(row["season_start_s1"])
        s1_end   = pd.to_datetime(row["season_end_s1"])
        s2_start = pd.to_datetime(row["season_start_s2"])
        s2_end   = pd.to_datetime(row["season_end_s2"])

        if any(pd.isna(v) for v in [s1_start, s1_end, s2_start, s2_end]):
            return {**_nan, "overlap_error": "Missing season dates"}

        s1_len = (s1_end - s1_start).days
        s2_len = (s2_end - s2_start).days
        ov_start = max(s1_start, s2_start)
        ov_end   = min(s1_end, s2_end)
        ov_days  = max(0, (ov_end - ov_start).days) if ov_start <= ov_end else 0
        max_len  = max(s1_len, s2_len)

        return {
            "overlap_days": int(ov_days),
            "overlap_fraction_s1": float(ov_days / s1_len) if s1_len > 0 else np.nan,
            "overlap_fraction_s2": float(ov_days / s2_len) if s2_len > 0 else np.nan,
            "overlap_fraction_max": float(ov_days / max_len) if max_len > 0 else np.nan,
            "seasons_distinct": bool(ov_days / max_len < 0.3) if max_len > 0 else np.nan,
            "overlap_error": None,
        }
    except Exception as e:
        return {**_nan, "overlap_error": str(e)}


# =============================================================================
# DOY / date utilities
# =============================================================================

def doy_to_date(doy: int, year: int = 2024) -> pd.Timestamp:
    """Convert day-of-year (1–365) to a date in the given reference year."""
    return pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(doy - 1, unit="D")
