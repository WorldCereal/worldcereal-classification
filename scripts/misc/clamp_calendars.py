"""clamp_calendars.py — Clamp S1∪S2 season union to ≤ 365 days.

Operates on both the ``seasonality_lookup.parquet`` and the crop-calendar
GeoTIFFs that live in the same folder.  Only the files that are actually
modified are backed up into an ``orig_files/`` sub-folder:

  --target parquet  →  backs up ``seasonality_lookup.parquet``
  --target tifs     →  backs up ``S1_SOS_WGS84.tif`` and ``S2_SOS_WGS84.tif``
  --target both     →  backs up all three of the above

Usage
-----
    # Clamp both the parquet and all TIF files (default)
    python clamp_calendars.py <folder>

    # Clamp only the parquet
    python clamp_calendars.py <folder> --target parquet

    # Clamp only the TIF files
    python clamp_calendars.py <folder> --target tifs

Algorithm
---------
For every dual-season grid point whose S1∪S2 bounding-box union exceeds
365 days the earlier-ending season's SOS is moved forward to
``annual_sos_doy``, guaranteeing union == exactly 365 days.

  • S2 ends later (95 % of cases) → clamp ``s1_sos_doy``
  • S1 ends later ( 5 % of cases) → clamp ``s2_sos_doy``

Single-season points and points already within 365 days are not touched.

TIF patching
------------
The TIF rasters are 0.5° global grids (360×720, int16, nodata=0).  Each pixel
center maps exactly to one row in the parquet.  The same clamping masks derived
from the parquet are re-projected to (row, col) raster indices and applied
directly to the in-memory raster arrays before writing back with rasterio
(preserving all original metadata — CRS, transform, nodata, etc.).

After patching the TIFs, re-running ``regenerate_seasonality_lookup.py`` on the
same folder will reproduce the clamped parquet from scratch.
"""

import argparse
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ── Column / file name mappings ──
DOY_COLS = [
    "s1_sos_doy",
    "s1_eos_doy",
    "s2_sos_doy",
    "s2_eos_doy",
    "annual_sos_doy",
    "annual_eos_doy",
]

# Parquet column → TIF file name
TIF_NAMES: dict[str, str] = {
    "s1_sos_doy":     "S1_SOS_WGS84.tif",
    "s1_eos_doy":     "S1_EOS_WGS84.tif",
    "s2_sos_doy":     "S2_SOS_WGS84.tif",
    "s2_eos_doy":     "S2_EOS_WGS84.tif",
    "annual_sos_doy": "ANNUAL_SOS_WGS84.tif",
    "annual_eos_doy": "ANNUAL_EOS_WGS84.tif",
}

PARQUET_NAME = "seasonality_lookup.parquet"
ORIG_SUBDIR  = "orig_files"

def _d(doy: int, yr: int) -> date:
    return date(yr, 1, 1) + timedelta(days=int(doy) - 1)


def _union_span_days(s1_sos, s1_eos, s2_sos, s2_eos, ref_year: int = 2023) -> int:
    """Calendar days in the S1∪S2 bounding box (handles year-wrapping)."""
    s1a = _d(s1_sos, ref_year - 1) if s1_sos > s1_eos else _d(s1_sos, ref_year)
    s1b = _d(s1_eos, ref_year)
    s2a = _d(s2_sos, ref_year - 1) if s2_sos > s2_eos else _d(s2_sos, ref_year)
    s2b = _d(s2_eos, ref_year)
    return (max(s1b, s2b) - min(s1a, s2a)).days + 1


def compute_union_spans(df: pd.DataFrame) -> np.ndarray:
    """Return an int64 array of union span (days) for every row in *df*."""
    spans = np.zeros(len(df), dtype=np.int64)
    for i, (_, r) in enumerate(df.iterrows()):
        if r["s2_sos_doy"] == 0:
            sos, eos = int(r["s1_sos_doy"]), int(r["s1_eos_doy"])
            spans[i] = (365 - sos) + eos + 1 if sos > eos else eos - sos + 1
        else:
            spans[i] = _union_span_days(
                r["s1_sos_doy"], r["s1_eos_doy"],
                r["s2_sos_doy"], r["s2_eos_doy"],
            )
    return spans

def _backup(src: Path, orig_dir: Path) -> None:
    """Copy *src* into *orig_dir* if it hasn't been backed up yet."""
    orig_dir.mkdir(parents=True, exist_ok=True)
    dest = orig_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
        print(f"    backed up → {dest}")
    else:
        print(f"    backup already exists, skipping → {dest}")


def _compute_clamp_masks(
    df: pd.DataFrame,
) -> tuple[
    pd.Series,  # dual_mask
    pd.Series,  # s2_anchored
    pd.Series,  # s2_normal
    pd.Series,  # s2_zero
    pd.Series,  # s1_anchored
    pd.Series,  # s1_normal
    pd.Series,  # s1_zero
]:
    """Return all boolean masks needed for clamping, derived from *df*.

    *df* must have all DOY_COLS cast to int64.
    """
    df["_union_days"] = compute_union_spans(df)

    dual_mask = df["s2_sos_doy"] != 0
    over365   = df["_union_days"] > 365

    n_over = (over365 & dual_mask).sum()
    print(f"  Dual-season points with union > 365 days: {n_over:,}")

    # Anchor: which season ends later (annual_eos == that season's eos)
    s2_anchored = over365 & dual_mask & (df["annual_eos_doy"] == df["s2_eos_doy"])
    s1_anchored = (
        over365 & dual_mask
        & (df["annual_eos_doy"] == df["s1_eos_doy"])
        & ~s2_anchored
    )
    print(f"  S2-anchored (clamp s1_sos): {s2_anchored.sum():,}")
    print(f"  S1-anchored (clamp s2_sos): {s1_anchored.sum():,}")

    # Guard: annual_sos_doy == 0 → use DOY=1 instead
    s2_normal = s2_anchored & (df["annual_sos_doy"] != 0)
    s2_zero   = s2_anchored & (df["annual_sos_doy"] == 0)
    s1_normal = s1_anchored & (df["annual_sos_doy"] != 0)
    s1_zero   = s1_anchored & (df["annual_sos_doy"] == 0)

    print(f"  annual_sos=0 guard rows:    {(s2_zero.sum() + s1_zero.sum()):,}  (→ DOY=1)")

    df.drop(columns=["_union_days"], inplace=True)
    return dual_mask, s2_anchored, s2_normal, s2_zero, s1_anchored, s1_normal, s1_zero

def clamp_parquet(folder: Path) -> None:
    parquet_path = folder / PARQUET_NAME
    orig_dir     = folder / ORIG_SUBDIR

    print(f"\n{'='*65}")
    print(f"  PARQUET  →  {parquet_path}")
    print(f"{'='*65}")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} grid points")
    print(f"  Dual-season:   {(df['s2_sos_doy'] != 0).sum():,}")
    print(f"  Single-season: {(df['s2_sos_doy'] == 0).sum():,}")

    for c in DOY_COLS:
        df[c] = df[c].astype(np.int64)

    print("\nComputing S1∪S2 union spans …")
    dual_mask, s2_anchored, s2_normal, s2_zero, s1_anchored, s1_normal, s1_zero = (
        _compute_clamp_masks(df)
    )

    if (s2_anchored.sum() + s1_anchored.sum()) == 0:
        print("Nothing to do — all unions already ≤ 365 days.")
        return

    # Apply
    df.loc[s2_normal, "s1_sos_doy"] = df.loc[s2_normal, "annual_sos_doy"]
    df.loc[s2_zero,   "s1_sos_doy"] = 1
    df.loc[s1_normal, "s2_sos_doy"] = df.loc[s1_normal, "annual_sos_doy"]
    df.loc[s1_zero,   "s2_sos_doy"] = 1

    # Verify
    print("\nVerifying …")
    df["_check"] = compute_union_spans(df)
    still_over = (df["_check"] > 365) & dual_mask
    print(f"  Points still > 365 days: {still_over.sum():,}  (should be 0)")
    if still_over.sum() > 0:
        print(" Verification failed — aborting.")
        sys.exit(1)
    df.drop(columns=["_check"], inplace=True)

    # Backup into orig_files/ & write
    _backup(parquet_path, orig_dir)
    out = df[["lat", "lon"] + DOY_COLS].copy()
    for c in DOY_COLS:
        out[c] = out[c].astype(np.uint16)
    out.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"  Written → {parquet_path}  ({out.shape[0]:,} rows)")

    n_changed = s2_anchored.sum() + s1_anchored.sum()
    print(f"\n  Points clamped: {n_changed:,} ({100*n_changed/dual_mask.sum():.1f}% of dual-season)")
    print(f"    └ s1_sos clamped (S2 ends later): {s2_anchored.sum():>6,}")
    print(f"    └ s2_sos clamped (S1 ends later): {s1_anchored.sum():>6,}")
    print(f"    └ annual_sos=0 guard (→ DOY=1):   {s2_zero.sum()+s1_zero.sum():>6,}")

def clamp_tifs(folder: Path) -> None:
    """Patch the six crop-calendar TIFs in *folder* with the same clamp logic.

    Strategy
    --------
    1. Load all six TIFs into int16 numpy arrays.
    2. Build a tiny DataFrame of all pixel-center lat/lons (same as
       regenerate_seasonality_lookup does) and attach the raster values.
    3. Run ``_compute_clamp_masks`` on that DataFrame.
    4. Convert the DataFrame row indices back to (row, col) raster indices.
    5. Patch the in-memory arrays for S1_SOS / S2_SOS.
    6. Backup originals → orig_files/, then write updated arrays back with
       rasterio (preserving all original metadata).
    """
    orig_dir = folder / ORIG_SUBDIR

    print(f"\n{'='*65}")
    print(f"  TIF FILES  →  {folder}")
    print(f"{'='*65}")

    # ── Load all TIFs ──
    arrays:    dict[str, np.ndarray] = {}
    profiles:  dict[str, dict]       = {}
    transform = None

    for col, tif_name in TIF_NAMES.items():
        tif_path = folder / tif_name
        if not tif_path.exists():
            raise FileNotFoundError(f"Expected TIF not found: {tif_path}")
        with rasterio.open(tif_path) as src:
            arrays[col]   = src.read(1).astype(np.int16)
            profiles[col] = src.profile.copy()
            if transform is None:
                transform = src.transform
                height, width = src.height, src.width
        print(f"  Loaded {tif_name}: shape={arrays[col].shape}, dtype={arrays[col].dtype}")

    assert transform is not None

    # ── Build pixel-center DataFrame (mirrors regenerate_seasonality_lookup) ──
    col_idx = np.arange(width)
    row_idx = np.arange(height)
    lon_centers = (transform.c + (col_idx + 0.5) * transform.a).astype(np.float32)
    lat_centers = (transform.f + (row_idx + 0.5) * transform.e).astype(np.float32)

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    pixel_df = pd.DataFrame({"lat": lat_flat, "lon": lon_flat})
    for col in DOY_COLS:
        pixel_df[col] = arrays[col].ravel().astype(np.int64)

    # Keep track of which flat indices are "active" (any band non-zero), but
    # we need to run mask computation on ALL pixels (including all-zero ones)
    # so that the row indexing stays consistent with the 2-D raster layout.
    print(f"\n  Total pixels: {len(pixel_df):,}  ({height}×{width})")
    print(f"  Non-zero in S1_SOS: {(pixel_df['s1_sos_doy'] != 0).sum():,}")

    # ── Compute clamp masks ──
    print("\nComputing clamp masks from raster pixel values …")
    (
        dual_mask, s2_anchored, s2_normal, s2_zero,
        s1_anchored, s1_normal, s1_zero,
    ) = _compute_clamp_masks(pixel_df)

    n_changed = s2_anchored.sum() + s1_anchored.sum()
    if n_changed == 0:
        print("Nothing to do — all unions already ≤ 365 days.")
        return

    # ── Apply clamp to in-memory arrays ──
    # Flat indices → (row, col) in the 2-D raster
    flat_idx_s2_normal = np.where(s2_normal.values)[0]
    flat_idx_s2_zero   = np.where(s2_zero.values)[0]
    flat_idx_s1_normal = np.where(s1_normal.values)[0]
    flat_idx_s1_zero   = np.where(s1_zero.values)[0]

    s1_sos_flat = arrays["s1_sos_doy"].ravel()   # view into 2-D array
    s2_sos_flat = arrays["s2_sos_doy"].ravel()
    ann_sos_flat = arrays["annual_sos_doy"].ravel()

    # S2-anchored: set S1_SOS = annual_sos (or 1 if annual_sos==0)
    s1_sos_flat[flat_idx_s2_normal] = ann_sos_flat[flat_idx_s2_normal].astype(np.int16)
    s1_sos_flat[flat_idx_s2_zero]   = np.int16(1)

    # S1-anchored: set S2_SOS = annual_sos (or 1 if annual_sos==0)
    s2_sos_flat[flat_idx_s1_normal] = ann_sos_flat[flat_idx_s1_normal].astype(np.int16)
    s2_sos_flat[flat_idx_s1_zero]   = np.int16(1)

    print(f"\n  Patched S1_SOS: {len(flat_idx_s2_normal) + len(flat_idx_s2_zero):,} pixels")
    print(f"  Patched S2_SOS: {len(flat_idx_s1_normal) + len(flat_idx_s1_zero):,} pixels")

    # ── Write only the two modified SOS TIFs; backup only those two ──
    modified_cols = ["s1_sos_doy", "s2_sos_doy"]
    tifs_to_write = {col: TIF_NAMES[col] for col in modified_cols}

    print(f"\n  Backing up modified SOS TIFs to {orig_dir} …")
    for col in modified_cols:
        _backup(folder / TIF_NAMES[col], orig_dir)

    print("\n  Writing updated TIFs …")
    for col, tif_name in tifs_to_write.items():
        tif_path = folder / tif_name
        profile  = profiles[col]
        # Ensure dtype is int16 (matches originals)
        profile.update(dtype=rasterio.int16)
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(arrays[col], 1)
        print(f" Written → {tif_path}")

    print(f"\n  Points clamped: {n_changed:,}")
    print(f"    └ S1_SOS pixels updated (S2 ends later): {s2_anchored.sum():>6,}")
    print(f"    └ S2_SOS pixels updated (S1 ends later): {s1_anchored.sum():>6,}")
    print(f"    └ annual_sos=0 guard (→ DOY=1):          {s2_zero.sum()+s1_zero.sum():>6,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clamp S1∪S2 season union to ≤ 365 days. "
            "Operates on seasonality_lookup.parquet and/or the six crop-calendar TIFs."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing seasonality_lookup.parquet and the *_WGS84.tif files.",
    )
    parser.add_argument(
        "--target",
        choices=["both", "parquet", "tifs"],
        default="both",
        help=(
            "Which files to clamp: 'parquet' (only the .parquet), "
            "'tifs' (only the 6 GeoTIFs), or 'both' (default)."
        ),
    )
    args = parser.parse_args()

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory.")
        sys.exit(1)

    if args.target in ("both", "parquet"):
        parquet_path = folder / PARQUET_NAME
        if not parquet_path.exists():
            print(f"ERROR: {parquet_path} not found.")
            sys.exit(1)
        clamp_parquet(folder)

    if args.target in ("both", "tifs"):
        missing = [
            name for name in TIF_NAMES.values()
            if not (folder / name).exists()
        ]
        if missing:
            print(f"ERROR: Missing TIF files in {folder}:")
            for m in missing:
                print(f"  {m}")
            sys.exit(1)
        clamp_tifs(folder)

    print("\nDone.")


if __name__ == "__main__":
    main()
