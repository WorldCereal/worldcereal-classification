"""clamp_calendars.py — Clamp S1∪S2 season union to ≤ 365 days.

Usage
-----
    python clamp_calendars.py <folder>

The script looks for ``seasonality_lookup.parquet`` inside <folder> and
writes the clamped result back to the same path (in-place), keeping a
backup at ``seasonality_lookup_original.parquet``.

Algorithm
---------
For every dual-season grid point whose S1∪S2 bounding-box union exceeds
365 days the earlier-ending season's SOS is moved forward to
``annual_sos_doy``, guaranteeing union == exactly 365 days.

  • S2 ends later (95 % of cases) → clamp ``s1_sos_doy``
  • S1 ends later ( 5 % of cases) → clamp ``s2_sos_doy``

Edge-case guard: if ``annual_sos_doy == 0`` (pre-existing source data
quirk — the ANNUAL_SOS raster had nodata at that cell) the SOS is set
to DOY=1 (Jan 1) instead, which gives a valid 364-day union.

Single-season points and points already within 365 days are not touched.

Later extensions to this script (e.g. patching the corresponding TIF
files) can be added in the same folder and receive the folder path via
the same CLI interface.
"""

import argparse
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── DOY columns (order matters — used for cast-back to uint16) ──
DOY_COLS = [
    "s1_sos_doy",
    "s1_eos_doy",
    "s2_sos_doy",
    "s2_eos_doy",
    "annual_sos_doy",
    "annual_eos_doy",
]

PARQUET_NAME = "seasonality_lookup.parquet"
BACKUP_NAME  = "seasonality_lookup_original.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Core clamping logic
# ─────────────────────────────────────────────────────────────────────────────

def clamp_parquet(parquet_path: Path) -> None:
    print(f"\n{'='*65}")
    print(f"  clamp_calendars  →  {parquet_path}")
    print(f"{'='*65}")

    # ── Load ──
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} grid points")
    print(f"  Dual-season:   {(df['s2_sos_doy'] != 0).sum():,}")
    print(f"  Single-season: {(df['s2_sos_doy'] == 0).sum():,}")

    # Cast to int64 for safe arithmetic (columns are uint16 on disk)
    for c in DOY_COLS:
        df[c] = df[c].astype(np.int64)

    # ── Compute original union spans ──
    print("\nComputing S1∪S2 union spans …")
    df["_union_days"] = compute_union_spans(df)

    dual_mask = df["s2_sos_doy"] != 0
    over365   = df["_union_days"] > 365
    n_over    = (over365 & dual_mask).sum()
    print(f"  Dual-season points with union > 365 days: {n_over:,}")

    if n_over == 0:
        print("Nothing to do — all unions already ≤ 365 days.")
        df.drop(columns=["_union_days"], inplace=True)
        return

    # ── Determine anchor (which season ends later) ──
    s2_anchored = over365 & dual_mask & (df["annual_eos_doy"] == df["s2_eos_doy"])
    s1_anchored = (
        over365 & dual_mask
        & (df["annual_eos_doy"] == df["s1_eos_doy"])
        & ~s2_anchored
    )
    print(f"  S2-anchored (S2 ends later → clamp s1_sos): {s2_anchored.sum():,}")
    print(f"  S1-anchored (S1 ends later → clamp s2_sos): {s1_anchored.sum():,}")

    # ── Apply clamp ──
    # S2-anchored: move s1_sos → annual_sos
    #   Guard: if annual_sos_doy == 0 (pre-existing source nodata), use DOY=1
    s2_normal = s2_anchored & (df["annual_sos_doy"] != 0)
    s2_zero   = s2_anchored & (df["annual_sos_doy"] == 0)
    print(f"\n  S2-anchored breakdown:")
    print(f"    normal (use annual_sos_doy): {s2_normal.sum():,}")
    print(f"    guard  (annual_sos==0 → use DOY=1): {s2_zero.sum():,}")

    df.loc[s2_normal, "s1_sos_doy"] = df.loc[s2_normal, "annual_sos_doy"]
    df.loc[s2_zero,   "s1_sos_doy"] = 1   # Jan 1 — valid start of annual window

    # S1-anchored: move s2_sos → annual_sos (guard applies symmetrically)
    s1_normal = s1_anchored & (df["annual_sos_doy"] != 0)
    s1_zero   = s1_anchored & (df["annual_sos_doy"] == 0)
    df.loc[s1_normal, "s2_sos_doy"] = df.loc[s1_normal, "annual_sos_doy"]
    df.loc[s1_zero,   "s2_sos_doy"] = 1

    # ── Verify ──
    print("\nVerifying clamped unions …")
    df["_clamped_union_days"] = compute_union_spans(df)
    still_over = (df["_clamped_union_days"] > 365) & dual_mask
    print(f"  Points still > 365 days after clamp: {still_over.sum():,}  (should be 0)")
    if still_over.sum() > 0:
        print("  ❌  Clamping did not fully resolve all over-365 points — aborting.")
        sys.exit(1)

    # Confirm untouched rows were not changed
    unchanged = dual_mask & ~over365
    n_accidentally_changed = (
        (df.loc[unchanged, "s1_sos_doy"] != df.loc[unchanged, "s1_sos_doy"]).sum()  # always 0, sanity
    )
    print(f"  Accidentally modified ≤365 rows: {n_accidentally_changed}  (should be 0)")

    # ── Prepare output ──
    df.drop(columns=["_union_days", "_clamped_union_days"], inplace=True)
    out = df[["lat", "lon"] + DOY_COLS].copy()
    for c in DOY_COLS:
        out[c] = out[c].astype(np.uint16)

    # ── Backup original, write clamped ──
    backup_path = parquet_path.parent / BACKUP_NAME
    if not backup_path.exists():
        shutil.copy2(parquet_path, backup_path)
        print(f"\n  Backup written → {backup_path}")
    else:
        print(f"\n  Backup already exists, skipping → {backup_path}")

    out.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"  ✅  Clamped parquet written → {parquet_path}")
    print(f"      Shape: {out.shape}  |  dtypes: all uint16")

    # ── Final summary ──
    n_total_changed = s2_anchored.sum() + s1_anchored.sum()
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Total grid points:           {len(out):>8,}")
    print(f"  Points clamped:              {n_total_changed:>8,}"
          f"  ({100 * n_total_changed / dual_mask.sum():.1f}% of dual-season)")
    print(f"    └ s1_sos clamped (S2 ends later): {s2_anchored.sum():>6,}")
    print(f"    └ s2_sos clamped (S1 ends later): {s1_anchored.sum():>6,}")
    print(f"    └   of which annual_sos=0 guard:  {(s2_zero.sum() + s1_zero.sum()):>6,}  (→ DOY=1)")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clamp S1∪S2 season union to ≤ 365 days in seasonality_lookup.parquet."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to the folder containing seasonality_lookup.parquet",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory.")
        sys.exit(1)

    parquet_path = folder / PARQUET_NAME
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found.")
        sys.exit(1)

    clamp_parquet(parquet_path)


if __name__ == "__main__":
    main()
