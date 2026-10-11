#!/usr/bin/env python
"""Check the engine's windowed extraction against its scalar production code.

Three checks on real patches:

  S2 chip     composite_s2_chip sliced per point, against composite_s2 called
              per point. Both are the 10 m grid, so this is a pure
              same-grid-same-values check.
  S1 grid     the S1 window process_patch produces, against an independent
              reconstruction: for every S2 cell of the window, the SCALAR
              composite_s1 evaluated at the S1 pixel covering that cell's
              centre. S1 is 20 m where S2 is 10 m, so this is what keeps the
              bands co-registered, and the reference uses only production code.
  repeat      how many distinct S1 pixels a window touches. With 20 m S1 and a
              10 m window a 5x5 should touch about 9, not 25; 25 would mean the
              window was taken as a block of S1 pixels and covers four times
              the ground of the S2 window.

Usage:
    python verify_window_grid.py --ref 2019_USA_USDA-CDL_POINT_110 \
        --patches 3 --points 12 --half 2
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import ptp_engine as E
from pyproj import CRS, Transformer

warnings.filterwarnings("ignore")

INDEX = Path("/vitodata/worldcereal/data/test_spark_runs/"
             "patches_index_extended.geoparquet")
# The index stores cluster paths; this host mounts the same export elsewhere.
PATH_FIX = ("/data/worldcereal_data/", "/vitodata/worldcereal_data/")


def _local(p: str) -> str:
    return p.replace(*PATH_FIX) if p else p


def _periods(start: str, end: str, freq: str = "month"):
    """Period starts: day 1 monthly, days 1/11/21 for dekad."""
    t = pd.Timestamp(start).normalize().replace(day=1)
    stop = pd.Timestamp(end)
    days = (1,) if freq == "month" else (1, 11, 21)
    out = []
    while t < stop:
        out += [(t.year, t.month, d) for d in days]
        t += pd.DateOffset(months=1)
    return out


def _rowcols(patch: dict, n: int, seed: int = 1):
    """n (row, col) pairs over the chip, with corners for the NODATA padding."""
    rng = np.random.default_rng(seed)
    ny, nx = len(patch["y"]), len(patch["x"])
    rows = list(rng.integers(0, ny, max(n - 4, 0)))
    cols = list(rng.integers(0, nx, max(n - 4, 0)))
    rows += [0, 0, ny - 1, ny - 1][:min(n, 4)]
    cols += [0, nx - 1, 0, nx - 1][:min(n, 4)]
    return np.array(rows[:n]), np.array(cols[:n])


def _ref_s1_window(s2_patch, s1p, tf, off, row, col, half, periods, t0, t1,
                   freq):
    """The S1 window rebuilt from scalar composite_s1, cell by cell.

    Deliberately re-derives the S1 cell itself instead of calling
    _s1_window_idx, so a mistake there cannot pass unnoticed.
    """
    w = 2 * half + 1
    ny, nx = len(s2_patch["y"]), len(s2_patch["x"])
    out = np.full((2, len(periods), w, w), E.NODATA, np.uint16)
    touched = set()
    for i, r in enumerate(range(row - half, row + half + 1)):
        for j, c in enumerate(range(col - half, col + half + 1)):
            if not (0 <= r < ny and 0 <= c < nx):
                continue
            qx, qy = tf.transform(float(s2_patch["x"][c]),
                                  float(s2_patch["y"][r]))
            a = int(np.argmin(np.abs(s1p["y"] - qy))) + off["row_off"]
            b = int(np.argmin(np.abs(s1p["x"] - qx))) + off["col_off"]
            if not (0 <= a < len(s1p["y"]) and 0 <= b < len(s1p["x"])):
                continue
            touched.add((a, b))
            out[:, :, i, j] = E.composite_s1(s1p, a, b, periods, t0, t1, freq)
    return out, touched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="2019_USA_USDA-CDL_POINT_110")
    ap.add_argument("--patches", type=int, default=3)
    ap.add_argument("--points", type=int, default=12)
    ap.add_argument("--half", type=int, default=2)
    ap.add_argument("--s2-mask", default="dilated",
                    choices=["dilated", "raw_scl"])
    ap.add_argument("--freq", default="month", choices=["month", "dekad"])
    args = ap.parse_args()
    if args.half < 1:
        print("--half must be 1 or more; the scalar path has no window")
        return 1

    idx = pd.read_parquet(INDEX, columns=["ref_id", "s2_path", "s1_asc",
                                          "s1_desc", "start_date", "end_date",
                                          "tile"])
    sub = idx[idx.ref_id == args.ref]
    if not len(sub):
        print(f"no patches for {args.ref}")
        return 1

    scl = E.SCL_RAW_BAND if args.s2_mask == "raw_scl" else E.SCL_DILATED_BAND
    bad_s2 = val_s2 = bad_s1 = val_s1 = 0
    centre_bad = 0
    touch = []
    t_point = t_chip = 0.0
    checked = 0

    for _, r in sub.iterrows():
        p = _local(r.s2_path)
        if not Path(p).exists():
            continue
        patch = E._read_patch(Path(p), E.S2_BANDS + [scl])
        periods = _periods(r.start_date, r.end_date, args.freq)
        t0 = np.datetime64(pd.Timestamp(r.start_date).date())
        t1 = np.datetime64(pd.Timestamp(r.end_date).date())
        rows, cols = _rowcols(patch, args.points)
        s2_crs = CRS.from_wkt(patch["crs_wkt"])

        # --- S2: chip against per-point, same grid ---
        t = time.perf_counter()
        ref_s2 = [E.composite_s2(patch, int(a), int(b), periods, t0, t1,
                                 s2_mask=args.s2_mask, freq=args.freq,
                                 half=args.half)
                  for a, b in zip(rows, cols)]
        t_point += time.perf_counter() - t
        t = time.perf_counter()
        chip2 = E.composite_s2_chip(patch, periods, t0, t1, args.s2_mask,
                                    args.freq)
        new_s2 = [E._slice_chip(chip2, int(a), int(b), args.half)
                  for a, b in zip(rows, cols)]
        t_chip += time.perf_counter() - t
        for a, b in zip(ref_s2, new_s2):
            val_s2 += a.size
            bad_s2 += int((a != b).sum())

        # --- S1: engine window against the scalar reconstruction ---
        s1p = tf = case = None
        for orbit, cand in (("DESCENDING", r.s1_desc), ("ASCENDING", r.s1_asc)):
            if cand and Path(_local(cand)).exists():
                s1p = E._read_patch(Path(_local(cand)), E.S1_BANDS)
                s1_crs = CRS.from_wkt(s1p["crs_wkt"])
                case = ("s1_same_crs" if s1_crs.equals(s2_crs)
                        else "s1_cross_crs")
                tf = Transformer.from_crs(s2_crs, s1_crs, always_xy=True)
                break
        if s1p is not None:
            off = E.DEFAULT_CONVENTIONS[case]
            chip1 = E.composite_s1_chip(s1p, periods, t0, t1, args.freq)
            for a, b in zip(rows, cols):
                ir, ic = E._s1_window_idx(patch, s1p, tf, off, int(a), int(b),
                                          args.half)
                got = E._gather_s1_window(chip1, ir, ic)
                want, touched = _ref_s1_window(
                    patch, s1p, tf, off, int(a), int(b), args.half, periods,
                    t0, t1, args.freq)
                val_s1 += want.size
                bad_s1 += int((got != want).sum())
                touch.append(len(touched))
                # the centre cell must be the production single-pixel value
                h = args.half
                if ir[h, h] >= 0:
                    scalar = E.composite_s1(s1p, int(ir[h, h]), int(ic[h, h]),
                                            periods, t0, t1, args.freq)
                    centre_bad += int((got[:, :, h, h] != scalar).sum())

        checked += 1
        if checked >= args.patches:
            break

    if not checked:
        print("no readable patches on this host")
        return 1

    w = 2 * args.half + 1
    print(f"{args.ref}  {checked} patch(es) x {args.points} points, "
          f"N={args.half} ({w}x{w}), mask={args.s2_mask}, freq={args.freq}")
    print(f"  S2 chip   {val_s2:12,d} values  {bad_s2:,} mismatch   "
          f"per-point {t_point:5.2f}s  chip {t_chip:5.2f}s  "
          f"({t_point / max(t_chip, 1e-9):.1f}x)")
    print(f"  S1 grid   {val_s1:12,d} values  {bad_s1:,} mismatch   "
          f"centre vs scalar composite_s1: {centre_bad:,} mismatch")
    if touch:
        print(f"  S1 pixels touched per {w}x{w} window: "
              f"min {min(touch)}, median {int(np.median(touch))}, "
              f"max {max(touch)}  (a block of S1 pixels would be {w * w})")
    ok = bad_s2 == 0 and bad_s1 == 0 and centre_bad == 0
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
