#!/usr/bin/env python
"""Check a dilated/raw_scl patch pair is complete and genuinely controlled.

Run this before shipping the trees anywhere: it is the difference between
"the files exist" and "these two trees are a valid comparison".

Checks, per pair:
  1. Both trees hold exactly the same patch files (same regions, same names).
  2. Every patch carries the full 16-band model input set, nothing extra.
  3. Grids match (t/y/x, coords, CRS) between the two conventions.
  4. Only the 10 S2 bands differ; S1, slope, elevation and AGERA5 are
     byte-identical — i.e. the S2 masking is the ONLY moving part.
  5. raw_scl has no more S2 NODATA than dilated (it is strictly less
     aggressive, so it must let at least as much data through).
  6. Every patch window can supply the model's timestep count.

    python verify_patch_pair.py --base <BASE> [--expected-timesteps 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

S2_BANDS = [
    "S2-L2A-B02", "S2-L2A-B03", "S2-L2A-B04", "S2-L2A-B05", "S2-L2A-B06",
    "S2-L2A-B07", "S2-L2A-B08", "S2-L2A-B8A", "S2-L2A-B11", "S2-L2A-B12",
]
NON_S2 = ["S1-SIGMA0-VH", "S1-SIGMA0-VV", "slope", "elevation",
          "AGERA5-PRECIP", "AGERA5-TMEAN"]
REQUIRED = set(S2_BANDS) | set(NON_S2)
NODATA = 65535


def rel_paths(root: Path) -> set:
    return {p.relative_to(root).as_posix() for p in root.rglob("*.nc")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--dilated-subdir", default="PATCHES_DILATED")
    ap.add_argument("--rawscl-subdir", default="PATCHES_RAWSCL")
    ap.add_argument("--expected-timesteps", type=int, default=12,
                    help="Timestep count the model requires (monthly: 12).")
    args = ap.parse_args()

    dil = args.base / args.dilated_subdir
    raw = args.base / args.rawscl_subdir
    for d in (dil, raw):
        if not d.is_dir():
            logger.error(f"Missing tree: {d}")
            return 1

    a, b = rel_paths(dil), rel_paths(raw)
    problems = 0
    if a != b:
        logger.error(
            f"Trees differ: {len(a - b)} only in dilated, {len(b - a)} only in raw_scl"
        )
        for p in sorted(a ^ b)[:10]:
            logger.error(f"  unpaired: {p}")
        problems += 1
    common = sorted(a & b)
    logger.info(f"{len(common)} paired patches across "
                f"{len({p.split('/')[0] for p in common})} regions")

    n_short = 0
    s2_dil_tot = s2_raw_tot = cells = 0
    for rel in common:
        da = xr.open_dataset(dil / rel, mask_and_scale=False)
        db = xr.open_dataset(raw / rel, mask_and_scale=False)
        tag = rel

        have = {v for v in da.data_vars if v != "crs"}
        if have != REQUIRED:
            logger.error(f"{tag}: band set wrong "
                         f"(missing={sorted(REQUIRED - have)}, extra={sorted(have - REQUIRED)})")
            problems += 1

        if da.sizes != db.sizes:
            logger.error(f"{tag}: grid mismatch {dict(da.sizes)} vs {dict(db.sizes)}")
            problems += 1
        elif not (np.array_equal(da.x.values, db.x.values)
                  and np.array_equal(da.y.values, db.y.values)
                  and np.array_equal(da.t.values, db.t.values)):
            logger.error(f"{tag}: coordinate mismatch between conventions")
            problems += 1

        for v in NON_S2:
            if v in da and v in db and not np.array_equal(da[v].values, db[v].values):
                logger.error(f"{tag}: {v} differs between conventions "
                             f"(masking must not touch it)")
                problems += 1

        # A band whose source directory is unreachable composites to NODATA
        # everywhere, which every cross-convention check above still passes.
        for v in sorted(have):
            if int((da[v].values.view(np.uint16) != NODATA).sum()) == 0:
                logger.error(f"{tag}: {v} is NODATA at every cell "
                             f"(source archive not readable at build time?)")
                problems += 1

        if all(v in da and v in db for v in S2_BANDS):
            nd_a = sum(int((da[v].values.view(np.uint16) == NODATA).sum()) for v in S2_BANDS)
            nd_b = sum(int((db[v].values.view(np.uint16) == NODATA).sum()) for v in S2_BANDS)
            s2_dil_tot += nd_a
            s2_raw_tot += nd_b
            cells += sum(da[v].size for v in S2_BANDS)
            if nd_b > nd_a:
                logger.error(f"{tag}: raw_scl has MORE S2 nodata than dilated "
                             f"({nd_b:,} > {nd_a:,}) — masking is inverted?")
                problems += 1
            if nd_a == nd_b and da[S2_BANDS[0]].equals(db[S2_BANDS[0]]):
                logger.warning(f"{tag}: conventions produced identical S2 "
                               f"(no cloud flagged in this patch)")

        if da.sizes.get("t", 0) < args.expected_timesteps:
            logger.error(f"{tag}: only {da.sizes.get('t')} timesteps, model needs "
                         f"{args.expected_timesteps}")
            n_short += 1
            problems += 1

        da.close()
        db.close()

    if cells:
        logger.info(
            f"S2 NODATA overall: dilated {100*s2_dil_tot/cells:.1f}% -> "
            f"raw_scl {100*s2_raw_tot/cells:.1f}% "
            f"({100*(s2_dil_tot-s2_raw_tot)/max(s2_dil_tot,1):.1f}% of masked data recovered)"
        )
    if n_short:
        logger.error(f"{n_short} patch(es) too short for inference")

    if problems:
        logger.error(f"FAILED with {problems} problem(s)")
        return 2
    logger.success("All checks passed — trees are a valid controlled pair, "
                   "complete, and inference-ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
