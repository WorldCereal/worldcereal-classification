#!/usr/bin/env python
"""Pick a small, region-spread set of extraction patches for spatial inference.

Chooses N patches from the global patch index, allocated across UN subregions
(Europe weighted higher, since that is where the current model suites are
evaluated) and spatially spread *within* each region so a region's patches are
not all neighbours from one reference dataset.

The output is a manifest that `ptp_patch_composite.py --manifest` consumes, so
the same patch set can be composited under BOTH masking conventions — which is
the whole point: a dilated/raw_scl comparison is only meaningful if the two
runs cover identical ground.

    python select_test_patches.py --out-manifest patches_manifest.parquet

Selection is deterministic for a given --seed, --total and allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

PATCH_INDEX = Path(
    "/vitodata/worldcereal/data/PATCH_TO_POINT_LOCAL/global_patches_index/"
    "patches_index_extended.geoparquet"
)
WORLD_BOUNDARIES = (
    Path(__file__).resolve().parents[2]
    / "src/worldcereal/data/world-administrative-boundaries"
    / "world-administrative-boundaries.geoparquet"
)

# Regions get a share of the total proportional to how much data they hold,
# floored at --min-per-region so every region is represented and capped at
# --max-per-region so one data-rich region cannot swamp the set. Europe is
# boosted because that is where the model suites are evaluated.
EUROPE_BOOST = 2.0
EUROPE_REGIONS = frozenset(
    {"Western Europe", "Northern Europe", "Eastern Europe", "Southern Europe"}
)


def region_seed(seed: int, region: str) -> int:
    """Stable per-region seed.

    Python's `hash()` on str is salted per process, so it cannot be used for
    anything that must reproduce across runs. Hash the region name explicitly.
    """
    h = hashlib.sha256(f"{seed}:{region}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def allocate(
    available: Dict[str, int], total: int, lo: int, hi: int
) -> Dict[str, int]:
    """Split `total` patches across regions by data volume, clamped to [lo, hi].

    sqrt damps the spread: Northern America holds ~500x the patches of Western
    Asia, and a linear share would give the small regions nothing. Largest
    remainder makes the parts sum to exactly `total`.
    """
    regions = sorted(available)
    weights = {
        r: math.sqrt(available[r]) * (EUROPE_BOOST if r in EUROPE_REGIONS else 1.0)
        for r in regions
    }

    alloc = {r: lo for r in regions}
    remaining = total - lo * len(regions)
    if remaining < 0:
        raise ValueError(
            f"--total {total} cannot cover {len(regions)} regions at "
            f"--min-per-region {lo} (needs >= {lo * len(regions)})"
        )

    # Distribute the surplus by weight, then hand out the rounding remainder to
    # the largest fractional parts. Regions at `hi` drop out and their share is
    # redistributed, so the total is always met while the cap holds.
    while remaining > 0:
        open_regions = [r for r in regions if alloc[r] < hi and available[r] > alloc[r]]
        if not open_regions:
            break
        wsum = sum(weights[r] for r in open_regions)
        exact = {r: remaining * weights[r] / wsum for r in open_regions}
        add = {r: int(exact[r]) for r in open_regions}
        left = remaining - sum(add.values())
        for r in sorted(open_regions, key=lambda r: (-(exact[r] - add[r]), r))[:left]:
            add[r] += 1
        progressed = False
        for r in open_regions:
            room = min(hi - alloc[r], available[r] - alloc[r])
            take = min(add[r], room)
            if take > 0:
                alloc[r] += take
                remaining -= take
                progressed = True
        if not progressed:
            break

    return {r: n for r, n in alloc.items() if n > 0}


def _region_label(region: str) -> str:
    """'Western Europe' -> 'Western_Europe', the folder name the inference
    script's --continents/REGIONS list uses."""
    return str(region).replace(" ", "_")


def spread_within_region(
    df: pd.DataFrame, n: int, seed: int = 0
) -> pd.DataFrame:
    """Greedy farthest-point selection, preferring unused ref_ids first.

    Picking the n patches that are simply first in the index would cluster them
    inside one reference dataset. This walks the candidates keeping whichever is
    farthest from everything already chosen, and only revisits a ref_id once
    every ref_id in the region has contributed.
    """
    if len(df) <= n:
        return df

    rng = np.random.default_rng(seed)
    # Cap the candidate pool: farthest-point is O(n * pool) and a region can
    # hold 250k patches. A few thousand is far more than enough to spread over.
    pool = df.sample(n=min(len(df), 4000), random_state=seed % (2**32)).reset_index(drop=True)

    lon = pool["lon"].to_numpy()
    lat = pool["lat"].to_numpy()
    chosen: List[int] = [int(rng.integers(len(pool)))]
    # Great-circle-ish distance is unnecessary here; squared degrees ranks the
    # same way at these scales and keeps the loop cheap.
    mind = (lon - lon[chosen[0]]) ** 2 + (lat - lat[chosen[0]]) ** 2

    while len(chosen) < n:
        used_refs = {pool.loc[i, "ref_id"] for i in chosen}
        fresh = ~pool["ref_id"].isin(used_refs).to_numpy()
        # Prefer an unused ref_id while any remain; otherwise fall back to the
        # full pool so we always reach n.
        score = np.where(fresh, mind, -1.0) if fresh.any() else mind.copy()
        score[chosen] = -1.0
        nxt = int(np.argmax(score))
        if score[nxt] < 0:
            break
        chosen.append(nxt)
        mind = np.minimum(mind, (lon - lon[nxt]) ** 2 + (lat - lat[nxt]) ** 2)

    return pool.loc[chosen]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-manifest", required=True, type=Path)
    ap.add_argument("--patch-index", type=Path, default=PATCH_INDEX)
    ap.add_argument("--boundaries", type=Path, default=WORLD_BOUNDARIES)
    ap.add_argument("--total", type=int, default=100,
                    help="Total patches to select (default: 100).")
    ap.add_argument("--min-per-region", type=int, default=3,
                    help="Floor per region, so every region is represented.")
    ap.add_argument("--max-per-region", type=int, default=10,
                    help="Cap per region, so data-rich regions cannot swamp it.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Change this to draw a different set of the same size.")
    ap.add_argument("--require-s1", action="store_true", default=True,
                    help="Only pick patches that have at least one S1 orbit.")
    args = ap.parse_args()

    logger.info(f"Reading patch index: {args.patch_index}")
    idx = gpd.read_parquet(
        args.patch_index,
        columns=["ref_id", "sample_id", "tile", "zone", "s2_path", "s1_asc",
                 "s1_desc", "start_date", "end_date", "geometry"],
    )
    logger.info(f"{len(idx):,} patches in index")

    # Patch footprints are ~640 m boxes; their centroid in degrees is fine for
    # a point-in-polygon region lookup and for spreading.
    pts = idx.copy()
    pts["lon"] = idx.geometry.centroid.x
    pts["lat"] = idx.geometry.centroid.y
    pts = pts.set_geometry(gpd.points_from_xy(pts["lon"], pts["lat"], crs=idx.crs))

    wb = gpd.read_parquet(
        args.boundaries, columns=["name", "region", "continent", "geometry"]
    ).to_crs(pts.crs)

    logger.info("Joining patches to regions")
    j = gpd.sjoin(pts, wb, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    j = j[j["region"].notna()]
    logger.info(f"{len(j):,} patches matched to a region")

    # ~1.5% of index rows are S1-only (no S2 patch on disk). Nothing can be
    # composited from those, so drop them before allocating, otherwise a
    # region's share is spent on patches that cannot be produced.
    has_s2 = j["s2_path"].apply(lambda v: isinstance(v, str) and bool(v))
    logger.info(f"Dropping {int((~has_s2).sum()):,} patches with no S2 path")
    j = j[has_s2]

    if args.require_s1:
        has_s1 = j["s1_asc"].notna() | j["s1_desc"].notna()
        logger.info(f"Dropping {int((~has_s1).sum()):,} patches with no S1 orbit")
        j = j[has_s1]

    available = j.groupby("region").size().to_dict()
    allocation = allocate(
        available, args.total, args.min_per_region, args.max_per_region
    )
    logger.info(
        f"Allocation (total={sum(allocation.values())}, seed={args.seed}): "
        + ", ".join(f"{r}={n}" for r, n in sorted(allocation.items()))
    )

    rows = []
    for region, n in sorted(allocation.items()):
        cand = j[j["region"] == region]
        if cand.empty:
            logger.warning(f"{region}: no patches available, skipping")
            continue
        take = spread_within_region(
            pd.DataFrame(cand.drop(columns="geometry")),
            n,
            seed=region_seed(args.seed, region),
        )
        if len(take) < n:
            logger.warning(f"{region}: wanted {n}, got {len(take)}")
        take = take.copy()
        take["region_folder"] = _region_label(region)
        rows.append(take)
        logger.info(
            f"{region:20s}: {len(take)} patches from {take['ref_id'].nunique()} ref(s), "
            f"lon {take['lon'].min():.1f}..{take['lon'].max():.1f} "
            f"lat {take['lat'].min():.1f}..{take['lat'].max():.1f}"
        )

    out = pd.concat(rows, ignore_index=True)
    keep = ["region_folder", "region", "continent", "name", "ref_id", "sample_id",
            "tile", "zone", "s2_path", "s1_asc", "s1_desc", "start_date",
            "end_date", "lon", "lat"]
    out = out[keep].sort_values(["region_folder", "sample_id"]).reset_index(drop=True)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_manifest, index=False)
    out.to_csv(args.out_manifest.with_suffix(".csv"), index=False)
    logger.success(
        f"Selected {len(out)} patches across {out['region_folder'].nunique()} "
        f"regions -> {args.out_manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
