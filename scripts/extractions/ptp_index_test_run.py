"""End-to-end test of the global-index-driven patch-to-point workflow.

For one ref: load patches_index.geoparquet -> slice to a RefCatalog ->
select_and_assign (RDM streaming + spatial join) -> extract_host with N
workers -> compare a sample of rows against an existing reference output
(e.g. the campaign's MERGED_PARQUETS) if one exists.

Usage:
  python ptp_index_test_run.py --index ~/ptp_index_test/patches_index.geoparquet \
      --ref-id 2021_SVK_Eurocrops_POLY_110 --sample-limit 200 --workers 4 \
      --out-dir ~/ptp_index_test/out \
      --reference /vitodata/worldcereal/data/PATCH_TO_POINT_LOCAL/MERGED_PARQUETS/2021_SVK_Eurocrops_POLY_110.geoparquet
"""

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptp_engine  # noqa: E402
from ptp_campaign_rdm import RDM_DIR_DEFAULT, select_and_assign  # noqa: E402
from ptp_engine import DEFAULT_CONVENTIONS, extract_host  # noqa: E402
from ptp_global_index import catalog_for_ref, read_index_meta  # noqa: E402

VALUE_COLS = [
    "S2-L2A-B02", "S2-L2A-B03", "S2-L2A-B04", "S2-L2A-B05", "S2-L2A-B06",
    "S2-L2A-B07", "S2-L2A-B08", "S2-L2A-B8A", "S2-L2A-B11", "S2-L2A-B12",
    "S1-SIGMA0-VH", "S1-SIGMA0-VV", "slope", "elevation",
    "AGERA5-PRECIP", "AGERA5-TMEAN",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--ref-id", required=True)
    ap.add_argument("--rdm-dir", type=Path, default=RDM_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sample-limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reference", type=Path, default=None,
                    help="existing output geoparquet to compare against")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ptp_engine.AGERA5_CACHE = args.out_dir / "_agera5_cache"

    t0 = time.time()
    index = gpd.read_parquet(args.index)
    meta = read_index_meta(args.index)
    logger.info(f"Index: {len(index):,} patches / {index.ref_id.nunique()} "
                f"refs, built {meta.get('built_at')} "
                f"(loaded in {time.time() - t0:.1f}s)")

    t1 = time.time()
    catalog = catalog_for_ref(index, args.ref_id)
    logger.info(f"{args.ref_id}: {len(catalog.entries)} patches sliced from "
                f"index in {time.time() - t1:.2f}s")

    t2 = time.time()
    points, stats = select_and_assign(
        args.ref_id, catalog, args.rdm_dir, sample_limit=args.sample_limit)
    logger.info(f"select+assign: {len(points)} points in "
                f"{time.time() - t2:.1f}s | {stats}")
    if points.empty:
        logger.error("no points selected — nothing to test")
        return

    t3 = time.time()
    out_path = args.out_dir / f"{args.ref_id}_INDEXTEST.geoparquet"
    if out_path.exists():
        out_path.unlink()
    df, _ = extract_host(
        args.ref_id, DEFAULT_CONVENTIONS, workers=args.workers,
        out_path=out_path, points=points, index=catalog.entries)
    dt = time.time() - t3
    n = df.sample_id.nunique()
    logger.success(f"extract: {n} samples / {len(df):,} rows in {dt:.1f}s "
                   f"({n / dt:.1f} samples/s with {args.workers} workers)")

    if args.reference and args.reference.exists():
        ref = pd.read_parquet(args.reference,
                              columns=["sample_id", "timestamp"] + VALUE_COLS)
        ours = df[["sample_id", "timestamp"] + VALUE_COLS].copy()
        merged = ours.merge(ref, on=["sample_id", "timestamp"],
                            suffixes=("_new", "_ref"))
        if merged.empty:
            logger.warning("no overlapping (sample_id, timestamp) rows with "
                           "the reference — cannot compare")
            return
        logger.info(f"comparing {len(merged):,} overlapping rows "
                    f"({merged.sample_id.nunique()} samples) vs "
                    f"{args.reference.name}")
        bad = 0
        for c in VALUE_COLS:
            a = merged[f"{c}_new"].to_numpy()
            b = merged[f"{c}_ref"].to_numpy()
            neq = int((a != b).sum())
            if neq:
                bad += neq
                ex = np.flatnonzero(a != b)[:3]
                logger.warning(f"  {c}: {neq}/{len(merged)} rows differ "
                               f"(e.g. new={a[ex].tolist()} "
                               f"ref={b[ex].tolist()})")
        if bad == 0:
            logger.success("VALUE CHECK: bit-identical to the reference "
                           "output on all compared rows/columns")
        else:
            logger.error(f"VALUE CHECK: {bad} differing cells — investigate "
                         "before using the index route")


if __name__ == "__main__":
    main()
