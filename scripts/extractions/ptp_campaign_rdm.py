"""Campaign driver: full-RDM reprocessing via the local patch-to-point engine.

For each ref_id, extracts the RDM samples from that ref's OWN patch
extractions — the standard patch-to-point flow, minus openEO.

Per ref, three steps:

  1. SELECT — stream the harmonized RDM geoparquet
     (<rdm-dir>/<ref>/harmonized/<ref>.geoparquet) row-group by row-group,
     keeping samples that (a) fall inside any patch footprint of the ref and
     (b) have valid_time inside the ref's patch window. Polygon samples are
     reduced to their centroid (EPSG:3857 centroid, dropped if outside the
     polygon) — the same `gdf_to_points` semantics the openEO flow used.
  2. ASSIGN — each sample is mapped to ONE patch:
       primary   : the sample's own patch (sample_id == patch id), when it
                   exists — the vast majority;
       collateral: otherwise, the covering patch whose centre is nearest to
                   the point (deterministic; openEO's mosaic pick here was
                   arbitrary and irreproducible).
  3. EXTRACT — ptp_engine.extract_host with `points=` supplying the
     assignment; output is one long-format geoparquet per ref, keyed by the
     ref's own ref_id. No rekey stage: unlike the in-patch campaign, samples
     already belong to the ref they are extracted from.

Patch discovery + footprints come from ref_catalog (STAC-primary, seconds
per ref; use --catalog-cache to persist).

Usage:
  ptp_campaign_rdm.py --mode assign  --ref-ids <ref> ...   # stats only, no extraction
  ptp_campaign_rdm.py --mode extract --ref-ids <ref> ... --out-dir DIR
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from shapely import wkb as shapely_wkb
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptp_engine  # noqa: E402
from ptp_engine import DEFAULT_CONVENTIONS, extract_host  # noqa: E402
from ref_catalog import RefCatalog  # noqa: E402

RDM_DIR_DEFAULT = Path("/vitodata/worldcereal/data/RDM")

# The attribute columns the engine's output assembly reads from the points
# frame (plus sample_id/geometry). Matches RDM_DEFAULT_COLUMNS minus ref_id.
RDM_ATTR_COLUMNS = [
    "sample_id", "ewoc_code", "valid_time", "irrigation_status",
    "quality_score_lc", "quality_score_ct", "extract", "h3_l3_cell",
    "geometry",
]


# --- Step 1+2: select and assign ------------------------------------------


def _centroid_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """gdf_to_points semantics: EPSG:3857 centroid, drop if outside its own
    polygon, return EPSG:4326 points. Point geometries pass through."""
    if gdf.empty:
        return gdf
    merc = gdf.to_crs(epsg=3857)
    centroids = merc.geometry.centroid
    inside = merc.geometry.geom_type.isin(("Point", "MultiPoint")) | \
        centroids.within(merc.geometry)
    dropped = int((~inside).sum())
    if dropped:
        logger.warning(f"dropped {dropped} sample(s) whose centroid lies "
                       "outside their own polygon")
    out = gdf.loc[inside].copy()
    out["geometry"] = gpd.GeoSeries(centroids[inside], crs=3857).to_crs(4326)
    return out


def select_and_assign(
    ref_id: str,
    catalog: RefCatalog,
    rdm_dir: Path,
    only_flagged: bool = False,
    sample_limit: Optional[int] = None,
) -> Tuple[gpd.GeoDataFrame, dict]:
    """Stream the ref's harmonized RDM file; return the engine-ready points
    frame (RDM attrs + host_sample_id) and selection statistics."""
    src = rdm_dir / ref_id / "harmonized" / f"{ref_id}.geoparquet"
    if not src.exists():
        raise FileNotFoundError(f"{ref_id}: no harmonized RDM file at {src}")

    tree, tree_sids = catalog.strtree()
    if not tree_sids:
        raise ValueError(f"{ref_id}: catalog has no footprints (built via "
                         "'fs'?). Use --index-source stac/auto.")
    # h3 prefilter set from catalog (STAC h3 property); empty set disables it.
    h3_cells = {e["h3"] for e in catalog.entries.values() if e.get("h3")}

    # Ref-wide valid_time window from patch filenames (…_<start>_<end>.nc).
    starts, ends = [], []
    for e in catalog.entries.values():
        if e.get("s2") is None:
            continue
        stem = e["s2"].stem.split("_")
        starts.append(stem[-2]); ends.append(stem[-1])
    t_lo, t_hi = min(starts), max(ends)

    stats = {"ref_id": ref_id, "rows_read": 0, "kept_h3": 0, "kept_time": 0,
             "kept_spatial": 0, "primary": 0, "collateral": 0,
             "multi_cover_resolved": 0, "centroid_dropped": 0}

    pf = pq.ParquetFile(src)
    read_cols = [c for c in RDM_ATTR_COLUMNS if c != "geometry"] + ["geometry"]
    chunks: List[gpd.GeoDataFrame] = []

    # Row-group skip on h3 statistics (cheap; same trick as the flow) —
    # then stream the surviving groups in bounded batches: harmonized files
    # can hold >1M rows in a single row group, and materializing that many
    # polygons at once (plus the 3857 reprojection copy) blows past the
    # ~4.3 GB per-process ceiling on the VMs.
    eligible_rgs = []
    for rg in range(pf.metadata.num_row_groups):
        keep = True
        if h3_cells:
            meta = pf.metadata.row_group(rg)
            for ci in range(meta.num_columns):
                col = meta.column(ci)
                if col.path_in_schema == "h3_l3_cell" and col.statistics \
                        and col.statistics.has_min_max:
                    lo, hi = col.statistics.min, col.statistics.max
                    keep = any(lo <= c <= hi for c in h3_cells)
                    break
        if keep:
            eligible_rgs.append(rg)

    for batch in pf.iter_batches(batch_size=50_000, row_groups=eligible_rgs,
                                 columns=read_cols):
        df = batch.to_pandas()
        stats["rows_read"] += len(df)

        if h3_cells and "h3_l3_cell" in df.columns:
            df = df[df["h3_l3_cell"].isin(h3_cells)]
        stats["kept_h3"] += len(df)
        if df.empty:
            continue

        vt = pd.to_datetime(df["valid_time"], errors="coerce")
        df = df[(vt >= t_lo) & (vt <= t_hi)]
        stats["kept_time"] += len(df)
        if only_flagged and "extract" in df.columns:
            df = df[df["extract"] > 0]
        if df.empty:
            continue

        df = df.copy()
        df["geometry"] = df["geometry"].apply(lambda b: shapely_wkb.loads(bytes(b)))
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        n_before = len(gdf)
        gdf = _centroid_points(gdf)
        stats["centroid_dropped"] += n_before - len(gdf)
        if gdf.empty:
            continue

        hits = tree.query(gdf.geometry.values, predicate="intersects")
        covered = sorted(set(hits[0]))
        gdf = gdf.iloc[covered]
        stats["kept_spatial"] += len(gdf)
        if not gdf.empty:
            chunks.append(gdf)

        if sample_limit and sum(len(c) for c in chunks) >= sample_limit:
            break

    if not chunks:
        return gpd.GeoDataFrame(columns=RDM_ATTR_COLUMNS + ["host_sample_id"],
                                geometry=[], crs="EPSG:4326"), stats
    points = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True),
                              crs="EPSG:4326")
    points = points.drop_duplicates(subset="sample_id", keep="first")
    if sample_limit:
        points = points.head(sample_limit)

    # --- assignment ---
    entries = catalog.entries
    centre_cache: Dict[str, Point] = {}

    def _assign(row) -> Optional[str]:
        sid = row.sample_id
        own = entries.get(sid)
        if own is not None and own.get("s2") is not None:
            stats["primary"] += 1
            return sid
        covering = catalog.covering(row.geometry)
        covering = [c for c in covering
                    if entries[c].get("s2") is not None]
        if not covering:
            return None
        if len(covering) > 1:
            stats["multi_cover_resolved"] += 1
            for c in covering:
                if c not in centre_cache:
                    centre_cache[c] = entries[c]["footprint"].centroid
            covering = sorted(
                covering,
                key=lambda c: (row.geometry.distance(centre_cache[c]), c))
        stats["collateral"] += 1
        return covering[0]

    points["host_sample_id"] = [
        _assign(r) for r in points.itertuples()]
    unassigned = points["host_sample_id"].isna()
    if unassigned.any():
        logger.warning(f"{ref_id}: {int(unassigned.sum())} sample(s) inside a "
                       "footprint whose patch lacks an S2 file; dropped")
        points = points[~unassigned]
    return points, stats


# --- Step 3: extract -------------------------------------------------------


def run_ref(
    ref_id: str,
    args,
    conventions: dict,
) -> Optional[dict]:
    out_path = (Path(args.out_dir) / f"{ref_id}.geoparquet"
                if args.out_dir else None)
    if args.mode == "extract" and out_path.exists():
        logger.warning(f"{ref_id}: output exists, skipping ({out_path})")
        return None

    catalog = RefCatalog.load(
        ref_id, source=args.index_source,
        cache_dir=Path(args.catalog_cache) if args.catalog_cache else None)
    if not catalog.entries:
        logger.error(f"{ref_id}: no patches found ({catalog.source}); skipping")
        return None

    points, stats = select_and_assign(
        ref_id, catalog, Path(args.rdm_dir),
        only_flagged=args.only_flagged, sample_limit=args.sample_limit)
    logger.info(f"{ref_id}: {stats}")
    if args.mode == "assign" or points.empty:
        return stats

    extract_host(
        ref_id, conventions,
        workers=args.workers,
        out_path=out_path,
        points=points,
        index_source=args.index_source,
        catalog_cache=Path(args.catalog_cache) if args.catalog_cache else None,
    )
    stats["out_path"] = str(out_path)

    if args.verify or args.verify_pct:
        from ptp_verify import verify_ref
        n = args.verify
        if args.verify_pct:
            n = max(10, min(200, int(stats["primary"] * args.verify_pct / 100)))
        v = verify_ref(ref_id, out_path, catalog,
                       store=Path(args.verify_store), n_samples=n,
                       max_divergence_frac=args.max_divergence_frac)
        stats["verify_status"] = v["status"]
        stats["verify"] = {k: val for k, val in v.items()
                           if k not in ("ref_id",)}
        (logger.success if v["status"] == "PASS" else
         logger.warning if v["status"].startswith("SKIP") else
         logger.error)(f"{ref_id}: verification {v['status']}")
        vdir = Path(args.out_dir) / "_verify"
        vdir.mkdir(exist_ok=True)
        import json as _json
        (vdir / f"{ref_id}.json").write_text(_json.dumps(v, indent=2))
        if v["status"] == "FAIL":
            raise RuntimeError(f"verification FAILED for {ref_id} "
                               f"(unexplained differences vs openEO store; "
                               f"see {vdir / (ref_id + '.json')})")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["assign", "extract"], required=True,
                    help="assign: selection/assignment stats only (no patch "
                         "reads); extract: full extraction")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref-ids", nargs="+")
    src.add_argument("--ref-ids-file", type=str)
    ap.add_argument("--rdm-dir", type=Path, default=RDM_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="required for --mode extract; one "
                         "<ref_id>.geoparquet per ref")
    ap.add_argument("--index-source", choices=["stac", "auto"], default="auto",
                    help="'fs' is not offered: assignment needs footprints, "
                         "which only the STAC source provides")
    ap.add_argument("--catalog-cache", type=Path, default=None)
    ap.add_argument("--agera5-cache", type=Path, default=None,
                    help="cache dir for AGERA5 monthly composites "
                         "(default: <out-dir>/_agera5_cache)")
    ap.add_argument("--conventions", type=Path, default=None,
                    help="JSON conventions file (default: engine built-ins, "
                         "i.e. the validated locked conventions)")
    ap.add_argument("--only-flagged", action="store_true",
                    help="keep only samples with extract > 0 (the flow's "
                         "only_flagged_samples; default keeps collaterals)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-limit", type=int, default=None)
    ap.add_argument("--verify-pct", type=float, default=None, metavar="P",
                    help="verify P%% of a ref's primary samples, clamped to "
                         "[10, 200] points (overrides --verify). Fixed N is "
                         "statistically sufficient for systematic faults; "
                         "use pct only if per-ref proportionality matters.")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="after each ref's extraction, verify N random "
                         "primary samples against the openEO-era store: "
                         "every difference must be explainable (neighbour-"
                         "pixel bug / orbit / float32 edge noise / documented "
                         "aux tolerance) or the ref FAILS. 0 disables.")
    ap.add_argument("--max-divergence-frac", type=float, default=0.3,
                    help="verification tripwire: FAIL a ref when more than "
                         "this fraction of checked points are geometry-"
                         "divergent (see ptp_verify.py)")
    ap.add_argument("--verify-store", type=Path,
                    default=None,
                    help="openEO-era reference store "
                         "(default: the canonical WITH_ANOMALY store)")
    args = ap.parse_args()

    if args.mode == "extract":
        if args.out_dir is None:
            ap.error("--out-dir is required for --mode extract")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ptp_engine.AGERA5_CACHE = (args.agera5_cache
                                   or args.out_dir / "_agera5_cache")
        ptp_engine.AGERA5_CACHE.mkdir(parents=True, exist_ok=True)

    if (args.verify or args.verify_pct) and args.verify_store is None:
        from ptp_verify import STORE_DEFAULT
        args.verify_store = STORE_DEFAULT

    conventions = DEFAULT_CONVENTIONS
    if args.conventions and Path(args.conventions).exists():
        import json
        conventions = json.loads(Path(args.conventions).read_text())
        logger.info(f"Loaded conventions from {args.conventions}")

    refs = (args.ref_ids if args.ref_ids else
            [line.strip() for line in
             Path(args.ref_ids_file).read_text().splitlines()
             if line.strip() and not line.startswith("#")])

    all_stats = []
    failed: List[str] = []
    for ref in refs:
        try:
            s = run_ref(ref, args, conventions)
            if s:
                all_stats.append(s)
        except Exception as exc:
            logger.exception(f"{ref}: FAILED — continuing ({exc})")
            failed.append(ref)

    if all_stats:
        summary = pd.DataFrame(all_stats)
        print(summary.to_string(index=False))
        if args.out_dir:
            summary.to_csv(args.out_dir / "_rdm_campaign_stats.csv",
                           mode="a", header=not (args.out_dir / "_rdm_campaign_stats.csv").exists(),
                           index=False)
    if failed:
        logger.error(f"{len(failed)} ref(s) failed: {failed}")
        raise SystemExit(1)
    logger.success("Done.")


if __name__ == "__main__":
    main()
