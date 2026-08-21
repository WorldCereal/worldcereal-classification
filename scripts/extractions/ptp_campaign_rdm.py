"""Campaign driver: full-RDM reprocessing via the local patch-to-point engine.

For each ref_id, extracts the RDM samples from that ref's OWN patch
extractions — the standard patch-to-point flow, minus openEO.

Per ref, three steps:

  1. SELECT — stream the harmonized RDM geoparquet
     (<rdm-dir>/<ref>/harmonized/<ref>.geoparquet) row-group by row-group,
     keeping samples that (a) fall inside any patch footprint of the ref and
     (b) have valid_time inside the ref's patch window. Polygon samples are
     reduced to a point by the HYBRID rule (2026-08-21):
       * true centroid — the EPSG:3857 centroid of the full polygon, when it
         lies inside its own polygon and inside a patch footprint;
       * clipped fallback — otherwise, the production/openEO-era point: the
         3857 centroid of the polygon clipped to the job's patch-footprint
         MultiPolygon shrunk 20 m inward (rdm_interaction.get_samples,
         buffer=-20, cap_style=3; ST_Simplify 1e-6 on both geometries), kept
         only if that centroid lies inside the clipped piece (gdf_to_points).
     The fallback recovers the collateral polygons that overlap a patch while
     their centroid falls just outside it (~1.3 M samples campaign-wide),
     placing them exactly where the openEO-era store had them. `point_kind`
     ('centroid' | 'clipped' | 'point') records the rule per sample.
     --edge-fallback additionally applies the fallback to centroids inside the
     outer 20 m band of their patch (fully production-faithful; moves ~5-10 %
     of existing samples). --legacy-centroid disables the fallback.
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
  # targeted delta: extract only samples NOT already in an earlier output
  ptp_campaign_rdm.py --mode extract --ref-ids <ref> --out-dir DIR_DELTA --delta-from DIR
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
import numpy as np
from pyproj import Transformer
from shapely import make_valid
from shapely import wkb as shapely_wkb
from shapely.geometry import MultiPolygon, Point
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree

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


EDGE_MARGIN_M_DEFAULT = 20.0   # production: rdm_interaction.get_samples buffer=-20
SIMPLIFY_DEG = 1e-6            # production: ST_Simplify(geometry, 0.000001)
_TO_MERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
_TO_LL = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform


def _tile_epsg(tile: Optional[str], zone: Optional[str]) -> Optional[int]:
    """EPSG of a patch from its S2 tile id (31TDJ -> 32631, 33KXA -> 32733)."""
    try:
        z = int(zone) if zone not in (None, "") else int(str(tile)[:2])
    except (TypeError, ValueError):
        return None
    band = str(tile)[2].upper() if tile and len(str(tile)) >= 3 else "N"
    return (32600 if band >= "N" else 32700) + z


def shrunk_extents(catalog: RefCatalog,
                   margin_m: float = EDGE_MARGIN_M_DEFAULT) -> Dict[int, list]:
    """Production's spatial extent per openEO job (one ref x one EPSG), as
    eroded footprint PARTS.

    patch_to_point_worldcereal.get_label_points builds a MultiPolygon of the
    job's S2 STAC item footprints (each .buffer(1e-9)); rdm_interaction.
    get_samples estimates one UTM CRS for it, buffers it by -20 m with square
    corners (cap_style=3) — GEOS erodes each member and unions the results —
    returns to EPSG:4326 and ST_Simplify(1e-6)s it before clipping the RDM
    polygons. Reproduced here from the catalog footprints grouped by patch
    EPSG; the eroded members are kept as a list so that a polygon is clipped
    only against the few parts it touches (intersection distributes over the
    union; intersecting each polygon with a 10k-part union is ~100x slower)."""
    groups: Dict[int, list] = {}
    for e in catalog.entries.values():
        fp = e.get("footprint")
        if fp is None:
            continue
        parts = list(fp.geoms) if fp.geom_type == "MultiPolygon" else [fp]
        epsg = _tile_epsg(e.get("tile"), e.get("zone"))
        groups.setdefault(epsg, []).extend(p.buffer(1e-9) for p in parts)
    out: Dict[int, list] = {}
    for epsg, polys in groups.items():
        g = gpd.GeoSeries(polys, crs="EPSG:4326")
        utm = gpd.GeoSeries([MultiPolygon(polys)], crs="EPSG:4326").estimate_utm_crs()
        shr = g.to_crs(utm).buffer(-margin_m, cap_style=3).to_crs("EPSG:4326")
        keep = [s.simplify(SIMPLIFY_DEG, preserve_topology=True)
                for s in shr.values if not s.is_empty]
        if keep:
            out[epsg] = keep
    return out


def _clipped_centroid(poly, parts: List[Any]):
    """gdf_to_points on the production-clipped polygon: 3857 centroid of
    make_valid(simplify(poly)) ∩ (union of the eroded footprint parts the
    polygon touches), kept only if inside the clipped piece. Returns a 4326
    Point or None."""
    spoly = make_valid(poly.simplify(SIMPLIFY_DEG, preserve_topology=True))
    pieces = [spoly.intersection(p) for p in parts]
    pieces = [q for q in pieces if not q.is_empty]
    if not pieces:
        return None
    clip = pieces[0] if len(pieces) == 1 else unary_union(pieces)
    clip_m = shapely_transform(_TO_MERC, clip)
    c = clip_m.centroid
    if clip_m.is_empty or not c.within(clip_m):
        return None
    return shapely_transform(_TO_LL, c)


def _place_points(
    gdf: gpd.GeoDataFrame,
    raw_tree: STRtree,
    shrunk: Dict[int, Any],
    stats: dict,
    edge_fallback: bool = False,
    legacy_centroid: bool = False,
) -> gpd.GeoDataFrame:
    """Hybrid point rule (see module docstring). Returns the kept rows with
    EPSG:4326 point geometry and a `point_kind` column."""
    if gdf.empty:
        return gdf
    is_pt = gdf.geometry.geom_type.isin(("Point", "MultiPoint")).to_numpy()
    merc = gdf.to_crs(epsg=3857)
    cen_m = merc.geometry.centroid
    inside_poly = (cen_m.within(merc.geometry).to_numpy() | is_pt)
    cen_ll = gpd.GeoSeries(cen_m, crs=3857).to_crs(4326)
    cen_vals = np.asarray(cen_ll.values, dtype=object)
    covered = np.zeros(len(gdf), dtype=bool)
    hit = raw_tree.query(cen_vals, predicate="intersects")
    if len(hit[0]):
        covered[np.unique(hit[0])] = True
    keep_true = inside_poly & covered
    shr_parts = [g for parts in shrunk.values() for g in parts]
    shr_tree = STRtree(shr_parts) if shr_parts else None
    if edge_fallback and shr_tree is not None:
        deep = np.zeros(len(gdf), dtype=bool)
        h2 = shr_tree.query(cen_vals, predicate="intersects")
        if len(h2[0]):
            deep[np.unique(h2[0])] = True
        keep_true = keep_true & (deep | is_pt)
    kinds = np.where(is_pt, "point", "centroid").astype(object)
    geoms = list(cen_vals)
    sel = keep_true.copy()
    if not legacy_centroid and shr_tree is not None:
        for i in np.where(~keep_true & ~is_pt)[0]:
            poly = gdf.geometry.iloc[i]
            cand = shr_tree.query(poly, predicate="intersects")
            pt = _clipped_centroid(poly, [shr_parts[int(j)] for j in cand]) \
                if len(cand) else None
            if pt is not None:
                geoms[i] = pt
                kinds[i] = "clipped"
                sel[i] = True
                stats["clipped_fallback"] += 1
                if inside_poly[i] and covered[i]:   # only under --edge-fallback
                    stats["edge_relocated"] += 1
            elif len(raw_tree.query(poly, predicate="intersects")):
                # the polygon does touch a patch, yet no point could be placed:
                # either it only grazes the outer 20 m band, or its centroid
                # lies outside the polygon and so does the clipped piece's.
                stats["outside_patches" if inside_poly[i] else "centroid_dropped"] += 1
            # else: polygon nowhere near a patch — the ordinary spatial exclusion
    else:
        stats["centroid_dropped"] += int((~sel & ~inside_poly & covered).sum())
        stats["outside_patches"] += int((~sel & inside_poly & covered).sum())
    out = gdf.loc[sel].copy()
    out["geometry"] = gpd.GeoSeries([geoms[i] for i in np.where(sel)[0]],
                                    index=out.index, crs="EPSG:4326")
    out["point_kind"] = kinds[sel]
    return out


def select_and_assign(
    ref_id: str,
    catalog: RefCatalog,
    rdm_dir: Path,
    only_flagged: bool = False,
    sample_limit: Optional[int] = None,
    edge_fallback: bool = False,
    legacy_centroid: bool = False,
    edge_margin_m: float = EDGE_MARGIN_M_DEFAULT,
    delta_from: Optional[Path] = None,
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
        starts.append(stem[-2])
        ends.append(stem[-1])
    t_lo, t_hi = min(starts), max(ends)

    stats: Dict[str, Any] = {"ref_id": ref_id, "rows_read": 0, "kept_h3": 0, "kept_time": 0,
             "kept_spatial": 0, "primary": 0, "collateral": 0,
             "multi_cover_resolved": 0, "centroid_dropped": 0,
             "clipped_fallback": 0, "edge_relocated": 0, "outside_patches": 0,
             "skipped_existing": 0}
    shrunk = {} if legacy_centroid else shrunk_extents(catalog, edge_margin_m)
    if not legacy_centroid and not shrunk:
        logger.warning(f"{ref_id}: no shrunk extents could be built; "
                       "clipped fallback disabled for this ref")

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

        gdf = _place_points(gdf, tree, shrunk, stats,
                            edge_fallback=edge_fallback,
                            legacy_centroid=legacy_centroid)
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
    if delta_from is not None:
        prev = Path(delta_from) / f"{ref_id}.geoparquet"
        if prev.exists():
            existing = set(pq.read_table(prev, columns=["sample_id"])
                           .column("sample_id").unique().to_pylist())
            n0 = len(points)
            points = points[~points["sample_id"].isin(existing)]
            stats["skipped_existing"] = n0 - len(points)
            logger.info(f"{ref_id}: delta mode — {stats['skipped_existing']:,} "
                        f"sample(s) already in {prev.name} skipped, "
                        f"{len(points):,} new to extract")
        else:
            logger.info(f"{ref_id}: delta mode — no earlier output at {prev}; "
                        "extracting the full selection")
    if stats["clipped_fallback"]:
        logger.info(f"{ref_id}: {stats['clipped_fallback']:,} sample(s) placed "
                    "by the clipped fallback (point_kind='clipped')")
    if stats["centroid_dropped"] or stats["outside_patches"]:
        logger.warning(
            f"{ref_id}: {stats['centroid_dropped']:,} sample(s) touching a patch "
            "have no usable point (centroid outside the polygon, no clipped "
            f"fallback) and {stats['outside_patches']:,} only graze a patch's "
            "outer 20 m band; both dropped")
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
    if args.mode == "extract" and out_path is not None and out_path.exists():
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
        only_flagged=args.only_flagged, sample_limit=args.sample_limit,
        edge_fallback=args.edge_fallback, legacy_centroid=args.legacy_centroid,
        edge_margin_m=args.edge_margin_m,
        delta_from=Path(args.delta_from) if args.delta_from else None)
    logger.info(f"{ref_id}: {stats}")
    if args.dump_points:
        dp = Path(args.dump_points)
        dp.mkdir(parents=True, exist_ok=True)
        points.to_parquet(dp / f"{ref_id}.points.geoparquet", index=False)
        logger.info(f"{ref_id}: points frame dumped to {dp}")
    if args.mode == "assign" or points.empty:
        return stats

    extract_host(
        ref_id, conventions,
        workers=args.workers,
        out_path=out_path,
        points=points,
        index_source=args.index_source,
        catalog_cache=Path(args.catalog_cache) if args.catalog_cache else None,
        index=catalog.entries,
    )
    stats["out_path"] = str(out_path)

    if (args.verify or args.verify_pct) and out_path is not None:
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
        # tmp+rename: certs may be owned by another user (not group-
        # writable); replacing needs only directory write permission.
        vtmp = vdir / f"{ref_id}.json.tmp{os.getpid()}"
        vtmp.write_text(json.dumps(v, indent=2))
        try:
            os.chmod(vtmp, 0o664)
        except OSError:
            pass
        vtmp.rename(vdir / f"{ref_id}.json")
        if v["status"] == "FAIL":
            raise RuntimeError(f"verification FAILED for {ref_id} "
                               f"(unexplained differences vs openEO store; "
                               f"see {vdir / (ref_id + '.json')})")
    return stats


def main() -> None:
    # Campaign logs are INFO-level: per-batch detail (e.g. centroid drops)
    # is logger.debug and floods a large ref's log otherwise. Override with
    # PTP_LOG_LEVEL=DEBUG when actually debugging.
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("PTP_LOG_LEVEL", "INFO"))
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
    ap.add_argument("--delta-from", type=Path, default=None, metavar="DIR",
                    help="targeted re-extraction: skip samples whose sample_id "
                         "is already in DIR/<ref>.geoparquet (an earlier "
                         "campaign output); write only the new ones to "
                         "--out-dir (use a separate dir, then ptp_merge_delta)")
    ap.add_argument("--edge-fallback", action="store_true",
                    help="also apply the clipped fallback to centroids in the "
                         "outer --edge-margin-m band of their patch (fully "
                         "production-faithful; moves ~5-10%% of samples)")
    ap.add_argument("--edge-margin-m", type=float, default=EDGE_MARGIN_M_DEFAULT,
                    help="inward shrink of the patch footprints for the "
                         "clipped fallback (production: 20 m)")
    ap.add_argument("--legacy-centroid", action="store_true",
                    help="disable the clipped fallback (pre-2026-08-21 rule)")
    ap.add_argument("--dump-points", type=Path, default=None, metavar="DIR",
                    help="write the selected points frame (with point_kind and "
                         "host_sample_id) to DIR/<ref>.points.geoparquet")
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
            # One JSON per ref (idempotent on rerun), then regenerate the
            # CSV whole and rename into place. Appending would interleave
            # and duplicate: three shard machines write here concurrently,
            # and NFS appends are not atomic.
            stats_dir = args.out_dir / "_stats"
            stats_dir.mkdir(exist_ok=True)
            for s in all_stats:
                stmp = stats_dir / f"{s['ref_id']}.json.tmp{os.getpid()}"
                stmp.write_text(json.dumps(s, default=str, indent=1))
                try:
                    os.chmod(stmp, 0o664)
                except OSError:
                    pass
                stmp.rename(stats_dir / f"{s['ref_id']}.json")
            rows = [json.loads(p.read_text())
                    for p in sorted(stats_dir.glob("*.json"))]
            tmp = args.out_dir / f"_rdm_campaign_stats.csv.tmp{os.getpid()}"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.rename(args.out_dir / "_rdm_campaign_stats.csv")
    if failed:
        logger.error(f"{len(failed)} ref(s) failed: {failed}")
        raise SystemExit(1)
    logger.success("Done.")


if __name__ == "__main__":
    main()
