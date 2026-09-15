"""Patch-to-point extraction for ARBITRARY point sources (no RDM required).

The RDM campaign route (ptp_campaign_rdm.py) discovers work by REF NAME: it
reads <rdm_dir>/<ref_id>/harmonized/<ref_id>.geoparquet, derives centroids, and
assigns each point to a patch of the SAME ref. That makes it impossible to
extract points that did not come from RDM, or whose patches live under a
different ref -- the `-INPATCH` case, where points were sampled inside other
datasets' patches.

This script drops that coupling. Give it:
  * a point file with `ref_id`, `sample_id` and either a point `geometry` or
    `lat`/`lon` columns (Overture, OSM, hand-digitised, GEE exports, active
    -learning candidates, ...);
  * a patch index built by ptp_global_index.py scan (or the campaign
    index -- any parquet with ref_id/sample_id/s2_path/s1_*/tile/zone/geometry).

and it assigns every point to whichever patch on disk actually contains it,
REGARDLESS of ref_id, then runs the same ptp_engine extraction. Outputs use the
same 34-column schema as the campaign, so they drop straight into the training
data.

Why this is faster than the RDM route: no harmonized-file dependency, no RDM
ingestion step, and the point->patch join is one STRtree query over the whole
archive instead of a per-ref catalog fetch. Adding a new point source becomes
"write a parquet, run this".

Grouping / output naming
------------------------
Extraction is driven per HOST ref (the ref whose patches are being opened),
because the month axis and the patch index are host-scoped. Outputs are written
per SOURCE ref_id (the value in your point file) so they stay attributable:
    <out-dir>/<source_ref_id>.geoparquet
A source ref whose points span several hosts is extracted once per host and the
parts are concatenated.

Attributes
----------
ptp_engine emits RDM attribute columns. Any your file lacks are filled with the
documented defaults below, so a bare lat/lon source still produces a schema
-compatible output. Supply real values by including the columns.

CLI
---
  # local, one process
  python ptp_points_extract.py \
      --points my_points.geoparquet \
      --index /vitodata/worldcereal/data/test_spark_runs/patches_index_extended.geoparquet \
      --out-dir /vitodata/worldcereal/data/POINTS_TEST

  # only some source refs, capped for a smoke test
  python ptp_points_extract.py ... --ref-ids 2022_EU_OVERTURE-BUILTUP-INPATCH_POINT_100 \
      --sample-limit 500

  # what would it do? (no extraction)
  python ptp_points_extract.py ... --assign-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Defaults for RDM attributes a generic point source will not have.
# 0 / "" are the engine's own "unknown" encodings; keeping them explicit here
# means a downstream reader can tell a real value from a filled one.
ATTR_DEFAULTS: Dict[str, object] = {
    "ewoc_code": 0,
    "valid_time": "",          # passed through to the output as-is
    "irrigation_status": 0,
    "quality_score_lc": 0,
    "quality_score_ct": 0,
    "extract": 0,
    "h3_l3_cell": "",
}


# --- Point loading -----------------------------------------------------------

def load_points(path: Path,
                ref_ids: Optional[List[str]] = None) -> gpd.GeoDataFrame:
    """Read the point source; accept geometry or lat/lon; fill attributes."""
    filters = [("ref_id", "in", ref_ids)] if ref_ids else None
    try:
        gdf = gpd.read_parquet(path, filters=filters)
    except ValueError:
        # No geometry column -- lat/lon source.
        df = pd.read_parquet(path, filters=filters)
        gdf = None
    else:
        df = None

    if gdf is None:
        cols = {c.lower(): c for c in df.columns}
        lat = cols.get("lat") or cols.get("latitude")
        lon = cols.get("lon") or cols.get("longitude")
        if not (lat and lon):
            raise SystemExit(
                f"{path}: no geometry column and no lat/lon columns "
                f"(have: {list(df.columns)})")
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")

    if gdf.crs is None:
        logger.warning("point file has no CRS; assuming EPSG:4326")
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)

    for col in ("ref_id", "sample_id"):
        if col not in gdf.columns:
            raise SystemExit(f"{path}: required column '{col}' is missing")

    # Non-point geometries (someone handed us polygons): use the same rule as
    # the RDM route -- EPSG:3857 centroid, dropped when outside its own shape.
    if not (gdf.geom_type == "Point").all():
        from ptp_campaign_rdm import _centroid_points
        n0 = len(gdf)
        gdf = _centroid_points(gdf)
        logger.info(f"reduced {n0:,} non-point geometries to {len(gdf):,} "
                    "centroids (EPSG:3857 rule, outside-own-shape dropped)")

    # Quantise to 1e-11 deg (~1 um), matching ptp_campaign_rdm._centroid_points.
    # Supplied coordinates carry full float64 precision, and the aux samplers
    # run them through PROJ, whose inverse transforms contract multiply-adds
    # into FMA where the CPU offers it — so the same point sampled on a VM
    # (AVX2+FMA3) and a YARN node (SSE only) can land 1-5 ulp apart and, at a
    # pixel boundary, pick a different slope/elevation cell. Rounding here is
    # ~1e7x finer than a 10 m pixel and makes runs bit-comparable across hosts.
    gdf["geometry"] = gpd.points_from_xy(
        np.round(gdf.geometry.x.to_numpy(), 11),
        np.round(gdf.geometry.y.to_numpy(), 11), crs=4326)

    for col, default in ATTR_DEFAULTS.items():
        if col not in gdf.columns:
            gdf[col] = default

    n0 = len(gdf)
    gdf = gdf.drop_duplicates(subset=["ref_id", "sample_id"], keep="first")
    if len(gdf) != n0:
        logger.warning(f"dropped {n0 - len(gdf):,} duplicate "
                       "(ref_id, sample_id) rows")
    logger.info(f"{len(gdf):,} points across {gdf.ref_id.nunique()} source refs")
    return gdf


# --- Assignment --------------------------------------------------------------

def _point_year(row) -> Optional[int]:
    """Target year of a point: explicit `year`, else `valid_time`, else the
    leading 4 digits of its ref_id (the campaign's own naming convention)."""
    y = getattr(row, "year", None)
    if y is not None and not (isinstance(y, float) and np.isnan(y)):
        try:
            return int(y)
        except (TypeError, ValueError):
            pass
    vt = getattr(row, "valid_time", None)
    if isinstance(vt, str) and len(vt) >= 4 and vt[:4].isdigit():
        return int(vt[:4])
    rid = getattr(row, "ref_id", "")
    if isinstance(rid, str) and len(rid) >= 4 and rid[:4].isdigit():
        return int(rid[:4])
    return None


def assign_to_patches(points: gpd.GeoDataFrame,
                      index: gpd.GeoDataFrame,
                      require_temporal: bool = True) -> Tuple[
                          gpd.GeoDataFrame, dict]:
    """Attach host_ref_id/host_sample_id by point-in-footprint AND time.

    Spatial containment alone is not enough: patch footprints from different
    years overlap, so a 2019 point sits inside 2018..2024 patches of the same
    LPIS region. Extracting it from a 2024 patch yields a 2023-2025 time series
    for a 2019 label — silently wrong training data. The index carries each
    patch's real window (`start_date`/`end_date`, parsed from the filename), so
    a candidate patch is only valid when it COVERS the point's target year.

    Resolution mirrors ptp_campaign_rdm.select_and_assign so results stay
    comparable: prefer the patch whose sample_id equals the point's own; else
    the covering patch whose footprint centroid is nearest (ties broken by
    sample_id, making the choice deterministic across machines).
    """
    idx = index[index.geometry.notna() & index.s2_path.notna()].reset_index(
        drop=True)
    if idx.empty:
        raise SystemExit("index has no rows with both a footprint and an S2 "
                         "path -- rebuild it with footprints")
    logger.info(f"index: {len(idx):,} usable patches across "
                f"{idx.ref_id.nunique()} refs")

    idx_cols = ["ref_id", "sample_id", "geometry"]
    have_dates = {"start_date", "end_date"} <= set(idx.columns)
    if have_dates:
        idx_cols += ["start_date", "end_date"]
    elif require_temporal:
        raise SystemExit(
            "index has no start_date/end_date columns, so points cannot be "
            "matched to the right YEAR. Rebuild it with "
            "ptp_global_index.py scan, or pass --no-temporal-check to "
            "accept spatially-correct but possibly wrong-year extractions.")

    joined = gpd.sjoin(points, idx[idx_cols].rename(
        columns={"ref_id": "host_ref_id", "sample_id": "host_sample_id"}),
        how="left", predicate="within")

    stats = {
        "points_in": int(len(points)),
        "points_covered": int(joined.index_right.notna()
                              .groupby(joined.index).any().sum()),
        "multi_cover_resolved": 0,
        "own_patch": 0,
        "dropped_wrong_year": 0,
        "points_covered_in_year": 0,
    }
    joined = joined[joined.index_right.notna()].copy()

    if have_dates and require_temporal and not joined.empty:
        # Prefer a DATE-granular test: a patch window like 2022-08-01..2024-03-31
        # "covers 2024" by the year test yet excludes a 2024-06-01 valid_time,
        # producing a series centred on the wrong season. ptp_campaign_rdm
        # compares full timestamps (`vt >= t_lo & vt <= t_hi`); match that
        # wherever valid_time is present, and fall back to the year test only
        # for points that have none.
        vt = pd.to_datetime(joined.get("valid_time"), errors="coerce") \
            if "valid_time" in joined.columns else pd.Series(
                pd.NaT, index=joined.index)
        sd = pd.to_datetime(joined.start_date, errors="coerce")
        ed = pd.to_datetime(joined.end_date, errors="coerce")
        by_date = (vt >= sd) & (vt <= ed)

        want = np.array([_point_year(r) for r in joined.itertuples()])
        lo = joined.start_date.str.slice(0, 4).astype("float")
        hi = joined.end_date.str.slice(0, 4).astype("float")
        by_year = ((lo.to_numpy() <= want.astype("float"))
                   & (want.astype("float") <= hi.to_numpy()))

        has_vt = vt.notna().to_numpy()
        keep = np.array([w is None for w in want]) | np.where(
            has_vt, by_date.to_numpy(), by_year)
        stats["temporal_by_date"] = int(has_vt.sum())
        before = joined.index.nunique()
        joined = joined[keep]
        stats["dropped_wrong_year"] = int(before - joined.index.nunique())
        stats["points_covered_in_year"] = int(joined.index.nunique())
        if stats["dropped_wrong_year"]:
            logger.warning(
                f"{stats['dropped_wrong_year']:,} point(s) fell inside a patch "
                "but no patch covering their target year — dropped")
    if joined.empty:
        return points.iloc[0:0].assign(host_ref_id=None, host_sample_id=None), \
            stats

    # Centroids only for the patches that actually tie, and computed on the
    # geometry as-is (4326). Distances are compared between candidates a few
    # hundred metres apart at most, so degrees rank them the same as metres
    # would; projecting 886k footprints to get that ordering is not worth it.
    # `.centroid` on a geographic CRS warns for this reason -- silence it
    # rather than compute a value we would only use for ranking.
    import warnings

    _centroids: Dict[tuple, object] = {}

    def _centroid(key: tuple, geom) -> object:
        if key not in _centroids:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _centroids[key] = geom.centroid
        return _centroids[key]

    idx_geom = dict(zip(zip(idx.ref_id, idx.sample_id), idx.geometry))

    def _pick(grp: pd.DataFrame) -> pd.Series:
        if len(grp) == 1:
            return grp.iloc[0]
        own = grp[grp.host_sample_id == grp.sample_id.iloc[0]]
        if len(own):
            return own.iloc[0]
        pt = grp.geometry.iloc[0]
        d = []
        for i, r in enumerate(grp.itertuples()):
            key = (r.host_ref_id, r.host_sample_id)
            d.append((pt.distance(_centroid(key, idx_geom[key])),
                      r.host_sample_id, i))
        d.sort()
        return grp.iloc[d[0][2]]

    dup = joined.index.duplicated(keep=False)
    stats["multi_cover_resolved"] = int(
        joined[dup].index.nunique()) if dup.any() else 0
    if dup.any():
        singles = joined[~dup]
        picked = (joined[dup].groupby(level=0, group_keys=False)
                  .apply(_pick))
        joined = pd.concat([singles, picked]).sort_index()

    stats["own_patch"] = int(
        (joined.host_sample_id == joined.sample_id).sum())
    out = gpd.GeoDataFrame(joined.drop(columns=["index_right"]),
                           geometry="geometry", crs="EPSG:4326")
    logger.info(f"assigned {len(out):,}/{stats['points_in']:,} points "
                f"({stats['own_patch']:,} to their own patch, "
                f"{stats['multi_cover_resolved']:,} multi-cover resolved)")
    return out, stats


# --- Extraction --------------------------------------------------------------

def extract_source_ref(source_ref: str,
                       pts: gpd.GeoDataFrame,
                       index: gpd.GeoDataFrame,
                       out_path: Path,
                       workers: int,
                       agera5_cache: Optional[Path],
                       conventions: Optional[dict] = None) -> dict:
    """Run ptp_engine for one SOURCE ref, once per host ref it touches."""
    import ptp_engine
    from ptp_engine import DEFAULT_CONVENTIONS, extract_host
    conv = conventions if conventions is not None else DEFAULT_CONVENTIONS

    if agera5_cache is not None:
        ptp_engine.AGERA5_CACHE = Path(agera5_cache)
        ptp_engine.AGERA5_CACHE.mkdir(parents=True, exist_ok=True)

    parts: List[gpd.GeoDataFrame] = []
    stats = {"source_ref_id": source_ref, "points": int(len(pts)),
             "hosts": int(pts.host_ref_id.nunique()), "rows": 0}

    for host, grp in pts.groupby("host_ref_id"):
        sub = index[index.ref_id == host]
        entries = _index_to_entries(sub)
        tmp = out_path.parent / f".{out_path.stem}__{host}.part.geoparquet"
        try:
            extract_host(
                host, conv,
                workers=workers,
                out_path=tmp,
                points=grp,
                index_source="auto",
                catalog_cache=None,
                index=entries,
            )
        except Exception as exc:
            logger.error(f"{source_ref} via host {host}: {exc}")
            continue
        if tmp.exists():
            parts.append(gpd.read_parquet(tmp))
            tmp.unlink()

    if not parts:
        logger.warning(f"{source_ref}: nothing extracted")
        return stats
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                           crs="EPSG:4326")
    # The engine writes the HOST ref_id (its own convention); restore ours so
    # the file is attributable to the source dataset.
    if "ref_id" in out.columns:
        out["host_ref_id"] = out["ref_id"]
    out["ref_id"] = source_ref
    out.to_parquet(out_path)
    stats["rows"] = int(len(out))
    logger.success(f"{source_ref}: {len(out):,} rows -> {out_path}")
    return stats


def _index_to_entries(sub: gpd.GeoDataFrame) -> Dict[str, dict]:
    """Index rows -> the entries dict ptp_engine expects."""
    def _path(v) -> Optional[Path]:
        # Absent paths arrive as NaN (float) from parquet, not None, so a bare
        # truthiness test still hands Path() a float.
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            return None
        return Path(v)

    entries: Dict[str, dict] = {}
    for r in sub.itertuples():
        s1: Dict[str, Path] = {}
        asc, desc = _path(getattr(r, "s1_asc", None)), \
            _path(getattr(r, "s1_desc", None))
        if asc is not None:
            s1["ASCENDING"] = asc
        if desc is not None:
            s1["DESCENDING"] = desc
        h3 = getattr(r, "h3", None)
        if isinstance(h3, float) and np.isnan(h3):
            h3 = None
        entries[r.sample_id] = {
            "tile": r.tile, "zone": r.zone,
            "s2": _path(r.s2_path),
            "s1": s1,
            "h3": h3,
            "footprint": r.geometry,
        }
    return entries


# --- Driver ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, required=True,
                    help="parquet/geoparquet with ref_id, sample_id and "
                         "geometry (or lat/lon)")
    ap.add_argument("--index", type=Path, required=True,
                    help="patch index geoparquet (extended recommended)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ref-ids", nargs="+",
                    help="only these SOURCE ref_ids")
    ap.add_argument("--workers", type=int, default=2,
                    help="engine worker processes per host")
    ap.add_argument("--sample-limit", type=int, default=None,
                    help="cap points per source ref (smoke tests)")
    ap.add_argument("--freq", choices=["month", "dekad"], default="month",
                    help="compositing period (patch_to_point.py --period): "
                         "'month' (default) or 'dekad' (3 composites per "
                         "month at days 1/11/21). Must match the campaign "
                         "run this output will sit alongside.")
    ap.add_argument("--s2-mask", choices=["dilated", "raw_scl"],
                    default="dilated",
                    help="S2 cloud masking: 'dilated' (precomputed "
                         "erosion/dilation band) or 'raw_scl'.")
    ap.add_argument("--agera5-cache", type=Path, default=None,
                    help="default: <out-dir>/_agera5_cache")
    ap.add_argument("--assign-only", action="store_true",
                    help="report the point->patch assignment and stop")
    ap.add_argument("--no-temporal-check", action="store_true",
                    help="DANGEROUS: assign on space alone. Patch footprints "
                         "from different years overlap, so this can extract a "
                         "2023-2025 series for a 2019 point. Only for an index "
                         "without start_date/end_date.")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract refs whose output already exists")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    agera5 = args.agera5_cache or args.out_dir / "_agera5_cache"

    points = load_points(args.points, args.ref_ids)
    index = gpd.read_parquet(args.index)
    assigned, astats = assign_to_patches(
        points, index, require_temporal=not args.no_temporal_check)

    if assigned.empty:
        raise SystemExit("no point fell inside any patch footprint")

    per_ref = (assigned.groupby("ref_id")
               .agg(points=("sample_id", "size"),
                    hosts=("host_ref_id", "nunique"))
               .sort_values("points", ascending=False))
    logger.info(f"assignment summary:\n{per_ref.to_string()}")
    (args.out_dir / "_assignment_summary.csv").write_text(
        per_ref.to_csv())

    uncovered = astats["points_in"] - astats["points_covered"]
    if uncovered:
        logger.warning(f"{uncovered:,} point(s) fell in NO patch and cannot "
                       "be extracted")

    if args.assign_only:
        logger.info("--assign-only: stopping before extraction")
        return

    from ptp_engine import DEFAULT_CONVENTIONS, seed_meteo_cache
    conv = {**DEFAULT_CONVENTIONS, "freq": args.freq,
            "s2_mask": args.s2_mask}
    logger.info(f"period={args.freq}  s2_mask={args.s2_mask}")

    # Seed AGERA5 before extraction: a cold shared cache makes every worker
    # fetch the same rasters at once and read each other's half-written files.
    if agera5 is not None and {"start_date", "end_date"} <= set(index.columns):
        try:
            seed_meteo_cache(agera5, str(index.start_date.min()),
                             str(index.end_date.max()), freq=args.freq)
        except Exception as exc:                # noqa: BLE001
            logger.warning(f"AGERA5 seeding skipped: {exc}")

    all_stats = []
    for source_ref, pts in assigned.groupby("ref_id"):
        out_path = args.out_dir / f"{source_ref}.geoparquet"
        if out_path.exists() and not args.overwrite:
            logger.warning(f"{source_ref}: output exists, skipping")
            continue
        if args.sample_limit:
            pts = pts.head(args.sample_limit)
        all_stats.append(extract_source_ref(
            source_ref, pts, index, out_path, args.workers, agera5,
            conventions=conv))

    if all_stats:
        pd.DataFrame(all_stats).to_csv(
            args.out_dir / "_points_extract_stats.csv", index=False)
    done = sum(1 for s in all_stats if s.get("rows"))
    logger.success(f"{done}/{len(all_stats)} source refs extracted")


if __name__ == "__main__":
    main()
