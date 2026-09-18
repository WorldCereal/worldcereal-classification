"""Global patches index for local patch-to-point extraction.

Builds ONE geoparquet holding, for every patch of every ref: its footprint
(EPSG:4326), file paths (S2 .nc, S1 per orbit), tile/zone/h3 and ref_id.
This turns point-to-patch assignment into a single spatial join over the
whole archive and gives coverage reporting ("which points have no patch
anywhere?") for free.

Built strictly on top of ref_catalog.RefCatalog (STAC-primary with fs
fallback and retries) — no new discovery code. Per-ref catalog parquets are
cached in --cache-dir, so rebuilding the global index only refetches refs
that were never cached.

Known caveats, addressed here:
  * STALENESS: the index is a snapshot. `built_at` (UTC) is stored in the
    parquet metadata, and --reconcile-sample N stat()s N random referenced
    files per ref at build time to quantify STAC/disk drift. At use time the
    extractor still degrades per-file at open, so a stale row costs one
    dropped point, never a wrong value.
  * REF-SCOPED SEMANTICS: the month axis and output/verification are per
    ref. `catalog_for_ref` slices the index back into a RefCatalog, so the
    existing ref-scoped machinery (select_and_assign, extract_host,
    ptp_verify) is reused unchanged.

CLI:
  # Build (append refs incrementally; existing refs in the index are kept):
  python ptp_global_index.py build --ref-ids <ref> ... --out patches_index.geoparquet
  python ptp_global_index.py build --ref-ids-file rdm_campaign_refs.txt --out ...

  # Inspect:
  python ptp_global_index.py info --index patches_index.geoparquet

  # Coverage of arbitrary points (geoparquet/gpkg with point geometry):
  python ptp_global_index.py coverage --index patches_index.geoparquet --points pts.geoparquet

TWO WAYS TO BUILD AN INDEX
--------------------------
`build` discovers patches BY REF NAME (STAC per ref_id, filesystem fallback
under <root>/<ref_id>/...). Anything not named after a ref in the campaign list
is invisible to it, which hides two things: orphaned ref dirs that exist on
disk but are absent from rdm_campaign_refs.txt, and the -INPATCH refs whose
points were sampled inside ANOTHER ref's patches (they own no patch directory
at all, so only a spatial lookup finds them).

`scan` instead walks both EXTRACTION roots directly and indexes every
<ref_dir>/<zone>/<tile>/<sample_id>/*.nc it finds, recording `in_campaign` so
the extra material is easy to isolate. Output schema is build's INDEX_COLUMNS
plus `in_campaign`, `footprint_source`, `start_date`, `end_date`, so
`catalog_for_ref` works on either index.

  # Fast: full disk walk, STAC footprints (~minutes)
  python ptp_global_index.py scan --out patches_index_extended.geoparquet

  # Exhaustive: footprints read from the NetCDFs themselves (hours; use screen)
  python ptp_global_index.py scan --footprints nc --workers 16 \
      --out patches_index_extended.geoparquet

  # What did the scan add over a build index?
  python ptp_global_index.py compare --extended ext.geoparquet \
      --campaign patches_index.geoparquet
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_catalog import RefCatalog  # noqa: E402
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from shapely.geometry import box
from ref_catalog import S1_ROOT, S2_ROOT, _iter_items  # noqa: E402

INDEX_COLUMNS = [
    "ref_id", "sample_id", "tile", "zone",
    "s2_path", "s1_asc", "s1_desc", "h3", "source", "geometry",
]


# --- Build ------------------------------------------------------------------


def _ref_to_frame(ref_id: str, cat: RefCatalog) -> gpd.GeoDataFrame:
    rows = []
    for sid, e in cat.entries.items():
        rows.append({
            "ref_id": ref_id,
            "sample_id": sid,
            "tile": e.get("tile"),
            "zone": e.get("zone"),
            "s2_path": str(e["s2"]) if e.get("s2") else None,
            "s1_asc": (str(e["s1"]["ASCENDING"])
                       if e.get("s1", {}).get("ASCENDING") else None),
            "s1_desc": (str(e["s1"]["DESCENDING"])
                        if e.get("s1", {}).get("DESCENDING") else None),
            "h3": e.get("h3"),
            "source": cat.source,
            "geometry": e.get("footprint"),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def build_index(
    ref_ids: List[str],
    out_path: Path,
    cache_dir: Optional[Path] = None,
    workers: int = 4,
    reconcile_sample: int = 0,
) -> gpd.GeoDataFrame:
    """Build/extend the global index. Refs already present in an existing
    index file are kept as-is (delete the file to force a full rebuild)."""
    existing: Optional[gpd.GeoDataFrame] = None
    done: set = set()
    if out_path.exists():
        existing = gpd.read_parquet(out_path)
        done = set(existing["ref_id"].unique())
        logger.info(f"Extending existing index: {len(existing):,} patches "
                    f"across {len(done)} refs already present")
    todo = [r for r in ref_ids if r not in done]
    if not todo:
        logger.info("Nothing to do — all requested refs already indexed")
        return existing if existing is not None else gpd.GeoDataFrame()

    drift_report: Dict[str, dict] = {}

    def _one(ref: str) -> Optional[gpd.GeoDataFrame]:
        try:
            cat = RefCatalog.load(ref, source="auto", cache_dir=cache_dir)
        except Exception as exc:  # a bad ref must not kill the build
            logger.error(f"{ref}: catalog build failed ({exc}); skipped")
            return None
        if not cat.entries:
            logger.warning(f"{ref}: no patches found; skipped")
            return None
        frame = _ref_to_frame(ref, cat)
        n_fp = int(frame.geometry.notna().sum())
        logger.info(f"{ref}: {len(frame):,} patches "
                    f"({n_fp:,} with footprint) via {cat.source}")
        if reconcile_sample > 0:
            import random
            with_s2 = frame[frame.s2_path.notna()]
            picks = with_s2.sample(
                n=min(reconcile_sample, len(with_s2)), random_state=0)
            missing = [p for p in picks.s2_path if not Path(p).exists()]
            drift_report[ref] = {"checked": len(picks), "missing": len(missing)}
            if missing:
                logger.warning(f"{ref}: {len(missing)}/{len(picks)} sampled "
                               "S2 paths missing on disk (STAC/disk drift)")
        return frame

    frames: List[gpd.GeoDataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for frame in pool.map(_one, todo):
            if frame is not None:
                frames.append(frame)

    parts = ([existing] if existing is not None else []) + frames
    if not parts:
        raise RuntimeError("index would be empty — nothing was built")
    index = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    # Uniqueness contract: one row per (ref_id, sample_id).
    dup = index.duplicated(subset=["ref_id", "sample_id"])
    if dup.any():
        logger.warning(f"dropping {int(dup.sum())} duplicate "
                       "(ref_id, sample_id) rows")
        index = index[~dup]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp{__import__('os').getpid()}")
    _write_with_meta(index, tmp, drift_report)
    tmp.rename(out_path)
    logger.success(f"Index: {len(index):,} patches / "
                   f"{index.ref_id.nunique()} refs -> {out_path}")
    return index


def _write_with_meta(gdf: gpd.GeoDataFrame, path: Path,
                     drift: Dict[str, dict]) -> None:
    """to_parquet, then re-attach our metadata (built_at, drift) losslessly."""
    import pyarrow.parquet as pq

    gdf.to_parquet(path, index=False)
    table = pq.read_table(path)
    meta = dict(table.schema.metadata or {})
    meta[b"ptp_index"] = json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_patches": len(gdf),
        "n_refs": int(gdf.ref_id.nunique()),
        "drift_sample": drift,
    }).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def read_index_meta(path: Path) -> dict:
    import pyarrow.parquet as pq
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(b"ptp_index")
    return json.loads(raw) if raw else {}


# --- Use --------------------------------------------------------------------


def catalog_for_ref(index: gpd.GeoDataFrame, ref_id: str) -> RefCatalog:
    """Slice the global index back into a RefCatalog so all existing
    ref-scoped consumers (select_and_assign, extract_host) work unchanged."""
    sub = index[index.ref_id == ref_id]
    entries: Dict[str, dict] = {}
    for r in sub.itertuples():
        s1 = {}
        if isinstance(r.s1_asc, str) and r.s1_asc:
            s1["ASCENDING"] = Path(r.s1_asc)
        if isinstance(r.s1_desc, str) and r.s1_desc:
            s1["DESCENDING"] = Path(r.s1_desc)
        entries[r.sample_id] = {
            "tile": r.tile if isinstance(r.tile, str) else None,
            "zone": r.zone if isinstance(r.zone, str) else None,
            "s2": Path(r.s2_path) if isinstance(r.s2_path, str) else None,
            "s1": s1,
            "h3": r.h3 if isinstance(r.h3, str) else None,
            "footprint": r.geometry if r.geometry is not None else None,
        }
    return RefCatalog(ref_id, entries, "global_index")


def coverage(index: gpd.GeoDataFrame,
             points: gpd.GeoDataFrame) -> pd.DataFrame:
    """For each point: how many patches cover it, and from which refs.
    Pure sjoin — the 'which points are even extractable' question."""
    pts = points.to_crs(4326) if points.crs else points.set_crs(4326)
    usable = index[index.s2_path.notna() & index.geometry.notna()]
    joined = gpd.sjoin(pts[["geometry"]].reset_index(names="_pt"),
                       usable[["ref_id", "sample_id", "geometry"]],
                       how="left", predicate="intersects")
    agg = joined.groupby("_pt").agg(
        n_patches=("sample_id", lambda s: int(s.notna().sum())),
        refs=("ref_id", lambda s: sorted(set(s.dropna()))))
    return agg


# --- CLI --------------------------------------------------------------------


OUT_COLUMNS = [
    "ref_id", "sample_id", "tile", "zone",
    "s2_path", "s1_asc", "s1_desc", "h3", "source",
    "in_campaign", "footprint_source",
    "start_date", "end_date", "geometry",
]


_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.nc$")


def _dates_from_path(*paths: Optional[str]) -> tuple:
    """(start, end) from the first path that carries the pattern."""
    for p in paths:
        if not p:
            continue
        m = _DATE_RE.search(p)
        if m:
            return m.group(1), m.group(2)
    return None, None


def _scan_ref_dir(ref_id: str, root: Path, key: str) -> Dict[str, dict]:
    """One ref dir under one root. Mirrors ref_catalog.build_from_fs's layout
    assumptions (and its os.scandir/readdir trick, which avoids a stat per
    entry on NFS)."""
    entries: Dict[str, dict] = {}
    base = root / ref_id
    if not base.is_dir():
        return entries
    try:
        with os.scandir(base) as zones:
            for zone in zones:
                if not zone.is_dir():
                    continue
                with os.scandir(zone.path) as tiles:
                    for tile in tiles:
                        if not tile.is_dir():
                            continue
                        with os.scandir(tile.path) as sdirs:
                            for sdir in sdirs:
                                if not sdir.is_dir():
                                    continue
                                e = entries.setdefault(sdir.name, {
                                    "tile": tile.name, "zone": zone.name,
                                    "s2": None, "s1_asc": None,
                                    "s1_desc": None,
                                })
                                with os.scandir(sdir.path) as files:
                                    for f in files:
                                        if not f.name.endswith(".nc"):
                                            continue
                                        if key == "s2":
                                            e["s2"] = f.path
                                            e["tile"] = tile.name
                                            e["zone"] = zone.name
                                        elif "_ASCENDING_" in f.name:
                                            e["s1_asc"] = f.path
                                        else:
                                            e["s1_desc"] = f.path
    except (PermissionError, FileNotFoundError) as exc:
        logger.warning(f"{ref_id} under {root}: {exc}")
    return entries


def walk_all(s2_root: Path, s1_root: Path, workers: int,
             only_refs: Optional[List[str]] = None) -> pd.DataFrame:
    """Every ref dir under both roots -> one row per (ref_id, sample_id)."""
    ref_dirs = set()
    for root in (s2_root, s1_root):
        if not root.is_dir():
            logger.error(f"root does not exist: {root}")
            continue
        for name in os.listdir(root):
            if (root / name).is_dir():
                ref_dirs.add(name)
    if only_refs:
        ref_dirs &= set(only_refs)
    logger.info(f"{len(ref_dirs)} ref directories found on disk")

    rows: List[dict] = []

    def _one(ref_id: str) -> List[dict]:
        merged: Dict[str, dict] = {}
        for root, key in ((s2_root, "s2"), (s1_root, "s1")):
            for sid, e in _scan_ref_dir(ref_id, root, key).items():
                tgt = merged.setdefault(sid, {
                    "tile": e["tile"], "zone": e["zone"],
                    "s2": None, "s1_asc": None, "s1_desc": None})
                for k in ("s2", "s1_asc", "s1_desc"):
                    if e.get(k):
                        tgt[k] = e[k]
                if e.get("s2"):
                    tgt["tile"], tgt["zone"] = e["tile"], e["zone"]
        out = []
        for sid, v in merged.items():
            start, end = _dates_from_path(
                v["s2"], v["s1_asc"], v["s1_desc"])
            out.append({
                "ref_id": ref_id, "sample_id": sid, "tile": v["tile"],
                "zone": v["zone"], "s2_path": v["s2"],
                "s1_asc": v["s1_asc"], "s1_desc": v["s1_desc"],
                "h3": None, "source": "fs",
                "start_date": start, "end_date": end,
            })
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, r): r for r in sorted(ref_dirs)}
        for i, fut in enumerate(as_completed(futs), 1):
            ref = futs[fut]
            try:
                got = fut.result()
            except Exception as exc:
                logger.error(f"{ref}: scan failed ({exc})")
                continue
            rows.extend(got)
            if i % 25 == 0 or i == len(futs):
                logger.info(f"[{i}/{len(futs)}] {len(rows):,} patches so far "
                            f"(last: {ref} +{len(got):,})")
    return pd.DataFrame(rows)


def footprints_from_stac(ref_ids: List[str], workers: int) -> pd.DataFrame:
    """(ref_id, sample_id) -> geometry, from the STAC item geometries."""
    from shapely.geometry import shape
    out: List[dict] = []

    def _one(ref_id: str) -> List[dict]:
        got: List[dict] = []
        for coll in ("sentinel2-patch-extraction",
                     "sentinel1-patch-extraction"):
            try:
                for feat in _iter_items(ref_id, coll):
                    props = feat.get("properties", {})
                    sid = props.get("sample_id") or feat.get("id")
                    geom = feat.get("geometry")
                    if sid and geom:
                        got.append({"ref_id": ref_id, "sample_id": str(sid),
                                    "geometry": shape(geom)})
                if got:
                    break          # S2 is enough; S1 shares the footprint
            except Exception as exc:
                logger.warning(f"{ref_id}/{coll}: STAC failed ({exc})")
        return got

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, r): r for r in ref_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out.extend(fut.result())
            except Exception:
                pass
            if i % 25 == 0 or i == len(futs):
                logger.info(f"[{i}/{len(futs)}] STAC footprints: {len(out):,}")
    if not out:
        return pd.DataFrame(columns=["ref_id", "sample_id", "geometry"])
    df = pd.DataFrame(out)
    return df.drop_duplicates(subset=["ref_id", "sample_id"])


def _footprint_from_nc(path: str) -> Optional[object]:
    """Bounding box of one patch .nc, in EPSG:4326, from its CRS + x/y axes."""
    try:
        import netCDF4
        import numpy as np
        from pyproj import CRS, Transformer
        with netCDF4.Dataset(path) as ds:
            if "x" not in ds.variables or "y" not in ds.variables:
                return None
            x = np.asarray(ds.variables["x"][:], dtype="float64")
            y = np.asarray(ds.variables["y"][:], dtype="float64")
            epsg = None
            for cand in ("crs", "spatial_ref", "transverse_mercator"):
                if cand in ds.variables:
                    v = ds.variables[cand]
                    for attr in ("epsg_code", "spatial_ref", "crs_wkt"):
                        if hasattr(v, attr):
                            try:
                                epsg = CRS.from_user_input(getattr(v, attr))
                                break
                            except Exception:
                                continue
                if epsg is not None:
                    break
            if epsg is None:
                return None
            # cell centres -> outer edge
            dx = float(abs(x[1] - x[0])) / 2 if len(x) > 1 else 5.0
            dy = float(abs(y[1] - y[0])) / 2 if len(y) > 1 else 5.0
            minx, maxx = float(x.min()) - dx, float(x.max()) + dx
            miny, maxy = float(y.min()) - dy, float(y.max()) + dy
            if epsg.to_epsg() != 4326:
                tf = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
                xs, ys = tf.transform([minx, maxx, minx, maxx],
                                      [miny, maxy, maxy, miny])
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
            return box(minx, miny, maxx, maxy)
    except Exception:
        return None


def _footprint_wkb(path: str) -> Optional[bytes]:
    """Process-pool worker: shapely geometries do not pickle cheaply, and WKB
    keeps the IPC payload small."""
    g = _footprint_from_nc(path)
    return None if g is None else g.wkb


def footprints_from_nc(df: pd.DataFrame, workers: int) -> pd.Series:
    """Footprint per row by opening each patch file. SLOW — hours at ~800k.

    Uses PROCESSES, not threads: the netCDF4/HDF5 stack is not thread-safe for
    concurrent opens and segfaults the interpreter under a ThreadPoolExecutor
    (reproduced at 16+ threads). Processes also sidestep the GIL, which matters
    because the CRS transform is CPU work, not just NFS I/O.
    """
    from concurrent.futures import ProcessPoolExecutor

    from shapely import wkb as _wkb

    paths = df["s2_path"].fillna(df["s1_asc"]).fillna(df["s1_desc"])
    geoms: List[Optional[object]] = [None] * len(paths)
    todo = [(i, p) for i, p in enumerate(paths) if isinstance(p, str)]
    logger.info(f"reading footprints from {len(todo):,} NetCDFs "
                f"with {workers} processes — this is the slow path")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # chunksize keeps IPC overhead well below the ~0.4 s per-file cost
        for (i, _), blob in zip(
                todo, pool.map(_footprint_wkb, [p for _, p in todo],
                               chunksize=64)):
            if blob is not None:
                geoms[i] = _wkb.loads(blob)
            done += 1
            if done % 20000 == 0 or done == len(todo):
                ok = sum(1 for g in geoms if g is not None)
                logger.info(f"[{done:,}/{len(todo):,}] footprints read, "
                            f"{ok:,} resolved")
    return pd.Series(geoms, index=df.index)


def cmd_scan(args: argparse.Namespace) -> None:
    campaign_refs: set = set()
    if args.refs_file and args.refs_file.exists():
        campaign_refs = {ln.split()[0] for ln in
                         args.refs_file.read_text().splitlines()
                         if ln.strip() and not ln.startswith("#")}
        logger.info(f"campaign list: {len(campaign_refs)} refs")

    df = walk_all(args.s2_root, args.s1_root, args.workers,
                  only_refs=args.only_refs)
    if df.empty:
        raise SystemExit("no patches found — check the roots")
    df["in_campaign"] = df.ref_id.isin(campaign_refs)
    logger.info(f"walk complete: {len(df):,} patches across "
                f"{df.ref_id.nunique()} refs "
                f"({int((~df.in_campaign).sum()):,} rows outside the "
                "campaign list)")

    if args.footprints == "stac":
        fp = footprints_from_stac(sorted(df.ref_id.unique()), args.workers)
        df = df.merge(fp, on=["ref_id", "sample_id"], how="left")
        df["footprint_source"] = df.geometry.notna().map(
            {True: "stac", False: None})
    else:
        df["geometry"] = footprints_from_nc(df, args.workers)
        df["footprint_source"] = df.geometry.notna().map(
            {True: "nc", False: None})

    n_fp = int(df.geometry.notna().sum())
    logger.info(f"footprints resolved: {n_fp:,}/{len(df):,} "
                f"({100 * n_fp / len(df):.1f}%)")
    n_dt = int(df.start_date.notna().sum())
    if n_dt:
        logger.info(f"temporal range parsed for {n_dt:,}/{len(df):,} patches: "
                    f"{df.start_date.min()} .. {df.end_date.max()}")
        # A patch whose window does not CONTAIN the year in its ref_id is the
        # case the ref name cannot tell you about. Windows normally straddle
        # the target year (~18 months centred on it), so testing the start or
        # end year for equality would flag ~89% of a healthy archive.
        yr = df.ref_id.str.slice(0, 4)
        odd = df.start_date.notna() & ~(
            (df.start_date.str.slice(0, 4) <= yr)
            & (yr <= df.end_date.str.slice(0, 4)))
        if int(odd.sum()):
            logger.warning(f"{int(odd.sum()):,} patches whose window does not "
                           "overlap the year in their ref_id")

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    for c in OUT_COLUMNS:
        if c not in gdf.columns:
            gdf[c] = None
    gdf = gdf[OUT_COLUMNS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(args.out)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_patches": int(len(gdf)),
        "n_refs": int(gdf.ref_id.nunique()),
        "n_outside_campaign": int((~gdf.in_campaign).sum()),
        "footprints": args.footprints,
        "n_footprints": n_fp,
        "n_with_dates": int(gdf.start_date.notna().sum()),
        "date_min": (None if gdf.start_date.isna().all()
                     else str(gdf.start_date.min())),
        "date_max": (None if gdf.end_date.isna().all()
                     else str(gdf.end_date.max())),
    }
    logger.success(f"wrote {args.out}: {json.dumps(meta)}")


def cmd_compare(args: argparse.Namespace) -> None:
    ext = gpd.read_parquet(args.extended)
    cam = gpd.read_parquet(args.campaign)
    ke = set(zip(ext.ref_id, ext.sample_id))
    kc = set(zip(cam.ref_id, cam.sample_id))
    print(f"extended : {len(ext):,} patches / {ext.ref_id.nunique()} refs")
    print(f"campaign : {len(cam):,} patches / {cam.ref_id.nunique()} refs")
    print(f"in both  : {len(ke & kc):,}")
    print(f"NEW in extended: {len(ke - kc):,}")
    print(f"only in campaign (missing from disk walk): {len(kc - ke):,}")
    new_refs = sorted(set(ext.ref_id) - set(cam.ref_id))
    print(f"\nrefs only in extended ({len(new_refs)}):")
    sub = ext[ext.ref_id.isin(new_refs)]
    for r, n in sub.ref_id.value_counts().items():
        fp = int(sub[sub.ref_id == r].geometry.notna().sum())
        print(f"   {n:>7,} patches ({fp:,} with footprint)  {r}")
    lost = sorted(set(cam.ref_id) - set(ext.ref_id))
    if lost:
        print(f"\nrefs in campaign index but NOT on disk ({len(lost)}):")
        for r in lost[:20]:
            print("   ", r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref-ids", nargs="+")
    src.add_argument("--ref-ids-file", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--cache-dir", type=Path, default=None)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--reconcile-sample", type=int, default=0,
                   help="stat() N random referenced S2 files per ref to "
                        "quantify STAC/disk drift (0 = off)")

    i = sub.add_parser("info")
    i.add_argument("--index", type=Path, required=True)

    c = sub.add_parser("coverage")
    c.add_argument("--index", type=Path, required=True)
    c.add_argument("--points", type=Path, required=True,
                   help="geoparquet or gpkg with point geometries")

    s = sub.add_parser("scan",
                       help="EXHAUSTIVE disk walk: every patch on disk, "
                            "whether or not it belongs to a campaign ref")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--s2-root", type=Path, default=S2_ROOT)
    s.add_argument("--s1-root", type=Path, default=S1_ROOT)
    s.add_argument("--refs-file", type=Path,
                   default=Path(__file__).resolve().parent
                   / "rdm_campaign_refs.txt",
                   help="only used to flag in_campaign; never filters")
    s.add_argument("--only-refs", nargs="+", help="restrict the walk (debug)")
    s.add_argument("--footprints", choices=["stac", "nc"], default="stac",
                   help="stac = fast, needs STAC to know the ref; "
                        "nc = exhaustive but opens every file (hours)")
    s.add_argument("--workers", type=int, default=8)

    cm = sub.add_parser("compare",
                        help="what the exhaustive scan adds over a build index")
    cm.add_argument("--extended", type=Path, required=True)
    cm.add_argument("--campaign", type=Path, required=True)

    args = ap.parse_args()

    if args.cmd == "scan":
        return cmd_scan(args)
    if args.cmd == "compare":
        return cmd_compare(args)



    if args.cmd == "build":
        refs = (args.ref_ids if args.ref_ids else
                [ln.split()[0] for ln in
                 args.ref_ids_file.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")])
        build_index(refs, args.out, cache_dir=args.cache_dir,
                    workers=args.workers,
                    reconcile_sample=args.reconcile_sample)
    elif args.cmd == "info":
        idx = gpd.read_parquet(args.index)
        meta = read_index_meta(args.index)
        print(json.dumps(meta, indent=2))
        per_ref = idx.groupby("ref_id").agg(
            patches=("sample_id", "size"),
            with_s2=("s2_path", lambda s: int(s.notna().sum())),
            with_footprint=("geometry", lambda s: int(s.notna().sum())))
        print(per_ref.to_string())
    elif args.cmd == "coverage":
        idx = gpd.read_parquet(args.index)
        pts = (gpd.read_parquet(args.points)
               if args.points.suffix in (".parquet", ".geoparquet")
               else gpd.read_file(args.points))
        cov = coverage(idx, pts)
        n0 = int((cov.n_patches == 0).sum())
        print(cov.to_string())
        print(f"\n{len(cov)} points, {n0} uncovered "
              f"({100 * n0 / max(len(cov), 1):.1f}%)")


if __name__ == "__main__":
    main()
