"""RefCatalog: fast per-ref patch index for local patch-to-point.

Answers, for one ref_id, the two questions every local extraction needs:

  1. WHERE are the patch files?   sample_id -> S2 .nc path, S1 paths per orbit
  2. WHAT do they cover?          sample_id -> footprint polygon (EPSG:4326)

Two sources, same output shape:

  * "stac" (default): paginate the S1/S2 patch-extraction STAC collections
    over HTTP (~5-30 s per ref, zero NFS traffic). Asset hrefs in these
    collections ARE local /data paths, and items carry the footprint geometry
    — which the collateral-assignment STRtree needs anyway. Dangling items
    (catalogue entries whose file is missing on disk) are tolerated: the
    extractor already degrades per-file at open time.
  * "fs": walk <root>/<ref>/<zone>/<tile>/<sample_id>/*.nc with os.scandir
    (1-4 min per ref on NFS). Ground truth for what is actually on disk;
    used as fallback and by `reconcile` to quantify STAC/disk drift.

The returned entries are a SUPERSET of ptp_engine.index_patches()
output — {sid: {"tile","zone","s2","s1":{orbit:path}}} plus "footprint" and
"h3" — so existing consumers can switch by replacing the index_patches call.

CLI:
  python ref_catalog.py --ref-ids 2022_BGR_Eurocrops_POLY_110 --compare-fs
  python ref_catalog.py --ref-ids-file refs.txt --cache-dir _CATALOG
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger
from shapely import wkb as shapely_wkb
from shapely.geometry import shape
from shapely.strtree import STRtree

STAC_SEARCH = "https://stac.openeo.vito.be/search"
COLLECTIONS = {
    "s2": "worldcereal_sentinel_2_patch_extractions",
    "s1": "worldcereal_sentinel_1_patch_extractions",
}
S2_ROOT = Path("/data/worldcereal_data/EXTRACTIONS/SENTINEL_2")
S1_ROOT = Path("/data/worldcereal_data/EXTRACTIONS/SENTINEL_1")


# --- STAC source ----------------------------------------------------------


def _iter_items(ref_id: str, collection: str, page_size: int = 500):
    """Yield STAC features for a ref_id, following `next` links."""
    token = None
    while True:
        body = {
            "collections": [collection],
            "limit": page_size,
            "query": {"ref_id": {"eq": ref_id}},
            "fields": {"include": [
                "properties.sample_id", "properties.s2_tile",
                "properties.h3_l3_cell", "properties.sat:orbit_state",
                "geometry", "assets",
            ]},
        }
        if token:
            body["token"] = token
        # Transient 5xx happen on this server; a single blip must not dump
        # callers to the minutes-long filesystem walk (or worse, a
        # footprint-less catalog). Retry with backoff before giving up.
        last_exc: Optional[requests.RequestException] = None
        for attempt in range(4):
            try:
                resp = requests.post(STAC_SEARCH, json=body, timeout=300)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise  # our request is wrong; retrying won't help
                import time as _time
                _time.sleep(2 ** attempt)
        else:
            assert last_exc is not None  # loop only falls through after raises
            raise last_exc
        payload = resp.json()
        yield from payload.get("features", [])
        token = None
        for link in payload.get("links", []):
            if link.get("rel") == "next":
                token = (link.get("body", {}).get("token")
                         or link.get("href", "").split("token=")[-1])
        if not token:
            break


def _href_to_path(feature: dict) -> Optional[Path]:
    for asset in feature.get("assets", {}).values():
        href = str(asset.get("href", "")).replace("file://", "")
        if href.startswith("/"):
            return Path(href)
    return None


def _zone_tile_from_path(path: Path, ref_id: str) -> tuple:
    """Layout <root>/<ref>/<zone>/<tile>/<sid>/<file>.nc -> (zone, tile)."""
    parts = path.parts
    try:
        i = parts.index(ref_id)
        return parts[i + 1], parts[i + 2]
    except (ValueError, IndexError):
        return None, None


def build_from_stac(ref_id: str) -> Dict[str, dict]:
    entries: Dict[str, dict] = {}
    for feature in _iter_items(ref_id, COLLECTIONS["s2"]):
        props = feature.get("properties", {})
        sid = props.get("sample_id")
        path = _href_to_path(feature)
        if not sid or path is None:
            continue
        zone, tile = _zone_tile_from_path(path, ref_id)
        entries[sid] = {
            "tile": tile or props.get("s2_tile"),
            "zone": zone,
            "s2": path,
            "s1": {},
            "h3": props.get("h3_l3_cell"),
            "footprint": shape(feature["geometry"]) if feature.get("geometry") else None,
        }
    for feature in _iter_items(ref_id, COLLECTIONS["s1"]):
        props = feature.get("properties", {})
        sid = props.get("sample_id")
        orbit = props.get("sat:orbit_state")
        path = _href_to_path(feature)
        if not sid or path is None or orbit is None:
            continue
        entry = entries.get(sid)
        if entry is None:
            # S1-only sample (no S2 patch): record it anyway; extraction
            # requires S2, so consumers will skip it, but reconcile() and
            # integrity tooling want to see it.
            zone, tile = _zone_tile_from_path(path, ref_id)
            entry = entries.setdefault(sid, {
                "tile": tile, "zone": zone, "s2": None, "s1": {},
                "h3": props.get("h3_l3_cell"), "footprint": None,
            })
        entry["s1"][orbit.upper()] = path

    # Patch STAC incompleteness: some samples have their S2 .nc on disk but no
    # item in the S2 collection (observed: 165/3236 = 5% on 2022_BGR). Their
    # S1 items reveal the exact directory (same <zone>/<tile>/<sid> layout,
    # SENTINEL_1 -> SENTINEL_2), so a handful of targeted globs recovers them
    # without a tree walk.
    patched = 0
    for sid, entry in entries.items():
        if entry["s2"] is not None or not entry["s1"]:
            continue
        s1_path = next(iter(entry["s1"].values()))
        s2_dir = Path(str(s1_path.parent).replace("SENTINEL_1", "SENTINEL_2"))
        try:
            nc = next(s2_dir.glob("*.nc"), None)
        except OSError:
            nc = None
        if nc is not None:
            entry["s2"] = nc
            patched += 1
    if patched:
        logger.info(f"{ref_id}: recovered {patched} S2 path(s) missing from "
                    "the S2 STAC collection via their S1 item directories")
    return entries


# --- Filesystem source ----------------------------------------------------


def build_from_fs(ref_id: str, s2_root: Path = S2_ROOT,
                  s1_root: Path = S1_ROOT) -> Dict[str, dict]:
    """os.scandir walk; DirEntry.is_dir() uses readdir type info, avoiding a
    stat per entry on NFS (the pathlib.iterdir walk this replaces did ~2x the
    RPCs)."""
    entries: Dict[str, dict] = {}
    for root, key in ((s2_root, "s2"), (s1_root, "s1")):
        base = root / ref_id
        if not base.exists():
            continue
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
                                entry = entries.setdefault(sdir.name, {
                                    "tile": tile.name, "zone": zone.name,
                                    "s2": None, "s1": {},
                                    "h3": None, "footprint": None,
                                })
                                with os.scandir(sdir.path) as files:
                                    for f in files:
                                        if not f.name.endswith(".nc"):
                                            continue
                                        if key == "s2":
                                            entry["s2"] = Path(f.path)
                                            entry["tile"] = tile.name
                                            entry["zone"] = zone.name
                                        else:
                                            orbit = ("ASCENDING"
                                                     if "_ASCENDING_" in f.name
                                                     else "DESCENDING")
                                            entry["s1"][orbit] = Path(f.path)
    return entries


# --- Catalog --------------------------------------------------------------


class RefCatalog:
    """Per-ref patch index with lazy footprint STRtree and parquet cache."""

    def __init__(self, ref_id: str, entries: Dict[str, dict], source: str):
        self.ref_id = ref_id
        self.entries = entries
        self.source = source
        self._tree = None
        self._tree_sids: List[str] = []

    # -- construction

    @classmethod
    def load(cls, ref_id: str, source: str = "auto",
             cache_dir: Optional[Path] = None) -> "RefCatalog":
        """source: "stac" | "fs" | "auto" (stac, fs fallback if stac empty).
        With cache_dir, a previously saved catalog is reused and new builds
        are saved there."""
        if cache_dir is not None:
            cached = Path(cache_dir) / f"{ref_id}.catalog.parquet"
            if cached.exists():
                cat = cls.from_parquet(ref_id, cached)
                has_fp = any(e.get("footprint") is not None
                             for e in cat.entries.values())
                if has_fp or source == "fs":
                    return cat
                logger.warning(f"{ref_id}: cached catalog has no footprints "
                               "(old fs build?); rebuilding from STAC")
        entries, used = {}, source
        if source in ("stac", "auto"):
            try:
                entries = build_from_stac(ref_id)
                used = "stac"
            except requests.RequestException as exc:
                logger.warning(f"{ref_id}: STAC unavailable ({exc}); "
                               "falling back to filesystem walk")
        if not entries and source in ("fs", "auto"):
            entries = build_from_fs(ref_id)
            used = "fs"
        cat = cls(ref_id, entries, used)
        # Never cache fs-built catalogs: they have no footprints, and a cached
        # footprint-less catalog would poison every future run of this ref.
        if cache_dir is not None and entries and used == "stac":
            cat.to_parquet(Path(cache_dir) / f"{ref_id}.catalog.parquet")
        return cat

    # -- spatial

    def strtree(self):
        """STRtree over footprints (4326). Only STAC-built catalogs have
        footprints; returns (tree, [sample_id per tree geometry])."""
        if self._tree is None:
            geoms, sids = [], []
            for sid, e in self.entries.items():
                if e.get("footprint") is not None:
                    geoms.append(e["footprint"])
                    sids.append(sid)
            self._tree = STRtree(geoms) if geoms else STRtree([])
            self._tree_sids = sids
        return self._tree, self._tree_sids

    def covering(self, point) -> List[str]:
        """sample_ids of patches whose footprint intersects `point` (4326)."""
        tree, sids = self.strtree()
        return [sids[int(i)] for i in tree.query(point, predicate="intersects")]

    # -- persistence

    def to_parquet(self, path: Path) -> None:
        rows = []
        for sid, e in self.entries.items():
            rows.append({
                "sample_id": sid, "tile": e.get("tile"), "zone": e.get("zone"),
                "s2_path": str(e["s2"]) if e.get("s2") else None,
                "s1_asc": str(e["s1"].get("ASCENDING")) if e.get("s1", {}).get("ASCENDING") else None,
                "s1_desc": str(e["s1"].get("DESCENDING")) if e.get("s1", {}).get("DESCENDING") else None,
                "h3": e.get("h3"),
                "footprint_wkb": (e["footprint"].wkb
                                  if e.get("footprint") is not None else None),
                "source": self.source,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, ref_id: str, path: Path) -> "RefCatalog":
        df = pd.read_parquet(path)

        def _str(v):
            # parquet round-trips None as NaN (a truthy float!) — guard hard
            return v if isinstance(v, str) and v else None

        entries: Dict[str, dict] = {}
        for r in df.itertuples():
            s1 = {}
            if _str(r.s1_asc):
                s1["ASCENDING"] = Path(r.s1_asc)
            if _str(r.s1_desc):
                s1["DESCENDING"] = Path(r.s1_desc)
            fp = r.footprint_wkb
            entries[r.sample_id] = {
                "tile": _str(r.tile), "zone": _str(r.zone),
                "s2": Path(r.s2_path) if _str(r.s2_path) else None,
                "s1": s1, "h3": _str(r.h3),
                "footprint": (shapely_wkb.loads(bytes(fp))
                              if isinstance(fp, (bytes, bytearray)) else None),
            }
        source = df["source"].iloc[0] if len(df) else "cache"
        return cls(ref_id, entries, f"cache({source})")

    # -- diagnostics

    def reconcile(self, stat_workers: int = 16) -> dict:
        """Compare this (STAC-built) catalog against the filesystem: which
        referenced files are missing on disk (dangling), per sensor."""
        paths = []
        for sid, e in self.entries.items():
            if e.get("s2"):
                paths.append(("s2", sid, e["s2"]))
            for orbit, p in e.get("s1", {}).items():
                paths.append((f"s1_{orbit[:3].lower()}", sid, p))
        with ThreadPoolExecutor(max_workers=stat_workers) as pool:
            exists = list(pool.map(lambda t: t[2].exists(), paths))
        missing = [(k, sid, str(p)) for (k, sid, p), ok in zip(paths, exists)
                   if not ok]
        out = {"ref_id": self.ref_id, "checked": len(paths),
               "missing": len(missing),
               "missing_by_kind": dict(pd.Series([m[0] for m in missing])
                                       .value_counts()) if missing else {},
               "examples": missing[:5]}
        return out


# --- CLI ------------------------------------------------------------------


def _compare_with_fs(ref_id: str) -> None:
    """Parity check: STAC-built catalog vs filesystem walk (the current
    index_patches behaviour). Proves drop-in-ness per ref."""
    import time
    t0 = time.time()
    stac = build_from_stac(ref_id)
    t1 = time.time()
    fs = build_from_fs(ref_id)
    t2 = time.time()
    logger.info(f"{ref_id}: STAC {len(stac)} entries in {t1-t0:.1f}s | "
                f"FS {len(fs)} entries in {t2-t1:.1f}s")
    s_only = set(stac) - set(fs)
    f_only = set(fs) - set(stac)
    both = set(stac) & set(fs)
    path_mismatch = orbit_mismatch = 0
    for sid in both:
        if str(stac[sid].get("s2")) != str(fs[sid].get("s2")):
            path_mismatch += 1
        if set(stac[sid]["s1"]) != set(fs[sid]["s1"]):
            orbit_mismatch += 1
    n_fp = sum(1 for e in stac.values() if e.get("footprint") is not None)
    print(f"{ref_id}:")
    print(f"  common={len(both)}  stac-only={len(s_only)} (dangling risk)  "
          f"fs-only={len(f_only)} (missing from STAC)")
    print(f"  s2-path mismatches={path_mismatch}  s1-orbit-set mismatches={orbit_mismatch}")
    print(f"  footprints present: {n_fp}/{len(stac)}")
    if s_only:
        print(f"  stac-only examples: {sorted(s_only)[:3]}")
    if f_only:
        print(f"  fs-only examples: {sorted(f_only)[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref-ids", nargs="+")
    src.add_argument("--ref-ids-file", type=str)
    ap.add_argument("--source", choices=["stac", "fs", "auto"], default="stac")
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="Write/reuse <ref>.catalog.parquet files here.")
    ap.add_argument("--compare-fs", action="store_true",
                    help="Parity check STAC vs filesystem walk per ref.")
    ap.add_argument("--reconcile", action="store_true",
                    help="Report catalog entries whose files are missing on disk.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Refs processed in parallel (cache builds only).")
    args = ap.parse_args()

    refs = (args.ref_ids if args.ref_ids else
            [line.strip() for line in Path(args.ref_ids_file).read_text().splitlines()
             if line.strip() and not line.startswith("#")])

    if args.compare_fs:
        for ref in refs:
            _compare_with_fs(ref)
        return

    def _one(ref):
        cat = RefCatalog.load(ref, source=args.source,
                              cache_dir=Path(args.cache_dir) if args.cache_dir else None)
        line = f"{ref}: {len(cat.entries)} entries via {cat.source}"
        if args.reconcile:
            rec = cat.reconcile()
            line += (f" | dangling: {rec['missing']}/{rec['checked']}"
                     f" {rec['missing_by_kind'] or ''}")
        return line

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_one, refs):
            print(line)


if __name__ == "__main__":
    main()
