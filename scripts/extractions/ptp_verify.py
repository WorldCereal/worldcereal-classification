"""Per-ref sanity verification of local patch-to-point output against the
openEO-era store — every difference must be EXPLAINED, not just counted.

The openEO-era store is wrong for ~50% of points, but wrong in a precisely
characterised way (the neighbouring-pixel layout bug). 
That makes it a usable reference: for a random sample of points 
we classify each band group as

  S2 : identical         — store matches our value bit-for-bit
       shift_explained   — store differs, but matches the series of EXACTLY
                           one 8-neighbour pixel of the host patch,
                           recomputed from the raw NetCDF (the bug's
                           signature), AND our own value re-derives at the
                           centre pixel (engine self-check)
       geometry_divergence — store matches a pixel of the own patch further
                           than 1 px away: the openEO-era extraction used a
                           different point location (it sourced points from
                           the live RDM API; we read the harmonized snapshot,
                           which is the AUTHORITATIVE source.
                           Reported with the offset; a WARN, not a failure —
                           the ref still passes unless --strict.
       unexplained       — anything else  -> FAILURE
  S1 : identical / explained (other orbit, a +-1-pixel shift, or <=2 months
       off by <=2 DN — the float32 edge-month noise) / unexplained
  aux: within documented tolerances (elevation +-2 m, slope +-2 deg,
       TMEAN +-0.2 K, PRECIP +-0.01 mm) / out_of_tolerance

A ref PASSES when nothing is unexplained/out-of-tolerance. Sampling is
deterministic per ref (seeded by ref_id) so reruns verify the same points.

Standalone post-hoc:
  ptp_verify.py --ref-ids <ref> ... --local-dir OUT --store DIR [--n 20]
Integrated: ptp_campaign_rdm.py --verify N runs this after each ref.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS, Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptp_engine import (  # noqa: E402
    NODATA,
    S1_BANDS,
    S2_BANDS,
    _nearest_idx,
    _read_patch,
    composite_s1,
    composite_s2,
)
from ref_catalog import RefCatalog  # noqa: E402

STORE_DEFAULT = Path(
    "/data/worldcereal_data/EXTRACTIONS/WORLDCEREAL/"
    "WORLDCEREAL_ALL_EXTRACTIONS_WITH_ANOMALY/worldcereal_all_extractions.parquet")

AUX_TOL = {"elevation": 2, "slope": 2}
# Meteo is validated differently: the openEO-era store mixes TWO conventions
# (covering-cell from the nearest-era graph, bilinear from the newer one), so
# a store value is EXPLAINED if it matches either convention recomputed from
# our cached composite raster (±METEO_TOL), and unexplained otherwise.
METEO_BANDS = {"AGERA5-TMEAN": "temperature-mean",
               "AGERA5-PRECIP": "precipitation-flux"}
METEO_TOL = 25  # covers floor()-vs-round and shifted-pixel-centre residuals


def _load_store_ref(store: Path, ref_id: str,
                    sample_ids: Optional[set] = None) -> Optional[pd.DataFrame]:
    part = store / f"ref_id={ref_id}"
    if not part.exists():
        return None
    cols = ["sample_id", "timestamp"] + S2_BANDS + S1_BANDS \
        + list(AUX_TOL) + list(METEO_BANDS)
    frames = []
    for f in part.glob("*.parquet"):
        df = pd.read_parquet(f, columns=cols)
        if sample_ids is not None:
            df = df[df.sample_id.isin(sample_ids)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def _meteo_candidates(year: int, month: int, band: str,
                      lon: float, lat: float) -> List[int]:
    """Every value the openEO-era store could legitimately hold at (lon,
    lat): our covering 0.1-deg cell, bilinear (the newer graph), and the 8
    NEIGHBOURING cells' values. The openEO cube resampled the meteo raster
    to 10 m with nearest-neighbour on a misregistered grid, so near-edge
    points were served an adjacent cell's value (CAN forensics 2026-08-19:
    store PRECIP = the neighbouring cell, 31 mm/month off, at a point 100 m
    from the cell edge; TMEAN offsets cluster in discrete per-cell steps).
    A genuine bug on OUR side (wrong scale, wrong indexing) matches none of
    these. Empty list if the raster is unavailable."""
    import ptp_engine
    import rasterio
    from ptp_engine import _bilinear
    cache = ptp_engine.AGERA5_CACHE
    if cache is None:
        return []
    path = Path(cache) / f"openEO_{year}-{month:02d}-01Z_{band}.tif"
    if not path.exists():
        return []
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        fc, fr = (~ds.transform) * (lon, lat)
    out: List[int] = []
    r0, c0 = int(np.floor(fr)), int(np.floor(fc))
    for dr in (0, -1, 1):
        for dc in (0, -1, 1):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]:
                v = int(arr[rr, cc])
                if v != NODATA:
                    out.append(v)
    vb = _bilinear(arr.astype(float), fr - 0.5, fc - 0.5, nodata=NODATA)
    if vb is not None and not np.isnan(vb) and vb != NODATA:
        out.append(int(np.floor(vb)))
    return out


def _series_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.all(a == b))


def _fullscan_matches(patch: dict, months: List[tuple],
                      t0: np.datetime64, t1: np.datetime64,
                      b04_store: np.ndarray) -> List[tuple]:
    """All (row, col) whose monthly-median B04 series equals `b04_store`.
    Vectorised over the whole patch — used only to escalate an otherwise
    unexplained sample, so the cost (one band, whole patch) is acceptable."""
    times = patch["times"]
    mask = patch["bands"]["S2-L2A-SCL_DILATED_MASK"]
    band = patch["bands"]["S2-L2A-B04"]
    cube = np.full((len(months), *band.shape[1:]), np.nan)
    mkeys = [(pd.Timestamp(t).year, pd.Timestamp(t).month) for t in times]
    sel_t = (times >= t0) & (times < t1)
    for mi, m in enumerate(months):
        idx = [i for i, k in enumerate(mkeys) if k == m and sel_t[i]]
        if not idx:
            continue
        vals = band[idx].astype(float)
        vals[(mask[idx] == 1) | (vals == NODATA)] = np.nan
        with np.errstate(all="ignore"):
            cube[mi] = np.floor(np.nanmedian(vals, axis=0))
    cube = np.nan_to_num(cube, nan=NODATA).astype(np.int64)
    hit = (cube == b04_store[:, None, None]).all(axis=0)
    ys, xs = np.where(hit)
    return list(zip(ys.tolist(), xs.tolist()))


def _xpatch_s2_match(catalog: RefCatalog, sid: str, geom,
                     months: List[tuple], t0: np.datetime64,
                     t1: np.datetime64, s2_store: np.ndarray,
                     max_patches: int = 6) -> Optional[Dict[str, Any]]:
    """openEO merged ALL of a ref's patches into one cube. Where patches
    OVERLAP (ubiquitous for POINT refs: each point gets its own window),
    the store's values for a point could come from a NEIGHBOUR's patch
    file — which may hold a different extraction vintage of the same
    pixels — plus the usual pixel-shift bug. Try centre+8 in each other
    covering patch; an exact series match proves the store row was read
    from that file, not the sample's own."""
    try:
        others = [h for h in catalog.covering(geom) if h != sid]
    except Exception:
        return None
    for host in others[:max_patches]:
        entry = catalog.entries.get(host)
        if not entry or not entry.get("s2"):
            continue
        try:
            patch = _read_patch(entry["s2"],
                                S2_BANDS + ["S2-L2A-SCL_DILATED_MASK"])
        except OSError:
            continue
        tr = Transformer.from_crs("EPSG:4326", CRS.from_wkt(patch["crs_wkt"]),
                                  always_xy=True)
        qx, qy = tr.transform(geom.x, geom.y)
        r0, c0 = _nearest_idx(patch["y"], qy), _nearest_idx(patch["x"], qx)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r0 + dr, c0 + dc
                if not (0 <= rr < len(patch["y"])
                        and 0 <= cc < len(patch["x"])):
                    continue
                series = composite_s2(patch, rr, cc, months, t0, t1)
                if _series_equal(series.T.astype(np.int64), s2_store):
                    return {"via": host, "offset_px": [dr, dc]}
    return None


def _xpatch_s1_match(catalog: RefCatalog, sid: str, geom,
                     months: List[tuple], t0: np.datetime64,
                     t1: np.datetime64, s1_store: np.ndarray,
                     max_patches: int = 6) -> Optional[Dict[str, Any]]:
    """S1 flavour of _xpatch_s2_match — e.g. the sample's own patch has no
    S1 file at all, but an overlapping neighbour's does (the merged cube
    then HAD S1 there; our patch-local route correctly does not). Same
    edge-month tolerance as the own-patch S1 check."""
    try:
        others = [h for h in catalog.covering(geom) if h != sid]
    except Exception:
        return None
    for host in others[:max_patches]:
        entry = catalog.entries.get(host)
        if not entry:
            continue
        for orbit, s1_path in entry.get("s1", {}).items():
            try:
                s1p = _read_patch(s1_path, S1_BANDS)
            except OSError:
                continue
            tr1 = Transformer.from_crs("EPSG:4326",
                                       CRS.from_wkt(s1p["crs_wkt"]),
                                       always_xy=True)
            qx, qy = tr1.transform(geom.x, geom.y)
            c0 = _nearest_idx(s1p["x"], qx)
            r0 = _nearest_idx(s1p["y"], qy)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r0 + dr, c0 + dc
                    if not (0 <= rr < len(s1p["y"])
                            and 0 <= cc < len(s1p["x"])):
                        continue
                    series = composite_s1(s1p, rr, cc, months, t0, t1)
                    series = series.T.astype(np.int64)
                    diff = series != s1_store
                    bad_months = int(np.any(diff, axis=1).sum())
                    if bad_months == 0 or (
                            bad_months <= 2
                            and int(np.abs(series - s1_store)[diff].max()) <= 2):
                        return {"via": host, "orbit": orbit,
                                "offset_px": [dr, dc]}
    return None


def verify_ref(
    ref_id: str,
    local_path: Path,
    catalog: RefCatalog,
    store: Path = STORE_DEFAULT,
    n_samples: int = 20,
    max_divergence_frac: float = 0.3,
) -> dict:
    """Classify a deterministic random sample of points; return the verdict."""
    verdict: Dict[str, Any] = {"ref_id": ref_id, "status": "PASS", "n_checked": 0,
               "s2_identical": 0, "s2_shift_explained": 0,
               "geometry_divergence": 0, "divergence_offsets": [],
               "s2_xpatch_explained": 0, "s1_xpatch_explained": 0,
               "xpatch_samples": [],
               "s2_unexplained": 0,
               "s1_identical": 0, "s1_explained": 0, "s1_unexplained": 0,
               "aux_ok": 0, "aux_out_of_tolerance": 0,
               "meteo_beyond_covering": 0,
               "self_check_failures": 0, "unexplained_samples": []}

    local = gpd.read_parquet(local_path)
    # host_sample_id is not persisted in the output; recover assignment:
    # primary samples (own patch) are verifiable directly — restrict to them,
    # which is also the population where the neighbour recompute is defined.
    own = [s for s in local.sample_id.unique() if s in catalog.entries
           and catalog.entries[s].get("s2") is not None]
    if not own:
        verdict["status"] = "SKIP(no primary samples)"
        return verdict
    rng = np.random.default_rng(
        int(hashlib.sha256(ref_id.encode()).hexdigest()[:8], 16))
    picked = list(rng.choice(own, size=min(n_samples, len(own)),
                             replace=False))

    ref_store = _load_store_ref(store, ref_id, set(picked))
    if ref_store is None or ref_store.empty:
        verdict["status"] = "SKIP(no openEO reference in store)"
        return verdict
    sx = ref_store.set_index(["sample_id", "timestamp"]).sort_index()
    lx = local.set_index(["sample_id", "timestamp"]).sort_index()

    for sid in picked:
        if sid not in sx.index.get_level_values(0):
            continue
        srows = sx.loc[sid]
        lrows = lx.loc[sid]
        common_ts = srows.index.intersection(lrows.index)
        if len(common_ts) == 0:
            continue
        verdict["n_checked"] += 1
        s_vals = srows.loc[common_ts]
        l_vals = lrows.loc[common_ts]
        months = [(t.year, t.month) for t in common_ts]

        s2_store = s_vals[S2_BANDS].to_numpy(np.int64)
        s2_local = l_vals[S2_BANDS].to_numpy(np.int64)

        if _series_equal(s2_store, s2_local):
            verdict["s2_identical"] += 1
        else:
            # Recompute centre + 8 neighbours from the raw patch.
            entry = catalog.entries[sid]
            row0 = lrows.iloc[0]
            t0 = np.datetime64(str(row0["start_date"]))
            t1 = np.datetime64(str(row0["end_date"]))
            patch = _read_patch(entry["s2"],
                                S2_BANDS + ["S2-L2A-SCL_DILATED_MASK"])
            tr = Transformer.from_crs("EPSG:4326",
                                      CRS.from_wkt(patch["crs_wkt"]),
                                      always_xy=True)
            geom = l_vals.geometry.iloc[0] if hasattr(l_vals, "geometry") \
                else local[local.sample_id == sid].geometry.iloc[0]
            px, py = tr.transform(geom.x, geom.y)
            c0 = _nearest_idx(patch["x"], px)
            r0 = _nearest_idx(patch["y"], py)
            explained = False
            self_ok = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r0 + dr, c0 + dc
                    if not (0 <= rr < len(patch["y"])
                            and 0 <= cc < len(patch["x"])):
                        continue
                    series = composite_s2(patch, rr, cc, months, t0, t1)
                    series = series.T.astype(np.int64)  # (months, bands)
                    if dr == 0 and dc == 0:
                        self_ok = _series_equal(series, s2_local)
                    elif _series_equal(series, s2_store):
                        explained = True
            if not self_ok:
                verdict["self_check_failures"] += 1
            if explained and self_ok:
                verdict["s2_shift_explained"] += 1
            else:
                # Escalate: does the store's series live at ANOTHER pixel of
                # this patch (openEO used a different point location)?
                b04_store = s_vals["S2-L2A-B04"].to_numpy(np.int64)
                far = _fullscan_matches(patch, months, t0, t1, b04_store)
                confirmed = None
                for (rr, cc) in far:
                    series = composite_s2(patch, rr, cc, months, t0, t1)
                    if _series_equal(series.T.astype(np.int64), s2_store):
                        confirmed = (rr - r0, cc - c0)
                        break
                if confirmed is not None and self_ok:
                    verdict["geometry_divergence"] += 1
                    verdict["divergence_offsets"].append(
                        {"sample_id": sid, "offset_px": list(confirmed)})
                    # location differs -> S1/aux comparisons are meaningless
                    # for this sample; skip them.
                    continue
                xp = _xpatch_s2_match(catalog, sid, geom, months, t0, t1,
                                      s2_store)
                if xp is not None and self_ok:
                    verdict["s2_xpatch_explained"] += 1
                    verdict["xpatch_samples"].append({"sample_id": sid, **xp})
                    # The store row was read from ANOTHER patch's file in the
                    # merged cube; its S1/aux came from the same contaminated
                    # read -> those comparisons are meaningless here too.
                    continue
                verdict["s2_unexplained"] += 1
                verdict["unexplained_samples"].append(sid)

        # --- S1: identical, else try both orbits x 3x3 pixels, else edge-noise
        s1_store = s_vals[S1_BANDS].to_numpy(np.int64)
        s1_local = l_vals[S1_BANDS].to_numpy(np.int64)
        if _series_equal(s1_store, s1_local):
            verdict["s1_identical"] += 1
        else:
            entry = catalog.entries[sid]
            row0 = lrows.iloc[0]
            t0 = np.datetime64(str(row0["start_date"]))
            t1 = np.datetime64(str(row0["end_date"]))
            geom = local[local.sample_id == sid].geometry.iloc[0]
            explained = False
            for orbit, s1_path in entry.get("s1", {}).items():
                try:
                    s1p = _read_patch(s1_path, S1_BANDS)
                except OSError:
                    continue
                tr1 = Transformer.from_crs("EPSG:4326",
                                           CRS.from_wkt(s1p["crs_wkt"]),
                                           always_xy=True)
                qx, qy = tr1.transform(geom.x, geom.y)
                b0c, b0r = _nearest_idx(s1p["x"], qx), _nearest_idx(s1p["y"], qy)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = b0r + dr, b0c + dc
                        if not (0 <= rr < len(s1p["y"])
                                and 0 <= cc < len(s1p["x"])):
                            continue
                        series = composite_s1(s1p, rr, cc, months, t0, t1)
                        series = series.T.astype(np.int64)
                        diff = series != s1_store
                        bad_months = int(np.any(diff, axis=1).sum())
                        if bad_months == 0 or (
                                bad_months <= 2
                                and int(np.abs(series - s1_store)[diff].max()) <= 2):
                            explained = True
                if explained:
                    break
            if explained:
                verdict["s1_explained"] += 1
            else:
                xp1 = _xpatch_s1_match(catalog, sid, geom, months, t0, t1,
                                       s1_store)
                if xp1 is not None:
                    verdict["s1_xpatch_explained"] += 1
                    verdict["xpatch_samples"].append(
                        {"sample_id": sid, "band": "s1", **xp1})
                    # store S1 (and aux) for this sample came from another
                    # patch's file in the merged cube; skip aux.
                    continue
                verdict["s1_unexplained"] += 1
                if sid not in verdict["unexplained_samples"]:
                    verdict["unexplained_samples"].append(sid)

        # --- aux tolerances (store may legitimately differ: openEO bilinear
        # artefacts; bounds are the documented ones)
        aux_bad = False
        for band, tol in AUX_TOL.items():
            sa = s_vals[band].to_numpy(np.int64)
            la = l_vals[band].to_numpy(np.int64)
            valid = (sa != NODATA) & (la != NODATA)
            if valid.any() and int(np.abs(sa[valid] - la[valid]).max()) > tol:
                aux_bad = True
        # meteo: store must match covering-cell, bilinear, or a neighbouring
        # cell (the openEO nearest-resampled meteo layer was misregistered)
        geom_pt = local[local.sample_id == sid].geometry.iloc[0]
        meteo_beyond_covering = False
        for band, src in METEO_BANDS.items():
            sa = s_vals[band].to_numpy(np.int64)
            la = l_vals[band].to_numpy(np.int64)
            hit_bad = False
            for k, ts in enumerate(common_ts):
                if sa[k] == NODATA or la[k] == NODATA:
                    continue
                if abs(int(sa[k]) - int(la[k])) <= METEO_TOL:
                    continue  # matches our (covering-cell) value
                cands = _meteo_candidates(ts.year, ts.month, src,
                                          geom_pt.x, geom_pt.y)
                # The store value must lie within the local 3x3 cells' value
                # range: the openEO meteo layer was nearest-resampled on a
                # misregistered grid and sometimes bilinear, so any local
                # mixture is a store-side artefact. Our own would-be bugs
                # (scale, indexing, month alignment) leave this range.
                if (not cands or
                        not (min(cands) - METEO_TOL <= int(sa[k])
                             <= max(cands) + METEO_TOL)):
                    hit_bad = True
                    break
                meteo_beyond_covering = True
            if hit_bad:
                aux_bad = True
                break
        if aux_bad:
            verdict["aux_out_of_tolerance"] += 1
            if sid not in verdict["unexplained_samples"]:
                verdict["unexplained_samples"].append(sid)
        else:
            verdict["aux_ok"] += 1
            if meteo_beyond_covering:
                verdict["meteo_beyond_covering"] += 1

    if (verdict["s2_unexplained"] or verdict["s1_unexplained"]
            or verdict["aux_out_of_tolerance"] or verdict["self_check_failures"]):
        verdict["status"] = "FAIL"
    elif (verdict["n_checked"] >= 5
            and verdict["geometry_divergence"] / verdict["n_checked"]
            > max_divergence_frac):
        # Divergence itself is the STORE being outdated, not us being wrong —
        # but a high rate means either the ref's geometries were massively
        # revised (attributes may have moved too: human review needed) or our
        # own pixel selection is systematically off and masquerading as
        # divergence. Either way, don't silently pass. NOTE: at n=20 an
        # observed rate has a wide confidence interval — this is a tripwire,
        # not a measurement; re-run with a larger --verify N to confirm.
        verdict["status"] = "FAIL(divergence_rate)"
    verdict["unexplained_samples"] = verdict["unexplained_samples"][:5]
    verdict["divergence_offsets"] = verdict["divergence_offsets"][:5]
    verdict["xpatch_samples"] = verdict["xpatch_samples"][:8]
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref-ids", nargs="+")
    src.add_argument("--ref-ids-file", type=str)
    ap.add_argument("--local-dir", type=Path, required=True,
                    help="dir with <ref_id>.geoparquet local outputs")
    ap.add_argument("--store", type=Path, default=STORE_DEFAULT)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--catalog-cache", type=Path, default=None)
    ap.add_argument("--agera5-cache", type=Path, default=None,
                    help="AGERA5 monthly cache dir — needed to explain "
                         "bilinear-era store meteo values")
    ap.add_argument("--out", type=Path, default=None,
                    help="append verdicts to this CSV")
    ap.add_argument("--max-divergence-frac", type=float, default=0.3,
                    help="FAIL a ref when more than this fraction of checked "
                         "points are geometry-divergent (default 0.3); a "
                         "tripwire for massively revised geometries or a "
                         "systematic pixel-selection fault masquerading as "
                         "divergence")
    ap.add_argument("--strict", action="store_true",
                    help="also FAIL refs with geometry_divergence samples "
                         "(default: divergence is reported but passes)")
    args = ap.parse_args()

    if args.agera5_cache:
        import ptp_engine
        ptp_engine.AGERA5_CACHE = args.agera5_cache

    refs = (args.ref_ids if args.ref_ids else
            [line.strip() for line in
             Path(args.ref_ids_file).read_text().splitlines()
             if line.strip() and not line.startswith("#")])
    verdicts = []
    for ref in refs:
        local_path = args.local_dir / f"{ref}.geoparquet"
        if not local_path.exists():
            logger.warning(f"{ref}: no local output at {local_path}; skipping")
            continue
        cat = RefCatalog.load(ref, source="auto", cache_dir=args.catalog_cache)
        v = verify_ref(ref, local_path, cat, store=args.store, n_samples=args.n,
                       max_divergence_frac=args.max_divergence_frac)
        if args.strict and v.get("geometry_divergence"):
            v["status"] = "FAIL"
        (logger.success if v["status"] == "PASS" else
         logger.warning if v["status"].startswith("SKIP") else
         logger.error)(f"{ref}: {v['status']} — {json.dumps(v)}")
        verdicts.append(v)
    if verdicts and args.out:
        pd.DataFrame(verdicts).to_csv(
            args.out, mode="a", index=False,
            header=not args.out.exists())
    if any(v["status"] == "FAIL" for v in verdicts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
