"""Native-date (uncomposited) patch-to-point extraction.

Emits one row per sample holding the ACTUAL acquisitions rather than monthly or
dekadal composites, for models that consume irregular time series with an
explicit date encoding (TESSERA v2 uses a sinusoidal day-of-year encoding plus
a validity mask; its S2 branch takes the 10 bands below and its S1 branch VV/VH).

Two facts drive the schema:

  * S2 and S1 are different satellites, so their acquisition dates do not line
    up. Each sample therefore carries two independent time axes, `s2_date` and
    `s1_date`, and the band lists are aligned to their own axis.
  * Cloud masking is per pixel, so the number of valid S2 acquisitions varies
    per point (a handful under persistent cloud, ~100 in clear regions). Lists
    are ragged by design; there is no padding.

Meteo is sampled from the AGERA5 DAILY archive on every acquisition date of
both sensors: `AGERA5-TMEAN_s2` / `AGERA5-PRECIP_s2` align element-wise with
`s2_date`, and the `_s1` pair with `s1_date`. slope and elevation are static
and stay scalars.

With --spatial-context N each S2/S1 band element becomes a (2N+1)^2 array
(row-major, centre at index 2N(N+1)), so the column is list<list<uint16>>.
Validity is decided on the centre pixel so the sequence length is the same
across the window.

Output: one <host>.geoparquet per host, one row per sample.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import ptp_engine as E
import rasterio
from loguru import logger

AGERA5_DAILY_BANDS = {"AGERA5-TMEAN": "temperature-mean",
                      "AGERA5-PRECIP": "precipitation-flux"}
# Emitted once per sensor axis, e.g. AGERA5-TMEAN_s2 aligned with s2_date.
METEO_COLS = [f"{m}_{ax}" for ax in ("s2", "s1") for m in AGERA5_DAILY_BANDS]
# Band whose NODATA decides whether an acquisition counts as observed.
REF_BAND = "S2-L2A-B04"


class DailyMeteo:
    """AGERA5 daily archive, sampled in bulk. Value of the covering 0.1-deg cell.

    The daily rasters are global, 512x512-tiled GeoTIFFs. Requests are grouped
    by date and each date reads only the window spanning the requested points,
    instead of decoding the whole globe once per point per date.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or E.AGERA5_DAILY)
        self.missing: set = set()

    def _path(self, day, band: str) -> Path:
        stamp = day.strftime("%Y%m%d")
        # Both naming conventions: the archive renamed its files from
        # 11 June 2026 (see ptp_engine.agera5_daily_file).
        try:
            return E.agera5_daily_file(day.year, day.month, day.day, band)
        except FileNotFoundError:
            return self.root / f"{day.year}" / stamp / f"AgERA5_{band}_{stamp}.tif"

    def sample_many(self, requests: List[tuple], band: str) -> List[np.ndarray]:
        """requests: (lon, lat, dates) per series -> one uint16 array each."""
        from collections import defaultdict

        from rasterio.windows import Window

        out = [np.full(len(d), E.NODATA, dtype=np.uint16) for _, _, d in requests]
        by_day: Dict[object, List[Tuple[int, int]]] = defaultdict(list)
        for ri, (_, _, days) in enumerate(requests):
            for j, d in enumerate(days):
                by_day[d].append((ri, j))
        lons = np.array([r[0] for r in requests], dtype=np.float64)
        lats = np.array([r[1] for r in requests], dtype=np.float64)
        for day, idx in by_day.items():
            path = self._path(day, band)
            if not path.exists():
                self.missing.add(day)
                continue
            ri = np.fromiter((i for i, _ in idx), dtype=np.int64, count=len(idx))
            with rasterio.open(path) as ds:
                inv = ~ds.transform
                cc, rr = inv * (lons[ri], lats[ri])
                rr = np.floor(rr).astype(np.int64)
                cc = np.floor(cc).astype(np.int64)
                inside = (rr >= 0) & (rr < ds.height) & (cc >= 0) & (cc < ds.width)
                if not inside.any():
                    continue
                r0, r1 = rr[inside].min(), rr[inside].max()
                c0, c1 = cc[inside].min(), cc[inside].max()
                arr = ds.read(1, window=Window(int(c0), int(r0),
                                               int(c1 - c0 + 1), int(r1 - r0 + 1)))
            vals = np.full(len(idx), E.NODATA, dtype=np.uint16)
            vals[inside] = arr[rr[inside] - r0, cc[inside] - c0]
            for k, (i, j) in enumerate(idx):
                out[i][j] = vals[k]
        return out


def _doy(dates: np.ndarray) -> np.ndarray:
    """Day-of-year per acquisition (TESSERA's sinusoidal encoding input)."""
    if len(dates) == 0:
        return np.empty(0, dtype=np.int16)
    return (pd.to_datetime(dates).dayofyear.to_numpy().astype(np.int16))


def _valid_s2(patch: dict, row: int, col: int, t0, t1, s2_mask: str) -> np.ndarray:
    """Boolean per time step: in window, not cloud-masked, and observed."""
    times = patch["times"]
    sel = (times >= t0) & (times < t1)
    if s2_mask == "raw_scl":
        bad = np.isin(patch["bands"][E.SCL_RAW_BAND][:, row, col],
                      list(E.SCL_REJECT_CLASSES))
    else:
        bad = patch["bands"][E.SCL_DILATED_BAND][:, row, col] == 1
    ref = patch["bands"][REF_BAND][:, row, col]
    return sel & ~bad & (ref != E.NODATA)


def _valid_s1(s1: dict, row: int, col: int, t0, t1) -> np.ndarray:
    """Boolean per time step: in window and at least one polarisation observed."""
    times = s1["times"]
    ok = (times >= t0) & (times < t1)
    seen = np.zeros(len(times), dtype=bool)
    for b in E.S1_BANDS:
        if b in s1["bands"]:
            v = s1["bands"][b][:, row, col]
            seen |= (v != 0) & (v != E.NODATA)
    return ok & seen


def _empty(half: int):
    """Typed empty series, so a band with no acquisitions still reads back as
    uint16 rather than a null-typed column that will not concat."""
    return [] if half > 0 else np.empty(0, dtype=np.uint16)


def _series(patch: dict, bands: List[str], row: int, col: int,
            keep: np.ndarray, half: int) -> Dict[str, object]:
    """Per band, the values at the kept time steps (scalars, or windows).

    Values stay numpy uint16 rather than Python lists: pyarrow infers int64
    from a list of Python ints, which would quadruple the stored size.
    """
    out: Dict[str, object] = {}
    for b in bands:
        if b not in patch["bands"]:
            out[b] = _empty(half)
            continue
        if half > 0:
            w = E._window_slice(patch["bands"][b], row, col, half)[keep]
            out[b] = [w[i].reshape(-1).astype(np.uint16)
                      for i in range(w.shape[0])]
        else:
            out[b] = patch["bands"][b][keep, row, col].astype(np.uint16)
    return out


def process_patch_native(task: dict) -> List[dict]:
    """Native-date series for every point hosted by one patch (worker process)."""
    conv = task["conventions"]
    half = int(conv.get("spatial_context", 0) or 0)
    s2_mask = conv.get("s2_mask", "dilated")
    t0 = np.datetime64(task["t_start"])
    t1 = np.datetime64(task["t_end_excl"])
    from pyproj import CRS, Transformer

    try:
        s2p = E._read_patch(task["s2_path"], E.S2_BANDS + [
            E.SCL_RAW_BAND if s2_mask == "raw_scl" else E.SCL_DILATED_BAND])
    except OSError as exc:
        logger.warning(f"S2 patch unreadable, dropping "
                       f"{len(task['points'])} point(s): {exc}")
        return []
    s2_crs = CRS.from_wkt(s2p["crs_wkt"])
    to_s2 = Transformer.from_crs("EPSG:4326", s2_crs, always_xy=True)

    s1_patches = []
    for orbit, p in task["s1_paths"].items():
        if not p or not Path(p).exists():
            continue
        try:
            s1p = E._read_patch(p, E.S1_BANDS)
        except OSError:
            continue
        s1_patches.append((orbit, s1p,
                           Transformer.from_crs(s2_crs, CRS.from_wkt(s1p["crs_wkt"]),
                                                always_xy=True)))
    s1_patches.sort(key=lambda kv: -Path(task["s1_paths"][kv[0]]).stat().st_size)

    recs = []
    for sample_id, lon, lat in task["points"]:
        px, py = to_s2.transform(lon, lat)
        col = E._nearest_idx(s2p["x"], px)
        row = E._nearest_idx(s2p["y"], py)
        if not (0 <= row < len(s2p["y"]) and 0 <= col < len(s2p["x"])):
            continue
        keep = _valid_s2(s2p, row, col, t0, t1, s2_mask)
        rec = {"sample_id": sample_id, "tile": task["tile"],
               "s2_date": [pd.Timestamp(d).date()
                           for d in s2p["times"][keep]],
               "s2_pixel_xy": (float(s2p["x"][col]), float(s2p["y"][row])),
               "s2_crs_wkt": s2p["crs_wkt"], "s1_orbit": None}
        rec.update(_series(s2p, E.S2_BANDS, row, col, keep, half))

        # S1: pick the orbit with the most valid acquisitions at this pixel.
        best = None
        for orbit, s1p, tf in s1_patches:
            qx, qy = tf.transform(*rec["s2_pixel_xy"])
            r = E._nearest_idx(s1p["y"], qy)
            c = E._nearest_idx(s1p["x"], qx)
            if not (0 <= r < len(s1p["y"]) and 0 <= c < len(s1p["x"])):
                continue
            k = _valid_s1(s1p, r, c, t0, t1)
            if best is None or int(k.sum()) > best[0]:
                best = (int(k.sum()), orbit, s1p, r, c, k)
        if best is None:
            rec["s1_date"] = []
            for b in E.S1_BANDS:
                rec[b] = _empty(half)
        else:
            _, orbit, s1p, r, c, k = best
            rec["s1_orbit"] = orbit
            rec["s1_date"] = [pd.Timestamp(d).date()
                              for d in s1p["times"][k]]
            rec.update(_series(s1p, E.S1_BANDS, r, c, k, half))
        recs.append(rec)
    return recs


def extract_host_native(
    host_ref_id: str,
    conventions: dict,
    points: gpd.GeoDataFrame,
    index: Dict[str, dict],
    out_path: Optional[Path] = None,
    workers: int = 2,
) -> pd.DataFrame:
    """One row per sample, with per-acquisition lists. Mirrors extract_host."""
    from concurrent.futures import ProcessPoolExecutor

    needed = set(points.host_sample_id)
    axes = E.month_axis_from_patches(index, needed, "month")
    tasks = []
    for hsid, grp in points.groupby("host_sample_id"):
        e = index.get(hsid)
        if not e or not e.get("s2"):
            continue
        axis = axes.get(e["zone"]) or next(iter(axes.values()))
        tasks.append({
            "s2_path": str(e["s2"]),
            "s1_paths": {o: str(p) for o, p in e["s1"].items()},
            "tile": e["tile"], "zone": e["zone"],
            "points": [(r.sample_id, r.geometry.x, r.geometry.y)
                       for r in grp.itertuples()],
            "t_start": axis["start"], "t_end_excl": axis["end"],
            "conventions": conventions,
        })
    logger.info(f"{host_ref_id}: {len(points)} points in {len(tasks)} patches "
                "(native dates)")
    records: List[dict] = []
    if workers <= 1:
        for t in tasks:
            records.extend(process_patch_native(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for out in pool.map(process_patch_native, tasks, chunksize=2):
                records.extend(out)
    # Same rule as extract_host's all-nodata drop: a point with no valid
    # acquisition from either sensor carries no signal. Keeps the sample set
    # identical to the composite stores.
    n0 = len(records)
    records = [r for r in records if len(r["s2_date"]) or len(r["s1_date"])]
    if len(records) < n0:
        logger.warning(f"{host_ref_id}: dropping {n0 - len(records)} sample(s) "
                       "with no valid S2 or S1 acquisition")
    if not records:
        logger.warning(f"{host_ref_id}: no records")
        return pd.DataFrame()

    meteo = DailyMeteo()
    slope_s, elev_s = E.SlopeSampler(), E.ElevationSampler()
    attrs = points.set_index("sample_id")
    geom = dict(zip(points.sample_id, points.geometry))
    from pyproj import CRS, Transformer
    tfc: Dict[str, Transformer] = {}

    rows = []
    centres: List[Tuple[float, float]] = []
    for rec in records:
        sid = rec["sample_id"]
        wkt = rec["s2_crs_wkt"]
        if wkt not in tfc:
            tfc[wkt] = Transformer.from_crs(CRS.from_wkt(wkt), "EPSG:4326",
                                            always_xy=True)
        clon, clat = tfc[wkt].transform(*rec["s2_pixel_xy"])
        dates = rec["s2_date"]
        a = attrs.loc[sid]
        row = {
            "sample_id": sid,
            "ref_id": host_ref_id,
            "host_sample_id": a.get("host_sample_id"),
            "tile": rec["tile"],
            "s1_orbit": rec["s1_orbit"] or "none",
            "lon": geom[sid].x, "lat": geom[sid].y,
            "geometry": geom[sid],
            "s2_date": dates,
            "s2_doy": _doy(dates),
            "n_s2": int(len(dates)),
            "s1_date": rec["s1_date"],
            "s1_doy": _doy(rec["s1_date"]),
            "n_s1": int(len(rec["s1_date"])),
            # Filled below, grouped by tile; placed here to keep column order
            # identical to parts written before the grouping was added.
            "slope": None,
            "elevation": None,
        }
        for b in E.S2_BANDS + E.S1_BANDS:
            row[b] = rec.get(b, [])
        centres.append((clon, clat))
        for c in ("ewoc_code", "valid_time", "h3_l3_cell", "irrigation_status",
                  "quality_score_lc", "quality_score_ct", "extract"):
            if c in attrs.columns:
                row[c] = a[c]
        rows.append(row)

    # Slope/elevation grouped by raster tile, as extract_host does: records
    # arrive in host_sample_id order, which scatters them over many tiles, and
    # the samplers keep only 4 tiles (240 MB each for slope) in memory.
    lons = np.array([c[0] for c in centres])
    lats = np.array([c[1] for c in centres])
    elev_all = elev_s.sample_many(lons, lats, conventions["elevation"])
    by_tile: Dict[str, List[int]] = {}
    for ri, r in enumerate(rows):
        by_tile.setdefault(r["tile"], []).append(ri)
    for tile, idxs in by_tile.items():
        for ri in idxs:
            rows[ri]["slope"] = slope_s.sample(tile, lons[ri], lats[ri],
                                               conventions["slope"])
            rows[ri]["elevation"] = int(elev_all[ri])

    # Meteo on both axes, in bulk: one windowed read per (date, band).
    for ax in ("s2", "s1"):
        reqs = [(lo, la, r[f"{ax}_date"]) for (lo, la), r in zip(centres, rows)]
        for out_name, band in AGERA5_DAILY_BANDS.items():
            for r, v in zip(rows, meteo.sample_many(reqs, band)):
                r[f"{out_name}_{ax}"] = v

    df = pd.DataFrame(rows)
    if meteo.missing:
        logger.warning(f"{host_ref_id}: AGERA5 daily missing for "
                       f"{len(meteo.missing)} date(s); those entries are NODATA")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(f".tmp{os.getpid()}.parquet")
        _write_native_parquet(df, tmp, half=int(
            conventions.get("spatial_context", 0) or 0))
        tmp.rename(out_path)
        logger.success(f"{host_ref_id}: {len(df):,} samples "
                       f"(median {int(df.n_s2.median())} S2 / "
                       f"{int(df.n_s1.median())} S1 acquisitions) -> {out_path}")
    return df


def _write_native_parquet(df: pd.DataFrame, path: Path, half: int) -> None:
    """GeoParquet with an EXPLICIT schema for the ragged list columns.

    Type inference is not usable here: a ref where every sample lacks S1 gives
    pyarrow an all-empty column and it infers `null`, which then refuses to
    concat with the uint16 column of any other ref. Stating the types makes
    every output file schema-identical regardless of content.
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    band_t = (pa.list_(pa.uint16()) if half == 0
              else pa.list_(pa.list_(pa.uint16())))
    list_types = {"s2_date": pa.list_(pa.date32()),
                  "s1_date": pa.list_(pa.date32()),
                  "s2_doy": pa.list_(pa.int16()),
                  "s1_doy": pa.list_(pa.int16())}
    for b in E.S2_BANDS + E.S1_BANDS:
        list_types[b] = band_t
    for b in METEO_COLS:
        list_types[b] = pa.list_(pa.uint16())

    plain = df.drop(columns=[c for c in list_types if c in df.columns]
                    + ["geometry"])
    table = pa.Table.from_pandas(plain, preserve_index=False)
    npix = (2 * half + 1) ** 2
    for name, typ in list_types.items():
        if name in df.columns:
            table = table.append_column(
                pa.field(name, typ), _list_array(df[name].tolist(), typ, npix))
    table = table.append_column(
        pa.field("geometry", pa.binary()),
        pa.array([g.wkb for g in df["geometry"]], type=pa.binary()))

    meta = dict(table.schema.metadata or {})
    meta[b"geo"] = json.dumps({
        "version": "1.0.0", "primary_column": "geometry",
        "columns": {"geometry": {
            "encoding": "WKB", "geometry_types": ["Point"],
            "crs": {"$schema": "https://proj.org/schemas/v0.5/projjson.schema.json",
                    "type": "GeographicCRS", "name": "WGS 84",
                    "datum": {"type": "GeodeticReferenceFrame", "name":
                              "World Geodetic System 1984",
                              "ellipsoid": {"name": "WGS 84",
                                            "semi_major_axis": 6378137,
                                            "inverse_flattening": 298.257223563}},
                    "coordinate_system": {"subtype": "ellipsoidal", "axis": [
                        {"name": "Geodetic latitude", "abbreviation": "Lat",
                         "direction": "north", "unit": "degree"},
                        {"name": "Geodetic longitude", "abbreviation": "Lon",
                         "direction": "east", "unit": "degree"}]},
                    "id": {"authority": "EPSG", "code": 4326}}}}}).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def _list_array(values: list, typ, npix: int):
    """Build a list column from flat values + offsets (no per-element Python
    objects). Handles list<scalar> and, for windows, list<list<uint16>>."""
    import pyarrow as pa

    lens = np.fromiter((len(v) for v in values), dtype=np.int64,
                       count=len(values))
    outer = pa.array(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
    inner_t = typ.value_type
    if pa.types.is_list(inner_t):          # windows: each element is npix values
        flat = (np.concatenate([np.concatenate(v) for v in values if len(v)])
                if lens.sum() else np.empty(0)).astype(np.uint16)
        inner = pa.ListArray.from_arrays(
            pa.array(np.arange(0, int(lens.sum()) * npix + 1, npix,
                               dtype=np.int32)),
            pa.array(flat, type=inner_t.value_type))
        return pa.ListArray.from_arrays(outer, inner)
    if pa.types.is_date32(inner_t):
        flat = np.array([d for v in values for d in v], dtype="datetime64[D]")
    else:
        flat = (np.concatenate([np.asarray(v) for v in values if len(v)])
                if lens.sum() else np.empty(0))
    return pa.ListArray.from_arrays(outer, pa.array(flat, type=inner_t))
