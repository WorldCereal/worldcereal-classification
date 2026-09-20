#!/usr/bin/env python
"""Build spatial-inference test patches from the LOCAL extraction patches.

`spatial_inference.py` consumes already-composited patch NetCDFs (the
GLOBAL_TEST_PATCHES layout: monthly `t`, full `y`/`x`, 10 S2 bands + S1 VH/VV
+ slope/elevation + AGERA5). Those came from an openEO batch job, which bakes
the S2 cloud-masking convention in server-side, so a raw_scl-consistent patch
cannot be derived from an existing dilated one, and re-running the openEO job
is not affordable.

The local extraction patches under S2_ROOT are raw time series carrying BOTH
mask bands (S2-L2A-SCL and S2-L2A-SCL_DILATED_MASK), so the same patch can be
composited either way with no openEO call. This module applies the per-point
recipe of `ptp_engine.composite_s2` to the whole 2-D grid and writes the result
in the layout `spatial_inference.py` expects.

The two conventions differ only in which observations enter the per-period
median. S1, slope, elevation and AGERA5 come from the same code in both runs,
so a dilated/raw_scl pair generated here is controlled by construction.

S2 output is bit-identical to `ptp_engine.composite_s2`. S1 reproduces
`composite_s1`'s arithmetic, but its per-period mean is a vectorised `nanmean`,
so float32 summation order can differ by <=1 ulp (occasionally +-1 DN). That
difference is common-mode and cancels between the two runs.

Usage:
    python ptp_patch_composite.py \
        --host-ref-id 2017_BEL_LPIS-Flanders_POLY_110 \
        --s2-mask raw_scl --out-dir /path/out --limit 5

Run it twice with the same --limit/--seed and --s2-mask dilated / raw_scl to
generate the controlled pair.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS, Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ptp_engine as E  # noqa: E402

# --- Grid compositing (ptp_engine's recipe, applied to every pixel) --------


def _month_time_selector(times, months, t_start, t_end_excl, freq="month"):
    """Per period, the boolean time-step mask that composite_* would use."""
    sel = (times >= t_start) & (times < t_end_excl)
    mkeys = [E._period_key(t, freq) for t in times]
    return [
        np.array(
            [sel[ti] and mkeys[ti] == month for ti in range(len(times))],
            dtype=bool,
        )
        for month in months
    ]


def composite_s2_grid(
    patch: dict,
    months: List[Tuple[int, int, int]],
    t_start: np.datetime64,
    t_end_excl: np.datetime64,
    s2_mask: str = "dilated",
    freq: str = "month",
) -> np.ndarray:
    """(10, n_months, y, x) uint16 — composite_s2 evaluated at every pixel.

    `s2_mask` picks the masking convention as DEFAULT_CONVENTIONS["s2_mask"]
    does: "dilated" uses the erosion/dilation band, "raw_scl" the raw classes.
    """
    times = patch["times"]
    if s2_mask == "raw_scl":
        bad = np.isin(patch["bands"][E.SCL_RAW_BAND], list(E.SCL_REJECT_CLASSES))
    else:
        bad = patch["bands"][E.SCL_DILATED_BAND] == 1

    ny, nx = len(patch["y"]), len(patch["x"])
    out = np.full((len(E.S2_BANDS), len(months), ny, nx), E.NODATA, np.uint16)
    tsels = _month_time_selector(times, months, t_start, t_end_excl, freq)

    for mi, tsel in enumerate(tsels):
        if not tsel.any():
            continue
        keep = ~bad[tsel]  # (k, y, x)
        for bi, band in enumerate(E.S2_BANDS):
            series = patch["bands"][band][tsel].astype(np.float64)
            vals = np.where(keep & (series != E.NODATA), series, np.nan)
            with warnings.catch_warnings():
                # Months with no surviving observation are all-NaN by design;
                # they become NODATA just below.
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(vals, axis=0)
            ok = np.isfinite(med)
            out[bi, mi][ok] = np.floor(med[ok]).astype(np.uint16)
    return out


def composite_s1_grid(
    s1: dict,
    months: List[Tuple[int, int, int]],
    t_start: np.datetime64,
    t_end_excl: np.datetime64,
    freq: str = "month",
) -> np.ndarray:
    """(2, n_months, y, x) uint16 — composite_s1 evaluated at every pixel."""
    times = s1["times"]
    ny, nx = len(s1["y"]), len(s1["x"])
    out = np.full((len(E.S1_BANDS), len(months), ny, nx), E.NODATA, np.uint16)
    tsels = _month_time_selector(times, months, t_start, t_end_excl, freq)

    for bi, band in enumerate(E.S1_BANDS):
        if band not in s1["bands"]:
            continue
        arr = s1["bands"][band]
        for mi, tsel in enumerate(tsels):
            if not tsel.any():
                continue
            s = arr[tsel].astype(np.float32)
            valid = (s != 0) & (s != E.NODATA)
            if not valid.any():
                continue
            dns = np.where(valid, s.astype(np.float64), np.nan)
            with warnings.catch_warnings(), np.errstate(
                invalid="ignore", divide="ignore"
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                power = (
                    10.0 ** ((20.0 * np.log10(dns) - 83.0) / 10.0)
                ).astype(np.float32)
                mean_power = np.nanmean(power, axis=0, dtype=np.float32)
                dn = (
                    10.0
                    ** ((10.0 * np.log10(mean_power.astype(np.float64)) + 83.0)
                        / 20.0)
                ).astype(np.float32)
            ok = np.isfinite(dn)
            out[bi, mi][ok] = np.clip(np.floor(dn[ok]), 1, 65534).astype(np.uint16)
    return out


def _s1_nodata_score(composite: np.ndarray) -> int:
    """Total NODATA cells — the grid analogue of ptp's per-point coverage rule."""
    return int((composite == E.NODATA).sum())


def s1_index_maps(
    s2_patch: dict, s1: dict, conventions: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Nearest S1 (row, col) for every S2 pixel centre, plus an in-bounds mask.

    S1 is on a 20 m grid and sometimes another UTM zone, so it is not aligned
    with the 10 m S2 grid. This is ptp_engine's per-point rule (S2 pixel centre
    into the S1 CRS, nearest index, frozen offsets) applied to every pixel.
    """
    s2_crs = CRS.from_wkt(s2_patch["crs_wkt"])
    s1_crs = CRS.from_wkt(s1["crs_wkt"])
    case = "s1_same_crs" if s1_crs.equals(s2_crs) else "s1_cross_crs"

    xx, yy = np.meshgrid(s2_patch["x"], s2_patch["y"])
    if case == "s1_same_crs":
        qx, qy = xx, yy
    else:
        tf = Transformer.from_crs(s2_crs, s1_crs, always_xy=True)
        qx, qy = tf.transform(xx.ravel(), yy.ravel())
        qx = np.asarray(qx).reshape(xx.shape)
        qy = np.asarray(qy).reshape(yy.shape)

    col = np.abs(s1["x"][None, None, :] - qx[..., None]).argmin(axis=-1)
    row = np.abs(s1["y"][None, None, :] - qy[..., None]).argmin(axis=-1)
    col = col + conventions[case]["col_off"]
    row = row + conventions[case]["row_off"]

    inside = (
        (row >= 0) & (row < len(s1["y"])) & (col >= 0) & (col < len(s1["x"]))
    )
    return np.clip(row, 0, len(s1["y"]) - 1), np.clip(col, 0, len(s1["x"]) - 1), inside, case


def s1_on_s2_grid(
    s1_composite: np.ndarray, row: np.ndarray, col: np.ndarray, inside: np.ndarray
) -> np.ndarray:
    """Gather an S1 composite (2, m, y1, x1) onto the S2 grid (2, m, y2, x2)."""
    out = s1_composite[:, :, row, col]
    out[:, :, ~inside] = E.NODATA
    return out


# --- Auxiliary bands on the patch grid ------------------------------------


def _pixel_lonlat(patch: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-centre lon/lat for every cell, flattened row-major (y, x)."""
    crs = CRS.from_wkt(patch["crs_wkt"])
    xx, yy = np.meshgrid(patch["x"], patch["y"])
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(xx.ravel(), yy.ravel())
    return np.asarray(lon), np.asarray(lat)


def aux_grids(
    patch: dict,
    tile: str,
    months: List[Tuple[int, int]],
    conventions: dict,
    meteo: E.MonthlyMeteo,
    slope_s: E.SlopeSampler,
    elev_s: E.ElevationSampler,
) -> Dict[str, np.ndarray]:
    """slope/elevation (y, x) and AGERA5 (n_months, y, x), uint16 as in ptp."""
    ny, nx = len(patch["y"]), len(patch["x"])
    lons, lats = _pixel_lonlat(patch)

    slope = np.array(
        [slope_s.sample(tile, lo, la, conventions["slope"])
         for lo, la in zip(lons, lats)],
        dtype=np.int64,
    )
    elev = elev_s.sample_many(lons, lats, conventions["elevation"])

    tmean = np.full((len(months), ny * nx), E.NODATA, dtype=np.uint16)
    precip = np.full((len(months), ny * nx), E.NODATA, dtype=np.uint16)
    for mi, (yy_, mm_, dd_) in enumerate(months):
        t = meteo.sample(yy_, mm_, dd_, "temperature-mean", lons, lats,
                         conventions["meteo"])
        p = meteo.sample(yy_, mm_, dd_, "precipitation-flux", lons, lats,
                         conventions["meteo"])
        tmean[mi] = np.minimum(np.floor(t), E.NODATA).astype(np.uint16)
        precip[mi] = np.minimum(np.floor(p), E.NODATA).astype(np.uint16)

    return {
        "slope": slope.astype(np.uint16).reshape(ny, nx),
        "elevation": elev.astype(np.uint16).reshape(ny, nx),
        "AGERA5-TMEAN": tmean.reshape(len(months), ny, nx),
        "AGERA5-PRECIP": precip.reshape(len(months), ny, nx),
    }


# --- NetCDF writing (GLOBAL_TEST_PATCHES layout) --------------------------


def write_patch_netcdf(
    out_path: Path,
    patch: dict,
    months: List[Tuple[int, int]],
    s2: np.ndarray,
    s1: np.ndarray,
    aux: Dict[str, np.ndarray],
    attrs: Dict[str, str],
) -> None:
    """Write the composited patch in the layout spatial_inference.py reads.

    Matches the openEO-produced files: int `t` as days since 1990-01-01, a
    `crs` scalar carrying the WKT, and every band as int16-with-_Unsigned
    (uint16 bit pattern) so xarray's mask_and_scale reads them back correctly.
    """
    import netCDF4

    ny, nx = len(patch["y"]), len(patch["x"])
    epoch = np.datetime64("1990-01-01")
    tvals = [
        int((np.datetime64(f"{y:04d}-{m:02d}-{d:02d}") - epoch)
            / np.timedelta64(1, "D"))
        for (y, m, d) in months
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".tmp{os.getpid()}.nc")
    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as ds:
        ds.createDimension("t", None)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        ds.createDimension("string1", 1)

        v = ds.createVariable("t", "i4", ("t",))
        v.standard_name = v.long_name = "t"
        v.axis = "T"
        v.units = "days since 1990-01-01"
        v.calendar = "proleptic_gregorian"
        v[:] = tvals

        for name, data, sn, ln in (
            ("x", patch["x"], "projection_x_coordinate",
             "x coordinate of projection"),
            ("y", patch["y"], "projection_y_coordinate",
             "y coordinate of projection"),
        ):
            v = ds.createVariable(name, "f8", (name,), fill_value=np.nan)
            v.standard_name, v.long_name, v.units = sn, ln, "m"
            v[:] = data

        v = ds.createVariable("crs", "S1", ("string1",))
        v.crs_wkt = v.spatial_ref = patch["crs_wkt"]

        def _band(name: str, arr: np.ndarray, dims):
            var = ds.createVariable(name, "i2", dims, fill_value=np.int16(-1),
                                    zlib=True, complevel=4)
            var.set_auto_maskandscale(False)
            var.long_name = name
            var.units = ""
            var.grid_mapping = "crs"
            var._Unsigned = "true"
            var[:] = arr.astype(np.uint16).view(np.int16)

        for bi, b in enumerate(E.S2_BANDS):
            _band(b, s2[bi], ("t", "y", "x"))
        for bi, b in enumerate(E.S1_BANDS):
            _band(b, s1[bi], ("t", "y", "x"))
        # slope/elevation are static; broadcast over t so the band stack the
        # model sees is uniform, exactly as in the openEO patches.
        for b in ("slope", "elevation"):
            _band(b, np.broadcast_to(aux[b], (len(months), ny, nx)),
                  ("t", "y", "x"))
        for b in ("AGERA5-PRECIP", "AGERA5-TMEAN"):
            _band(b, aux[b], ("t", "y", "x"))

        ds.Conventions = "CF-1.9"
        for k, val in attrs.items():
            setattr(ds, k, val)

    try:
        os.chmod(tmp, 0o664)
    except OSError:
        pass
    tmp.rename(out_path)


# --- Per-patch driver -----------------------------------------------------


def build_patch(
    sample_id: str,
    entry: dict,
    months: List[Tuple[int, int, int]],
    t_start: np.datetime64,
    t_end_excl: np.datetime64,
    conventions: dict,
    meteo: E.MonthlyMeteo,
    slope_s: E.SlopeSampler,
    elev_s: E.ElevationSampler,
    out_path: Path,
) -> bool:
    """Composite one extraction patch and write it. False if unusable."""
    s2_mask = conventions.get("s2_mask", "dilated")
    freq = conventions.get("freq", "month")
    mask_band = E.SCL_RAW_BAND if s2_mask == "raw_scl" else E.SCL_DILATED_BAND

    try:
        s2_patch = E._read_patch(entry["s2"], E.S2_BANDS + [mask_band])
    except OSError as exc:
        logger.warning(f"{sample_id}: S2 patch unreadable, skipping ({exc})")
        return False
    if mask_band not in s2_patch["bands"]:
        logger.warning(f"{sample_id}: {mask_band} absent from patch, skipping")
        return False

    s2 = composite_s2_grid(s2_patch, months, t_start, t_end_excl,
                           s2_mask=s2_mask, freq=freq)

    # S1: composite every readable orbit on the patch grid and keep the one
    # with the best coverage. ptp_engine chooses per point; a patch needs a
    # single orbit for the whole grid, so the rule is applied grid-wide.
    ny, nx = len(s2_patch["y"]), len(s2_patch["x"])
    s1 = np.full((len(E.S1_BANDS), len(months), ny, nx), E.NODATA, np.uint16)
    chosen_orbit = None
    chosen_case = None
    best_score = None
    for orbit, path in sorted(
        ((o, p) for o, p in entry["s1"].items() if p and Path(p).exists()),
        key=lambda kv: -Path(kv[1]).stat().st_size,
    ):
        try:
            s1p = E._read_patch(path, E.S1_BANDS)
        except OSError as exc:
            logger.warning(f"{sample_id}: S1 {orbit} unreadable ({exc})")
            continue
        row, col, inside, case = s1_index_maps(s2_patch, s1p, conventions)
        cand = s1_on_s2_grid(
            composite_s1_grid(s1p, months, t_start, t_end_excl, freq),
            row, col, inside
        )
        score = _s1_nodata_score(cand)
        if best_score is None or score < best_score:
            s1, best_score, chosen_orbit, chosen_case = cand, score, orbit, case

    aux = aux_grids(s2_patch, entry["tile"], months, conventions,
                    meteo, slope_s, elev_s)

    write_patch_netcdf(
        out_path, s2_patch, months, s2, s1, aux,
        attrs={
            "institution": "local ptp_patch_composite (ptp_engine recipe)",
            "title": sample_id,
            "description": (
                f"Monthly composite from local extraction patches; "
                f"s2_mask={s2_mask}"
            ),
            "s2_mask": s2_mask,
            "s1_orbit": chosen_orbit or "none",
            "s1_case": chosen_case or "none",
            "source_s2_patch": str(entry["s2"]),
            "tile": entry["tile"],
        },
    )
    return True


def _out_name(sample_id: str, year: int, suffix: str) -> str:
    """spatial_inference.py parses the year from `patch_<id>_<year>_<suffix>`."""
    return f"patch_{sample_id}_{year}_{suffix}.nc"


def run_manifest(args) -> int:
    """Composite the patch set named by a select_test_patches.py manifest.

    The manifest carries s2/s1 paths, tile, zone and the patch window, so
    nothing is rediscovered by walking the extraction tree, and the same
    manifest runs under either --s2-mask, which is what makes the two output
    trees comparable.
    """
    man = pd.read_parquet(args.manifest)
    logger.info(f"{len(man)} patches in manifest, s2_mask={args.s2_mask}")

    conventions = dict(E.DEFAULT_CONVENTIONS)
    conventions["s2_mask"] = args.s2_mask
    conventions["freq"] = args.freq

    meteo = E.MonthlyMeteo(args.agera5_cache)
    slope_s, elev_s = E.SlopeSampler(), E.ElevationSampler()

    written = skipped = failed = 0
    for i, r in man.iterrows():
        region = r["region_folder"]
        sid = r["sample_id"]
        if not isinstance(r["s2_path"], str) or not r["s2_path"]:
            logger.warning(f"{sid}: manifest has no S2 path, skipping")
            failed += 1
            continue
        entry = {
            "s2": Path(r["s2_path"]),
            "s1": {o: (Path(r[c]) if isinstance(r[c], str) and r[c] else None)
                   for o, c in (("ASCENDING", "s1_asc"), ("DESCENDING", "s1_desc"))},
            "tile": r["tile"],
            "zone": str(r["zone"]),
        }
        if not entry["s2"].exists():
            logger.warning(f"{sid}: S2 patch missing on disk, skipping")
            failed += 1
            continue

        # Month axis from this patch's own window, which is what
        # month_axis_from_patches derives per zone from the same filename dates.
        start = pd.Timestamp(r["start_date"]).replace(day=1)
        today = pd.Timestamp.today().normalize()
        last_complete = today.replace(day=1) - pd.Timedelta(days=1)
        end = min(pd.Timestamp(r["end_date"]), last_complete)
        end = end.replace(day=1) + pd.offsets.MonthEnd(0)
        months = [(t.year, t.month, d)
                  for t in pd.date_range(start, end, freq="MS")
                  for d in E.PERIOD_DAYS[getattr(args, "freq", "month")]]
        if not months:
            logger.warning(f"{sid}: empty month axis, skipping")
            failed += 1
            continue
        t_start = np.datetime64(start.strftime("%Y-%m-%d"))
        t_end_excl = np.datetime64(
            (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )

        year = months[len(months) // 2][0]
        out_path = args.out_dir / region / _out_name(sid, year, args.suffix)
        if out_path.exists() and not args.overwrite:
            logger.info(f"[{i+1}/{len(man)}] {sid}: exists, skipping")
            skipped += 1
            continue

        logger.info(
            f"[{i+1}/{len(man)}] {region}/{sid}: {len(months)} months "
            f"({months[0][0]}-{months[0][1]:02d} -> {months[-1][0]}-{months[-1][1]:02d})"
        )
        try:
            ok = build_patch(sid, entry, months, t_start, t_end_excl, conventions,
                             meteo, slope_s, elev_s, out_path)
        except Exception as exc:  # one bad patch must not lose the whole run
            logger.error(f"{sid}: failed ({type(exc).__name__}: {exc})")
            failed += 1
            continue
        if ok:
            written += 1
        else:
            failed += 1

    logger.success(
        f"s2_mask={args.s2_mask}: {written} written, {skipped} skipped, "
        f"{failed} failed -> {args.out_dir}"
    )
    if meteo.missing:
        logger.warning(f"AGERA5 months unavailable (NODATA): {sorted(meteo.missing)}")
    return 0 if failed == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Manifest from select_test_patches.py. Output is "
                         "written to <out-dir>/<region>/, the layout "
                         "spatial_inference.py expects.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute patches whose output already exists.")
    ap.add_argument("--host-ref-id", default=None,
                    help="Extraction host under S2_ROOT, e.g. 2017_BEL_LPIS-Flanders_POLY_110")
    ap.add_argument("--freq", choices=["month", "dekad"], default="month",
                    help="compositing period (patch_to_point.py --period): "
                         "'month' (default) or 'dekad' (3 per month at days "
                         "1/11/21)")
    ap.add_argument("--s2-mask", choices=["dilated", "raw_scl"], default="dilated",
                    help="S2 cloud-masking convention (default: dilated).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="Composite at most this many patches.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the patch sample when --limit is set.")
    ap.add_argument("--sample-ids", nargs="*", default=None,
                    help="Explicit host_sample_ids instead of a random sample.")
    ap.add_argument("--suffix", default="local",
                    help="Filename suffix (default: local).")
    ap.add_argument("--agera5-cache", type=Path, required=True,
                    help="Directory for staged AGERA5 monthly composites.")
    ap.add_argument("--s2-root", type=Path, default=None)
    ap.add_argument("--s1-root", type=Path, default=None)
    args = ap.parse_args()
    if not args.manifest and not args.host_ref_id:
        ap.error("one of --manifest or --host-ref-id is required")

    if args.s2_root:
        E.S2_ROOT = args.s2_root
    if args.s1_root:
        E.S1_ROOT = args.s1_root
    E.AGERA5_CACHE = args.agera5_cache
    args.agera5_cache.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        return run_manifest(args)

    conventions = dict(E.DEFAULT_CONVENTIONS)
    conventions["s2_mask"] = args.s2_mask
    conventions["freq"] = args.freq

    index = E.index_patches(args.host_ref_id)
    usable = {sid: e for sid, e in index.items() if e["s2"]}
    if not usable:
        logger.error(f"No S2 patches found for host {args.host_ref_id}")
        return 1
    logger.info(f"{args.host_ref_id}: {len(usable)} patches with S2 available")

    if args.sample_ids:
        chosen = [s for s in args.sample_ids if s in usable]
        missing = set(args.sample_ids) - set(chosen)
        if missing:
            logger.warning(f"Requested sample_ids not found: {sorted(missing)}")
    else:
        chosen = sorted(usable)
        if args.limit is not None and args.limit < len(chosen):
            rng = np.random.default_rng(args.seed)
            chosen = [chosen[i] for i in
                      sorted(rng.choice(len(chosen), args.limit, replace=False))]

    axes = E.month_axis_from_patches(index, set(chosen), args.freq)
    meteo = E.MonthlyMeteo(args.agera5_cache, freq=args.freq)
    # Seed AGERA5 up front: a cold cache otherwise has every patch fetch the
    # same rasters on demand (and dekad needs 3x as many as month).
    if axes:
        try:
            E.seed_meteo_cache(
                args.agera5_cache,
                min(a["start"] for a in axes.values()),
                max(a["end"] for a in axes.values()), freq=args.freq)
        except Exception as exc:                # noqa: BLE001
            logger.warning(f"AGERA5 seeding skipped: {exc}")
    slope_s, elev_s = E.SlopeSampler(), E.ElevationSampler()

    written = 0
    for sid in chosen:
        entry = usable[sid]
        axis = axes.get(entry["zone"])
        if not axis:
            logger.warning(f"{sid}: no month axis for zone {entry['zone']}, skipping")
            continue
        months = [tuple(m) for m in axis["months"]]
        t_start = np.datetime64(axis["start"])
        t_end_excl = np.datetime64(
            (pd.Timestamp(axis["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        year = months[len(months) // 2][0]  # mid-window year labels the patch
        out_path = args.out_dir / _out_name(sid, year, args.suffix)
        if out_path.exists():
            logger.info(f"{sid}: exists, skipping -> {out_path.name}")
            continue
        logger.info(f"{sid}: compositing {len(months)} {args.freq} period(s), "
                    f"s2_mask={args.s2_mask}")
        if build_patch(sid, entry, months, t_start, t_end_excl, conventions,
                       meteo, slope_s, elev_s, out_path):
            written += 1
            logger.success(f"{sid}: wrote {out_path}")

    logger.success(f"Done: {written} patch(es) written to {args.out_dir}")
    if meteo.missing:
        logger.warning(f"AGERA5 months unavailable (NODATA): {sorted(meteo.missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
