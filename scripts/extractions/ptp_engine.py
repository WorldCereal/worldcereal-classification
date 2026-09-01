"""ENGINE for local (openEO-free) patch-to-point extraction.

Campaign-agnostic core shared by all extraction campaigns: it turns
(points assigned to patches) into validated time-series values. Campaign
drivers own everything else — which points, mapped to which patches, written
where:

  * ptp_campaign_inpatch.py — in-patch hard negatives (host_sample_id
    routing, h3 remap, rekey to our ref_ids). First user of this engine;
    its point loader (`load_host_points`) is also still the default when
    `extract_host(points=None)` — supply `points=` to bypass it.
  * ptp_campaign_rdm.py — full-RDM reprocessing: points from the
    harmonized RDM files, primaries at their own patch, collaterals assigned
    via ref_catalog footprints.

Patch discovery is pluggable too: `index_source="fs"` walks the extraction
tree (original behaviour), "stac"/"auto" use ref_catalog.RefCatalog
(seconds per ref instead of minutes).

Reads the S2/S1 patch NetCDFs directly from /data/worldcereal_data and
reproduces the openEO patch-to-point output bit-for-bit, per the empirically
validated recipe:

  S2   : drop obs where SCL_DILATED_MASK == 1 or DN == 65535; per-calendar-month
         MEDIAN per band; floor to uint16.
  S1   : uint16 DN -> dB = 20*log10(DN) - 83 -> linear power; per-month MEAN in
         the linear domain; DN = 10**((10*log10(mean)+83)/20); FLOOR (truncation);
         clamp [1, 65534].
  METEO: AGERA5 monthly composites (public CloudFerro S3, identical to the
         collection openEO loads; local-daily fallback). Value of the covering
         0.1-degree cell.
  SLOPE: the exact Terrascope product openEO loads has LOCAL hrefs:
         /data/worldcereal_data/AUXDATA/COP-DEM_GLO-30_SLOPE/S2grid_20m/slope_<TILE>.tif
  ELEV : /data/MTDA/DEM/COPERNICUS-DEM-30 (bilinear at the S2 pixel centre,
         matches the store within ~2 m).

Pixel selection is deterministic geometry: the point's coordinates plus the
patch's own georeferencing identify the containing pixel with certainty. 
DEFAULT_CONVENTIONS below records the frozen SEMANTIC/ENCODING decisions of 
the recipe (interpolation modes, the aux-at-S2-pixel-centre rule, float32 S1 cell
reads). NOTE: the S1 *arithmetic* was float32 to mirror openEO; it is now done in
float64 (see _s1_monthly) because it is measurably more accurate at the point and
because float32 vector math is not reproducible across CPU generations. S1 DNs may
therefore differ by +-1 from pre-2026-08-20 outputs and from openEO.
They were established once by empirical calibration against openEO ground truth 
(bit-exact on 1,188 samples / 3 hosts, then 59 hosts) and subsequently confirmed line-by-line in
the openEO backend source.

Why local at all: this whole local route exist because openEO's aggregate_spatial 
returns a NEIGHBOURING pixel (not the point's own) for ~48% of points. 
The bug was reported to the openEO team on 2026-08-13; this route was validated against 
openEO ground truth on 1188 samples across 3 hosts. And it's much faster.

Output: the same host-keyed <host>_<run-suffix>.geoparquet the openEO route
produced, into --merged-dir.
"""

import argparse
import calendar
import json
import os
import sys as _sys

# The ptp_* modules and ref_catalog live as flat scripts in this directory
# (not installed as a package). Drivers insert this dir on sys.path and
# direct execution gets it automatically; this line covers the remaining
# case — ptp_engine imported via an exotic mechanism (importlib-by-path,
# notebooks) — so the lazy `from ref_catalog import ...` inside
# extract_host always resolves for every user.
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import requests
from loguru import logger
from pyproj import CRS, Transformer

# --- Paths ----------------------------------------------------------------
# Stable Terrascope/project locations (defaults are correct on the Terrascope
# clusters; overridable via the CLI flags of the same name).

S2_ROOT = Path("/data/worldcereal_data/EXTRACTIONS/SENTINEL_2")
S1_ROOT = Path("/data/worldcereal_data/EXTRACTIONS/SENTINEL_1")
SLOPE_DIR = Path("/data/worldcereal_data/AUXDATA/COP-DEM_GLO-30_SLOPE/S2grid_20m")
DEM_DIR = Path("/data/MTDA/DEM/COPERNICUS-DEM-30")
AGERA5_DAILY = Path("/data/MTDA/AgERA5")
AGERA5_S3 = "https://s3.waw3-1.cloudferro.com/agera_monthly_v2/agera5_monthly_composite"

# Campaign-specific locations: no baked-in defaults — main() fills these from
# the CLI (--gt-dir, --merged-dir, --run-suffix, --agera5-cache,
# --reference-dir) before any of the functions below run.
GT_DIR: Optional[Path] = None
MERGED_DIR: Optional[Path] = None
AGERA5_CACHE: Optional[Path] = None
RUN_SUFFIX = "LOCAL"

S2_BANDS = [
    "S2-L2A-B02", "S2-L2A-B03", "S2-L2A-B04", "S2-L2A-B05", "S2-L2A-B06",
    "S2-L2A-B07", "S2-L2A-B08", "S2-L2A-B8A", "S2-L2A-B11", "S2-L2A-B12",
]
S1_BANDS = ["S1-SIGMA0-VH", "S1-SIGMA0-VV"]
NODATA = 65535
# Big ref_ids (ESP/POL/AUT: 200k+ samples -> millions of rows) might OOM'd a 15 GB
# VM when the whole long-format frame plus its Arrow copy were materialized at
# once. extract_host assembles and writes in blocks of at most this many rows:
# peak memory is one block regardless of ref size. Small refs (the common
# case) stay single-shot.
CHUNK_ROWS = 1_500_000

# Frozen semantic/encoding decisions (see header). Row/col offsets are grid
# steps relative to the nearest-pixel-centre match, keyed per case; all zero =
# pure geometry.
DEFAULT_CONVENTIONS = {
    "s2": {"row_off": 0, "col_off": 0},
    "s1_same_crs": {"row_off": 0, "col_off": 0},
    "s1_cross_crs": {"row_off": 0, "col_off": 0},
    # Meteo = value of the COVERING 0.1-deg cell. The openEO-era store is a
    # MIXTURE (mostly covering-cell from the nearest-era graph; some
    # near-cell-edge samples bilinear from the newer graph, 
    # so no single choice matches it everywhere; covering_cell
    # matches the dominant historical semantics and keeps "monthly total of
    # the containing cell" interpretable. ptp_verify accepts either
    # convention on the store side.
    "meteo": "covering_cell",       # vs "bilinear"
    "slope": "bilinear_floor",      # vs "nearest"
    "elevation": "bilinear_floor",  # vs "nearest"
    # S2 cloud masking. Two methods, matching patch_to_point.py's
    # --optical-mask-method exactly (see optimized_mask_precomputed /
    # optimized_mask_raw_scl_values in patch_to_point_worldcereal.py):
    #   "dilated" -> drop obs where S2-L2A-SCL_DILATED_MASK == 1. The
    #       precomputed band has already had a large EROSION/DILATION applied,
    #       so pixels merely NEAR cloud/shadow are masked as well. Production
    #       default; this is what the whole openEO-era store used.
    #   "raw_scl" -> drop obs whose raw S2-L2A-SCL class is in
    #       SCL_REJECT_CLASSES. No erosion/dilation, so ONLY pixels actually
    #       classified as bad are dropped. Strictly less aggressive: more
    #       observations survive per month (denser composites, fewer NODATA
    #       months) at the price of some cloud-edge contamination.
    "s2_mask": "dilated",           # vs "raw_scl"
}

# Raw-SCL invalid classes, verbatim from optimized_mask_raw_scl_values:
#   0 no data | 1 saturated/defective | 3 cloud shadow |
#   8 medium-probability cloud | 9 high-probability cloud |
#   10 thin cirrus | 11 snow/ice
# Everything else (4 vegetation, 5 bare, 6 water, 7 unclassified, 2 dark
# area, 12 ...) is kept.
SCL_REJECT_CLASSES = frozenset({0, 1, 3, 8, 9, 10, 11})
SCL_RAW_BAND = "S2-L2A-SCL"
SCL_DILATED_BAND = "S2-L2A-SCL_DILATED_MASK"
# Derived from --merged-dir in main(): <merged-dir>/_local_extractor_conventions.json
CONVENTIONS_FILE: Optional[Path] = None


# --- Patch file index -----------------------------------------------------


def index_patches(host_ref_id: str) -> Dict[str, dict]:
    """One filesystem walk per host: host_sample_id -> file paths + tile/zone.

    Filename dates vary per sample, so files are discovered by listing, never
    constructed. Layout: <root>/<host>/<zone>/<tile>/<sample_id>/<file>.nc
    """
    index: Dict[str, dict] = {}
    for root, key in ((S2_ROOT, "s2"), (S1_ROOT, "s1")):
        base = root / host_ref_id
        if not base.exists():
            continue
        for zone_dir in base.iterdir():
            if not zone_dir.is_dir():
                continue
            for tile_dir in zone_dir.iterdir():
                if not tile_dir.is_dir():
                    continue
                for sdir in tile_dir.iterdir():
                    sid = sdir.name
                    entry = index.setdefault(
                        sid, {"tile": tile_dir.name, "zone": zone_dir.name,
                              "s2": None, "s1": {}}
                    )
                    for nc in sdir.glob("*.nc"):
                        if key == "s2":
                            entry["s2"] = nc
                            entry["tile"] = tile_dir.name
                            entry["zone"] = zone_dir.name
                        else:
                            orbit = ("ASCENDING" if "_ASCENDING_" in nc.name
                                     else "DESCENDING")
                            entry["s1"][orbit] = nc
    return index


# --- NetCDF reading -------------------------------------------------------


def _read_uint16(var) -> np.ndarray:
    """Read an int16-with-_Unsigned-true variable as its uint16 bit pattern."""
    var.set_auto_maskandscale(False)
    raw = np.asarray(var[:])
    if raw.dtype == np.int16:
        return raw.view(np.uint16)
    return raw.astype(np.uint16)


def _read_patch(path: Path, band_names: List[str]) -> dict:
    """Read a patch NetCDF: times, pixel-centre coords, CRS, uint16 bands."""
    with netCDF4.Dataset(path) as ds:
        tvar = ds.variables["t"]
        # 'days since 1990-01-01', proleptic_gregorian
        times = (np.datetime64("1990-01-01") +
                 np.asarray(tvar[:]).astype("timedelta64[D]"))
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        crs_wkt = ds.variables["crs"].crs_wkt
        bands = {}
        for name in band_names:
            if name in ds.variables:
                bands[name] = _read_uint16(ds.variables[name])
    return {"times": times, "x": x, "y": y, "crs_wkt": crs_wkt, "bands": bands}


def _nearest_idx(coords: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(coords - value)))


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values (consecutive NODATA months)."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


# --- Compositing (the exact openEO recipe) --------------------------------


def _month_key(t: np.datetime64) -> Tuple[int, int]:
    ts = pd.Timestamp(t)
    return ts.year, ts.month


def composite_s2(
    patch: dict, row: int, col: int, months: List[Tuple[int, int]],
    t_start: np.datetime64, t_end_excl: np.datetime64,
    s2_mask: str = "dilated",
) -> np.ndarray:
    """(10, n_months) uint16: masked per-month median per band, floor-cast.

    `s2_mask` selects the cloud-masking method — "dilated" (the precomputed
    erosion/dilation band, production) or "raw_scl" (raw SCL classes, no
    erosion/dilation). See DEFAULT_CONVENTIONS["s2_mask"]."""
    times = patch["times"]
    sel = (times >= t_start) & (times < t_end_excl)
    if s2_mask == "raw_scl":
        scl = patch["bands"][SCL_RAW_BAND][:, row, col]
        # bad[ti] is True where the observation must be dropped, so the
        # `mask[ti] != 1` test below reads identically for both methods.
        mask = np.isin(scl, list(SCL_REJECT_CLASSES)).astype(np.uint8)
    else:
        mask = patch["bands"][SCL_DILATED_BAND][:, row, col]
    out = np.full((len(S2_BANDS), len(months)), NODATA, dtype=np.uint16)
    mkeys = [_month_key(t) for t in times]
    for bi, band in enumerate(S2_BANDS):
        series = patch["bands"][band][:, row, col]
        for mi, month in enumerate(months):
            vals = [
                int(series[ti])
                for ti in range(len(times))
                if sel[ti] and mkeys[ti] == month
                and mask[ti] != 1 and series[ti] != NODATA
            ]
            if vals:
                out[bi, mi] = np.uint16(np.floor(np.median(vals)))
    return out


def s1_month_index(
    s1: dict, months: List[Tuple[int, int]],
    t_start: np.datetime64, t_end_excl: np.datetime64,
) -> np.ndarray:
    """Per time step: index into `months`, or -1 when out of window / month.

    Computed once per patch (not per point) and consumed by
    s1_nodata_months, which is called for every candidate orbit of every
    point during orbit selection.
    """
    times = s1["times"]
    sel = (times >= t_start) & (times < t_end_excl)
    midx = {m: i for i, m in enumerate(months)}
    out = np.full(len(times), -1, dtype=np.int64)
    for ti, t in enumerate(times):
        if sel[ti]:
            out[ti] = midx.get(_month_key(t), -1)
    return out


def s1_nodata_months(
    s1: dict, row: int, col: int, month_index: np.ndarray, n_months: int,
) -> np.ndarray:
    """Boolean per month: would composite_s1 leave BOTH bands at NODATA?

    Same rule as composite_s1 (an in-window observation whose DN is neither 0
    nor NODATA), without the power/log arithmetic — so orbit selection is
    cheap and only the winning orbit pays for the full composite.
    """
    has = np.zeros(n_months, dtype=bool)
    for band in S1_BANDS:
        if band not in s1["bands"]:
            continue
        series = s1["bands"][band][:, row, col]
        ok = (month_index >= 0) & (series != 0) & (series != NODATA)
        if ok.any():
            has[month_index[ok]] = True
    return ~has


def composite_s1(
    s1: dict, row: int, col: int, months: List[Tuple[int, int]],
    t_start: np.datetime64, t_end_excl: np.datetime64,
) -> np.ndarray:
    """(2, n_months) uint16: linear-power monthly mean, recompressed, floored."""
    times = s1["times"]
    sel = (times >= t_start) & (times < t_end_excl)
    mkeys = [_month_key(t) for t in times]
    out = np.full((len(S1_BANDS), len(months)), NODATA, dtype=np.uint16)
    for bi, band in enumerate(S1_BANDS):
        if band not in s1["bands"]:
            continue
        series = s1["bands"][band][:, row, col].astype(np.float32)
        for mi, month in enumerate(months):
            dns = np.array([
                series[ti] for ti in range(len(times))
                if sel[ti] and mkeys[ti] == month
                and series[ti] not in (0, NODATA)
            ], dtype=np.float32)
            if len(dns) == 0:
                continue
            # Transcendentals in float64, rounded back to float32 after each
            # step. More accurate and reproducible across machines
            power = np.float32(
                10.0 ** ((20.0 * np.log10(dns.astype(np.float64)) - 83.0)
                         / 10.0))
            mean_power = np.float32(power.mean())
            dn = np.float32(
                10.0 ** ((10.0 * np.log10(np.float64(mean_power)) + 83.0)
                         / 20.0))
            out[bi, mi] = np.uint16(np.clip(np.floor(dn), 1, 65534))
    return out


# --- Per-patch worker -----------------------------------------------------


def process_patch(task: dict) -> List[dict]:
    """Extract S2+S1 monthly series for all points hosted by one patch.

    Runs in a worker process. `task` carries everything needed (paths, points
    as (sample_id, lon, lat), month axis, conventions) so no globals are shared.
    """
    conv = task["conventions"]
    months: List[Tuple[int, int]] = [tuple(m) for m in task["months"]]
    t_start = np.datetime64(task["t_start"])
    t_end_excl = np.datetime64(task["t_end_excl"])
    results = []
    s1_only = bool(task.get("s1_only", False))

    try:
        s2_patch = _read_patch(
            task["s2_path"],
            [] if s1_only else S2_BANDS + [
                SCL_RAW_BAND if conv.get("s2_mask") == "raw_scl"
                else SCL_DILATED_BAND]
        )
    except OSError as exc:
        # Corrupt/truncated NetCDF on disk. Without S2 there is nothing to
        # extract for these points — drop them (same net effect as openEO's
        # all-nodata drop) instead of killing the whole host.
        logger.warning(f"S2 patch unreadable, dropping {len(task['points'])} "
                       f"point(s): {task['s2_path']} ({exc})")
        return []
    s2_crs = CRS.from_wkt(s2_patch["crs_wkt"])
    to_s2 = Transformer.from_crs("EPSG:4326", s2_crs, always_xy=True)

    # S1: read every readable orbit patch, largest file first. The
    # orbit is then chosen PER POINT on coverage — see the point loop.
    cand = {o: p for o, p in task["s1_paths"].items() if p and Path(p).exists()}
    s1_patches: List[Tuple[str, dict]] = []
    s1_cases: Dict[str, str] = {}
    s1_tf: Dict[str, Transformer] = {}
    s1_midx: Dict[str, np.ndarray] = {}
    for orbit in sorted(cand, key=lambda o: -Path(cand[o]).stat().st_size):
        try:
            s1p = _read_patch(cand[orbit], S1_BANDS)
        except OSError as exc:
            logger.warning(f"S1 {orbit} unreadable, skipping it: "
                           f"{cand[orbit]} ({exc})")
            continue
        s1_crs = CRS.from_wkt(s1p["crs_wkt"])
        s1_patches.append((orbit, s1p))
        s1_cases[orbit] = "s1_same_crs" if s1_crs.equals(s2_crs) else "s1_cross_crs"
        s1_tf[orbit] = Transformer.from_crs(s2_crs, s1_crs, always_xy=True)
        s1_midx[orbit] = s1_month_index(s1p, months, t_start, t_end_excl)

    for sample_id, lon, lat in task["points"]:
        px, py = to_s2.transform(lon, lat)
        col = _nearest_idx(s2_patch["x"], px) + conv["s2"]["col_off"]
        row = _nearest_idx(s2_patch["y"], py) + conv["s2"]["row_off"]
        # aggregate_spatial reads the MERGED cube on the S2 10 m grid, so every
        # non-S2 source must be evaluated at the centre of the point's S2
        # pixel, not at the point itself. This is what makes the cross-CRS S1
        # "sometimes +1 row" quirk deterministic.
        cx = float(s2_patch["x"][min(max(col, 0), len(s2_patch["x"]) - 1)])
        cy = float(s2_patch["y"][min(max(row, 0), len(s2_patch["y"]) - 1)])
        rec = {"sample_id": sample_id, "tile": task["tile"],
               "s1_orbit": None, "s1_case": None,
               "s2_pixel_xy": (cx, cy), "s2_crs_wkt": s2_patch["crs_wkt"]}
        if s1_only:
            rec["s2"] = None
        elif 0 <= row < len(s2_patch["y"]) and 0 <= col < len(s2_patch["x"]):
            rec["s2"] = composite_s2(s2_patch, row, col, months,
                                     t_start, t_end_excl,
                                     s2_mask=conv.get("s2_mask", "dilated"))
        else:
            rec["s2"] = np.full((len(S2_BANDS), len(months)), NODATA, np.uint16)

        # S1: coverage-aware orbit choice per point. Each readable
        # orbit is composited at the point's S2 pixel centre; the orbit with
        # the fewest NODATA months wins, then the shortest NODATA run, then
        # the larger file (list order). Rationale: the openEO-era flow chose
        # one orbit per job with a max-temporal-gap rule; the file-size proxy
        # picked a denser orbit with a seasonal hole in ~20 refs.
        best = None
        for orbit, s1p in s1_patches:
            qx, qy = s1_tf[orbit].transform(cx, cy)
            case = s1_cases[orbit]
            s1_col = _nearest_idx(s1p["x"], qx) + conv[case]["col_off"]
            s1_row = _nearest_idx(s1p["y"], qy) + conv[case]["row_off"]
            inside = (0 <= s1_row < len(s1p["y"])
                      and 0 <= s1_col < len(s1p["x"]))
            nod = (s1_nodata_months(s1p, s1_row, s1_col, s1_midx[orbit],
                                    len(months)) if inside
                   else np.ones(len(months), dtype=bool))
            key = (int(nod.sum()), _longest_run(nod))
            if best is None or key < best[0]:
                best = (key, orbit, case, s1p, s1_row, s1_col, inside)
        if best is None:
            rec["s1"] = np.full((2, len(months)), NODATA, np.uint16)
        else:
            _, orbit, case, s1p, s1_row, s1_col, inside = best
            rec["s1"] = (composite_s1(s1p, s1_row, s1_col, months,
                                      t_start, t_end_excl) if inside
                         else np.full((2, len(months)), NODATA, np.uint16))
            rec["s1_orbit"], rec["s1_case"] = orbit, case
        results.append(rec)
    return results


# --- Auxiliary bands ------------------------------------------------------


class MonthlyMeteo:
    """AGERA5 monthly composites: S3-staged primary, local-daily fallback.

    In-season refs (patch windows extending past what AGERA5 covers yet)
    get NODATA meteo for those months instead of a hard failure — but ONLY
    for months beyond the daily archive's last complete month. A missing
    month *behind* that horizon means archive corruption and still raises.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        resolved = cache_dir if cache_dir is not None else AGERA5_CACHE
        if resolved is None:
            raise ValueError(
                "MonthlyMeteo needs a cache_dir (or the module-level "
                "AGERA5_CACHE set, which main() does from --agera5-cache)")
        self.cache_dir = Path(resolved)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._open: Dict[Tuple[int, int, str], Optional[tuple]] = {}
        self._horizon: Optional[Tuple[int, int]] = None
        self.missing: set = set()  # (year, month) served as NODATA

    def _local_path(self, year: int, month: int, band: str) -> Path:
        return self.cache_dir / f"openEO_{year}-{month:02d}-01Z_{band}.tif"

    def _daily_horizon(self) -> Tuple[int, int]:
        """Last month FULLY covered by the local daily archive."""
        if self._horizon is None:
            for ydir in sorted((p for p in AGERA5_DAILY.iterdir()
                                if p.name.isdigit()), reverse=True):
                days = sorted(d.name for d in ydir.iterdir()
                              if len(d.name) == 8 and d.name.isdigit())
                if days:
                    y, m, d = (int(days[-1][:4]), int(days[-1][4:6]),
                               int(days[-1][6:8]))
                    if d < calendar.monthrange(y, m)[1]:
                        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
                    self._horizon = (y, m)
                    break
            if self._horizon is None:
                raise RuntimeError(f"no day folders under {AGERA5_DAILY}")
        return self._horizon

    def _ensure(self, year: int, month: int, band: str) -> Optional[Path]:
        p = self._local_path(year, month, band)
        if p.exists() and p.stat().st_size > 0:
            return p
        url = f"{AGERA5_S3}/openEO_{year}-{month:02d}-01Z_{band}.tif"
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            # Per-process temp name: parallel shards may fetch the same
            # month; rename is atomic and last-writer-wins with identical
            # content. chmod so any group member can refresh it later.
            tmp = p.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(r.content)
            try:
                os.chmod(tmp, 0o664)
            except OSError:
                pass
            tmp.rename(p)
            return p
        if (year, month) > self._daily_horizon():
            h = self._daily_horizon()
            logger.warning(
                f"AGERA5 {year}-{month:02d} not on S3 and beyond the daily "
                f"archive horizon ({h[0]}-{h[1]:02d}) — no source has it yet; "
                "meteo = NODATA for this month")
            return None
        # Fallback: composite from the local daily archive (proven identical:
        # temp = floor(mean of raw K*100), precip = sum of raw mm*100).
        logger.warning(f"S3 miss for {year}-{month:02d} {band}; compositing "
                       "from /data/MTDA/AgERA5 dailies")
        ndays = calendar.monthrange(year, month)[1]
        acc: Optional[np.ndarray] = None
        profile = None
        for day in range(1, ndays + 1):
            f = (AGERA5_DAILY / f"{year}" / f"{year}{month:02d}{day:02d}" /
                 f"AgERA5_{band}_{year}{month:02d}{day:02d}.tif")
            with rasterio.open(f) as ds:
                arr = ds.read(1).astype(np.float64)
                profile = profile or ds.profile
            acc = arr if acc is None else acc + arr
        assert acc is not None and profile is not None  # ndays >= 28
        comp = (np.floor(acc / ndays) if band == "temperature-mean" else acc)
        profile.update(dtype="uint16", nodata=NODATA)
        tmp = p.with_suffix(f".tmp{os.getpid()}.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(comp.astype(np.uint16), 1)
        try:
            os.chmod(tmp, 0o664)
        except OSError:
            pass
        tmp.rename(p)
        return p

    def sample(self, year: int, month: int, band: str,
               lons: np.ndarray, lats: np.ndarray, mode: str) -> np.ndarray:
        key = (year, month, band)
        if key not in self._open:
            path = self._ensure(year, month, band)
            if path is None:
                self._open[key] = None
            else:
                with rasterio.open(path) as ds:
                    self._open[key] = (ds.read(1), ds.transform)
            if len(self._open) > 25:  # keep memory bounded (13 MB per raster)
                self._open.pop(next(iter(self._open)))
        entry = self._open[key]
        if entry is None:
            self.missing.add((year, month))
            return np.full(len(lons), NODATA, dtype=np.float64)
        arr, transform = entry
        out = np.full(len(lons), NODATA, dtype=np.float64)
        inv = ~transform
        for i, (lon, lat) in enumerate(zip(lons, lats)):
            fc, fr = inv * (lon, lat)
            if mode == "bilinear":
                out[i] = _bilinear(arr, fr - 0.5, fc - 0.5, nodata=NODATA)
            else:
                r, c = int(np.floor(fr)), int(np.floor(fc))
                if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                    out[i] = arr[r, c]
        return out


def _bilinear(arr: np.ndarray, fr: float, fc: float,
              nodata: Optional[float] = None) -> float:
    """Bilinear interpolation at fractional (row, col) in pixel-centre space."""
    r0, c0 = int(np.floor(fr)), int(np.floor(fc))
    dr, dc = fr - r0, fc - c0
    vals, wts = [], []
    for (r, c, w) in ((r0, c0, (1 - dr) * (1 - dc)), (r0, c0 + 1, (1 - dr) * dc),
                      (r0 + 1, c0, dr * (1 - dc)), (r0 + 1, c0 + 1, dr * dc)):
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            v = float(arr[r, c])
            if nodata is not None and v == nodata:
                continue
            vals.append(v)
            wts.append(w)
    if not vals or sum(wts) == 0:
        return nodata if nodata is not None else np.nan
    return float(np.dot(vals, wts) / sum(wts))


class SlopeSampler:
    """Terrascope 20 m slope tiles (uint8 degrees, nodata 255), per S2 tile."""

    def __init__(self):
        self._open: Dict[str, Optional[tuple]] = {}
        self._tf: Dict[object, Transformer] = {}

    def sample(self, tile: str, lon: float, lat: float, mode: str) -> int:
        if tile not in self._open:
            path = SLOPE_DIR / f"slope_{tile}.tif"
            if not path.exists():
                self._open[tile] = None
            else:
                with rasterio.open(path) as ds:
                    # Cast once, on load: a 5488x5488 uint8 tile is 30 MB, but
                    # .astype(float64) per call allocated 240 MB *per point*,
                    # which made a 7.7k-point host take longer than the entire
                    # patch extraction that preceded it.
                    self._open[tile] = (ds.read(1).astype(np.float64),
                                        ds.transform, ds.crs)
            if len(self._open) > 4:  # 240 MB each as float64; keep few
                self._open.pop(next(iter(self._open)))
        entry = self._open[tile]
        if entry is None:
            return NODATA
        arr, transform, crs = entry
        if crs not in self._tf:
            self._tf[crs] = Transformer.from_crs("EPSG:4326", crs,
                                                 always_xy=True)
        x, y = self._tf[crs].transform(lon, lat)
        fc, fr = (~transform) * (x, y)
        if mode == "nearest":
            r, c = int(np.floor(fr)), int(np.floor(fc))
            if not (0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]):
                return NODATA
            v = int(arr[r, c])
            return NODATA if v == 255 else v
        vb = _bilinear(arr, fr - 0.5, fc - 0.5, nodata=255)
        return NODATA if vb == 255 or np.isnan(vb) else int(np.floor(vb))


class ElevationSampler:
    """Copernicus 30 m DEM 1x1-degree COG tiles (float32 metres)."""

    def __init__(self):
        self._open: Dict[str, Optional[tuple]] = {}

    @staticmethod
    def _tile_name(lon: float, lat: float) -> str:
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        return (f"Copernicus_DSM_COG_10_{ns}{abs(int(np.floor(lat))):02d}_00_"
                f"{ew}{abs(int(np.floor(lon))):03d}_00_DEM")

    def sample(self, lon: float, lat: float, mode: str) -> int:
        """Sample one point. Cheap ONLY if consecutive calls stay within a few
        DEM tiles — a 1x1-degree COG is 32 MB and casting it to float64 costs
        ~150 ms. Callers with points spread over many tiles must group by
        `tile_name()` first (see `sample_many`), otherwise every point evicts
        the cache and reloads a tile from NFS.
        """
        name = self._tile_name(lon, lat)
        if name not in self._open:
            path = DEM_DIR / name / f"{name}.tif"
            if not path.exists():
                self._open[name] = None
            else:
                with rasterio.open(path) as ds:
                    # Cast once here, never per call.
                    self._open[name] = (ds.read(1).astype(np.float64),
                                        ds.transform)
            if len(self._open) > 4:  # ~66 MB each as float64
                self._open.pop(next(iter(self._open)))
        entry = self._open[name]
        if entry is None:
            return NODATA
        arr, transform = entry
        fc, fr = (~transform) * (lon, lat)
        if mode == "nearest":
            r, c = int(np.floor(fr)), int(np.floor(fc))
            if not (0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]):
                return NODATA
            return int(np.floor(float(arr[r, c])))
        v = _bilinear(arr, fr - 0.5, fc - 0.5)
        return NODATA if np.isnan(v) else int(np.floor(max(v, 0.0)))

    def sample_many(self, lons: np.ndarray, lats: np.ndarray,
                    mode: str) -> np.ndarray:
        """Vectorised over points: group by DEM tile so each tile loads once."""
        out = np.full(len(lons), NODATA, dtype=np.int64)
        names = [self._tile_name(lo, la) for lo, la in zip(lons, lats)]
        order: Dict[str, List[int]] = {}
        for i, n in enumerate(names):
            order.setdefault(n, []).append(i)
        for name, idxs in order.items():
            for i in idxs:
                out[i] = self.sample(lons[i], lats[i], mode)
        return out


# --- Month axis -----------------------------------------------------------


def month_axis_from_patches(index: Dict[str, dict], needed: set) -> dict:
    """Per-zone month axis from S2 patch filename dates (openEO derived the
    job window per EPSG from the S2 STAC; zone dir == UTM zone == EPSG)."""
    zone_bounds: Dict[str, list] = {}
    for sid in needed:
        entry = index.get(sid)
        if not entry or not entry["s2"]:
            continue
        stem = entry["s2"].stem  # ..._<start>_<end>
        start, end = stem.split("_")[-2], stem.split("_")[-1]
        b = zone_bounds.setdefault(entry["zone"], [start, end])
        b[0], b[1] = min(b[0], start), max(b[1], end)

    today = pd.Timestamp.today().normalize()
    last_complete = today.replace(day=1) - pd.Timedelta(days=1)
    axes = {}
    for zone, (start, end) in zone_bounds.items():
        s = pd.Timestamp(start).replace(day=1)
        e = min(pd.Timestamp(end), last_complete)
        e = e.replace(day=1) + pd.offsets.MonthEnd(0)
        months = [(t.year, t.month) for t in pd.date_range(s, e, freq="MS")]
        axes[zone] = {"start": s.strftime("%Y-%m-%d"),
                      "end": e.strftime("%Y-%m-%d"), "months": months}
    return axes


# --- Host extraction ------------------------------------------------------


def load_host_points(host_ref_id: str) -> gpd.GeoDataFrame:
    if GT_DIR is None:
        raise ValueError("GT_DIR is not set (main() sets it from --gt-dir; "
                         "campaign drivers pass points= instead)")
    gt = gpd.read_parquet(GT_DIR / f"{host_ref_id}.geoparquet")
    prov = pd.read_parquet(GT_DIR / "provenance.parquet")
    prov = prov[prov.host_ref_id == host_ref_id]
    gdf = gt.merge(prov[["sample_id", "host_sample_id"]], on="sample_id")
    if len(gdf) != len(gt):
        raise ValueError(
            f"{host_ref_id}: provenance join lost rows "
            f"({len(gt)} -> {len(gdf)})"
        )
    return gdf


def _assemble_s1_only(host_ref_id: str, records: List[dict], sid_axis: dict,
                      out_path: Optional[Path]) -> pd.DataFrame:
    """Long-format frame with only the S1 columns + s1_orbit (S1 refresh)."""
    cols: Dict[str, List[Any]] = {"sample_id": [], "timestamp": [], "S1-SIGMA0-VH": [],
            "S1-SIGMA0-VV": [], "s1_orbit": [], "tile": []}
    for rec in records:
        months = [tuple(m) for m in sid_axis[rec["sample_id"]][0]]
        T = len(months)
        cols["sample_id"].extend([rec["sample_id"]] * T)
        cols["timestamp"].extend(
            np.datetime64(f"{y:04d}-{m:02d}-01") for (y, m) in months)
        cols["S1-SIGMA0-VH"].extend(rec["s1"][0, :T].tolist())
        cols["S1-SIGMA0-VV"].extend(rec["s1"][1, :T].tolist())
        cols["s1_orbit"].extend([rec["s1_orbit"] or "none"] * T)
        cols["tile"].extend([rec["tile"]] * T)
    df = pd.DataFrame(cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for b in ("S1-SIGMA0-VH", "S1-SIGMA0-VV"):
        df[b] = df[b].astype(np.uint16)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(f".tmp{os.getpid()}.parquet")
        df.to_parquet(tmp, index=False)
        tmp.rename(out_path)
        logger.success(f"{host_ref_id}: S1 refresh wrote {df.sample_id.nunique():,} "
                       f"samples / {len(df):,} rows -> {out_path}")
    return df


def extract_host(
    host_ref_id: str,
    conventions: dict,
    workers: int = 16,
    out_path: Optional[Path] = None,
    t_axis_override: Optional[dict] = None,
    sample_limit: Optional[int] = None,
    index_source: str = "fs",
    catalog_cache: Optional[Path] = None,
    points: Optional[gpd.GeoDataFrame] = None,
    index: Optional[Dict[str, dict]] = None,
    s1_only: bool = False,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Extract one host. Returns (long dataframe, raw per-point records).

    index_source: "fs" walks the extraction tree (original behaviour);
    "stac"/"auto" build the patch index from the STAC catalogue via
    RefCatalog (seconds instead of minutes per ref; identical entry shape).
    `points` lets a campaign driver supply its own point set (columns
    sample_id, host_sample_id, geometry); defaults to the in-patch
    ground-truth+provenance loader.
    """
    if points is None:
        points = load_host_points(host_ref_id)
    if sample_limit:
        points = points.head(sample_limit)
    if index is None:
        if index_source == "fs":
            index = index_patches(host_ref_id)
        else:
            from ref_catalog import RefCatalog
            index = RefCatalog.load(host_ref_id, source=index_source,
                                    cache_dir=catalog_cache).entries
    needed = set(points.host_sample_id)
    missing_s2 = [s for s in needed if s not in index or not index[s]["s2"]]
    if missing_s2:
        logger.warning(f"{host_ref_id}: {len(missing_s2)} host patches have no "
                       "local S2 file; their points will be dropped")

    axes = t_axis_override or month_axis_from_patches(index, needed)

    tasks = []
    for hsid, grp in points.groupby("host_sample_id"):
        entry = index.get(hsid)
        if not entry or not entry["s2"]:
            continue
        if entry["zone"] in axes:
            axis = axes[entry["zone"]]
        else:
            axis = next(iter(axes.values()))
            logger.warning(
                f"{host_ref_id}: no month axis for zone {entry['zone']} "
                f"(patch {hsid}); falling back to the axis of zone "
                f"{next(iter(axes))} ({axis['start']}..{axis['end']})")
        tasks.append({
            "s2_path": str(entry["s2"]),
            "s1_paths": {o: str(p) for o, p in entry["s1"].items()},
            "tile": entry["tile"],
            "zone": entry["zone"],
            "points": [(r.sample_id, r.geometry.x, r.geometry.y)
                       for r in grp.itertuples()],
            "months": axis["months"],
            "t_start": axis["start"],
            "t_end_excl": axis["end"],
            "conventions": conventions,
            "s1_only": s1_only,
        })

    logger.info(f"{host_ref_id}: {len(points)} points in {len(tasks)} patches; "
                f"axes: { {z: (a['start'], a['end']) for z, a in axes.items()} }")

    records: List[dict] = []
    if workers <= 1:
        for t in tasks:
            records.extend(process_patch(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for out in pool.map(process_patch, tasks, chunksize=4):
                records.extend(out)
    logger.info(f"{host_ref_id}: extracted S2/S1 for {len(records)} points")

    # Attach the axis used per record (via host_sample_id -> zone).
    sid_axis = {}
    for t in tasks:
        for (sample_id, _, _) in t["points"]:
            sid_axis[sample_id] = (t["months"], t["t_start"], t["t_end_excl"])

    if s1_only:
        return _assemble_s1_only(host_ref_id, records, sid_axis, out_path), records

    # --- Auxiliary bands (main process; cheap) ---
    meteo = MonthlyMeteo()
    slope_s = SlopeSampler()
    elev_s = ElevationSampler()
    pt_geom = dict(zip(points.sample_id, points.geometry))
    attrs = points.set_index("sample_id")

    all_months = sorted({m for t in tasks for m in map(tuple, t["months"])})
    # Sampling location for aux bands = S2 pixel centre in EPSG:4326 (the
    # merged cube lives on the S2 grid; see process_patch).
    centre_ll = []
    _tf_cache: Dict[str, Transformer] = {}
    for r in records:
        wkt = r["s2_crs_wkt"]
        if wkt not in _tf_cache:
            _tf_cache[wkt] = Transformer.from_crs(
                CRS.from_wkt(wkt), "EPSG:4326", always_xy=True)
        centre_ll.append(_tf_cache[wkt].transform(*r["s2_pixel_xy"]))
    lons = np.array([c[0] for c in centre_ll])
    lats = np.array([c[1] for c in centre_ll])
    meteo_vals = {}
    for (yy, mm) in all_months:
        meteo_vals[(yy, mm, "T")] = meteo.sample(
            yy, mm, "temperature-mean", lons, lats, conventions["meteo"])
        meteo_vals[(yy, mm, "P")] = meteo.sample(
            yy, mm, "precipitation-flux", lons, lats, conventions["meteo"])
    if meteo.missing:
        mm_str = ", ".join(f"{y}-{m:02d}" for (y, m) in sorted(meteo.missing))
        logger.warning(
            f"{host_ref_id}: AGERA5-TMEAN/PRECIP = NODATA for month(s) "
            f"{mm_str} — in-season ref, no AGERA5 source covers them yet. "
            "Rows keep their S2/S1 values.")

    # Vectorised column-wise assembly. Flat numpy columns keep the
    # peak memory at a few hundred MB.
    axis_of = {r["sample_id"]: sid_axis[r["sample_id"]] for r in records}
    ts_cache: Dict[tuple, np.ndarray] = {}
    n_per: List[int] = []
    for rec in records:
        months = tuple(map(tuple, axis_of[rec["sample_id"]][0]))
        if months not in ts_cache:
            ts_cache[months] = np.array(
                [np.datetime64(f"{y:04d}-{m:02d}-01") for (y, m) in months],
                dtype="datetime64[ns]")
        n_per.append(len(months))
    reps = np.asarray(n_per)
    total = int(reps.sum())

    # Elevation and slope must be sampled grouped by raster tile, not in point
    # order: the DEM COGs are 32 MB each and points of a host are scattered
    # over many 1-degree tiles, so per-point ordering reloads a tile per point
    # (measured: 150 ms/point, i.e. ~20 min for a 7.7k-point host, vs ~1 s
    # when grouped).
    elev_all = elev_s.sample_many(lons, lats, conventions["elevation"])
    slope_all = np.full(len(records), NODATA, dtype=np.int64)
    by_tile: Dict[str, List[int]] = {}
    for ri, rec in enumerate(records):
        by_tile.setdefault(rec["tile"], []).append(ri)
    for tile, idxs in by_tile.items():
        for ri in idxs:
            slope_all[ri] = slope_s.sample(tile, lons[ri], lats[ri],
                                           conventions["slope"])

    def _build_block(i0: int, i1: int) -> pd.DataFrame:
        """Assemble the long-format frame for records[i0:i1]. Rows of one
        sample never straddle blocks, so per-block all-nodata dropping is
        identical to the global rule."""
        recs = records[i0:i1]
        reps_b = reps[i0:i1]
        total_b = int(reps_b.sum())
        cols: Dict[str, np.ndarray] = {
            b: np.full(total_b, NODATA, dtype=np.uint16)
            for b in S2_BANDS + S1_BANDS
            + ["slope", "elevation", "AGERA5-PRECIP", "AGERA5-TMEAN"]}
        ts_col = np.empty(total_b, dtype="datetime64[ns]")
        fidx_col = np.empty(total_b, dtype=np.int64)

        offs = 0
        for bi_r, rec in enumerate(recs):
            ri = i0 + bi_r
            months = tuple(map(tuple, axis_of[rec["sample_id"]][0]))
            T = len(months)
            sl = slice(offs, offs + T)
            for bi, b in enumerate(S2_BANDS):
                cols[b][sl] = rec["s2"][bi, :T]
            cols["S1-SIGMA0-VH"][sl] = rec["s1"][0, :T]
            cols["S1-SIGMA0-VV"][sl] = rec["s1"][1, :T]
            ts_col[sl] = ts_cache[months]
            fidx_col[sl] = ri
            cols["slope"][sl] = np.uint16(slope_all[ri])
            cols["elevation"][sl] = np.uint16(elev_all[ri])
            for mi, (yy, mm) in enumerate(months):
                cols["AGERA5-TMEAN"][offs + mi] = np.uint16(
                    min(np.floor(meteo_vals[(yy, mm, "T")][ri]), NODATA))
                cols["AGERA5-PRECIP"][offs + mi] = np.uint16(
                    min(np.floor(meteo_vals[(yy, mm, "P")][ri]), NODATA))
            offs += T

        # Per-record scalars, repeated per month row. np.repeat on object
        # arrays copies references, not values.
        rec_sids = np.array([r["sample_id"] for r in recs], dtype=object)
        a = attrs.loc[list(rec_sids)]
        geom_arr = np.array([pt_geom[s] for s in rec_sids], dtype=object)
        df = pd.DataFrame({
            "feature_index": fidx_col,
            "sample_id": np.repeat(rec_sids, reps_b),
            "timestamp": ts_col,
            **{b: cols[b] for b in S2_BANDS},
            "S1-SIGMA0-VH": cols["S1-SIGMA0-VH"],
            "S1-SIGMA0-VV": cols["S1-SIGMA0-VV"],
            "slope": cols["slope"],
            "elevation": cols["elevation"],
            "AGERA5-PRECIP": cols["AGERA5-PRECIP"],
            "AGERA5-TMEAN": cols["AGERA5-TMEAN"],
            "lon": np.repeat(np.array([g.x for g in geom_arr]), reps_b),
            "lat": np.repeat(np.array([g.y for g in geom_arr]), reps_b),
            "geometry": np.repeat(geom_arr, reps_b),
            "tile": np.repeat(
                np.array([r["tile"] for r in recs], dtype=object), reps_b),
            "s1_orbit": np.repeat(
                np.array([r["s1_orbit"] or "none" for r in recs], dtype=object),
                reps_b),
            "h3_l3_cell": np.repeat(
                a["h3_l3_cell"].astype(str).to_numpy(dtype=object), reps_b),
            "start_date": np.repeat(np.array(
                [axis_of[s][1] for s in rec_sids], dtype=object), reps_b),
            "end_date": np.repeat(np.array(
                [axis_of[s][2] for s in rec_sids], dtype=object), reps_b),
            "year": np.full(total_b, int(host_ref_id.split("_")[0]),
                            dtype=np.int64),
            "valid_time": np.repeat(
                a["valid_time"].astype(str).str[:10].to_numpy(dtype=object),
                reps_b),
            "ewoc_code": np.repeat(a["ewoc_code"].to_numpy(np.int64), reps_b),
            "irrigation_status": np.repeat(
                a["irrigation_status"].to_numpy(np.int64), reps_b),
            "quality_score_lc": np.repeat(
                a["quality_score_lc"].to_numpy(np.int64), reps_b),
            "quality_score_ct": np.repeat(
                a["quality_score_ct"].to_numpy(np.int64), reps_b),
            "extract": np.repeat(a["extract"].to_numpy(np.int64), reps_b),
            # point_kind ('centroid' | 'clipped' | 'point'): set by the RDM
            # campaign driver's hybrid placement rule; absent for callers that
            # do not provide it (in-patch campaign), keeping their schema.
            **({"point_kind": np.repeat(
                a["point_kind"].astype(str).to_numpy(dtype=object), reps_b)}
               if "point_kind" in a.columns else {}),
            # parent_sample_id: '' for normal samples, the parent's id for
            # multi-point children (point_kind='sampled'); driver-provided.
            **({"parent_sample_id": np.repeat(
                a["parent_sample_id"].astype(str).to_numpy(dtype=object),
                reps_b)}
               if "parent_sample_id" in a.columns else {}),
        })
        # all-nodata drop, same rule as post_job_action: every S2/S1 column
        # at NODATA for every timestep of the sample.
        sensor_cols = [c for c in df.columns if "S2" in c or "S1" in c]
        nodata_rows = (df[sensor_cols] == NODATA).all(axis=1)
        drop = df.assign(_nd=nodata_rows).groupby(
            "sample_id")["_nd"].transform("all")
        if drop.any():
            n = df.loc[drop, "sample_id"].nunique()
            logger.warning(f"{host_ref_id}: dropping {n} all-nodata sample(s)")
            df = df[~drop]
        df["ref_id"] = pd.Series(
            [host_ref_id] * len(df), index=df.index,
            dtype=pd.CategoricalDtype(categories=[host_ref_id], ordered=False))
        return df

    if total <= CHUNK_ROWS or out_path is None:
        df = _build_block(0, len(records))
        if out_path is not None:
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_parquet(out_path, index=False)
            logger.success(
                f"{host_ref_id}: wrote {df.sample_id.nunique():,} samples "
                f"/ {len(df):,} rows -> {out_path}")
        return df, records

    blocks: List[Tuple[int, int]] = []
    start, acc = 0, 0
    for ri in range(len(records)):
        acc += int(reps[ri])
        if acc >= CHUNK_ROWS:
            blocks.append((start, ri + 1))
            start, acc = ri + 1, 0
    if start < len(records):
        blocks.append((start, len(records)))
    logger.info(f"{host_ref_id}: {total:,} rows -> chunked write "
                f"({len(blocks)} blocks)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_blocks: List[Path] = []
    n_rows = 0
    n_samples = 0
    try:
        for (i0, i1) in blocks:
            df_b = _build_block(i0, i1)
            gdf_b = gpd.GeoDataFrame(df_b, geometry="geometry",
                                     crs="EPSG:4326")
            f = out_path.with_suffix(
                f".tmp{os.getpid()}.b{len(tmp_blocks)}.parquet")
            gdf_b.to_parquet(f, index=False)
            tmp_blocks.append(f)
            n_rows += len(df_b)
            n_samples += df_b["sample_id"].nunique()
            del df_b, gdf_b
        # Merge: stream block tables through one writer. read_schema keeps
        # the geoparquet 'geo' metadata geopandas wrote into block 0.
        schema = pq.read_schema(tmp_blocks[0])
        tmp_out = out_path.with_suffix(f".tmp{os.getpid()}.parquet")
        with pq.ParquetWriter(tmp_out, schema) as writer:
            for f in tmp_blocks:
                writer.write_table(pq.read_table(f, schema=schema))
        tmp_out.rename(out_path)
    finally:
        for f in tmp_blocks:
            f.unlink(missing_ok=True)
    logger.success(f"{host_ref_id}: wrote {n_samples:,} samples "
                   f"/ {n_rows:,} rows -> {out_path} "
                   f"({len(blocks)} blocks)")
    return None, records


# --- CLI ------------------------------------------------------------------


def main():
    global S2_ROOT, S1_ROOT, SLOPE_DIR, DEM_DIR, AGERA5_DAILY, AGERA5_S3
    global GT_DIR, MERGED_DIR, AGERA5_CACHE, RUN_SUFFIX
    global CONVENTIONS_FILE

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["extract"], default="extract",
                    help="kept for CLI compatibility; extraction is the only "
                         "mode (calibration was a one-time bootstrap, see "
                         "header)")
    ap.add_argument("--hosts", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample-limit", type=int, default=None)
    ap.add_argument(
        "--index-source", choices=["fs", "stac", "auto"], default="fs",
        help="patch index: 'fs' walks the extraction tree (original "
             "behaviour, minutes/ref on NFS); 'stac' paginates the STAC "
             "catalogue (seconds/ref); 'auto' = stac with fs fallback")
    ap.add_argument(
        "--catalog-cache", type=Path, default=None,
        help="with --index-source stac/auto: reuse/write "
             "<ref>.catalog.parquet files here (see ref_catalog.py)")
    # Campaign-specific locations (no defaults).
    ap.add_argument(
        "--gt-dir", type=Path, required=True,
        help="ground-truth dir holding <host>.geoparquet files plus "
             "provenance.parquet")
    ap.add_argument(
        "--merged-dir", type=Path, required=True,
        help="output dir for <host>_<run-suffix>.geoparquet; also holds the "
             "conventions file and the calibration report")
    ap.add_argument(
        "--run-suffix", type=str, default=RUN_SUFFIX,
        help="suffix in the output filename <host>_<suffix>.geoparquet "
             "(default: %(default)s); must match the --run-suffix used by "
             "ptp_campaign_inpatch.py at rekey/gate time")
    ap.add_argument(
        "--agera5-cache", type=Path, default=None,
        help="cache dir for AGERA5 monthly composites "
             "(default: <merged-dir>/_agera5_cache)")
    ap.add_argument(
        "--conventions", type=str, default=None,
        help="JSON file with locked conventions (default: "
             "<merged-dir>/_local_extractor_conventions.json if present, "
             "else built-in defaults)")
    # Stable Terrascope/project locations (defaults correct on Terrascope).
    ap.add_argument("--s2-root", type=Path, default=S2_ROOT,
                    help="S2 patch NetCDF root (default: %(default)s)")
    ap.add_argument("--s1-root", type=Path, default=S1_ROOT,
                    help="S1 patch NetCDF root (default: %(default)s)")
    ap.add_argument("--slope-dir", type=Path, default=SLOPE_DIR,
                    help="Terrascope 20 m slope tiles (default: %(default)s)")
    ap.add_argument("--dem-dir", type=Path, default=DEM_DIR,
                    help="Copernicus 30 m DEM COGs (default: %(default)s)")
    ap.add_argument("--agera5-daily", type=Path, default=AGERA5_DAILY,
                    help="AGERA5 local daily archive, fallback compositing "
                         "source (default: %(default)s)")
    ap.add_argument("--agera5-s3", type=str, default=AGERA5_S3,
                    help="AGERA5 monthly composite S3 prefix "
                         "(default: %(default)s)")
    args = ap.parse_args()

    S2_ROOT, S1_ROOT = args.s2_root, args.s1_root
    SLOPE_DIR, DEM_DIR = args.slope_dir, args.dem_dir
    AGERA5_DAILY, AGERA5_S3 = args.agera5_daily, args.agera5_s3
    GT_DIR, MERGED_DIR = args.gt_dir, args.merged_dir
    AGERA5_CACHE = args.agera5_cache or MERGED_DIR / "_agera5_cache"
    RUN_SUFFIX = args.run_suffix
    CONVENTIONS_FILE = MERGED_DIR / "_local_extractor_conventions.json"

    conv = DEFAULT_CONVENTIONS
    conv_path = Path(args.conventions) if args.conventions else CONVENTIONS_FILE
    if conv_path.exists():
        conv = json.loads(conv_path.read_text())
        logger.info(f"Loaded conventions from {conv_path}")

    for host in args.hosts:
        out_path = MERGED_DIR / f"{host}_{RUN_SUFFIX}.geoparquet"
        if out_path.exists():
            logger.warning(f"{host}: output exists, skipping ({out_path})")
            continue
        extract_host(host, conv, workers=args.workers, out_path=out_path,
                         sample_limit=args.sample_limit,
                         index_source=args.index_source,
                         catalog_cache=args.catalog_cache)
    logger.success("Done.")


if __name__ == "__main__":
    main()
