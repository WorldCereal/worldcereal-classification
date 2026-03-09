"""
season_overlap.py
-----------------
Computes season overlap between S1 and S2 crop calendars from rasters
(day-of-year encoded, nodata=0) and produces:

  1.  distinct_seasons.tif  -- uint8, 1=distinct, 0=overlapping, 255=nodata
  2.  S1_SOS_merged.tif     -- S1 SOS with nodata where seasons are not distinct
  3.  S1_EOS_merged.tif     -- S1 EOS expanded to cover S2 where seasons are not distinct
                               (i.e. in non-distinct pixels, EOS = max(S1_EOS, S2_EOS))
  4.  S2_SOS_masked.tif     -- S2 SOS masked (nodata) where seasons are not distinct
  5.  S2_EOS_masked.tif     -- S2 EOS masked (nodata) where seasons are not distinct
  6.  S1_LOS_before.tif     -- S1 length-of-season in days (original)
  7.  S2_LOS_before.tif     -- S2 length-of-season in days (original, nodata=0)
  8.  S1_LOS_after.tif      -- S1 length-of-season after expansion (merged)
  9.  S2_LOS_after.tif      -- S2 length-of-season after masking (nodata=0 where non-distinct)

Overlap is computed purely per-pixel on the rasters (day-of-year values).
Cross-year wrapping is handled: if EOS < SOS the season wraps around the
year boundary (e.g. sowing in day 300, harvest in day 60 of next year).

Usage (CLI)
-----------
python season_overlap.py \\
    --s1-sos  /path/to/S1_SOS_WGS84.tif \\
    --s1-eos  /path/to/S1_EOS_WGS84.tif \\
    --s2-sos  /path/to/S2_SOS_WGS84.tif \\
    --s2-eos  /path/to/S2_EOS_WGS84.tif \\
    --out-dir /path/to/output/ \\
    [--overlap-threshold-days  60] \\
    [--overlap-threshold-frac  0.30] \\
    [--threshold-mode          days|fraction|either|both]

Threshold logic (--threshold-mode):
  days      : distinct if overlap_days  < threshold_days
  fraction  : distinct if overlap_frac  < threshold_frac  (fraction of longer season)
  either    : distinct if EITHER condition holds  (more permissive → more distinct)
  both      : distinct if BOTH conditions hold    (more strict   → fewer distinct)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

def _doy_season_length(sos: np.ndarray, eos: np.ndarray) -> np.ndarray:
    """
    Season length in days, handling cross-year wrap (eos < sos → +365).
    Both arrays are int16 day-of-year (1-365).  Returns float32.
    """
    length = (eos - sos).astype(np.float32)
    length[length < 0] += 365.0  # wrap-around year boundary
    return length


def _compute_los(sos: np.ndarray, eos: np.ndarray, nodata_val: int = 0) -> np.ndarray:
    """
    Compute length-of-season (days) from DOY arrays.
    Pixels where sos or eos == nodata_val are returned as 0 (nodata).
    Cross-year wrap (eos < sos) is handled by adding 365.
    Returns int16 array.
    """
    nd_mask = (sos == nodata_val) | (eos == nodata_val)
    los = _doy_season_length(sos, eos)  # float32
    los = np.where(nd_mask, 0, np.round(los)).astype(np.int16)
    return los


def _doy_overlap_days(
    s1_sos: np.ndarray,
    s1_eos: np.ndarray,
    s2_sos: np.ndarray,
    s2_eos: np.ndarray,
) -> np.ndarray:
    """
    Per-pixel overlap in days between two seasons expressed as day-of-year.
    Cross-year wrapping is supported by projecting everything onto a
    0..730-day timeline anchored at s1_sos.

    Returns float32 array (0 where no overlap, positive where overlap).
    """
    shape = s1_sos.shape
    overlap = np.zeros(shape, dtype=np.float32)

    # Flatten for vectorised work
    ss1 = s1_sos.astype(np.float32).ravel()
    se1 = s1_eos.astype(np.float32).ravel()
    ss2 = s2_sos.astype(np.float32).ravel()
    se2 = s2_eos.astype(np.float32).ravel()

    # Expand eos if season wraps (eos < sos)
    se1_exp = np.where(se1 < ss1, se1 + 365.0, se1)
    se2_exp = np.where(se2 < ss2, se2 + 365.0, se2)

    # Shift s2 into the same reference frame as s1
    # If s2_sos is before s1_sos, add 365 so we're in the same cycle
    shift = np.where(ss2 < ss1, 365.0, 0.0)
    ss2_shifted = ss2 + shift
    se2_shifted = se2_exp + shift

    # Standard interval overlap: max(0, min(e1, e2) - max(s1, s2))
    ov = np.maximum(0.0, np.minimum(se1_exp, se2_shifted) - np.maximum(ss1, ss2_shifted))

    # Also try shifting s2 the other way in case s1 wraps but s2 doesn't
    shift2 = np.where(ss2 > se1_exp, -365.0, 0.0)
    ss2_s2 = ss2 + shift2
    se2_s2 = se2_exp + shift2
    ov2 = np.maximum(0.0, np.minimum(se1_exp, se2_s2) - np.maximum(ss1, ss2_s2))

    ov_flat = np.maximum(ov, ov2)
    return ov_flat.reshape(shape)


def compute_distinct_seasons(
    s1_sos: np.ndarray,
    s1_eos: np.ndarray,
    s2_sos: np.ndarray,
    s2_eos: np.ndarray,
    nodata_val: int = 0,
    threshold_days: float = 100.0,
    threshold_frac: float = 0.35,
    threshold_mode: str = "both",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pixel distinct-season flag.

    Parameters
    ----------
    s1_sos, s1_eos, s2_sos, s2_eos : int16 day-of-year arrays (nodata=0)
    nodata_val   : value treated as nodata in all input rasters
    threshold_days : overlap in days below which seasons are 'distinct'
    threshold_frac : overlap as fraction of longer season below which distinct
    threshold_mode : 'days' | 'fraction' | 'either' | 'both'

    Returns
    -------
    distinct   : uint8 array  (1=distinct, 0=not distinct, 255=nodata)
    overlap_days : float32 array of computed overlap days (nan where nodata)
    overlap_frac : float32 array of overlap fraction of longer season (nan where nodata)
    """
    nodata_mask = (
        (s1_sos == nodata_val)
        | (s1_eos == nodata_val)
        | (s2_sos == nodata_val)
        | (s2_eos == nodata_val)
    )

    ov_days = _doy_overlap_days(s1_sos, s1_eos, s2_sos, s2_eos)

    s1_len = _doy_season_length(s1_sos, s1_eos)
    s2_len = _doy_season_length(s2_sos, s2_eos)
    max_len = np.maximum(s1_len, s2_len)

    with np.errstate(invalid="ignore", divide="ignore"):
        ov_frac = np.where(max_len > 0, ov_days / max_len, np.nan)

    # Apply threshold logic
    dist_days = ov_days < threshold_days
    dist_frac = ov_frac < threshold_frac

    if threshold_mode == "days":
        is_distinct = dist_days
    elif threshold_mode == "fraction":
        is_distinct = dist_frac
    elif threshold_mode == "either":
        is_distinct = dist_days | dist_frac
    else:  # "both"
        is_distinct = dist_days & dist_frac

    distinct = np.where(nodata_mask, 255, np.where(is_distinct, 1, 0)).astype(np.uint8)

    ov_days_out = np.where(nodata_mask, np.nan, ov_days).astype(np.float32)
    ov_frac_out = np.where(nodata_mask, np.nan, ov_frac.astype(np.float32))

    return distinct, ov_days_out, ov_frac_out


def merge_seasons(
    s1_sos: np.ndarray,
    s1_eos: np.ndarray,
    s2_sos: np.ndarray,
    s2_eos: np.ndarray,
    distinct: np.ndarray,
    nodata_val: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Where seasons are NOT distinct (distinct==0), expand the longer of S1/S2
    symmetrically to exactly 365 days and store the result as the merged S1 season.
    S2 is masked with nodata_val.

    Strategy
    --------
    1. Compute the length-of-season (LOS) for both S1 and S2 in absolute days
       (cross-year wrap handled by adding 365 when EOS < SOS).
    2. Choose the season with the greater LOS as the *base* season.
    3. Calculate padding = 365 - base_LOS  (split evenly: floor(pad/2) on the
       SOS side, ceil(pad/2) on the EOS side so the total is always exactly 365).
    4. The merged SOS = base_SOS - pad_sos  (in DOY, wrapping into previous year
       if needed, but stored mod 365 in the 1-365 range).
    5. The merged EOS is implied: base_EOS + pad_eos, similarly wrapped.
    6. Where base_LOS >= 365 already (degenerate near-full-year season), no
       expansion is done — the season is kept at 365 days exactly.

    Where distinct==1 or nodata (255): keep original values unchanged.

    Returns
    -------
    s1_sos_out, s1_eos_out, s2_sos_out, s2_eos_out  -- all int16
    """
    not_distinct = distinct == 0  # pixels to merge

    s1_sos_f = s1_sos.astype(np.float32)
    s1_eos_f = s1_eos.astype(np.float32)
    s2_sos_f = s2_sos.astype(np.float32)
    s2_eos_f = s2_eos.astype(np.float32)

    # --- Absolute LOS for each season (handle cross-year wrap per season) ---
    s1_eos_abs = np.where(s1_eos_f < s1_sos_f, s1_eos_f + 365.0, s1_eos_f)
    s2_eos_abs = np.where(s2_eos_f < s2_sos_f, s2_eos_f + 365.0, s2_eos_f)
    s1_los = s1_eos_abs - s1_sos_f   # float, may be 0 for degenerate pixels
    s2_los = s2_eos_abs - s2_sos_f

    # --- Choose the longer season as the base ---
    s1_is_longer = s1_los >= s2_los
    base_sos = np.where(s1_is_longer, s1_sos_f, s2_sos_f)
    base_los = np.where(s1_is_longer, s1_los, s2_los)

    # --- Compute symmetric padding to reach 365 days ---
    # Clamp base_los to 365 so we never get negative padding.
    base_los_clamped = np.minimum(base_los, 365.0)
    padding = 365.0 - base_los_clamped
    pad_sos = np.floor(padding / 2.0)    # days pulled earlier on SOS side
    pad_eos = padding - pad_sos          # days pushed later on EOS side (ceil)

    # --- New SOS: pull back by pad_sos, wrap into 1-365 range ---
    new_sos_raw = base_sos - pad_sos
    # Wrap: if new_sos_raw <= 0, it crossed Jan 1 into previous year
    new_sos = np.where(new_sos_raw <= 0, new_sos_raw + 365.0, new_sos_raw)
    new_sos = np.round(new_sos).astype(np.int16)

    # --- New EOS: base_EOS (abs) + pad_eos, wrap into 1-365 range ---
    base_eos_abs = base_sos + base_los_clamped
    new_eos_abs = base_eos_abs + pad_eos
    new_eos_raw = np.where(new_eos_abs > 365.0, new_eos_abs - 365.0, new_eos_abs)
    new_eos = np.round(new_eos_raw).astype(np.int16)

    # --- Apply only where not distinct ---
    s1_sos_out = np.where(not_distinct, new_sos, s1_sos).astype(np.int16)
    s1_eos_out = np.where(not_distinct, new_eos, s1_eos).astype(np.int16)

    # --- Mask S2 with nodata where not distinct ---
    s2_sos_out = np.where(not_distinct, nodata_val, s2_sos).astype(np.int16)
    s2_eos_out = np.where(not_distinct, nodata_val, s2_eos).astype(np.int16)

    return s1_sos_out, s1_eos_out, s2_sos_out, s2_eos_out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read first band and return (data, profile)."""
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def _write_raster(path: str | Path, data: np.ndarray, profile: dict) -> None:
    """Write single-band raster."""
    out_profile = profile.copy()
    out_profile.update(
        dtype=data.dtype,
        count=1,
        compress="lzw",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Main pipeline function (importable)
# ---------------------------------------------------------------------------

def run_season_overlap_pipeline(
    s1_sos_path: str,
    s1_eos_path: str,
    s2_sos_path: str,
    s2_eos_path: str,
    out_dir: str,
    threshold_days: float = 100.0,
    threshold_frac: float = 0.35,
    threshold_mode: str = "both",
    nodata_val: int = 0,
) -> dict[str, str]:
    """
    Full pipeline: read rasters → compute overlap → write outputs.

    Returns dict mapping output name → file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading input rasters...")
    s1_sos, profile = _read_raster(s1_sos_path)
    s1_eos, _ = _read_raster(s1_eos_path)
    s2_sos, _ = _read_raster(s2_sos_path)
    s2_eos, _ = _read_raster(s2_eos_path)

    print(f"Raster shape: {s1_sos.shape}")
    print(f"Threshold: {threshold_days} days  /  {threshold_frac*100:.0f}% fraction  [mode={threshold_mode}]")

    # --- Step 1: compute distinct flag ---
    print("Computing season overlap...")
    distinct, ov_days, ov_frac = compute_distinct_seasons(
        s1_sos, s1_eos, s2_sos, s2_eos,
        nodata_val=nodata_val,
        threshold_days=threshold_days,
        threshold_frac=threshold_frac,
        threshold_mode=threshold_mode,
    )

    valid = distinct != 255
    n_distinct = int((distinct == 1).sum())
    n_overlap = int((distinct == 0).sum())
    n_nodata = int((distinct == 255).sum())
    print(f"  Distinct pixels  : {n_distinct:,}  ({n_distinct / max(valid.sum(), 1) * 100:.1f}%)")
    print(f"  Overlapping pixels: {n_overlap:,}  ({n_overlap / max(valid.sum(), 1) * 100:.1f}%)")
    print(f"  Nodata pixels    : {n_nodata:,}")

    # --- Step 2: produce merged/masked seasons ---
    print("Merging/masking seasons for non-distinct pixels...")
    s1_sos_m, s1_eos_m, s2_sos_m, s2_eos_m = merge_seasons(
        s1_sos, s1_eos, s2_sos, s2_eos, distinct, nodata_val=nodata_val
    )

    # --- Step 3: write outputs ---
    print("Writing outputs...")
    out_paths: dict[str, str] = {}

    # distinct_seasons.tif
    dist_profile = profile.copy()
    dist_profile.update(dtype="uint8", nodata=255)
    p = out_dir / "distinct_seasons.tif"
    _write_raster(p, distinct, dist_profile)
    out_paths["distinct_seasons"] = str(p)

    # Merged season rasters (same profile as inputs: int16, nodata=0)
    for name, arr in [
        ("S1_SOS_merged", s1_sos_m),
        ("S1_EOS_merged", s1_eos_m),
        ("S2_SOS_masked", s2_sos_m),
        ("S2_EOS_masked", s2_eos_m),
    ]:
        p = out_dir / f"{name}.tif"
        _write_raster(p, arr, profile)
        out_paths[name] = str(p)

    # --- Step 4: compute and write LOS rasters (before and after) ---
    print("Computing length-of-season rasters...")
    los_profile = profile.copy()
    los_profile.update(dtype="int16", nodata=0)

    s1_los_before = _compute_los(s1_sos, s1_eos, nodata_val=nodata_val)
    s2_los_before = _compute_los(s2_sos, s2_eos, nodata_val=nodata_val)

    # S1_LOS_after: the merged season is always exactly 365 days for non-distinct
    # pixels (by construction in merge_seasons). For distinct pixels the original
    # S1 season is unchanged. Use _compute_los on the merged arrays but guard
    # against the degenerate wrap case where merged SOS ≈ merged EOS by using
    # the known 365-day result directly where not_distinct.
    nd_mask_s1 = (s1_sos == nodata_val) | (s1_eos == nodata_val)
    not_distinct_mask = (distinct == 0) & ~nd_mask_s1
    s1_los_after = _compute_los(s1_sos, s1_eos, nodata_val=nodata_val)  # start from before
    s1_los_after = np.where(not_distinct_mask, 365, s1_los_after).astype(np.int16)

    s2_los_after = _compute_los(s2_sos_m, s2_eos_m, nodata_val=nodata_val)

    for name, arr in [
        ("S1_LOS_before", s1_los_before),
        ("S2_LOS_before", s2_los_before),
        ("S1_LOS_after",  s1_los_after),
        ("S2_LOS_after",  s2_los_after),
    ]:
        p = out_dir / f"{name}.tif"
        _write_raster(p, arr, los_profile)
        out_paths[name] = str(p)

    print("Done.")
    return out_paths


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_season_overlap(
    out_dir: str = None,
    s1_sos_before: str = None,
    s1_eos_before: str = None,
    s2_sos_before: str = None,
    s2_eos_before: str = None,
    s1_sos_after: str = None,
    s1_eos_after: str = None,
    s2_sos_after: str = None,
    s2_eos_after: str = None,
    s1_los_before: str = None,
    s2_los_before: str = None,
    s1_los_after: str = None,
    s2_los_after: str = None,
    distinct_seasons: str = None,
    figsize: tuple = (22, 28),
    cmap_doy: str = "twilight_shifted",
    cmap_los: str = "YlOrRd",
    nodata_val: int = 0,
    vmin_doy: int = 1,
    vmax_doy: int = 365,
    vmin_los: int = 0,
    vmax_los: int = 365,
    save_path: Optional[str] = None,
    dpi: int = 120,
) -> None:
    """
    Side-by-side visualisation of season rasters before and after the
    overlap merging/masking step.

    Panels (rows × cols  =  5 × 4):
      Row 0  : S1 SOS  before | S1 SOS  after  | S2 SOS  before | S2 SOS  after
      Row 1  : S1 EOS  before | S1 EOS  after  | S2 EOS  before | S2 EOS  after
      Row 2  : S1 LOS  before | S1 LOS  after  | S2 LOS  before | S2 LOS  after
      Row 3  : (distinct_seasons spanning all 4 columns)
      (optional colour-bars on the right)

    All path arguments default to ``<out_dir>/<name>.tif`` from the pipeline
    when ``out_dir`` is provided and the individual path is not set.

    Parameters
    ----------
    out_dir :
        Directory where ``run_season_overlap_pipeline`` wrote its outputs.
        Used to build default file paths when individual paths are not given.
    s1_sos_before, s1_eos_before, s2_sos_before, s2_eos_before :
        Original SOS / EOS rasters for S1 and S2 (DOY 1-365, nodata=0).
    s1_sos_after, s1_eos_after, s2_sos_after, s2_eos_after :
        Post-pipeline merged/masked SOS / EOS rasters.
    s1_los_before, s2_los_before, s1_los_after, s2_los_after :
        Length-of-season rasters (days, nodata=0).
    distinct_seasons :
        distinct_seasons.tif (uint8, 1=distinct, 0=overlap, 255=nodata).
    figsize : tuple
        Overall figure size (width, height) in inches.
    cmap_doy : str
        Colormap for day-of-year panels.
    cmap_los : str
        Colormap for length-of-season panels.
    nodata_val : int
        Value to mask in DOY / LOS rasters before display.
    vmin_doy, vmax_doy :
        Colour scale range for DOY panels.
    vmin_los, vmax_los :
        Colour scale range for LOS panels.
    save_path : str, optional
        If set, the figure is saved to this path instead of (only) shown.
    dpi : int
        Resolution for saved figure.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    # ---- resolve default paths via out_dir ----
    def _resolve(explicit: Optional[str], key: str) -> Optional[str]:
        if explicit is not None:
            return explicit
        if out_dir is not None:
            candidate = Path(out_dir) / f"{key}.tif"
            if candidate.exists():
                return str(candidate)
        return None

    _CAL_DIR = (
        Path(__file__).resolve().parent.parent
        / "worldcereal-classification/src/worldcereal/data/cropcalendars"
    )

    paths = {
        "S1_SOS_before":    _resolve(s1_sos_before, "S1_SOS_WGS84") or str(_CAL_DIR / "S1_SOS_WGS84.tif"),
        "S1_EOS_before":    _resolve(s1_eos_before, "S1_EOS_WGS84") or str(_CAL_DIR / "S1_EOS_WGS84.tif"),
        "S2_SOS_before":    _resolve(s2_sos_before, "S2_SOS_WGS84") or str(_CAL_DIR / "S2_SOS_WGS84.tif"),
        "S2_EOS_before":    _resolve(s2_eos_before, "S2_EOS_WGS84") or str(_CAL_DIR / "S2_EOS_WGS84.tif"),
        "S1_SOS_after":     _resolve(s1_sos_after,  "S1_SOS_merged"),
        "S1_EOS_after":     _resolve(s1_eos_after,  "S1_EOS_merged"),
        "S2_SOS_after":     _resolve(s2_sos_after,  "S2_SOS_masked"),
        "S2_EOS_after":     _resolve(s2_eos_after,  "S2_EOS_masked"),
        "S1_LOS_before":    _resolve(s1_los_before, "S1_LOS_before"),
        "S2_LOS_before":    _resolve(s2_los_before, "S2_LOS_before"),
        "S1_LOS_after":     _resolve(s1_los_after,  "S1_LOS_after"),
        "S2_LOS_after":     _resolve(s2_los_after,  "S2_LOS_after"),
        "distinct_seasons": _resolve(distinct_seasons, "distinct_seasons"),
    }

    # ---- read helper ----
    def _load(key: str) -> Optional[np.ndarray]:
        p = paths.get(key)
        if p is None or not Path(p).exists():
            print(f"[WARN] Raster not found, skipping: {key}  (path={p})")
            return None
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
            nd = src.nodata
        arr = np.where(arr == nodata_val, np.nan, arr)
        if nd is not None and not np.isnan(float(nd)):
            arr = np.where(arr == nd, np.nan, arr)
        return arr

    # -----------------------------------------------------------------------
    # Layout: 2 columns (before | after), one row per variable × season.
    # Row order:
    #   0  S1 SOS
    #   1  S1 EOS
    #   2  S1 LOS
    #   3  S2 SOS
    #   4  S2 EOS
    #   5  S2 LOS
    #   6  distinct seasons  (spans both columns)
    # -----------------------------------------------------------------------
    rows = [
        # (row_label,   cmap,     vmin,    vmax,    key_before,     key_after)
        ("S1  SOS",  cmap_doy, vmin_doy, vmax_doy, "S1_SOS_before", "S1_SOS_after"),
        ("S1  EOS",  cmap_doy, vmin_doy, vmax_doy, "S1_EOS_before", "S1_EOS_after"),
        ("S1  LOS",  cmap_los, vmin_los, vmax_los, "S1_LOS_before", "S1_LOS_after"),
        ("S2  SOS",  cmap_doy, vmin_doy, vmax_doy, "S2_SOS_before", "S2_SOS_after"),
        ("S2  EOS",  cmap_doy, vmin_doy, vmax_doy, "S2_EOS_before", "S2_EOS_after"),
        ("S2  LOS",  cmap_los, vmin_los, vmax_los, "S2_LOS_before", "S2_LOS_after"),
    ]
    n_map_rows = len(rows)

    fig, axes = plt.subplots(
        n_map_rows + 1,   # +1 for distinct row
        2,
        figsize=figsize,
        gridspec_kw={"hspace": 0.22, "wspace": 0.04},
    )

    extent = [-180, 180, -90, 90]

    def _imshow(ax, data, cmap, vmin, vmax, title):
        if data is None:
            ax.set_visible(False)
            return None
        im = ax.imshow(
            data, cmap=cmap, vmin=vmin, vmax=vmax,
            extent=extent, origin="upper", interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        return im

    last_doy_im = None
    last_los_im = None

    for row_idx, (lbl, cmap_, vmin_, vmax_, key_b, key_a) in enumerate(rows):
        im_b = _imshow(axes[row_idx, 0], _load(key_b), cmap_, vmin_, vmax_, f"{lbl}  —  before")
        im_a = _imshow(axes[row_idx, 1], _load(key_a), cmap_, vmin_, vmax_, f"{lbl}  —  after")
        im = im_b or im_a
        if im is not None:
            if cmap_ == cmap_doy:
                last_doy_im = im
            else:
                last_los_im = im
        # shared colourbar for this row, attached to right side of right panel
        if im is not None:
            cb = fig.colorbar(im, ax=axes[row_idx, 1], fraction=0.035, pad=0.02, shrink=0.95)
            cb.ax.tick_params(labelsize=7)
            if cmap_ == cmap_doy:
                cb.set_label("Day of year", fontsize=7)
            else:
                cb.set_label("Days", fontsize=7)

    # ---- last row: distinct seasons spanning both columns ----
    ax_dist_l = axes[n_map_rows, 0]
    ax_dist_r = axes[n_map_rows, 1]
    # hide the right cell and expand the left one manually
    ax_dist_r.set_visible(False)

    bb_l = ax_dist_l.get_position()
    bb_r = ax_dist_r.get_position()
    ax_dist_l.set_position([
        bb_l.x0, bb_l.y0,
        bb_r.x1 - bb_l.x0,   # full width of both columns
        bb_l.height,
    ])

    distinct_arr = _load("distinct_seasons")
    if distinct_arr is not None:
        cmap_dist = mcolors.ListedColormap(["#d7191c", "#1a9641"])
        norm_dist  = mcolors.BoundaryNorm([0, 0.5, 1.5], cmap_dist.N)
        ax_dist_l.imshow(
            distinct_arr, cmap=cmap_dist, norm=norm_dist,
            extent=extent, origin="upper", interpolation="nearest",
            aspect="auto",
        )
        legend_patches = [
            Patch(facecolor="#1a9641", label="Distinct seasons (1)"),
            Patch(facecolor="#d7191c", label="Overlapping seasons (0)"),
            Patch(facecolor="white", edgecolor="grey", linewidth=0.6, label="Nodata"),
        ]
        ax_dist_l.legend(handles=legend_patches, loc="lower left", fontsize=8,
                         framealpha=0.85, edgecolor="grey")
    ax_dist_l.set_title("Distinct seasons  (green = distinct  |  red = overlapping)",
                        fontsize=9, pad=4)
    ax_dist_l.set_xticks([])
    ax_dist_l.set_yticks([])
    for spine in ax_dist_l.spines.values():
        spine.set_linewidth(0.5)

    # ---- column headers ----
    axes[0, 0].annotate("Before", xy=(0.5, 1.12), xycoords="axes fraction",
                        ha="center", fontsize=11, fontweight="bold")
    axes[0, 1].annotate("After",  xy=(0.5, 1.12), xycoords="axes fraction",
                        ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Season overlap analysis — SOS / EOS / LOS  before vs after merging/masking",
        fontsize=12, y=0.995, fontweight="bold",
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute season overlap and produce distinct_seasons.tif + merged season rasters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cal_dir = (
        Path(__file__).resolve().parent.parent
        / "worldcereal-classification/src/worldcereal/data/cropcalendars"
    )
    p.add_argument("--s1-sos",  default=str(cal_dir / "S1_SOS_WGS84.tif"),  help="S1 start-of-season raster (DOY)")
    p.add_argument("--s1-eos",  default=str(cal_dir / "S1_EOS_WGS84.tif"),  help="S1 end-of-season raster (DOY)")
    p.add_argument("--s2-sos",  default=str(cal_dir / "S2_SOS_WGS84.tif"),  help="S2 start-of-season raster (DOY)")
    p.add_argument("--s2-eos",  default=str(cal_dir / "S2_EOS_WGS84.tif"),  help="S2 end-of-season raster (DOY)")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "season_overlap_output"),
                   help="Output directory")
    p.add_argument("--overlap-threshold-days",  type=float, default=60.0,
                   help="Overlap in days below which seasons are considered distinct")
    p.add_argument("--overlap-threshold-frac",  type=float, default=0.30,
                   help="Overlap as fraction of longer season below which distinct (0–1)")
    p.add_argument("--threshold-mode", choices=["days", "fraction", "either", "both"],
                   default="both",
                   help="How to combine the two thresholds: "
                        "'days'=only day threshold, 'fraction'=only fraction threshold, "
                        "'either'=distinct if either holds, 'both'=distinct only if both hold")
    p.add_argument("--nodata", type=int, default=0,
                   help="Nodata value in input rasters")
    return p


def main(args=None):
    parser = _build_parser()
    ns = parser.parse_args(args)
    run_season_overlap_pipeline(
        s1_sos_path=ns.s1_sos,
        s1_eos_path=ns.s1_eos,
        s2_sos_path=ns.s2_sos,
        s2_eos_path=ns.s2_eos,
        out_dir=ns.out_dir,
        threshold_days=ns.overlap_threshold_days,
        threshold_frac=ns.overlap_threshold_frac,
        threshold_mode=ns.threshold_mode,
        nodata_val=ns.nodata,
    )


if __name__ == "__main__":
    main()
