"""season_overlap — Season overlap analysis and zonal aggregation.

Computes season overlap between S1 and S2 crop calendars from
day-of-year rasters (nodata=0), optionally smooths the result with a
median filter, and optionally aggregates to spatial zones.

Outputs
-------
  distinct_seasons_raw.tif  -- pixel-level classification before smoothing/zones
  distinct_seasons.tif      -- final uint8: 1=distinct, 0=overlapping, 255=nodata
  S1_SOS_merged.tif         -- S1 SOS (expanded to 365-day season where overlapping)
  S1_EOS_merged.tif         -- S1 EOS (expanded symmetrically where overlapping)
  S2_SOS_masked.tif         -- S2 SOS masked (nodata) where overlapping
  S2_EOS_masked.tif         -- S2 EOS masked (nodata) where overlapping
  S1/S2_LOS_before.tif      -- length-of-season (days) before merging
  S1/S2_LOS_after.tif       -- length-of-season (days) after merging
  zone_stats.parquet         -- per-zone overlap statistics (when zones provided)
  zone_id_raster.tif         -- numeric zone-ID raster for traceability
  zone_id_legend.json        -- JSON mapping numeric IDs → zone names

Pipeline stages
---------------
  1. Per-pixel overlap computation (cross-year wrapping handled)
  2. Spatial smoothing via median filter (default kernel=3)
  3. Zonal aggregation: per-zone majority vote with configurable threshold
  4. Season merging: longer season expanded to 365 days, S2 masked

Zone helpers
------------
  ``polygonize_classification_raster()`` — convert a classification raster
      (e.g. GAEZ, Köppen-Geiger) to polygon GeoDataFrame.
  ``intersect_zone_layers()`` — overlay multiple zone GeoDataFrames to
      produce composite zones (e.g. Country × GAEZ).

Usage (CLI)
-----------
::

    python -m worldcereal.season_overlap \\
        --s1-sos  S1_SOS_WGS84.tif --s1-eos  S1_EOS_WGS84.tif \\
        --s2-sos  S2_SOS_WGS84.tif --s2-eos  S2_EOS_WGS84.tif \\
        --out-dir output/ \\
        [--overlap-threshold-days 100] [--overlap-threshold-frac 0.35] \\
        [--threshold-mode both] [--smooth-kernel 5]

Threshold modes (--threshold-mode):
  days      : distinct if overlap_days  < threshold_days
  fraction  : distinct if overlap_frac  < threshold_frac
  either    : distinct if EITHER condition holds  (more permissive)
  both      : distinct if BOTH conditions hold    (stricter, default)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

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
# Spatial smoothing
# ---------------------------------------------------------------------------

def smooth_distinct_seasons(
    distinct: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply a median filter to the distinct_seasons raster to remove
    salt-and-pepper noise (isolated pixels that flip between
    distinct/overlapping while their neighbours are uniform).

    Only operates on valid pixels (0 or 1); nodata (255) is preserved.

    Parameters
    ----------
    distinct : uint8 array (1=distinct, 0=overlapping, 255=nodata)
    kernel_size : int
        Size of the square median filter window.  3 means a 3×3 kernel,
        5 means 5×5 etc.  Set to 0 or 1 to skip smoothing entirely.

    Returns
    -------
    uint8 array with the same shape / nodata convention.
    """
    if kernel_size <= 1:
        return distinct.copy()

    try:
        from scipy.ndimage import median_filter
    except ImportError:
        print("[WARN] scipy not available — skipping spatial smoothing.")
        return distinct.copy()

    nodata_mask = distinct == 255
    valid_mask = ~nodata_mask

    # Work on a float copy: 1=distinct, 0=overlap, nan=nodata
    work = distinct.astype(np.float32)
    work[nodata_mask] = np.nan

    # Use median filter which is ideal for binary salt-and-pepper removal.
    # We need to handle NaN carefully: replace with -1 temporarily.
    work_filled = np.where(nodata_mask, -1.0, work)
    filtered = median_filter(work_filled, size=kernel_size)

    # Restore: only update valid pixels, keep nodata unchanged
    result = distinct.copy()
    result[valid_mask] = np.where(filtered[valid_mask] >= 0.5, 1, 0).astype(np.uint8)
    result[nodata_mask] = 255

    return result


# ---------------------------------------------------------------------------
# Zone creation helpers (raster → vector, intersect, clean)
# ---------------------------------------------------------------------------

def polygonize_classification_raster(
    raster_path: str,
    class_names: Optional[dict[int, str]] = None,
    nodata_val: int = 0,
    target_resolution: Optional[float] = None,
    simplify_tolerance: float = 0.05,
    min_area_deg2: float = 0.01,
) -> "gpd.GeoDataFrame":
    """Convert a classified raster (uint8) to a polygon GeoDataFrame.

    The raster is optionally downsampled first (to avoid millions of tiny
    polygons from 1 km data), then polygonized, simplified, and cleaned.

    Parameters
    ----------
    raster_path : str
        Path to the input classification raster (uint8, EPSG:4326).
    class_names : dict[int, str], optional
        Mapping from raster value → human-readable class name.
        If None, uses ``"class_<value>"`` as the name.
    nodata_val : int
        Raster value to treat as nodata (excluded from output).
    target_resolution : float, optional
        If set, the raster is first resampled to this resolution (degrees)
        using nearest-neighbour before polygonizing. Recommended for 1 km
        inputs (~0.008333°) to reduce polygon count. E.g. 0.1° ≈ 11 km.
    simplify_tolerance : float
        Douglas-Peucker tolerance in degrees for simplifying polygon edges.
    min_area_deg2 : float
        Minimum polygon area in square degrees. Smaller polygons (slivers)
        are dropped.

    Returns
    -------
    gpd.GeoDataFrame with columns: zone_value (int), zone_name (str), geometry.
    CRS = EPSG:4326.
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    from shapely.geometry import shape

    with rasterio.open(raster_path) as src:
        if target_resolution is not None and abs(src.res[0] - target_resolution) > 1e-6:
            # Resample to coarser resolution
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height,
                *src.bounds,
                resolution=target_resolution,
            )
            data = np.empty((dst_height, dst_width), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest,
            )
            transform = dst_transform
        else:
            data = src.read(1)
            transform = src.transform

    # Mask out nodata
    mask = data != nodata_val

    # Polygonize
    records = []
    for geom_dict, value in shapes(data, mask=mask, transform=transform):
        value = int(value)
        geom = shape(geom_dict)
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
        if geom.is_empty or geom.area < min_area_deg2:
            continue
        name = class_names.get(value, f"class_{value}") if class_names else f"class_{value}"
        records.append({"zone_value": value, "zone_name": name, "geometry": geom})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    print(f"  Polygonized: {len(gdf)} polygons from {gdf['zone_value'].nunique()} classes")
    return gdf


def intersect_zone_layers(
    gdf_list: Sequence["gpd.GeoDataFrame"],
    zone_columns: Sequence[str],
    min_area_deg2: float = 0.01,
) -> "gpd.GeoDataFrame":
    """Intersect two or more zone GeoDataFrames and produce a combined zone layer.

    The resulting GeoDataFrame has a composite ``zone_id`` column formed by
    concatenating the zone labels from each input layer (e.g.
    ``"FRA__Cfa"`` for France × Koppen Cfa).

    Parameters
    ----------
    gdf_list : list of GeoDataFrames
        Each must have a geometry column and the column named in *zone_columns*.
    zone_columns : list of str
        Column name in each corresponding GeoDataFrame to use as the zone label.
    min_area_deg2 : float
        Minimum intersection area in square degrees. Smaller slivers are dropped.

    Returns
    -------
    gpd.GeoDataFrame with columns: zone_id (str), geometry, plus original zone columns.
    """
    import geopandas as gpd

    if len(gdf_list) != len(zone_columns):
        raise ValueError("gdf_list and zone_columns must have the same length.")
    if len(gdf_list) < 1:
        raise ValueError("At least one GeoDataFrame is required.")
    if len(gdf_list) == 1:
        result = gdf_list[0].copy()
        result["zone_id"] = result[zone_columns[0]].astype(str)
        return result

    # Start with the first layer
    combined = gdf_list[0][[zone_columns[0], "geometry"]].copy()
    combined = combined.rename(columns={zone_columns[0]: zone_columns[0]})

    for i in range(1, len(gdf_list)):
        right = gdf_list[i][[zone_columns[i], "geometry"]].copy()
        print(f"  Intersecting layer {i} ({zone_columns[i]}, {len(right)} polygons) "
              f"with accumulated {len(combined)} polygons...")
        combined = gpd.overlay(combined, right, how="intersection", keep_geom_type=True)
        # Drop slivers
        combined = combined[combined.geometry.area >= min_area_deg2].copy()
        print(f"    → {len(combined)} polygons after sliver removal")

    # Build composite zone_id
    combined["zone_id"] = combined[zone_columns[0]].astype(str)
    for col in zone_columns[1:]:
        combined["zone_id"] = combined["zone_id"] + "__" + combined[col].astype(str)

    combined = combined.reset_index(drop=True)
    print(f"  Final intersected zones: {len(combined)} polygons, "
          f"{combined['zone_id'].nunique()} unique zone IDs")
    return combined


def aggregate_by_zones(
    distinct: np.ndarray,
    transform: "rasterio.Affine",
    zone_gdf: "gpd.GeoDataFrame",
    zone_column: str = "zone_id",
    threshold: float = 0.75,
) -> tuple[np.ndarray, "gpd.GeoDataFrame"]:
    """Aggregate the pixel-level distinct_seasons array by spatial zones.

    For each zone polygon, compute the fraction of valid pixels that are
    NOT distinct (overlapping, value=0). If that fraction >= ``threshold``,
    the entire zone is set to overlapping (0). Otherwise the entire zone is
    set to distinct (1).

    Parameters
    ----------
    distinct : uint8 array (1=distinct, 0=overlapping, 255=nodata)
    transform : rasterio Affine transform matching *distinct*.
    zone_gdf : GeoDataFrame
        Zone polygons with a *zone_column* and geometry in EPSG:4326.
    zone_column : str
        Column in *zone_gdf* to use as the zone label.
    threshold : float
        Fraction of overlapping pixels (among valid) required to set the
        entire zone to overlapping. Default 0.75 means ≥75% overlapping
        → whole zone = overlapping.

    Returns
    -------
    distinct_zonal : uint8 array same shape as *distinct*.
        Zonally aggregated version: 0 or 1 per zone, 255=nodata.
    zone_stats : GeoDataFrame
        Per-zone statistics: zone_column, n_valid, n_overlapping,
        frac_overlapping, zonal_decision (0 or 1).
    """
    from rasterio.features import rasterize

    h, w = distinct.shape

    # Rasterize zone IDs to a label array matching distinct's grid
    # Assign a numeric ID to each zone polygon
    zone_ids = zone_gdf[zone_column].unique()
    id_map = {name: idx + 1 for idx, name in enumerate(zone_ids)}  # 1-based
    rev_map = {v: k for k, v in id_map.items()}

    shapes_iter = [
        (geom, id_map[zone_name])
        for geom, zone_name in zip(zone_gdf.geometry, zone_gdf[zone_column])
        if zone_name in id_map
    ]

    zone_raster = rasterize(
        shapes_iter,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )

    # Compute per-zone statistics
    valid_mask = (distinct == 0) | (distinct == 1)
    overlapping_mask = distinct == 0

    stats_records = []
    for numeric_id, zone_name in rev_map.items():
        zone_mask = zone_raster == numeric_id
        zone_valid = zone_mask & valid_mask
        n_valid = int(zone_valid.sum())
        if n_valid == 0:
            stats_records.append({
                zone_column: zone_name,
                "n_valid": 0, "n_overlapping": 0,
                "frac_overlapping": np.nan, "zonal_decision": 255,
            })
            continue
        n_overlapping = int((zone_valid & overlapping_mask).sum())
        frac_ov = n_overlapping / n_valid
        decision = 0 if frac_ov >= threshold else 1  # 0=overlapping, 1=distinct
        stats_records.append({
            zone_column: zone_name,
            "n_valid": n_valid,
            "n_overlapping": n_overlapping,
            "frac_overlapping": round(frac_ov, 4),
            "zonal_decision": decision,
        })

    import pandas as pd  # local import — avoid hard dependency for non-zone use
    zone_stats_df = pd.DataFrame(stats_records)
    # Merge back to get geometry
    zone_stats = zone_gdf[[zone_column, "geometry"]].drop_duplicates(subset=zone_column)
    zone_stats = zone_stats.merge(zone_stats_df, on=zone_column, how="left")

    # Build the zonally-aggregated distinct raster
    # Start from original, then overwrite valid zone pixels with the zonal decision
    distinct_zonal = distinct.copy()
    for _, row in zone_stats_df.iterrows():
        numeric_id = id_map.get(row[zone_column])
        if numeric_id is None or row["zonal_decision"] == 255:
            continue
        zone_mask = zone_raster == numeric_id
        # Only overwrite valid pixels within this zone
        update_mask = zone_mask & valid_mask
        distinct_zonal[update_mask] = row["zonal_decision"]

    n_changed = int((distinct_zonal != distinct)[valid_mask].sum())
    n_zones_ov = int((zone_stats_df["zonal_decision"] == 0).sum())
    n_zones_dist = int((zone_stats_df["zonal_decision"] == 1).sum())
    n_zones_nd = int((zone_stats_df["zonal_decision"] == 255).sum())
    print(f"  Zonal aggregation: {n_zones_ov} zones → overlapping, "
          f"{n_zones_dist} zones → distinct, {n_zones_nd} zones → nodata")
    print(f"  Pixels changed by zonal aggregation: {n_changed:,}")

    return distinct_zonal, zone_stats


# ---------------------------------------------------------------------------
# Season diagnostics — differences between S1/S2/Annual
# ---------------------------------------------------------------------------
# Cropland mask utilities
# ---------------------------------------------------------------------------

def build_cropland_mask(
    mask_file: str,
    mask_attribute: str,
    mask_threshold: float,
    reference_raster_path: str,
) -> np.ndarray:
    """Build a boolean raster mask from a vector grid file.

    Reads *mask_file* (a geoparquet / shapefile / GPKG whose geometry column
    may be stored as raw WKB bytes), filters features where
    ``feature[mask_attribute] > mask_threshold``, and rasterizes the result
    onto the same grid as *reference_raster_path*.

    Parameters
    ----------
    mask_file : str
        Path to a vector file (geoparquet, shapefile, GPKG, …) with a
        geometry column and at least one numeric attribute.
    mask_attribute : str
        Name of the numeric column used for filtering.
    mask_threshold : float
        Threshold value; features with ``attribute > threshold`` are kept.
    reference_raster_path : str
        Path to any of the input DOY rasters — used only for its shape,
        transform and CRS so the mask aligns pixel-for-pixel.

    Returns
    -------
    np.ndarray
        Boolean array (True = inside filtered grid cells = keep).
    """
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize as _rasterize_fn

    # --- Load the vector file ------------------------------------------------
    try:
        gdf = gpd.read_parquet(mask_file)
    except (ValueError, Exception):
        # Fallback: geometry stored as raw WKB bytes (non-geo parquet)
        df = pd.read_parquet(mask_file)
        if "geometry" not in df.columns:
            raise ValueError(f"No 'geometry' column found in {mask_file}")
        from shapely import wkb

        geoms = df["geometry"].apply(wkb.loads)
        gdf = gpd.GeoDataFrame(df.drop(columns=["geometry"]), geometry=geoms, crs="EPSG:4326")

    # --- Filter by attribute -------------------------------------------------
    if mask_attribute not in gdf.columns:
        raise ValueError(
            f"Column '{mask_attribute}' not found in {mask_file}. "
            f"Available columns: {list(gdf.columns)}"
        )
    gdf_filtered = gdf[gdf[mask_attribute] > mask_threshold].copy()
    print(
        f"  Cropland mask: {len(gdf_filtered):,} / {len(gdf):,} features "
        f"with {mask_attribute} > {mask_threshold}"
    )

    # --- Rasterize onto the reference grid -----------------------------------
    with rasterio.open(reference_raster_path) as ref:
        out_shape = (ref.height, ref.width)
        transform = ref.transform

    if len(gdf_filtered) == 0:
        print("  WARNING: no features passed the mask filter — all pixels will be masked out!")
        return np.zeros(out_shape, dtype=bool)

    # Ensure CRS matches (input rasters are EPSG:4326)
    if gdf_filtered.crs is not None and not gdf_filtered.crs.equals("EPSG:4326"):
        gdf_filtered = gdf_filtered.to_crs("EPSG:4326")

    shapes = [(geom, 1) for geom in gdf_filtered.geometry if geom is not None and geom.is_valid]
    mask_arr = _rasterize_fn(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    mask_bool = mask_arr.astype(bool)
    print(f"  Cropland mask: {mask_bool.sum():,} pixels inside, "
          f"{(~mask_bool).sum():,} pixels outside")
    return mask_bool


# ---------------------------------------------------------------------------

def _doy_difference(doy_a: np.ndarray, doy_b: np.ndarray) -> np.ndarray:
    """Signed circular difference between two DOY arrays: a − b.

    Result is in the range (−182, +182] days (shortest arc on the 365-day
    circle).  Positive means *a* is later in the year than *b*.

    Both inputs are int/float arrays with DOY in 1..365.  Returns float32.
    """
    a = doy_a.astype(np.float32)
    b = doy_b.astype(np.float32)
    diff = a - b
    # Wrap into (−182, +182]  (half-year)
    diff = np.where(diff > 182.0, diff - 365.0, diff)
    diff = np.where(diff <= -183.0, diff + 365.0, diff)
    return diff


def run_season_diagnostics(
    s1_sos_path: str,
    s1_eos_path: str,
    s2_sos_path: str,
    s2_eos_path: str,
    annual_sos_path: str,
    annual_eos_path: str,
    out_dir: str,
    nodata_val: int = 0,
    mask_file: Optional[str] = None,
    mask_attribute: Optional[str] = None,
    mask_threshold: float = 0.0,
) -> dict[str, str]:
    """Compare S1, S2, and Annual seasons pixel-by-pixel.

    Computes and writes:
      - ``S1_SOS_minus_ANN_SOS.tif`` — DOY difference: S1 SOS − Annual SOS
      - ``S1_EOS_minus_ANN_EOS.tif`` — DOY difference: S1 EOS − Annual EOS
      - ``S2_SOS_minus_ANN_SOS.tif`` — DOY difference: S2 SOS − Annual SOS
      - ``S2_EOS_minus_ANN_EOS.tif`` — DOY difference: S2 EOS − Annual EOS
      - ``S1_LOS.tif``              — S1 length-of-season (days)
      - ``S2_LOS.tif``              — S2 length-of-season (days)
      - ``ANN_LOS.tif``             — Annual length-of-season (days)
      - ``S1_LOS_minus_ANN_LOS.tif``— LOS difference: S1 LOS − Annual LOS
      - ``S2_LOS_minus_ANN_LOS.tif``— LOS difference: S2 LOS − Annual LOS
      - ``longer_season.tif``       — which season is longer: 1=S1, 2=S2,
                                       0=equal, 255=nodata

    All DOY differences use circular (wrap-aware) arithmetic on a 365-day
    cycle, returning values in (−182, +182] days.  LOS differences are
    ordinary signed integers (no wrapping needed since LOS is 0–365).

    Parameters
    ----------
    s1_sos_path, s1_eos_path : str
        S1 start/end-of-season DOY rasters (nodata = *nodata_val*).
    s2_sos_path, s2_eos_path : str
        S2 start/end-of-season DOY rasters.
    annual_sos_path, annual_eos_path : str
        Annual start/end-of-season DOY rasters.
    out_dir : str
        Directory where diagnostic rasters are written.
    nodata_val : int
        Value treated as nodata in all input rasters (default 0).
    mask_file : str, optional
        Path to a vector file (geoparquet, shapefile, …) with grid
        polygons and a numeric attribute.  When provided together with
        *mask_attribute*, pixels falling outside the filtered grid cells
        are set to nodata before processing.
    mask_attribute : str, optional
        Numeric column in *mask_file* used for filtering.
    mask_threshold : float
        Features with ``mask_attribute > mask_threshold`` are kept.
        Default 0.0.

    Returns
    -------
    dict mapping output name → file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading input rasters for diagnostics...")
    s1_sos, profile = _read_raster(s1_sos_path)
    s1_eos, _ = _read_raster(s1_eos_path)
    s2_sos, _ = _read_raster(s2_sos_path)
    s2_eos, _ = _read_raster(s2_eos_path)
    ann_sos, _ = _read_raster(annual_sos_path)
    ann_eos, _ = _read_raster(annual_eos_path)

    print(f"  Raster shape: {s1_sos.shape}")

    # Optional: apply cropland mask — set outside pixels to nodata
    if mask_file is not None and mask_attribute is not None:
        print("Building cropland mask for diagnostics...")
        crop_mask = build_cropland_mask(
            mask_file, mask_attribute, mask_threshold, s1_sos_path
        )
        outside = ~crop_mask
        for arr in (s1_sos, s1_eos, s2_sos, s2_eos, ann_sos, ann_eos):
            arr[outside] = nodata_val

    # Nodata masks — one per pair of seasons being compared
    nd_s1  = (s1_sos == nodata_val) | (s1_eos == nodata_val)
    nd_s2  = (s2_sos == nodata_val) | (s2_eos == nodata_val)
    nd_ann = (ann_sos == nodata_val) | (ann_eos == nodata_val)
    nd_s1_ann = nd_s1 | nd_ann
    nd_s2_ann = nd_s2 | nd_ann
    nd_all = nd_s1 | nd_s2

    out_paths: dict[str, str] = {}

    # ── SOS / EOS differences (circular, int16, nodata = 0) ──────────────
    diff_profile = profile.copy()
    diff_profile.update(dtype="int16", nodata=0)

    diffs = [
        ("S1_SOS_minus_ANN_SOS", s1_sos, ann_sos, nd_s1_ann),
        ("S1_EOS_minus_ANN_EOS", s1_eos, ann_eos, nd_s1_ann),
        ("S2_SOS_minus_ANN_SOS", s2_sos, ann_sos, nd_s2_ann),
        ("S2_EOS_minus_ANN_EOS", s2_eos, ann_eos, nd_s2_ann),
    ]
    for name, a, b, nd_mask in diffs:
        diff = _doy_difference(a, b)
        arr = np.where(nd_mask, 0, np.round(diff)).astype(np.int16)
        p = out_dir / f"{name}.tif"
        _write_raster(p, arr, diff_profile)
        out_paths[name] = str(p)

        # Summary stats (valid pixels only)
        valid = ~nd_mask
        vals = diff[valid]
        print(f"  {name}: mean={vals.mean():.1f}d  median={np.median(vals):.0f}d  "
              f"std={vals.std():.1f}d  |abs|_mean={np.abs(vals).mean():.1f}d  "
              f"range=[{vals.min():.0f}, {vals.max():.0f}]  valid={valid.sum():,}")

    # ── LOS per season (int16, nodata = 0) ───────────────────────────────
    los_profile = profile.copy()
    los_profile.update(dtype="int16", nodata=0)

    s1_los = _compute_los(s1_sos, s1_eos, nodata_val=nodata_val)
    s2_los = _compute_los(s2_sos, s2_eos, nodata_val=nodata_val)
    ann_los = _compute_los(ann_sos, ann_eos, nodata_val=nodata_val)

    for name, arr in [("S1_LOS", s1_los), ("S2_LOS", s2_los), ("ANN_LOS", ann_los)]:
        p = out_dir / f"{name}.tif"
        _write_raster(p, arr, los_profile)
        out_paths[name] = str(p)

    # ── LOS differences (ordinary signed, int16, nodata = 0) ─────────────
    los_diffs = [
        ("S1_LOS_minus_ANN_LOS", s1_los, ann_los, nd_s1_ann),
        ("S2_LOS_minus_ANN_LOS", s2_los, ann_los, nd_s2_ann),
    ]
    for name, a, b, nd_mask in los_diffs:
        diff = (a.astype(np.int16) - b.astype(np.int16))
        arr = np.where(nd_mask, 0, diff).astype(np.int16)
        p = out_dir / f"{name}.tif"
        _write_raster(p, arr, los_profile)
        out_paths[name] = str(p)

        valid = ~nd_mask
        vals = diff[valid].astype(np.float32)
        print(f"  {name}: mean={vals.mean():.1f}d  median={np.median(vals):.0f}d  "
              f"std={vals.std():.1f}d  range=[{vals.min():.0f}, {vals.max():.0f}]")

    # ── Which season is longer? (uint8: 1=S1, 2=S2, 0=equal, 255=nodata)
    longer_profile = profile.copy()
    longer_profile.update(dtype="uint8", nodata=255)

    s1_los_f = s1_los.astype(np.float32)
    s2_los_f = s2_los.astype(np.float32)
    longer = np.where(
        nd_all, 255,
        np.where(s1_los_f > s2_los_f, 1,
                 np.where(s2_los_f > s1_los_f, 2, 0))
    ).astype(np.uint8)

    p = out_dir / "longer_season.tif"
    _write_raster(p, longer, longer_profile)
    out_paths["longer_season"] = str(p)

    # Summary
    valid_longer = ~nd_all
    n_s1 = int((longer[valid_longer] == 1).sum())
    n_s2 = int((longer[valid_longer] == 2).sum())
    n_eq = int((longer[valid_longer] == 0).sum())
    n_valid = int(valid_longer.sum())
    print(f"\n  Longer season: S1={n_s1:,} ({100*n_s1/n_valid:.1f}%)  "
          f"S2={n_s2:,} ({100*n_s2/n_valid:.1f}%)  "
          f"equal={n_eq:,} ({100*n_eq/n_valid:.1f}%)  "
          f"nodata={int(nd_all.sum()):,}")

    # LOS summary table
    for lbl, los, nd in [("S1", s1_los, nd_s1), ("S2", s2_los, nd_s2), ("ANN", ann_los, nd_ann)]:
        v = los[~nd].astype(np.float32)
        print(f"  {lbl} LOS: mean={v.mean():.0f}d  median={np.median(v):.0f}d  "
              f"std={v.std():.0f}d  range=[{v.min():.0f}, {v.max():.0f}]")

    print(f"\nDone. {len(out_paths)} files written to {out_dir}")
    return out_paths


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
    smooth_kernel_size: int = 3,
    zone_gdf: Optional["gpd.GeoDataFrame"] = None,
    zone_column: str = "zone_id",
    zone_threshold: float = 0.75,
    mask_file: Optional[str] = None,
    mask_attribute: Optional[str] = None,
    mask_threshold: float = 0.0,
) -> dict[str, str]:
    """
    Full pipeline: read rasters → compute overlap → (smooth) → (aggregate zones) → write outputs.

    Parameters
    ----------
    s1_sos_path, s1_eos_path, s2_sos_path, s2_eos_path : str
        Paths to the 4 input DOY rasters.
    out_dir : str
        Output directory for all generated rasters.
    threshold_days, threshold_frac, threshold_mode : overlap thresholds
    nodata_val : int
        Nodata value in input rasters (default 0).
    smooth_kernel_size : int
        Median-filter kernel size for spatial smoothing of the distinct_seasons
        raster before zone aggregation. Default 3 (light smoothing). Set to 0
        or 1 to disable smoothing.
    zone_gdf : GeoDataFrame, optional
        Zone polygons for spatial aggregation. Must have a *zone_column* and
        geometry in EPSG:4326.  When ``None`` (default), no zonal aggregation
        is applied and the result is purely pixel-based (+ optional smoothing).
    zone_column : str
        Column in *zone_gdf* to use as the zone label.  Default ``"zone_id"``.
    zone_threshold : float
        Fraction of overlapping pixels per zone above which the entire zone
        is set to overlapping.  Default 0.75.
    mask_file : str, optional
        Path to a vector file (geoparquet, shapefile, …) with grid
        polygons and a numeric attribute.  When provided together with
        *mask_attribute*, pixels falling outside the filtered grid cells
        are set to nodata before processing.
    mask_attribute : str, optional
        Numeric column in *mask_file* used for filtering.
    mask_threshold : float
        Features with ``mask_attribute > mask_threshold`` are kept.
        Default 0.0.

    Returns
    -------
    dict mapping output name → file path.
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

    # Optional: apply cropland mask — set outside pixels to nodata
    if mask_file is not None and mask_attribute is not None:
        print("Building cropland mask...")
        crop_mask = build_cropland_mask(
            mask_file, mask_attribute, mask_threshold, s1_sos_path
        )
        outside = ~crop_mask
        for arr in (s1_sos, s1_eos, s2_sos, s2_eos):
            arr[outside] = nodata_val

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

    # --- Step 1b: write raw pixel-level distinct_seasons before any smoothing/aggregation ---
    dist_profile = profile.copy()
    dist_profile.update(dtype="uint8", nodata=255)
    p = out_dir / "distinct_seasons_raw.tif"
    _write_raster(p, distinct, dist_profile)
    out_paths: dict[str, str] = {}
    out_paths["distinct_seasons_raw"] = str(p)

    # --- Step 1c: optional spatial smoothing ---
    if smooth_kernel_size and smooth_kernel_size > 1:
        print(f"Applying spatial smoothing (median filter, kernel={smooth_kernel_size})...")
        distinct = smooth_distinct_seasons(distinct, kernel_size=smooth_kernel_size)
        n_distinct_s = int((distinct == 1).sum())
        n_overlap_s = int((distinct == 0).sum())
        print(f"  After smoothing — distinct: {n_distinct_s:,}, overlapping: {n_overlap_s:,}")

    # --- Step 1d: optional zonal aggregation ---
    zone_stats_gdf = None
    if zone_gdf is not None:
        print(f"Aggregating by zones (column={zone_column!r}, threshold={zone_threshold})...")
        raster_transform = profile["transform"]
        distinct, zone_stats_gdf = aggregate_by_zones(
            distinct, raster_transform, zone_gdf,
            zone_column=zone_column, threshold=zone_threshold,
        )
        n_distinct_z = int((distinct == 1).sum())
        n_overlap_z = int((distinct == 0).sum())
        print(f"  After zonal aggregation — distinct: {n_distinct_z:,}, overlapping: {n_overlap_z:,}")

        # Write zone statistics
        p_stats = out_dir / "zone_stats.parquet"
        zone_stats_gdf.to_parquet(str(p_stats), index=False)
        out_paths["zone_stats"] = str(p_stats)
        print(f"  Written: {p_stats}")

        # Write zone_id raster for traceability
        from rasterio.features import rasterize as _rasterize_fn
        zone_ids_unique = zone_gdf[zone_column].unique()
        _zid_map = {name: idx + 1 for idx, name in enumerate(zone_ids_unique)}
        _shapes_for_zones = [
            (geom, _zid_map[zn])
            for geom, zn in zip(zone_gdf.geometry, zone_gdf[zone_column])
            if zn in _zid_map
        ]
        zone_id_raster = _rasterize_fn(
            _shapes_for_zones,
            out_shape=distinct.shape,
            transform=profile["transform"],
            fill=0, dtype=np.int32,
        )
        zid_profile = profile.copy()
        zid_profile.update(dtype="int32", nodata=0)
        p_zid = out_dir / "zone_id_raster.tif"
        _write_raster(p_zid, zone_id_raster, zid_profile)
        out_paths["zone_id_raster"] = str(p_zid)

        # Write zone_id legend (JSON mapping numeric_id → zone_name)
        p_legend = out_dir / "zone_id_legend.json"
        with open(p_legend, "w") as f:
            json.dump({str(v): k for k, v in _zid_map.items()}, f, indent=2)
        out_paths["zone_id_legend"] = str(p_legend)
        print(f"  Written: {p_legend}")

    # --- Step 2: produce merged/masked seasons ---
    print("Merging/masking seasons for non-distinct pixels...")
    s1_sos_m, s1_eos_m, s2_sos_m, s2_eos_m = merge_seasons(
        s1_sos, s1_eos, s2_sos, s2_eos, distinct, nodata_val=nodata_val
    )

    # --- Step 3: write outputs ---
    print("Writing outputs...")

    # distinct_seasons.tif (final, after smoothing + zonal aggregation)
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

    _CAL_DIR = Path(__file__).resolve().parent / "data" / "cropcalendars"

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
    cal_dir = Path(__file__).resolve().parent / "data" / "cropcalendars"
    p.add_argument("--s1-sos",  default=str(cal_dir / "S1_SOS_WGS84.tif"),  help="S1 start-of-season raster (DOY)")
    p.add_argument("--s1-eos",  default=str(cal_dir / "S1_EOS_WGS84.tif"),  help="S1 end-of-season raster (DOY)")
    p.add_argument("--s2-sos",  default=str(cal_dir / "S2_SOS_WGS84.tif"),  help="S2 start-of-season raster (DOY)")
    p.add_argument("--s2-eos",  default=str(cal_dir / "S2_EOS_WGS84.tif"),  help="S2 end-of-season raster (DOY)")
    p.add_argument("--out-dir", default=str(cal_dir / "season_overlap_outputs"),
                   help="Output directory")
    p.add_argument("--overlap-threshold-days",  type=float, default=100.0,
                   help="Overlap in days below which seasons are considered distinct")
    p.add_argument("--overlap-threshold-frac",  type=float, default=0.35,
                   help="Overlap as fraction of longer season below which distinct (0–1)")
    p.add_argument("--threshold-mode", choices=["days", "fraction", "either", "both"],
                   default="both",
                   help="How to combine the two thresholds: "
                        "'days'=only day threshold, 'fraction'=only fraction threshold, "
                        "'either'=distinct if either holds, 'both'=distinct only if both hold")
    p.add_argument("--smooth-kernel", type=int, default=3,
                   help="Median-filter kernel size for spatial smoothing (0 or 1 to disable)")
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
        smooth_kernel_size=ns.smooth_kernel,
    )


if __name__ == "__main__":
    main()
