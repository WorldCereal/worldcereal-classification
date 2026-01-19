"""Utility to enrich a GeoParquet with per-feature UTM CRS and geometry (S2-based).

For every feature in the input GeoDataFrame:
1. Determine the Sentinel-2 MGRS tile (by centroid) using the same S2 grid
    parquet as used in `openeo-gfmap` (see job_splitters.split_job_s2grid).
2. Derive the appropriate UTM EPSG code from the MGRS tile (zone + hemisphere).
3. Append new attributes:
    - `tile_name` (string) Identifier copied from a user-provided column (--id-source)
    - `epsg_utm`  (int) EPSG code of feature-specific UTM zone (always WGS84 UTM)
    - `geometry_utm_wkt` (str) WKT of reprojected geometry in its per-row UTM CRS

The result is written to a new GeoParquet.

Notes
-----
* Efficiency: Reprojection is done per unique EPSG to avoid many single-row
    reprojections.
* Storage: Because each feature may have a different CRS, we store the UTM
    geometry as WKT text in a separate column (`geometry_utm_wkt`) instead
    of as a second active GeoSeries (GeoParquet only supports one geometry with a
    single CRS). Downstream code can reconstruct a geometry per row using
    shapely.from_wkt and the EPSG stored in `epsg_utm`.

The script DOES NOT derive `tile_name` from Sentinel-2; it is an external
identifier you supply via --id-source. The S2 grid is only used to determine
the UTM EPSG per feature.

Example
-------
python scripts/convert_gdf_to_production_grid.py \
    --input path/to/input.geoparquet \
    --id-source production_unit_id \
    --output path/to/output_with_utm.geoparquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from loguru import logger
from openeo_gfmap.manager.job_splitters import load_s2_grid


def _load_gdf(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Determine file format based on extension
    ext = path.suffix.lower()
    if ext == ".parquet":
        gdf = gpd.read_parquet(path)
    elif ext in [".gpkg", ".geojson", ".shp"]:
        gdf = gpd.read_file(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS.")
    if "geometry" not in gdf:
        raise ValueError("Input file must contain a geometry column.")

    return gdf


def _centroids_in_crs(gdf: gpd.GeoDataFrame, epsg: int = 4326) -> gpd.GeoSeries:
    """Compute centroids robustly by projecting to Web Mercator first.

    Always project to 3857 for centroid calculation to avoid planar assumptions
    on geographic CRS, then transform the resulting point centroids back to the
    requested EPSG (default 4326).
    """
    tmp = gdf.to_crs(epsg=3857)
    cent_3857 = tmp.geometry.centroid
    return gpd.GeoSeries(cent_3857, crs="EPSG:3857").to_crs(epsg=epsg)


def _mgrs_tile_to_utm_epsg(tile: str) -> Optional[int]:
    """Derive a UTM EPSG from an S2 MGRS tile string.

    MGRS tile (e.g. '31UFS') starts with zone number (2 digits) + latitude band + two grid letters.
    UTM EPSG (northern hemisphere): 326 + zone (2-digit) -> e.g. 32631.
    Southern hemisphere: 327 + zone.

    Latitude band letters (C–M southern, N–X northern; I and O excluded).
    We'll classify by the band letter at position 3 (tile[2]).
    """
    if not tile or len(tile) < 3:
        return None
    # Extract zone number (first 2 chars may be 1 or 2 digits; handle both)
    # Sentinel-2 tiles usually: two digits + letter + two letters: 31UFS, 05QLF etc.
    # We'll parse initial numeric part.
    i = 0
    while i < len(tile) and tile[i].isdigit():
        i += 1
    if i == 0:
        return None
    zone_str = tile[:i]
    band_letter = tile[i : i + 1]
    try:
        zone = int(zone_str)
    except ValueError:
        return None
    if zone < 1 or zone > 60:
        return None
    if not band_letter:
        return None
    band = band_letter.upper()
    # Latitude bands: C..M south, N..X north.
    north = band >= "N"
    base = 326 if north else 327
    return base * 100 + zone  # e.g. 32600 + zone -> 32631


def _assign_utm_epsg_by_s2(
    gdf: gpd.GeoDataFrame, web_mercator: bool = False
) -> gpd.Series:
    # Ensure CRS matches grid method expectation (split_job_s2grid uses 4326 or 3857 depending on flag)
    epsg_target = 3857 if web_mercator else 4326
    centroids = _centroids_in_crs(gdf, epsg=epsg_target)
    cent_gdf = gpd.GeoDataFrame(
        gdf[[gdf.geometry.name]].copy(), geometry=centroids, crs=f"EPSG:{epsg_target}"
    )
    s2_grid = load_s2_grid(web_mercator=web_mercator)[["tile", "geometry"]]
    joined = gpd.sjoin(cent_gdf, s2_grid, predicate="intersects", how="left")
    tiles = joined["tile"]
    return tiles.apply(_mgrs_tile_to_utm_epsg)


def _batch_reproject(
    original: gpd.GeoDataFrame, epsg_series: pd.Series
) -> gpd.GeoDataFrame:
    """Given a GeoDataFrame and a parallel Series of EPSG codes, build WKT UTM geometry column.

    We cannot store multiple CRSes in a single GeoSeries; instead we serialize
    each per-feature UTM geometry as WKT text in `geometry_utm_wkt` while
    keeping the original geometry as the active geometry column.
    """
    from shapely import to_wkt

    wkt_series = pd.Series(index=original.index, dtype=object)
    for epsg, idxs in epsg_series.dropna().groupby(epsg_series).groups.items():
        subset = original.loc[idxs]
        try:
            reproj = subset.to_crs(epsg=int(epsg))
        except Exception as e:  # pragma: no cover - log & skip
            logger.warning(f"Warning: failed to reproject subset to EPSG:{epsg} -> {e}")
            continue
        wkt_series.loc[idxs] = reproj.geometry.apply(
            lambda g: to_wkt(g, rounding_precision=6)
        )

    result = original.copy()
    result["geometry_utm_wkt"] = wkt_series
    result["epsg_utm"] = epsg_series.astype("Int64")  # preserve NA
    return result


def enrich_with_utm(
    gdf: gpd.GeoDataFrame,
    web_mercator_grid: bool = False,
) -> gpd.GeoDataFrame:
    """Add per-feature UTM EPSG and geometry_utm columns using the Sentinel-2 grid."""
    epsg_series = _assign_utm_epsg_by_s2(gdf, web_mercator=web_mercator_grid)
    return _batch_reproject(gdf, epsg_series)


def convert_gdf_to_utm_grid(
    in_path: Path | str,
    out_path: Path | str,
    id_col: str,
    web_mercator_grid: bool = False,
) -> None:
    in_path = Path(in_path)
    out_path = Path(out_path)

    logger.info(f"Loading GeoDataFrame: {in_path}")
    gdf = _load_gdf(in_path)
    logger.info("Converting to WGS84 ...")
    gdf = gdf.to_crs(epsg=4326)

    # Validate id-source column
    if id_col not in gdf.columns:
        raise ValueError(f"--id-source column '{id_col}' not found in input data")
    duplicates = gdf[id_col][gdf[id_col].duplicated()]
    if not duplicates.empty:
        raise ValueError(
            f"--id-source column '{id_col}' contains duplicates (sample: {duplicates.head().tolist()})."
        )
    if gdf[id_col].isna().any():
        raise ValueError(f"--id-source column '{id_col}' contains null values")

    enriched = enrich_with_utm(
        gdf,
        web_mercator_grid=web_mercator_grid,
    )
    # Attach tile_name from id-source
    enriched["tile_name"] = gdf[id_col].astype(str).values

    missing = enriched["epsg_utm"].isna().sum()
    if missing:
        logger.warning(
            f"{missing} features have no UTM EPSG (outside bounds or join failure)."
        )

    logger.info(f"Writing output to {out_path}")
    # Note: Only one active geometry w/ single CRS allowed. UTM variants stored as WKT per row.
    enriched.to_parquet(out_path)
    logger.info("Done.")
