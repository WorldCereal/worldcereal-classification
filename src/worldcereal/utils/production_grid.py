"""Utilities for production grid creation and UTM enrichment.

This module supports two complementary steps:
1) Grid creation (tiling): build a production grid from a WGS84 bounding box
   and temporal extent. The resulting GeoDataFrame includes:
   `tile_name`, `epsg`, `bounds_epsg`, `start_date`, and `end_date`.
2) UTM enrichment: attach per-feature UTM geometry and EPSG information using
   the Sentinel-2 MGRS grid:
   `geometry_utm_wkt` and `epsg_utm`.

Notes
-----
* Efficiency: Reprojection is done per unique EPSG to avoid many single-row
  reprojections.
* Storage: Because each feature may have a different CRS, we store the UTM
  geometry as WKT text in a separate column (`geometry_utm_wkt`) instead of as
  a second active GeoSeries (GeoParquet only supports one geometry with a
  single CRS). Downstream code can reconstruct geometry per row using
  shapely.from_wkt and the EPSG stored in `epsg_utm`.

Examples
--------
Create a production grid and persist it:
    grid = create_production_grid(spatial_extent, temporal_extent, resolution=20)
    grid.to_parquet("production_grid.parquet")

Enrich an existing grid with UTM columns:
    convert_gdf_to_utm_grid(
        in_path="production_grid.parquet",
        out_path="production_grid_utm.parquet",
        id_col="tile_name",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.manager.job_splitters import load_s2_grid
from shapely.geometry import box


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
    """Load a grid file, enrich it with UTM columns, and write a GeoParquet.

    The input file must contain a geometry column and the provided `id_col`,
    which is copied into the output `tile_name` column. The output contains
    `geometry_utm_wkt` and `epsg_utm` columns derived from the Sentinel-2 grid.
    """
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


def lon_to_utm_zone(lon: float) -> int:
    return int((lon + 180) / 6) + 1


def utm_zone_bounds(zone: int) -> tuple[float, float]:
    min_lon = (zone - 1) * 6 - 180
    max_lon = min_lon + 6
    return min_lon, max_lon


def split_bbox_by_utm_and_hemisphere(
    west: float, south: float, east: float, north: float
) -> list[dict]:
    """
    Split a bounding box into UTM zones and hemispheres.

    Coordinates must be in WGS84 (EPSG:4326) format.
    """
    zone_start = lon_to_utm_zone(west)
    zone_end = lon_to_utm_zone(east)
    hemi_splits = []

    if south < 0 and north > 0:
        lat_splits = [("S", south, 0), ("N", 0, north)]
    else:
        hemisphere = "N" if south >= 0 else "S"
        lat_splits = [(hemisphere, south, north)]

    for zone in range(zone_start, zone_end + 1):
        zmin_lon, zmax_lon = utm_zone_bounds(zone)
        lon_min_clipped = max(west, zmin_lon)
        lon_max_clipped = min(east, zmax_lon)

        if lon_min_clipped >= lon_max_clipped:
            continue

        for hemisphere, hemi_min_lat, hemi_max_lat in lat_splits:
            hemi_min_clipped = max(south, hemi_min_lat)
            hemi_max_clipped = min(north, hemi_max_lat)

            if hemi_min_clipped < hemi_max_clipped:
                hemi_splits.append(
                    {
                        "west": lon_min_clipped,
                        "south": hemi_min_clipped,
                        "east": lon_max_clipped,
                        "north": hemi_max_clipped,
                        "crs": "EPSG:4326",
                        "zone": zone,
                        "hemisphere": hemisphere,
                    }
                )

    return hemi_splits


def create_tiling_grid(
    bbox: dict,
    basename: str = "tile",
    output_crs: str = "EPSG:4326",
    grid_size_m: float = 20000,
    tiling_crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Create a grid of square tiles over a bounding box (with CRS).

    Tiles in `tiling_crs`, output in `output_crs`.
    """
    if not {"west", "south", "east", "north", "crs"}.issubset(bbox):
        raise ValueError("bbox must include 'west', 'south', 'east', 'north', 'crs'.")

    bbox_geom = box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=bbox["crs"])
    if tiling_crs is None:
        crs = bbox_gdf.estimate_utm_crs()
        epsg = int(crs.to_epsg())
        tiling_crs = f"EPSG:{epsg}"
    bbox_gdf = bbox_gdf.to_crs(tiling_crs)

    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    x_coords = np.arange(minx, maxx, grid_size_m)
    y_coords = np.arange(miny, maxy, grid_size_m)
    coordinates = [
        (x, y, min(x + grid_size_m, maxx), min(y + grid_size_m, maxy))
        for x in x_coords
        for y in y_coords
    ]
    bounds_tiling = [repr(coords) for coords in coordinates]

    geometries_tiling = [box(*coords) for coords in coordinates]

    grid = gpd.GeoDataFrame(geometry=geometries_tiling, crs=tiling_crs).to_crs(
        output_crs
    )
    grid["tile_name"] = [f"{basename}_{i}" for i in range(len(grid))]
    grid["epsg"] = (
        int(tiling_crs.split(":")[1]) if tiling_crs.startswith("EPSG:") else None
    )
    grid["bounds_epsg"] = bounds_tiling

    return grid


def create_production_grid(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    resolution: int = 20,
    tiling_crs: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Create a production grid GeoDataFrame for the given extent."""
    if not spatial_extent.epsg == 4326:
        logger.info(
            '"Spatial extent is not in WGS84 (EPSG:4326). Reprojecting to WGS84.")'
        )
        bbox_geom = box(
            spatial_extent.west,
            spatial_extent.south,
            spatial_extent.east,
            spatial_extent.north,
        )
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=spatial_extent.epsg)
        bbox_gdf = bbox_gdf.to_crs("EPSG:4326")
        spatial_extent = BoundingBoxExtent(
            west=bbox_gdf.total_bounds[0],
            south=bbox_gdf.total_bounds[1],
            east=bbox_gdf.total_bounds[2],
            north=bbox_gdf.total_bounds[3],
            epsg=4326,
        )

    bbox_splits = split_bbox_by_utm_and_hemisphere(
        spatial_extent.west,
        spatial_extent.south,
        spatial_extent.east,
        spatial_extent.north,
    )

    logger.info(f"Splitted bounding box into {len(bbox_splits)} UTM zone splits.")
    grid_dfs = []
    for split in bbox_splits:
        tile_name = f"tile_{split['zone']}{split['hemisphere']}"
        grid_dfs.append(
            create_tiling_grid(
                split,
                basename=tile_name,
                grid_size_m=resolution * 1000,
                tiling_crs=tiling_crs,
            )
        )

    grid = gpd.GeoDataFrame(pd.concat(grid_dfs, ignore_index=True))
    grid["start_date"] = temporal_extent.start_date
    grid["end_date"] = temporal_extent.end_date

    logger.info(f"Created production grid with {len(grid)} tiles.")

    return grid
