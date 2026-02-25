"""Utilities for production grid creation and UTM enrichment.

This module supports three complementary steps:
1) Grid creation (create_production_grid):
    Splits AOI geometries by Sentinel-2 MGRS tiles and creates sub-tiles
    within each S2 tile based on a specified tiling size.
    The resulting GeoDataFrame includes `tile_name`, a unique identifier for each tile,
    and geometries in EPSG:4326.
2) UTM enrichment (enrich_with_utm):
    For each tile, determine the appropriate UTM zone based on the S2 MGRS tile it intersects.
    Reproject the tile geometry to that UTM CRS and store it as WKT in a new column `geometry_utm_wkt`.
    The corresponding EPSG code is stored in `epsg_utm`.
3) Small-tile merging (_merge_small_tiles):
    Merge tiles with width or height below a specified threshold.
    Merging is done within the same UTM EPSG to ensure spatial consistency, based on shared edges or proximity.

Notes
-----
* Storage: Because each feature may have a different CRS, we store the UTM
  geometry as WKT text in a separate column (`geometry_utm_wkt`) instead of as
  a second active GeoSeries (GeoParquet only supports one geometry with a
  single CRS). Downstream code can reconstruct geometry per row using
  shapely.from_wkt and the EPSG stored in `epsg_utm`.

Example
-------
Create a production grid from AOIs:
    grid = create_production_grid(aoi_gdf, tiling_size_km=20)
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from openeo_gfmap.manager.job_splitters import load_s2_grid
from shapely import wkt as shapely_wkt
from shapely.geometry import box


def _centroids_in_crs(gdf: gpd.GeoDataFrame, epsg: int = 4326) -> gpd.GeoSeries:
    """Compute centroids robustly in a target CRS.

    Always project to 3857 for centroid calculation to avoid planar assumptions
    on geographic CRS, then transform the resulting point centroids back to the
    requested EPSG (default 4326).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with a geometry column.
    epsg : int, optional
        EPSG code for the output centroids, by default 4326.

    Returns
    -------
    gpd.GeoSeries
        Centroids in the requested CRS.
    """
    tmp = gdf.to_crs(epsg=3857)
    cent_3857 = tmp.geometry.centroid
    return gpd.GeoSeries(cent_3857, crs="EPSG:3857").to_crs(epsg=epsg)


def _mgrs_tile_to_utm_epsg(tile: str) -> Optional[int]:
    """Derive a UTM EPSG from an S2 MGRS tile string.

    MGRS tile (e.g. "31UFS") starts with zone number (2 digits) + latitude band
    + two grid letters. UTM EPSG (northern hemisphere): 326 + zone (2-digit)
    -> e.g. 32631. Southern hemisphere: 327 + zone.

    Latitude band letters (C-M southern, N-X northern; I and O excluded).
    We classify by the band letter at position 3 (tile[2]).

    Parameters
    ----------
    tile : str
        Sentinel-2 MGRS tile identifier (e.g. "31UFS").

    Returns
    -------
    Optional[int]
        UTM EPSG code, or None if the tile cannot be parsed.
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
    """Assign UTM EPSG codes by intersecting centroids with the S2 grid.

    Centroids are computed in 4326 or 3857 depending on the grid mode, then
    spatially joined to the Sentinel-2 MGRS grid to pick the tile-based UTM
    EPSG code.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with a geometry column.
    web_mercator : bool, optional
        Whether to use the Web Mercator S2 grid, by default False.

    Returns
    -------
    pd.Series
        Series of EPSG codes aligned to the input rows (nullable Int64).
    """
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
    """Reproject features by EPSG and store UTM geometries as WKT.

    We cannot store multiple CRSes in a single GeoSeries; instead we serialize
    each per-feature UTM geometry as WKT text in `geometry_utm_wkt` while
    keeping the original geometry as the active geometry column.

    Parameters
    ----------
    original : gpd.GeoDataFrame
        Input GeoDataFrame with a geometry column.
    epsg_series : pd.Series
        Per-row EPSG codes (nullable).

    Returns
    -------
    gpd.GeoDataFrame
        Copy of the input with `geometry_utm_wkt` and `epsg_utm` columns added.
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
    """Add per-feature UTM EPSG and geometry_utm columns using the S2 grid.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with a geometry column.
    web_mercator_grid : bool, optional
        Whether to use the Web Mercator S2 grid, by default False.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with `geometry_utm_wkt` and `epsg_utm` columns.
    """
    epsg_series = _assign_utm_epsg_by_s2(gdf, web_mercator=web_mercator_grid)
    return _batch_reproject(gdf, epsg_series)


def ensure_utm_grid(
    gdf: gpd.GeoDataFrame,
    web_mercator_grid: bool = False,
) -> gpd.GeoDataFrame:
    """Ensure a grid has UTM geometry/EPSG columns.

    The input GeoDataFrame is reprojected to WGS84 and enriched with per-feature
    UTM geometry (WKT) and EPSG codes. A unique `tile_name` column must already
    be present and is preserved as-is.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with a geometry column and `tile_name`.
    web_mercator_grid : bool, optional
        Whether to use the Web Mercator S2 grid, by default False.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with `tile_name`, `geometry_utm_wkt`, and `epsg_utm`.
    """
    # Some basic checks never hurt...
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS.")
    if "tile_name" not in gdf.columns:
        raise ValueError("Input GeoDataFrame must contain a 'tile_name' column.")
    duplicates = gdf["tile_name"][gdf["tile_name"].duplicated()]
    if not duplicates.empty:
        raise ValueError(
            "'tile_name' contains duplicates (sample: "
            + ", ".join(duplicates.head().astype(str).tolist())
            + ")."
        )
    if gdf["tile_name"].isna().any():
        raise ValueError("'tile_name' contains null values")

    # Reproject to WGS84
    gdf = gdf.to_crs(epsg=4326)
    # Actual enrichment with UTM geometry and EPSG based on S2 grid intersection
    enriched = enrich_with_utm(
        gdf,
        web_mercator_grid=web_mercator_grid,
    )
    # Preserve tile_name as string type in the output
    enriched["tile_name"] = gdf["tile_name"].astype(str).values
    return enriched


def _create_tiling_grid_for_polygon(
    polygon_wgs84,
    basename: str,
    grid_size_m: float,
    utm_epsg: int,
    attributes: Optional[dict[str, object]] = None,
) -> gpd.GeoDataFrame:
    """Tile a WGS84 polygon into square grids in a projected CRS.

    Tiles are generated in the tiling CRS and clipped to the polygon before
    being reprojected to EPSG:4326.

    Parameters
    ----------
    polygon_wgs84 : shapely.geometry.base.BaseGeometry
        Polygon geometry in EPSG:4326.
    basename : str
        Prefix for generated tile names.
    grid_size_m : float
        Tile size in meters in the tiling CRS.
    utm_epsg : int
        UTM EPSG code used for tiling (e.g., 32631).
    attributes : Optional[dict[str, object]], optional
        Additional attributes to include in the output grid, by default None.

    Returns
    -------
    gpd.GeoDataFrame
        Tiled GeoDataFrame with `tile_name` and geometry in EPSG:4326.
    """
    tiling_crs = f"EPSG:{utm_epsg}"
    poly_gdf = gpd.GeoDataFrame(geometry=[polygon_wgs84], crs="EPSG:4326")
    poly_gdf = poly_gdf.to_crs(tiling_crs)
    poly_epsg = poly_gdf.geometry.iloc[0]

    minx, miny, maxx, maxy = poly_epsg.bounds
    x_coords = np.arange(minx, maxx, grid_size_m)
    y_coords = np.arange(miny, maxy, grid_size_m)

    geometries_epsg = []
    for x in x_coords:
        for y in y_coords:
            tile_geom = box(
                x,
                y,
                min(x + grid_size_m, maxx),
                min(y + grid_size_m, maxy),
            )
            if not tile_geom.intersects(poly_epsg):
                continue
            clipped = tile_geom.intersection(poly_epsg)
            if clipped.is_empty:
                continue
            geometries_epsg.append(clipped)

    grid_epsg = gpd.GeoDataFrame(geometry=geometries_epsg, crs=tiling_crs)
    grid = grid_epsg.to_crs(epsg=4326)

    # create unique tile names based on the basename and index;
    # ensure it's a string type
    grid["tile_name"] = [f"{basename}_{i}" for i in range(len(grid))]

    # Conserve specific attributes from the original AOI if provided (e.g. season dates)
    if attributes:
        for key, value in attributes.items():
            grid[key] = value

    return grid


def _merge_small_tiles(
    grid: gpd.GeoDataFrame,
    min_size_m: int = 100,
    max_gap_m: int = 20,
) -> gpd.GeoDataFrame:
    """Merge tiles smaller than the given size within the same UTM EPSG.

    Tiles are merged into a neighboring tile within the same UTM EPSG based on
    shared edge length (preferred) or a small gap distance between polygons.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid with `geometry_utm_wkt` and `epsg_utm` columns.
    min_size_m : int, optional
        Minimum width or height in meters before a tile is merged, by default 100.
    max_gap_m : int, optional
        Maximum edge-to-edge distance (meters) to still merge tiles, by default 20.

    Returns
    -------
    gpd.GeoDataFrame
        Grid with merged small tiles and updated UTM geometry.
    """
    if grid.empty:
        return grid
    if "geometry_utm_wkt" not in grid.columns or "epsg_utm" not in grid.columns:
        return grid

    grid = grid.copy()

    def _load_utm(value: object):
        if isinstance(value, str):
            return shapely_wkt.loads(value)
        return None

    grid["_utm_geom"] = grid["geometry_utm_wkt"].apply(_load_utm)

    # Compute width/height from bounds in the tile CRS.
    grid["_width_m"] = grid["_utm_geom"].apply(
        lambda g: (g.bounds[2] - g.bounds[0]) if g is not None else None
    )
    grid["_height_m"] = grid["_utm_geom"].apply(
        lambda g: (g.bounds[3] - g.bounds[1]) if g is not None else None
    )

    small_mask = (grid["_width_m"] < min_size_m) | (grid["_height_m"] < min_size_m)
    if not small_mask.any():
        return grid.drop(columns=["_utm_geom", "_width_m", "_height_m"])

    to_drop: set[int] = set()
    updates: dict[int, dict[str, object]] = {}

    from shapely import to_wkt

    for epsg, group in grid.groupby("epsg_utm"):
        idxs = group.index
        if len(idxs) <= 1:
            continue

        # Work in the UTM CRS for adjacency and merge operations.
        geom_epsg = group["_utm_geom"]
        gdf_epsg = gpd.GeoDataFrame(
            group.drop(columns=["geometry", "_utm_geom"]).copy(),
            geometry=geom_epsg,
            crs=f"EPSG:{int(epsg)}",
        )

        small_idxs = gdf_epsg.index[
            (gdf_epsg["_width_m"] < min_size_m) | (gdf_epsg["_height_m"] < min_size_m)
        ].tolist()

        for idx in small_idxs:
            if idx in to_drop:
                continue
            if idx not in gdf_epsg.index:
                continue

            geom = gdf_epsg.at[idx, "geometry"]
            candidates = gdf_epsg.drop(index=idx)
            if candidates.empty:
                continue

            # Prefer neighbors with the largest shared edge; fall back to nearby tiles.
            shared = candidates.geometry.boundary.intersection(geom.boundary)
            shared_len = shared.length
            touching = shared_len > 0
            if touching.any():
                neighbor_idx = shared_len.idxmax()
            else:
                distances = candidates.geometry.distance(geom)
                min_distance = distances.min()
                if pd.isna(min_distance) or min_distance > max_gap_m:
                    continue
                neighbor_idx = distances.idxmin()

            neighbor_geom = gdf_epsg.at[neighbor_idx, "geometry"]
            merged = neighbor_geom.union(geom)
            gdf_epsg.at[neighbor_idx, "geometry"] = merged

            to_drop.add(idx)
            gdf_epsg = gdf_epsg.drop(index=idx)

        # Persist updated geometries back to the main grid.
        for idx, row in gdf_epsg.iterrows():
            geom = row.geometry
            updates[idx] = {
                "geometry_utm_wkt": to_wkt(geom, rounding_precision=6),
                "geometry": gpd.GeoSeries([geom], crs=gdf_epsg.crs)
                .to_crs(epsg=4326)
                .iloc[0],
                "_utm_geom": geom,
                "_width_m": geom.bounds[2] - geom.bounds[0],
                "_height_m": geom.bounds[3] - geom.bounds[1],
            }

    if to_drop:
        grid = grid.drop(index=list(to_drop))

    for idx, values in updates.items():
        for key, value in values.items():
            grid.at[idx, key] = value

    return grid.drop(columns=["_utm_geom", "_width_m", "_height_m"])


def create_production_grid(
    aoi_gdf: gpd.GeoDataFrame,
    tiling_size_km: int,
    web_mercator_grid: bool = False,
    min_tile_size_m: int = 100,
) -> gpd.GeoDataFrame:
    """Split AOIs into an S2-aligned production grid.

    Each AOI is intersected with the Sentinel-2 MGRS grid and then tiled within
    each S2 tile using a UTM CRS derived from the MGRS code.

    Parameters
    ----------
    aoi_gdf : gpd.GeoDataFrame
        AOI geometries with a valid CRS.
    tiling_size_km : int
        Tile size in kilometers.
    web_mercator_grid : bool, optional
        Whether to use the Web Mercator S2 grid, by default False.
    min_tile_size_m : int, optional
        Minimum width or height (m) before merging, by default 100.

    Returns
    -------
    gpd.GeoDataFrame
        Production grid with `tile_name`, `geometry_utm_wkt`, and `epsg_utm`.
    """

    # Ensure we work in WGS84
    gdf = aoi_gdf.to_crs(epsg=4326)

    # Basic check on ID column presence, uniqueness, and nulls
    if "id" not in gdf.columns:
        raise ValueError("Input GeoDataFrame must contain an 'id' column.")
    if gdf["id"].isna().any():
        raise ValueError("AOI 'id' column contains null values.")
    if gdf["id"].duplicated().any():
        raise ValueError("AOI 'id' values must be unique.")

    # Checking which attributes need to be preserved
    preserve_cols = (
        ["start_date", "end_date"]
        if "start_date" in gdf.columns and "end_date" in gdf.columns
        else []
    )
    # any season specifications also need to be preserved
    season_cols = [
        col
        for col in gdf.columns
        if col.startswith("season_start_") or col.startswith("season_end_")
    ]
    preserve_cols.extend(season_cols)
    logger.info(f"Preserving AOI attributes in the grid: {preserve_cols}")

    # Load the S2 grid once and reuse for all AOIs; ensure it has a CRS.
    s2_grid = load_s2_grid(web_mercator=web_mercator_grid)[["tile", "geometry"]]
    if s2_grid.crs is None:
        raise ValueError("Sentinel-2 grid is missing a CRS.")

    # Treat each AOI separately to handle different S2 tile intersections
    # and tiling in the appropriate UTM CRS.
    grid_dfs = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            raise ValueError(f"AOI geometry at index {idx} is empty.")

        aoi_s2 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(s2_grid.crs).iloc[0]
        s2_hits = s2_grid[s2_grid.intersects(aoi_s2)]
        aoi_id = str(row["id"])
        attributes = row[preserve_cols].to_dict() if preserve_cols else {}

        for _, s2_row in s2_hits.iterrows():
            tile_id = str(s2_row["tile"])
            utm_epsg = _mgrs_tile_to_utm_epsg(tile_id)
            if utm_epsg is None:
                continue

            tile_geom = s2_row.geometry.intersection(aoi_s2)
            if tile_geom.is_empty:
                continue

            tile_geom_wgs84 = (
                gpd.GeoSeries([tile_geom], crs=s2_grid.crs).to_crs(epsg=4326).iloc[0]
            )
            grid_dfs.append(
                _create_tiling_grid_for_polygon(
                    polygon_wgs84=tile_geom_wgs84,
                    basename=f"{aoi_id}_{tile_id}",
                    grid_size_m=tiling_size_km * 1000,
                    utm_epsg=utm_epsg,
                    attributes=attributes,
                )
            )

    if not grid_dfs:
        raise ValueError("No S2 tiles intersect the provided AOIs.")

    grid = gpd.GeoDataFrame(pd.concat(grid_dfs, ignore_index=True))
    grid = ensure_utm_grid(
        grid,
        web_mercator_grid=web_mercator_grid,
    )
    if min_tile_size_m > 0:
        grid = _merge_small_tiles(grid, min_size_m=min_tile_size_m)

    logger.info(f"Created production grid with {len(grid)} tiles.")
    return grid
