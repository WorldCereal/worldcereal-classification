import json
import random
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import openeo
import pandas as pd
import shapely
import xarray as xr
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import MultiBackendJobManager
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_splitters import load_s2_grid
from tabulate import tabulate

from worldcereal.job import create_inputs_process_graph

from .extractions import (
    NODATAVALUE,
    WORLDCEREAL_BANDS,
    _apply_band_scaling,
)

MAX_RETRIES = 50
BASE_DELAY = 0.1  # initial delay in seconds
MAX_DELAY = 10

REQUIRED_ATTRIBUTES = ["tile_name", "geometry_utm_wkt", "epsg_utm"]


class InferenceJobManager(MultiBackendJobManager):
    def on_job_done(self, job: BatchJob, row):
        logger.info(f"Job {job.job_id} completed")
        output_dir = generate_output_path_inference(self._root_dir, 0, row)

        # Get job results
        job_result = job.get_results()

        # Get metadata
        job_metadata = job.describe()
        result_metadata = job_result.get_metadata()
        job_metadata_path = output_dir / f"job_{job.job_id}.json"
        result_metadata_path = output_dir / f"result_{job.job_id}.json"

        # Get the products
        assets = job_result.get_assets()
        for asset in assets:
            filepath = asset.download(target=output_dir)

            # We want to add the tile name to the filename
            new_filename = f"{filepath.stem}_{row.tile_name}.nc"
            new_filepath = filepath.parent / new_filename

            shutil.move(filepath, new_filepath)

        with job_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(job_metadata, f, ensure_ascii=False)
        with result_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(result_metadata, f, ensure_ascii=False)

        logger.success("Job completed")


def create_worldcereal_inputsjob(
    row: pd.Series,
    connection: openeo.Connection,
    provider,
    connection_provider,
    s1_orbit_state: Literal["ASCENDING", "DESCENDING"] | None,
):
    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)
    bounds = shapely.from_wkt(row.geometry_utm_wkt).bounds
    rounded_bounds = tuple(round(coord / 20) * 20 for coord in bounds)
    spatial_extent = BoundingBoxExtent(*rounded_bounds, epsg=int(row["epsg_utm"]))

    preprocessed_inputs = create_inputs_process_graph(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        s1_orbit_state=s1_orbit_state,
        target_epsg=int(row["epsg_utm"]),
        tile_size=None,
    )

    # Submit the job
    job_options = {
        "driver-memory": "4g",
        "executor-memory": "2g",
        "executor-memoryOverhead": "1g",
        "python-memory": "4g",
        "soft-errors": 0.1,
        "image-name": "python311",
        "max-executors": 10,
    }

    return preprocessed_inputs.create_job(
        title=f"WorldCereal collect inputs for {row.tile_name}",
        job_options=job_options,
    )


def generate_output_path_inference(
    root_folder: Path,
    geometry_index: int,
    row: pd.Series,
    asset_id: Optional[str] = None,
) -> Path:
    """Method to generate the output path for inference jobs.

    Parameters
    ----------
    root_folder : Path
        root folder where the output parquet file will be saved
    geometry_index : int
        For point extractions, only one asset (a geoparquet file) is generated per job.
        Therefore geometry_index is always 0. It has to be included in the function signature
        to be compatible with the GFMapJobManager
    row : pd.Series
        the current job row from the GFMapJobManager
    asset_id : str, optional
        Needed for compatibility with GFMapJobManager but not used.

    Returns
    -------
    Path
        output path for the point extractions parquet file
    """

    tile_name = row.tile_name

    # Create the subfolder to store the output
    subfolder = root_folder / str(tile_name)
    subfolder.mkdir(parents=True, exist_ok=True)

    return subfolder


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


def build_production_grid(
    in_path: Path,
    out_path: Path,
    id_source: Optional[str] = None,
    web_mercator_grid: bool = True,
):
    """Build production grid with UTM geometries and EPSG codes.

    Parameters
    ----------
    in_path : Path
        Path to input shapefile or geoparquet with geometries.
    out_path : Path
        Path to output production grid GeoParquet file.
    id_source : Optional[str]
        Column name to use as unique identifier for each geometry.
    web_mercator_grid : bool, optional
        Whether to use Web Mercator S2 grid for tile assignment, by default True.
    """

    logger.info(f"Loading GeoDataFrame: {in_path}")
    gdf = _load_gdf(in_path)
    logger.info("Converting to WGS84 ...")
    gdf = gdf.to_crs(epsg=4326)

    # Validate id-source column
    if id_source is None:
        # create a default id column
        gdf["id"] = range(len(gdf))
        id_col = "id"
    else:
        id_col = id_source
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
            f"Warning: {missing} features have no UTM EPSG (outside bounds or join failure)."
        )

    logger.info(f"Writing output to {out_path}")
    # Note: Only one active geometry w/ single CRS allowed. UTM variants stored as WKT per row.
    enriched.to_parquet(out_path)
    logger.info("Production grid created.")
    return


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
) -> pd.Series:
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
            print(
                f"Warning: failed to reproject subset to EPSG:{epsg} -> {e}",
                file=sys.stderr,
            )
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


def collect_worldcereal_inputs_patches(
    shapefile: Path,
    output_folder: Path,
    start_date: str,
    end_date: str,
    id_source: Optional[str] = None,
    web_mercator_grid: bool = True,
    parallel_jobs: int = 20,
    randomize_production_grid: bool = False,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    restart_failed: bool = True,
):
    """Collect WorldCereal input data patches using GFMap InferenceJobManager."""

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Construct production grid with all openeo jobs
    production_grid_file = output_folder / "production_grid.geoparquet"
    if not production_grid_file.is_file():
        build_production_grid(
            shapefile,
            production_grid_file,
            id_source=id_source,
            web_mercator_grid=web_mercator_grid,
        )

    # Create a job dataframe if it does not exist
    job_tracking_csv = output_folder / "job_tracking.csv"
    if job_tracking_csv.is_file():
        logger.info("Job tracking file already exists, skipping job creation.")
        job_df = pd.read_csv(job_tracking_csv)

        if restart_failed:
            logger.info("Resetting failed jobs.")
            job_df.loc[
                job_df["status"].isin(["error", "start_failed"]),
                "status",
            ] = "not_started"

            # Save new job tracking dataframe
            job_df.to_csv(job_tracking_csv, index=False)

    else:
        logger.info("Job tracking file does not exist, creating new jobs.")

        production_gdf = gpd.read_parquet(production_grid_file)

        # Check if all required attributes are present in the production_gdf
        missing_attributes = [
            attr for attr in REQUIRED_ATTRIBUTES if attr not in production_gdf.columns
        ]
        if missing_attributes:
            raise ValueError(
                f"The following required attributes are missing in the production grid: {missing_attributes}"
            )

        if randomize_production_grid:
            logger.info("Randomizing the production grid tiles.")
            production_gdf = production_gdf.sample(frac=1).reset_index(drop=True)

        job_df = production_gdf[REQUIRED_ATTRIBUTES].copy()
        job_df["start_date"] = start_date
        job_df["end_date"] = end_date

    # Retry loop starts here
    attempt = 0
    while True:
        try:
            # Setup connection + manager
            connection = cdse_connection()
            logger.info("Setting up the job manager.")
            manager = InferenceJobManager(root_dir=output_folder)
            manager.add_backend(
                "cdse", connection=connection, parallel_jobs=parallel_jobs
            )

            # Kick off all jobs
            manager.run_jobs(
                df=job_df,
                start_job=partial(
                    create_worldcereal_inputsjob,
                    s1_orbit_state=s1_orbit_state,
                ),
                job_db=job_tracking_csv,
            )
            logger.info("All jobs submitted successfully.")
            break  # success: exit loop

        except Exception as exc:
            if attempt < MAX_RETRIES:
                attempt += 1
                # Exponential backoff with full jitter, capped at MAX_DELAY seconds
                backoff = min(BASE_DELAY * 2**attempt, MAX_DELAY)
                jitter = random.uniform(
                    -0.2 * backoff, 0.2 * backoff
                )  # ±20% of backoff
                delay = max(0, backoff + jitter)
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue
            # Non-retryable or maxed-out
            logger.error(f"Max retries reached. Last error: {exc}")
            raise

    return


# ---------------------------------------------------------------------------
# Utility functions to inspect locally downloaded preprocessed input NetCDFs
# ---------------------------------------------------------------------------


def get_band_statistics_netcdf(ds: xr.Dataset) -> pd.DataFrame:
    """Compute scaled statistics for supported WorldCereal bands in an xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing band variables; 'crs' is ignored if present.

    Returns
    -------
    pd.DataFrame
        Index: band name. Columns: %_nodata, min, max, mean, std.

    Notes
    -----
    Nodata defined by NODATAVALUE (65535) or non-finite values. Unsupported bands skipped.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray.Dataset")

    stats = {}
    skipped = []

    for bandname, da in ds.data_vars.items():
        if bandname == "crs":
            continue
        if not any(bandname in v for v in WORLDCEREAL_BANDS.values()):
            skipped.append(bandname)
            continue

        arr = da.values
        flat = arr.ravel()
        total = flat.size
        if total == 0:
            continue

        nodata_mask = (flat == NODATAVALUE) | (~np.isfinite(flat))
        valid = flat[~nodata_mask]
        nodata_pct = (nodata_mask.sum() / total * 100.0) if total else 0.0

        if valid.size == 0:
            stats[bandname] = {
                "%_nodata": f"{nodata_pct:.2f}",
                "min": "N/A",
                "max": "N/A",
                "mean": "N/A",
                "std": "N/A",
            }
            continue

        scaled = _apply_band_scaling(valid.astype(np.float32), bandname)
        stats[bandname] = {
            "%_nodata": f"{nodata_pct:.2f}",
            "min": f"{scaled.min():.4f}",
            "max": f"{scaled.max():.4f}",
            "mean": f"{scaled.mean():.4f}",
            "std": f"{scaled.std():.4f}",
        }

    if not stats:
        raise ValueError("No supported bands with data found.")

    df = pd.DataFrame(stats).T
    print("-------------------------------------")
    print("Band statistics (NetCDF):")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=True))
    if skipped:
        logger.warning(f"Skipped unsupported bands: {', '.join(skipped)}")
    return df


def visualize_timeseries_netcdf(
    ds: xr.Dataset,
    band: str = "NDVI",
    npixels: int = 5,
    pixel_coords: Optional[list[tuple]] = None,
    time_dim: str = "t",
    y_dim: str = "y",
    x_dim: str = "x",
    random_seed: Optional[int] = None,
    outfile: Optional[Path] = None,
    nodata: int = NODATAVALUE,
) -> None:
    """Visualize time series for random or specified pixels in a preprocessed inputs Dataset.

    If band == 'NDVI', it is computed from S2-L2A-B08 (NIR) and S2-L2A-B04 (RED).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    band : str, default 'NDVI'
        Band to plot or 'NDVI'.
    npixels : int, default 5
        Number of random pixels if pixel_coords not given.
    pixel_coords : list[tuple], optional
        Explicit (row, col) coordinates.
    time_dim, y_dim, x_dim : str
        Dimension names.
    random_seed : int, optional
        Seed for reproducible sampling.
    outfile : Path, optional
        File path to save figure.
    nodata : int
        Nodata sentinel value.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray.Dataset")

    for dim in (time_dim, y_dim, x_dim):
        if dim not in ds.dims:
            raise KeyError(f"Dimension '{dim}' not in dataset.")

    if band == "NDVI":
        for needed in ["S2-L2A-B08", "S2-L2A-B04"]:
            if needed not in ds.data_vars:
                raise KeyError(
                    f"Band '{needed}' required for NDVI but missing from dataset"
                )
        nir = (
            ds["S2-L2A-B08"]
            .astype("float32")
            .where(lambda v: (v != nodata) & np.isfinite(v))
        )
        red = (
            ds["S2-L2A-B04"]
            .astype("float32")
            .where(lambda v: (v != nodata) & np.isfinite(v))
        )
        data_da = (nir - red) / (nir + red)
        ylabel = "NDVI"
    else:
        if band not in ds.data_vars:
            raise KeyError(f"Band '{band}' not found in dataset")
        data_da = (
            ds[band].astype("float32").where(lambda v: (v != nodata) & np.isfinite(v))
        )
        # Apply scaling if supported
        try:
            vals = data_da.values
            mask_valid = (vals != nodata) & np.isfinite(vals)
            scaled = vals.copy()
            scaled[mask_valid] = _apply_band_scaling(scaled[mask_valid], band)
            data_da = data_da.copy(data=scaled)
        except ValueError:
            pass
        ylabel = band

    ny, nx = data_da.sizes[y_dim], data_da.sizes[x_dim]
    if pixel_coords is None:
        rng = np.random.default_rng(random_seed)
        finite_mask = np.isfinite(data_da).any(dim=time_dim).values
        valid_idx = np.argwhere(finite_mask)
        if valid_idx.size == 0:
            raise ValueError("No valid pixels found for plotting.")
        if npixels > len(valid_idx):
            logger.warning(
                f"Requested {npixels} pixels; only {len(valid_idx)} valid. Using all valid."
            )
            npixels = len(valid_idx)
        sel = rng.choice(len(valid_idx), size=npixels, replace=False)
        pixel_coords = [tuple(valid_idx[i]) for i in sel]
    else:
        for r, c in pixel_coords:
            if not (0 <= r < ny and 0 <= c < nx):
                raise IndexError(f"Pixel ({r},{c}) out of bounds ({ny},{nx}).")

    fig, ax = plt.subplots(figsize=(12, 6))
    time_values = pd.to_datetime(data_da[time_dim].values)
    for r, c in pixel_coords:
        ts = data_da.isel({y_dim: r, x_dim: c}).values
        ax.plot(time_values, ts, marker="o", linestyle="-", label=f"(row={r}, col={c})")

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=90)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.subplots_adjust(right=0.75)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if outfile is not None:
        plt.savefig(outfile)
    return
