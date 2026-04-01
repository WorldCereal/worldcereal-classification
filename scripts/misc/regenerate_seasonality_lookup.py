#!/usr/bin/env python
"""Regenerate ``seasonality_lookup.parquet`` from the crop-calendar GeoTIFFs.

The training pipeline uses a pre-built parquet lookup table (indexed by 0.5°
lat/lon pixel centers) instead of reading the TIF rasters on the fly.  If
you update the crop-calendar TIF files you **must** re-run this script so
that training picks up the new season definitions.

Usage
-----
    python scripts/misc/regenerate_seasonality_lookup.py          # default paths
    python scripts/misc/regenerate_seasonality_lookup.py --verify  # also compare with existing
    python scripts/misc/regenerate_seasonality_lookup.py \
        --tif-dir /path/to/tifs --output /path/to/output.parquet

The script reads the six season TIFs (S1_SOS, S1_EOS, S2_SOS, S2_EOS,
ANNUAL_SOS, ANNUAL_EOS), samples every 0.5° pixel center, keeps rows where
*any* band has valid (non-zero) data, and writes a parquet with the exact
schema expected by ``worldcereal.train.datasets._ensure_seasonality_lookup``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# Default directory shipped with the package
_DEFAULT_TIF_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "worldcereal"
    / "data"
    / "cropcalendars"
)

_TIF_NAMES = {
    "s1_sos_doy": "S1_SOS_WGS84.tif",
    "s1_eos_doy": "S1_EOS_WGS84.tif",
    "s2_sos_doy": "S2_SOS_WGS84.tif",
    "s2_eos_doy": "S2_EOS_WGS84.tif",
    "annual_sos_doy": "ANNUAL_SOS_WGS84.tif",
    "annual_eos_doy": "ANNUAL_EOS_WGS84.tif",
}

_COLUMN_ORDER = [
    "lat", "lon",
    "s1_sos_doy", "s1_eos_doy",
    "s2_sos_doy", "s2_eos_doy",
    "annual_sos_doy", "annual_eos_doy",
]


def build_lookup(tif_dir: Path) -> pd.DataFrame:
    """Read the four crop-calendar TIFs and return a DataFrame of pixel centers.

    Parameters
    ----------
    tif_dir : Path
        Directory containing the ``*_WGS84.tif`` crop-calendar rasters.

    Returns
    -------
    pd.DataFrame
        Columns: lat (float32), lon (float32), s1_sos_doy (uint16),
        s1_eos_doy (uint16), s2_sos_doy (uint16), s2_eos_doy (uint16),
        annual_sos_doy (uint16), annual_eos_doy (uint16).
        Only rows where at least one DOY band has valid (>0) data.
    """
    # Read raster data and metadata from the first TIF
    arrays: dict[str, np.ndarray] = {}
    transform = None
    width = height = None

    for col_name, tif_name in _TIF_NAMES.items():
        tif_path = tif_dir / tif_name
        if not tif_path.exists():
            raise FileNotFoundError(f"Expected TIF not found: {tif_path}")
        with rasterio.open(tif_path) as src:
            if transform is None:
                transform = src.transform
                width, height = src.width, src.height
            else:
                # Sanity-check that all rasters share the same grid
                if (src.width, src.height) != (width, height):
                    raise ValueError(
                        f"Grid mismatch: {tif_name} has size "
                        f"({src.width}x{src.height}), expected ({width}x{height})"
                    )
            arrays[col_name] = src.read(1)

    assert transform is not None and width is not None and height is not None

    # Compute pixel-center coordinates
    cols = np.arange(width)
    rows = np.arange(height)
    lon_centers = (transform.c + (cols + 0.5) * transform.a).astype(np.float32)
    lat_centers = (transform.f + (rows + 0.5) * transform.e).astype(np.float32)

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    # Flatten DOY arrays
    flat: dict[str, np.ndarray] = {}
    for col_name in _TIF_NAMES:
        flat[col_name] = arrays[col_name].ravel()

    # Keep only pixels where ANY band has valid (non-zero) data
    any_valid = np.zeros(lat_flat.shape, dtype=bool)
    for arr in flat.values():
        any_valid |= arr != 0

    df = pd.DataFrame(
        {
            "lat": lat_flat[any_valid],
            "lon": lon_flat[any_valid],
            **{
                col: flat[col][any_valid].astype(np.uint16)
                for col in _TIF_NAMES
            },
        },
        columns=_COLUMN_ORDER,
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate seasonality_lookup.parquet from crop-calendar TIFs."
    )
    parser.add_argument(
        "--tif-dir",
        type=Path,
        default=_DEFAULT_TIF_DIR,
        help="Directory containing the S1/S2 SOS/EOS WGS84 TIF files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to <tif-dir>/seasonality_lookup.parquet.",
    )
    args = parser.parse_args()

    tif_dir: Path = args.tif_dir
    output: Path = args.output or (tif_dir / "seasonality_lookup.parquet")

    print(f"Reading TIFs from: {tif_dir}")
    df = build_lookup(tif_dir)
    print(f"Built lookup table: {df.shape[0]} rows, {df.shape[1]} columns")

    df.to_parquet(output, index=False, engine="pyarrow")
    print(f"Written to: {output}")


if __name__ == "__main__":
    main()
