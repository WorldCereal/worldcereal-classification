"""Cropland mapping inference script, demonstrating the use of the GFMAP, Presto and WorldCereal classifiers in a first inference pipeline."""

import argparse
from pathlib import Path

from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import Backend, BackendContext

from worldcereal.job import generate_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="WC - Cropland Inference",
        description="Cropland inference using GFMAP, Presto and WorldCereal classifiers",
    )

    parser.add_argument("minx", type=float, help="Minimum X coordinate (west)")
    parser.add_argument("miny", type=float, help="Minimum Y coordinate (south)")
    parser.add_argument("maxx", type=float, help="Maximum X coordinate (east)")
    parser.add_argument("maxy", type=float, help="Maximum Y coordinate (north)")
    parser.add_argument(
        "--epsg",
        type=int,
        default=4326,
        help="EPSG code of the input `minx`, `miny`, `maxx`, `maxy` parameters.",
    )
    parser.add_argument(
        "start_date", type=str, help="Starting date for data extraction."
    )
    parser.add_argument("end_date", type=str, help="Ending date for data extraction.")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to folder where to save the resulting NetCDF.",
    )

    args = parser.parse_args()

    minx = args.minx
    miny = args.miny
    maxx = args.maxx
    maxy = args.maxy
    epsg = args.epsg

    start_date = args.start_date
    end_date = args.end_date

    spatial_extent = BoundingBoxExtent(minx, miny, maxx, maxy, epsg)
    temporal_extent = TemporalContext(start_date, end_date)

    backend_context = BackendContext(Backend.FED)

    generate_map(
        spatial_extent,
        temporal_extent,
        backend_context,
        args.output_path,
        product="cropland",
        out_format="GTiff",
    )
