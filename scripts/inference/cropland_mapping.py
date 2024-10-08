"""Cropland mapping inference script, demonstrating the use of the GFMAP, Presto and WorldCereal classifiers in a first inference pipeline."""

import argparse
from pathlib import Path

from loguru import logger
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import Backend, BackendContext

from worldcereal.job import WorldCerealProductType, generate_map
from worldcereal.parameters import PostprocessParameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="WC - Crop Mapping Inference",
        description="Crop Mapping inference using GFMAP, Presto and WorldCereal classifiers",
    )

    parser.add_argument("minx", type=float, help="Minimum X coordinate (west)")
    parser.add_argument("miny", type=float, help="Minimum Y coordinate (south)")
    parser.add_argument("maxx", type=float, help="Maximum X coordinate (east)")
    parser.add_argument("maxy", type=float, help="Maximum Y coordinate (north)")
    parser.add_argument(
        "start_date", type=str, help="Starting date for data extraction."
    )
    parser.add_argument("end_date", type=str, help="Ending date for data extraction.")
    parser.add_argument(
        "product",
        type=str,
        help="Product to generate. One of ['cropland', 'croptype']",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to folder where to save the resulting GeoTiff.",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=4326,
        help="EPSG code of the input `minx`, `miny`, `maxx`, `maxy` parameters.",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Run postprocessing on the croptype and/or the cropland product after inference.",
    )
    parser.add_argument(
        "--class-probabilities",
        action="store_true",
        help="Output per-class probabilities in the resulting product",
    )

    args = parser.parse_args()

    minx = args.minx
    miny = args.miny
    maxx = args.maxx
    maxy = args.maxy
    epsg = args.epsg

    start_date = args.start_date
    end_date = args.end_date

    product = args.product

    # minx, miny, maxx, maxy = (664000, 5611134, 665000, 5612134)  # Small test
    # minx, miny, maxx, maxy = (664000, 5611134, 684000, 5631134)  # Large test
    # epsg = 32631
    # start_date = "2020-11-01"
    # end_date = "2021-10-31"
    # product = "croptype"

    spatial_extent = BoundingBoxExtent(minx, miny, maxx, maxy, epsg)
    temporal_extent = TemporalContext(start_date, end_date)

    backend_context = BackendContext(Backend.CDSE)

    job_results = generate_map(
        spatial_extent,
        temporal_extent,
        args.output_path,
        product_type=WorldCerealProductType(product),
        postprocess_parameters=PostprocessParameters(
            enable=args.postprocess, keep_class_probs=args.class_probabilities
        ),
        out_format="GTiff",
        backend_context=backend_context,
    )
    logger.success(f"Job finished:\n\t{job_results}")
