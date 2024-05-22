"""Cropland mapping inference script, demonstrating the use of the GFMAP, Presto and WorldCereal classifiers in a first inference pipeline."""

import argparse

from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import Backend, BackendContext, cdse_connection
from openeo_gfmap.features.feature_extractor import PatchFeatureExtractor

from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap


class PrestoFeatureExtractor(PatchFeatureExtractor):
    def __init__(self):
        pass

    def extract(self, image):
        pass


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
        "start_date", type=str, help="Starting date for data extraction."
    )
    parser.add_argument("end_date", type=str, help="Ending date for data extraction.")
    parser.add_argument(
        "output_folder", type=str, help="Path to folder where to save results."
    )

    args = parser.parse_args()

    minx = args.minx
    miny = args.miny
    maxx = args.maxx
    maxy = args.maxy

    start_date = args.start_date
    end_date = args.end_date

    spatial_extent = BoundingBoxExtent(minx, miny, maxx, maxy)
    temporal_extent = TemporalContext(start_date, end_date)

    backend = BackendContext(Backend.CDSE)

    # Preparing the input cube for the inference
    input_cube = worldcereal_preprocessed_inputs_gfmap(
        connection=cdse_connection(),
        backend_context=backend,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    # Start the job and download
    job = input_cube.create_job(
        title=f"Cropland inference BBOX: {minx} {miny} {maxx} {maxy}",
        description="Cropland inference using WorldCereal, Presto and GFMAP classifiers",
        out_format="NetCDF",
    )

    job.start_and_wait()
    job.get_results().download_files(args.output_folder)
