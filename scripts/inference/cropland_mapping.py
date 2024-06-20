"""Cropland mapping inference script, demonstrating the use of the GFMAP, Presto and WorldCereal classifiers in a first inference pipeline."""

import argparse
from pathlib import Path

import openeo
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import Backend, BackendContext
from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from openeo_gfmap.inference.model_inference import apply_model_inference
from openeo_gfmap.preprocessing.scaling import compress_uint8

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"

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

    connection = openeo.connect(
        "https://openeo.creo.vito.be/openeo/"
    ).authenticate_oidc()

    # Preparing the input cube for the inference
    inputs = worldcereal_preprocessed_inputs_gfmap(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    # Test feature computer
    presto_parameters = {
        "rescale_s1": False,  # Will be done in the Presto UDF itself!
    }

    features = apply_feature_extractor(
        feature_extractor_class=PrestoFeatureExtractor,
        cube=inputs,
        parameters=presto_parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    catboost_parameters = {}

    classes = apply_model_inference(
        model_inference_class=CroplandClassifier,
        cube=features,
        parameters=catboost_parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
            {"dimension": "t", "value": "P1D"},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Cast to uint8
    classes = compress_uint8(classes)

    classes.execute_batch(
        outputfile=args.output_path,
        out_format="GTiff",
        job_options={
            "driver-memory": "4g",
            "executor-memoryOverhead": "12g",
            "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
        },
    )
