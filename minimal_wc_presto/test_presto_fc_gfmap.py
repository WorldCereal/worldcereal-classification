"""Test the presto feature computer running with GFMAP"""

import openeo
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from openeo_gfmap.inference.model_inference import apply_model_inference

from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap
from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier

EXTENT = dict(
    zip(["west", "south", "east", "north"], [664000.0, 5611120.0, 665000.0, 5612120.0])
)
EXTENT["crs"] = "EPSG:32631"
EXTENT["srs"] = "EPSG:32631"
STARTDATE = "2020-11-01"
ENDDATE = "2021-10-31"

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


if __name__ == "__main__":
    # Test extent
    spatial_extent = BoundingBoxExtent(
        west=EXTENT["west"],
        south=EXTENT["south"],
        east=EXTENT["east"],
        north=EXTENT["north"],
        epsg=32631,
    )

    temporal_extent = TemporalContext(
        start_date=STARTDATE,
        end_date=ENDDATE,
    )
    backend_context = BackendContext(Backend.FED)

    connection = openeo.connect(
        "https://openeo.creo.vito.be/openeo/"
    ).authenticate_oidc()

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
        ]
    )

    classes.execute_batch(
        outputfile=".notebook-tests/presto_prediction_gfmap.nc",
        out_format="NetCDF",
        job_options={
            "driver-memory": "4g",
            "executor-memoryOverhead": "8g",
            "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
        },
    )
