
import openeo

from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from openeo_gfmap.inference.model_inference import apply_model_inference

from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap
from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CroplandClassifier

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"


def run_inference(spatial_extent, temporal_extent, backend_context,
                  output_path, product='cropland', format='NetCDF'):

    # Connect to openeo
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

    # Run feature computer
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

    if product == 'cropland':
        # initiate default cropland model
        model = CroplandClassifier
        catboost_parameters = {}
    else:
        raise ValueError(f"Product {product} not supported.")

    classes = apply_model_inference(
        model_inference_class=model,
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

    classes.execute_batch(
        outputfile=output_path,
        out_format=format,
        job_options={
            "driver-memory": "4g",
            "executor-memoryOverhead": "12g",
            "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
        },
    )

    return output_path
