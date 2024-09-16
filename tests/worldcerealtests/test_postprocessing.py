from openeo_gfmap.inference.model_inference import (
    EPSG_HARMONIZED_NAME,
    apply_model_inference_local,
)

from worldcereal.openeo.postprocess import PostProcessor


def test_cropland_postprocessing(WorldCerealCroplandClassification):
    """Test the local postprocessing of a cropland product"""

    print("Postprocessing cropland product ...")
    _ = apply_model_inference_local(
        PostProcessor,
        WorldCerealCroplandClassification,
        parameters={"ignore_dependencies": True, EPSG_HARMONIZED_NAME: None},
    )

    print("Running postprocessing UDF locally")
