import xarray as xr
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData

# Import the UDF functions directly
from worldcereal.openeo.inference_catboost import apply_udf_data as inference_udf
from worldcereal.parameters import CropLandParameters, CropTypeParameters


def create_udf_data(arr: xr.DataArray, parameters: dict, epsg: int = 32631) -> UdfData:
    """Create UdfData object from xarray DataArray and parameters."""
    cube = XarrayDataCube(arr)
    udf_data = UdfData(
        datacube_list=[cube], user_context=parameters, proj={"EPSG": epsg}
    )
    return udf_data


# Helper to assert classification range while ignoring sentinel (no cropland) value 254
def _assert_value_range(
    arr: xr.DataArray, band_name: str, min_val: int, max_val: int, ignore: int = 254
):
    vals = arr.sel(bands=band_name).values
    valid = vals[vals != ignore]
    if valid.size == 0:
        return
    assert valid.max() <= max_val
    assert valid.min() >= min_val


def test_cropland_inference(WorldCerealPreprocessedInputs):
    """Test the local generation of a cropland product using direct UDF calls"""

    # Initialize CropLandParameters
    cropland_params = CropLandParameters().model_dump()

    # Create UDF data for the dual workflow
    udf_data = create_udf_data(WorldCerealPreprocessedInputs, cropland_params)

    # Run the unified dual UDF
    result = inference_udf(udf_data)
    cropland_classification = result.datacube_list[0].get_array()

    assert list(cropland_classification.bands.values) == [
        "classification",
        "probability",
        "probability_other",
        "probability_cropland",
    ]
    _assert_value_range(cropland_classification, "classification", 0, 1)
    assert cropland_classification.sel(bands="probability").values.max() <= 100
    assert cropland_classification.sel(bands="probability").values.min() >= 0
    assert cropland_classification.shape == (4, 100, 100)


def test_cropland_inference_with_intermediate(WorldCerealPreprocessedInputs):
    """Test the local generation of a cropland product using direct UDF calls"""

    # Initialize CropLandParameters
    cropland_params = CropLandParameters().model_dump()
    cropland_params["postprocess_parameters"]["save_intermediate"] = True

    # Create UDF data for the dual workflow
    udf_data = create_udf_data(WorldCerealPreprocessedInputs, cropland_params)

    # Run the unified dual UDF
    result = inference_udf(udf_data)
    cropland_classification = result.datacube_list[0].get_array()

    assert list(cropland_classification.bands.values) == [
        "classification",
        "probability",
        "probability_other",
        "probability_cropland",
        "raw_classification",
        "raw_probability",
        "raw_probability_other",
        "raw_probability_cropland",
    ]
    _assert_value_range(cropland_classification, "classification", 0, 1)
    assert cropland_classification.sel(bands="probability").values.max() <= 100
    assert cropland_classification.sel(bands="probability").values.min() >= 0
    assert cropland_classification.shape == (8, 100, 100)


def test_croptype_inference(WorldCerealPreprocessedInputs):
    """Test the local generation of a croptype product using direct UDF calls"""

    print("Get Presto croptype features")

    # Initialize CropTypeParameters with custom target_date
    croptype_params = CropTypeParameters(
        target_date=None, mask_cropland=False
    ).model_dump()

    # Create UDF data for the dual workflow
    udf_data = create_udf_data(WorldCerealPreprocessedInputs, croptype_params)

    # Run classification UDF
    classification_result = inference_udf(udf_data)
    croptype_classification = classification_result.datacube_list[0].get_array()

    assert list(croptype_classification.bands.values) == [
        "classification",
        "probability",
        "probability_barley",
        "probability_dry_pulses_legumes",
        "probability_fibre_crops",
        "probability_flower_crops",
        "probability_grass_fodder_crops",
        "probability_herb_spice_medicinal_crops",
        "probability_maize",
        "probability_millet",
        "probability_no-crop",
        "probability_oats",
        "probability_other_oilseed",
        "probability_other_root_tuber",
        "probability_potatoes",
        "probability_rapeseed_rape",
        "probability_rice",
        "probability_rye",
        "probability_sorghum",
        "probability_soy_soybeans",
        "probability_spring_barley",
        "probability_spring_oats",
        "probability_spring_rye",
        "probability_spring_triticale",
        "probability_spring_wheat",
        "probability_sugarbeet",
        "probability_sunflower",
        "probability_triticale",
        "probability_vegetables_fruits",
        "probability_wheat",
    ]

    #  First assert below depends on the amount of classes in the model
    _assert_value_range(croptype_classification, "classification", 0, 27)
    assert croptype_classification.sel(bands="probability").values.max() <= 100
    assert croptype_classification.sel(bands="probability").values.min() >= 0
    assert croptype_classification.shape == (30, 100, 100)


def test_dual_workflow(WorldCerealPreprocessedInputs):
    """Test that dual workflow outputs contain exact replicas of individual classifications with prefixes."""

    cropland_parameters = CropLandParameters()
    croptype_parameters = CropTypeParameters()

    parameters = {
        "cropland_params": cropland_parameters.model_dump(),
        "croptype_params": croptype_parameters.model_dump(),
    }

    # Create UDF data for the dual workflow
    udf_data = create_udf_data(WorldCerealPreprocessedInputs, parameters)

    # Run the unified dual UDF
    result = inference_udf(udf_data)
    combined_output = result.datacube_list[0].get_array()

    # Extract cropland and croptype components (last 4 bands are cropland)
    all_bands = list(combined_output.bands.values)
    cropland_bands = [b for b in all_bands if b.startswith("cropland_")]
    croptype_bands = [b for b in all_bands if b.startswith("croptype_")]
    cropland_output = combined_output.sel(bands=cropland_bands)
    croptype_output = combined_output.sel(bands=croptype_bands)

    # Define expected PREFIXED band names (this is what we want in dual workflow)
    expected_cropland_bands = [
        "cropland_classification",
        "cropland_probability",
        "cropland_probability_other",
        "cropland_probability_cropland",
    ]

    expected_croptype_bands = [
        "croptype_classification",
        "croptype_probability",
        "croptype_probability_barley",
        "croptype_probability_dry_pulses_legumes",
        "croptype_probability_fibre_crops",
        "croptype_probability_flower_crops",
        "croptype_probability_grass_fodder_crops",
        "croptype_probability_herb_spice_medicinal_crops",
        "croptype_probability_maize",
        "croptype_probability_millet",
        "croptype_probability_no-crop",
        "croptype_probability_oats",
        "croptype_probability_other_oilseed",
        "croptype_probability_other_root_tuber",
        "croptype_probability_potatoes",
        "croptype_probability_rapeseed_rape",
        "croptype_probability_rice",
        "croptype_probability_rye",
        "croptype_probability_sorghum",
        "croptype_probability_soy_soybeans",
        "croptype_probability_spring_barley",
        "croptype_probability_spring_oats",
        "croptype_probability_spring_rye",
        "croptype_probability_spring_triticale",
        "croptype_probability_spring_wheat",
        "croptype_probability_sugarbeet",
        "croptype_probability_sunflower",
        "croptype_probability_triticale",
        "croptype_probability_vegetables_fruits",
        "croptype_probability_wheat",
    ]

    # Get actual band names
    actual_cropland_bands = list(cropland_output.bands.values)
    actual_croptype_bands = list(croptype_output.bands.values)

    # Assert that bands are exactly the prefixed versions
    assert actual_cropland_bands == expected_cropland_bands, (
        f"Cropland bands mismatch. Got {actual_cropland_bands}, expected {expected_cropland_bands}"
    )

    assert actual_croptype_bands == expected_croptype_bands, (
        f"Croptype bands mismatch. Got {actual_croptype_bands}, expected {expected_croptype_bands}"
    )

    # Verify shapes
    assert cropland_output.shape == (4, 100, 100)
    assert croptype_output.shape == (30, 100, 100)
    assert combined_output.shape == (34, 100, 100)

    # Value checks using the PREFIXED names
    _assert_value_range(cropland_output, "cropland_classification", 0, 1)
    _assert_value_range(cropland_output, "cropland_probability", 0, 100)

    _assert_value_range(croptype_output, "croptype_classification", 0, 27)
    _assert_value_range(croptype_output, "croptype_probability", 0, 100)

    # Check if crop type is masked in the right locations
    cropland_mask = cropland_output.sel(bands="cropland_classification").values == 0
    croptype_data = croptype_output.values
    assert (croptype_data[:, cropland_mask] == 254).all()


def test_dual_workflow_with_intermediate_results(WorldCerealPreprocessedInputs):
    """Test that dual workflow outputs contain exact replicas of individual classifications with prefixes."""

    cropland_parameters = CropLandParameters().model_dump()
    croptype_parameters = CropTypeParameters().model_dump()

    cropland_parameters["postprocess_parameters"]["save_intermediate"] = True
    croptype_parameters["postprocess_parameters"]["save_intermediate"] = True

    parameters = {
        "cropland_params": cropland_parameters,
        "croptype_params": croptype_parameters,
    }

    # Create UDF data for the dual workflow
    udf_data = create_udf_data(WorldCerealPreprocessedInputs, parameters)

    # Run the unified dual UDF
    result = inference_udf(udf_data)
    combined_output = result.datacube_list[0].get_array()

    # Extract cropland and croptype components
    all_bands = list(combined_output.bands.values)
    cropland_bands = [b for b in all_bands if b.startswith("cropland_")]
    croptype_bands = [b for b in all_bands if b.startswith("croptype_")]
    cropland_output = combined_output.sel(bands=cropland_bands)
    croptype_output = combined_output.sel(bands=croptype_bands)

    # Define expected PREFIXED band names (this is what we want in dual workflow)
    expected_cropland_bands = [
        "cropland_classification",
        "cropland_probability",
        "cropland_probability_other",
        "cropland_probability_cropland",
    ]
    expected_cropland_bands += [
        b.replace("cropland_", "cropland_raw_") for b in expected_cropland_bands
    ]

    expected_croptype_bands = [
        "croptype_classification",
        "croptype_probability",
        "croptype_probability_barley",
        "croptype_probability_dry_pulses_legumes",
        "croptype_probability_fibre_crops",
        "croptype_probability_flower_crops",
        "croptype_probability_grass_fodder_crops",
        "croptype_probability_herb_spice_medicinal_crops",
        "croptype_probability_maize",
        "croptype_probability_millet",
        "croptype_probability_no-crop",
        "croptype_probability_oats",
        "croptype_probability_other_oilseed",
        "croptype_probability_other_root_tuber",
        "croptype_probability_potatoes",
        "croptype_probability_rapeseed_rape",
        "croptype_probability_rice",
        "croptype_probability_rye",
        "croptype_probability_sorghum",
        "croptype_probability_soy_soybeans",
        "croptype_probability_spring_barley",
        "croptype_probability_spring_oats",
        "croptype_probability_spring_rye",
        "croptype_probability_spring_triticale",
        "croptype_probability_spring_wheat",
        "croptype_probability_sugarbeet",
        "croptype_probability_sunflower",
        "croptype_probability_triticale",
        "croptype_probability_vegetables_fruits",
        "croptype_probability_wheat",
    ]

    expected_croptype_bands += [
        b.replace("croptype_", "croptype_raw_") for b in expected_croptype_bands
    ]

    # Get actual band names
    actual_cropland_bands = list(cropland_output.bands.values)
    actual_croptype_bands = list(croptype_output.bands.values)

    # Assert that bands are exactly the prefixed versions
    assert actual_cropland_bands == expected_cropland_bands, (
        f"Cropland bands mismatch. Got {actual_cropland_bands}, expected {expected_cropland_bands}"
    )

    assert actual_croptype_bands == expected_croptype_bands, (
        f"Croptype bands mismatch. Got {actual_croptype_bands}, expected {expected_croptype_bands}"
    )

    # Value checks using the PREFIXED names
    _assert_value_range(cropland_output, "cropland_classification", 0, 1)
    _assert_value_range(cropland_output, "cropland_probability", 0, 100)

    _assert_value_range(croptype_output, "croptype_classification", 0, 27)
    _assert_value_range(croptype_output, "croptype_probability", 0, 100)

    # Check if crop type is masked in the right locations
    cropland_mask = cropland_output.sel(bands="cropland_classification").values == 0
    croptype_data = croptype_output.values
    assert (croptype_data[:, cropland_mask] == 254).all()
