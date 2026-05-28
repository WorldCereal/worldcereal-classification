import openeo

# -------------------------------------------------------------------------------
# SET PARAMETERS
# -------------------------------------------------------------------------------

minx, miny, maxx, maxy = (664200, 5611334, 664400, 5611534)  # Very Small test
# minx, miny, maxx, maxy = (664000, 5611134, 665000, 5612134)  # Small test
# minx, miny, maxx, maxy = (664000, 5611134, 684000, 5631134)  # Large test
# minx, miny, maxx, maxy = (634000, 5601134, 684000, 5651134)  # Very large test

# EPSG in which bbox is defined.
epsg = 32631

spatial_extent = {
    "west": minx,
    "south": miny,
    "east": maxx,
    "north": maxy,
    "crs": f"EPSG:{epsg}",
}

temporal_extent = ["2020-11-01", "2021-10-31"]

orbit_state = "DESCENDING"  # "ASCENDING" or "DESCENDING"

season_ids = ["tc-s1"]
season_windows = {"tc-s1": ("2020-12-01", "2021-07-31")}

##########
# Optional parameters that can be used to override the defaults in the UDP.
##########
# seasonal_model_zip = "http_link_to_custom_presto_backbone_model.zip"
# landcover_head_zip = "http_link_to_custom_landcover_head.zip"
# croptype_head_zip = "http_link_to_custom_croptype_head.zip"
# enable_cropland_head = True # Set to False to disable the landcover head and only run the croptype head.
# mask_cropland = True # Set to False to disable masking of the croptype predictions based on the cropland predictions.
# postprocess_method_cropland = "majority_vote" # or "smooth_probabilities"
# postprocess_kernel_size_cropland = 3 # Must be an odd integer not larger than 25.
# postprocess_method_croptype = "majority_vote" # or "smooth_probabilities"
# postprocess_kernel_size_croptype = 3 # Must be an odd integer not larger than 25.
###########

# -------------------------------------------------------------------------------
# CREATE OPENEO PROCESS GRAPH FROM UDP AND RUN IT
# -------------------------------------------------------------------------------

c = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()

inference_cube = c.datacube_from_process(
    process_id="worldcereal_crop_type",
    namespace="https://raw.githubusercontent.com/WorldCereal/worldcereal-classification/refs/tags/worldcereal_crop_type_v3.0.0/scripts/udp/worldcereal_crop_type.json",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    orbit_state=orbit_state,
    season_ids=season_ids,
    season_windows=season_windows,
    # seasonal_model_zip=seasonal_model_zip,
    # landcover_head_zip=landcover_head_zip,
    # croptype_head_zip=croptype_head_zip,
    # enable_cropland_head=enable_cropland_head,
    # mask_cropland=mask_cropland,
    # postprocess_method_cropland=postprocess_method_cropland,
    # postprocess_kernel_size_cropland=postprocess_kernel_size_cropland,
    # postprocess_method_croptype=postprocess_method_croptype,
    # postprocess_kernel_size_croptype=postprocess_kernel_size_croptype,
)


job = inference_cube.execute_batch(
    title="WorldCereal Crop Type UDP Test",
    auto_add_save_result=False,
)
