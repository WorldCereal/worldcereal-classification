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

season_ids = ["2021"]
season_windows = {"2021": ("2020-12-01", "2021-07-31")}

##########
# Optional parameters that can be used to override the defaults in the UDP.
##########
# seasonal_model_zip = "http_link_to_custom_presto_backbone_model.zip"
# landcover_head_zip = "http_link_to_custom_landcover_head.zip"
# postprocess_method = "majority_vote" # or "smooth_probabilities"
# postprocess_kernel_size = 3 # Must be an odd integer not larger than 25.
###########

# -------------------------------------------------------------------------------
# CREATE OPENEO PROCESS GRAPH FROM UDP AND RUN IT
# -------------------------------------------------------------------------------

c = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()

inference_cube = c.datacube_from_process(
    process_id="worldcereal_crop_extent",
    namespace="https://raw.githubusercontent.com/WorldCereal/worldcereal-classification/refs/tags/worldcereal_crop_extent_v3.0.0/scripts/udp/worldcereal_crop_extent.json",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    orbit_state=orbit_state,
    season_ids=season_ids,
    season_windows=season_windows,
    # seasonal_model_zip=seasonal_model_zip,
    # landcover_head_zip=landcover_head_zip,
    # postprocess_method=postprocess_method,
    # postprocess_kernel_size=postprocess_kernel_size,
)


job = inference_cube.execute_batch(
    title="WorldCereal Crop Extent UDP Test",
    auto_add_save_result=False,
)
