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

# URL to the seasonal model zip
model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/MultiTask/presto-prometheo-dualtask-SeasonalMultiTaskLoss-month-augment%3DTrue-balance%3DTrue-timeexplicit%3DTrue-masking%3Denabled-run%3D202601240103.zip"

# I would recommend already running the select_s1_orbit_state_vvvh from GFMap during job database creation
orbit_state = "DESCENDING"  # "ASCENDING" or "DESCENDING"

season_ids = ["tc-s1"]
season_windows = {"tc-s1": ("2020-12-01", "2021-07-31")}

postprocess_method = "majority_vote"  # "majority_vote" or "smooth_probabilities", defaults to "majority_vote" if not provided

postprocess_kernel_size = 3  # Only used if postprocessing_method is "majority_vote". Must be an odd integer, defaults to 5 if not provided.


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# CREATE OPENEO PROCESS GRAPH FROM UDP AND RUN IT
# -------------------------------------------------------------------------------

c = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()

inference_cube = c.datacube_from_process(
    process_id="worldcereal_crop_type",
    namespace="https://raw.githubusercontent.com/WorldCereal/worldcereal-classification/refs/heads/582-create-pytoch-udps/scripts/udp/worldcereal_crop_type.json",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    model_url=model_url,
    orbit_state=orbit_state,
    season_ids=season_ids,
    season_windows=season_windows,
    postprocess_method=postprocess_method,
    postprocess_kernel_size=postprocess_kernel_size,
)


job = inference_cube.execute_batch(
    title="WorldCereal Crop Type UDP Test",
    auto_add_save_result=False,
)
