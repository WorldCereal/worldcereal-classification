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


# -------------------------------------------------------------------------------
# CREATE OPENEO PROCESS GRAPH FROM UDP AND RUN IT
# -------------------------------------------------------------------------------

c = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()

inference_cube = c.datacube_from_process(
    process_id="worldcereal_crop_type",
    namespace="https://raw.githubusercontent.com/WorldCereal/worldcereal-classification/refs/heads/582-create-pytoch-udps/scripts/udp/worldcereal_crop_type.json",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    orbit_state=orbit_state,
    season_ids=season_ids,
    season_windows=season_windows
)


job = inference_cube.execute_batch(
    title="WorldCereal Crop Type UDP Test",
    auto_add_save_result=False,
)
