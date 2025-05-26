"""This short script demonstrates how to run WorldCereal crop type inference through
an OpenEO User-Defined Process (UDP) on CDSE.

No default model is currently available, meaning that a user first needs to
train a custom model before using this script.

A user is required to specify a spatial and temporal extent, as well as the link to
the crop type model to be used. The model should be hosted on a publicly accessible server.

The user needs to manually download the resulting map through the OpenEO Web UI:
https://openeo.dataspace.copernicus.eu/
Or use the openeo API to fetch the resulting map.
"""

import openeo

###### USER INPUTS ######
# Define the spatial and temporal extent
spatial_extent = {
    "west": 622694.5968575787,
    "south": 5672232.857114074,
    "east": 623079.000934101,
    "north": 5672519.995940826,
    "crs": "EPSG:32631",
    "srs": "EPSG:32631",
}

# Temporal extent needs to consist of exactly twelve months,
# always starting first day of the month and ending last day of the month.
temporal_extent = ["2018-11-01", "2019-10-30"]

# Provide the link to your custom model
model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/downstream/tests/be_multiclass-test_custommodel.onnx"

# Optional parameters
orbit_state = "ASCENDING"  # optional parameter, default is "DESCENDING"
postprocess_method = "smooth_probabilities"  # optional parameter, available choices are "majority_vote" and "smooth_probabilities". Default is "smooth_probabilities"
postprocess_kernel_size = 5  # optional parameter,  default is 5 (only used if postprocess_method is "majority_vote")

###### END OF USER INPUTS ######

# Connect to openeo backend
c = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

# Define the operations
cube = c.datacube_from_process(
    process_id="worldcereal_crop_type",
    namespace="https://raw.githubusercontent.com/WorldCereal/worldcereal-classification/refs/heads/327-lut-crop-type/src/worldcereal/udp/worldcereal_crop_type.json",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    model_url=model_url,
    orbit_state=orbit_state,
    postprocess_method=postprocess_method,
    postprocess_kernel_size=postprocess_kernel_size,
)

# Run the job
job = cube.execute_batch(
    title="Test worldcereal_crop_type UDP",
)
