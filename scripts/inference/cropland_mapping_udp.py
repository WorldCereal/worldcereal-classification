"""This short script demonstrates how to run WorldCereal cropland extent inference through
    an OpenEO User-Defined Process (UDP) on CDSE.

    The WorldCereal default cropland model is used and default post-processing is applied
    (using smooth_probabilities method).

    The user needs to manually download the resulting map through the OpenEO Web UI:
    https://openeofed.dataspace.copernicus.eu/
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
temporal_extent = ["2020-01-01", "2020-12-31"]

# This argument is optional and determines the projection of the output products
target_projection = "EPSG:4326"

# Here you can set custom job options, such as driver-memory, executor-memory, etc.
# In case you would like to use the default options, set job_options to None.

# job_options = {
#     "driver-memory": "4g",
#     "executor-memory": "2g",
#     "python-memory": "3g",
# }
job_options = None

###### END OF USER INPUTS ######

# Connect to openeo backend
c = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

# Define the operations
classes = c.datacube_from_process(
    process_id="worldcereal_crop_extent",
    namespace="https://github.com/WorldCereal/worldcereal-classification/blob/worldcereal_crop_extent_v1.0.1/src/worldcereal/udp/worldcereal_crop_extent.json",
    temporal_extent=temporal_extent,
    spatial_extent=spatial_extent,
    projection=target_projection,
)

# Run the job
job = classes.execute_batch(
    title="WorldCereal Crop Extent UDP test",
    out_format="GTiff",
    job_options=job_options,
)
