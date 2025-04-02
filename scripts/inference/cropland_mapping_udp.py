"""This short script demonstrates how to run WorldCereal cropland extent inference through
an OpenEO User-Defined Process (UDP) on CDSE.

The WorldCereal default cropland model is used and default post-processing is applied
(using smooth_probabilities method).

The user needs to manually download the resulting map through the OpenEO Web UI:
https://openeo.dataspace.copernicus.eu/
Or use the openeo API to fetch the resulting map.
"""

import openeo

###### USER INPUTS ######
# Define the spatial and temporal extent
spatial_extent = {
    "west": 3.809252,
    "south": 51.232365,
    "east": 3.833542,
    "north": 51.245477,
    "crs": "EPSG:4326",
    "srs": "EPSG:4326",
}

# Temporal extent needs to consist of exactly twelve months,
# always starting first day of the month and ending last day of the month.
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
    namespace="https://raw.githubusercontent.com/WorldCereal/worldcereal-classification/refs/tags/worldcereal_crop_extent_v1.0.2/src/worldcereal/udp/worldcereal_crop_extent.json",
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

###### EXPLANATION OF RESULTS ######
# Two GeoTiff images will be generated:
# 1. "worldcereal-cropland-extent.tif": The original results of the WorldCereal cropland model
# 2. "worldcereal-cropland-extent-postprocessed.tif": The results after applying the default post-processing,
#   which includes a smoothing of the probabilities and relabelling of pixels according to the smoothed probabilities.

# Each tif file contains 4 bands, all at 10m resolution:
# - Band 1: Predicted cropland labels (0: non-cropland, 1: cropland)
# - Band 2: Probability of the winning class (50 - 100)
# - Band 3: Probability of the non-cropland class (0 - 100)
# - Band 4: Probability of the cropland class (0 - 100)
