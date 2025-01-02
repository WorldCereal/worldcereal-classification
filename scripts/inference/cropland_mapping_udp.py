"""This short script demonstrates how to run WorldCereal cropland extent inference through
    an OpenEO User-Defined Process (UDP) on CDSE.

    The WorldCereal default cropland model is used and default post-processing is applied
    (using smooth_probabilities method).

    The user needs to manually download the resulting map through the OpenEO Web UI:
    https://openeofed.dataspace.copernicus.eu/
    Or use the openeo API to fetch the resulting map.
    """

import openeo

spatial_extent = {
    "west": 622694.5968575787,
    "south": 5672232.857114074,
    "east": 623079.000934101,
    "north": 5672519.995940826,
    "crs": "EPSG:32631",
    "srs": "EPSG:32631",
}

temporal_extent = ["2020-01-01", "2020-12-31"]

c = openeo.connect("openeofed.dataspace.copernicus.eu").authenticate_oidc()

# This argument is optional and determines the projection of the output products
target_projection = "EPSG:4326"

classes = c.datacube_from_process(
    process_id="worldcereal_crop_extent",
    namespace="https://raw.githubusercontent.com/ESA-APEx/apex_algorithms/refs/heads/main/openeo_udp/worldcereal_crop_extent.json",
    temporal_extent=temporal_extent,
    spatial_extent=spatial_extent,
    projection=target_projection,
)

ONNX_DEPS_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"

JOB_OPTIONS = {
    "driver-memory": "4g",
    "executor-memory": "2g",
    "executor-memoryOverhead": "1g",
    "python-memory": "3g",
    "soft-errors": "true",
    "udf-dependency-archives": [f"{ONNX_DEPS_URL}#onnx_deps"],
}

job = classes.execute_batch(
    title="WorldCereal Crop Extent UDP test",
    out_format="GTiff",
    job_options=JOB_OPTIONS,
)
