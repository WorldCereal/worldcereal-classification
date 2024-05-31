"""Test the presto feature computer running with GFMAP"""

import openeo
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.features.feature_extractor import apply_feature_extractor
from presto_feature_computer import PrestoFeatureExtractor

from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap

EXTENT = dict(
    zip(["west", "south", "east", "north"], [664000.0, 5611120.0, 665000.0, 5612120.0])
)
EXTENT["crs"] = "EPSG:32631"
EXTENT["srs"] = "EPSG:32631"
STARTDATE = "2020-11-01"
ENDDATE = "2021-10-31"


if __name__ == "__main__":
    # Test extent
    spatial_extent = BoundingBoxExtent(
        west=EXTENT["west"],
        south=EXTENT["south"],
        east=EXTENT["east"],
        north=EXTENT["north"],
        epsg=32631,
    )

    temporal_extent = TemporalContext(
        start_date=STARTDATE,
        end_date=ENDDATE,
    )
    backend_context = BackendContext(Backend.FED)

    connection = openeo.connect(
        "https://openeo.creo.vito.be/openeo/"
    ).authenticate_oidc()

    inputs = worldcereal_preprocessed_inputs_gfmap(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    # Test feature computer
    presto_parameters = {
        "rescale_s1": False,  # Will be done in the Presto UDF itself!
    }

    features = apply_feature_extractor(
        feature_extractor_class=PrestoFeatureExtractor,
        cube=inputs,
        parameters=presto_parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 100},
            {"dimension": "y", "unit": "px", "value": 100},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    features.execute_batch(
        outputfile=".notebook-tests/presto_features_gfmap_nointerp.nc",
        out_format="NetCDF",
        job_options={"driver-memory": "4g", "executor-memoryOverhead": "8g"},
    )
