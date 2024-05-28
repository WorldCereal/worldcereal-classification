"""Test the presto feature computer running with GFMAP"""
import openeo

from openeo_gfmap import (
    Backend, BackendContext, BoundingBoxExtent, TemporalContext
)
from openeo_gfmap.features.feature_extractor import apply_feature_extractor

from presto_feature_computer import PrestoFeatureExtractor

from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap

EXTENT = dict(zip(["west", "south", "east", "north"], [664000.0, 5611120.0, 665000.0, 5612120.0]))
EXTENT['crs'] = "EPSG:32631"
EXTENT['srs'] = "EPSG:32631"
STARTDATE = '2020-11-01'
ENDDATE = '2021-10-31'


if __name__ == '__main__':
    # Test extent
    spatial_extent = BoundingBoxExtent(
        west=EXTENT['west'],
        south=EXTENT['south'],
        east=EXTENT['east'],
        north=EXTENT['north'],
        epsg=32631,
    )

    temporal_extent = TemporalContext(
        start_date=STARTDATE,
        end_date=ENDDATE,
    )
    backend_context = BackendContext(Backend.FED)

    connection = openeo.connect("openeofed.dataspace.copernicus.eu").authenticate_oidc()

    inputs = worldcereal_preprocessed_inputs_gfmap(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    # Test feature computer
    presto_parameters = {}
    features = apply_feature_extractor(
        feature_extractor_class=PrestoFeatureExtractor,
        cube=inputs,
        parameters=presto_parameters,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ]
    )

    job = features.create_job(out_format="NetCDF", title="Presto FC GFMAP")

    job.start_and_wait()

    for asset in job.get_results().get_assets():
        if asset.metadata["type"].startswith("application/x-netcdf"):
            asset.download("presto_features_gfmap.nc")
