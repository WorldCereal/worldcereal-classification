import logging
from typing import List, Optional

import geojson
import geopandas as gpd
import openeo
import pandas as pd
from extract_sar import (
    _buffer_geometry,
    _filter_extract_true,
    _upload_geoparquet_artifactory,
)
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.fetching.s2 import build_sentinel2_l2a_extractor
from openeo_gfmap.manager import _log

# Logger for this current pipeline
_pipeline_log: Optional[logging.Logger] = None


def _setup_logger(level=logging.INFO) -> None:
    global _pipeline_log
    """Setup the logger from the openeo_gfmap package to the assigned level."""
    _pipeline_log = logging.getLogger("pipeline_sar")

    _pipeline_log.setLevel(level)
    _log.setLevel(level)

    stream_handler = logging.StreamHandler()
    _log.addHandler(stream_handler)
    _pipeline_log.addHandler(stream_handler)

    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s:  %(message)s")
    stream_handler.setFormatter(formatter)

    # Exclude the other loggers from other libraries
    class ManagerLoggerFilter(logging.Filter):
        """Filter to only accept the OpenEO-GFMAP manager logs."""

        def filter(self, record):
            return record.name in [_log.name, _pipeline_log.name]

    stream_handler.addFilter(ManagerLoggerFilter())


def get_job_nb_polygons(row: pd.Series) -> int:
    """Get the number of polygons in the geometry."""
    return len(
        list(
            filter(
                lambda feat: feat.properties.get("extract"),
                geojson.loads(row.geometry)["features"],
            )
        )
    )


def create_job_dataframe_s2(
    backend: Backend, split_jobs: List[gpd.GeoDataFrame]
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    columns = [
        "backend_name",
        "out_prefix",
        "out_extension",
        "start_date",
        "end_date",
        "s2_tile",
        "h3index",
        "geometry",
    ]
    rows = []
    for job in split_jobs:
        # Compute the average in the valid date and make a buffer of 1.5 year around
        median_time = pd.to_datetime(job.valid_date).mean()
        start_date = median_time - pd.Timedelta(days=275)  # A bit more than 9 months
        end_date = median_time + pd.Timedelta(days=275)  # A bit more than 9 months
        s2_tile = job.tile.iloc[0]  # Job dataframes are split depending on the
        h3index = job.h3index.iloc[0]

        rows.append(
            pd.Series(
                dict(
                    zip(
                        columns,
                        [
                            backend.value,
                            "S2-L2A-10m",
                            ".nc",
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                            s2_tile,
                            h3index,
                            job.to_json(),
                        ],
                    )
                )
            )
        )

    return pd.DataFrame(rows)


def create_datacube_optical(
    row: pd.Series,
    connection: openeo.DataCube,
    provider=None,
    connection_provider=None,
    executor_memory: str = "5G",
    executor_memory_overhead: str = "2G",
) -> gpd.GeoDataFrame:
    start_date = row.start_date
    end_date = row.end_date
    temporal_context = TemporalContext(start_date, end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)

    # Filter the geometry to the rows with the extract only flag
    geometry = _filter_extract_true(geometry)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Performs a buffer of 64 px around the geometry
    geometry_df = _buffer_geometry(geometry, 320)
    spatial_extent_url = _upload_geoparquet_artifactory(geometry_df, row.name)

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    fetch_type = FetchType.POLYGON
    bands_to_download = [
        "S2-L2A-B01",
        "S2-L2A-B02",
        "S2-L2A-B03",
        "S2-L2A-B04",
        "S2-L2A-B05",
        "S2-L2A-B06",
        "S2-L2A-B07",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B09",
        "S2-L2A-B11",
        "S2-L2A-B12",
        "S2-L2A-SCL",
    ]

    # Create the job to extract S2
    extraction_parameters = {
        "target_resolution": 10,
        "load_collection": {
            "eo:cloud_cover": lambda val: val <= 95.0,
            "tileId": lambda val: val == row.s2_tile,
        },
    }
    extractor = build_sentinel2_l2a_extractor(
        backend_context,
        bands=bands_to_download,
        fetch_type=fetch_type.POLYGON,
        **extraction_parameters,
    )

    cube = extractor.get_cube(connection, spatial_extent_url, temporal_context)

    # Compute the SCL dilation and add it to the cube
    scl_dilated_mask = cube.process(
        "to_scl_dilation_mask",
        data=cube,
        scl_band_name="S2-L2A-SCL",
        kernel1_size=17,  # 17px dilation on a 20m layer
        kernel2_size=77,  # 77px dilation on a 20m layer
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3,
    ).rename_labels("bands", ["S2-L2A-SCL_DILATED_MASK"])

    cube = cube.merge_cubes(scl_dilated_mask)
    cube = cube.linear_scale_range(0, 65534, 0, 65534)

    # Get the h3index to use in the tile
    h3index = geometry.features[0].properties["h3index"]
    valid_date = geometry.features[0].properties["valid_date"]

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_polygons = get_job_nb_polygons(row)
    _log.debug(f"Number of polygons to extract: {number_polygons}")

    job_options = {
        "executor-memory": "5G",
        "executor-memoryOverhead": "2G",
    }

    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_S2_{h3index}_{valid_date}",
        sample_by_feature=True,
        job_options=job_options,
    )
