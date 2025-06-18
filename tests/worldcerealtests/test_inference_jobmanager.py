from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from openeo import Connection
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from shapely.geometry import box

from worldcereal.job import (
    INFERENCE_JOB_OPTIONS,
    CropLandParameters,
    CropTypeParameters,
    WorldCerealProductType,
    create_inference_job,
    run_largescale_inference,
)


def test_run_largescale_inference_with_geodataframe():
    """
    Dummy test for the run_largescale_inference method using a GeoDataFrame.
    Ensures the method can be called without submitting jobs to the OpenEO backend.
    """
    # Create a dummy GeoDataFrame
    data = {
        "start_date": ["2023-01-01"],
        "end_date": ["2023-12-31"],
        "geometry": [box(0, 0, 1, 1)],
        "tile_name": ["tile_1"],
    }
    production_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    output_dir = Path(".")
    product_type = WorldCerealProductType.CROPLAND
    cropland_parameters = CropLandParameters()

    # Mock dependencies to avoid actual backend calls
    with patch("worldcereal.job.InferenceJobManager") as mock_job_manager_class:
        # Mock the job manager
        mock_job_manager = MagicMock()
        mock_job_manager_class.return_value = mock_job_manager

        # Call the method
        run_largescale_inference(
            production_grid=production_gdf,
            output_dir=output_dir,
            product_type=product_type,
            cropland_parameters=cropland_parameters,
            parallel_jobs=1,
        )

        # Assertions to ensure the method was called correctly
        mock_job_manager.run_jobs.assert_called_once()


def test_create_inference_job_logic():
    """
    Test the logic of the create_inference_job function without submitting jobs to the OpenEO backend.
    """
    # Create a dummy row with required fields
    row = pd.Series(
        {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "geometry": box(0, 0, 1, 1),
            "tile_name": "tile_1",
        }
    )

    # Mock the OpenEO connection
    mock_connection = MagicMock(spec=Connection)

    # Mock the process graph creation
    with patch("worldcereal.job.create_inference_process_graph") as mock_create_graph:
        mock_data_cube = MagicMock()
        mock_create_graph.return_value = mock_data_cube

        # Call the function
        create_inference_job(
            row=row,
            connection=mock_connection,
            provider="dummy_provider",
            connection_provider="dummy_connection_provider",
            epsg=4326,
            product_type=WorldCerealProductType.CROPTYPE,
            cropland_parameters=CropLandParameters(),
            croptype_parameters=CropTypeParameters(),
            postprocess_parameters=None,
            s1_orbit_state=None,
            target_epsg=None,
            job_options=None,
        )

        # Assertions to ensure the logic is executed correctly
        mock_create_graph.assert_called_once_with(
            spatial_extent=BoundingBoxExtent(*(0, 0, 1, 1), epsg=4326),
            temporal_extent=TemporalContext("2023-01-01", "2023-12-31"),
            product_type=WorldCerealProductType.CROPTYPE,
            croptype_parameters=CropTypeParameters(),
            cropland_parameters=CropLandParameters(),
            postprocess_parameters=None,
            s1_orbit_state=None,
            target_epsg=None,
        )
        mock_data_cube.create_job.assert_called_once_with(
            title="WorldCereal [croptype] job",
            description="Job that performs end-to-end WorldCereal inference",
            job_options=INFERENCE_JOB_OPTIONS,
        )
