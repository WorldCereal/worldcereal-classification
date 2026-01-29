from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from openeo import Connection
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from shapely.geometry import box

from worldcereal.job import (
    DEFAULT_SEASONAL_WORKFLOW_PRESET,
    INFERENCE_JOB_OPTIONS,
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
        "epsg": [4326],
        "bounds_epsg": ["(0, 0, 1, 1)"],
    }
    production_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    output_dir = Path(".")
    product_type = WorldCerealProductType.CROPLAND
    mock_job_manager = MagicMock()
    mock_job_db = MagicMock()
    mock_job_db.df.empty = False
    mock_start_job = MagicMock()

    # Patch setup_inference_job_manager to avoid actual backend calls
    with patch("worldcereal.job.setup_inference_job_manager") as mock_setup:
        mock_setup.return_value = (mock_job_manager, mock_job_db, mock_start_job)

        run_largescale_inference(
            production_grid=production_gdf,
            output_dir=output_dir,
            product_type=product_type,
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
            "epsg": 4326,
            "bounds_epsg": "(0, 0, 1, 1)",
        }
    )

    # Mock the OpenEO connection
    mock_connection = MagicMock(spec=Connection)

    # Dummy inference result returned by process graph creation
    mock_inference_result = MagicMock(name="inference_result")

    # Mock the process graph creation
    with patch("worldcereal.job.create_inference_process_graph") as mock_create_graph:
        mock_create_graph.return_value = mock_inference_result

        # Call the function
        create_inference_job(
            row=row,
            connection=mock_connection,
            provider="dummy_provider",
            connection_provider="dummy_connection_provider",
            product_type=WorldCerealProductType.CROPTYPE,
            s1_orbit_state=None,
            target_epsg=None,
            job_options=None,
        )

        mock_create_graph.assert_called_once()
        _, kwargs = mock_create_graph.call_args
        assert kwargs["spatial_extent"] == BoundingBoxExtent(*(0, 0, 1, 1), epsg=4326)
        assert kwargs["temporal_extent"] == TemporalContext(
            "2023-01-01", "2023-12-31"
        )
        assert kwargs["product_type"] == WorldCerealProductType.CROPTYPE
        assert kwargs["s1_orbit_state"] is None
        assert kwargs["target_epsg"] == 4326
        assert kwargs["connection"] is mock_connection
        assert kwargs["seasonal_preset"] == DEFAULT_SEASONAL_WORKFLOW_PRESET
        assert kwargs["row"].equals(row)

        # first positional arg is the inference result, options via 'additional'
        assert mock_connection.create_job.call_count == 1
        args, kwargs = mock_connection.create_job.call_args
        assert args[0] is mock_inference_result
        assert kwargs["title"] == "WorldCereal [croptype] job_tile_1"
        assert (
            kwargs["description"]
            == "Job that performs end-to-end WorldCereal inference"
        )
        assert "additional" in kwargs
        assert kwargs["additional"] == INFERENCE_JOB_OPTIONS
