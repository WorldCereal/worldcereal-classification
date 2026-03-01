from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from openeo import Connection
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from shapely import wkt as shapely_wkt
from shapely.geometry import box

from worldcereal.job import (
    DEFAULT_INFERENCE_JOB_OPTIONS,
    DEFAULT_INPUTS_JOB_OPTIONS,
    DEFAULT_SEASONAL_WORKFLOW_PRESET,
    WorldCerealProductType,
    WorldCerealTask,
)
from worldcereal.jobmanager import WorldCerealJobManager
from worldcereal.parameters import EmbeddingsParameters


def test_run_inference_jobs_with_geodataframe():
    """Ensure inference jobs are dispatched via the unified job manager."""
    # Create a dummy GeoDataFrame
    utm_geom = box(500000, 0, 501000, 1000)
    data = {
        "start_date": ["2023-01-01"],
        "end_date": ["2023-12-31"],
        "geometry": [box(0, 0, 1, 1)],
        "tile_name": ["tile_1"],
        "geometry_utm_wkt": [shapely_wkt.dumps(utm_geom)],
        "epsg_utm": [32631],
    }
    production_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    output_dir = Path(".")
    mock_job_db = MagicMock()
    mock_job_db.read.return_value = pd.DataFrame(
        [{"status": "not_started", "tile_name": "tile_1"}]
    )

    with patch.object(
        WorldCerealJobManager, "_create_job_database", return_value=mock_job_db
    ):
        manager = WorldCerealJobManager(
            output_dir=output_dir,
            task=WorldCerealTask.INFERENCE,
            backend_context=BackendContext(Backend.CDSE),
            aoi_gdf=production_gdf,
            temporal_extent=TemporalContext("2023-01-01", "2023-12-31"),
        )

    with (
        patch(
            "worldcereal.jobmanager.MultiBackendJobManager.run_jobs"
        ) as mock_run_jobs,
        patch.object(manager, "add_default_backend"),
    ):
        manager.run_jobs(
            parallel_jobs=1,
            product_type=WorldCerealProductType.CROPLAND,
        )

    mock_run_jobs.assert_called_once()


def test_run_inputs_jobs_with_geodataframe():
    """Ensure inputs jobs are dispatched via the unified job manager."""
    utm_geom = box(500000, 0, 501000, 1000)
    data = {
        "start_date": ["2023-01-01"],
        "end_date": ["2023-12-31"],
        "geometry": [box(0, 0, 1, 1)],
        "tile_name": ["tile_1"],
        "geometry_utm_wkt": [shapely_wkt.dumps(utm_geom)],
        "epsg_utm": [32631],
    }
    production_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    output_dir = Path(".")
    mock_job_db = MagicMock()
    mock_job_db.read.return_value = pd.DataFrame(
        [{"status": "not_started", "tile_name": "tile_1"}]
    )

    with patch.object(
        WorldCerealJobManager, "_create_job_database", return_value=mock_job_db
    ):
        manager = WorldCerealJobManager(
            output_dir=output_dir,
            task=WorldCerealTask.INPUTS,
            backend_context=BackendContext(Backend.CDSE),
            aoi_gdf=production_gdf,
            temporal_extent=TemporalContext("2023-01-01", "2023-12-31"),
        )

    with (
        patch(
            "worldcereal.jobmanager.MultiBackendJobManager.run_jobs"
        ) as mock_run_jobs,
        patch.object(manager, "add_default_backend"),
    ):
        manager.run_jobs(
            parallel_jobs=1,
        )

    mock_run_jobs.assert_called_once()


def test_run_embeddings_jobs_with_geodataframe():
    """Ensure embeddings jobs are dispatched via the unified job manager."""
    utm_geom = box(500000, 0, 501000, 1000)
    data = {
        "start_date": ["2023-01-01"],
        "end_date": ["2023-12-31"],
        "geometry": [box(0, 0, 1, 1)],
        "tile_name": ["tile_1"],
        "geometry_utm_wkt": [shapely_wkt.dumps(utm_geom)],
        "epsg_utm": [32631],
    }
    production_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    output_dir = Path(".")
    mock_job_db = MagicMock()
    mock_job_db.read.return_value = pd.DataFrame(
        [{"status": "not_started", "tile_name": "tile_1"}]
    )

    with patch.object(
        WorldCerealJobManager, "_create_job_database", return_value=mock_job_db
    ):
        manager = WorldCerealJobManager(
            output_dir=output_dir,
            task=WorldCerealTask.EMBEDDINGS,
            backend_context=BackendContext(Backend.CDSE),
            aoi_gdf=production_gdf,
            temporal_extent=TemporalContext("2023-01-01", "2023-12-31"),
        )

    with (
        patch(
            "worldcereal.jobmanager.MultiBackendJobManager.run_jobs"
        ) as mock_run_jobs,
        patch.object(manager, "add_default_backend"),
    ):
        manager.run_jobs(
            parallel_jobs=1,
        )

    mock_run_jobs.assert_called_once()


def test_create_inference_job_logic():
    """Test the job manager inference job builder without backend calls."""
    # Create a dummy row with required fields
    utm_geom = box(500000, 0, 501000, 1000)
    row = pd.Series(
        {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "geometry": box(0, 0, 1, 1),
            "tile_name": "tile_1",
            "geometry_utm_wkt": shapely_wkt.dumps(utm_geom),
            "epsg_utm": 32631,
        }
    )

    # Mock the OpenEO connection
    mock_connection = MagicMock(spec=Connection)

    # Dummy inference result returned by process graph creation
    mock_inference_result = MagicMock(name="inference_result")

    # Mock the process graph creation
    manager = WorldCerealJobManager(
        output_dir=Path("."),
        task=WorldCerealTask.INFERENCE,
        backend_context=BackendContext(Backend.CDSE),
        aoi_gdf=gpd.GeoDataFrame(
            {
                "tile_name": ["tile_1"],
                "geometry_utm_wkt": [row.geometry_utm_wkt],
                "epsg_utm": [row.epsg_utm],
            },
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        ),
        temporal_extent=TemporalContext("2023-01-01", "2023-12-31"),
        season_specifications={
            "s1": TemporalContext("2023-01-01", "2023-06-30"),
        },
    )

    with patch(
        "worldcereal.jobmanager.create_worldcereal_process_graph"
    ) as mock_create_graph:
        mock_create_graph.return_value = mock_inference_result

        # Call the function
        manager._create_inference_job(
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
        assert kwargs["spatial_extent"] == BoundingBoxExtent(
            *(500000, 0, 501000, 1000), epsg=32631
        )
        assert kwargs["temporal_extent"] == TemporalContext("2023-01-01", "2023-12-31")
        assert kwargs["product_type"] == WorldCerealProductType.CROPTYPE
        assert kwargs["s1_orbit_state"] is None
        assert kwargs["target_epsg"] == 32631
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
        assert kwargs["additional"] == DEFAULT_INFERENCE_JOB_OPTIONS


def test_create_inputs_job_logic():
    """Test the job manager inputs job builder without backend calls."""
    utm_geom = box(500000, 0, 501000, 1000)
    row = pd.Series(
        {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "geometry": box(0, 0, 1, 1),
            "tile_name": "tile_1",
            "geometry_utm_wkt": shapely_wkt.dumps(utm_geom),
            "epsg_utm": 32631,
        }
    )

    mock_connection = MagicMock(spec=Connection)
    mock_inputs = MagicMock(name="inputs_cube")
    mock_inputs.create_job = MagicMock()

    manager = WorldCerealJobManager(
        output_dir=Path("."),
        task=WorldCerealTask.INPUTS,
        backend_context=BackendContext(Backend.CDSE),
        aoi_gdf=gpd.GeoDataFrame(
            {
                "tile_name": ["tile_1"],
                "geometry_utm_wkt": [row.geometry_utm_wkt],
                "epsg_utm": [row.epsg_utm],
            },
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        ),
        temporal_extent=TemporalContext("2023-01-01", "2023-12-31"),
    )

    with patch(
        "worldcereal.jobmanager.create_worldcereal_process_graph"
    ) as mock_create_graph:
        mock_create_graph.return_value = mock_inputs

        manager._create_inputs_job(
            row=row,
            connection=mock_connection,
            provider="dummy_provider",
            connection_provider="dummy_connection_provider",
        )

        mock_create_graph.assert_called_once()
        _, kwargs = mock_create_graph.call_args
        assert kwargs["spatial_extent"] == BoundingBoxExtent(
            *(500000, 0, 501000, 1000), epsg=32631
        )
        assert kwargs["temporal_extent"] == TemporalContext("2023-01-01", "2023-12-31")
        assert kwargs["s1_orbit_state"] is None
        assert kwargs["target_epsg"] == 32631
        assert kwargs["compositing_window"] == "month"
        assert kwargs["connection"] is mock_connection

        mock_inputs.create_job.assert_called_once()
        _, kwargs = mock_inputs.create_job.call_args
        assert kwargs["title"] == "WorldCereal collect inputs for tile_1"
        assert kwargs["job_options"] == DEFAULT_INPUTS_JOB_OPTIONS


def test_create_embeddings_job_logic():
    """Test the job manager embeddings job builder without backend calls."""
    utm_geom = box(500000, 0, 501000, 1000)
    row = pd.Series(
        {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "geometry": box(0, 0, 1, 1),
            "tile_name": "tile_1",
            "geometry_utm_wkt": shapely_wkt.dumps(utm_geom),
            "epsg_utm": 32631,
        }
    )

    mock_connection = MagicMock(spec=Connection)
    mock_embeddings = MagicMock(name="embeddings_cube")
    mock_embeddings.create_job = MagicMock()

    manager = WorldCerealJobManager(
        output_dir=Path("."),
        task=WorldCerealTask.EMBEDDINGS,
        backend_context=BackendContext(Backend.CDSE),
        aoi_gdf=gpd.GeoDataFrame(
            {
                "tile_name": ["tile_1"],
                "geometry_utm_wkt": [row.geometry_utm_wkt],
                "epsg_utm": [row.epsg_utm],
            },
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        ),
        temporal_extent=TemporalContext("2023-01-01", "2023-12-31"),
    )

    with patch(
        "worldcereal.jobmanager.create_worldcereal_process_graph"
    ) as mock_create_graph:
        mock_create_graph.return_value = mock_embeddings

        manager._create_embeddings_job(
            row=row,
            connection=mock_connection,
            provider="dummy_provider",
            connection_provider="dummy_connection_provider",
        )

        mock_create_graph.assert_called_once()
        _, kwargs = mock_create_graph.call_args
        assert kwargs["spatial_extent"] == BoundingBoxExtent(
            *(500000, 0, 501000, 1000), epsg=32631
        )
        assert kwargs["temporal_extent"] == TemporalContext("2023-01-01", "2023-12-31")
        assert kwargs["s1_orbit_state"] is None
        assert kwargs["target_epsg"] == 32631
        assert isinstance(kwargs["embeddings_parameters"], EmbeddingsParameters)
        assert kwargs["scale_uint16"] is True
        assert kwargs["connection"] is mock_connection

        mock_embeddings.create_job.assert_called_once()
        _, kwargs = mock_embeddings.create_job.call_args
        assert kwargs["title"] == "WorldCereal embeddings for tile_1"
        assert kwargs["job_options"] == DEFAULT_INFERENCE_JOB_OPTIONS
