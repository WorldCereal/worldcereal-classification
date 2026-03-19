from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from openeo import Connection, DataCube
from openeo.internal.graph_building import PGNode
from openeo.metadata import CollectionMetadata
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from shapely import wkt as shapely_wkt
from shapely.geometry import box

from worldcereal.job import (
    DEFAULT_INFERENCE_JOB_OPTIONS,
    DEFAULT_INPUTS_JOB_OPTIONS,
    DEFAULT_SEASONAL_WORKFLOW_PRESET,
    WorldCerealProductType,
    WorldCerealTask,
    create_inference_process_graph,
)
from worldcereal.jobmanager import WorldCerealJobManager
from worldcereal.openeo.workflow_config import WorldCerealWorkflowConfigBuilder
from worldcereal.parameters import EmbeddingsParameters


def _dummy_input_cube() -> DataCube:
    return DataCube(PGNode("load_collection", arguments={"id": "dummy-input"}))


def _metadata_for_bands(bands: list[str]) -> CollectionMetadata:
    return CollectionMetadata(
        metadata={
            "cube:dimensions": {
                "bands": {
                    "type": "bands",
                    "values": bands,
                },
                "t": {
                    "type": "temporal",
                    "extent": ["2023-01-01", "2023-12-31"],
                },
                "x": {
                    "type": "spatial",
                    "axis": "x",
                    "extent": [0, 1],
                },
                "y": {
                    "type": "spatial",
                    "axis": "y",
                    "extent": [0, 1],
                },
            }
        }
    )


def _nodes_with_process_id(cube: DataCube, process_id: str) -> list[dict]:
    graph = cube.flat_graph()
    return [node for node in graph.values() if node.get("process_id") == process_id]


def _single_filter_bands_arg(cube: DataCube) -> list[str]:
    filter_nodes = _nodes_with_process_id(cube, "filter_bands")
    assert len(filter_nodes) == 1
    return list(filter_nodes[0]["arguments"]["bands"])


def _single_rename_target(cube: DataCube) -> list[str]:
    rename_nodes = _nodes_with_process_id(cube, "rename_labels")
    assert len(rename_nodes) == 1
    return list(rename_nodes[0]["arguments"]["target"])


def test_run_largescale_inference_with_geodataframe(tmp_path: Path):
    """
    Dummy test for the run_largescale_inference method using a GeoDataFrame.
    Ensures the method can be called without submitting jobs to the OpenEO backend.
    """
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

    output_dir = tmp_path
    mock_job_db = MagicMock()
    mock_job_db.read.return_value = pd.DataFrame(
        [{"status": "not_started", "tile_name": "tile_1"}]
    )

    with patch.object(
        WorldCerealJobManager, "_create_job_database", return_value=mock_job_db
    ):
        manager = WorldCerealJobManager(
            output_dir=output_dir,
            task=WorldCerealTask.CLASSIFICATION,
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


def test_run_inputs_jobs_with_geodataframe(tmp_path: Path):
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

    output_dir = tmp_path
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


def test_run_embeddings_jobs_with_geodataframe(tmp_path: Path):
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

    output_dir = tmp_path
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


def test_create_inference_job_logic(tmp_path: Path):
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
        output_dir=tmp_path,
        task=WorldCerealTask.CLASSIFICATION,
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
        "worldcereal.jobmanager.create_inference_process_graph"
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


def test_create_inference_process_graph_cropland_splits_auxiliary_products():
    spatial_extent = BoundingBoxExtent(0, 0, 1, 1, epsg=4326)
    temporal_extent = TemporalContext("2023-01-01", "2023-12-31")
    mock_connection = MagicMock(spec=Connection)

    udf_bands = [
        "cropland_classification",
        "probability_cropland",
        "probability_other",
        "ndvi:ts_0",
        "global_embedding:dim_0",
        "global_embedding:scale",
    ]
    workflow_config = (
        WorldCerealWorkflowConfigBuilder()
        .season_ids(["s1"])
        .season_windows({"s1": ("2023-01-01", "2023-12-31")})
        .export_embeddings(True)
        .export_ndvi(True)
        .build()
    )

    with (
        patch("worldcereal.job.worldcereal_preprocessed_inputs") as mock_inputs,
        patch("worldcereal.openeo.mapping.apply_metadata") as mock_apply_metadata,
    ):
        mock_inputs.return_value = _dummy_input_cube()
        mock_apply_metadata.return_value = _metadata_for_bands(udf_bands)

        results = create_inference_process_graph(
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            product_type=WorldCerealProductType.CROPLAND,
            connection=mock_connection,
            workflow_config=workflow_config,
        )

    assert len(results) == 4

    main_graph, ndvi_graph, emb_graph, scale_graph = results
    assert len(_nodes_with_process_id(main_graph, "save_result")) == 1
    assert len(_nodes_with_process_id(ndvi_graph, "save_result")) == 1
    assert len(_nodes_with_process_id(emb_graph, "save_result")) == 1
    assert len(_nodes_with_process_id(scale_graph, "save_result")) == 1
    assert len(_nodes_with_process_id(main_graph, "apply_neighborhood")) >= 1

    assert _single_filter_bands_arg(main_graph) == [
        "cropland_classification",
        "probability_cropland",
        "probability_other",
    ]
    assert _single_filter_bands_arg(ndvi_graph) == ["ndvi:ts_0"]
    assert _single_filter_bands_arg(emb_graph) == ["global_embedding:dim_0"]
    assert _single_filter_bands_arg(scale_graph) == ["global_embedding:scale"]


def test_create_inference_process_graph_croptype_has_expected_save_nodes_and_labels():
    spatial_extent = BoundingBoxExtent(0, 0, 1, 1, epsg=4326)
    temporal_extent = TemporalContext("2023-01-01", "2023-12-31")
    mock_connection = MagicMock(spec=Connection)

    udf_bands = [
        "croptype_classification:s1",
        "croptype_probability:s1",
        "croptype_probability:s1:wheat",
        "croptype_classification:s2",
        "croptype_probability:s2",
        "croptype_probability:s2:wheat",
        "cropland_classification",
        "probability_cropland",
        "probability_other",
        "ndvi:ts_0",
        "global_embedding:dim_0",
        "global_embedding:scale",
    ]
    workflow_config = (
        WorldCerealWorkflowConfigBuilder()
        .season_ids(["s1", "s2"])
        .season_windows(
            {
                "s1": ("2023-01-01", "2023-06-30"),
                "s2": ("2023-07-01", "2023-12-31"),
            }
        )
        .export_class_probabilities(True)
        .export_embeddings(True)
        .export_ndvi(True)
        .build()
    )

    with (
        patch("worldcereal.job.worldcereal_preprocessed_inputs") as mock_inputs,
        patch("worldcereal.openeo.mapping.apply_metadata") as mock_apply_metadata,
    ):
        mock_inputs.return_value = _dummy_input_cube()
        mock_apply_metadata.return_value = _metadata_for_bands(udf_bands)

        results = create_inference_process_graph(
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            product_type=WorldCerealProductType.CROPTYPE,
            connection=mock_connection,
            workflow_config=workflow_config,
        )

    assert len(results) == 6
    assert all(
        len(_nodes_with_process_id(cube, "save_result")) == 1 for cube in results
    )
    assert len(_nodes_with_process_id(results[0], "apply_neighborhood")) >= 1

    assert _single_rename_target(results[0]) == [
        "classification",
        "probability",
        "probability_wheat",
    ]
    assert _single_rename_target(results[1]) == [
        "classification",
        "probability",
        "probability_wheat",
    ]

    assert _single_filter_bands_arg(results[2]) == [
        "cropland_classification",
        "probability_cropland",
        "probability_other",
    ]
    assert _single_filter_bands_arg(results[3]) == ["ndvi:ts_0"]
    assert _single_filter_bands_arg(results[4]) == ["global_embedding:dim_0"]
    assert _single_filter_bands_arg(results[5]) == ["global_embedding:scale"]


def test_create_inference_process_graph_croptype_merged_products():
    temporal_extent = TemporalContext("2023-01-01", "2023-12-31")

    udf_bands = [
        "croptype_classification:s1",
        "croptype_probability:s1",
        "croptype_classification:s2",
        "croptype_probability:s2",
        "cropland_classification",
        "probability_cropland",
        "probability_other",
        "ndvi:ts_0",
        "global_embedding:dim_0",
        "global_embedding:scale",
    ]
    workflow_config = (
        WorldCerealWorkflowConfigBuilder()
        .season_ids(["s1", "s2"])
        .export_embeddings(True)
        .export_ndvi(True)
        .merge_classification_products(True)
        .build()
    )
    workflow_context = workflow_config.to_dict()

    with (
        patch("worldcereal.job.worldcereal_preprocessed_inputs") as mock_inputs,
        patch("worldcereal.openeo.mapping.apply_metadata") as mock_apply_metadata,
    ):
        from worldcereal.openeo.mapping import _croptype_map

        mock_inputs.return_value = _dummy_input_cube()
        mock_apply_metadata.return_value = _metadata_for_bands(udf_bands)

        dummy_cube = _dummy_input_cube()
        results = _croptype_map(dummy_cube, temporal_extent, workflow_context)

    # merged: 1 combined classification product + 1 ndvi + 1 embedding dims + 1 scale
    assert len(results) == 4

    # first result is the merged classification product (no per-season filter_bands rename)
    merged_bands = _single_filter_bands_arg(results[0])
    assert "cropland_classification" in merged_bands
    assert "croptype_classification:s1" in merged_bands
    assert "croptype_classification:s2" in merged_bands

    assert _single_filter_bands_arg(results[1]) == ["ndvi:ts_0"]
    assert _single_filter_bands_arg(results[2]) == ["global_embedding:dim_0"]
    assert _single_filter_bands_arg(results[3]) == ["global_embedding:scale"]


def test_create_inputs_job_logic(tmp_path: Path):
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
        output_dir=tmp_path,
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
        "worldcereal.jobmanager.create_inputs_process_graph"
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


def test_create_embeddings_job_logic(tmp_path: Path):
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
        output_dir=tmp_path,
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
        "worldcereal.jobmanager.create_embeddings_process_graph"
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
