import json
import os
from pathlib import Path

import pytest
from openeo_gfmap import BoundingBoxExtent, FetchType
from openeo_gfmap.backend import Backend, BackendContext, cdse_connection
from openeo_gfmap.temporal import TemporalContext

from worldcereal.extract.patch_to_point_worldcereal import (
    worldcereal_preprocessed_inputs_from_patches,
)
from worldcereal.openeo.preprocessing import (
    InvalidTemporalContextError,
    _validate_temporal_context,
    correct_temporal_context,
    worldcereal_preprocessed_inputs,
)

basedir = Path(os.path.dirname(os.path.realpath(__file__)))


def test_temporal_context_validation():
    """Test the validation of temporal context."""

    temporal_context = TemporalContext("2020-01-01", "2020-12-31")
    _validate_temporal_context(temporal_context)

    incorrect_temporal_context = TemporalContext("2020-01-05", "2020-03-15")

    with pytest.raises(InvalidTemporalContextError):
        _validate_temporal_context(incorrect_temporal_context)

    more_than_one_year = TemporalContext("2019-01-05", "2021-03-15")

    with pytest.raises(InvalidTemporalContextError):
        _validate_temporal_context(more_than_one_year)


def test_temporal_context_correction():
    """Test the automatic correction of invalid temporal context."""

    incorrect_temporal_context = TemporalContext("2022-01-05", "2020-03-15")
    corrected_temporal_context = correct_temporal_context(incorrect_temporal_context)

    # Should no longer raise an exception
    _validate_temporal_context(corrected_temporal_context)


def test_worldcereal_preprocessed_inputs_graph(SpatialExtent):
    """Test the worldcereal_preprocessed_inputs function.
    This is based on constructing the openEO graph for the job
    that would run, without actually running it."""

    temporal_extent = TemporalContext("2020-06-01", "2021-05-31")

    cube = worldcereal_preprocessed_inputs(
        connection=cdse_connection(),
        backend_context=BackendContext(Backend.CDSE),
        spatial_extent=SpatialExtent,
        temporal_extent=temporal_extent,
        fetch_type=FetchType.POLYGON,
    )

    # Ref file with processing graph
    ref_graph = basedir / "testresources" / "preprocess_graph.json"

    # # uncomment to save current graph to the ref file
    # with open(ref_graph, "w") as f:
    #     f.write(json.dumps(cube.flat_graph(), indent=4))

    with open(ref_graph, "r") as f:
        expected = json.load(f)
        assert expected == cube.flat_graph()


def test_worldcereal_preprocessed_inputs_graph_withslope():
    """This version has fetchtype.TILE and should include slope."""

    temporal_extent = TemporalContext("2018-03-01", "2019-02-28")

    cube = worldcereal_preprocessed_inputs(
        connection=cdse_connection(),
        backend_context=BackendContext(Backend.CDSE),
        spatial_extent=BoundingBoxExtent(
            west=44.432274, south=51.317362, east=44.698802, north=51.428224, epsg=4326
        ),
        temporal_extent=temporal_extent,
    )

    # Ref file with processing graph
    ref_graph = basedir / "testresources" / "preprocess_graphwithslope.json"

    # # uncomment to save current graph to the ref file
    # with open(ref_graph, "w") as f:
    #     f.write(json.dumps(cube.flat_graph(), indent=4))

    with open(ref_graph, "r") as f:
        expected = json.load(f)
        assert expected == cube.flat_graph()


def test_worldcereal_preprocessed_inputs_from_patches_graph():
    """This version gets a preprocessed cube from extracted patches."""

    temporal_extent = TemporalContext("2020-01-01", "2020-12-31")

    cube = worldcereal_preprocessed_inputs_from_patches(
        connection=cdse_connection(),
        temporal_extent=temporal_extent,
        ref_id="test_ref_id",
        epsg=32631,
    )

    # Ref file with processing graph
    ref_graph = basedir / "testresources" / "preprocess_from_patches_graph.json"

    # # uncomment to save current graph to the ref file
    # with open(ref_graph, "w") as f:
    #     f.write(json.dumps(cube.flat_graph(), indent=4))

    with open(ref_graph, "r") as f:
        expected = json.load(f)
        assert expected == cube.flat_graph()
