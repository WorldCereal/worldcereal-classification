import pytest
from openeo_gfmap.temporal import TemporalContext

from worldcereal.openeo.preprocessing import (
    InvalidTemporalContextError,
    _validate_temporal_context,
    correct_temporal_context,
)


def test_temporal_context_validation():
    """Test the validation of temporal context."""

    temporal_context = TemporalContext("2020-01-01", "2022-03-31")
    _validate_temporal_context(temporal_context)

    incorrect_temporal_context = TemporalContext("2022-01-05", "2020-03-15")

    with pytest.raises(InvalidTemporalContextError):
        _validate_temporal_context(incorrect_temporal_context)


def test_temporal_context_correction():
    """Test the automatic correction of invalid temporal context."""

    incorrect_temporal_context = TemporalContext("2022-01-05", "2020-03-15")
    corrected_temporal_context = correct_temporal_context(incorrect_temporal_context)

    # Should no longer raise an exception
    _validate_temporal_context(corrected_temporal_context)
