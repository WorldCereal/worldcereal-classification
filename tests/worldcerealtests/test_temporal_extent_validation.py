#!/usr/bin/env python3
"""
Test script to validate the temporal extent checking functionality.
This creates a mock xarray DataArray with time coordinates and tests the validation.
"""

import numpy as np
import pytest
import xarray as xr

from worldcereal.openeo.feature_extractor import select_timestep_from_temporal_features


def create_mock_temporal_features():
    """Create a mock xarray DataArray with temporal features for testing."""

    # Create time coordinates for 12 monthly timesteps in 2024
    time_coords = [np.datetime64(f"2024-{month:02d}-01") for month in range(1, 13)]

    # Create mock spatial coordinates
    x_coords = np.array([0, 1, 2])
    y_coords = np.array([0, 1])

    # Create mock feature data (2 spatial x 12 temporal x 5 feature bands)
    data = np.random.rand(len(x_coords), len(y_coords), len(time_coords), 5)

    # Create the DataArray
    features = xr.DataArray(
        data,
        dims=["x", "y", "t", "bands"],
        coords={
            "x": x_coords,
            "y": y_coords,
            "t": time_coords,
            "bands": [f"feature_{i}" for i in range(5)],
        },
    )

    return features


def test_valid_target_dates():
    """Test that valid target dates work correctly."""
    print("=== Testing Valid Target Dates ===")

    features = create_mock_temporal_features()
    print(
        f"Features temporal extent: {features.t.min().values} to {features.t.max().values}"
    )

    # Test 1: Valid date within range
    result = select_timestep_from_temporal_features(features, "2024-06-01")
    assert result is not None, "Should return a valid result for date within range"
    assert "t" not in result.dims, "Time dimension should be removed after selection"
    print("✓ Valid target date 2024-06-01 worked correctly")

    # Test 2: Middle timestep (no target date)
    result = select_timestep_from_temporal_features(features, None)
    assert result is not None, "Should return a valid result for middle timestep"
    assert "t" not in result.dims, "Time dimension should be removed after selection"
    print("✓ Middle timestep selection worked correctly")


def test_invalid_target_dates():
    """Test that invalid target dates raise appropriate errors."""
    print("\n=== Testing Invalid Target Dates ===")

    features = create_mock_temporal_features()

    # Test 1: Date before temporal extent
    with pytest.raises(ValueError, match="outside the temporal extent"):
        select_timestep_from_temporal_features(features, "2023-06-01")
    print("✓ Correctly raised error for date before extent")

    # Test 2: Date after temporal extent
    with pytest.raises(ValueError, match="outside the temporal extent"):
        select_timestep_from_temporal_features(features, "2025-06-01")
    print("✓ Correctly raised error for date after extent")

    # Test 3: Invalid date format (should be caught by numpy.datetime64)
    with pytest.raises(ValueError):
        select_timestep_from_temporal_features(features, "invalid-date")
    print("✓ Correctly raised error for invalid date format")


def test_edge_cases():
    """Test edge cases for temporal extent validation."""
    print("\n=== Testing Edge Cases ===")

    features = create_mock_temporal_features()

    # Test 1: Exact start date
    result = select_timestep_from_temporal_features(features, "2024-01-01")
    assert result is not None, "Should work for exact start date"
    assert "t" not in result.dims, "Time dimension should be removed"
    print("✓ Exact start date worked correctly")

    # Test 2: Exact end date
    result = select_timestep_from_temporal_features(features, "2024-12-01")
    assert result is not None, "Should work for exact end date"
    assert "t" not in result.dims, "Time dimension should be removed"
    print("✓ Exact end date worked correctly")

    # Test 3: Date very close to boundary (one day before start)
    with pytest.raises(ValueError, match="outside the temporal extent"):
        select_timestep_from_temporal_features(features, "2023-12-31")
    print("✓ Correctly raised error for date just before extent")


def run_all_tests():
    """Run all tests when executed directly (not via pytest)."""
    print("Testing Temporal Extent Validation")
    print("=" * 50)

    try:
        test_valid_target_dates()
        test_invalid_target_dates()
        test_edge_cases()
        print("\n" + "=" * 50)
        print("All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
