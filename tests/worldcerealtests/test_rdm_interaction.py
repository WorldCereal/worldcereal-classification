from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely import Point, Polygon

from worldcereal.rdm_api.rdm_interaction import (
    RDM_ENDPOINT,
    _collections_from_rdm,
    query_ground_truth,
)


@pytest.fixture
def sample_polygon():
    return Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])


@pytest.fixture
def sample_temporal_extent():
    return ["2021-01-01", "2021-12-31"]


@patch("requests.get")
def test_collections_from_rdm(
    mock_requests_get, sample_polygon, sample_temporal_extent
):

    mock_requests_get.return_value.json.return_value = [
        {"collectionId": "Foo"},
        {"collectionId": "Bar"},
    ]

    collection_ids = _collections_from_rdm(
        geometry=sample_polygon, temporal_extent=sample_temporal_extent
    )

    assert collection_ids == ["Foo", "Bar"]

    bbox = sample_polygon.bounds
    geom = f"Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}"
    temporal = f"&ValidityTime.Start={sample_temporal_extent[0]}T00%3A00%3A00Z&ValidityTime.End={sample_temporal_extent[1]}T00%3A00%3A00Z"
    expected_url = f"{RDM_ENDPOINT}/collections/search?{geom}{temporal}"
    mock_requests_get.assert_called_with(expected_url)


@patch("worldcereal.rdm_api.rdm_interaction._get_download_urls")
def test_query_ground_truth(
    mock_get_download_urls, sample_polygon, sample_temporal_extent, tmp_path
):

    data = {
        "col1": ["must", "include", "this", "column"],
        "col2": ["and", "this", "One", "Too"],
        "col3": ["but", "not", "This", "One"],
        "valid_time": [
            "2021-01-01",
            "2021-12-31",
            "2021-06-01",
            "2025-05-22",
        ],  # Last date not within sample_temporal_extent
        "geometry": [
            Point(0.5, 0.5),
            Point(0.25, 0.25),
            Point(2, 3),
            Point(0.75, 0.75),
        ],  # Third point not within sample_polygon
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    file_path = tmp_path / "sample.parquet"
    gdf.to_parquet(file_path)

    mock_get_download_urls.return_value = [file_path]

    query_ground_truth(
        geometry=sample_polygon,
        output_path=tmp_path / "output.parquet",
        temporal_extent=sample_temporal_extent,
        columns=["col1", "col2"],
    )

    result_gdf = gpd.read_parquet(tmp_path / "output.parquet")

    # Check that col3 and valid_time indeed not included
    assert result_gdf.columns.tolist() == ["col1", "col2", "geometry"]

    # Check that the third and fourth geometry are not included, as they are outside the spatiotemporal extent
    assert len(result_gdf) == 2
