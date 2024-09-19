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


@patch("requests.get")
def test_collections_from_rdm(mock_requests_get, sample_polygon):

    mock_requests_get.return_value.json.return_value = [
        {"collectionId": "Foo"},
        {"collectionId": "Bar"},
    ]

    collection_ids = _collections_from_rdm(sample_polygon)

    assert collection_ids == ["Foo", "Bar"]

    bbox = sample_polygon.bounds
    expected_url = f"{RDM_ENDPOINT}/collections/search?Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}"
    mock_requests_get.assert_called_with(expected_url)


@patch("worldcereal.rdm_api.rdm_interaction._get_download_urls")
def test_query_ground_truth(mock_get_download_urls, sample_polygon, tmp_path):

    data = {
        "col1": ["include", "this", "column"],
        "col2": ["This", "One", "Too"],
        "col3": ["Not", "This", "One"],
        "geometry": [Point(0.5, 0.5), Point(0.25, 0.25), Point(2, 3)],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    file_path = tmp_path / "sample.parquet"
    gdf.to_parquet(file_path)

    mock_get_download_urls.return_value = [file_path]

    query_ground_truth(
        poly=sample_polygon,
        output_path=tmp_path / "output.parquet",
        columns=["col1", "col2"],
    )

    result_gdf = gpd.read_parquet(tmp_path / "output.parquet")

    # Check that col3 indeed was not included
    assert result_gdf.columns.tolist() == ["col1", "col2", "geometry"]

    # Check that the last geometry was not included, since it's outside the sample_polygon
    assert len(result_gdf) == 2
