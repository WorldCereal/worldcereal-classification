from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely import Point, Polygon

from worldcereal.rdm_api.rdm_interaction import RdmInteraction


@pytest.fixture
def sample_polygon():
    return Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])


@pytest.fixture
def sample_temporal_extent():
    return ["2021-01-01", "2021-12-31"]


class TestRdmInteraction:
    @patch("requests.get")
    def test_collections_from_rdm(
        self, mock_requests_get, sample_polygon, sample_temporal_extent
    ):

        mock_requests_get.return_value.status_code = 200
        mock_requests_get.return_value.json.return_value = [
            {"collectionId": "Foo"},
            {"collectionId": "Bar"},
        ]
        interaction = RdmInteraction()
        collection_ids = interaction._collections_from_rdm(
            geometry=sample_polygon, temporal_extent=sample_temporal_extent
        )

        assert collection_ids == ["Foo", "Bar"]

        bbox = sample_polygon.bounds
        geom = f"Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}"
        temporal = f"&ValidityTime.Start={sample_temporal_extent[0]}T00%3A00%3A00Z&ValidityTime.End={sample_temporal_extent[1]}T00%3A00%3A00Z"
        expected_url = f"{interaction.RDM_ENDPOINT}/collections/search?{geom}{temporal}"
        mock_requests_get.assert_called_with(
            url=expected_url, headers={"accept": "*/*"}
        )

    @patch("worldcereal.rdm_api.rdm_interaction.RdmInteraction._get_download_urls")
    @patch("worldcereal.rdm_api.rdm_interaction.RdmInteraction._collections_from_rdm")
    def test_query_rdm(
        self,
        mock_get_download_urls,
        mock_collections_from_rdm,
        sample_polygon,
        sample_temporal_extent,
        tmp_path,
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

        mock_collections_from_rdm.return_value = [file_path]
        mock_get_download_urls.return_value = [file_path]

        interaction = RdmInteraction()
        result_gdf = interaction.query_rdm(
            geometry=sample_polygon,
            temporal_extent=sample_temporal_extent,
            columns=["col1", "col2"],
        )

        # Check that col3 and valid_time indeed not included
        assert result_gdf.columns.tolist() == ["col1", "col2", "geometry"]

        # Check that the third and fourth geometry are not included, as they are outside the spatiotemporal extent
        assert len(result_gdf) == 2
