from unittest.mock import patch

import geopandas as gpd
import pytest
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from shapely import Point

from worldcereal.rdm_api.rdm_interaction import RdmCollection, RdmInteraction


@pytest.fixture
def sample_bbox():
    return BoundingBoxExtent(west=0, east=1, north=1, south=0)


@pytest.fixture
def sample_temporal_extent():
    return TemporalContext(start_date="2021-01-01", end_date="2021-12-31")


class TestRdmInteraction:
    @patch("requests.Session.get")
    def test_collections_from_rdm(
        self, mock_requests_get, sample_bbox, sample_temporal_extent
    ):

        mock_requests_get.return_value.status_code = 200
        mock_requests_get.return_value.json.return_value = [
            {
                "collectionId": "Foo",
                "title": "Foo_title",
                "accessType": "Public",
            },
            {
                "collectionId": "Bar",
                "title": "Bar_title",
                "accessType": "Public",
            },
        ]
        interaction = RdmInteraction()
        collections = interaction.get_collections(
            bbox=sample_bbox, temporal_extent=sample_temporal_extent
        )
        ref_ids = [collection.id for collection in collections]

        assert ref_ids == ["Foo", "Bar"]

        bbox_str = f"Bbox={sample_bbox.west}&Bbox={sample_bbox.south}&Bbox={sample_bbox.east}&Bbox={sample_bbox.north}"
        temporal = f"&ValidityTime.Start={sample_temporal_extent.start_date}T00%3A00%3A00Z&ValidityTime.End={sample_temporal_extent.end_date}T00%3A00%3A00Z"
        expected_url = (
            f"{interaction.RDM_ENDPOINT}/collections/search?{bbox_str}{temporal}"
        )
        mock_requests_get.assert_called_with(
            url=expected_url, headers={"accept": "*/*"}, timeout=10
        )

    @patch("worldcereal.rdm_api.rdm_interaction.RdmInteraction.get_collections")
    @patch("worldcereal.rdm_api.rdm_interaction.RdmInteraction._get_download_urls")
    def test_download_samples(
        self,
        mock_get_download_urls,
        mock_collections_from_rdm,
        sample_bbox,
        sample_temporal_extent,
        tmp_path,
    ):

        data = {
            "col1": ["must", "include", "this", "column", "definitely", "check"],
            "col2": ["and", "this", "One", "Too", "please", "check"],
            "col3": ["but", "not", "This", "One", "please", "check"],
            "valid_time": [
                "2021-01-01",
                "2021-12-31",
                "2021-06-01",
                "2025-05-22",
                "2021-06-01",
                "2021-06-01",
            ],  # Fourth date not within sample_temporal_extent
            "ewoc_code": ["1", "2", "3", "4", "5", "1"],
            # Fifth crop code not within list of ewoc_codes
            "extract": [1, 1, 1, 1, 2, 0],
            "geometry": [
                Point(0.5, 0.5),
                Point(0.25, 0.25),
                Point(2, 3),
                Point(0.75, 0.75),
                Point(0.75, 0.78),
                Point(0.78, 0.75),
            ],  # Third point not within sample_polygon
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        file_path = tmp_path / "sample.parquet"
        gdf.to_parquet(file_path)

        mock_collections_from_rdm.return_value = [
            RdmCollection(
                **{
                    "collectionId": "Foo",
                    "title": "Foo_title",
                    "accessType": "Public",
                }
            ),
        ]
        mock_get_download_urls.return_value = [str(file_path)]

        interaction = RdmInteraction()
        result_gdf = interaction.get_samples(
            bbox=sample_bbox,
            temporal_extent=sample_temporal_extent,
            columns=["col1", "col2", "ref_id", "geometry"],
            ewoc_codes=["1", "2", "3", "4"],
            subset=True,
        )

        # Check that col3 and valid_time indeed not included
        assert result_gdf.columns.tolist() == [
            "col1",
            "col2",
            "ref_id",
            "geometry",
        ]

        # Check that the third up till last geometry are not included
        # third and fourth are outside the spatiotemporal extent
        # fifth has a crop type not in the list of ewoc_codes
        # last sample is not to be extracted
        assert len(result_gdf) == 2
