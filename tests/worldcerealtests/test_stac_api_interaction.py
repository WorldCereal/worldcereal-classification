from unittest.mock import MagicMock, patch

import pystac
import pytest
from requests.auth import AuthBase

from worldcereal.stac.stac_api_interaction import StacApiInteraction


@pytest.fixture
def mock_auth():
    return MagicMock(spec=AuthBase)


def mock_stac_item(item_id):
    item = MagicMock(spec=pystac.Item)
    item.id = item_id
    item.to_dict.return_value = {"id": item_id, "some_property": "value"}
    return item


class TestStacApiInteraction:
    @patch("requests.post")
    @patch("worldcereal.stac.stac_api_interaction.StacApiInteraction.exists")
    def test_upload_items_single_chunk(
        self, mock_exists, mock_requests_post, mock_auth
    ):
        """Test bulk upload of STAC items in one single chunk."""

        mock_requests_post.return_value.status_code = 200
        mock_requests_post.return_value.json.return_value = {"status": "success"}
        mock_requests_post.reason = "OK"

        mock_exists.return_value = True

        items = [mock_stac_item(f"item-{i}") for i in range(10)]

        interaction = StacApiInteraction(
            sensor="Sentinel1",
            base_url="http://fake-stac-api",
            auth=mock_auth,
            bulk_size=10,  # To ensure all 10 items are uploaded in one bulk
        )
        interaction.upload_items_bulk(items)

        mock_requests_post.assert_called_with(
            url=f"http://fake-stac-api/collections/{interaction.collection_id}/bulk_items",
            auth=mock_auth,
            json={
                "method": "upsert",
                "items": {item.id: item.to_dict() for item in items},
            },
        )
        assert mock_requests_post.call_count == 1

    @patch("requests.post")
    @patch("worldcereal.stac.stac_api_interaction.StacApiInteraction.exists")
    def test_upload_items_multiple_chunk(
        self, mock_exists, mock_requests_post, mock_auth
    ):
        """Test bulk upload of STAC items in mulitiple chunks."""

        mock_requests_post.return_value.status_code = 200
        mock_requests_post.return_value.json.return_value = {"status": "success"}
        mock_requests_post.reason = "OK"

        mock_exists.return_value = True

        items = [mock_stac_item(f"item-{i}") for i in range(10)]

        interaction = StacApiInteraction(
            sensor="Sentinel1",
            base_url="http://fake-stac-api",
            auth=mock_auth,
            bulk_size=3,  # This would require 4 chunk for 10 items
        )
        interaction.upload_items_bulk(items)

        assert mock_requests_post.call_count == 4

        expected_calls = [
            {
                "url": f"http://fake-stac-api/collections/{interaction.collection_id}/bulk_items",
                "auth": mock_auth,
                "json": {
                    "method": "upsert",
                    "items": {
                        "item-0": {"id": "item-0", "some_property": "value"},
                        "item-1": {"id": "item-1", "some_property": "value"},
                        "item-2": {"id": "item-2", "some_property": "value"},
                    },
                },
            },
            {
                "url": f"http://fake-stac-api/collections/{interaction.collection_id}/bulk_items",
                "auth": mock_auth,
                "json": {
                    "method": "upsert",
                    "items": {
                        "item-3": {"id": "item-3", "some_property": "value"},
                        "item-4": {"id": "item-4", "some_property": "value"},
                        "item-5": {"id": "item-5", "some_property": "value"},
                    },
                },
            },
            {
                "url": f"http://fake-stac-api/collections/{interaction.collection_id}/bulk_items",
                "auth": mock_auth,
                "json": {
                    "method": "upsert",
                    "items": {
                        "item-6": {"id": "item-6", "some_property": "value"},
                        "item-7": {"id": "item-7", "some_property": "value"},
                        "item-8": {"id": "item-8", "some_property": "value"},
                    },
                },
            },
            {
                "url": f"http://fake-stac-api/collections/{interaction.collection_id}/bulk_items",
                "auth": mock_auth,
                "json": {
                    "method": "upsert",
                    "items": {
                        "item-9": {"id": "item-9", "some_property": "value"},
                    },
                },
            },
        ]

        for i, call in enumerate(mock_requests_post.call_args_list):
            assert call[1] == expected_calls[i]
