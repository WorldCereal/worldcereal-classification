from unittest.mock import Mock

import pytest

from worldcereal.extract.utils import upload_geoparquet_artifact


@pytest.mark.parametrize(
    ("backend", "connection"),
    [(None, None), ("cdse", Mock())],
)
def test_upload_geoparquet_artifact_requires_one_auth_source(backend, connection):
    with pytest.raises(ValueError, match="exactly one"):
        upload_geoparquet_artifact(
            Mock(), "geometry", backend=backend, connection=connection
        )


def test_upload_geoparquet_artifact_uses_authenticated_connection(monkeypatch):
    artifact_helper = Mock()
    artifact_helper.upload_file.return_value = "s3://bucket/key"
    artifact_helper.get_presigned_url.return_value = "https://example.test/presigned"
    factory = Mock(return_value=artifact_helper)
    monkeypatch.setattr(
        "worldcereal.extract.utils.OpenEOArtifactHelper.from_openeo_connection",
        factory,
    )
    connection = Mock()
    geodataframe = Mock()

    result = upload_geoparquet_artifact(
        geodataframe,
        "samples",
        collection="SENTINEL2",
        connection=connection,
    )

    factory.assert_called_once_with(connection)
    geodataframe.to_parquet.assert_called_once()
    assert artifact_helper.upload_file.call_args.args[0] == (
        "openeogfmap_dataframe_SENTINEL2_samples.parquet"
    )
    artifact_helper.get_presigned_url.assert_called_once_with("s3://bucket/key")
    assert result == "https://example.test/presigned"


def test_upload_geoparquet_artifact_authenticates_with_backend(monkeypatch):
    artifact_helper = Mock()
    artifact_helper.upload_file.return_value = "s3://bucket/key"
    artifact_helper.get_presigned_url.return_value = "https://example.test/presigned"
    factory = Mock(return_value=artifact_helper)
    monkeypatch.setattr(
        "worldcereal.extract.utils.OpenEOArtifactHelper.from_openeo_backend",
        factory,
    )

    result = upload_geoparquet_artifact(Mock(), "samples", backend="cdse")

    factory.assert_called_once_with("cdse")
    assert result == "https://example.test/presigned"
