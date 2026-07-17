from pathlib import Path

from worldcereal.data.cropcalendars import (
    SEASONALITY_REQUIRED_COLUMNS,
    load_seasonality_lookup,
    seasonality_lookup_file,
)


def test_seasonality_lookup_file_exists():
    with seasonality_lookup_file() as lookup_path:
        assert isinstance(lookup_path, Path)
        assert lookup_path.exists()
        assert lookup_path.name == "seasonality_lookup.parquet"


def test_load_seasonality_lookup_columns():
    table = load_seasonality_lookup()

    assert not table.empty
    assert set(SEASONALITY_REQUIRED_COLUMNS).issubset(table.columns)
