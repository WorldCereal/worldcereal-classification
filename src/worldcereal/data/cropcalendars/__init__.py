"""Crop calendar resources distributed with worldcereal."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pandas as pd

SEASONALITY_LOOKUP_FILENAME = "seasonality_lookup.parquet"
SEASONALITY_REQUIRED_COLUMNS = (
    "lat",
    "lon",
    "s1_sos_doy",
    "s1_eos_doy",
    "s2_sos_doy",
    "s2_eos_doy",
)
SEASONALITY_LOOKUP_COLUMNS: Tuple[str, ...] = (
    "s1_sos_doy",
    "s1_eos_doy",
    "s2_sos_doy",
    "s2_eos_doy",
    "annual_sos_doy",
    "annual_eos_doy",
    "s1_sos_dekad",
    "s1_eos_dekad",
    "s2_sos_dekad",
    "s2_eos_dekad",
    "annual_sos_dekad",
    "annual_eos_dekad"
)
SEASONALITY_COLUMN_MAP: Dict[str, Tuple[str, str]] = {
    "tc-s1": ("s1_sos_doy", "s1_eos_doy"),
    "tc-s2": ("s2_sos_doy", "s2_eos_doy"),
    "tc-annual": ("annual_sos_doy", "annual_eos_doy"),
}
SEASONALITY_LAT_RANGE = (-89.999, 89.999)
SEASONALITY_LON_RANGE = (-179.999, 179.999)


@contextmanager
def seasonality_lookup_file() -> Iterator[Path]:
    """Yield a filesystem path to the bundled seasonality lookup parquet."""

    local_path = Path(__file__).resolve().parent / SEASONALITY_LOOKUP_FILENAME
    if local_path.exists():
        yield local_path
        return

    resource = resources.files(__package__) / SEASONALITY_LOOKUP_FILENAME
    with resources.as_file(resource) as resource_path:
        yield resource_path


def load_seasonality_lookup() -> pd.DataFrame:
    """Load the bundled seasonality lookup table as a pandas DataFrame."""

    with seasonality_lookup_file() as lookup_path:
        table = pd.read_parquet(lookup_path)

    missing_columns = set(SEASONALITY_REQUIRED_COLUMNS).difference(table.columns)
    if missing_columns:
        raise ValueError(
            "Seasonality lookup parquet is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return table


__all__ = [
    "SEASONALITY_LOOKUP_FILENAME",
    "SEASONALITY_REQUIRED_COLUMNS",
    "SEASONALITY_LOOKUP_COLUMNS",
    "SEASONALITY_COLUMN_MAP",
    "SEASONALITY_LAT_RANGE",
    "SEASONALITY_LON_RANGE",
    "seasonality_lookup_file",
    "load_seasonality_lookup",
]
