"""Lightweight exports for training defaults and shared constants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

MIN_EDGE_BUFFER = 2

GLOBAL_SEASON_IDS: Tuple[str, ...] = ("tc-s1", "tc-s2")

SEASONALITY_LOOKUP_FILENAME = "seasonality_lookup.parquet"
SEASONALITY_LOOKUP_PACKAGE = "worldcereal.data.cropcalendars"
SEASONALITY_LOOKUP_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cropcalendars"
    / SEASONALITY_LOOKUP_FILENAME
)
SEASONALITY_LOOKUP_COLUMNS: Tuple[str, ...] = (
    "s1_sos_doy",
    "s1_eos_doy",
    "s2_sos_doy",
    "s2_eos_doy",
)
SEASONALITY_COLUMN_MAP: Dict[str, Tuple[str, str]] = {
    "tc-s1": ("s1_sos_doy", "s1_eos_doy"),
    "tc-s2": ("s2_sos_doy", "s2_eos_doy"),
}
SEASONALITY_LAT_RANGE = (-89.999, 89.999)
SEASONALITY_LON_RANGE = (-179.999, 179.999)

__all__ = [
    "GLOBAL_SEASON_IDS",
    "MIN_EDGE_BUFFER",
    "SEASONALITY_LOOKUP_FILENAME",
    "SEASONALITY_LOOKUP_PACKAGE",
    "SEASONALITY_LOOKUP_PATH",
    "SEASONALITY_LOOKUP_COLUMNS",
    "SEASONALITY_COLUMN_MAP",
    "SEASONALITY_LAT_RANGE",
    "SEASONALITY_LON_RANGE",
]
