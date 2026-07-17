"""Lightweight exports for training defaults and shared constants."""

from __future__ import annotations

from typing import Tuple

from worldcereal.data.cropcalendars import SEASONALITY_LOOKUP_FILENAME

MIN_EDGE_BUFFER = 2

GLOBAL_SEASON_IDS: Tuple[str, ...] = ("tc-s1", "tc-s2")

OUTLIER_COLUMNS: dict = {
    "CT_outlier_score": "CTY24_confidence_nonoutlier",
    "LC_outlier_score": "LC10_confidence_nonoutlier",
    "CT_outlier_flag": "CTY24_anomaly_flag",
    "LC_outlier_flag": "LC10_anomaly_flag",
}

__all__ = [
    "GLOBAL_SEASON_IDS",
    "OUTLIER_COLUMNS",
    "MIN_EDGE_BUFFER",
    "SEASONALITY_LOOKUP_FILENAME",
]
