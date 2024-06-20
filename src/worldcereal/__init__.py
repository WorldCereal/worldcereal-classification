#!/usr/bin/env python3

from ._version import __version__
from .utils.spatial import BoundingBoxExtent

__all__ = ["__version__", "BoundingBoxExtent"]

SUPPORTED_SEASONS = [
    "tc-s1",
    "tc-s2",
    "tc-annual",
    "custom",
]

SEASONAL_MAPPING = {
    "tc-s1": "S1",
    "tc-s2": "S2",
    "tc-annual": "ANNUAL",
    "custom": "custom",
}


# Default buffer (days) prior to
# season start
SEASON_PRIOR_BUFFER = {
    "tc-s1": 0,
    "tc-s2": 0,
    "tc-annual": 0,
    "custom": 0,
}


# Default buffer (days) after
# season end
SEASON_POST_BUFFER = {
    "tc-s1": 0,
    "tc-s2": 0,
    "tc-annual": 0,
    "custom": 0,
}
