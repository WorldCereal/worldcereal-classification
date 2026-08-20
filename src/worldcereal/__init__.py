#!/usr/bin/env python3

from ._version import __version__

__all__ = ["__version__"]

SUPPORTED_SEASONS = [
    "tc-s1",
    "tc-s2",
    "tc-annual",
    "custom",
]


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
