#!/usr/bin/env python3

from ._version import __version__

__all__ = ['__version__']


SUPPORTED_SEASONS = [
    'tc-wintercereals',
    'tc-maize-main',
    'tc-maize-second',
    'tc-annual',
    'custom'
]

SEASONAL_MAPPING = {
    'tc-wintercereals': 'WW',
    'tc-maize-main': 'M1',
    'tc-maize-second': 'M2',
    'tc-annual': 'ANNUAL',
    'custom': 'custom'
}


# Default buffer (days) prior to
# season start
SEASON_PRIOR_BUFFER = {
    'tc-wintercereals': 15,
    'tc-maize-main': 15,
    'tc-maize-second': 15,
    'tc-annual': 0,
    'custom': 0
}


# Default buffer (days) after
# season end
SEASON_POST_BUFFER = {
    'tc-wintercereals': 0,
    'tc-maize-main': 0,
    'tc-maize-second': 0,
    'tc-annual': 0,
    'custom': 0
}


# Base temperatures used for
# crop-specific GDD accumulation
TBASE = {
    'tc-wintercereals': 0,
    'tc-maize-main': 10,
    'tc-maize-second': 10
}


# Upper limit temperatures for
# GDD accumulation
GDDTLIMIT = {
    'tc-wintercereals': 25,
    'tc-maize-main': 30,
    'tc-maize-second': 30
}
