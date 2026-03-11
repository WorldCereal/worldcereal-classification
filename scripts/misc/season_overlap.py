"""CLI wrapper for worldcereal.season_overlap.

This is a convenience entry point for running the season overlap pipeline
from the command line. The actual implementation lives in the installed
worldcereal package at ``src/worldcereal/season_overlap.py``.

Usage::

    python scripts/misc/season_overlap.py \
        --s1-sos  S1_SOS_WGS84.tif --s1-eos  S1_EOS_WGS84.tif \
        --s2-sos  S2_SOS_WGS84.tif --s2-eos  S2_EOS_WGS84.tif \
        --out-dir output/

Or, equivalently, via the package module:

    python -m worldcereal.season_overlap [same args]
"""

from worldcereal.season_overlap import main

if __name__ == "__main__":
    main()
