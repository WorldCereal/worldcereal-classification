"""Seasonal utility helpers shared between training and inference."""

from __future__ import annotations

from typing import Literal

import numpy as np


def align_to_composite_window(
    dt_in: np.datetime64, timestep_freq: Literal["month", "dekad"]
) -> np.datetime64:
    """Return the start timestamp for the composite window containing ``dt_in``."""

    ts = dt_in.astype("object")
    year = ts.year
    month = ts.month
    day = ts.day

    if timestep_freq == "dekad":
        if day <= 10:
            correct_date = np.datetime64(f"{year}-{month:02d}-01")
        elif 11 <= day <= 20:
            correct_date = np.datetime64(f"{year}-{month:02d}-11")
        else:
            correct_date = np.datetime64(f"{year}-{month:02d}-21")
    elif timestep_freq == "month":
        correct_date = np.datetime64(f"{year}-{month:02d}-01")
    else:
        raise ValueError(f"Unknown compositing window: {timestep_freq}")

    return correct_date
