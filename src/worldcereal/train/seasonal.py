"""Seasonal utility helpers shared between training and inference."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd

CompositeFreq = Literal["month", "dekad"]

# Anything that can be normalized to a calendar date: a numpy datetime64, a
# pandas Timestamp, a python datetime/date, or an ISO date string.
DateLike = object


@dataclass(frozen=True)
class SeasonWindow:
    """Month/day extent of a season, applied per year.

    ``year_offset`` is 0 for in-year seasons and 1 when the window crosses the
    year boundary (e.g. Nov -> Mar). Days that do not exist in a given month/year are
    clamped to the month's last day when the window is resolved to a concrete
    date (see `resolve_season_bounds`).
    """

    start_month: int
    start_day: int
    end_month: int
    end_day: int
    year_offset: int = 0


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


def advance_composite_slot(
    dt_in: np.datetime64, timestep_freq: CompositeFreq
) -> np.datetime64:
    """Return the start of the composite window immediately after ``dt_in``'s.
    This is the single source of truth for advancing one composite slot
    (month or dekad).
    """

    start = align_to_composite_window(dt_in, timestep_freq).astype("datetime64[D]")
    if timestep_freq == "month":
        return (start.astype("datetime64[M]") + np.timedelta64(1, "M")).astype(
            "datetime64[D]"
        )
    if timestep_freq == "dekad":
        offset_days = int((start - start.astype("datetime64[M]")).astype(int))  # 0/10/20
        if offset_days in (0, 10):
            return start + np.timedelta64(10, "D")
        return (start.astype("datetime64[M]") + np.timedelta64(1, "M")).astype(
            "datetime64[D]"
        )
    raise ValueError(f"Unknown compositing window: {timestep_freq}")


def composite_window_end(
    dt_in: np.datetime64, timestep_freq: CompositeFreq
) -> np.datetime64:
    """Return the inclusive END date of the composite window containing ``dt_in``.

    End-edge counterpart to `align_to_composite_window` (the start edge): together
    they bracket one slot (month: 1st <-> last day; dekad: 1/11/21 <-> 10/20/last).
    """

    return advance_composite_slot(dt_in, timestep_freq) - np.timedelta64(1, "D")


def enumerate_composite_slots(
    start: np.datetime64, end: np.datetime64, timestep_freq: CompositeFreq
) -> List[np.datetime64]:
    """List the composite-window starts from ``start`` to ``end`` (inclusive).

    Both bounds are expected to be composite-aligned (see
    `align_to_composite_window`). Returns an empty list when ``end < start``.
    """

    if end < start:
        return []
    slots: List[np.datetime64] = [start]
    current = start
    while current < end:
        current = advance_composite_slot(current, timestep_freq)
        slots.append(current)
    return slots


def _as_day(value: DateLike) -> np.datetime64:
    """Normalize any date-like value to ``datetime64[D]``."""

    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    return np.datetime64(pd.Timestamp(value), "D")


def coerce_date_for_year(year: int, month: int, day: int) -> np.datetime64:
    """Build a numpy datetime64, clamping the day to the month's max if needed."""

    last_day = calendar.monthrange(year, month)[1]
    safe_day = min(day, last_day)
    return np.datetime64(f"{year:04d}-{month:02d}-{safe_day:02d}", "D")


def season_window_from_dates(start: DateLike, end: DateLike) -> SeasonWindow:
    """Build a :class:`SeasonWindow` from two concrete dates.

    The window spans at most 12 consecutive months; ``year_offset`` is inferred
    from the two years (0 for in-year, 1 for a single new-year crossing).
    """

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError(
            f"Season window end {end_ts.date()} precedes start {start_ts.date()}."
        )
    year_offset = end_ts.year - start_ts.year
    if year_offset > 1:
        raise ValueError(
            "Season window may span at most 12 consecutive months "
            f"(got {start_ts.date()} -> {end_ts.date()})."
        )
    return SeasonWindow(
        start_month=start_ts.month,
        start_day=start_ts.day,
        end_month=end_ts.month,
        end_day=end_ts.day,
        year_offset=year_offset,
    )


def resolve_season_bounds(
    reference: DateLike, window: SeasonWindow
) -> Tuple[np.datetime64, np.datetime64]:
    """Resolve ``window`` to concrete (start, end) dates for ``reference``'s cycle.

    For year-crossing windows (``year_offset == 1``) the cycle is chosen so that
    ``reference`` is attributed to the season it most plausibly belongs to: if it
    falls on/after the window's end month/day it opens the cycle in its own year,
    otherwise it belongs to the cycle that opened the previous year.
    """

    ref = pd.Timestamp(reference)
    if window.year_offset == 0:
        start_year = ref.year
    else:
        after_end = (ref.month, ref.day) > (window.end_month, window.end_day)
        start_year = ref.year if after_end else ref.year - 1

    start = coerce_date_for_year(start_year, window.start_month, window.start_day)
    end = coerce_date_for_year(
        start_year + window.year_offset, window.end_month, window.end_day
    )
    return start, end


def date_in_season(
    label: DateLike,
    start: DateLike,
    end: DateLike,
    *,
    freq: CompositeFreq | None = None,
) -> bool:
    """Whether ``label`` falls within the ``[start, end]`` season range.

    With ``freq is None`` the comparison is at exact day resolution. With ``freq``
    set, the season *start* is snapped down to the start of its composite window
    and the season *end* up to the inclusive end of its composite window, while
    the label keeps its exact date. This compares exact label datetimes against
    exact composite-window edges (month or dekad), so a label counts as in-season
    only when its composite slot is part of the season.
    """

    label_d = _as_day(label)
    start_d = _as_day(start)
    end_d = _as_day(end)
    if freq is not None:
        start_d = align_to_composite_window(start_d, freq)
        end_d = composite_window_end(end_d, freq)
    return bool(start_d <= label_d <= end_d)


def in_season_window(
    ts: DateLike, window: SeasonWindow, *, freq: CompositeFreq | None = None
) -> bool:
    """Whether ``ts`` falls inside ``window`` (optionally composite-aligned).

    Single source of truth for "is this date in the season window?", shared by
    the season-alignment pre-filter, the manual-window dataset filter, and the
    ``in_season`` flag, so they can no longer disagree on the boundary.
    """

    start, end = resolve_season_bounds(ts, window)
    return date_in_season(ts, start, end, freq=freq)
