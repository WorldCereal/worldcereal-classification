"""Tests for the shared season-window helpers in ``worldcereal.train.seasonal``.

These cover the composite-grid primitives (start/end edges, slot advance) and
the season-membership stack (``SeasonWindow`` resolution + ``in_season_window``)
that unify how dates are compared against a season across the pipeline.

Regression focus: the ``in_season`` flag must agree with the season the user
asked for at *composite* resolution. A label whose valid_time falls in the final
month of the window (e.g. Aug 15 for a Feb->Aug season) must count as in-season,
while staying correct for dekadal compositing (an Aug 31 label must fall out of a
season that ends in the 2nd dekad of August).
"""

import numpy as np
import pandas as pd
import pytest

from worldcereal.train.seasonal import (
    SeasonWindow,
    advance_composite_slot,
    align_to_composite_window,
    composite_window_end,
    date_in_season,
    in_season_window,
    resolve_season_bounds,
    season_window_from_dates,
)


def _d(value: str) -> np.datetime64:
    return np.datetime64(value, "D")


# ---------------------------------------------------------------------------
# Composite-grid primitives
# ---------------------------------------------------------------------------


class TestAlignToCompositeWindow:
    @pytest.mark.parametrize(
        "date, expected",
        [
            ("2023-08-01", "2023-08-01"),
            ("2023-08-15", "2023-08-01"),
            ("2023-08-31", "2023-08-01"),
            ("2024-02-29", "2024-02-01"),
        ],
    )
    def test_month_snaps_to_first(self, date, expected):
        assert align_to_composite_window(_d(date), "month") == _d(expected)

    @pytest.mark.parametrize(
        "date, expected",
        [
            ("2023-08-01", "2023-08-01"),  # 1st dekad
            ("2023-08-10", "2023-08-01"),
            ("2023-08-11", "2023-08-11"),  # 2nd dekad
            ("2023-08-20", "2023-08-11"),
            ("2023-08-21", "2023-08-21"),  # 3rd dekad
            ("2023-08-31", "2023-08-21"),
        ],
    )
    def test_dekad_snaps_to_slot_start(self, date, expected):
        assert align_to_composite_window(_d(date), "dekad") == _d(expected)

    def test_unknown_freq_raises(self):
        with pytest.raises(ValueError):
            align_to_composite_window(_d("2023-08-15"), "weekly")


class TestAdvanceCompositeSlot:
    @pytest.mark.parametrize(
        "date, expected",
        [
            ("2023-08-15", "2023-09-01"),
            ("2023-12-15", "2024-01-01"),  # year roll-over
            ("2024-02-10", "2024-03-01"),
        ],
    )
    def test_month(self, date, expected):
        assert advance_composite_slot(_d(date), "month") == _d(expected)

    @pytest.mark.parametrize(
        "date, expected",
        [
            ("2023-08-05", "2023-08-11"),  # 1st -> 2nd dekad
            ("2023-08-15", "2023-08-21"),  # 2nd -> 3rd dekad
            ("2023-08-25", "2023-09-01"),  # 3rd -> next month
            ("2023-02-25", "2023-03-01"),  # short month, non-leap
            ("2024-02-25", "2024-03-01"),  # short month, leap
        ],
    )
    def test_dekad(self, date, expected):
        assert advance_composite_slot(_d(date), "dekad") == _d(expected)

    def test_accepts_unaligned_input(self):
        # Unlike the strict dataset method, any date in a slot advances correctly.
        assert advance_composite_slot(_d("2023-08-17"), "dekad") == _d("2023-08-21")

    def test_unknown_freq_raises(self):
        with pytest.raises(ValueError):
            advance_composite_slot(_d("2023-08-15"), "weekly")


class TestCompositeWindowEnd:
    @pytest.mark.parametrize(
        "date, expected",
        [
            ("2023-08-15", "2023-08-31"),
            ("2023-02-10", "2023-02-28"),  # non-leap February
            ("2024-02-10", "2024-02-29"),  # leap February
            ("2023-04-05", "2023-04-30"),
            ("2023-12-31", "2023-12-31"),
        ],
    )
    def test_month(self, date, expected):
        assert composite_window_end(_d(date), "month") == _d(expected)

    @pytest.mark.parametrize(
        "date, expected",
        [
            ("2023-08-05", "2023-08-10"),  # 1st dekad ends on the 10th
            ("2023-08-15", "2023-08-20"),  # 2nd dekad ends on the 20th
            ("2023-08-25", "2023-08-31"),  # 3rd dekad runs to month end
            ("2023-02-25", "2023-02-28"),  # 3rd dekad, non-leap
            ("2024-02-25", "2024-02-29"),  # 3rd dekad, leap
        ],
    )
    def test_dekad(self, date, expected):
        assert composite_window_end(_d(date), "dekad") == _d(expected)

    def test_is_start_edge_complement(self):
        # start edge and end edge bracket exactly one slot
        for date in ("2023-08-05", "2023-08-15", "2023-08-25"):
            start = align_to_composite_window(_d(date), "dekad")
            end = composite_window_end(_d(date), "dekad")
            assert start <= _d(date) <= end


# ---------------------------------------------------------------------------
# SeasonWindow construction & resolution
# ---------------------------------------------------------------------------


class TestSeasonWindowFromDates:
    def test_in_year(self):
        assert season_window_from_dates("2023-02-01", "2023-08-31") == SeasonWindow(
            2, 1, 8, 31, 0
        )

    def test_year_crossing_sets_offset(self):
        assert season_window_from_dates("2023-11-01", "2024-03-31") == SeasonWindow(
            11, 1, 3, 31, 1
        )

    def test_accepts_timestamps(self):
        w = season_window_from_dates(
            pd.Timestamp("2023-02-01"), np.datetime64("2023-08-31")
        )
        assert w == SeasonWindow(2, 1, 8, 31, 0)

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError):
            season_window_from_dates("2023-08-31", "2023-02-01")

    def test_more_than_one_year_crossing_raises(self):
        with pytest.raises(ValueError):
            season_window_from_dates("2022-02-01", "2024-08-31")


class TestResolveSeasonBounds:
    def test_in_year_uses_reference_year(self):
        w = SeasonWindow(2, 1, 8, 31, 0)
        start, end = resolve_season_bounds("2023-05-10", w)
        assert (start, end) == (_d("2023-02-01"), _d("2023-08-31"))

    def test_in_year_independent_of_reference_year(self):
        w = SeasonWindow(2, 1, 8, 31, 0)
        start, end = resolve_season_bounds("2020-05-10", w)
        assert (start, end) == (_d("2020-02-01"), _d("2020-08-31"))

    @pytest.mark.parametrize(
        "reference, exp_start, exp_end",
        [
            ("2024-01-10", "2023-11-01", "2024-03-31"),  # inside the crossing season
            ("2023-12-15", "2023-11-01", "2024-03-31"),  # before new year
            ("2024-11-15", "2024-11-01", "2025-03-31"),  # next cycle
        ],
    )
    def test_year_crossing_cycle_selection(self, reference, exp_start, exp_end):
        w = SeasonWindow(11, 1, 3, 31, 1)
        start, end = resolve_season_bounds(reference, w)
        assert (start, end) == (_d(exp_start), _d(exp_end))

    @pytest.mark.parametrize(
        "ref, expected_end",
        [("2023-05-01", "2023-02-28"), ("2024-05-01", "2024-02-29")],
    )
    def test_invalid_day_is_clamped_to_month_end(self, ref, expected_end):
        # A Feb-29 boundary clamps to 28/29 depending on the resolved year.
        w = SeasonWindow(1, 1, 2, 29, 0)
        _, end = resolve_season_bounds(ref, w)
        assert end == _d(expected_end)


# ---------------------------------------------------------------------------
# Membership: date_in_season / in_season_window
# ---------------------------------------------------------------------------


class TestDateInSeasonDayResolution:
    """freq=None keeps exact day-resolution, inclusive on both ends."""

    @pytest.mark.parametrize(
        "label, expected",
        [
            ("2023-02-01", True),  # inclusive start
            ("2023-08-31", True),  # inclusive end
            ("2023-01-31", False),
            ("2023-09-01", False),
        ],
    )
    def test_exact_bounds(self, label, expected):
        assert date_in_season(label, "2023-02-01", "2023-08-31") is expected


class TestDateInSeasonComposite:
    def test_month_end_rounds_up_to_month(self):
        # season end Aug 15 -> whole August in-season at month resolution
        assert date_in_season("2023-08-31", "2023-02-10", "2023-08-15", freq="month")
        assert not date_in_season(
            "2023-09-01", "2023-02-10", "2023-08-15", freq="month"
        )

    def test_month_start_rounds_down_to_month(self):
        # season start Feb 10 -> all of February in-season
        assert date_in_season("2023-02-01", "2023-02-10", "2023-08-15", freq="month")

    @pytest.mark.parametrize(
        "label, expected",
        [
            ("2023-08-20", True),  # last day of the end dekad is still in
            ("2023-08-21", False),  # the next dekad is out
        ],
    )
    def test_dekad_end_edge_rounds_up(self, label, expected):
        # season ends Aug 12, in the 2nd dekad of August -> snaps up to Aug 20
        assert (
            date_in_season(label, "2023-02-01", "2023-08-12", freq="dekad") is expected
        )

    @pytest.mark.parametrize(
        "label, expected",
        [
            ("2023-08-11", True),  # first day of the start dekad is in
            ("2023-08-10", False),  # the previous dekad is out
        ],
    )
    def test_dekad_start_edge_rounds_down(self, label, expected):
        # season starts Aug 12, in the 2nd dekad of August -> snaps down to Aug 11
        assert (
            date_in_season(label, "2023-08-12", "2023-12-31", freq="dekad") is expected
        )


class TestInSeasonWindow:
    @pytest.mark.parametrize(
        "label, expected",
        [
            ("2023-11-15", True),
            ("2024-01-10", True),
            ("2024-03-15", True),
            ("2023-10-20", False),
            ("2024-04-05", False),
        ],
    )
    def test_year_crossing_membership(self, label, expected):
        w = season_window_from_dates("2023-11-01", "2024-03-31")
        assert in_season_window(label, w, freq="month") is expected

    @pytest.mark.parametrize(
        "label",
        ["2023-08-15", pd.Timestamp("2023-08-15"), np.datetime64("2023-08-15")],
    )
    def test_accepts_mixed_input_types(self, label):
        w = season_window_from_dates("2023-02-01", "2023-08-31")
        assert in_season_window(label, w, freq="month")

    def test_day_resolution_is_stricter_at_sub_month_end(self):
        # freq=None respects the exact end day; the composite path rounds up.
        w = season_window_from_dates("2023-02-01", "2023-08-10")
        assert not in_season_window("2023-08-15", w, freq=None)
        assert in_season_window("2023-08-15", w, freq="month")
