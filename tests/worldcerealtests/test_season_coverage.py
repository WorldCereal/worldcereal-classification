"""Tests for the min_season_coverage threshold in season mask computation.

Verifies that:
- WorldCerealDataset stores and validates min_season_coverage.
- _season_mask_from_window applies the threshold so that a season partially
  outside the augmented window still yields non-zero supervision when the
  fraction of covered slots meets the threshold.
- _season_mask_from_calendar applies the same threshold logic.
- prepare_training_datasets routes train_min_season_coverage to the training
  split only; validation and test splits always use 1.0 (full coverage required).

These tests are the regression suite for the augmentation-resilience fix
described in the design analysis: random window shifts during training can
leave a season only partially inside the 12-timestamp window, causing the
original binary coverage gate to drop all crop-type supervision for that
sample. The new threshold-based gate preserves supervision whenever the
fraction of covered season slots is ≥ min_season_coverage.
"""

from typing import Literal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from worldcereal.train.datasets import (
    SeasonWindow,
    WorldCerealDataset,
)
from worldcereal.train.finetuning_utils import prepare_training_datasets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_dataset(
    timestep_freq: Literal["month", "dekad"] = "month",
    min_season_coverage: float = 1.0,
) -> WorldCerealDataset:
    """Instantiate a WorldCerealDataset with a trivial dataframe."""
    df = pd.DataFrame({"x": range(2)})
    return WorldCerealDataset(
        df,
        timestep_freq=timestep_freq,
        min_season_coverage=min_season_coverage,
    )


def _monthly_dates(start_year: int, start_month: int, n_months: int) -> np.ndarray:
    """Return n_months monthly datetime64[D] composites from start_year/start_month.

    Each date is the 1st of the month, matching what align_to_composite_window
    produces for monthly timestep_freq.
    """
    dates = []
    year, month = start_year, start_month
    for _ in range(n_months):
        dates.append(np.datetime64(f"{year:04d}-{month:02d}-01", "D"))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return np.array(dates, dtype="datetime64[D]")


# April–September window, 6 monthly slots: Apr, May, Jun, Jul, Aug, Sep
_APR_SEP_WINDOW = SeasonWindow(
    start_month=4, start_day=1, end_month=9, end_day=30, year_offset=0
)


# ---------------------------------------------------------------------------
# 1. Parameter validation
# ---------------------------------------------------------------------------


class TestMinSeasonCoverageParameter:
    def test_default_is_full_coverage(self):
        ds = _minimal_dataset()
        assert ds.min_season_coverage == 1.0

    def test_custom_value_stored(self):
        ds = _minimal_dataset(min_season_coverage=0.5)
        assert ds.min_season_coverage == 0.5

    def test_boundary_value_one_is_valid(self):
        ds = _minimal_dataset(min_season_coverage=1.0)
        assert ds.min_season_coverage == 1.0

    def test_small_positive_is_valid(self):
        ds = _minimal_dataset(min_season_coverage=0.1)
        assert ds.min_season_coverage == pytest.approx(0.1)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="min_season_coverage"):
            _minimal_dataset(min_season_coverage=0.0)

    def test_above_one_raises(self):
        with pytest.raises(ValueError, match="min_season_coverage"):
            _minimal_dataset(min_season_coverage=1.01)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="min_season_coverage"):
            _minimal_dataset(min_season_coverage=-0.5)


# ---------------------------------------------------------------------------
# 2. _season_mask_from_window: full / partial / zero coverage
# ---------------------------------------------------------------------------


class TestSeasonMaskFromWindowCoverage:
    """Unit tests for _season_mask_from_window with an April-September (6-slot) window.

    Naming convention for composite_date windows:
      - "full"  : Jan–Dec 2021  – all 6 season slots present
      - "half"  : Jul2021–Jun2022 – 3/6 slots per annual cycle
      - "third" : Aug2021–Jul2022 – 2/6 from the 2021 cycle, 4/6 from 2022 cycle
      - "none"  : Jan–Mar 2021  – 0/6 slots present
    """

    # --- Full coverage (Jan–Dec 2021, 12 months) ---

    def test_full_coverage_strict_threshold_enables_mask(self):
        """All 6 season slots present → mask is enabled with min_season_coverage=1.0."""
        ds = _minimal_dataset(min_season_coverage=1.0)
        dates = _monthly_dates(2021, 1, 12)
        mask, _ = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, None)
        # Apr(idx 3)–Sep(idx 8)
        assert int(mask.sum()) == 6
        assert bool(mask[3])  # April
        assert bool(mask[8])  # September

    def test_full_coverage_relaxed_threshold_also_enables_mask(self):
        """Full coverage should pass any threshold."""
        ds = _minimal_dataset(min_season_coverage=0.5)
        dates = _monthly_dates(2021, 1, 12)
        mask, _ = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, None)
        assert int(mask.sum()) == 6

    # --- 50% coverage (3/6 per annual cycle, Jul2021–Jun2022) ---

    def test_partial_50pct_coverage_strict_threshold_all_false(self):
        """3 of 6 season slots present → mask is all-False with min_season_coverage=1.0.

        Jul2021–Jun2022 window:
          2021 cycle (Apr-Sep 2021): Jul, Aug, Sep covered → 3/6
          2022 cycle (Apr-Sep 2022): Apr, May, Jun covered → 3/6
        Both cycles need 6/6 slots → both fail → all-False.
        """
        ds = _minimal_dataset(min_season_coverage=1.0)
        dates = _monthly_dates(2021, 7, 12)
        mask, in_flag = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, None)
        assert int(mask.sum()) == 0
        assert in_flag is False

    def test_partial_50pct_coverage_relaxed_threshold_non_zero(self):
        """3 of 6 season slots present → mask is non-zero with min_season_coverage=0.5.

        n_required = max(1, round(6 * 0.5)) = 3.  Both cycles provide exactly 3 → pass.
        Expected True entries: Jul,Aug,Sep 2021 + Apr,May,Jun 2022 = 6 total.
        """
        ds = _minimal_dataset(min_season_coverage=0.5)
        dates = _monthly_dates(2021, 7, 12)
        mask, _ = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, None)
        assert int(mask.sum()) == 6  # 3 from 2021 cycle + 3 from 2022 cycle

    def test_partial_50pct_threshold_boundary_at_exactly_required(self):
        """Exactly n_required slots → threshold is inclusive (>=), so it passes."""
        # n_required at 0.5 for 6-slot season = 3; test that 3 == 3 qualifies
        ds = _minimal_dataset(min_season_coverage=0.5)
        dates = _monthly_dates(2021, 7, 12)  # 3/6 per cycle → exactly at boundary
        mask, _ = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, None)
        assert int(mask.sum()) > 0, (
            "Exactly n_required covered slots should pass the threshold"
        )

    # --- Mixed per-cycle coverage (Aug2021–Jul2022) ---

    def test_mixed_cycle_coverage_each_cycle_evaluated_independently(self):
        """Each annual cycle is evaluated independently against the threshold.

        Aug2021–Jul2022 window:
          2021 cycle (Apr-Sep 2021): Aug, Sep covered → 2/6 < 3 → fails
          2022 cycle (Apr-Sep 2022): Apr, May, Jun, Jul covered → 4/6 >= 3 → passes
        Only the 2022 cycle contributes to the mask.
        """
        ds = _minimal_dataset(min_season_coverage=0.5)
        dates = _monthly_dates(2021, 8, 12)
        mask, _ = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, None)
        # Only Apr,May,Jun,Jul 2022 → 4 True entries
        assert int(mask.sum()) == 4
        # Aug,Sep 2021 must NOT be in the mask (their cycle failed the threshold)
        aug_2021 = np.datetime64("2021-08-01", "D")
        sep_2021 = np.datetime64("2021-09-01", "D")
        assert not bool(mask[list(dates).index(aug_2021)])
        assert not bool(mask[list(dates).index(sep_2021)])

    # --- in_seasons flag with partial coverage ---

    def test_in_flag_true_when_label_inside_qualifying_cycle(self):
        """in_flag is True when label_datetime falls inside a cycle that meets the threshold."""
        ds = _minimal_dataset(min_season_coverage=0.5)
        dates = _monthly_dates(2021, 7, 12)  # Jul2021–Jun2022; 2022 cycle qualifies
        label_dt = np.datetime64("2022-05-15", "D")  # inside Apr-Sep 2022
        _, in_flag = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, label_dt)
        assert in_flag is True

    def test_in_flag_false_when_cycle_fails_strict_threshold(self):
        """in_flag is False when the cycle containing the label fails the strict threshold."""
        ds = _minimal_dataset(min_season_coverage=1.0)
        dates = _monthly_dates(2021, 7, 12)
        label_dt = np.datetime64("2022-05-15", "D")
        _, in_flag = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, label_dt)
        assert in_flag is False

    # --- Zero coverage ---

    def test_zero_coverage_relaxed_threshold_still_all_false(self):
        """No season slots in the window → mask all-False regardless of threshold."""
        ds = _minimal_dataset(min_season_coverage=0.5)
        # Jan–Mar 2021 only; Apr-Sep season has no overlap
        dates = _monthly_dates(2021, 1, 3)
        mask, in_flag = ds._season_mask_from_window(_APR_SEP_WINDOW, dates, None)
        assert int(mask.sum()) == 0
        assert in_flag is False


# ---------------------------------------------------------------------------
# 3. Core regression: augmentation-shift scenario (train vs eval)
# ---------------------------------------------------------------------------


class TestAugmentationResilienceRegression:
    """
    Regression tests for the specific augmentation problem that motivated this fix.

    During training with augment=True, the 12-timestamp window can drift up to
    ±5 positions from the valid_position.  A +5 shift can push the early part
    of a season (e.g. the first 3 months of a 6-month season) outside the window,
    leaving only 50% coverage.  With the old binary gate this silently dropped
    all crop-type supervision; with the new threshold gate the training split
    (min_season_coverage=0.5) retains supervision while the eval split
    (min_season_coverage=1.0) correctly withholds it.
    """

    def test_train_retains_supervision_when_window_shifted(self):
        """Training dataset produces a non-zero mask for a 50%-covered season."""
        train_ds = _minimal_dataset(min_season_coverage=0.5)
        # Simulates augmented window: season partially outside on the left
        shifted_dates = _monthly_dates(2021, 7, 12)  # Jul2021–Jun2022; 3/6 per cycle
        mask, _ = train_ds._season_mask_from_window(
            _APR_SEP_WINDOW, shifted_dates, None
        )
        assert int(mask.sum()) > 0, (
            "Training dataset should still produce a non-zero season mask "
            "when 50% of season slots are covered after augmentation shift."
        )

    def test_eval_withholds_supervision_when_window_shifted(self):
        """Eval dataset produces an all-False mask for the same 50%-covered window."""
        eval_ds = _minimal_dataset(min_season_coverage=1.0)
        shifted_dates = _monthly_dates(2021, 7, 12)
        mask, _ = eval_ds._season_mask_from_window(_APR_SEP_WINDOW, shifted_dates, None)
        assert int(mask.sum()) == 0, (
            "Eval dataset should produce an all-False mask when full season "
            "coverage is not met, preserving evaluation integrity."
        )

    def test_both_seasons_survive_moderate_shift(self):
        """Both tc-s1 and tc-s2 seasons retain supervision with a moderate shift.

        This mirrors the real-world dual-season setup where both seasons must
        survive the shift to avoid total supervision loss for a sample.
        Uses two non-overlapping windows spanning the full calendar year.
        """
        # Season A: Jan–Jun (6 slots)
        season_a = SeasonWindow(
            start_month=1, start_day=1, end_month=6, end_day=30, year_offset=0
        )
        # Season B: Jul–Dec (6 slots)
        season_b = SeasonWindow(
            start_month=7, start_day=1, end_month=12, end_day=31, year_offset=0
        )

        # Moderate shift: window covers Apr2021–Mar2022
        # Season A 2021: Jan-Jun 2021 → Apr,May,Jun covered = 3/6 = 50%
        # Season B 2021: Jul-Dec 2021 → Jul-Dec all covered = 6/6 = 100%
        # Season A 2022: Jan-Jun 2022 → Jan,Feb,Mar covered = 3/6 = 50%
        # Season B 2022: Jul-Dec 2022 → 0/6 (not in window)
        shifted_dates = _monthly_dates(2021, 4, 12)  # Apr2021–Mar2022

        train_ds = _minimal_dataset(min_season_coverage=0.5)
        mask_a, _ = train_ds._season_mask_from_window(season_a, shifted_dates, None)
        mask_b, _ = train_ds._season_mask_from_window(season_b, shifted_dates, None)

        assert int(mask_a.sum()) > 0, "Season A should survive with 50% threshold"
        assert int(mask_b.sum()) > 0, "Season B should survive with 100% coverage"


# ---------------------------------------------------------------------------
# 4. _season_mask_from_calendar: threshold applied consistently
# ---------------------------------------------------------------------------


class TestSeasonMaskFromCalendarCoverage:
    """Verify that _season_mask_from_calendar uses the same threshold logic.

    We monkeypatch _season_context_for to avoid a dependency on the
    seasonality lookup file so the test runs in any environment.
    """

    _SEASON_START = np.datetime64("2021-04-01", "D")
    _SEASON_END = np.datetime64("2021-09-30", "D")

    def _patch_context(self, ds: WorldCerealDataset):
        return patch.object(
            ds,
            "_season_context_for",
            return_value=(self._SEASON_START, self._SEASON_END),
        )

    def test_full_coverage_strict_threshold_enables_mask(self):
        ds = _minimal_dataset(min_season_coverage=1.0)
        dates = _monthly_dates(2021, 1, 12)
        with self._patch_context(ds):
            mask, _ = ds._season_mask_from_calendar(
                "tc-s1", dates, 2021, {}, None, 50.0, 10.0
            )
        assert int(mask.sum()) == 6

    def test_partial_coverage_strict_threshold_all_false(self):
        """3/6 slots present → all-False with min_season_coverage=1.0."""
        ds = _minimal_dataset(min_season_coverage=1.0)
        # Jul-Dec 2021 only; Apr-Jun outside window → 3/6
        dates = _monthly_dates(2021, 7, 6)
        with self._patch_context(ds):
            mask, in_flag = ds._season_mask_from_calendar(
                "tc-s1", dates, 2021, {}, None, 50.0, 10.0
            )
        assert int(mask.sum()) == 0
        assert in_flag is False

    def test_partial_coverage_relaxed_threshold_non_zero(self):
        """3/6 slots present → non-zero mask with min_season_coverage=0.5."""
        ds = _minimal_dataset(min_season_coverage=0.5)
        dates = _monthly_dates(2021, 7, 6)
        with self._patch_context(ds):
            mask, _ = ds._season_mask_from_calendar(
                "tc-s1", dates, 2021, {}, None, 50.0, 10.0
            )
        # Jul, Aug, Sep 2021 are in both the composite window and the Apr-Sep season
        assert int(mask.sum()) == 3

    def test_in_flag_gated_on_threshold(self):
        """in_flag is True only when the threshold is met and label is in season."""
        label_dt = np.datetime64("2021-07-15", "D")  # inside Apr-Sep 2021
        dates = _monthly_dates(2021, 7, 6)

        ds_strict = _minimal_dataset(min_season_coverage=1.0)
        ds_relaxed = _minimal_dataset(min_season_coverage=0.5)

        with self._patch_context(ds_strict):
            _, in_flag_strict = ds_strict._season_mask_from_calendar(
                "tc-s1", dates, 2021, {}, label_dt, 50.0, 10.0
            )
        with self._patch_context(ds_relaxed):
            _, in_flag_relaxed = ds_relaxed._season_mask_from_calendar(
                "tc-s1", dates, 2021, {}, label_dt, 50.0, 10.0
            )

        assert in_flag_strict is False, (
            "Strict threshold: label-in-season but threshold not met → in_flag False"
        )
        assert in_flag_relaxed is True, (
            "Relaxed threshold: threshold met and label in season → in_flag True"
        )


# ---------------------------------------------------------------------------
# 5. prepare_training_datasets: split-level coverage settings
# ---------------------------------------------------------------------------


class TestPrepareTrainingDatasetsCoverage:
    """Verify that prepare_training_datasets routes min_season_coverage correctly.

    Training split must receive train_min_season_coverage.
    Validation and test splits must always receive 1.0 regardless of what the
    caller passes for train_min_season_coverage, so evaluation is never
    inflated by partial-season pooling.
    """

    @pytest.fixture
    def tiny_dfs(self):
        """Minimal DataFrames sufficient to construct WorldCerealLabelledDataset."""
        df = pd.DataFrame({"dummy_col": range(4)})
        return df.copy(), df.copy(), df.copy()

    def test_default_train_coverage_is_half(self, tiny_dfs):
        train_df, val_df, test_df = tiny_dfs
        train_ds, val_ds, test_ds = prepare_training_datasets(
            train_df, val_df, test_df, emit_label_tensor=False
        )
        assert train_ds.min_season_coverage == pytest.approx(0.5)
        assert val_ds.min_season_coverage == pytest.approx(1.0)
        assert test_ds.min_season_coverage == pytest.approx(1.0)

    def test_custom_train_coverage_propagates_to_train_only(self, tiny_dfs):
        train_df, val_df, test_df = tiny_dfs
        train_ds, val_ds, test_ds = prepare_training_datasets(
            train_df,
            val_df,
            test_df,
            emit_label_tensor=False,
            train_min_season_coverage=0.7,
        )
        assert train_ds.min_season_coverage == pytest.approx(0.7)
        assert val_ds.min_season_coverage == pytest.approx(1.0)
        assert test_ds.min_season_coverage == pytest.approx(1.0)

    def test_val_test_coverage_is_always_full_even_with_strict_train(self, tiny_dfs):
        """Even when train uses 1.0, val/test must also be explicitly 1.0."""
        train_df, val_df, test_df = tiny_dfs
        train_ds, val_ds, test_ds = prepare_training_datasets(
            train_df,
            val_df,
            test_df,
            emit_label_tensor=False,
            train_min_season_coverage=1.0,
        )
        assert train_ds.min_season_coverage == pytest.approx(1.0)
        assert val_ds.min_season_coverage == pytest.approx(1.0)
        assert test_ds.min_season_coverage == pytest.approx(1.0)

    def test_augment_false_still_applies_train_coverage_param(self, tiny_dfs):
        """train_min_season_coverage is applied regardless of the augment flag."""
        train_df, val_df, test_df = tiny_dfs
        train_ds, _, _ = prepare_training_datasets(
            train_df,
            val_df,
            test_df,
            emit_label_tensor=False,
            augment=False,
            train_min_season_coverage=0.6,
        )
        assert train_ds.min_season_coverage == pytest.approx(0.6)
