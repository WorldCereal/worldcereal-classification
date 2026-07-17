"""Tests for the temporal subsetting in ``worldcereal.local_inference``.

The model is trained on exactly ``max_timesteps`` real composite slots, so
inference must hand it the same. These cover the three ways that contract can
break, so that silently-wrong maps are not produced.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from worldcereal.local_inference import subset_ds_temporally


def _monthly_ds(start: str, periods: int) -> xr.Dataset:
    """Cube with monthly composite slots labelled at month start."""
    return xr.Dataset(
        {"B02": (("t",), np.arange(periods))},
        coords={"t": pd.date_range(start, periods=periods, freq="MS")},
    )


def _dekadal_ds(year: int) -> xr.Dataset:
    """Cube with dekadal composite slots (days 1/11/21 of each month)."""
    stamps = [
        pd.Timestamp(f"{year}-{m:02d}-{d:02d}")
        for m in range(1, 13)
        for d in (1, 11, 21)
    ]
    return xr.Dataset(
        {"B02": (("t",), np.arange(len(stamps)))},
        coords={"t": pd.DatetimeIndex(stamps)},
    )


def _slots(ds: xr.Dataset) -> list:
    return [str(t.date()) for t in pd.to_datetime(ds.t.values)]


def test_two_season_union_keeps_its_own_slots_without_padding():
    """The raw union is honoured as-is when pad_to_max is off."""
    ds = _monthly_ds("2021-04-01", 28)

    out = subset_ds_temporally(ds, ("2021-11-19", "2022-09-03"), min_coverage=0.5)

    assert _slots(out) == [
        f"{y}-{m:02d}-01"
        for y, m in [(2021, 11), (2021, 12)] + [(2022, m) for m in range(1, 10)]
    ]


def test_multi_year_window_trims_instead_of_collapsing():
    """A >12-month window must trim to the last max_timesteps slots.
    """
    ds = _monthly_ds("2021-04-01", 28)

    out = subset_ds_temporally(ds, ("2021-04-01", "2023-07-31"), min_coverage=0.5)

    assert len(out.t) == 12
    assert _slots(out)[0] == "2022-08-01"
    assert _slots(out)[-1] == "2023-07-01"


def test_short_union_widens_to_exactly_max_timesteps_toward_the_tail():
    """prefer_tail=True grows the END first: 11 slots -> 12 by adding 2022-10."""
    ds = _monthly_ds("2021-04-01", 28)

    out = subset_ds_temporally(
        ds, ("2021-11-19", "2022-09-03"), min_coverage=0.5, pad_to_max=True
    )

    assert len(out.t) == 12
    assert _slots(out)[0] == "2021-11-01"
    assert _slots(out)[-1] == "2022-10-01"


def test_widening_grows_the_head_when_prefer_tail_is_false():
    ds = _monthly_ds("2021-04-01", 28)

    out = subset_ds_temporally(
        ds,
        ("2021-11-19", "2022-09-03"),
        min_coverage=0.5,
        pad_to_max=True,
        prefer_tail=False,
    )

    assert len(out.t) == 12
    assert _slots(out)[0] == "2021-10-01"
    assert _slots(out)[-1] == "2022-09-01"


def test_widening_falls_back_to_the_head_at_the_tail_data_boundary():
    """No room to grow forward (cube ends at the window end) -> grow backwards."""
    ds = _monthly_ds("2021-01-01", 21)  # ends 2022-09

    out = subset_ds_temporally(
        ds, ("2021-11-19", "2022-09-03"), min_coverage=0.5, pad_to_max=True
    )

    assert len(out.t) == 12
    assert _slots(out)[0] == "2021-10-01"
    assert _slots(out)[-1] == "2022-09-01"


def test_widening_raises_when_the_cube_cannot_supply_enough_real_slots():
    """Fail loudly rather than pad with nodata timesteps."""
    ds = _monthly_ds("2021-11-01", 11)  # only the 11 union slots exist

    with pytest.raises(ValueError, match="Cannot widen the season window"):
        subset_ds_temporally(
            ds, ("2021-11-19", "2022-09-03"), min_coverage=0.5, pad_to_max=True
        )


def test_dekadal_windows_use_the_dekad_grid():
    """Slot enumeration must follow the model's composite frequency."""
    ds = _dekadal_ds(2022)

    out = subset_ds_temporally(
        ds,
        ("2022-03-15", "2022-06-25"),
        min_coverage=1.0,
        max_timesteps=36,
        timestep_freq="dekad",
    )

    # 2022-03-15 snaps back to the 2nd dekad (11th); 06-25 to the 3rd (21st).
    assert _slots(out)[0] == "2022-03-11"
    assert _slots(out)[-1] == "2022-06-21"
    assert len(out.t) == 11


def test_missing_slots_within_coverage_are_nodata_filled():
    """Gaps inside the window are reindexed with nodata, above min_coverage."""
    ds = _monthly_ds("2021-11-01", 9)  # window tail (2022-08/09) absent

    out = subset_ds_temporally(
        ds,
        ("2021-11-19", "2022-09-03"),
        min_coverage=0.5,
        nodata_value=65535,
    )

    assert len(out.t) == 11
    assert int(out["B02"].values[-1]) == 65535


def test_coverage_below_threshold_raises():
    ds = _monthly_ds("2021-11-01", 2)

    with pytest.raises(ValueError, match="below min_coverage"):
        subset_ds_temporally(ds, ("2021-11-19", "2022-09-03"), min_coverage=0.9)
