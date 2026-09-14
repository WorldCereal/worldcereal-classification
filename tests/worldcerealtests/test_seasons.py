import json

import numpy as np
import pandas as pd
import pytest
from openeo_gfmap import BoundingBoxExtent, TemporalContext

from worldcereal import seasons


@pytest.fixture
def lookup_table():
	index = pd.MultiIndex.from_tuples(
		[(10.25, 20.25), (10.25, 21.25), (11.25, 179.75), (11.25, -179.75)],
		names=["lat", "lon"],
	)
	return pd.DataFrame(
		{
			"s1_sos_doy": [100, 110, 120, 130],
			"s1_eos_doy": [200, 210, 220, 230],
			"s2_sos_doy": [250, 260, 270, 280],
			"s2_eos_doy": [350, 360, 340, 330],
			"annual_sos_doy": [40, 50, 60, 70],
			"annual_eos_doy": [300, 310, 320, 330],
			"s1_sos_dekad": [30, 33, 36, 39],
			"s1_eos_dekad": [60, 63, 66, 69],
			"s2_sos_dekad": [75, 78, 81, 84],
			"s2_eos_dekad": [105, 108, 102, 99],
			"annual_sos_dekad": [12, 15, 18, 21],
			"annual_eos_dekad": [90, 93, 96, 99],
		},
		index=index,
	)


@pytest.fixture
def patched_lookup(monkeypatch, lookup_table):
	monkeypatch.setattr(
		seasons, "ensure_seasonality_lookup_table", lambda: lookup_table
	)
	return lookup_table


def test_snap_coordinate_to_lookup_grid_clamps_and_snaps():
	assert seasons._snap_coordinate_to_lookup_grid(10.49, (0, 20)) == 10.25
	assert seasons._snap_coordinate_to_lookup_grid(20, (0, 20)) == 20.25
	assert seasons._snap_coordinate_to_lookup_grid(-1, (0, 20)) == 0.25


@pytest.mark.parametrize(
	("season", "expected"),
	[
		("tc-s1", ("s1_sos_dekad", "s1_eos_dekad")),
		("tc-s2", ("s2_sos_dekad", "s2_eos_dekad")),
		("tc-annual", ("annual_sos_dekad", "annual_eos_dekad")),
	],
)
def test_resolve_cropcalendar_columns(season, expected):
	assert seasons.resolve_cropcalendar_columns(season) == expected


def test_resolve_cropcalendar_columns_rejects_unknown_values():
	with pytest.raises(ValueError, match="not available"):
		seasons.resolve_cropcalendar_columns("tc-unknown")


def test_fetch_cropcalendar_points_uses_snapped_cell(patched_lookup):
	assert seasons.fetch_cropcalendar_dekad_point("tc-s2", 10.4, 20.4) == (75, 105)


def test_fetch_cropcalendar_point_falls_back_to_nearest_cell(patched_lookup, caplog):
	result = seasons.fetch_cropcalendar_dekad_point("tc-s1", 12, 22)
	assert result == (33, 63)


def test_fetch_cropcalendar_point_can_disable_nearest_fallback(patched_lookup):
	with pytest.raises(ValueError, match="No seasonality record"):
		seasons.fetch_cropcalendar_dekad_point(
			"tc-s1", 12, 22, fallback_to_nearest=False
		)


def test_fetch_cropcalendar_point_rejects_distant_nearest_cell(patched_lookup):
	with pytest.raises(ValueError, match="too far"):
		seasons.fetch_cropcalendar_dekad_point(
			"tc-s1", 12, 22, max_fallback_distance_degrees=1.0
		)


def test_fetch_cropcalendar_point_rejects_invalid_values(patched_lookup):
	patched_lookup.loc[(10.25, 20.25), "s1_sos_dekad"] = 109

	with pytest.raises(ValueError, match="Valid .* range is 1-108"):
		seasons.fetch_cropcalendar_dekad_point("tc-s1", 10.4, 20.4)


def test_fetch_cropcalendar_dekads_extent_returns_medoid(patched_lookup):
	extent = BoundingBoxExtent(west=20, south=10, east=22, north=11, epsg=4326)

	assert seasons.fetch_cropcalendar_dekads_extent(["tc-s1"], extent) == {
		"tc-s1": (30, 60)
	}


def test_fetch_cropcalendar_dekads_extent_handles_dateline(patched_lookup):
	extent = BoundingBoxExtent(west=179, south=11, east=-179, north=12, epsg=4326)

	assert seasons.fetch_cropcalendar_dekads_extent(["tc-s1"], extent) == {
		"tc-s1": (36, 66)
	}


def test_fetch_cropcalendar_dekads_extent_falls_back_to_centroid(patched_lookup):
	extent = BoundingBoxExtent(
		west=20.5, south=10.5, east=20.75, north=10.75, epsg=4326
	)

	assert seasons.fetch_cropcalendar_dekads_extent(["tc-s1"], extent) == {
		"tc-s1": (30, 60)
	}


def test_fetch_cropcalendar_dekads_extent_rejects_distant_centroid(patched_lookup):
	extent = BoundingBoxExtent(west=30, south=30, east=31, north=31, epsg=4326)

	with pytest.raises(ValueError, match="too far"):
		seasons.fetch_cropcalendar_dekads_extent(["tc-s1"], extent)


def test_fetch_cropcalendar_dekads_extent_requires_a_season(patched_lookup):
	extent = BoundingBoxExtent(west=20, south=10, east=22, north=11, epsg=4326)

	with pytest.raises(ValueError, match="At least one season"):
		seasons.fetch_cropcalendar_dekads_extent([], extent)


def test_get_season_dates_for_extent_returns_temporal_context(patched_lookup):
	extent = BoundingBoxExtent(west=20, south=10, east=22, north=11, epsg=4326)

	assert seasons.get_season_dates_for_extent(extent, 2024, "tc-s1") == TemporalContext(
		"2023-11-01", "2024-08-31"
	)


def test_get_season_dates_for_extent_rejects_unknown_season(patched_lookup):
	extent = BoundingBoxExtent(west=20, south=10, east=22, north=11, epsg=4326)

	with pytest.raises(ValueError, match="not supported"):
		seasons.get_season_dates_for_extent(extent, 2024, "tc-unknown")


@pytest.mark.parametrize(
	("dekad", "mode", "expected"),
	[
		(1, "first", "1999-01-01"),
		(3, "last", "1999-01-31"),
		(36, "last", "1999-12-31"),
		(37, "first", "2000-01-01"),
		(72, "last", "2000-12-31"),
		(73, "first", "2001-01-01"),
	],
)
def test_season_dekad_to_date_handles_three_year_window(dekad, mode, expected):
	assert seasons.season_dekad_to_date(dekad, target_year=2000, mode=mode).isoformat() == expected


def test_season_dekad_to_date_handles_leap_year_and_invalid_mode():
	assert seasons.season_dekad_to_date(8, target_year=2024, mode="last") == pd.Timestamp(
		"2023-03-31"
	).date()
	with pytest.raises(ValueError, match="mode"):
		seasons.season_dekad_to_date(1, mode="middle")


def test_row_spatial_extent_supports_bbox_schema():
	row = pd.Series({"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4, "epsg": 4326})

	assert seasons._row_spatial_extent_from_grid_row(row) == BoundingBoxExtent(
		west=1, south=2, east=3, north=4, epsg=4326
	)


def test_row_spatial_extent_rejects_incomplete_schema():
	with pytest.raises(ValueError, match="Cannot infer spatial extent"):
		seasons._row_spatial_extent_from_grid_row(pd.Series({"xmin": 1}))


def test_enrich_production_grid_from_crop_calendars(patched_lookup):
	grid = pd.DataFrame(
		{"xmin": [20], "ymin": [10], "xmax": [22], "ymax": [11], "epsg": [4326]}
	)

	enriched = seasons.enrich_production_grid_from_crop_calendars(grid, 2024)

	# Union of both seasons spans 25 months, so it is trimmed symmetrically to 12
	# and the individual seasons are clipped to that period.
	assert enriched.loc[0, "start_date"] == "2024-05-01"
	assert enriched.loc[0, "end_date"] == "2025-04-30"
	assert enriched.loc[0, "season_ids"] == "tc-s1,tc-s2"
	assert json.loads(enriched.loc[0, "season_windows"]) == {
		"tc-s1": ["2024-05-01", "2024-08-31"],
		"tc-s2": ["2025-02-01", "2025-04-30"],
	}


def test_dekad_medoid_index_returns_an_observed_row():
	# Two disagreeing regimes: a short early season and a long late one.
	# Independent medians would yield (16, 30), a 15-dekad season that occurs
	# nowhere in the extent, while both real regimes last 10 resp. 19 dekads.
	values = np.array([[10, 19], [11, 20], [22, 40], [23, 41]])

	assert seasons.dekad_medoid_index(values) == 1


def test_dekad_medoid_index_keeps_year_offsets_distinct():
	# Three seasons in one year and two in the next: a circular metric would
	# collapse both groups, a linear one keeps the majority group.
	values = np.array([[2, 14], [3, 15], [4, 16], [38, 50], [39, 51]])

	assert seasons.dekad_medoid_index(values) == 2


def test_dekad_medoid_index_rejects_empty_input():
	with pytest.raises(ValueError, match="non-empty 2D array"):
		seasons.dekad_medoid_index(np.empty((0, 2)))


@pytest.fixture
def two_regime_lookup(monkeypatch):
	"""Extent holding two seasonality regimes that disagree per season."""

	table = pd.DataFrame(
		{
			"s1_sos_dekad": [10, 11, 12, 40],
			"s1_eos_dekad": [19, 20, 21, 60],
			"s2_sos_dekad": [90, 60, 61, 62],
			"s2_eos_dekad": [99, 70, 71, 72],
		},
		index=pd.MultiIndex.from_tuples(
			[(10.25, 20.25), (10.25, 21.25), (10.25, 22.25), (10.25, 23.25)],
			names=["lat", "lon"],
		),
	)
	monkeypatch.setattr(seasons, "ensure_seasonality_lookup_table", lambda: table)
	return table


def test_fetch_cropcalendar_dekads_extent_selects_seasons_jointly(two_regime_lookup):
	extent = BoundingBoxExtent(west=20, south=10, east=24, north=11, epsg=4326)

	# Taken on its own, season 1 is best represented by the second point ...
	assert seasons.fetch_cropcalendar_dekads_extent(["tc-s1"], extent) == {
		"tc-s1": (11, 20)
	}
	# ... but jointly both seasons come from the third point, so the returned
	# cropping calendar is one that really occurs somewhere in the extent.
	assert seasons.fetch_cropcalendar_dekads_extent(["tc-s1", "tc-s2"], extent) == {
		"tc-s1": (12, 21),
		"tc-s2": (61, 71),
	}


def test_fetch_cropcalendar_dekads_extent_can_reject_heterogeneous_extent(
	two_regime_lookup,
):
	extent = BoundingBoxExtent(west=20, south=10, east=24, north=11, epsg=4326)

	with pytest.raises(ValueError, match="heterogeneous"):
		seasons.fetch_cropcalendar_dekads_extent(
			["tc-s1", "tc-s2"], extent, on_heterogeneity="raise"
		)


def test_fetch_cropcalendar_dekads_extent_drops_nodata_season(two_regime_lookup):
	two_regime_lookup["s2_sos_dekad"] = 0
	extent = BoundingBoxExtent(west=20, south=10, east=24, north=11, epsg=4326)

	assert seasons.fetch_cropcalendar_dekads_extent(["tc-s1", "tc-s2"], extent) == {
		"tc-s1": (11, 20)
	}


def test_enrich_production_grid_rejects_heterogeneous_cell(two_regime_lookup):
	grid = pd.DataFrame(
		{"xmin": [20], "ymin": [10], "xmax": [24], "ymax": [11], "epsg": [4326]}
	)

	with pytest.raises(ValueError, match="heterogeneous"):
		seasons.enrich_production_grid_from_crop_calendars(grid, 2024)


def test_clip_season_windows_to_period_clips_and_drops():
	period = TemporalContext("2024-03-01", "2025-02-28")
	season_windows = {
		"tc-s1": ["2023-11-01", "2024-09-30"],
		"tc-s2": ["2024-04-01", "2024-08-31"],
		"tc-s3": ["2025-03-01", "2025-08-31"],
	}

	assert seasons.clip_season_windows_to_period(season_windows, period) == {
		"tc-s1": ["2024-03-01", "2024-09-30"],
		"tc-s2": ["2024-04-01", "2024-08-31"],
	}


@pytest.mark.parametrize(
	("season_windows", "expected"),
	[
		# Union already exactly 12 months: kept as is.
		(
			{"tc-s1": ["2024-03-01", "2024-08-31"], "tc-s2": ["2024-09-01", "2025-02-28"]},
			("2024-03-01", "2025-02-28"),
		),
		# Shorter union: ends with the latest season and extends backwards.
		(
			{"tc-s1": ["2024-04-01", "2024-10-31"]},
			("2023-11-01", "2024-10-31"),
		),
		# Single season already longer than 12 months: trimmed symmetrically.
		(
			{"tc-s1": ["2024-01-01", "2025-02-28"]},
			("2024-02-01", "2025-01-31"),
		),
	],
)
def test_consolidate_processing_period_always_spans_twelve_months(
	season_windows, expected
):
	context = seasons.consolidate_processing_period(season_windows)

	assert (context.start_date, context.end_date) == expected
	start = pd.Timestamp(context.start_date)
	assert pd.Timestamp(context.end_date) + pd.Timedelta(days=1) == start + pd.DateOffset(
		years=1
	)


def test_consolidate_processing_period_requires_season_windows():
	with pytest.raises(ValueError, match="without any season window"):
		seasons.consolidate_processing_period({})


def test_fetch_cropcalendar_dekads_extent_filters_nodata(patched_lookup):
	patched_lookup.loc[(10.25, 20.25), "s1_sos_dekad"] = 0
	extent = BoundingBoxExtent(west=20, south=10, east=21.5, north=11, epsg=4326)

	assert seasons.fetch_cropcalendar_dekads_extent(["tc-s1"], extent) == {
		"tc-s1": (33, 63)
	}

