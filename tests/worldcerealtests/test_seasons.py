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
	("season", "parameter", "expected"),
	[
		("tc-s1", "doy", ("s1_sos_doy", "s1_eos_doy")),
		("tc-s2", "dekad", ("s2_sos_dekad", "s2_eos_dekad")),
		("tc-annual", "doy", ("annual_sos_doy", "annual_eos_doy")),
	],
)
def test_resolve_cropcalendar_columns(season, parameter, expected):
	assert seasons.resolve_cropcalendar_columns(season, parameter) == expected


def test_resolve_cropcalendar_columns_rejects_unknown_values():
	with pytest.raises(ValueError, match="parameter"):
		seasons.resolve_cropcalendar_columns("tc-s1", "month")
	with pytest.raises(ValueError, match="not available"):
		seasons.resolve_cropcalendar_columns("tc-unknown", "doy")


def test_fetch_cropcalendar_points_uses_snapped_cell(patched_lookup):
	assert seasons.fetch_cropcalendar_doy_point("tc-s1", 10.4, 20.4) == (100, 200)
	assert seasons.fetch_cropcalendar_dekad_point("tc-s2", 10.4, 20.4) == (75, 105)


def test_fetch_cropcalendar_point_falls_back_to_nearest_cell(patched_lookup, caplog):
	result = seasons.fetch_cropcalendar_doy_point("tc-s1", 12, 22)

	assert result == (110, 210)


def test_fetch_cropcalendar_point_can_disable_nearest_fallback(patched_lookup):
	with pytest.raises(ValueError, match="No seasonality record"):
		seasons.fetch_cropcalendar_doy_point(
			"tc-s1", 12, 22, fallback_to_nearest=False
		)


@pytest.mark.parametrize(
	("function", "column", "invalid_value", "limit"),
	[
		(seasons.fetch_cropcalendar_doy_point, "s1_sos_doy", 367, "366"),
		(seasons.fetch_cropcalendar_dekad_point, "s1_sos_dekad", 109, "108"),
	],
)
def test_fetch_cropcalendar_point_rejects_invalid_values(
	patched_lookup, function, column, invalid_value, limit
):
	patched_lookup.loc[(10.25, 20.25), column] = invalid_value

	with pytest.raises(ValueError, match=f"Valid .* range is 1-{limit}"):
		function("tc-s1", 10.4, 20.4)


def test_fetch_cropcalendar_dekad_extent_returns_medians_and_values(patched_lookup):
	extent = BoundingBoxExtent(west=20, south=10, east=22, north=11, epsg=4326)

	assert seasons.fetch_cropcalendar_dekad_extent("tc-s1", extent) == (32, 62)


def test_fetch_cropcalendar_dekad_extent_handles_dateline(patched_lookup):
	extent = BoundingBoxExtent(west=179, south=11, east=-179, north=12, epsg=4326)

	assert seasons.fetch_cropcalendar_dekad_extent("tc-s1", extent) == (38, 68)


def test_fetch_cropcalendar_dekad_extent_falls_back_to_centroid(patched_lookup):
	extent = BoundingBoxExtent(west=30, south=30, east=31, north=31, epsg=4326)

	assert seasons.fetch_cropcalendar_dekad_extent("tc-s1", extent) == (33, 63)


def test_get_season_dates_for_extent_returns_temporal_context(patched_lookup):
	extent = BoundingBoxExtent(west=20, south=10, east=22, north=11, epsg=4326)

	assert seasons.get_season_dates_for_extent(extent, 2024, "tc-s1") == TemporalContext(
		"2023-11-01", "2024-09-30"
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


@pytest.mark.parametrize(
	("sos", "eos", "expected_start", "expected_end"),
	[
		(100, 200, "2020-04-10", "2020-07-19"),
		(300, 100, "2019-10-28", "2020-04-10"),
	],
)
def test_season_doys_to_dates_refyear(sos, eos, expected_start, expected_end):
	start, end = seasons.season_doys_to_dates_refyear(sos, eos, 2020)

	assert start.strftime("%Y-%m-%d") == expected_start
	assert end.strftime("%Y-%m-%d") == expected_end


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

	assert enriched.loc[0, "start_date"] == "2023-05-01"
	assert enriched.loc[0, "end_date"] == "2025-07-31"
	assert enriched.loc[0, "season_ids"] == "tc-s1,tc-s2"
	assert json.loads(enriched.loc[0, "season_windows"]) == {
		"tc-s1": ["2023-11-01", "2024-09-30"],
		"tc-s2": ["2025-02-01", "2025-11-30"],
	}


def test_collect_cropcalendar_values_filters_nodata(patched_lookup):
	patched_lookup.loc[(10.25, 20.25), "s1_sos_dekad"] = 0
	extent = BoundingBoxExtent(west=20, south=10, east=21.5, north=11, epsg=4326)

	sos, eos = seasons._collect_cropcalendar_dekad_extent_values("tc-s1", extent)

	np.testing.assert_array_equal(sos, np.array([33]))
	np.testing.assert_array_equal(eos, np.array([63]))

