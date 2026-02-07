import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr
from prometheo.models import Presto
from prometheo.models.presto.wrapper import load_presto_weights
from prometheo.predictors import DEM_BANDS, METEO_BANDS, NODATAVALUE, S1_BANDS, S2_BANDS

from worldcereal.train.datasets import (
    WorldCerealDataset,
    WorldCerealLabelledDataset,
    WorldCerealTrainingDataset,
    _spatial_bins_from_latlon,
    align_to_composite_window,
    get_class_weights,
    get_dekad_timestamp_components,
    get_monthly_timestamp_components,
    run_model_inference,
)


class TestWorldCerealDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data for datasets tests."""
        # Create a simple dataframe with minimal required columns
        self.num_samples = 5
        self.num_timesteps = 12

        # Create a dataframe with the required columns
        data = {
            "lat": [45.1, 45.2, 45.3, 45.4, 45.5],
            "lon": [5.1, 5.2, 5.3, 5.4, 5.5],
            "start_date": ["2021-01-01"] * self.num_samples,
            "end_date": ["2022-01-01"] * self.num_samples,
            "valid_time": ["2021-07-01"] * self.num_samples,
            "available_timesteps": [self.num_timesteps] * self.num_samples,
            "valid_position": [6] * self.num_samples,  # Middle of the time series
        }

        # Add band data for each timestep
        for ts in range(
            self.num_timesteps + 6
        ):  # 6 extra timesteps for augmentation possibility
            # Add optical bands
            for band_template, band_name in [
                ("OPTICAL-B02-ts{}-10m", "B2"),
                ("OPTICAL-B03-ts{}-10m", "B3"),
                ("OPTICAL-B04-ts{}-10m", "B4"),
                ("OPTICAL-B08-ts{}-10m", "B8"),
                ("OPTICAL-B05-ts{}-20m", "B5"),
                ("OPTICAL-B06-ts{}-20m", "B6"),
                ("OPTICAL-B07-ts{}-20m", "B7"),
                ("OPTICAL-B8A-ts{}-20m", "B8A"),
                ("OPTICAL-B11-ts{}-20m", "B11"),
                ("OPTICAL-B12-ts{}-20m", "B12"),
            ]:
                data[band_template.format(ts)] = [1000 + ts * 10] * self.num_samples

            # Add SAR bands
            for band_template, band_name in [
                ("SAR-VH-ts{}-20m", "VH"),
                ("SAR-VV-ts{}-20m", "VV"),
            ]:
                data[band_template.format(ts)] = [0.01 + ts * 0.001] * self.num_samples

            # Add METEO bands
            for band_template, band_name in [
                ("METEO-precipitation_flux-ts{}-100m", "precipitation"),
                ("METEO-temperature_mean-ts{}-100m", "temperature"),
            ]:
                data[band_template.format(ts)] = [10 + ts] * self.num_samples

        # Add DEM bands (not timestep dependent)
        data["DEM-alt-20m"] = [100] * self.num_samples
        data["DEM-slo-20m"] = [5] * self.num_samples

        self.df = pd.DataFrame(data)

        # Add finetune_class for labelled dataset tests
        self.df["finetune_class"] = [
            "cropland",
            "not_cropland",
            "cropland",
            "cropland",
            "not_cropland",
        ]
        self.df["label_task"] = [
            "landcover",
            "croptype",
            "croptype",
            "landcover",
            "croptype",
        ]
        self.df["landcover_label"] = [
            "lc_a",
            "lc_b",
            "lc_a",
            "lc_b",
            "lc_a",
        ]
        self.df["croptype_label"] = [
            "ct_a",
            "ct_b",
            "ct_a",
            "ct_b",
            "ct_a",
        ]
        self.df["quality_score_lc"] = [0.5, 1.0, 1.0, 2.0, 1.0]
        self.df["quality_score_ct"] = [1.5, 0.5, 2.0, 1.0, 0.8]

        # Initialize the datasets
        self.base_ds = WorldCerealDataset(
            self.df, num_timesteps=self.num_timesteps, augment=True
        )
        self.binary_ds = WorldCerealLabelledDataset(
            self.df, task_type="binary", num_outputs=1
        )
        self.multiclass_ds = WorldCerealLabelledDataset(
            self.df,
            task_type="multiclass",
            num_outputs=4,
            classes_list=["cropland", "not_cropland", "other1", "other2"],
        )
        self.time_explicit_ds = WorldCerealLabelledDataset(
            self.df, task_type="binary", num_outputs=1, time_explicit=True
        )
        self.expected_calendar_masks = np.array(
            [
                [False] * self.num_timesteps,
                [
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                ],
            ],
            dtype=bool,
        )
        self.expected_calendar_in_flags = np.array([False, True], dtype=bool)

    def test_dataset_length(self):
        """Test that dataset length matches dataframe length."""
        self.assertEqual(len(self.base_ds), self.num_samples)
        self.assertEqual(len(self.binary_ds), self.num_samples)
        self.assertEqual(len(self.multiclass_ds), self.num_samples)

    def test_get_timestep_positions(self):
        """Test getting timestep positions works correctly."""
        row = pd.Series.to_dict(self.df.iloc[0, :])
        timestep_positions, valid_position = self.base_ds.get_timestep_positions(row)

        # Check we got the right number of timesteps
        self.assertEqual(len(timestep_positions), self.num_timesteps)

        # Check the valid position is in the timestep positions
        self.assertIn(valid_position, timestep_positions)

        # Test with augmentation - ensure we have enough timesteps for augmentation
        # Modify the row to have a larger number of available timesteps
        augmented_row = row.copy()
        augmented_row["available_timesteps"] = (
            self.num_timesteps + 4
        )  # Increase number of available timesteps

        # Test with augmentation
        timestep_positions, valid_position = self.base_ds.get_timestep_positions(
            augmented_row
        )
        self.assertEqual(len(timestep_positions), self.num_timesteps)
        self.assertIn(valid_position, timestep_positions)

    def test_initialize_inputs(self):
        """Test input initialization creates correct array shapes."""
        s1, s2, meteo, dem = self.base_ds.initialize_inputs()

        # Check shapes
        self.assertEqual(s1.shape, (1, 1, self.num_timesteps, len(S1_BANDS)))
        self.assertEqual(s2.shape, (1, 1, self.num_timesteps, len(S2_BANDS)))
        self.assertEqual(meteo.shape, (1, 1, self.num_timesteps, len(METEO_BANDS)))
        self.assertEqual(dem.shape, (1, 1, len(DEM_BANDS)))

        # Check all initialized with NODATAVALUE
        self.assertTrue(np.all(s1 == NODATAVALUE))
        self.assertTrue(np.all(s2 == NODATAVALUE))
        self.assertTrue(np.all(meteo == NODATAVALUE))
        self.assertTrue(np.all(dem == NODATAVALUE))

    def test_get_inputs(self):
        """Test getting inputs from a row."""
        row = pd.Series.to_dict(self.df.iloc[0, :])
        timestep_positions, _ = self.base_ds.get_timestep_positions(row)

        inputs = self.base_ds.get_inputs(row, timestep_positions)

        # Check all required keys are in the inputs
        self.assertIn("s1", inputs)
        self.assertIn("s2", inputs)
        self.assertIn("meteo", inputs)
        self.assertIn("dem", inputs)
        self.assertIn("latlon", inputs)
        self.assertIn("timestamps", inputs)

        # Check shapes
        self.assertEqual(inputs["s1"].shape, (1, 1, self.num_timesteps, len(S1_BANDS)))
        self.assertEqual(inputs["s2"].shape, (1, 1, self.num_timesteps, len(S2_BANDS)))
        self.assertEqual(
            inputs["meteo"].shape, (1, 1, self.num_timesteps, len(METEO_BANDS))
        )
        self.assertEqual(inputs["dem"].shape, (1, 1, len(DEM_BANDS)))
        self.assertEqual(inputs["latlon"].shape, (1, 1, 2))
        self.assertEqual(inputs["timestamps"].shape, (self.num_timesteps, 3))

        # Check data has been filled in (not all NODATAVALUE)
        self.assertTrue(np.any(inputs["s1"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["s2"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["meteo"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["dem"] != NODATAVALUE))

    def test_task_balanced_sampler_combines_weights(self):
        """Ensure the sampler composes task, class, sample, and spatial weights."""

        class_column_map = {
            "landcover": "landcover_label",
            "croptype": "croptype_label",
        }

        spatial_bin_size = 0.2

        sampler = self.binary_ds.get_task_balanced_sampler(
            class_column_map=class_column_map,
            task_weight_method="balanced",
            class_weight_method="balanced",
            spatial_bin_size_degrees=spatial_bin_size,
            spatial_weight_method="log",
            normalize=True,
        )

        weights = sampler.weights.double().cpu().numpy()
        self.assertEqual(weights.shape[0], self.num_samples)

        df = self.binary_ds.dataframe
        tasks = df["label_task"].astype(str).to_numpy()

        # landcover supervises every sample when croptype labels exist
        effective_task_counts = Counter(tasks)
        if {"landcover", "croptype"}.issubset(effective_task_counts):
            effective_task_counts["landcover"] = len(tasks)
        task_weights = get_class_weights(
            tasks,
            method="balanced",
            normalize=True,
            counts_override=effective_task_counts,
        )
        expected_task = np.array([task_weights[t] for t in tasks], dtype=np.float64)

        expected_class = np.ones_like(expected_task)
        for task_name, column in class_column_map.items():
            mask = tasks == task_name
            if not np.any(mask):
                continue
            class_values = df.loc[mask, column].astype(str).to_numpy()
            class_weights = get_class_weights(
                class_values,
                method="balanced",
                clip_range=(0.1, 10.0),
                normalize=True,
            )
            expected_class[mask] = np.array(
                [class_weights[val] for val in class_values], dtype=np.float64
            )

        spatial_bins = _spatial_bins_from_latlon(df["lat"], df["lon"], spatial_bin_size)
        spatial_weights = get_class_weights(
            spatial_bins,
            method="log",
            clip_range=(0.1, 10.0),
            normalize=True,
        )
        expected_spatial = np.array(
            [spatial_weights[str(bin_id)] for bin_id in spatial_bins], dtype=np.float64
        )

        expected = expected_task * expected_class * expected_spatial

        np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-6)

    def test_getitem(self):
        """Test __getitem__ returns correct type."""
        item = self.base_ds[0]
        self.assertEqual(type(item).__name__, "Predictors")

        # Test labelled dataset
        predictors, attrs = self.binary_ds[0]
        self.assertEqual(type(predictors).__name__, "Predictors")
        self.assertTrue(hasattr(predictors, "label"))
        self.assertIn("season_masks", attrs)
        self.assertEqual(attrs["season_masks"].shape[1], self.num_timesteps)
        self.assertIn("in_seasons", attrs)
        self.assertEqual(attrs["in_seasons"].shape[0], attrs["season_masks"].shape[0])
        self.assertIn("in_seasons", attrs)
        self.assertEqual(attrs["in_seasons"].shape[0], attrs["season_masks"].shape[0])

    def test_labelled_dataset_season_mode_off(self):
        with mock.patch.object(
            WorldCerealLabelledDataset,
            "_compute_season_metadata",
        ) as mocked:
            ds = WorldCerealLabelledDataset(
                self.df,
                task_type="binary",
                num_outputs=1,
                season_calendar_mode="off",
            )
            _, attrs = ds[0]

        mocked.assert_not_called()
        self.assertIsNone(attrs["season_masks"])
        self.assertIsNone(attrs["in_seasons"])

    def test_labelled_dataset_season_calendar_enabled(self):
        mock_masks = np.ones((1, self.num_timesteps), dtype=bool)
        with mock.patch.object(
            WorldCerealLabelledDataset,
            "_compute_season_metadata",
            return_value=(mock_masks, None),
        ) as mocked:
            ds = WorldCerealLabelledDataset(
                self.df,
                task_type="binary",
                num_outputs=1,
                season_calendar_mode="calendar",
            )
            _, attrs = ds[0]

        self.assertTrue(mocked.call_args.kwargs["derive_from_calendar"])
        np.testing.assert_array_equal(attrs["season_masks"], mock_masks)
        self.assertNotIn("in_seasons", attrs)

    def test_labelled_dataset_season_auto_falls_back_to_calendar(self):
        mock_masks = np.ones((1, self.num_timesteps), dtype=bool)
        with mock.patch.object(
            WorldCerealLabelledDataset,
            "_compute_season_metadata",
            return_value=(mock_masks, None),
        ) as mocked:
            ds = WorldCerealLabelledDataset(
                self.df,
                task_type="binary",
                num_outputs=1,
                season_calendar_mode="auto",
            )
            _, attrs = ds[0]

        self.assertTrue(mocked.call_args.kwargs["derive_from_calendar"])
        np.testing.assert_array_equal(attrs["season_masks"], mock_masks)

    def test_labelled_dataset_season_auto_prefers_custom_windows(self):
        mock_masks = np.ones((1, self.num_timesteps), dtype=bool)
        season_windows = {
            "custom": (np.datetime64("2021-05-15"), np.datetime64("2021-08-15"))
        }
        with mock.patch.object(
            WorldCerealLabelledDataset,
            "_compute_season_metadata",
            return_value=(mock_masks, None),
        ) as mocked:
            ds = WorldCerealLabelledDataset(
                self.df,
                task_type="binary",
                num_outputs=1,
                season_calendar_mode="auto",
                season_ids=("custom",),
                season_windows=season_windows,
            )
            _, attrs = ds[0]

        self.assertFalse(mocked.call_args.kwargs["derive_from_calendar"])
        np.testing.assert_array_equal(attrs["season_masks"], mock_masks)

    def test_season_calendar_custom_requires_windows(self):
        with self.assertRaisesRegex(ValueError, "season_calendar_mode='custom'"):
            WorldCerealLabelledDataset(
                self.df,
                task_type="binary",
                num_outputs=1,
                season_calendar_mode="custom",
            )

    def test_training_dataset_season_mode_off(self):
        with mock.patch.object(
            WorldCerealTrainingDataset,
            "_compute_season_metadata",
        ) as mocked:
            ds = WorldCerealTrainingDataset(
                self.df,
                num_timesteps=self.num_timesteps,
                season_calendar_mode="off",
            )
            _, attrs = ds[0]

        mocked.assert_not_called()
        self.assertIsNone(attrs["season_masks"])

    def test_training_dataset_season_calendar_enabled(self):
        mock_masks = np.ones((2, self.num_timesteps), dtype=bool)
        with mock.patch.object(
            WorldCerealTrainingDataset,
            "_compute_season_metadata",
            return_value=(mock_masks, None),
        ) as mocked:
            ds = WorldCerealTrainingDataset(
                self.df,
                num_timesteps=self.num_timesteps,
                season_calendar_mode="calendar",
            )
            _, attrs = ds[0]

        self.assertTrue(mocked.call_args.kwargs["derive_from_calendar"])
        np.testing.assert_array_equal(attrs["season_masks"], mock_masks)

    def test_labelled_dataset_calendar_masks_real_data(self):
        ds = WorldCerealLabelledDataset(
            self.df,
            task_type="binary",
            num_outputs=1,
            season_calendar_mode="calendar",
        )
        _, attrs = ds[0]

        np.testing.assert_array_equal(
            attrs["season_masks"], self.expected_calendar_masks
        )
        np.testing.assert_array_equal(
            attrs["in_seasons"], self.expected_calendar_in_flags
        )

    def test_training_dataset_calendar_masks_real_data(self):
        ds = WorldCerealTrainingDataset(
            self.df,
            num_timesteps=self.num_timesteps,
            season_calendar_mode="calendar",
        )
        _, attrs = ds[0]

        np.testing.assert_array_equal(
            attrs["season_masks"], self.expected_calendar_masks
        )
        np.testing.assert_array_equal(
            attrs["in_seasons"], self.expected_calendar_in_flags
        )

    def test_custom_season_windows(self):
        df = self.df.copy()
        df["valid_time"] = ["2021-04-01"] * self.num_samples

        season_windows = {
            "custom": (np.datetime64("2021-03-15"), np.datetime64("2021-06-15"))
        }

        ds = WorldCerealLabelledDataset(
            df,
            task_type="binary",
            num_outputs=1,
            season_calendar_mode="custom",
            season_ids=("custom",),
            season_windows=season_windows,
        )

        row = pd.Series.to_dict(ds.dataframe.iloc[0, :])
        timestep_positions, _ = ds.get_timestep_positions(row)
        timestamps = ds.get_inputs(row, timestep_positions)["timestamps"]
        composite_dates = np.array(
            [
                np.datetime64(f"{int(year):04d}-{int(month):02d}-{int(day):02d}", "D")
                for day, month, year in timestamps
            ]
        )
        start_aligned = align_to_composite_window(np.datetime64("2021-03-15"), "month")
        end_aligned = align_to_composite_window(np.datetime64("2021-06-15"), "month")
        expected_mask = (composite_dates >= start_aligned) & (
            composite_dates <= end_aligned
        )

        _, attrs = ds[0]
        np.testing.assert_array_equal(attrs["season_masks"][0], expected_mask)
        self.assertTrue(attrs["in_seasons"][0])

    def test_custom_season_windows_different_year(self):
        df = self.df.copy()
        df["start_date"] = ["2023-01-01"] * self.num_samples
        df["end_date"] = ["2024-01-01"] * self.num_samples
        df["valid_time"] = ["2023-04-01"] * self.num_samples

        season_windows = {
            "custom": (np.datetime64("2021-03-15"), np.datetime64("2021-06-15"))
        }

        ds = WorldCerealLabelledDataset(
            df,
            task_type="binary",
            num_outputs=1,
            season_calendar_mode="custom",
            season_ids=("custom",),
            season_windows=season_windows,
        )

        row = pd.Series.to_dict(ds.dataframe.iloc[0, :])
        timestep_positions, _ = ds.get_timestep_positions(row)
        timestamps = ds.get_inputs(row, timestep_positions)["timestamps"]
        composite_dates = np.array(
            [
                np.datetime64(f"{int(year):04d}-{int(month):02d}-{int(day):02d}", "D")
                for day, month, year in timestamps
            ]
        )
        start_aligned = align_to_composite_window(np.datetime64("2023-03-15"), "month")
        end_aligned = align_to_composite_window(np.datetime64("2023-06-15"), "month")
        expected_mask = (composite_dates >= start_aligned) & (
            composite_dates <= end_aligned
        )

        _, attrs = ds[0]
        np.testing.assert_array_equal(attrs["season_masks"][0], expected_mask)
        self.assertTrue(attrs["in_seasons"][0])

    def test_custom_season_windows_cross_year_repeat(self):
        df = self.df.copy()
        df["start_date"] = ["2023-07-01"] * self.num_samples
        df["end_date"] = ["2024-07-01"] * self.num_samples
        df["valid_time"] = ["2023-12-15"] * self.num_samples

        season_windows = {
            "winter": (np.datetime64("2021-10-01"), np.datetime64("2022-02-01"))
        }

        ds = WorldCerealLabelledDataset(
            df,
            task_type="binary",
            num_outputs=1,
            season_calendar_mode="custom",
            season_ids=("winter",),
            season_windows=season_windows,
        )

        row = pd.Series.to_dict(ds.dataframe.iloc[0, :])
        timestep_positions, _ = ds.get_timestep_positions(row)
        timestamps = ds.get_inputs(row, timestep_positions)["timestamps"]
        composite_dates = np.array(
            [
                np.datetime64(f"{int(year):04d}-{int(month):02d}-{int(day):02d}", "D")
                for day, month, year in timestamps
            ]
        )
        months = (composite_dates.astype("datetime64[M]").astype(int) % 12) + 1
        expected_mask = np.isin(months, [10, 11, 12, 1, 2])

        _, attrs = ds[0]
        np.testing.assert_array_equal(attrs["season_masks"][0], expected_mask)
        self.assertTrue(attrs["in_seasons"][0])

    def test_training_dataset_custom_windows_without_valid_time(self):
        df = self.df.copy()
        df["valid_time"] = [np.nan] * self.num_samples

        season_windows = {
            "custom": (np.datetime64("2021-03-15"), np.datetime64("2021-06-15"))
        }

        ds = WorldCerealTrainingDataset(
            df,
            num_timesteps=self.num_timesteps,
            season_windows=season_windows,
            season_ids=("custom",),
            season_calendar_mode="custom",
        )

        row = pd.Series.to_dict(ds.dataframe.iloc[0, :])
        timestep_positions, _ = ds.get_timestep_positions(row)
        timestamps = ds.get_inputs(row, timestep_positions)["timestamps"]
        composite_dates = np.array(
            [
                np.datetime64(f"{int(year):04d}-{int(month):02d}-{int(day):02d}", "D")
                for day, month, year in timestamps
            ]
        )
        start_aligned = align_to_composite_window(np.datetime64("2021-03-15"), "month")
        end_aligned = align_to_composite_window(np.datetime64("2021-06-15"), "month")
        expected_mask = (composite_dates >= start_aligned) & (
            composite_dates <= end_aligned
        )

        _, attrs = ds[0]
        np.testing.assert_array_equal(attrs["season_masks"][0], expected_mask)
        self.assertNotIn("in_seasons", attrs)

    def test_calendar_seasons_require_label_datetime(self):
        df = self.df.copy()
        df["valid_time"] = [np.nan] * self.num_samples

        ds = WorldCerealTrainingDataset(
            df,
            num_timesteps=self.num_timesteps,
            season_calendar_mode="calendar",
        )

        with self.assertRaisesRegex(ValueError, "requires a label datetime"):
            ds[0]

    def test_calendar_seasons_require_lat_lon(self):
        df = self.df.copy()
        df["lat"] = [np.nan] * self.num_samples

        ds = WorldCerealTrainingDataset(
            df,
            num_timesteps=self.num_timesteps,
            season_calendar_mode="calendar",
        )

        with self.assertRaisesRegex(ValueError, "lat/lon"):
            ds[0]

    def test_manual_mode_rejects_missing_calendar_support(self):
        season_windows = {
            "custom": (np.datetime64("2021-05-15"), np.datetime64("2021-08-15"))
        }

        ds = WorldCerealLabelledDataset(
            self.df,
            task_type="binary",
            num_outputs=1,
            season_calendar_mode="auto",
            season_ids=("custom", "tc-s1"),
            season_windows=season_windows,
        )

        with self.assertRaisesRegex(ValueError, "manual windows"):
            ds[0]

    def test_custom_season_windows_require_full_coverage(self):
        df = self.df.copy()
        df["valid_time"] = ["2021-04-01"] * self.num_samples

        season_windows = {
            "spring": (np.datetime64("2021-03-15"), np.datetime64("2021-06-15"))
        }

        ds = WorldCerealLabelledDataset(
            df,
            task_type="binary",
            num_outputs=1,
            season_calendar_mode="custom",
            season_ids=("spring",),
            season_windows=season_windows,
            num_timesteps=4,  # Force insufficient coverage
        )

        _, attrs = ds[0]
        self.assertFalse(attrs["season_masks"][0].any())
        self.assertFalse(attrs["in_seasons"][0])

    def test_calendar_seasons_require_full_coverage(self):
        df = self.df.copy()
        df["valid_time"] = ["2021-04-01"] * self.num_samples

        ds = WorldCerealLabelledDataset(
            df,
            task_type="binary",
            num_outputs=1,
            season_calendar_mode="calendar",
            season_ids=("tc-s1",),
            num_timesteps=4,  # Force insufficient coverage
        )

        fake_window = (np.datetime64("2021-03-01"), np.datetime64("2021-06-01"))
        with mock.patch.object(
            WorldCerealLabelledDataset,
            "_season_context_for",
            return_value=fake_window,
        ):
            # Mock the season window to ensure coverage would be sufficient if all timesteps were present
            _, attrs = ds[0]

        self.assertFalse(attrs["season_masks"][0].any())
        self.assertFalse(attrs["in_seasons"][0])

    def test_binary_label(self):
        """Test binary labelled dataset returns correct labels."""
        item, _ = self.binary_ds[0]
        # Cropland should be mapped to 1 (positive class)
        self.assertEqual(item.label[0, 0, 0, 0], 1)

        item, _ = self.binary_ds[1]
        # not_cropland should be mapped to 0 (negative class)
        self.assertEqual(item.label[0, 0, 0, 0], 0)

    def test_multiclass_label(self):
        """Test multiclass labelled dataset returns correct labels."""
        item, _ = self.multiclass_ds[0]
        # First sample should have class index 0
        self.assertEqual(item.label[0, 0, 0, 0], 0)

        item, _ = self.multiclass_ds[4]
        # Fifth sample should have class index 1
        self.assertEqual(item.label[0, 0, 0, 0], 1)

    def test_time_explicit_label(self):
        """Test time explicit labelled dataset returns correct label shape."""
        item, attrs = self.time_explicit_ds[0]
        # Label should have temporal dimension
        self.assertEqual(item.label.shape, (1, 1, self.num_timesteps, 1))
        self.assertIn("season_masks", attrs)
        self.assertEqual(attrs["season_masks"].shape[1], self.num_timesteps)

        # For time_explicit, valid_position should have a value, other positions should be NODATAVALUE
        row = pd.Series.to_dict(self.df.iloc[0, :])
        _, valid_position = self.time_explicit_ds.get_timestep_positions(row)

        # Check only one position has a value
        valid_values = (item.label != NODATAVALUE).sum()
        self.assertEqual(valid_values, 1)

    def test_get_timestamps(self):
        row = pd.Series.to_dict(self.base_ds.dataframe.iloc[0, :])
        ref_timestamps = np.array(
            (
                [
                    [1, 1, 2021],
                    [1, 2, 2021],
                    [1, 3, 2021],
                    [1, 4, 2021],
                    [1, 5, 2021],
                    [1, 6, 2021],
                    [1, 7, 2021],
                    [1, 8, 2021],
                    [1, 9, 2021],
                    [1, 10, 2021],
                    [1, 11, 2021],
                    [1, 12, 2021],
                ]
            )
        )

        computed_timestamps = self.base_ds._get_timestamps(
            row, self.base_ds.get_timestep_positions(row)[0]
        )

        np.testing.assert_array_equal(computed_timestamps, ref_timestamps)


class TestWorldCerealDekadalDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data for dekadal datasets tests."""
        # Create a simple dataframe with minimal required columns
        self.num_samples = 5
        self.num_timesteps = 36

        # Create a dataframe with the required columns
        data = {
            "lat": [45.1, 45.2, 45.3, 45.4, 45.5],
            "lon": [5.1, 5.2, 5.3, 5.4, 5.5],
            "start_date": ["2022-07-17"] * self.num_samples,
            "end_date": ["2023-09-28"] * self.num_samples,
            "valid_time": ["2023-04-01"] * self.num_samples,
            "available_timesteps": [self.num_timesteps] * self.num_samples,
            "valid_position": [18] * self.num_samples,  # Middle of the time series
        }

        # Add band data for each timestep
        for ts in range(
            self.num_timesteps + 18
        ):  # 18 extra timesteps for augmentation possibility
            # Add optical bands
            for band_template, band_name in [
                ("OPTICAL-B02-ts{}-10m", "B2"),
                ("OPTICAL-B03-ts{}-10m", "B3"),
                ("OPTICAL-B04-ts{}-10m", "B4"),
                ("OPTICAL-B08-ts{}-10m", "B8"),
                ("OPTICAL-B05-ts{}-20m", "B5"),
                ("OPTICAL-B06-ts{}-20m", "B6"),
                ("OPTICAL-B07-ts{}-20m", "B7"),
                ("OPTICAL-B8A-ts{}-20m", "B8A"),
                ("OPTICAL-B11-ts{}-20m", "B11"),
                ("OPTICAL-B12-ts{}-20m", "B12"),
            ]:
                data[band_template.format(ts)] = [1000 + ts * 10] * self.num_samples

            # Add SAR bands
            for band_template, band_name in [
                ("SAR-VH-ts{}-20m", "VH"),
                ("SAR-VV-ts{}-20m", "VV"),
            ]:
                data[band_template.format(ts)] = [0.01 + ts * 0.001] * self.num_samples

            # Add METEO bands
            for band_template, band_name in [
                ("METEO-precipitation_flux-ts{}-100m", "precipitation"),
                ("METEO-temperature_mean-ts{}-100m", "temperature"),
            ]:
                data[band_template.format(ts)] = [10 + ts] * self.num_samples

        # Add DEM bands (not timestep dependent)
        data["DEM-alt-20m"] = [100] * self.num_samples
        data["DEM-slo-20m"] = [5] * self.num_samples

        self.df = pd.DataFrame(data)

        # Add finetune_class for labelled dataset tests
        self.df["finetune_class"] = [
            "cropland",
            "not_cropland",
            "cropland",
            "cropland",
            "not_cropland",
        ]

        # Initialize the datasets
        self.base_ds = WorldCerealDataset(
            self.df,
            num_timesteps=self.num_timesteps,
            timestep_freq="dekad",
            augment=True,
        )
        self.binary_ds = WorldCerealLabelledDataset(
            self.df,
            task_type="binary",
            num_outputs=1,
            timestep_freq="dekad",
            num_timesteps=self.num_timesteps,
        )
        self.multiclass_ds = WorldCerealLabelledDataset(
            self.df,
            task_type="multiclass",
            num_outputs=4,
            classes_list=["cropland", "not_cropland", "other1", "other2"],
            timestep_freq="dekad",
            num_timesteps=self.num_timesteps,
        )
        self.time_explicit_ds = WorldCerealLabelledDataset(
            self.df,
            task_type="binary",
            num_outputs=1,
            time_explicit=True,
            timestep_freq="dekad",
            num_timesteps=self.num_timesteps,
        )

    def test_dataset_length(self):
        """Test that dataset length matches dataframe length."""
        self.assertEqual(len(self.base_ds), self.num_samples)
        self.assertEqual(len(self.binary_ds), self.num_samples)
        self.assertEqual(len(self.multiclass_ds), self.num_samples)

    def test_get_timestep_positions(self):
        """Test getting timestep positions works correctly."""
        row = pd.Series.to_dict(self.df.iloc[0, :])
        timestep_positions, valid_position = self.base_ds.get_timestep_positions(row)

        # Check we got the right number of timesteps
        self.assertEqual(len(timestep_positions), self.num_timesteps)

        # Check the valid position is in the timestep positions
        self.assertIn(valid_position, timestep_positions)

        # Test with augmentation - ensure we have enough timesteps for augmentation
        # Modify the row to have a larger number of available timesteps
        augmented_row = row.copy()
        augmented_row["available_timesteps"] = (
            self.num_timesteps + 4
        )  # Increase number of available timesteps

        # Test with augmentation
        timestep_positions, valid_position = self.base_ds.get_timestep_positions(
            augmented_row
        )
        self.assertEqual(len(timestep_positions), self.num_timesteps)
        self.assertIn(valid_position, timestep_positions)

    def test_initialize_inputs(self):
        """Test input initialization creates correct array shapes."""
        s1, s2, meteo, dem = self.base_ds.initialize_inputs()

        # Check shapes
        self.assertEqual(s1.shape, (1, 1, self.num_timesteps, len(S1_BANDS)))
        self.assertEqual(s2.shape, (1, 1, self.num_timesteps, len(S2_BANDS)))
        self.assertEqual(meteo.shape, (1, 1, self.num_timesteps, len(METEO_BANDS)))
        self.assertEqual(dem.shape, (1, 1, len(DEM_BANDS)))

        # Check all initialized with NODATAVALUE
        self.assertTrue(np.all(s1 == NODATAVALUE))
        self.assertTrue(np.all(s2 == NODATAVALUE))
        self.assertTrue(np.all(meteo == NODATAVALUE))
        self.assertTrue(np.all(dem == NODATAVALUE))

    def test_get_inputs(self):
        """Test getting inputs from a row."""
        row = pd.Series.to_dict(self.df.iloc[0, :])
        timestep_positions, _ = self.base_ds.get_timestep_positions(row)

        inputs = self.base_ds.get_inputs(row, timestep_positions)

        # Check all required keys are in the inputs
        self.assertIn("s1", inputs)
        self.assertIn("s2", inputs)
        self.assertIn("meteo", inputs)
        self.assertIn("dem", inputs)
        self.assertIn("latlon", inputs)
        self.assertIn("timestamps", inputs)

        # Check shapes
        self.assertEqual(inputs["s1"].shape, (1, 1, self.num_timesteps, len(S1_BANDS)))
        self.assertEqual(inputs["s2"].shape, (1, 1, self.num_timesteps, len(S2_BANDS)))
        self.assertEqual(
            inputs["meteo"].shape, (1, 1, self.num_timesteps, len(METEO_BANDS))
        )
        self.assertEqual(inputs["dem"].shape, (1, 1, len(DEM_BANDS)))
        self.assertEqual(inputs["latlon"].shape, (1, 1, 2))
        self.assertEqual(inputs["timestamps"].shape, (self.num_timesteps, 3))

        # Check data has been filled in (not all NODATAVALUE)
        self.assertTrue(np.any(inputs["s1"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["s2"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["meteo"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["dem"] != NODATAVALUE))

    def test_getitem(self):
        """Test __getitem__ returns correct type."""
        item = self.base_ds[0]
        self.assertEqual(type(item).__name__, "Predictors")

        # Test labelled dataset
        item, attrs = self.binary_ds[0]
        self.assertEqual(type(item).__name__, "Predictors")
        self.assertTrue(hasattr(item, "label"))
        self.assertIn("season_masks", attrs)
        self.assertEqual(attrs["season_masks"].shape[1], self.num_timesteps)

    def test_binary_label(self):
        """Test binary labelled dataset returns correct labels."""
        item, _ = self.binary_ds[0]
        # Cropland should be mapped to 1 (positive class)
        self.assertEqual(item.label[0, 0, 0, 0], 1)

        item, _ = self.binary_ds[1]
        # not_cropland should be mapped to 0 (negative class)
        self.assertEqual(item.label[0, 0, 0, 0], 0)

    def test_multiclass_label(self):
        """Test multiclass labelled dataset returns correct labels."""
        item, _ = self.multiclass_ds[0]
        # First sample should have class index 0
        self.assertEqual(item.label[0, 0, 0, 0], 0)

        item, _ = self.multiclass_ds[4]
        # Fifth sample should have class index 1
        self.assertEqual(item.label[0, 0, 0, 0], 1)

    def test_time_explicit_label(self):
        """Test time explicit labelled dataset returns correct label shape."""
        item, attrs = self.time_explicit_ds[0]
        # Label should have temporal dimension
        self.assertEqual(item.label.shape, (1, 1, self.num_timesteps, 1))
        self.assertIn("season_masks", attrs)
        self.assertEqual(attrs["season_masks"].shape[1], self.num_timesteps)

        # For time_explicit, valid_position should have a value, other positions should be NODATAVALUE
        row = pd.Series.to_dict(self.df.iloc[0, :])
        _, valid_position = self.time_explicit_ds.get_timestep_positions(row)

        # Check only one position has a value
        valid_values = (item.label != NODATAVALUE).sum()
        self.assertEqual(valid_values, 1)

    def test_get_timestamps(self):
        row = pd.Series.to_dict(self.base_ds.dataframe.iloc[0, :])
        ref_timestamps = np.array(
            [
                [11, 7, 2022],
                [21, 7, 2022],
                [1, 8, 2022],
                [11, 8, 2022],
                [21, 8, 2022],
                [1, 9, 2022],
                [11, 9, 2022],
                [21, 9, 2022],
                [1, 10, 2022],
                [11, 10, 2022],
                [21, 10, 2022],
                [1, 11, 2022],
                [11, 11, 2022],
                [21, 11, 2022],
                [1, 12, 2022],
                [11, 12, 2022],
                [21, 12, 2022],
                [1, 1, 2023],
                [11, 1, 2023],
                [21, 1, 2023],
                [1, 2, 2023],
                [11, 2, 2023],
                [21, 2, 2023],
                [1, 3, 2023],
                [11, 3, 2023],
                [21, 3, 2023],
                [1, 4, 2023],
                [11, 4, 2023],
                [21, 4, 2023],
                [1, 5, 2023],
                [11, 5, 2023],
                [21, 5, 2023],
                [1, 6, 2023],
                [11, 6, 2023],
                [21, 6, 2023],
                [1, 7, 2023],
            ]
        )

        computed_timestamps = self.base_ds._get_timestamps(
            row, self.base_ds.get_timestep_positions(row)[0]
        )

        np.testing.assert_array_equal(computed_timestamps, ref_timestamps)


class TestTimeUtilities(unittest.TestCase):
    def test_align_to_composite_window(self):
        """Test aligning dates to composite window."""
        # Test with dekad frequency
        start_date = np.datetime64("2021-01-03", "D")
        end_date = np.datetime64("2021-01-24", "D")
        aligned_start = align_to_composite_window(start_date, "dekad")
        aligned_end = align_to_composite_window(end_date, "dekad")

        # Should align to first dekad of January
        self.assertEqual(aligned_start, np.datetime64("2021-01-01", "D"))
        self.assertEqual(aligned_end, np.datetime64("2021-01-21", "D"))

        # Test with monthly frequency
        start_date = np.datetime64("2021-01-15", "D")
        end_date = np.datetime64("2021-02-10", "D")
        aligned_start = align_to_composite_window(start_date, "month")
        aligned_end = align_to_composite_window(end_date, "month")

        # Should align to first day of month
        self.assertEqual(aligned_start, np.datetime64("2021-01-01", "D"))
        self.assertEqual(aligned_end, np.datetime64("2021-02-01", "D"))

    def test_get_monthly_timestamp_components(self):
        """Test getting month timestamp components."""
        start_date = np.datetime64("2021-01-03", "D")
        end_date = np.datetime64("2021-12-24", "D")

        days, months, years = get_monthly_timestamp_components(start_date, end_date)

        # Should have 12 months
        self.assertEqual(len(days), 12)
        self.assertEqual(len(months), 12)
        self.assertEqual(len(years), 12)

        # All days should be 1 (first day of month)
        self.assertTrue(np.all(days == 1))

        # Months should be 1-12
        self.assertTrue(np.all(months == np.arange(1, 13)))

        # All years should be 2021
        self.assertTrue(np.all(years == 2021))

    def test_get_dekad_timestamp_components(self):
        """Test getting dekad timestamp components."""
        start_date = np.datetime64("2021-01-03", "D")
        end_date = np.datetime64("2021-01-24", "D")

        days, months, years = get_dekad_timestamp_components(start_date, end_date)

        # Should have 3 dekads per month
        self.assertEqual(len(days), 3)

        # Days should be 1, 11, 21 for first month
        self.assertTrue(np.all(days == np.array([1, 11, 21])))

        # All months should be 1 (January)
        self.assertTrue(np.all(months == 1))

        # All years should be 2021
        self.assertTrue(np.all(years == 2021))


class TestGetLabel(unittest.TestCase):
    def setUp(self):
        # one‐row dataframes to drive get_label directly
        self.df_bin = pd.DataFrame([{"finetune_class": "cropland"}])
        self.df_bin_neg = pd.DataFrame([{"finetune_class": "not_cropland"}])
        self.df_multi = pd.DataFrame(
            [
                {"finetune_class": "cropland"},
                {"finetune_class": "shrubland"},
                {"finetune_class": "trees"},
            ]
        )
        # common args
        self.classes = ["cropland", "shrubland", "trees", "other"]

    def test_non_time_explicit_binary(self):
        # time_explicit=False → label only at t=0, shape T→1
        ds = WorldCerealLabelledDataset(
            self.df_bin,
            task_type="binary",
            num_outputs=1,
            time_explicit=False,
            num_timesteps=5,
        )
        row = self.df_bin.iloc[0].to_dict()
        lbl = ds.get_label(
            row, task_type="binary", classes_list=None, valid_position=None
        )
        # shape should be (1,1,1,1)
        self.assertEqual(lbl.shape, (1, 1, 1, 1))
        # positive class → 1
        self.assertEqual(int(lbl[0, 0, 0, 0]), 1)

        # negative
        dsn = WorldCerealLabelledDataset(
            self.df_bin_neg,
            task_type="binary",
            num_outputs=1,
            time_explicit=False,
            num_timesteps=5,
        )
        rown = self.df_bin_neg.iloc[0].to_dict()
        lbln = dsn.get_label(
            rown, task_type="binary", classes_list=None, valid_position=None
        )
        self.assertEqual(int(lbln[0, 0, 0, 0]), 0)

    def test_time_explicit_all_positions(self):
        # valid_position=None and time_explicit=True → label at every t
        ds = WorldCerealLabelledDataset(
            self.df_bin,
            task_type="binary",
            num_outputs=1,
            time_explicit=True,
            num_timesteps=4,
        )
        row = self.df_bin.iloc[0].to_dict()
        lbl = ds.get_label(
            row, task_type="binary", classes_list=None, valid_position=None
        )
        # shape (1,1,4,1) and all entries ==1
        self.assertEqual(lbl.shape, (1, 1, 4, 1))
        self.assertTrue((lbl[..., 0] == 1).all())

    def test_time_explicit_window(self):
        # window expansion around valid_position
        ds = WorldCerealLabelledDataset(
            self.df_bin,
            task_type="binary",
            num_outputs=1,
            time_explicit=True,
            num_timesteps=7,
            label_jitter=0,
            label_window=1,
        )
        row = self.df_bin.iloc[0].to_dict()
        # pick valid_position=3 → window [2,3,4]
        lbl = ds.get_label(row, task_type="binary", classes_list=None, valid_position=3)
        self.assertEqual(lbl.shape, (1, 1, 7, 1))
        # exactly three non‐NODATAVALUE entries
        non_na = np.nonzero(lbl[0, 0, :, 0] != NODATAVALUE)[0]
        self.assertListEqual(non_na.tolist(), [2, 3, 4])
        # values all ==1
        self.assertTrue((lbl[0, 0, non_na, 0] == 1).all())

    def test_time_explicit_multiclass(self):
        # multiclass label at a single position
        df = self.df_multi.iloc[[1]].copy()
        ds = WorldCerealLabelledDataset(
            df,
            task_type="multiclass",
            num_outputs=3,
            classes_list=self.classes,
            time_explicit=True,
            num_timesteps=5,
        )
        row = df.iloc[0].to_dict()
        # call at t=4
        lbl = ds.get_label(
            row, task_type="multiclass", classes_list=self.classes, valid_position=4
        )
        self.assertEqual(lbl.shape, (1, 1, 5, 1))
        # only index 4 is set, others are NODATAVALUE
        mask = lbl[0, 0, :, 0]
        # check that all except row 4 are NODATAVALUE
        for t in [0, 1, 2, 3]:
            self.assertTrue((mask[t] == NODATAVALUE).all())
        # at t=4, should match class index 1
        self.assertEqual(mask[4], 1)


class TestInference(unittest.TestCase):
    def test_run_model_inference(self):
        """Test the run_model_inference function. Based on reference features
        generated using the following code:

        arr = xr.open_dataarray(data_dir / "test_inference_array.nc")
        model_url = str(data_dir / "finetuned_presto_model.pt")
                presto_features = run_model_inference(
                    arr, model_url, batch_size=512, epsg=32631
                )
        presto_features.to_netcdf(data_dir / "test_presto_inference_features.nc")

        """
        data_dir = Path(__file__).parent / "testresources"
        arr = xr.open_dataarray(data_dir / "test_inference_array.nc")

        # Load a pretrained Presto model
        model_url = str(data_dir / "finetuned_presto_model.pt")
        presto_model = Presto()
        presto_model = load_presto_weights(presto_model, model_url)

        presto_features = run_model_inference(
            arr, presto_model, batch_size=512, epsg=32631
        )

        # Uncomment to regenerate ref features
        # presto_features.to_netcdf(data_dir / "test_presto_inference_features.nc")

        # Load ref features
        ref_presto_features = xr.open_dataarray(
            data_dir / "test_presto_inference_features.nc"
        )

        xr.testing.assert_allclose(
            presto_features,
            ref_presto_features,
            rtol=1e-04,
            atol=1e-04,
        )

        assert presto_features.dims == ref_presto_features.dims


class TestRepeatHandling(unittest.TestCase):
    """Tests for repeat logic in WorldCerealLabelledDataset and WorldCerealTrainingDataset.

    These were previously in a standalone file but are integrated here for cohesion.
    """

    def setUp(self):
        # Minimal dataframes (no band columns needed because we don't call __getitem__ which populates predictors)
        self.df_labelled = pd.DataFrame(
            [
                {
                    "lat": 45.0,
                    "lon": 5.0,
                    "start_date": "2021-01-01",
                    "end_date": "2021-12-31",
                    "available_timesteps": 12,
                    "valid_position": 6,
                    "finetune_class": "cropland",
                },
                {
                    "lat": 45.1,
                    "lon": 5.1,
                    "start_date": "2021-01-01",
                    "end_date": "2021-12-31",
                    "available_timesteps": 12,
                    "valid_position": 6,
                    "finetune_class": "not_cropland",
                },
                {
                    "lat": 45.2,
                    "lon": 5.2,
                    "start_date": "2021-01-01",
                    "end_date": "2021-12-31",
                    "available_timesteps": 12,
                    "valid_position": 6,
                    "finetune_class": "cropland",
                },
            ]
        )

        self.df_training = pd.DataFrame(
            [
                {
                    "lat": 46.0,
                    "lon": 6.0,
                    "start_date": "2021-01-01",
                    "end_date": "2021-12-31",
                    "available_timesteps": 12,
                    "valid_position": 6,
                },
                {
                    "lat": 46.1,
                    "lon": 6.1,
                    "start_date": "2021-01-01",
                    "end_date": "2021-12-31",
                    "available_timesteps": 12,
                    "valid_position": 6,
                },
            ]
        )

    def test_training_repeats_without_augmentation(self):
        ds = WorldCerealTrainingDataset(
            self.df_training,
            num_timesteps=12,
            repeats=1,
            augment=False,
            task_type="multiclass",
        )
        self.assertEqual(ds._repeats, 1)
        self.assertEqual(len(ds), len(self.df_training))
        self.assertListEqual(ds.indices, list(range(len(self.df_training))))

    def test_training_repeats_with_augmentation(self):
        ds = WorldCerealTrainingDataset(
            self.df_training,
            num_timesteps=12,
            repeats=4,
            augment=True,
            task_type="multiclass",
        )
        self.assertEqual(ds._repeats, 4)
        self.assertEqual(len(ds), len(self.df_training) * 4)
        expected = list(range(len(self.df_training))) * 4
        self.assertListEqual(ds.indices, expected)


if __name__ == "__main__":
    unittest.main()
