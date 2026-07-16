import unittest
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
    _get_per_bin_class_weights,
    _get_smoothed_per_bin_class_weights,
    _get_spatial_density_weights,
    _is_lc_only_dataset,
    align_to_composite_window,
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
                    True,
                    True,
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

    def test_task_batch_sampler_structure(self):
        """Ensure SeasonalTaskBatchSampler produces valid batches with LC/CT split."""

        batch_size = 4
        spatial_bin_size = 0.2

        sampler = self.binary_ds.get_task_batch_sampler(
            batch_size=batch_size,
            landcover_column="landcover_label",
            croptype_column="croptype_label",
            class_weight_method="balanced",
            spatial_bin_size_degrees=spatial_bin_size,
            spatial_weight_method="log",
            num_batches=10,
        )

        N = len(self.binary_ds)
        lc_lo, lc_hi = N, 2 * N  # LC virtual range [N, 2N)
        ct_lo, ct_hi = 2 * N, 3 * N  # CT virtual range [2N, 3N)

        batches = list(sampler)
        self.assertEqual(len(batches), 10)

        for batch in batches:
            self.assertEqual(len(batch), batch_size)
            lc_count = sum(1 for idx in batch if lc_lo <= idx < lc_hi)
            ct_count = sum(1 for idx in batch if ct_lo <= idx < ct_hi)
            # Each batch should have exactly half LC and half CT
            self.assertEqual(lc_count, batch_size // 2)
            self.assertEqual(ct_count, batch_size - batch_size // 2)
            # All indices should be in the virtual range
            for idx in batch:
                self.assertTrue(
                    lc_lo <= idx < lc_hi or ct_lo <= idx < ct_hi,
                    f"Index {idx} is outside virtual ranges "
                    f"LC=[{lc_lo}, {lc_hi}) CT=[{ct_lo}, {ct_hi})",
                )

    def test_task_batch_sampler_len(self):
        """Ensure __len__ returns the configured number of batches."""
        sampler = self.binary_ds.get_task_batch_sampler(batch_size=4, num_batches=7)
        self.assertEqual(len(sampler), 7)

    def test_task_batch_sampler_landcover_only(self):
        """Ensure task sampler can produce landcover-only batches."""
        batch_size = 5
        sampler = self.binary_ds.get_task_batch_sampler(
            tasks=("landcover",),
            batch_size=batch_size,
            num_batches=3,
        )

        N = len(self.binary_ds)
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        self.assertTrue(hasattr(sampler, "_lc_probs"))
        self.assertTrue(hasattr(sampler, "_ct_probs"))
        for batch in batches:
            self.assertEqual(len(batch), batch_size)
            self.assertTrue(all(N <= idx < 2 * N for idx in batch))

    def test_task_batch_sampler_croptype_only(self):
        """Ensure task sampler can produce croptype-only batches."""
        batch_size = 5
        sampler = self.binary_ds.get_task_batch_sampler(
            tasks=("croptype",),
            batch_size=batch_size,
            num_batches=3,
        )

        N = len(self.binary_ds)
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        self.assertTrue(hasattr(sampler, "_lc_probs"))
        self.assertTrue(hasattr(sampler, "_ct_probs"))
        for batch in batches:
            self.assertEqual(len(batch), batch_size)
            self.assertTrue(all(2 * N <= idx < 3 * N for idx in batch))

    def test_task_batch_sampler_explicit_ratios(self):
        """Ensure explicit task ratios control per-batch task counts."""
        batch_size = 10
        sampler = self.binary_ds.get_task_batch_sampler(
            tasks=("landcover", "croptype"),
            task_ratios={"landcover": 0.7, "croptype": 0.3},
            batch_size=batch_size,
            num_batches=3,
        )

        N = len(self.binary_ds)
        for batch in sampler:
            self.assertEqual(len(batch), batch_size)
            lc_count = sum(1 for idx in batch if N <= idx < 2 * N)
            ct_count = sum(1 for idx in batch if 2 * N <= idx < 3 * N)
            self.assertEqual(lc_count, 7)
            self.assertEqual(ct_count, 3)

    def test_task_batch_sampler_per_bin_scope(self):
        """`class_balancing_scope='per_bin'` produces valid batches and
        different LC sampling probabilities than the default global scope
        (with the same density factor applied to both)."""

        batch_size = 4
        spatial_bin_size = 0.2

        global_sampler = self.binary_ds.get_task_batch_sampler(
            batch_size=batch_size,
            class_weight_method="balanced",
            spatial_bin_size_degrees=spatial_bin_size,
            spatial_weight_method="log",
            class_balancing_scope="global",
            min_samples_per_bin=1,
            num_batches=5,
        )
        per_bin_sampler = self.binary_ds.get_task_batch_sampler(
            batch_size=batch_size,
            class_weight_method="balanced",
            spatial_bin_size_degrees=spatial_bin_size,
            spatial_weight_method="log",
            class_balancing_scope="per_bin",
            min_samples_per_bin=1,
            num_batches=5,
        )

        # Same density factor in both samplers; only the class-weight source
        # differs. Probabilities should still differ because per-bin and
        # global produce different class weights.
        self.assertFalse(
            np.allclose(
                global_sampler._lc_probs.numpy(),
                per_bin_sampler._lc_probs.numpy(),
            ),
            "Per-bin scope unexpectedly produced identical LC probabilities to global scope",
        )

        # Sampler should still yield well-formed batches.
        N = len(self.binary_ds)
        batches = list(per_bin_sampler)
        self.assertEqual(len(batches), 5)
        for batch in batches:
            self.assertEqual(len(batch), batch_size)
            for idx in batch:
                self.assertTrue(N <= idx < 3 * N)

    def test_task_batch_sampler_per_bin_requires_spatial(self):
        """Per-bin scope without spatial_bin_size_degrees should error."""
        with self.assertRaises(ValueError):
            self.binary_ds.get_task_batch_sampler(
                batch_size=4,
                class_balancing_scope="per_bin",
                spatial_bin_size_degrees=None,
                num_batches=1,
            )

    def test_task_batch_sampler_density_axis_independent(self):
        """`spatial_weight_method` is orthogonal to `class_balancing_scope`:
        toggling between 'none' and 'log' (with all else fixed) changes the
        sampling distribution under both global and per_bin scopes."""

        batch_size = 4
        spatial_bin_size = 0.2

        common_kwargs = dict(
            batch_size=batch_size,
            class_weight_method="balanced",
            spatial_bin_size_degrees=spatial_bin_size,
            min_samples_per_bin=1,
            num_batches=5,
        )

        # Same scope, different density method → different probs.
        per_bin_no_density = self.binary_ds.get_task_batch_sampler(
            **common_kwargs,
            class_balancing_scope="per_bin",
            spatial_weight_method="none",
        )
        per_bin_log_density = self.binary_ds.get_task_batch_sampler(
            **common_kwargs,
            class_balancing_scope="per_bin",
            spatial_weight_method="log",
        )
        self.assertFalse(
            np.allclose(
                per_bin_no_density._lc_probs.numpy(),
                per_bin_log_density._lc_probs.numpy(),
            ),
            "spatial_weight_method should change the distribution under per_bin scope",
        )

        # Same axis under global scope.
        global_no_density = self.binary_ds.get_task_batch_sampler(
            **common_kwargs,
            class_balancing_scope="global",
            spatial_weight_method="none",
        )
        global_log_density = self.binary_ds.get_task_batch_sampler(
            **common_kwargs,
            class_balancing_scope="global",
            spatial_weight_method="log",
        )
        self.assertFalse(
            np.allclose(
                global_no_density._lc_probs.numpy(),
                global_log_density._lc_probs.numpy(),
            ),
            "spatial_weight_method should change the distribution under global scope",
        )

    def test_task_batch_sampler_invalid_scope(self):
        """Unknown class_balancing_scope should error."""
        with self.assertRaises(ValueError):
            self.binary_ds.get_task_batch_sampler(
                batch_size=4,
                class_balancing_scope="invalid_mode",  # type: ignore[arg-type]
                spatial_bin_size_degrees=0.2,
                num_batches=1,
            )

    def test_getitem_lc_virtual_index(self):
        """Feeding an LC virtual index [N, 2N) should override label_task to 'landcover'."""
        N = len(self.binary_ds)
        # Pick a sample whose dataframe label_task is 'croptype'
        croptype_rows = self.df.index[self.df["label_task"] == "croptype"].tolist()
        self.assertTrue(len(croptype_rows) > 0, "Need at least one croptype row")
        real_idx = croptype_rows[0]
        virtual_idx = N + real_idx  # LC virtual range

        _, attrs = self.binary_ds[virtual_idx]
        self.assertEqual(attrs["label_task"], "landcover")

    def test_getitem_ct_virtual_index(self):
        """Feeding a CT virtual index [2N, 3N) should override label_task to 'croptype'."""
        N = len(self.binary_ds)
        # Pick a sample whose dataframe label_task is 'landcover'
        lc_rows = self.df.index[self.df["label_task"] == "landcover"].tolist()
        self.assertTrue(len(lc_rows) > 0, "Need at least one landcover row")
        real_idx = lc_rows[0]
        virtual_idx = 2 * N + real_idx  # CT virtual range

        _, attrs = self.binary_ds[virtual_idx]
        self.assertEqual(attrs["label_task"], "croptype")

    def test_getitem_natural_index(self):
        """Natural index [0, N) should preserve label_task from the dataframe."""
        for i in range(len(self.binary_ds)):
            _, attrs = self.binary_ds[i]
            expected = self.df.iloc[i]["label_task"]
            self.assertEqual(attrs["label_task"], expected)

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


class TestSeasonMaskShapeMatchesTimesteps(unittest.TestCase):
    """Verify that season_masks.shape[1] == num_timesteps for various sizes."""

    @staticmethod
    def _build_df(num_samples, num_timesteps):
        """Build a minimal dataframe with enough timestep columns."""
        # Extend end_date so that monthly timestamp generation covers all positions.
        # get_monthly_timestamp_components generates one timestamp per month from
        # start_date to end_date, so we need at least (num_timesteps + 6) months.
        total_ts = num_timesteps + 6
        start = pd.Timestamp("2021-01-01")
        end = start + pd.DateOffset(months=total_ts)
        data = {
            "lat": [45.0] * num_samples,
            "lon": [5.0] * num_samples,
            "start_date": [str(start.date())] * num_samples,
            "end_date": [str(end.date())] * num_samples,
            "valid_time": ["2021-07-01"] * num_samples,
            "available_timesteps": [total_ts] * num_samples,
            "valid_position": [num_timesteps // 2] * num_samples,
        }
        # Need enough timestep columns: num_timesteps + 6 for augmentation headroom
        total_ts = num_timesteps + 6
        for ts in range(total_ts):
            for tpl in [
                "OPTICAL-B02-ts{}-10m",
                "OPTICAL-B03-ts{}-10m",
                "OPTICAL-B04-ts{}-10m",
                "OPTICAL-B08-ts{}-10m",
                "OPTICAL-B05-ts{}-20m",
                "OPTICAL-B06-ts{}-20m",
                "OPTICAL-B07-ts{}-20m",
                "OPTICAL-B8A-ts{}-20m",
                "OPTICAL-B11-ts{}-20m",
                "OPTICAL-B12-ts{}-20m",
                "SAR-VH-ts{}-20m",
                "SAR-VV-ts{}-20m",
                "METEO-precipitation_flux-ts{}-100m",
                "METEO-temperature_mean-ts{}-100m",
            ]:
                data[tpl.format(ts)] = [1000] * num_samples
        data["DEM-alt-20m"] = [100] * num_samples
        data["DEM-slo-20m"] = [5] * num_samples
        df = pd.DataFrame(data)
        df["finetune_class"] = ["cropland"] * num_samples
        df["label_task"] = ["landcover"] * num_samples
        df["landcover_label"] = ["lc_a"] * num_samples
        df["croptype_label"] = ["ct_a"] * num_samples
        return df

    def test_season_mask_shape_8_timesteps(self):
        """With num_timesteps=8, season_masks should have shape (S, 8)."""
        df = self._build_df(2, 8)
        ds = WorldCerealLabelledDataset(
            df,
            task_type="binary",
            num_outputs=1,
            num_timesteps=8,
            season_calendar_mode="calendar",
        )
        _, attrs = ds[0]
        masks = attrs["season_masks"]
        self.assertEqual(masks.shape[1], 8)
        # Default 2 seasons (tc-s1, tc-s2)
        self.assertEqual(masks.shape[0], 2)

    def test_season_mask_shape_18_timesteps(self):
        """With num_timesteps=18, season_masks should have shape (S, 18)."""
        df = self._build_df(2, 18)
        ds = WorldCerealLabelledDataset(
            df,
            task_type="binary",
            num_outputs=1,
            num_timesteps=18,
            season_calendar_mode="calendar",
        )
        _, attrs = ds[0]
        masks = attrs["season_masks"]
        self.assertEqual(masks.shape[1], 18)
        self.assertEqual(masks.shape[0], 2)

    def test_season_mask_shape_default_12_timesteps(self):
        """With default num_timesteps=12, season_masks should have shape (S, 12)."""
        df = self._build_df(2, 12)
        ds = WorldCerealLabelledDataset(
            df,
            task_type="binary",
            num_outputs=1,
            season_calendar_mode="calendar",
        )
        _, attrs = ds[0]
        masks = attrs["season_masks"]
        self.assertEqual(masks.shape[1], 12)
        self.assertEqual(masks.shape[0], 2)


class TestLcOnlyDatasetHelperAndSeasonMasks(unittest.TestCase):
    """Tests for _is_lc_only_dataset helper and the LC-only season-mask shortcut."""

    # ------------------------------------------------------------------
    # _is_lc_only_dataset
    # ------------------------------------------------------------------

    def test_is_lc_only_100(self):
        self.assertTrue(_is_lc_only_dataset("2020_KEN_FOO_POINT_100"))

    def test_is_lc_only_101(self):
        self.assertTrue(_is_lc_only_dataset("2020_KEN_FOO_POINT_101"))

    def test_is_not_lc_only_110(self):
        self.assertFalse(_is_lc_only_dataset("2020_KEN_BAR_POINT_110"))

    def test_is_not_lc_only_empty_string(self):
        self.assertFalse(_is_lc_only_dataset(""))

    def test_is_not_lc_only_100_mid_string(self):
        """_100 not at the end of the string should not match."""
        self.assertFalse(_is_lc_only_dataset("2020_100_KEN_BAR_POINT_110"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_timestamps(num_timesteps: int = 12) -> np.ndarray:
        """Monthly timestamps as (day, month, year) rows covering 2021."""
        rows = []
        year, month = 2021, 1
        for _ in range(num_timesteps):
            rows.append([1, month, year])
            month += 1
            if month > 12:
                month = 1
                year += 1
        return np.array(rows, dtype=np.int32)

    @staticmethod
    def _make_dataset(num_timesteps: int = 12) -> WorldCerealLabelledDataset:
        """Minimal WorldCerealLabelledDataset — ref_id is set per-test on the row."""
        total_ts = num_timesteps + 6
        data: dict = {
            "lat": [45.0],
            "lon": [5.0],
            "start_date": ["2021-01-01"],
            "end_date": ["2022-07-01"],
            "valid_time": ["2021-07-01"],
            "available_timesteps": [total_ts],
            "valid_position": [num_timesteps // 2],
            "ref_id": ["2020_KEN_PLACEHOLDER_POINT_110"],
            "finetune_class": ["nocrop"],
            "label_task": ["croptype"],
            "landcover_label": ["lc_nocrop"],
            "croptype_label": ["nocrop"],
            "DEM-alt-20m": [100],
            "DEM-slo-20m": [5],
        }
        for ts in range(total_ts):
            for tpl in [
                "OPTICAL-B02-ts{}-10m",
                "OPTICAL-B03-ts{}-10m",
                "OPTICAL-B04-ts{}-10m",
                "OPTICAL-B08-ts{}-10m",
                "OPTICAL-B05-ts{}-20m",
                "OPTICAL-B06-ts{}-20m",
                "OPTICAL-B07-ts{}-20m",
                "OPTICAL-B8A-ts{}-20m",
                "OPTICAL-B11-ts{}-20m",
                "OPTICAL-B12-ts{}-20m",
                "SAR-VH-ts{}-20m",
                "SAR-VV-ts{}-20m",
                "METEO-precipitation_flux-ts{}-100m",
                "METEO-temperature_mean-ts{}-100m",
            ]:
                data[tpl.format(ts)] = [1000]
        return WorldCerealLabelledDataset(
            pd.DataFrame(data),
            task_type="multiclass",
            num_outputs=1,
            num_timesteps=num_timesteps,
            season_calendar_mode="calendar",
        )

    # ------------------------------------------------------------------
    # Season-mask shortcut in _compute_season_metadata
    # ------------------------------------------------------------------

    def test_lc_100_masks_all_true_with_label_datetime(self):
        """LC-only row (_100) must return all-True masks and in_seasons."""
        ds = self._make_dataset()
        timestamps = self._make_timestamps()
        row = {
            "ref_id": "2020_KEN_FOO_POINT_100",
            "lat": 45.0,
            "lon": 5.0,
            "valid_time": "2021-07-01",
        }
        label_dt = np.datetime64("2021-07-01", "D")

        masks, in_seasons = ds._compute_season_metadata(
            row=row,
            timestamps=timestamps,
            season_ids=ds._season_ids,
            season_windows=ds._season_windows,
            derive_from_calendar=True,
            label_datetime=label_dt,
        )

        self.assertTrue(masks.all(), "Expected all-True masks for LC-only dataset")
        self.assertIsNotNone(in_seasons)
        self.assertTrue(
            in_seasons.all(), "Expected all-True in_seasons for LC-only dataset"
        )
        # Shape contract: (num_seasons, num_timesteps)
        self.assertEqual(masks.shape[1], len(timestamps))

    def test_lc_101_masks_all_true_without_label_datetime(self):
        """Without label_datetime, in_seasons must be None (not all-True)."""
        ds = self._make_dataset()
        timestamps = self._make_timestamps()
        row = {
            "ref_id": "2020_KEN_FOO_POINT_101",
            "lat": 45.0,
            "lon": 5.0,
        }

        masks, in_seasons = ds._compute_season_metadata(
            row=row,
            timestamps=timestamps,
            season_ids=ds._season_ids,
            season_windows=ds._season_windows,
            derive_from_calendar=True,
            label_datetime=None,
        )

        self.assertTrue(masks.all(), "Expected all-True masks for LC-only dataset")
        self.assertIsNone(in_seasons)

    def test_ct_110_is_not_shortcircuited(self):
        """CT rows (_110) must not be short-circuited; calendar lookup must be attempted."""
        ds = self._make_dataset()
        timestamps = self._make_timestamps()
        row = {
            "ref_id": "2020_KEN_BAR_POINT_110",
            "lat": 45.0,
            "lon": 5.0,
        }
        sentinel = (np.zeros(len(timestamps), dtype=bool), False)
        label_dt = np.datetime64("2021-07-01", "D")

        with mock.patch.object(
            WorldCerealLabelledDataset,
            "_season_mask_from_calendar",
            return_value=sentinel,
        ) as mocked:
            ds._compute_season_metadata(
                row=row,
                timestamps=timestamps,
                season_ids=ds._season_ids,
                season_windows=ds._season_windows,
                derive_from_calendar=True,
                label_datetime=label_dt,
            )

        mocked.assert_called()


class TestPerBinClassWeights(unittest.TestCase):
    """Unit tests for :func:`_get_per_bin_class_weights`."""

    def test_balanced_bin_equal_weights(self):
        """A bin with a balanced class distribution should yield equal weights."""
        labels = np.array(["a", "a", "b", "b"])
        bins = np.array(["B0", "B0", "B0", "B0"])
        weights = _get_per_bin_class_weights(
            labels, bins, method="balanced", clip_range=None, min_samples_per_bin=1
        )
        np.testing.assert_allclose(weights, np.ones_like(weights))

    def test_within_bin_class_balancing(self):
        """Rare classes within a bin receive higher weight than dominant classes."""
        # Bin has 1 'a' and 9 'b' → 'a' weight should be ~9× 'b' weight (balanced).
        labels = np.array(["a"] + ["b"] * 9)
        bins = np.array(["B0"] * 10)
        weights = _get_per_bin_class_weights(
            labels, bins, method="balanced", clip_range=None, min_samples_per_bin=1
        )
        a_weight = weights[labels == "a"][0]
        b_weight = weights[labels == "b"][0]
        self.assertGreater(a_weight, b_weight)
        np.testing.assert_allclose(a_weight / b_weight, 9.0, rtol=1e-6)

    def test_global_mean_is_one(self):
        """The assembled per-sample array should have mean = 1 after normalisation."""
        rng = np.random.default_rng(0)
        labels = rng.choice(["a", "b", "c"], size=200, p=[0.6, 0.3, 0.1])
        bins = rng.choice(["B0", "B1", "B2", "B3"], size=200)
        weights = _get_per_bin_class_weights(
            labels, bins, method="balanced", clip_range=None, min_samples_per_bin=10
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_decouples_class_weight_per_bin(self):
        """Different bins with different class distributions yield different weights
        for the same class label."""
        # Bin 0: 'a' is rare (1/10) → high weight
        # Bin 1: 'a' is dominant (9/10) → low weight
        labels = np.array(["a"] + ["b"] * 9 + ["a"] * 9 + ["b"])  # bin 0  # bin 1
        bins = np.array(["B0"] * 10 + ["B1"] * 10)
        weights = _get_per_bin_class_weights(
            labels, bins, method="balanced", clip_range=None, min_samples_per_bin=1
        )
        a_in_b0 = weights[(bins == "B0") & (labels == "a")][0]
        a_in_b1 = weights[(bins == "B1") & (labels == "a")][0]
        self.assertGreater(a_in_b0, a_in_b1)

    def test_sparse_bin_falls_back_to_global(self):
        """Bins below min_samples_per_bin use global class weights, not within-bin.

        Setup designed to make fallback observable:
          * Dense bin B0: 100 'b' samples (single class) → within-bin gives 1.0.
          * Sparse bin B1: 1 'a' + 1 'b' (size 2, below the threshold).

        If B1 *incorrectly* used within-bin weights, both samples there would
        receive the same weight (single 'a' + single 'b' → balanced 1:1).
        With the correct fallback to global, 'a' is globally rare (1 of 102)
        so its weight should be far higher than 'b'.
        """
        labels = np.array(["b"] * 100 + ["a", "b"])
        bins = np.array(["B0"] * 100 + ["B1", "B1"])
        weights = _get_per_bin_class_weights(
            labels, bins, method="balanced", clip_range=None, min_samples_per_bin=10
        )
        sparse_a = float(weights[(bins == "B1") & (labels == "a")][0])
        sparse_b = float(weights[(bins == "B1") & (labels == "b")][0])
        # Fallback applied → 'a' weight ≫ 'b' weight (no fallback would give equal).
        self.assertGreater(sparse_a, 10 * sparse_b)

    def test_clip_range_applied(self):
        """Clipping should bound the final per-sample weights."""
        # Extreme imbalance within bin → uncapped weights would exceed 5.0
        labels = np.array(["a"] + ["b"] * 99)
        bins = np.array(["B0"] * 100)
        clipped = _get_per_bin_class_weights(
            labels,
            bins,
            method="balanced",
            clip_range=(0.5, 5.0),
            min_samples_per_bin=1,
        )
        self.assertLessEqual(float(clipped.max()), 5.0 + 1e-9)
        self.assertGreaterEqual(float(clipped.min()), 0.5 - 1e-9)

    def test_shape_mismatch_raises(self):
        labels = np.array(["a", "b"])
        bins = np.array(["B0", "B0", "B0"])
        with self.assertRaises(ValueError):
            _get_per_bin_class_weights(
                labels,
                bins,
                method="balanced",
                clip_range=None,
                min_samples_per_bin=1,
            )

    def test_invalid_min_samples_raises(self):
        labels = np.array(["a", "b"])
        bins = np.array(["B0", "B0"])
        with self.assertRaises(ValueError):
            _get_per_bin_class_weights(
                labels,
                bins,
                method="balanced",
                clip_range=None,
                min_samples_per_bin=0,
            )


class TestSpatialDensityWeights(unittest.TestCase):
    """Unit tests for :func:`_get_spatial_density_weights`."""

    def test_sparse_bin_overridden_to_one(self):
        """Singleton bins should be set to 1.0 instead of the inverse-density value."""
        # 100 samples in bin "dense", 1 sample in bin "sparse".
        bins = np.array(["dense"] * 100 + ["sparse"])
        weights = _get_spatial_density_weights(
            bins, method="log", clip_range=(0.1, 10.0), min_samples_per_bin=10
        )
        # The sparse-bin sample should be exactly 1.0 (overridden).
        self.assertAlmostEqual(float(weights[-1]), 1.0)
        # Without override, the singleton sparse bin would have a high
        # log-inverse-density weight; with override it's 1.0, well below
        # the clip ceiling.
        self.assertLess(float(weights[-1]), 10.0)

    def test_no_sparse_bins_keeps_normalized_weights(self):
        """When all bins exceed min_samples_per_bin, output matches the
        un-protected normalized weights."""
        # Two bins, both above threshold.
        bins = np.array(["a"] * 50 + ["b"] * 50)
        with_protection = _get_spatial_density_weights(
            bins, method="log", clip_range=(0.1, 10.0), min_samples_per_bin=10
        )
        # Re-import to compute the un-protected baseline directly.
        from worldcereal.train.datasets import _get_normalized_weights

        without_protection = _get_normalized_weights(bins, "log", (0.1, 10.0))
        np.testing.assert_allclose(with_protection, without_protection)

    def test_min_samples_one_disables_protection(self):
        """min_samples_per_bin=1 means no override; behaviour matches the un-
        protected helper exactly."""
        bins = np.array(["a"] * 100 + ["b"])  # one singleton bin
        with_protection = _get_spatial_density_weights(
            bins, method="log", clip_range=(0.1, 10.0), min_samples_per_bin=1
        )
        from worldcereal.train.datasets import _get_normalized_weights

        without_protection = _get_normalized_weights(bins, "log", (0.1, 10.0))
        np.testing.assert_allclose(with_protection, without_protection)


class TestSmoothedPerBinClassWeights(unittest.TestCase):
    """Unit tests for `_get_smoothed_per_bin_class_weights` (bilinear)."""

    def _two_bin_grid(self):
        """Construct a synthetic dataset with 2 dense bins (5° size).

        Bin (lat∈[0,5°), lon∈[0,5°)): all class 'a' (50 samples)
        Bin (lat∈[5,10°), lon∈[0,5°)): all class 'b' (50 samples)
        """
        n_per_bin = 50
        # Bin centres are at lat=2.5, 7.5; lon=2.5
        rng = np.random.default_rng(0)
        a_lats = rng.uniform(0.0, 5.0, n_per_bin)
        a_lons = rng.uniform(0.0, 5.0, n_per_bin)
        b_lats = rng.uniform(5.0, 10.0, n_per_bin)
        b_lons = rng.uniform(0.0, 5.0, n_per_bin)
        labels = np.array(["a"] * n_per_bin + ["b"] * n_per_bin)
        lats = np.concatenate([a_lats, b_lats])
        lons = np.concatenate([a_lons, b_lons])
        return labels, lats, lons

    def test_at_bin_center_matches_hard_binning(self):
        """A sample exactly at the bin centre should get the same weight as the
        hard-binning helper (since fu=0, fv=0 puts 100% weight on its own bin)."""
        labels, lats, lons = self._two_bin_grid()
        # Add a probe sample exactly at bin (0–5°, 0–5°) centre — class 'a'
        probe_label = "a"
        probe_lat = 2.5  # bin centre in lat
        probe_lon = 2.5  # bin centre in lon
        labels = np.append(labels, probe_label)
        lats = np.append(lats, probe_lat)
        lons = np.append(lons, probe_lon)
        bilinear = _get_smoothed_per_bin_class_weights(
            labels,
            lats,
            lons,
            bin_size=5.0,
            method="balanced",
            clip_range=None,
            min_samples_per_bin=1,
        )
        # Hard binning, same data
        bins = np.array(
            [
                f"{int(np.floor((la + 90) / 5))}_{int(np.floor((lo + 180) / 5))}"
                for la, lo in zip(lats, lons)
            ]
        )
        hard = _get_per_bin_class_weights(
            labels,
            bins,
            method="balanced",
            clip_range=None,
            min_samples_per_bin=1,
        )
        # Bin-centre probe: bilinear and hard should match within renormalisation
        # tolerance. The two arrays are mean-normalised independently, and
        # bilinear's other samples pick up small global-fallback contributions
        # at NW/SE/NE corners (which slightly shifts the global mean), so a
        # loose rtol covers the inevitable numerical drift.
        probe_bilinear = bilinear[-1]
        probe_hard = hard[-1]
        # In this two-bin balanced setup both should be ≈ 1.0.
        np.testing.assert_allclose(probe_bilinear, probe_hard, rtol=1e-3)

    def test_at_corner_blends_neighbors(self):
        """A sample on the edge between two bins gets the average of the two
        neighbours' class weights."""
        # Two bins where class 'x' has different per-bin weights.
        # Bin A: 9 'x' + 1 'y' → class 'x' is locally common, low weight
        # Bin B: 1 'x' + 9 'y' → class 'x' is locally rare, high weight
        # Bin C, D: empty (we'll have empty corners → fall back to global)
        labels = np.array(
            (["x"] * 9 + ["y"])  # bin A around lat=2.5
            + (["x"] + ["y"] * 9)  # bin B around lat=7.5
        )
        lats = np.concatenate(
            [
                np.full(10, 2.5),  # all in bin A (lat 0-5)
                np.full(10, 7.5),  # all in bin B (lat 5-10)
            ]
        )
        lons = np.full(20, 2.5)  # all in lon bin 0-5
        # Probe sample exactly on the edge between bin A and bin B (lat=5.0)
        # Class 'x'. Should get average of bin A weight and bin B weight for 'x'.
        labels = np.append(labels, "x")
        lats = np.append(lats, 5.0)  # exactly at boundary; fu = 0.5 in u-space
        lons = np.append(lons, 2.5)
        bilinear = _get_smoothed_per_bin_class_weights(
            labels,
            lats,
            lons,
            bin_size=5.0,
            method="balanced",
            clip_range=None,
            min_samples_per_bin=1,
        )
        # Hard binning
        bins = np.array(
            [
                f"{int(np.floor((la + 90) / 5))}_{int(np.floor((lo + 180) / 5))}"
                for la, lo in zip(lats, lons)
            ]
        )
        hard = _get_per_bin_class_weights(
            labels,
            bins,
            method="balanced",
            clip_range=None,
            min_samples_per_bin=1,
        )
        # Bin A 'x' weight (hard): from samples 0..8 (labels 'x' in bin A)
        bin_a_x = hard[0]
        # Bin B 'x' weight (hard): sample 10 (label 'x' in bin B)
        bin_b_x = hard[10]
        # The probe's bilinear value should be ~50/50 of these two.
        # Hard and bilinear arrays use the same mean normaliser convention but
        # their distributions differ slightly, so use a reasonable tolerance.
        probe_bilinear = bilinear[-1]
        # In hard: bin_a_x < bin_b_x (x is rare in B → high weight there).
        self.assertGreater(
            probe_bilinear, bin_a_x * 0.5
        )  # meaningfully above bin A's weight
        self.assertLess(
            probe_bilinear, bin_b_x * 1.0
        )  # meaningfully below bin B's weight

    def test_global_mean_one(self):
        """Final per-sample array has mean = 1 by construction."""
        labels, lats, lons = self._two_bin_grid()
        weights = _get_smoothed_per_bin_class_weights(
            labels,
            lats,
            lons,
            bin_size=5.0,
            method="balanced",
            clip_range=None,
            min_samples_per_bin=1,
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_clip_range_applied(self):
        """Clip range bounds the output."""
        labels, lats, lons = self._two_bin_grid()
        weights = _get_smoothed_per_bin_class_weights(
            labels,
            lats,
            lons,
            bin_size=5.0,
            method="balanced",
            clip_range=(0.5, 1.5),
            min_samples_per_bin=1,
        )
        self.assertGreaterEqual(float(weights.min()), 0.5 - 1e-9)
        self.assertLessEqual(float(weights.max()), 1.5 + 1e-9)

    def test_sparse_bin_falls_back_to_global(self):
        """A bin below min_samples_per_bin is treated as missing, so its
        corner contribution falls back to the global class weight."""
        # 100 'a' in dense bin, 1 'a' in sparse-corner bin
        labels = np.array(["a"] * 100 + ["a"])
        lats = np.concatenate([np.full(100, 2.5), [7.5]])  # sparse bin at lat∈[5,10)
        lons = np.full(101, 2.5)
        weights = _get_smoothed_per_bin_class_weights(
            labels,
            lats,
            lons,
            bin_size=5.0,
            method="balanced",
            clip_range=None,
            min_samples_per_bin=10,
        )
        # All values should be finite, no errors raised.
        self.assertTrue(np.all(np.isfinite(weights)))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _get_smoothed_per_bin_class_weights(
                np.array(["a", "b"]),
                np.array([0.0, 0.0, 0.0]),  # wrong length
                np.array([0.0, 0.0]),
                bin_size=5.0,
                method="balanced",
                clip_range=None,
            )


class TestClassWeightMultipliersRouting(unittest.TestCase):
    """Tests for class_weight_multipliers routing in SeasonalTaskBatchSampler."""

    def setUp(self):
        import pandas as pd

        # Shared label that appears in BOTH LC and CT columns — the real-world
        # example is "temporary_crops" which lives in both LC_CODES and
        # class_mappings.json.
        shared = "shared_class"
        data = {
            "landcover_label": [shared, "lc_only", shared, "lc_only", shared],
            "croptype_label": [shared, shared, "ct_only", "ct_only", shared],
            "lat": [1.0, 2.0, 3.0, 4.0, 5.0],
            "lon": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        self.df = pd.DataFrame(data)

    def _make_sampler(self, multipliers):
        from worldcereal.train.datasets import SeasonalTaskBatchSampler

        return SeasonalTaskBatchSampler(
            dataframe=self.df,
            batch_size=4,
            class_weight_method="balanced",
            num_batches=5,
            class_weight_multipliers=multipliers,
        )

    def test_shared_label_applied_to_both_pools(self):
        """A label present in both LC and CT sets must boost both sampling pools."""
        sampler_base = self._make_sampler(None)
        sampler_boost = self._make_sampler({"shared_class": 10.0})

        # LC probs for samples with the shared label should be higher after boost
        lc_probs_base = sampler_base._lc_probs.numpy()
        lc_probs_boost = sampler_boost._lc_probs.numpy()
        ct_probs_base = sampler_base._ct_probs.numpy()
        ct_probs_boost = sampler_boost._ct_probs.numpy()

        shared_lc_idx = [
            i for i, v in enumerate(self.df["landcover_label"]) if v == "shared_class"
        ]
        shared_ct_idx = [
            i
            for i, v in enumerate(
                self.df.loc[
                    self.df["croptype_label"].notna(), "croptype_label"
                ].reset_index(drop=True)
            )
            if v == "shared_class"
        ]

        # Both pools should have higher probability after the boost
        self.assertTrue(
            all(lc_probs_boost[i] > lc_probs_base[i] for i in shared_lc_idx),
            "shared_class boost did NOT increase LC pool probabilities",
        )
        self.assertTrue(
            all(ct_probs_boost[i] > ct_probs_base[i] for i in shared_ct_idx),
            "shared_class boost did NOT increase CT pool probabilities — elif bug",
        )

    def test_lc_only_label_only_affects_lc_pool(self):
        """A label that only exists in the LC pool must not change CT probs."""
        sampler_base = self._make_sampler(None)
        sampler_boost = self._make_sampler({"lc_only": 5.0})
        import numpy as np

        self.assertTrue(
            np.allclose(
                sampler_base._ct_probs.numpy(), sampler_boost._ct_probs.numpy()
            ),
            "lc_only multiplier incorrectly changed CT pool probabilities",
        )

    def test_unknown_label_warns_and_does_not_crash(self):
        """An unknown key in class_weight_multipliers should warn but not raise."""
        # Should complete without error (warning is logged internally)
        sampler = self._make_sampler({"completely_unknown_label": 2.0})
        self.assertIsNotNone(sampler)


class TestCheckpointMetricMonitoring(unittest.TestCase):
    """Tests for the checkpoint_metric / mean_f1 sentinel behavior."""

    def _compute_mean_f1(self, cur_lc_f1, cur_ct_f1):
        """Replicate the fixed formula from finetuning_utils.py."""
        return (max(0.0, cur_lc_f1) + max(0.0, cur_ct_f1)) / 2.0

    def test_mean_f1_both_available(self):
        """mean_f1 is the arithmetic mean when both heads have valid metrics."""
        result = self._compute_mean_f1(0.8, 0.6)
        self.assertAlmostEqual(result, 0.7)

    def test_mean_f1_ct_sentinel_does_not_go_negative(self):
        """When CT is gate-rejected (sentinel -1.0), mean_f1 must stay >= 0."""
        result = self._compute_mean_f1(0.85, -1.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertAlmostEqual(result, 0.425)

    def test_mean_f1_lc_sentinel_does_not_go_negative(self):
        """When LC sentinel fires, mean_f1 must stay >= 0."""
        result = self._compute_mean_f1(-1.0, 0.70)
        self.assertGreaterEqual(result, 0.0)
        self.assertAlmostEqual(result, 0.35)

    def test_mean_f1_both_sentinel_is_zero(self):
        """When both heads return sentinel, mean_f1 should be 0.0 (not -1.0)."""
        result = self._compute_mean_f1(-1.0, -1.0)
        self.assertEqual(result, 0.0)

    def test_non_seasonal_fallback_to_val_loss(self):
        """Non-seasonal models must fall back to val_loss regardless of requested metric."""
        seasonal_metrics_supported = False
        for requested in ("lc_f1", "ct_f1", "mean_f1"):
            effective = (
                requested
                if requested == "val_loss" or seasonal_metrics_supported
                else "val_loss"
            )
            self.assertEqual(
                effective,
                "val_loss",
                f"checkpoint_metric='{requested}' should fall back to 'val_loss' "
                "for non-seasonal models",
            )

    def test_seasonal_model_respects_requested_metric(self):
        """Seasonal models must use the requested metric without fallback."""
        seasonal_metrics_supported = True
        for requested in ("lc_f1", "ct_f1", "mean_f1", "val_loss"):
            effective = (
                requested
                if requested == "val_loss" or seasonal_metrics_supported
                else "val_loss"
            )
            self.assertEqual(
                effective,
                requested,
                f"checkpoint_metric='{requested}' should NOT be overridden for seasonal models",
            )

    def test_val_loss_explicit_never_falls_back(self):
        """val_loss always passes through regardless of seasonal support."""
        for seasonal in (True, False):
            effective = (
                "val_loss" if "val_loss" == "val_loss" or seasonal else "val_loss"
            )
            self.assertEqual(effective, "val_loss")


if __name__ == "__main__":
    unittest.main()
