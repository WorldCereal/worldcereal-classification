import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from prometheo.predictors import DEM_BANDS, METEO_BANDS, NODATAVALUE, S1_BANDS, S2_BANDS

from worldcereal.train.datasets import (
    WorldCerealDataset,
    WorldCerealLabelledDataset,
    get_correct_date,
    get_dekad_timestamp_components,
    get_monthly_timestamp_components,
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

        # Initialize the datasets
        self.base_ds = WorldCerealDataset(self.df, num_timesteps=self.num_timesteps)
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
        self.assertEqual(meteo.shape, (self.num_timesteps, len(METEO_BANDS)))
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
        self.assertEqual(inputs["meteo"].shape, (self.num_timesteps, len(METEO_BANDS)))
        self.assertEqual(inputs["dem"].shape, (1, 1, len(DEM_BANDS)))
        self.assertEqual(inputs["latlon"].shape, (2,))
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
        item = self.binary_ds[0]
        self.assertEqual(type(item).__name__, "Predictors")
        self.assertTrue(hasattr(item, "label"))

    def test_binary_label(self):
        """Test binary labelled dataset returns correct labels."""
        item = self.binary_ds[0]
        # Cropland should be mapped to 1 (positive class)
        self.assertEqual(item.label[0, 0, 0, 0], 1)

        item = self.binary_ds[1]
        # not_cropland should be mapped to 0 (negative class)
        self.assertEqual(item.label[0, 0, 0, 0], 0)

    def test_multiclass_label(self):
        """Test multiclass labelled dataset returns correct labels."""
        item = self.multiclass_ds[0]
        # First sample should have class index 0
        self.assertEqual(item.label[0, 0, 0, 0], 0)

        item = self.multiclass_ds[4]
        # Fifth sample should have class index 1
        self.assertEqual(item.label[0, 0, 0, 0], 1)


    def test_time_explicit_label(self):
        """Test time explicit labelled dataset returns correct label shape."""
        item = self.time_explicit_ds[0]
        # Label should have temporal dimension
        self.assertEqual(item.label.shape, (1, 1, self.num_timesteps, 1))

        # For time_explicit, valid_position should have a value, other positions should be NODATAVALUE
        row = pd.Series.to_dict(self.df.iloc[0, :])
        _, valid_position = self.time_explicit_ds.get_timestep_positions(row)

        # Check only one position has a value
        valid_values = (item.label != NODATAVALUE).sum()
        self.assertEqual(valid_values, 1)


class TestTimeUtilities(unittest.TestCase):
    def test_generate_month_sequence(self):
        """Test generating a sequence of months."""
        start_date = np.datetime64("2021-01-03", "D")
        end_date = np.datetime64("2021-12-24", "D")
        
        # make sure that start and end dates are month-aligned
        start_date = get_correct_date(start_date, timestep_freq="month")
        end_date = get_correct_date(end_date, timestep_freq="month")
        
        days, months, years = get_monthly_timestamp_components(start_date, end_date)

        # Should have 12 months
        self.assertEqual(len(months), 12)
        
        # First should be January 2021
        self.assertEqual(f"{years[0]}-{months[0]}", "2021-1")
        
        # Last should be December 2021
        self.assertEqual(f"{years[-1]}-{months[-1]}", "2021-12")

    def test_get_monthly_timestamp_components(self):
        """Test getting month timestamp components."""
        start_date = np.datetime64("2021-01-03", "D")
        end_date = np.datetime64("2021-12-24", "D")
        
        # make sure that start and end dates are month-aligned
        start_date = get_correct_date(start_date, timestep_freq="month")
        end_date = get_correct_date(end_date, timestep_freq="month")
        
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
        
        # make sure that start and end dates are dekad-aligned
        start_date = get_correct_date(start_date, timestep_freq="dekad")
        end_date = get_correct_date(end_date, timestep_freq="dekad")
        
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


if __name__ == "__main__":
    unittest.main()
