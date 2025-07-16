import unittest
from pathlib import Path

import pandas as pd
from prometheo.predictors import NODATAVALUE

from worldcereal.train.data import get_training_dfs_from_parquet
from worldcereal.train.datasets import (
    WorldCerealLabelledDataset,
    # MaskingMode,
    # MaskingStrategy,
)
from worldcereal.train.finetuning_utils import (
    evaluate_finetuned_model,
    prepare_training_datasets,
)
from worldcereal.utils.refdata import get_class_mappings

CLASS_MAPPINGS = get_class_mappings()
LANDCOVER_KEY = "LANDCOVER10"
CROPTYPE_KEY = "CROPTYPE27"


class TestPrepareTrainingDatasets(unittest.TestCase):
    def setUp(self):
        """Set up real data for dataset creation tests."""
        # Use the same path pattern as in other tests
        self.data_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "worldcereal"
            / "data"
            / "month"
        )

        parquet_files = list(self.data_path.rglob("*.geoparquet"))

        # Create a temporary directory
        self.temp_dir = Path("temp_dataset_test_dir")
        self.temp_dir.mkdir(exist_ok=True)

        # Prepare real data using prepare_training_df for binary classification
        self.train_df_binary, self.val_df_binary, self.test_df_binary = (
            get_training_dfs_from_parquet(
                parquet_files,
                finetune_classes=LANDCOVER_KEY,
                class_mappings=CLASS_MAPPINGS,
                debug=True,  # Limit processing for faster tests
            )
        )

        # Prepare real data for multiclass classification
        self.train_df_multiclass, self.val_df_multiclass, self.test_df_multiclass = (
            get_training_dfs_from_parquet(
                parquet_files,
                finetune_classes=CROPTYPE_KEY,
                class_mappings=CLASS_MAPPINGS,
                debug=True,
            )
        )

        # Get the classes lists from the real mappings
        self.binary_classes = list(set(CLASS_MAPPINGS[LANDCOVER_KEY].values()))
        self.multiclass_classes = list(set(CLASS_MAPPINGS[CROPTYPE_KEY].values()))

        # Filter multiclass classes to only those present in our data
        self.multiclass_classes = [
            c
            for c in self.multiclass_classes
            if c in self.train_df_multiclass["finetune_class"].unique()
        ]

        # Create corner case: add missing timestep data for masking tests
        # Make a copy to not affect other tests
        self.train_df_with_missing = self.train_df_binary.copy()
        if len(self.train_df_with_missing) > 0:
            # Simulate missing data in some timesteps by setting to NODATAVALUE
            for ts in range(3, 6):  # Create gap in middle timesteps
                for band_template in ["OPTICAL-B02-ts{}-10m", "OPTICAL-B03-ts{}-10m"]:
                    if band_template.format(ts) in self.train_df_with_missing.columns:
                        self.train_df_with_missing[band_template.format(ts)] = (
                            NODATAVALUE
                        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_prepare_training_datasets_binary(self):
        """Test preparing datasets for binary classification using real data."""

        train_ds, val_ds, test_ds = prepare_training_datasets(
            self.train_df_binary,
            self.val_df_binary,
            self.test_df_binary,
            augment=True,
            time_explicit=False,
            task_type="binary",
            num_outputs=1,
        )

        # Check dataset types
        self.assertIsInstance(train_ds, WorldCerealLabelledDataset)
        self.assertIsInstance(val_ds, WorldCerealLabelledDataset)
        self.assertIsInstance(test_ds, WorldCerealLabelledDataset)

        # Check dataset lengths match dataframe lengths
        self.assertEqual(len(train_ds), len(self.train_df_binary))
        self.assertEqual(len(val_ds), len(self.val_df_binary))
        self.assertEqual(len(test_ds), len(self.test_df_binary))

        # Check task type and augmentation settings
        self.assertEqual(train_ds.task_type, "binary")
        self.assertEqual(val_ds.task_type, "binary")
        self.assertEqual(test_ds.task_type, "binary")

        self.assertTrue(train_ds.augment)  # Training dataset should have augmentation
        self.assertFalse(
            val_ds.augment
        )  # Validation dataset should not have augmentation
        self.assertFalse(test_ds.augment)  # Test dataset should not have augmentation

        # Check actual class values from the dataset
        sample = train_ds[0]
        self.assertIsNotNone(sample)
        self.assertTrue(hasattr(sample, "label"))

    def test_prepare_training_datasets_multiclass(self):
        """Test preparing datasets for multiclass classification using real data."""

        train_ds, val_ds, test_ds = prepare_training_datasets(
            self.train_df_multiclass,
            self.val_df_multiclass,
            self.test_df_multiclass,
            augment=True,
            time_explicit=False,
            task_type="multiclass",
            num_outputs=len(self.multiclass_classes),
            classes_list=self.multiclass_classes,
        )

        # Check dataset types
        self.assertIsInstance(train_ds, WorldCerealLabelledDataset)
        self.assertIsInstance(val_ds, WorldCerealLabelledDataset)
        self.assertIsInstance(test_ds, WorldCerealLabelledDataset)

        # Check task type
        self.assertEqual(train_ds.task_type, "multiclass")
        self.assertEqual(val_ds.task_type, "multiclass")
        self.assertEqual(test_ds.task_type, "multiclass")

        # Check classes list
        self.assertEqual(train_ds.classes_list, self.multiclass_classes)
        self.assertEqual(val_ds.classes_list, self.multiclass_classes)
        self.assertEqual(test_ds.classes_list, self.multiclass_classes)

        # Verify dataset output shape
        if len(train_ds) > 0:
            sample = train_ds[0]
            self.assertEqual(sample.label.shape, (1, 1, 1, 1))  # (H, W, T, 1)

    def test_prepare_training_datasets_time_explicit(self):
        """Test preparing datasets with time_explicit=True using real data."""

        train_ds, val_ds, test_ds = prepare_training_datasets(
            self.train_df_binary,
            self.val_df_binary,
            self.test_df_binary,
            augment=False,
            time_explicit=True,
            task_type="binary",
            num_outputs=1,
        )

        # Check time_explicit setting
        self.assertTrue(train_ds.time_explicit)
        self.assertTrue(val_ds.time_explicit)
        self.assertTrue(test_ds.time_explicit)

        # Get a sample from dataset to check label shape
        if len(train_ds) > 0:
            sample = train_ds[0]
            # Check that label has temporal dimension
            self.assertEqual(sample.label.shape[2], train_ds.num_timesteps)
            # Verify correct shape using expected timesteps
            timesteps = 12  # Default number of timesteps in InSeasonLabelledDataset
            self.assertEqual(
                sample.label.shape, (1, 1, timesteps, train_ds.num_outputs)
            )  # (H, W, T, num_classes)

    # def test_prepare_training_datasets_with_masking(self):
    #     """Test dataset preparation with in-season masking using real data."""

    #     # Test with fixed mask position
    #     mask_position = 5
    #     train_ds, val_ds, test_ds = prepare_training_datasets(
    #         self.train_df_binary,
    #         self.val_df_binary,
    #         self.test_df_binary,
    #         augment=False,
    #         time_explicit=False,
    #         task_type="binary",
    #         num_outputs=1,
    #         masking_strategy_train=MaskingStrategy(
    #             MaskingMode.FIXED, from_position=mask_position
    #         ),
    #         masking_strategy_val=MaskingStrategy(
    #             MaskingMode.FIXED, from_position=mask_position
    #         ),
    #     )

    #     # Check mask position was set
    #     self.assertEqual(train_ds.masking_strategy.from_position, mask_position)
    #     self.assertEqual(val_ds.masking_strategy.from_position, mask_position)
    #     self.assertEqual(test_ds.masking_strategy.from_position, mask_position)

    #     # Verify random mask flag
    #     self.assertEqual(train_ds.masking_strategy.mode, MaskingMode.FIXED)
    #     self.assertEqual(val_ds.masking_strategy.mode, MaskingMode.FIXED)
    #     self.assertEqual(test_ds.masking_strategy.mode, MaskingMode.FIXED)

    #     # Check that masking is applied - we can verify this by getting a sample
    #     if len(train_ds) > 0:
    #         sample = train_ds[0]
    #         # The data beyond mask position should contain NODATAVALUE
    #         # Get the non-masked shape first from sample
    #         data_channels = sample.s2.shape[0]
    #         timesteps = sample.s2.shape[2]

    #         # Create expected mask for testing - all positions after mask_position should be NODATAVALUE
    #         for ts in range(mask_position, timesteps):
    #             # Check a subset of data channels for NODATAVALUE
    #             # Note: If there was already NODATAVALUE in the data, this might not catch everything
    #             # But it should detect if masking isn't being applied at all
    #             # channel_indices = [0, 1]  # Check the first two channels only
    #             for channel in range(data_channels):
    #                 # If masking is working, these positions should all be NODATAVALUE
    #                 data_at_masked_position = sample.s2[
    #                     0, 0, ts, channel
    #                 ]  # [H, W, T, BAND]
    #                 self.assertTrue(
    #                     (data_at_masked_position == NODATAVALUE).all()
    #                     or np.isnan(data_at_masked_position).all(),
    #                     f"Expected NODATAVALUE for S2 data at position {ts} but got {data_at_masked_position}",
    #                 )

    # def test_prepare_training_datasets_with_random_masking(self):
    #     """Test dataset preparation with random in-season masking using real data."""

    #     min_mask_position = 3
    #     train_ds, val_ds, test_ds = prepare_training_datasets(
    #         self.train_df_binary,
    #         self.val_df_binary,
    #         self.test_df_binary,
    #         augment=False,
    #         time_explicit=False,
    #         task_type="binary",
    #         num_outputs=1,
    #         masking_strategy_train=MaskingStrategy(
    #             MaskingMode.RANDOM, from_position=min_mask_position
    #         ),
    #     )

    #     # Check random mask settings
    #     self.assertEqual(train_ds.masking_strategy.mode, MaskingMode.RANDOM)
    #     self.assertEqual(
    #         val_ds.masking_strategy.mode, MaskingMode.NONE
    #     )  # No masking for val in this case
    #     self.assertEqual(
    #         test_ds.masking_strategy.mode, MaskingMode.NONE
    #     )  # No masking for test in this case

    #     # Check min mask position
    #     self.assertEqual(train_ds.masking_strategy.from_position, min_mask_position)


class TestEvaluateFinetunedModel(unittest.TestCase):
    def setUp(self):
        """Set up real data for evaluation tests."""
        # Use the same path pattern as in other tests
        self.data_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "worldcereal"
            / "data"
            / "month"
        )

        # Create a temporary directory for output files
        self.temp_dir = Path("temp_eval_test_dir")
        self.temp_dir.mkdir(exist_ok=True)

        parquet_files = list(self.data_path.rglob("*.geoparquet"))

        # Load real data in debug mode (limited number of files)
        _, _, test_df = get_training_dfs_from_parquet(
            parquet_files,
            finetune_classes=LANDCOVER_KEY,
            class_mappings=CLASS_MAPPINGS,
            debug=True,
        )
        # Use a small subset of the test data
        self.test_df = test_df.head(3) if len(test_df) >= 3 else test_df

        # Get the binary classes from CROPLAND2
        self.classes_list = list(set(CLASS_MAPPINGS[LANDCOVER_KEY].values()))

        # Create datasets for testing
        self.binary_ds = WorldCerealLabelledDataset(
            self.test_df, task_type="binary", num_outputs=1
        )

        # For multiclass, use CROPTYPE9
        multiclass_df, _, _ = get_training_dfs_from_parquet(
            parquet_files,
            finetune_classes=CROPTYPE_KEY,
            class_mappings=CLASS_MAPPINGS,
            debug=True,
        )
        self.multiclass_df = multiclass_df.sample(20)

        self.multiclass_classes = list(set(CLASS_MAPPINGS[CROPTYPE_KEY].values()))
        # Filter to only classes that exist in our data
        self.multiclass_classes = [
            c
            for c in self.multiclass_classes
            if c in self.multiclass_df["finetune_class"].unique()
        ]

        self.multiclass_ds = WorldCerealLabelledDataset(
            self.multiclass_df,
            task_type="multiclass",
            num_outputs=len(self.multiclass_classes),
            classes_list=self.multiclass_classes,
        )

        # Create a time-explicit dataset
        self.time_explicit_ds = WorldCerealLabelledDataset(
            self.test_df, task_type="binary", num_outputs=1, time_explicit=True
        )

        # Loading the pretrained model is slow and not necessary for most test runs
        # We'll conditionally load it only when needed
        self.model_loaded = False

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _load_model_if_needed(self):
        """Conditionally load the pretrained model."""
        if not self.model_loaded:
            # Try to load a pretrained model or use a minimal one for testing
            from prometheo.models import Presto
            from prometheo.utils import device

            # This is a minimal model that can be used for testing
            # It will return random outputs but that's fine for testing the evaluation flow
            self.model = Presto(
                pretrained_model_path="https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt",
                num_outputs=1,
                regression=False,
            )

            # explicit is better than implicit
            self.model.to(device)

            self.model_loaded = True

    def test_evaluate_binary_classification(self):
        """Test evaluation for binary classification task using real data."""

        self._load_model_if_needed()

        # Run evaluation with minimal settings
        results_df, _, _ = evaluate_finetuned_model(
            self.model,
            self.binary_ds,
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues in tests
            batch_size=1,
        )

        # Check results dataframe has expected structure
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn("macro avg", results_df["class"].values)
        self.assertTrue("precision" in results_df.columns)
        self.assertTrue("recall" in results_df.columns)
        self.assertTrue("f1-score" in results_df.columns)
        self.assertTrue("support" in results_df.columns)

        # Check that we have results for all classes and the "macro avg" and "weighted avg"
        expected_classes = ["not_crop", "crop", "accuracy", "macro avg", "weighted avg"]
        for cls in expected_classes:
            self.assertTrue(
                any(results_df["class"] == cls), f"Missing expected class: {cls}"
            )

    def test_evaluate_multiclass_classification(self):
        """Test evaluation for multiclass classification task using real data."""
        from prometheo.models.presto.single_file_presto import FinetuningHead
        from prometheo.utils import device

        self._load_model_if_needed()

        self.model.head = FinetuningHead(
            hidden_size=self.model.encoder.embedding_size,
            num_outputs=len(self.multiclass_classes),
            regression=False,
        ).to(device)

        results_df, _, _ = evaluate_finetuned_model(
            self.model,
            self.multiclass_ds,
            num_workers=0,
            batch_size=1,
            classes_list=self.multiclass_classes,
        )

        self.assertIsInstance(results_df, pd.DataFrame)
        for cls in self.multiclass_classes:
            if not any(results_df["class"] == cls):
                print(
                    f"[WARNING] Expected class '{cls}' was not present in evaluation report."
                )

        # Check results dataframe has expected structure
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn("macro avg", results_df["class"].values)
        self.assertTrue("precision" in results_df.columns)
        self.assertTrue("recall" in results_df.columns)
        self.assertTrue("f1-score" in results_df.columns)
        self.assertTrue("support" in results_df.columns)

    def test_evaluate_time_explicit_classification(self):
        """Test evaluation for time-explicit binary classification."""
        self._load_model_if_needed()

        results_df, _, _ = evaluate_finetuned_model(
            self.model,
            self.time_explicit_ds,
            num_workers=0,
            batch_size=1,
            time_explicit=True,
        )

        # Check results dataframe has expected structure
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn("macro avg", results_df["class"].values)
        self.assertTrue("precision" in results_df.columns)
        self.assertTrue("recall" in results_df.columns)
        self.assertTrue("f1-score" in results_df.columns)
        self.assertTrue("support" in results_df.columns)

    # def test_evaluate_binary_with_mask_positions(self):
    #     """Binary eval with mask_positions should tag each block correctly."""
    #     self._load_model_if_needed()

    #     mask_positions = [1, 3]
    #     df_masked, _, _ = evaluate_finetuned_model(
    #         self.model,
    #         self.binary_ds,
    #         num_workers=0,
    #         batch_size=1,
    #         mask_positions=mask_positions,
    #     )

    #     # common report columns still present
    #     for col in ("class", "precision", "recall", "f1-score", "support"):
    #         self.assertIn(col, df_masked.columns)

    #     # new column and correct values
    #     self.assertIn("masked_ts_from_pos", df_masked.columns)
    #     self.assertEqual(
    #         set(df_masked["masked_ts_from_pos"].unique()), set(mask_positions)
    #     )

    #     # for each k, ensure at least the macro avg row exists
    #     for k in mask_positions:
    #         sub = df_masked[df_masked["masked_ts_from_pos"] == k]
    #         self.assertTrue(len(sub) > 0)
    #         self.assertIn("macro avg", sub["class"].values)

    # def test_evaluate_multiclass_with_mask_positions(self):
    #     """Multiclass eval with mask_positions should tag each block correctly."""
    #     from prometheo.models.presto.single_file_presto import FinetuningHead
    #     from prometheo.utils import device

    #     self._load_model_if_needed()
    #     # attach a head for multiclass
    #     self.model.head = FinetuningHead(
    #         hidden_size=self.model.encoder.embedding_size,
    #         num_outputs=len(self.multiclass_classes),
    #         regression=False,
    #     ).to(device)

    #     mask_positions = [2]
    #     df_masked, _, _ = evaluate_finetuned_model(
    #         self.model,
    #         self.multiclass_ds,
    #         num_workers=0,
    #         batch_size=1,
    #         classes_list=self.multiclass_classes,
    #         mask_positions=mask_positions,
    #     )

    #     # report columns
    #     for col in ("class", "precision", "recall", "f1-score", "support"):
    #         self.assertIn(col, df_masked.columns)
    #     # mask column
    #     self.assertIn("masked_ts_from_pos", df_masked.columns)
    #     self.assertEqual(
    #         set(df_masked["masked_ts_from_pos"].unique()), set(mask_positions)
    #     )

    #     # for position 2, ensure we see at least one expected class row
    #     sub = df_masked[df_masked["masked_ts_from_pos"] == 2]
    #     self.assertTrue(len(sub) > 0)
    #     found = any(cls in sub["class"].values for cls in self.multiclass_classes)
    #     self.assertTrue(found, "No expected multiclass label in masked report")


if __name__ == "__main__":
    unittest.main()
