import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from prometheo.predictors import NODATAVALUE, Predictors
from torch import nn
from torch.utils.data import Dataset

from worldcereal.train.data import get_training_dfs_from_parquet
from worldcereal.train.datasets import (
    WorldCerealLabelledDataset,
    # MaskingMode,
    # MaskingStrategy,
)
from worldcereal.train.finetuning_utils import (
    _select_representative_season,
    evaluate_finetuned_model,
    prepare_training_datasets,
)
from worldcereal.train.seasonal_head import SeasonalHeadOutput
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
        predictors, attrs = train_ds[0]
        self.assertIsNotNone(predictors)
        self.assertTrue(hasattr(predictors, "label"))
        self.assertIn("season_masks", attrs)

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
            sample, attrs = train_ds[0]
            self.assertEqual(sample.label.shape, (1, 1, 1, 1))  # (H, W, T, 1)
            self.assertIn("season_masks", attrs)

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
            sample, attrs = train_ds[0]
            # Check that label has temporal dimension
            self.assertEqual(sample.label.shape[2], train_ds.num_timesteps)
            # Verify correct shape using expected timesteps
            timesteps = 12  # Default number of timesteps in InSeasonLabelledDataset
            self.assertEqual(
                sample.label.shape, (1, 1, timesteps, train_ds.num_outputs)
            )  # (H, W, T, num_classes)
            self.assertIn("season_masks", attrs)


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

        # Create a multiclass time-explicit dataset
        self.multiclass_time_explicit_ds = WorldCerealLabelledDataset(
            self.multiclass_df,
            task_type="multiclass",
            num_outputs=len(self.multiclass_classes),
            classes_list=self.multiclass_classes,
            time_explicit=True,
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

    class TestSeasonSelectionHelper(unittest.TestCase):
        def test_returns_all_overlapping_seasons(self):
            output = SeasonalHeadOutput(
                global_logits=None,
                season_logits=None,
                global_embedding=torch.zeros(1, 4),
                season_embeddings=torch.zeros(1, 2, 4),
                season_masks=torch.tensor([[[True, False], [True, False]]]),
            )
            attrs = {
                "in_seasons": np.array([[True, True]], dtype=bool),
                "valid_position": np.array([0]),
            }
            # With allow_multiple=True, should return both seasons
            selections = _select_representative_season(
                output, attrs, [0], allow_multiple=True
            )
            self.assertEqual(selections[0], [0, 1])

            # With allow_multiple=False, should return only the first season
            single = _select_representative_season(
                output, attrs, [0], allow_multiple=False
            )
            self.assertTrue(torch.equal(single.cpu(), torch.tensor([0])))

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

    def test_evaluate_multiclass_time_explicit_classification(self):
        """Test evaluation for time-explicit multiclass classification."""
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
            self.multiclass_time_explicit_ds,
            num_workers=0,
            batch_size=1,
            time_explicit=True,
            classes_list=self.multiclass_classes,
        )

        # Check results dataframe has expected structure
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn("macro avg", results_df["class"].values)
        self.assertTrue("precision" in results_df.columns)
        self.assertTrue("recall" in results_df.columns)
        self.assertTrue("f1-score" in results_df.columns)
        self.assertTrue("support" in results_df.columns)


class _DummySeasonalDataset(Dataset):
    task_type = "multiclass"
    time_explicit = False

    def __init__(self):
        base_mask = np.array([[True, False]], dtype=bool)
        base_in = np.array([True], dtype=bool)
        self.samples = [
            {
                "label": 0,
                "attrs": {
                    "landcover_label": "temporary_crops",
                    "croptype_label": "wheat",
                    "label_task": "landcover",
                    "season_masks": base_mask.copy(),
                    "in_seasons": base_in.copy(),
                    "valid_position": 0,
                },
            },
            {
                "label": 0,
                "attrs": {
                    "landcover_label": "temporary_crops",
                    "croptype_label": "wheat",
                    "label_task": "croptype",
                    "season_masks": base_mask.copy(),
                    "in_seasons": base_in.copy(),
                    "valid_position": 0,
                },
            },
            {
                "label": 0,
                "attrs": {
                    "landcover_label": "temporary_crops",
                    "croptype_label": "maize",
                    "label_task": "croptype",
                    "season_masks": base_mask.copy(),
                    "in_seasons": base_in.copy(),
                    "valid_position": 0,
                },
            },
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label_tensor = torch.tensor(sample["label"], dtype=torch.long).view(
            1, 1, 1, 1, 1
        )
        attrs = {}
        for key, value in sample["attrs"].items():
            if isinstance(value, np.ndarray):
                attrs[key] = value.copy()
            else:
                attrs[key] = value
        return Predictors(label=label_tensor), attrs


class _DummySeasonalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_logits = torch.tensor(
            [
                [5.0, 1.0],
                [4.0, 2.0],
                [1.0, 5.0],
            ]
        )
        self.season_logits = torch.tensor(
            [
                [[5.0, 1.0]],
                [[5.0, 1.0]],
                [[1.0, 5.0]],
            ]
        )

    def forward(self, predictors, attrs=None):
        assert attrs is not None
        batch = attrs["season_masks"].shape[0]
        season_masks = torch.as_tensor(attrs["season_masks"], dtype=torch.bool)
        global_logits = self.global_logits[:batch]
        season_logits = self.season_logits[:batch]
        emb_dim = 4
        global_embedding = torch.zeros(batch, emb_dim)
        season_embeddings = torch.zeros(batch, season_logits.shape[1], emb_dim)
        return SeasonalHeadOutput(
            global_logits=global_logits,
            season_logits=season_logits,
            global_embedding=global_embedding,
            season_embeddings=season_embeddings,
            season_masks=season_masks,
        )


class TestSeasonalEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataset = _DummySeasonalDataset()
        self.model = _DummySeasonalModel()
        self.landcover_classes = ["temporary_crops", "water"]
        self.croptype_classes = ["wheat", "maize"]

    def test_seasonal_evaluation_with_gating(self):
        results = evaluate_finetuned_model(
            self.model,
            self.dataset,
            num_workers=0,
            batch_size=len(self.dataset),
            seasonal_landcover_classes=self.landcover_classes,
            seasonal_croptype_classes=self.croptype_classes,
            cropland_class_names=["temporary_crops"],
        )
        self.assertIn("landcover", results)
        self.assertIn("croptype", results)

        landcover_df = results["landcover"]["results"]
        self.assertFalse(landcover_df.empty)

        croptype_df = results["croptype"]["results"]
        gate_row = croptype_df[croptype_df["class"] == "croptype_gate_rejections"]
        self.assertFalse(gate_row.empty)
        self.assertEqual(int(gate_row["support"].iloc[0]), 1)
        self.assertEqual(results["croptype"]["gate_rejections"], 1)
        self.assertEqual(results["croptype"]["num_samples"], 1)


if __name__ == "__main__":
    unittest.main()
