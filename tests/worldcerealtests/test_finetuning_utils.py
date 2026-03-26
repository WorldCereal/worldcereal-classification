import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from prometheo.predictors import NODATAVALUE, Predictors
from torch import nn
from torch.utils.data import Dataset
from worldcereal.train.data import get_training_dfs_from_parquet
from worldcereal.train.datasets import \
    WorldCerealLabelledDataset  # MaskingMode,; MaskingStrategy,
from worldcereal.train.finetuning_utils import (SeasonalMultiTaskLoss,
                                                _select_representative_season,
                                                evaluate_finetuned_model,
                                                prepare_training_datasets)
from worldcereal.train.seasonal_head import SeasonalHeadOutput
from worldcereal.utils.refdata import get_class_mappings

CLASS_MAPPINGS = get_class_mappings()
LANDCOVER_KEY = "LANDCOVER10"
CROPTYPE_KEY = "CROPTYPE28"


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
        gate_mask = croptype_df["class"] == "croptype_gate_rejections"
        self.assertFalse(gate_mask.any())
        self.assertEqual(results["croptype"]["gate_rejections"], 0)
        self.assertEqual(results["croptype"]["num_samples"], 2)


class TestSeasonalMultiTaskLoss(unittest.TestCase):
    """Unit tests for the SeasonalMultiTaskLoss forward pass."""

    def setUp(self):
        self.lc_classes = ["temporary_crops", "water"]
        self.ct_classes = ["wheat", "maize"]
        self.loss_fn = SeasonalMultiTaskLoss(
            landcover_classes=self.lc_classes,
            croptype_classes=self.ct_classes,
        )
        # Shared season mask: 1 season, 2 timesteps, season active at t=0
        self.base_mask = np.array([[True, False]], dtype=bool)
        self.base_in = np.array([True], dtype=bool)

    def _make_output(self, batch, n_seasons=1, n_lc=2, n_ct=2, emb_dim=4):
        """Build a SeasonalHeadOutput with random logits."""
        return SeasonalHeadOutput(
            global_logits=torch.randn(batch, n_lc),
            season_logits=torch.randn(batch, n_seasons, n_ct),
            global_embedding=torch.randn(batch, emb_dim),
            season_embeddings=torch.randn(batch, n_seasons, emb_dim),
            season_masks=torch.ones(batch, n_seasons, 2, dtype=torch.bool),
        )

    def _make_attrs(self, label_tasks, lc_labels, ct_labels):
        """Build an attrs dict mimicking DataLoader collation."""
        batch = len(label_tasks)
        return {
            "label_task": label_tasks,
            "landcover_label": lc_labels,
            "croptype_label": ct_labels,
            "season_masks": np.tile(self.base_mask, (batch, 1, 1)),
            "in_seasons": np.tile(self.base_in, (batch, 1)),
            "valid_position": [0] * batch,
        }

    def test_lc_only_batch(self):
        """A batch with only LC-assigned samples should produce LC loss only."""
        output = self._make_output(2)
        attrs = self._make_attrs(
            label_tasks=["landcover", "landcover"],
            lc_labels=["temporary_crops", "water"],
            ct_labels=[None, None],
        )
        predictors = Predictors(label=torch.zeros(2, 1, 1, 1, 1))
        loss = self.loss_fn(output, predictors, attrs)

        self.assertGreater(loss.item(), 0.0)
        self.assertIn("landcover", self.loss_fn.last_task_losses)
        self.assertNotIn("croptype", self.loss_fn.last_task_losses)

    def test_ct_only_batch(self):
        """A batch with only CT-assigned samples should produce CT loss only."""
        output = self._make_output(2)
        attrs = self._make_attrs(
            label_tasks=["croptype", "croptype"],
            lc_labels=["temporary_crops", "water"],
            ct_labels=["wheat", "maize"],
        )
        predictors = Predictors(label=torch.zeros(2, 1, 1, 1, 1))
        loss = self.loss_fn(output, predictors, attrs)

        self.assertGreater(loss.item(), 0.0)
        self.assertNotIn("landcover", self.loss_fn.last_task_losses)
        self.assertIn("croptype", self.loss_fn.last_task_losses)

    def test_mixed_batch(self):
        """A mixed LC+CT batch should produce both loss branches."""
        output = self._make_output(4)
        attrs = self._make_attrs(
            label_tasks=["landcover", "landcover", "croptype", "croptype"],
            lc_labels=["temporary_crops", "water", "temporary_crops", "water"],
            ct_labels=[None, None, "wheat", "maize"],
        )
        predictors = Predictors(label=torch.zeros(4, 1, 1, 1, 1))
        loss = self.loss_fn(output, predictors, attrs)

        self.assertGreater(loss.item(), 0.0)
        self.assertIn("landcover", self.loss_fn.last_task_losses)
        self.assertIn("croptype", self.loss_fn.last_task_losses)

    def test_loss_is_differentiable(self):
        """Loss should propagate gradients back through the logits."""
        output = self._make_output(2)
        output.global_logits.requires_grad_(True)
        output.season_logits.requires_grad_(True)
        attrs = self._make_attrs(
            label_tasks=["landcover", "croptype"],
            lc_labels=["temporary_crops", "water"],
            ct_labels=[None, "wheat"],
        )
        predictors = Predictors(label=torch.zeros(2, 1, 1, 1, 1))
        loss = self.loss_fn(output, predictors, attrs)
        loss.backward()

        self.assertIsNotNone(output.global_logits.grad)
        self.assertIsNotNone(output.season_logits.grad)

    def test_croptype_supervision_stats(self):
        """last_croptype_supervision should track eligible and supervised counts."""
        output = self._make_output(3)
        attrs = self._make_attrs(
            label_tasks=["landcover", "croptype", "croptype"],
            lc_labels=["temporary_crops", "water", "temporary_crops"],
            ct_labels=[None, "wheat", "maize"],
        )
        predictors = Predictors(label=torch.zeros(3, 1, 1, 1, 1))
        self.loss_fn(output, predictors, attrs)

        stats = self.loss_fn.last_croptype_supervision
        self.assertEqual(stats["eligible_samples"], 2.0)
        self.assertEqual(stats["supervised_samples"], 2.0)
        self.assertEqual(stats["missing_representative_season"], 0.0)

    def test_no_season_available_tracks_missing(self):
        """Samples with no active season should be counted as missing."""
        output = self._make_output(2, n_seasons=1)
        # Override season_masks to all-False so no season is active
        output.season_masks = torch.zeros(2, 1, 2, dtype=torch.bool)
        attrs = self._make_attrs(
            label_tasks=["croptype", "croptype"],
            lc_labels=["temporary_crops", "water"],
            ct_labels=["wheat", "maize"],
        )
        # Also clear in_seasons so the fallback via valid_position finds nothing
        attrs["in_seasons"] = np.array([[False], [False]], dtype=bool)
        predictors = Predictors(label=torch.zeros(2, 1, 1, 1, 1))
        self.loss_fn(output, predictors, attrs)

        stats = self.loss_fn.last_croptype_supervision
        self.assertEqual(stats["eligible_samples"], 2.0)
        self.assertEqual(stats["missing_representative_season"], 2.0)
        self.assertEqual(stats["supervised_samples"], 0.0)

    def test_sample_weights_applied(self):
        """When sample weight attrs are configured, loss should change."""
        loss_weighted = SeasonalMultiTaskLoss(
            landcover_classes=self.lc_classes,
            croptype_classes=self.ct_classes,
            task_sample_weight_attrs={"landcover": "quality_score_lc"},
            sample_weight_clip=(0.1, 10.0),
        )
        # Asymmetric logits so per-sample losses differ
        logits = torch.tensor([[5.0, 1.0], [1.0, 2.0]])
        output_a = SeasonalHeadOutput(
            global_logits=logits.clone(),
            season_logits=None,
            global_embedding=torch.zeros(2, 4),
            season_embeddings=torch.zeros(2, 1, 4),
            season_masks=torch.ones(2, 1, 2, dtype=torch.bool),
        )
        output_b = SeasonalHeadOutput(
            global_logits=logits.clone(),
            season_logits=None,
            global_embedding=torch.zeros(2, 4),
            season_embeddings=torch.zeros(2, 1, 4),
            season_masks=torch.ones(2, 1, 2, dtype=torch.bool),
        )
        base_attrs = {
            "label_task": ["landcover", "landcover"],
            "landcover_label": ["temporary_crops", "water"],
            "croptype_label": [None, None],
            "season_masks": np.tile(self.base_mask, (2, 1, 1)),
            "in_seasons": np.tile(self.base_in, (2, 1)),
            "valid_position": [0, 0],
        }
        p = Predictors(label=torch.zeros(2, 1, 1, 1, 1))

        # Uniform weights
        attrs_uniform = {**base_attrs, "quality_score_lc": [1.0, 1.0]}
        loss_uniform = loss_weighted(output_a, p, attrs_uniform)

        # Skewed weights — second sample gets 10× the weight
        attrs_skewed = {**base_attrs, "quality_score_lc": [1.0, 10.0]}
        loss_skewed = loss_weighted(output_b, p, attrs_skewed)

        # Losses should differ because of different weighting
        self.assertFalse(
            torch.allclose(loss_uniform, loss_skewed),
            "Sample weights should change the loss value",
        )


class TestSeasonSelectionPartialInSeasons(unittest.TestCase):
    """Verify _select_representative_season with partial in_seasons flags."""

    def _make_output(self, n_seasons=2, n_timesteps=2):
        return SeasonalHeadOutput(
            global_logits=None,
            season_logits=None,
            global_embedding=torch.zeros(1, 4),
            season_embeddings=torch.zeros(1, n_seasons, 4),
            season_masks=torch.ones(1, n_seasons, n_timesteps, dtype=torch.bool),
        )

    def test_first_season_only(self):
        """in_seasons=[True, False] should select only season 0."""
        output = self._make_output()
        attrs = {
            "in_seasons": np.array([[True, False]], dtype=bool),
            "valid_position": np.array([0]),
        }
        selections = _select_representative_season(
            output, attrs, [0], allow_multiple=True
        )
        self.assertEqual(selections[0], [0])

    def test_second_season_only(self):
        """in_seasons=[False, True] should select only season 1."""
        output = self._make_output()
        attrs = {
            "in_seasons": np.array([[False, True]], dtype=bool),
            "valid_position": np.array([0]),
        }
        selections = _select_representative_season(
            output, attrs, [0], allow_multiple=True
        )
        self.assertEqual(selections[0], [1])

    def test_no_season_active(self):
        """in_seasons=[False, False] with masks also False should return empty."""
        output = self._make_output()
        output.season_masks = torch.zeros(1, 2, 2, dtype=torch.bool)
        attrs = {
            "in_seasons": np.array([[False, False]], dtype=bool),
            "valid_position": np.array([0]),
        }
        selections = _select_representative_season(
            output, attrs, [0], allow_multiple=True
        )
        self.assertEqual(selections[0], [])

    def test_mixed_batch_different_patterns(self):
        """Two samples with different in_seasons patterns in the same batch."""
        output = SeasonalHeadOutput(
            global_logits=None,
            season_logits=None,
            global_embedding=torch.zeros(2, 4),
            season_embeddings=torch.zeros(2, 2, 4),
            season_masks=torch.ones(2, 2, 2, dtype=torch.bool),
        )
        attrs = {
            "in_seasons": np.array([[True, False], [False, True]], dtype=bool),
            "valid_position": np.array([0, 0]),
        }
        selections = _select_representative_season(
            output, attrs, [0, 1], allow_multiple=True
        )
        self.assertEqual(selections[0], [0])  # sample 0: season 0 only
        self.assertEqual(selections[1], [1])  # sample 1: season 1 only


class TestSeasonalLossTwoSeasonPartialCoverage(unittest.TestCase):
    """Verify SeasonalMultiTaskLoss uses only active season logits for CT loss."""

    def test_ct_loss_uses_only_active_season(self):
        """With in_seasons=[True, False], only season 0 logits should contribute.

        We set season 0 logits to predict class 0 confidently (low loss) and
        season 1 logits to predict class 1 confidently (high loss if target=0).
        If the loss incorrectly uses season 1, the loss would be much higher.
        """
        loss_fn = SeasonalMultiTaskLoss(
            landcover_classes=["temporary_crops", "water"],
            croptype_classes=["wheat", "maize"],
        )

        # Season 0: confident wheat (class 0); Season 1: confident maize (class 1)
        season_logits = torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]])  # [1, 2, 2]
        output = SeasonalHeadOutput(
            global_logits=torch.zeros(1, 2),
            season_logits=season_logits,
            global_embedding=torch.zeros(1, 4),
            season_embeddings=torch.zeros(1, 2, 4),
            season_masks=torch.ones(1, 2, 2, dtype=torch.bool),
        )
        attrs = {
            "label_task": ["croptype"],
            "landcover_label": ["temporary_crops"],
            "croptype_label": ["wheat"],  # target = class 0
            "season_masks": np.ones((1, 2, 2), dtype=bool),
            "in_seasons": np.array([[True, False]], dtype=bool),  # only season 0
            "valid_position": [0],
        }
        p = Predictors(label=torch.zeros(1, 1, 1, 1, 1))
        loss_active_only = loss_fn(output, p, attrs)

        # Now flip: only season 1 active (which predicts maize, but target is wheat)
        attrs_flipped = {**attrs, "in_seasons": np.array([[False, True]], dtype=bool)}
        loss_wrong_season = loss_fn(output, p, attrs_flipped)

        # Season 0 predicts wheat correctly → low loss
        # Season 1 predicts maize but target is wheat → high loss
        self.assertGreater(
            loss_wrong_season.item(),
            loss_active_only.item(),
            "Loss with the wrong season active should be higher",
        )


if __name__ == "__main__":
    unittest.main()
