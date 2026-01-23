import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import List
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from prometheo.finetune import Hyperparams
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper
from prometheo.predictors import NODATAVALUE, Predictors
from prometheo.utils import device
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from worldcereal.train.data import collate_fn, get_training_dfs_from_parquet
from worldcereal.train.finetuning_utils import (
    SeasonalMultiTaskLoss,
    evaluate_finetuned_model,
    prepare_training_datasets,
    run_finetuning,
)
from worldcereal.train.seasonal_head import (
    SeasonalFinetuningHead,
    SeasonalHeadOutput,
    WorldCerealSeasonalModel,
)

LANDCOVER_KEY = "TEST_BINARY"
CROPTYPE_KEY = "CROPTYPE28"


def _sample_to_predictors(sample):
    if isinstance(sample, tuple):
        return sample[0]
    return sample


class TestFinetunePrestoEndToEnd(unittest.TestCase):
    def setUp(self):
        self.data_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "worldcereal"
            / "data"
            / "month"
        )
        self.parquet_files = list(self.data_path.rglob("*.geoparquet"))
        self.pretrained_model_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"

    def _build_and_run_training(
        self, task_type, num_outputs, train_ds, val_ds, output_dir
    ):
        train_dl = DataLoader(
            train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn
        )
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

        model = PretrainedPrestoWrapper(
            num_outputs=num_outputs,
            regression=False,
            pretrained_model_path=self.pretrained_model_path,
        ).to(device)

        if task_type == "multiclass":
            loss_fn = nn.CrossEntropyLoss(ignore_index=NODATAVALUE)
        if "binary" in task_type:
            loss_fn = nn.BCEWithLogitsLoss()
        hyperparams = Hyperparams(
            max_epochs=3, batch_size=64, patience=2, num_workers=2
        )
        optimizer = AdamW(model.parameters(), lr=0.01)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return run_finetuning(
            model,
            train_dl,
            val_dl,
            experiment_name=f"{task_type}-test",
            output_dir=output_dir,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            hyperparams=hyperparams,
            setup_logging=False,
        )

    def test_multiclass_finetuning(self):
        with TemporaryDirectory() as tmpdir:
            train_df, val_df, test_df = get_training_dfs_from_parquet(
                self.parquet_files, finetune_classes=CROPTYPE_KEY, debug=True
            )
            classes_list = list(train_df["finetune_class"].unique())

            train_ds, val_ds, test_ds = prepare_training_datasets(
                train_df,
                val_df,
                test_df,
                task_type="multiclass",
                num_outputs=len(classes_list),
                classes_list=classes_list,
            )
            model = self._build_and_run_training(
                "multiclass", len(classes_list), train_ds, val_ds, Path(tmpdir)
            )
            results_df, _, _ = evaluate_finetuned_model(
                model, test_ds, num_workers=2, batch_size=64, classes_list=classes_list
            )
            preds = results_df[
                ~results_df["class"].isin(["accuracy", "macro avg", "weighted avg"])
            ]
            preds = preds[preds["support"] > 0]
            self.assertGreaterEqual(
                len(preds),
                len(classes_list),
                "Multiclass model collapsed to few classes",
            )

    def test_binary_finetuning(self):
        with TemporaryDirectory() as tmpdir:
            train_df, val_df, test_df = get_training_dfs_from_parquet(
                self.parquet_files, finetune_classes=LANDCOVER_KEY, debug=True
            )
            train_ds, val_ds, test_ds = prepare_training_datasets(
                train_df, val_df, test_df, task_type="binary", num_outputs=1
            )
            model = self._build_and_run_training(
                "binary", 1, train_ds, val_ds, Path(tmpdir)
            )
            results_df, _, _ = evaluate_finetuned_model(
                model, test_ds, num_workers=2, batch_size=64
            )
            self.assertIn("crop", results_df["class"].values)
            self.assertIn("not_crop", results_df["class"].values)

    def test_time_explicit_binary_finetuning(self):
        with TemporaryDirectory() as tmpdir:
            train_df, val_df, test_df = get_training_dfs_from_parquet(
                self.parquet_files, finetune_classes=LANDCOVER_KEY, debug=True
            )
            train_ds, val_ds, test_ds = prepare_training_datasets(
                train_df,
                val_df,
                test_df,
                task_type="binary",
                time_explicit=True,
                num_outputs=1,
            )
            model = self._build_and_run_training(
                "binary-time-explicit", 1, train_ds, val_ds, Path(tmpdir)
            )
            results_df, _, _ = evaluate_finetuned_model(
                model, test_ds, num_workers=2, batch_size=64, time_explicit=True
            )

            self.assertTrue("macro avg" in results_df["class"].values)

            # 🚨 Extra Sanity Check: Consistency of valid labels
            with torch.no_grad():
                for i in range(min(10, len(test_ds))):
                    label = _sample_to_predictors(test_ds[i]).label.squeeze()
                    if label.ndim == 1:
                        label = label.reshape(-1, 1)
                    valid = (label != 65535).sum()
                    self.assertGreaterEqual(
                        valid, 1, "Time-explicit labels are all masked!"
                    )

            # 🚨 Extra Sanity Check: No NaNs in predictions
            self.assertFalse(
                results_df.isna().any().any(), "NaN values found in evaluation results"
            )


class TestFinetunePrestoEndToEndDekad(unittest.TestCase):
    def setUp(self):
        self.data_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "worldcereal"
            / "data"
            / "dekad"
        )
        self.parquet_files = list(self.data_path.rglob("*.geoparquet"))
        self.pretrained_model_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
        self.timestep_freq = "dekad"

    def _build_and_run_training(
        self, task_type, num_outputs, train_ds, val_ds, output_dir
    ):
        train_dl = DataLoader(
            train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn
        )
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

        model = PretrainedPrestoWrapper(
            num_outputs=num_outputs,
            regression=False,
            pretrained_model_path=self.pretrained_model_path,
        ).to(device)

        if task_type == "multiclass":
            loss_fn = nn.CrossEntropyLoss(ignore_index=NODATAVALUE)
        if "binary" in task_type:
            loss_fn = nn.BCEWithLogitsLoss()
        hyperparams = Hyperparams(
            max_epochs=3, batch_size=64, patience=2, num_workers=2
        )
        optimizer = AdamW(model.parameters(), lr=0.01)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return run_finetuning(
            model,
            train_dl,
            val_dl,
            experiment_name=f"{task_type}-test",
            output_dir=output_dir,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            hyperparams=hyperparams,
            setup_logging=False,
        )

    def test_multiclass_finetuning(self):
        with TemporaryDirectory() as tmpdir:
            train_df, val_df, test_df = get_training_dfs_from_parquet(
                self.parquet_files,
                timestep_freq=self.timestep_freq,
                finetune_classes=CROPTYPE_KEY,
                debug=True,
            )
            classes_list = list(train_df["finetune_class"].unique())

            train_ds, val_ds, test_ds = prepare_training_datasets(
                train_df,
                val_df,
                test_df,
                timestep_freq=self.timestep_freq,
                num_timesteps=36,
                task_type="multiclass",
                num_outputs=len(classes_list),
                classes_list=classes_list,
            )
            model = self._build_and_run_training(
                "multiclass", len(classes_list), train_ds, val_ds, Path(tmpdir)
            )
            results_df, _, _ = evaluate_finetuned_model(
                model, test_ds, num_workers=2, batch_size=64, classes_list=classes_list
            )
            preds = results_df[
                ~results_df["class"].isin(["accuracy", "macro avg", "weighted avg"])
            ]
            preds = preds[preds["support"] > 0]
            self.assertGreaterEqual(
                len(preds),
                len(classes_list),
                "Multiclass model collapsed to few classes",
            )

    def test_binary_finetuning(self):
        with TemporaryDirectory() as tmpdir:
            train_df, val_df, test_df = get_training_dfs_from_parquet(
                self.parquet_files,
                timestep_freq=self.timestep_freq,
                finetune_classes=LANDCOVER_KEY,
                debug=True,
            )
            train_ds, val_ds, test_ds = prepare_training_datasets(
                train_df,
                val_df,
                test_df,
                timestep_freq=self.timestep_freq,
                num_timesteps=36,
                task_type="binary",
                num_outputs=1,
            )
            model = self._build_and_run_training(
                "binary", 1, train_ds, val_ds, Path(tmpdir)
            )
            results_df, _, _ = evaluate_finetuned_model(
                model, test_ds, num_workers=2, batch_size=64
            )
            self.assertIn("crop", results_df["class"].values)
            self.assertIn("not_crop", results_df["class"].values)

    def test_time_explicit_binary_finetuning(self):
        with TemporaryDirectory() as tmpdir:
            train_df, val_df, test_df = get_training_dfs_from_parquet(
                self.parquet_files,
                timestep_freq=self.timestep_freq,
                finetune_classes=LANDCOVER_KEY,
                debug=True,
            )
            train_ds, val_ds, test_ds = prepare_training_datasets(
                train_df,
                val_df,
                test_df,
                timestep_freq=self.timestep_freq,
                num_timesteps=36,
                task_type="binary",
                time_explicit=True,
                num_outputs=1,
            )
            model = self._build_and_run_training(
                "binary-time-explicit", 1, train_ds, val_ds, Path(tmpdir)
            )
            results_df, _, _ = evaluate_finetuned_model(
                model, test_ds, num_workers=2, batch_size=64, time_explicit=True
            )

            self.assertTrue("macro avg" in results_df["class"].values)

            # 🚨 Extra Sanity Check: Consistency of valid labels
            with torch.no_grad():
                for i in range(min(10, len(test_ds))):
                    label = _sample_to_predictors(test_ds[i]).label.squeeze()
                    if label.ndim == 1:
                        label = label.reshape(-1, 1)
                    valid = (label != 65535).sum()
                    self.assertGreaterEqual(
                        valid, 1, "Time-explicit labels are all masked!"
                    )

            # 🚨 Extra Sanity Check: No NaNs in predictions
            self.assertFalse(
                results_df.isna().any().any(), "NaN values found in evaluation results"
            )


class TestValidationImprovementCallback(unittest.TestCase):
    def test_callback_runs_only_on_improvements(self):
        model = _ToyBinaryModel().to(device)
        train_batch = _build_binary_predictor_batch()
        val_batch = _build_binary_predictor_batch()
        train_loader = _MemoryLoader([train_batch], task_type="binary")
        val_loader = _MemoryLoader([val_batch], task_type="binary")

        loss_fn = _CallbackProbeLoss(val_losses=[0.9, 0.8, 0.8])
        hyperparams = Hyperparams(max_epochs=3, batch_size=4, patience=3, num_workers=0)
        optimizer = AdamW(model.parameters(), lr=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        callback_calls = []

        def _on_improved(epoch_idx, model_snapshot, best_loss):
            callback_calls.append((epoch_idx, float(best_loss)))
            self.assertFalse(
                model_snapshot.training,
                "Model should remain in eval mode while running the callback",
            )
            context = getattr(model_snapshot, "_last_validation_context", None)
            self.assertIsNotNone(context)
            self.assertIn("metrics", context)

        with TemporaryDirectory() as tmpdir:
            run_finetuning(
                model=model,
                train_dl=train_loader,
                val_dl=val_loader,
                experiment_name="callback-test",
                output_dir=Path(tmpdir),
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                hyperparams=hyperparams,
                setup_logging=False,
                on_validation_improved=_on_improved,
            )

        self.assertEqual([1, 2], [epoch for epoch, _ in callback_calls])
        self.assertEqual(2, len(callback_calls))


SEASONAL_TIMESTEPS = 3
SEASONAL_LANDCOVER_CLASSES = ["background", "cropland"]
SEASONAL_CROPTYPE_CLASSES = ["maize", "wheat"]
_SEASONAL_MASK_TEMPLATES = [
    torch.tensor([[True, True, False], [False, False, True]], dtype=torch.bool),
    torch.tensor([[False, True, True], [True, False, False]], dtype=torch.bool),
]


class _MemoryLoader:
    def __init__(self, batches, task_type=None):
        self._batches = batches
        self.dataset = SimpleNamespace(task_type=task_type)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _build_binary_predictor_batch(batch_size: int = 4):
    label = torch.zeros(batch_size, 1, dtype=torch.float32)
    timestamps = torch.zeros(batch_size, 1, 3, dtype=torch.float32)
    return Predictors(label=label, timestamps=timestamps)


class _ToyBinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, predictors: Predictors, attrs=None):
        if predictors.label is None:
            raise ValueError("Predictors must include labels for the toy model")
        logits = torch.ones_like(predictors.label, dtype=torch.float32)
        logits = logits.to(self.weight.device)
        return logits * self.weight


class _CallbackProbeLoss(nn.Module):
    def __init__(self, val_losses: List[float]):
        super().__init__()
        if not val_losses:
            raise ValueError("val_losses must contain at least one value")
        self.val_losses = val_losses
        self.val_call_idx = 0

    def forward(self, inputs, targets):
        base = inputs.sum() * 0
        if torch.is_grad_enabled():
            value = 1.0
        else:
            idx = min(self.val_call_idx, len(self.val_losses) - 1)
            value = self.val_losses[idx]
            self.val_call_idx += 1
        loss_value = torch.tensor(float(value), device=inputs.device)
        return base + loss_value


class _DummySeasonalBackbone(nn.Module):
    def __init__(self, timesteps: int, embedding_dim: int):
        super().__init__()
        self.timesteps = timesteps
        self.proj = nn.Linear(1, embedding_dim)
        self.encoder = nn.Identity()
        setattr(self.encoder, "embedding_size", embedding_dim)

    def forward(self, predictors: Predictors, eval_pooling=None):
        if predictors.label is None:
            raise ValueError("Synthetic predictors must include labels")
        label_tensor = predictors.label.to(self.proj.weight.device)
        batch_size = label_tensor.shape[0]
        features = label_tensor.view(batch_size, self.timesteps, 1)
        return self.proj(features)


def _build_synthetic_seasonal_batch(label_tasks: List[str]):
    batch_size = len(label_tasks)
    label_values = torch.arange(
        batch_size * SEASONAL_TIMESTEPS, dtype=torch.float32
    ).view(batch_size, 1, 1, SEASONAL_TIMESTEPS, 1)
    timestamps = torch.zeros(batch_size, SEASONAL_TIMESTEPS, 3, dtype=torch.float32)
    predictors = Predictors(label=label_values, timestamps=timestamps)

    season_masks = torch.stack(
        [
            _SEASONAL_MASK_TEMPLATES[i % len(_SEASONAL_MASK_TEMPLATES)].clone()
            for i in range(batch_size)
        ],
        dim=0,
    )
    in_seasons = torch.stack(
        [
            torch.tensor([True, False], dtype=torch.bool)
            if task == "landcover"
            else torch.tensor([False, True], dtype=torch.bool)
            for task in label_tasks
        ],
        dim=0,
    )

    attrs = {
        "season_masks": season_masks,
        "in_seasons": in_seasons,
        "valid_position": [min(i, SEASONAL_TIMESTEPS - 1) for i in range(batch_size)],
        "landcover_label": [
            SEASONAL_LANDCOVER_CLASSES[i % len(SEASONAL_LANDCOVER_CLASSES)]
            for i in range(batch_size)
        ],
        "croptype_label": [
            SEASONAL_CROPTYPE_CLASSES[i % len(SEASONAL_CROPTYPE_CLASSES)]
            for i in range(batch_size)
        ],
        "label_task": label_tasks,
    }
    return predictors, attrs


class TestSeasonalHeadFinetuning(unittest.TestCase):
    def test_dual_branch_seasonal_training(self):
        backbone = _DummySeasonalBackbone(timesteps=SEASONAL_TIMESTEPS, embedding_dim=8)
        head = SeasonalFinetuningHead(
            embedding_dim=8,
            landcover_num_outputs=len(SEASONAL_LANDCOVER_CLASSES),
            crop_num_outputs=len(SEASONAL_CROPTYPE_CLASSES),
        )
        model = WorldCerealSeasonalModel(backbone=backbone, head=head).to(device)

        loss_fn = SeasonalMultiTaskLoss(
            landcover_classes=SEASONAL_LANDCOVER_CLASSES,
            croptype_classes=SEASONAL_CROPTYPE_CLASSES,
        )

        train_batches = [
            _build_synthetic_seasonal_batch(["landcover", "croptype"]),
            _build_synthetic_seasonal_batch(["croptype", "landcover"]),
        ]
        val_batches = [
            _build_synthetic_seasonal_batch(["landcover", "croptype"]),
        ]
        train_loader = _MemoryLoader(train_batches)
        val_loader = _MemoryLoader(val_batches)

        hyperparams = Hyperparams(max_epochs=2, batch_size=2, patience=1, num_workers=0)
        optimizer = AdamW(model.parameters(), lr=0.05)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        with TemporaryDirectory() as tmpdir:
            trained_model = run_finetuning(
                model=model,
                train_dl=train_loader,
                val_dl=val_loader,
                experiment_name="seasonal-dual-test",
                output_dir=Path(tmpdir),
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                hyperparams=hyperparams,
                setup_logging=False,
            )

            sample_predictors, sample_attrs = _build_synthetic_seasonal_batch(
                ["landcover", "croptype"]
            )
            trained_model.zero_grad(set_to_none=True)
            output = trained_model(sample_predictors, attrs=sample_attrs)

            self.assertIsNotNone(output.global_logits)
            self.assertIsNotNone(output.season_logits)
            self.assertEqual(
                output.global_logits.shape[-1], len(SEASONAL_LANDCOVER_CLASSES)
            )
            self.assertEqual(
                output.season_logits.shape[-1], len(SEASONAL_CROPTYPE_CLASSES)
            )

            loss = loss_fn(output, sample_predictors, sample_attrs)
            self.assertGreater(loss.item(), 0.0)

            loss.backward()

            landcover_grads = [
                p.grad for p in trained_model.head.landcover_head.parameters()
            ]
            croptype_grads = [p.grad for p in trained_model.head.crop_head.parameters()]

            self.assertTrue(
                any(g is not None and torch.any(g != 0) for g in landcover_grads),
                "Landcover branch did not receive gradients",
            )
            self.assertTrue(
                any(g is not None and torch.any(g != 0) for g in croptype_grads),
                "Crop-type branch did not receive gradients",
            )


class TestSeasonalLossWeighting(unittest.TestCase):
    def test_sample_weights_rescale_branch_losses(self):
        output = SeasonalHeadOutput(
            global_logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            season_logits=torch.tensor(
                [[[0.0, 1.0]], [[1.5, 0.5]]], dtype=torch.float32
            ),
            global_embedding=torch.zeros(2, 2),
            season_embeddings=torch.zeros(2, 1, 2),
            season_masks=torch.ones(2, 1, 1, dtype=torch.bool),
        )
        attrs = {
            "landcover_label": ["forest", None],
            "croptype_label": [None, "wheat"],
            "label_task": ["landcover", "croptype"],
            "valid_position": [0, 0],
            "in_seasons": np.ones((2, 1), dtype=bool),
            "quality_score_lc": [0.5, None],
            "quality_score_ct": [None, 2.0],
        }

        loss_fn = SeasonalMultiTaskLoss(
            landcover_classes=["forest", "water"],
            croptype_classes=["wheat", "maize"],
            task_sample_weight_attrs={
                "landcover": "quality_score_lc",
                "croptype": "quality_score_ct",
            },
            sample_weight_clip=(0.1, 5.0),
            sample_weight_default=1.0,
        )

        predictors = mock.MagicMock(spec=Predictors)
        weighted_loss = loss_fn(output, predictors, attrs)

        lc_ce = F.cross_entropy(
            output.global_logits[0:1], torch.tensor([0]), reduction="none"
        )
        lc_expected = torch.sum(lc_ce * torch.tensor([0.5])) / 0.5

        ct_logits = output.season_logits[1, 0:1, :]
        ct_ce = F.cross_entropy(ct_logits, torch.tensor([0]), reduction="none")
        ct_expected = torch.sum(ct_ce * torch.tensor([2.0])) / 2.0

        expected_total = lc_expected + ct_expected
        self.assertAlmostEqual(weighted_loss.item(), expected_total.item(), places=6)


if __name__ == "__main__":
    unittest.main()
