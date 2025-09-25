import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from prometheo.finetune import Hyperparams
from worldcereal.train.models import build_worldcereal_presto
from prometheo.utils import device
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from worldcereal.train.data import get_training_dfs_from_parquet
from worldcereal.train.finetuning_utils import (
    evaluate_finetuned_model,
    prepare_training_datasets,
    run_finetuning,
)

LANDCOVER_KEY = "TEST_BINARY"
CROPTYPE_KEY = "CROPTYPE27"


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
        self,
        task_type,
        num_outputs,
        train_ds,
        val_ds,
        output_dir,
        *,
        temporal_attention: bool = False,
        apply_temporal_weights: bool = False,
    ):
        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False)

        model = build_worldcereal_presto(
            num_outputs=num_outputs,
            regression=False,
            pretrained_model_path=self.pretrained_model_path,
            temporal_attention=temporal_attention,
        ).to(device)

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
            task_type=task_type,
            optimizer=optimizer,
            scheduler=scheduler,
            hyperparams=hyperparams,
            setup_logging=False,
            apply_temporal_weights=apply_temporal_weights,
        )

    def _build_and_run_temporal_training(
        self,
        task_type,
        num_outputs,
        train_ds,
        val_ds,
        output_dir,
        *,
        temporal_attention: bool = True,
        apply_temporal_weights: bool = True,
    ):
        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False)

        model = build_worldcereal_presto(
            num_outputs=num_outputs,
            regression=False,
            pretrained_model_path=self.pretrained_model_path,
            temporal_attention=temporal_attention,
        ).to(device)

        hyperparams = Hyperparams(
            max_epochs=1, batch_size=64, patience=1, num_workers=2
        )
        optimizer = AdamW(model.parameters(), lr=0.01)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return run_finetuning(
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            experiment_name=f"{task_type}-temporal-test",
            output_dir=output_dir,
            task_type=task_type,
            optimizer=optimizer,
            scheduler=scheduler,
            hyperparams=hyperparams,
            setup_logging=False,
            apply_temporal_weights=apply_temporal_weights,
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
                    label = test_ds[i].label.squeeze()  # [T, 1]
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

    def test_temporal_kernel_gaussian_finetuning(self):
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
                return_time_weights=True,
                time_kernel="gaussian",
                time_kernel_bandwidth=1.0,
            )

            self.assertTrue(train_ds.return_time_weights)
            self.assertGreaterEqual(train_ds.label_window, 1)

            model = self._build_and_run_temporal_training(
                "binary", 1, train_ds, val_ds, Path(tmpdir)
            )
            results_df, _, _ = evaluate_finetuned_model(
                model, test_ds, num_workers=2, batch_size=64, time_explicit=True
            )
            self.assertFalse(
                results_df.isna().any().any(),
                "NaN values found in temporal evaluation results",
            )

    def test_prepare_training_datasets_temporal_kernel(self):
        train_df, val_df, test_df = get_training_dfs_from_parquet(
            self.parquet_files, finetune_classes=LANDCOVER_KEY, debug=True
        )
        train_ds, _, _ = prepare_training_datasets(
            train_df,
            val_df,
            test_df,
            task_type="binary",
            time_explicit=True,
            num_outputs=1,
            return_time_weights=True,
            time_kernel="gaussian",
            time_kernel_bandwidth=0.0,
        )
        self.assertTrue(train_ds.return_time_weights)
        self.assertGreaterEqual(
            train_ds.label_window, 1, "Label window should be expanded for temporal kernels"
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
        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False)

        model = build_worldcereal_presto(
            num_outputs=num_outputs,
            regression=False,
            pretrained_model_path=self.pretrained_model_path,
        ).to(device)

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
            task_type=task_type,
            optimizer=optimizer,
            scheduler=scheduler,
            hyperparams=hyperparams,
            setup_logging=False,
            apply_temporal_weights=False,
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
                    label = test_ds[i].label.squeeze()  # [T, 1]
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


if __name__ == "__main__":
    unittest.main()
