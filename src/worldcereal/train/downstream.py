#!/usr/bin/env python3
"""Train a PyTorch classification head (Linear or small MLP) on embeddings."""

import json
import zipfile
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from numpy.typing import NDArray
from prometheo.utils import device
from seaborn import heatmap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm.auto import tqdm

from worldcereal.train.data import train_val_test_split
from worldcereal.train.datasets import get_class_weights


class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feat_prefix: str = "presto_ft_",
        label_col: str = "downstream_class",
    ):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.feat_cols = sorted([c for c in df.columns if c.startswith(feat_prefix)])
        logger.info(
            f"EmbeddingsDataset: {len(self.df)} samples, {len(self.feat_cols)} features"
        )
        self.X = self.df[self.feat_cols].to_numpy(dtype=np.float32)
        self.y = self.df[label_col].to_numpy()

    def __repr__(self) -> str:
        return (
            f"EmbeddingsDataset(num_samples={len(self.df)}, "
            f"num_features={len(self.feat_cols)}), label_col={self.label_col})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(
        self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchTrainer:
    def __init__(
        self,
        embeddings_df: pd.DataFrame,
        split_column: str = "split",
        head_type: str = "linear",
        output_dir: Union[Path, str] = "./downstream_classifier",
        modelversion: str = "",
        detector: str = "cropland",
        downstream_classes: Optional[dict] = None,
        batch_size: int = 1024,
        num_workers: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-2,  # can be high for lightweight head
        epochs: int = 30,
        seed: int = 42,
        label_smoothing: float = 0.0,
        use_balancing: bool = True,
        balancing_label: str = "downstream_class",
        balancing_method: str = "log",
        weights_clip_range: Tuple[float, float] = (0.3, 5.0),
        quality_col: Optional[str] = None,
        early_stopping_patience: int = 6,
        early_stopping_min_delta: float = 0.0,
        weight_decay_grid: Optional[Sequence[float]] = None,
        lr_grid: Optional[Sequence[float]] = None,
        cv_folds: int = 5,
        group_column: str = "sample_id",
    ):
        self.training_df = embeddings_df
        self.split_column = split_column
        self.head_type = head_type
        self.output_dir = Path(output_dir)
        self.modelversion = modelversion
        self.detector = detector
        self.downstream_classes = downstream_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_smoothing = label_smoothing
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.cv_folds = cv_folds
        self.group_column = group_column
        self.weight_decay_grid = (
            tuple(sorted({float(v) for v in weight_decay_grid}))
            if weight_decay_grid is not None
            else (0.0,)
        )
        self.lr_grid = (
            tuple(sorted({float(v) for v in lr_grid}))
            if lr_grid is not None
            else (self.lr,)
        )

        # Balancing / weighting
        self.use_balancing = use_balancing
        self.balancing_label = balancing_label
        self.balancing_method = balancing_method
        self.weights_clip_range = weights_clip_range
        self.quality_col = quality_col

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # Seeding
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sink = logger.add(
            self.output_dir / "logfile_torch_classifier.log", level="DEBUG"
        )

        # Initialize base config
        self.config: Dict[str, Any] = {}
        self.config.update(
            {
                "head_type": self.head_type,
                "output_dir": str(self.output_dir),
                "modelversion": self.modelversion,
                "detector": self.detector,
                "downstream_classes": self.downstream_classes,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "lr": self.lr,
                "epochs": self.epochs,
                "seed": self.seed,
                "use_balancing": self.use_balancing,
                "balancing_label": self.balancing_label,
                "balancing_method": self.balancing_method,
                "weights_clip_range": self.weights_clip_range,
                "quality_col": self.quality_col,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "weight_decay_grid": list(self.weight_decay_grid),
                "lr_grid": list(self.lr_grid),
                "cv_folds": self.cv_folds,
                "group_column": self.group_column,
            }
        )

    def create_config(self) -> None:
        """Create initial configuration and persist it."""
        self.save_config()

    def update_config_after_prepare(self, in_dim: int, num_classes: int) -> None:
        """Update config after data preparation when input/output dims are known."""
        self.config.update(
            {
                "in_dim": in_dim,
                "num_classes": num_classes,
                "classes_list": self.classes_list,
            }
        )
        self.save_config()

    def update_config_after_training(
        self,
        best_val_loss: float,
        best_epoch: int,
        epochs_trained: int,
        stopped_early: bool,
    ) -> None:
        """Update config with training outcomes (e.g., best val loss)."""
        self.config.update(
            {
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "epochs_trained": int(epochs_trained),
                "stopped_early": bool(stopped_early),
            }
        )
        self.save_config()

    def save_config(self) -> None:
        """Save configuration to JSON file, converting numpy types to native Python."""
        config_path = self.output_dir / "config.json"

        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        safe_config = convert_numpy_types(self.config)
        with open(config_path, "w") as f:
            json.dump(safe_config, f, indent=4)

    def _ensure_label_columns(self, df: pd.DataFrame) -> None:
        """Ensure that dataframe exposes ``finetune_class`` for downstream processing."""

        if "finetune_class" in df.columns and "downstream_class" in df.columns:
            logger.warning(
                "Both 'finetune_class' and 'downstream_class' found in dataframe. Using 'downstream_class'."
            )
        if "downstream_class" in df.columns:
            df["finetune_class"] = df["downstream_class"]
            return
        raise ValueError("Dataframe must contain either 'downstream_class'.")

    # --- Training ---
    def _build_model(self, in_dim: int, num_classes: int) -> nn.Module:
        if self.head_type == "linear":
            return LinearHead(in_dim, num_classes)
        elif self.head_type == "mlp":
            return MLPHead(
                in_dim, num_classes, hidden_dim=self.hidden_dim, dropout=self.dropout
            )
        else:
            raise ValueError("head_type must be one of {'linear','mlp'}")

    def _prepare_data(
        self, trn_df: pd.DataFrame, val_df: pd.DataFrame, tst_df: pd.DataFrame
    ):
        # Normalize class labels to range [0..C-1]
        classes = self.classes_list
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        for df in (trn_df, val_df, tst_df):
            df["label"] = df["downstream_class"].map(cls_to_idx)

        feat_cols = [c for c in trn_df.columns if c.startswith("presto_ft_")]
        in_dim = len(feat_cols)
        num_classes = len(classes)
        self.feat_cols = feat_cols
        return in_dim, num_classes

    def _get_balanced_sampler(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
    ) -> "WeightedRandomSampler":
        # extract the sampling class (strings or ints)
        bc_vals = df[self.balancing_label].values

        logger.info("Computing class weights ...")
        class_weights = get_class_weights(
            bc_vals,
            self.balancing_method,
            clip_range=self.weights_clip_range,
            normalize=normalize,
        )
        logger.info(f"Class weights: {class_weights}")

        # per‐sample weight
        sample_weights = np.ones_like(bc_vals).astype(np.float32)
        for k, v in class_weights.items():
            sample_weights[bc_vals == k] = v

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return sampler

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device_: torch.device,
    ) -> float:
        model.train()
        total_loss = 0.0
        for X, y in tqdm(loader, desc="Training", leave=False):
            X = X.to(device_)
            y = y.to(device_)
            logits = model(X)
            loss = self.loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * X.size(0)
        return total_loss / len(loader.dataset)

    def _eval_epoch(
        self, model: nn.Module, loader: DataLoader, device_: torch.device
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        total_loss = 0.0
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device_)
                y = y.to(device_)
                logits = model(X)
                loss = self.loss_fn(logits, y)
                total_loss += float(loss.item()) * X.size(0)
                preds = torch.argmax(logits, dim=1)
                preds_all.append(preds.cpu().numpy())
                labels_all.append(y.cpu().numpy())
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        return total_loss / len(loader.dataset), preds_all, labels_all

    def _drop_invalid_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        if "finetune_class" not in df.columns:
            return df
        return df[df["finetune_class"] != "remove"].reset_index(drop=True)

    def _apply_downstream_mapping(
        self, trainval_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.downstream_classes is not None:
            logger.info(f"Applying downstream class mapping: {self.downstream_classes}")
            missing_classes = set(trainval_df["finetune_class"].unique()) - set(
                self.downstream_classes.keys()
            )
            if missing_classes:
                raise ValueError(
                    f"Downstream mapping missing for classes: {missing_classes}"
                )
            for df in (trainval_df, test_df):
                df["downstream_class"] = df["finetune_class"].map(
                    self.downstream_classes
                )
        else:
            logger.info(
                "No downstream_classes specified, using finetune_classes directly"
            )
            self.downstream_classes = {
                cls: cls for cls in sorted(trainval_df["finetune_class"].unique())
            }
            trainval_df["downstream_class"] = trainval_df["finetune_class"]
            test_df["downstream_class"] = test_df["finetune_class"]

        self.target_column = "downstream_class"
        return trainval_df, test_df

    def _set_loss_function(self) -> None:
        self.classes_list = sorted(self.classes_list)
        if len(self.classes_list) == 2:
            logger.info(
                f"Binary classification detected with classes: {self.classes_list}"
            )
        else:
            logger.info(
                f"Multiclass classification with {len(self.classes_list)} classes"
            )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def _assign_label_indices(self, dfs: Sequence[pd.DataFrame]) -> None:
        cls_to_idx = {c: i for i, c in enumerate(self.classes_list)}
        for df in dfs:
            df["label"] = df[self.target_column].map(cls_to_idx)

    def _build_dataloader(self, df: pd.DataFrame, is_train: bool) -> DataLoader:
        dataset = EmbeddingsDataset(df, feat_prefix="presto_ft_", label_col="label")
        if is_train and self.use_balancing:
            sampler = self._get_balanced_sampler(df, normalize=True)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.num_workers,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=is_train,
            num_workers=self.num_workers,
        )

    def _run_training_loop(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        lr: float,
        weight_decay: float,
        max_epochs: int,
    ) -> Dict[str, Any]:
        device_ = device
        model = self._build_model(self.in_dim, self.num_classes).to(device_)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
            if val_loader is not None
            else None
        )

        best_val = float("inf")
        best_state = None
        best_epoch = 0
        best_macro_f1 = 0.0
        epochs_no_improve = 0
        stopped_early = False

        epochs_completed = 0
        for epoch in range(1, max_epochs + 1):
            tr_loss = self._train_epoch(model, train_loader, optimizer, device_)
            epochs_completed = epoch

            if val_loader is None:
                logger.info(
                    f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} (no validation set)"
                )
                continue

            val_loss, val_preds, val_labels = self._eval_epoch(
                model, val_loader, device_
            )
            macro_f1 = f1_score(val_labels, val_preds, average="macro")
            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < (best_val - self.early_stopping_min_delta):
                best_val = val_loss
                best_macro_f1 = macro_f1
                best_state = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if (
                    self.early_stopping_patience > 0
                    and epochs_no_improve >= self.early_stopping_patience
                ):
                    stopped_early = True
                    break

            logger.info(
                f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                f"macro_f1={macro_f1:.4f}"
            )

        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "model": model,
            "best_val_loss": best_val if val_loader is not None else None,
            "best_epoch": best_epoch if val_loader is not None else max_epochs,
            "epochs_ran": epochs_completed,
            "stopped_early": stopped_early,
            "val_macro_f1": best_macro_f1 if val_loader is not None else None,
        }

    def _iter_cv_splits(self, df: pd.DataFrame):
        if (
            self.group_column
            and self.group_column in df.columns
            and not df[self.group_column].isna().all()
        ):
            group_df = (
                df[[self.group_column, self.target_column]]
                .drop_duplicates(subset=self.group_column)
                .reset_index(drop=True)
            )
            if len(group_df) < self.cv_folds:
                logger.warning(
                    "Not enough unique groups for CV; falling back to sample-level splits."
                )
            else:
                sgkf = StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.seed
                )
                group_labels = group_df[self.target_column].to_numpy()
                dummy = np.zeros(len(group_df))
                for train_idx, val_idx in sgkf.split(dummy, group_labels):
                    train_groups = set(group_df.loc[train_idx, self.group_column])
                    val_groups = set(group_df.loc[val_idx, self.group_column])
                    train_mask = df[self.group_column].isin(train_groups)
                    val_mask = df[self.group_column].isin(val_groups)
                    yield np.where(train_mask)[0], np.where(val_mask)[0]
                return

        logger.warning(
            "Using row-level StratifiedKFold (group column missing or unusable)."
        )
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.seed
        )
        for train_idx, val_idx in skf.split(df, df[self.target_column]):
            yield train_idx, val_idx

    def _grid_search(
        self, trainval_df: pd.DataFrame
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        results: List[Dict[str, Any]] = []
        best_combo: Optional[Dict[str, float]] = None
        best_score = -np.inf

        for lr_val, wd_val in product(self.lr_grid, self.weight_decay_grid):
            fold_scores: List[float] = []
            fold_losses: List[float] = []
            fold_epochs: List[int] = []

            logger.info(f"Grid search combo lr={lr_val:.2e}, weight_decay={wd_val:.2e}")

            for fold_idx, (train_idx, val_idx) in enumerate(
                self._iter_cv_splits(trainval_df), start=1
            ):
                tr_df = trainval_df.iloc[train_idx].reset_index(drop=True)
                val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

                train_loader = self._build_dataloader(tr_df, is_train=True)
                val_loader = self._build_dataloader(val_df, is_train=False)

                result = self._run_training_loop(
                    train_loader,
                    val_loader,
                    lr=lr_val,
                    weight_decay=wd_val,
                    max_epochs=self.epochs,
                )

                if result["val_macro_f1"] is None or result["best_val_loss"] is None:
                    raise RuntimeError(
                        "Validation metrics missing during grid search run."
                    )

                fold_scores.append(float(result["val_macro_f1"]))
                fold_losses.append(float(result["best_val_loss"]))
                fold_epochs.append(int(result["best_epoch"]))

                logger.info(
                    f"Fold {fold_idx}/{self.cv_folds} -> macro_f1={result['val_macro_f1']:.4f}, "
                    f"val_loss={result['best_val_loss']:.4f}"
                )

            summary = {
                "lr": float(lr_val),
                "weight_decay": float(wd_val),
                "max_epochs": int(self.epochs),
                "fold_macro_f1": fold_scores,
                "fold_val_loss": fold_losses,
                "fold_best_epoch": fold_epochs,
                "mean_macro_f1": float(np.mean(fold_scores))
                if fold_scores
                else float("nan"),
                "mean_val_loss": float(np.mean(fold_losses))
                if fold_losses
                else float("nan"),
                "mean_best_epoch": float(np.mean(fold_epochs))
                if fold_epochs
                else float("nan"),
            }
            results.append(summary)

            if summary["mean_macro_f1"] > best_score:
                best_score = summary["mean_macro_f1"]
                mean_best_epoch = summary["mean_best_epoch"]
                final_epochs = (
                    max(1, int(round(mean_best_epoch)))
                    if not np.isnan(mean_best_epoch)
                    else self.epochs
                )
                best_combo = {
                    "lr": summary["lr"],
                    "weight_decay": summary["weight_decay"],
                    "final_epochs": min(self.epochs, final_epochs),
                    "mean_macro_f1": summary["mean_macro_f1"],
                    "mean_val_loss": summary["mean_val_loss"],
                    "mean_best_epoch": None
                    if np.isnan(summary["mean_best_epoch"])
                    else summary["mean_best_epoch"],
                }

        if best_combo is None:
            raise RuntimeError(
                "Grid search did not evaluate any hyperparameter combinations."
            )

        return best_combo, results

    def train(self) -> nn.Module:
        self.create_config()
        self._ensure_label_columns(self.training_df)

        trn_df, val_df, tst_df = train_val_test_split(
            self.training_df, self.split_column
        )
        trainval_df = pd.concat([trn_df, val_df]).reset_index(drop=True)
        test_df = tst_df.reset_index(drop=True)

        trainval_df = self._drop_invalid_samples(trainval_df)
        test_df = self._drop_invalid_samples(test_df)

        if trainval_df.empty:
            raise ValueError(
                "No training samples available after filtering train/val splits."
            )
        if test_df.empty:
            raise ValueError("No test samples available for evaluation.")

        trainval_df, test_df = self._apply_downstream_mapping(trainval_df, test_df)
        self.classes_list = sorted(trainval_df[self.target_column].unique())
        self._set_loss_function()

        self._assign_label_indices([trainval_df, test_df])

        self.feat_cols = [c for c in trainval_df.columns if c.startswith("presto_ft_")]
        self.in_dim = len(self.feat_cols)
        self.num_classes = len(self.classes_list)
        self.update_config_after_prepare(self.in_dim, self.num_classes)

        best_params, grid_results = self._grid_search(trainval_df)
        self.config.update(
            {
                "grid_search_results": grid_results,
                "selected_hyperparams": best_params,
            }
        )
        self.save_config()

        logger.info(
            "Selected hyperparameters -> lr=%.2e, weight_decay=%.2e, final_epochs=%d (macro-F1=%.4f)",
            best_params["lr"],
            best_params["weight_decay"],
            best_params["final_epochs"],
            best_params["mean_macro_f1"],
        )

        final_loader = self._build_dataloader(trainval_df, is_train=True)
        final_result = self._run_training_loop(
            final_loader,
            None,
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            max_epochs=best_params["final_epochs"],
        )
        model = final_result["model"]

        test_loader = self._build_dataloader(test_df, is_train=False)

        mean_best_epoch = best_params.get("mean_best_epoch")
        best_epoch_stat = (
            int(round(mean_best_epoch))
            if mean_best_epoch is not None
            else best_params["final_epochs"]
        )

        self.update_config_after_training(
            best_val_loss=best_params.get("mean_val_loss", 0.0),
            best_epoch=best_epoch_stat,
            epochs_trained=final_result["epochs_ran"],
            stopped_early=final_result["stopped_early"],
        )

        self.save_model(model)
        self.evaluate(model, test_loader)
        return model

    def save_model(self, model: nn.Module) -> None:
        modelname = f"PrestoDownstreamTorchHead_{self.head_type}_{self.detector}_v{self.modelversion}"
        pt_path = self.output_dir / f"{modelname}.pt"
        torch.save(model.state_dict(), pt_path)
        logger.info(f"Model weights saved: {pt_path}")

        # Package PT + config.json into an archive with the same base name
        try:
            self._zip_model_and_config(modelname, pt_path)
        except Exception as e:
            logger.warning(f"Packaging zip skipped: {e}")

    def _zip_model_and_config(self, modelname: str, pt_path: Path) -> None:
        """Create a zip archive containing the .pt model and the config as config.json.

        The archive is created in the output directory with the same base name
        as the model (e.g., `<modelname>.zip`).
        """
        zip_path = self.output_dir / f"{modelname}.zip"
        cfg_src = self.output_dir / "config.json"
        if not cfg_src.exists():
            raise FileNotFoundError(
                f"Config file not found at {cfg_src}. Ensure save_config() was called."
            )
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            # Store model with its filename
            zf.write(pt_path, arcname=pt_path.name)
            # Store config under a canonical name inside the archive
            zf.write(cfg_src, arcname="config.json")
        logger.info(f"Packaged artifacts into: {zip_path}")

    def _plot_confusion_matrices(
        self, true_labels: np.ndarray, preds: np.ndarray
    ) -> None:
        # Ensure numpy integer arrays and fixed label order
        true_labels = np.asarray(true_labels, dtype=np.int64)
        pred_labels = np.asarray(preds, dtype=np.int64)

        # Create confusion matrices
        f_abs = create_confusion_matrix(
            true_labels,
            pred_labels,
            self.classes_list,
            normalize=False,
            title="Confusion Matrix (Absolute)",
        )

        try:
            plt.tight_layout()
            plt.savefig(self.output_dir / "CM_abs.png")
        except Exception:
            plt.savefig(self.output_dir / "CM_abs.png", bbox_inches="tight")
        plt.close(f_abs)
        f_norm = create_confusion_matrix(
            true_labels,
            pred_labels,
            self.classes_list,
            normalize=True,
            title="Confusion Matrix (Normalized)",
        )
        try:
            plt.tight_layout()
            plt.savefig(self.output_dir / "CM_norm.png")
        except Exception:
            plt.savefig(self.output_dir / "CM_norm.png", bbox_inches="tight")
        plt.close(f_norm)

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for X, y in test_loader:
                logits = model(X.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(y.numpy())
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)

        # Metrics
        # Map integer labels back to string class names for readability
        target_names = [str(c) for c in self.classes_list]
        report = classification_report(
            labels_all,
            preds_all,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose().reset_index().round(2)
        report_df.columns = pd.Index(
            ["class", "precision", "recall", "f1-score", "support"]
        )
        report_df.to_csv(self.output_dir / "classification_report.csv")

        logger.info("Evaluation results:")
        logger.info("\n" + report_df.to_string(index=False))

        self._plot_confusion_matrices(labels_all, preds_all)

        metrics: Dict[str, float] = {}
        metrics["OA"] = round(accuracy_score(labels_all, preds_all), 3)
        metrics["Macro F1"] = round(f1_score(labels_all, preds_all, average="macro"), 3)
        metrics["Macro Precision"] = round(
            precision_score(labels_all, preds_all, average="macro"), 3
        )
        metrics["Macro Recall"] = round(
            recall_score(labels_all, preds_all, average="macro"), 3
        )

        with open(self.output_dir / "metrics.txt", "w") as f:
            f.write("Test results (Torch head):\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
                logger.info(f"{k} = {v}")
        return metrics


def create_confusion_matrix(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    class_names: NDArray[np.str_],
    title: Optional[str] = None,
    normalize: bool = True,
) -> plt.figure:
    """Create the confusion matrix."""
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)

    # Create the figure
    def _annot(x: Union[str, float]) -> str:
        """Annotation function."""
        x = float(x)
        if normalize:
            return f"{100 * x:.1f}%" if x > 0.0 else ""
        else:
            return f"{x}" if x > 0 else ""

    # class_tags = [f"{x}\n({y})" for x, y in zip(class_ids, class_names)]
    class_tags = class_names
    figsize = (ceil(0.7 * len(class_tags)) + 1, ceil(0.7 * len(class_tags)))
    figsize = (10, 9) if figsize[0] < 10 else figsize
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    heatmap(
        100 * cm if normalize else cm,
        vmin=0,
        vmax=100 if normalize else None,
        annot=np.asarray([_annot(x) for x in cm.flatten()]).reshape(cm.shape),
        fmt="",
        xticklabels=class_tags,
        yticklabels=class_tags,
        linewidths=0.01,
        square=True,
    )
    plt.xticks(
        [i + 0.5 for i in range(len(class_tags))],
        class_tags,
        rotation=90,
    )
    plt.yticks(
        [i + 0.5 for i in range(len(class_tags))],
        class_tags,
        rotation=0,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Target")
    plt.tight_layout()
    return fig
