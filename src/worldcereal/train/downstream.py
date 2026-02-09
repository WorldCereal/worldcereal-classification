#!/usr/bin/env python3
"""Train a PyTorch classification head (Linear or small MLP) on embeddings."""

import json
import zipfile
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm.auto import tqdm

from worldcereal.train.backbone import (
    checkpoint_fingerprint,
    resolve_seasonal_encoder,
)
from worldcereal.train.data import spatial_train_val_test_split, train_val_test_split
from worldcereal.train.datasets import get_class_weights
from worldcereal.train.seasonal_head import LinearHead, MLPHead


class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feat_prefix: str = "presto_ft_",
        label_col: str = "downstream_class",
        weight_col: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        # Sort columns numerically, not lexicographically to preserve embedding dimension order
        feat_cols_unsorted = [c for c in df.columns if c.startswith(feat_prefix)]
        self.feat_cols = sorted(
            feat_cols_unsorted, key=lambda x: int(x.replace(feat_prefix, ""))
        )
        logger.info(
            f"EmbeddingsDataset: {len(self.df)} samples, {len(self.feat_cols)} features"
        )
        self.X = self.df[self.feat_cols].to_numpy(dtype=np.float32)
        self.y = self.df[label_col].to_numpy()
        # Store per-sample weights if provided
        if weight_col and weight_col in df.columns:
            self.weights = self.df[weight_col].to_numpy(dtype=np.float32)
        else:
            self.weights = np.ones(len(self.df), dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"EmbeddingsDataset(num_samples={len(self.df)}, "
            f"num_features={len(self.feat_cols)}), label_col={self.label_col})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), int(self.y[idx]), float(self.weights[idx])


class TorchTrainer:
    def __init__(
        self,
        embeddings_df: pd.DataFrame,
        split_column: str = "split",
        head_type: str = "linear",
        head_task: Literal["croptype", "landcover"] = "croptype",
        output_dir: Union[Path, str] = "./downstream_classifier",
        modelversion: str = "",
        detector: Optional[str] = None,
        downstream_classes: Optional[dict] = None,
        cropland_class_names: Optional[Sequence[str]] = None,
        season_id: Optional[str] = None,
        presto_model_path: Optional[str] = None,
        presto_model_fingerprint: Optional[str] = None,
        batch_size: int = 1024,
        num_workers: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        lr: float = 1e-2,  # can be high for lightweight head
        epochs: int = 30,
        log_interval: int = 10,
        log_display_delta: float = 0.01,
        seed: int = 42,
        label_smoothing: float = 0.0,
        use_balancing: bool = True,
        balancing_label: str = "downstream_class",
        balancing_method: str = "log",
        weights_clip_range: Tuple[float, float] = (0.1, 10),
        quality_col: Optional[str] = None,
        early_stopping_patience: int = 6,
        early_stopping_min_delta: float = 0.0,
        weight_decay: float = 0.0,
        disable_progressbar: bool = False,
        use_spatial_split: bool = True,
        spatial_bin_size_degrees: float = 0.25,
    ):
        self.training_df = embeddings_df
        self.split_column = split_column
        self.head_type = head_type
        self.head_task = head_task
        if self.head_task == "croptype" and season_id is None:
            raise ValueError(
                "TorchTrainer requires 'season_id' when training croptype heads."
            )
        self.output_dir = Path(output_dir)
        self.modelversion = modelversion
        self.detector = detector or head_task
        self.downstream_classes = downstream_classes
        self.cropland_class_names = (
            [str(cls) for cls in cropland_class_names]
            if cropland_class_names is not None
            else []
        )
        self.season_id = season_id
        # Initialize attributes that will be set during prepare_data
        self.classes_list: list[str] = []
        self.in_dim: int = 0
        self.num_classes: int = 0
        if presto_model_path:
            # presto_model_path is expected to be the encoder checkpoint
            self.presto_model_path = presto_model_path
            if presto_model_fingerprint:
                self.presto_model_fingerprint = presto_model_fingerprint
            else:
                if str(self.presto_model_path).startswith(("http://", "https://")):
                    raise ValueError(
                        "presto_model_fingerprint must be provided when using a remote Presto URL."
                    )
                # Compute fingerprint directly from the encoder checkpoint path
                self.presto_model_fingerprint = checkpoint_fingerprint(
                    self.presto_model_path
                )
        else:
            # resolve_seasonal_encoder returns encoder checkpoint path + fingerprint
            (
                self.presto_model_path,
                self.presto_model_fingerprint,
            ) = resolve_seasonal_encoder()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_smoothing = label_smoothing
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.log_interval = max(1, int(log_interval))
        self.log_display_delta = log_display_delta
        self.last_logged_val: Optional[float] = None
        self.seed = seed
        self.disable_progressbar = disable_progressbar
        self.weight_decay = weight_decay

        # Balancing / weighting
        self.use_balancing = use_balancing
        self.balancing_label = balancing_label
        self.balancing_method = balancing_method
        self.weights_clip_range = weights_clip_range
        self.quality_col = quality_col

        # Spatial splitting
        self.use_spatial_split = use_spatial_split
        self.spatial_bin_size_degrees = spatial_bin_size_degrees

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # Seeding
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize base config
        self.config: Dict[str, Any] = {}
        self.config.update(
            {
                "head_type": self.head_type,
                "head_task": self.head_task,
                "output_dir": str(self.output_dir),
                "modelversion": self.modelversion,
                "detector": self.detector,
                "downstream_classes": self.downstream_classes,
                "cropland_class_names": self.cropland_class_names,
                "season_id": self.season_id,
                "presto_model_path": self.presto_model_path,
                "presto_model_fingerprint": self.presto_model_fingerprint,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "lr": self.lr,
                "epochs": self.epochs,
                "log_interval": self.log_interval,
                "seed": self.seed,
                "use_balancing": self.use_balancing,
                "balancing_label": self.balancing_label,
                "balancing_method": self.balancing_method,
                "weights_clip_range": self.weights_clip_range,
                "quality_col": self.quality_col,
                "use_spatial_split": self.use_spatial_split,
                "spatial_bin_size_degrees": self.spatial_bin_size_degrees,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "weight_decay": self.weight_decay,
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
        if "finetune_class" in df.columns:
            df["downstream_class"] = df["finetune_class"]
            return
        raise ValueError(
            "Dataframe must contain either 'downstream_class' or 'finetune_class'."
        )

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

    def _get_balanced_sampler(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
    ) -> "WeightedRandomSampler":
        # extract the sampling class (strings or ints)
        bc_vals = df[self.balancing_label].values

        logger.info("Computing class weights for balanced sampling ...")
        class_weights = get_class_weights(
            bc_vals,
            self.balancing_method,
            clip_range=self.weights_clip_range,
            normalize=normalize,
        )
        logger.info(f"Class weights: {class_weights}")

        # per‐sample weight for sampling (class balance only)
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
        total_samples = 0
        for X, y, weights in tqdm(
            loader, desc="Training", leave=False, disable=self.disable_progressbar
        ):
            X = X.to(device_)
            y = y.to(device_)
            weights = weights.to(device_)
            logits = model(X)
            # Compute per-sample loss
            loss_unreduced = nn.functional.cross_entropy(
                logits, y, reduction="none", label_smoothing=self.label_smoothing
            )
            # Apply per-sample weights
            loss = (loss_unreduced * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * X.size(0)
            total_samples += X.size(0)
        return total_loss / total_samples

    def _eval_epoch(
        self, model: nn.Module, loader: DataLoader, device_: torch.device
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for X, y, weights in loader:
                X = X.to(device_)
                y = y.to(device_)
                weights = weights.to(device_)
                logits = model(X)
                # Compute per-sample loss
                loss_unreduced = nn.functional.cross_entropy(
                    logits, y, reduction="none", label_smoothing=self.label_smoothing
                )
                # Apply per-sample weights
                loss = (loss_unreduced * weights).mean()
                total_loss += float(loss.item()) * X.size(0)
                total_samples += X.size(0)
                preds = torch.argmax(logits, dim=1)
                preds_all.append(preds.cpu().numpy())
                labels_all.append(y.cpu().numpy())
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        return total_loss / total_samples, preds_all, labels_all

    def _drop_invalid_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        if "finetune_class" not in df.columns:
            return df
        filtered = df[df["finetune_class"] != "remove"].reset_index(drop=True)
        if "in_season" in filtered.columns:
            if filtered["in_season"].any():
                filtered = filtered[filtered["in_season"]].reset_index(drop=True)
            else:
                logger.warning(
                    "No samples marked in-season; skipping in_season filtering."
                )
        return filtered

    def _build_head_manifest(self, checkpoint_name: str) -> Dict[str, Any]:
        task_type = "binary" if len(self.classes_list) == 2 else "multiclass"
        if self.head_task == "croptype":
            replacement_contract = {
                "input_tensor": "season_embeddings",
                "output_attr": "season_logits",
                "expects_season_masks": True,
            }
            logits_attr = "season_logits"
            state_dict_prefix = "head.crop_head"
        else:
            replacement_contract = {
                "input_tensor": "global_embedding",
                "output_attr": "global_logits",
                "expects_time_dimension": False,
            }
            logits_attr = "global_logits"
            state_dict_prefix = "head.landcover_head"

        head_entry = {
            "name": self.head_task,
            "task": self.head_task,
            "task_type": task_type,
            "num_classes": len(self.classes_list),
            "class_names": [str(cls) for cls in self.classes_list],
            "head_type": self.head_type,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout if self.head_type == "mlp" else 0.0,
            "label_column": self.target_column,
            "logits_attr": logits_attr,
            "state_dict_prefix": state_dict_prefix,
            "replacement_contract": replacement_contract,
            "gating": {
                "enabled": bool(self.cropland_class_names),
                "cropland_classes": self.cropland_class_names,
            },
        }

        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "experiment": {
                "name": checkpoint_name.replace(".pt", ""),
                "season_id": self.season_id,
            },
            "backbone": {
                "fingerprint": self.presto_model_fingerprint,
            }
            if self.presto_model_fingerprint
            else {},
            "heads": [head_entry],
            "artifacts": {
                "config": "config.json",
                "checkpoints": {"head": checkpoint_name},
                "packages": {"head": checkpoint_name.replace(".pt", ".zip")},
            },
        }
        return manifest

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
        dataset = EmbeddingsDataset(
            df, feat_prefix="presto_ft_", label_col="label", weight_col="_sample_weight"
        )
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
                if self._should_emit(epoch):
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

            improved = False
            if val_loss < (best_val - self.early_stopping_min_delta):
                best_val = val_loss
                best_macro_f1 = macro_f1
                best_state = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                epochs_no_improve = 0
                improved = True
            else:
                epochs_no_improve += 1
                if (
                    self.early_stopping_patience > 0
                    and epochs_no_improve >= self.early_stopping_patience
                ):
                    stopped_early = True
                    logger.info(
                        f"Early stopping triggered after {epoch} epochs (best epoch: {best_epoch}, best val_loss: {best_val:.4f})"
                    )
                    break

            if self._should_emit(epoch, improved):
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

    def _should_emit(self, epoch: int, improved: bool = False) -> bool:
        if epoch == 1 or epoch == self.epochs or epoch % self.log_interval == 0:
            return True
        if improved:
            return True
        return False

    def train(self) -> nn.Module:
        self.create_config()
        self._ensure_label_columns(self.training_df)

        if self.use_spatial_split:
            train_df, val_df, test_df = spatial_train_val_test_split(
                self.training_df,
                split_column=self.split_column,
                bin_size_degrees=self.spatial_bin_size_degrees,
            )
        else:
            train_df, val_df, test_df = train_val_test_split(
                self.training_df, self.split_column
            )

        train_df = self._drop_invalid_samples(train_df)
        val_df = self._drop_invalid_samples(val_df)
        test_df = self._drop_invalid_samples(test_df)

        if train_df.empty:
            raise ValueError("No training samples available after filtering.")
        if val_df.empty:
            raise ValueError("No validation samples available after filtering.")
        if test_df.empty:
            raise ValueError("No test samples available for evaluation.")

        # Apply downstream class mapping to all splits
        train_df, _ = self._apply_downstream_mapping(train_df, train_df.copy())
        val_df, _ = self._apply_downstream_mapping(val_df, val_df.copy())
        test_df, _ = self._apply_downstream_mapping(test_df, test_df.copy())

        self.classes_list = sorted(train_df[self.target_column].unique())
        self._set_loss_function()

        self._assign_label_indices([train_df, val_df, test_df])

        # Compute quality-based sample weights if quality_col is provided
        if self.quality_col is not None:
            for df in (train_df, val_df, test_df):
                if self.quality_col in df.columns:
                    quality_values = df[self.quality_col].values
                    # Normalize quality values to [0, 1] range
                    quality_min = quality_values.min()
                    quality_max = quality_values.max()
                    if quality_max > quality_min:
                        quality_normalized = (quality_values - quality_min) / (
                            quality_max - quality_min
                        )
                    else:
                        quality_normalized = np.ones_like(quality_values)
                    df["_sample_weight"] = quality_normalized
                else:
                    df["_sample_weight"] = 1.0
            logger.info(
                f"Applied quality-based sample weights from column '{self.quality_col}'"
            )
        else:
            for df in (train_df, val_df, test_df):
                df["_sample_weight"] = 1.0

        if "confidence_nonoutlier" in train_df.columns:
            logger.info(
                "Incorporating 'confidence_nonoutlier' into sample weights for train/val sets."
            )
            train_df["_sample_weight"] *= train_df["confidence_nonoutlier"].fillna(1.0)
            val_df["_sample_weight"] *= val_df["confidence_nonoutlier"].fillna(1.0)

        self.feat_cols = [c for c in train_df.columns if c.startswith("presto_ft_")]
        self.in_dim = len(self.feat_cols)
        self.num_classes = len(self.classes_list)
        self.update_config_after_prepare(self.in_dim, self.num_classes)

        # Train with validation monitoring
        train_loader = self._build_dataloader(train_df, is_train=True)
        val_loader = self._build_dataloader(val_df, is_train=False)

        logger.info(
            f"Training with validation monitoring for up to {self.epochs} epochs "
            f"(lr={self.lr:.2e}, weight_decay={self.weight_decay:.2e})"
        )

        result = self._run_training_loop(
            train_loader,
            val_loader,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.epochs,
        )
        model = result["model"]

        test_loader = self._build_dataloader(test_df, is_train=False)

        self.update_config_after_training(
            best_val_loss=result["best_val_loss"] if result["best_val_loss"] else 0.0,
            best_epoch=result["best_epoch"],
            epochs_trained=result["epochs_ran"],
            stopped_early=result["stopped_early"],
        )

        self.save_model(model)
        self.evaluate(model, test_loader)
        return model

    def save_model(self, model: nn.Module) -> None:
        modelname = self._build_model_name()
        pt_path = self.output_dir / f"{modelname}.pt"
        torch.save(model.state_dict(), pt_path)
        logger.info(f"Model weights saved: {pt_path}")

        manifest = self._build_head_manifest(pt_path.name)
        self.config.update(manifest)
        self.save_config()

        # Package PT + config.json into an archive with the same base name
        try:
            self._zip_model_and_config(modelname, pt_path)
        except Exception as e:
            logger.warning(f"Packaging zip skipped: {e}")

    def _build_model_name(self) -> str:
        base = f"PrestoDownstreamTorchHead_{self.head_type}_{self.head_task}"
        if self.season_id:
            base = f"{base}_{self.season_id}"
        if self.modelversion:
            base = f"{base}_version-{self.modelversion}"
        return base

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
            for X, y, weights in test_loader:
                logits = model(X.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(y.numpy())
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)

        # Metrics
        # Determine which classes are actually present in the test set
        unique_labels = np.unique(np.concatenate([labels_all, preds_all]))
        labels_in_test = [
            self.classes_list[i] for i in unique_labels if i < len(self.classes_list)
        ]

        report = classification_report(
            labels_all,
            preds_all,
            labels=unique_labels,
            target_names=[str(c) for c in labels_in_test],
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
