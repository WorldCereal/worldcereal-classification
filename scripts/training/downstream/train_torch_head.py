#!/usr/bin/env python3
"""Train a PyTorch classification head (Linear or small MLP) on Presto embeddings.

This complements `train_catboost.py` by using a neural head instead of CatBoost.
It supports either computing embeddings on-the-fly or loading precomputed embeddings
and benchmarks accuracy/F1 against the CatBoost approach.
"""

import argparse
import json
import zipfile
from math import ceil
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from numpy.typing import NDArray
from prometheo.models import Presto
from prometheo.models.presto.wrapper import load_presto_weights
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

from worldcereal.train.data import get_training_df
from worldcereal.train.datasets import (
    SensorMaskingConfig,
    WorldCerealTrainingDataset,
    get_class_weights,
)


class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feat_prefix: str = "presto_ft_",
        label_col: str = "label",
    ):
        self.df = df.reset_index(drop=True)
        self.feat_cols = [c for c in df.columns if c.startswith(feat_prefix)]
        self.X = self.df[self.feat_cols].to_numpy(dtype=np.float32)
        self.y = self.df[label_col].to_numpy()

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


class TorchHeadTrainer:
    def __init__(
        self,
        head_type: str,
        presto_model_path: str,
        data_dir: str,
        output_dir: str,
        modelversion: str = "001",
        detector: str = "cropland",
        downstream_classes: Optional[dict] = None,
        classes_scheme: str = "LANDCOVER10",
        timestep_freq: Literal["month", "dekad"] = "month",
        batch_size: int = 1024,
        num_workers: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-2,  # can be high for linear head
        epochs: int = 30,
        seed: int = 42,
        label_smoothing: float = 0.0,
        ref_id_to_keep_path: Optional[str] = None,
        use_balancing: bool = False,
        sampling_class: str = "downstream_class",
        balancing_method: str = "log",
        clip_range: Tuple[float, float] = (0.3, 5.0),
        quality_col: Optional[str] = None,
        early_stopping_patience: int = 6,
        early_stopping_min_delta: float = 0.0,
    ):
        self.head_type = head_type
        self.presto_model_path = Path(presto_model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.modelversion = modelversion
        self.detector = detector
        self.downstream_classes = downstream_classes
        self.classes_scheme = classes_scheme
        self.timestep_freq = timestep_freq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.ref_id_to_keep_path = ref_id_to_keep_path
        # Balancing / weighting
        self.use_balancing = use_balancing
        self.sampling_class = sampling_class
        self.balancing_method = balancing_method
        self.clip_range = clip_range
        self.quality_col = quality_col
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # Determine if binary classification based on downstream_classes
        if self.downstream_classes is not None:
            unique_downstream = set(self.downstream_classes.values())
            if len(unique_downstream) == 2:
                self.is_binary = True
                logger.info(
                    f"Detected binary classification from downstream_classes: {unique_downstream}"
                )
            else:
                self.is_binary = False
                logger.info(
                    f"Detected multiclass classification from downstream_classes: {unique_downstream}"
                )
        else:
            # Default case: no downstream mapping, will be determined later based on finetune_classes
            self.is_binary = False

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sink = logger.add(
            self.output_dir / "logfile_torch_head.log", level="DEBUG"
        )
        self.config: Dict[str, Any] = {}
        # Initialize base config
        self.config.update(
            {
                "head_type": self.head_type,
                "presto_model_path": str(self.presto_model_path),
                "data_dir": str(self.data_dir),
                "output_dir": str(self.output_dir),
                "modelversion": self.modelversion,
                "detector": self.detector,
                "downstream_classes": self.downstream_classes,
                "timestep_freq": self.timestep_freq,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "lr": self.lr,
                "epochs": self.epochs,
                "seed": self.seed,
                "ref_id_to_keep_path": self.ref_id_to_keep_path,
                "use_balancing": self.use_balancing,
                "sampling_class": self.sampling_class,
                "balancing_method": self.balancing_method,
                "clip_range": self.clip_range,
                "quality_col": self.quality_col,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
            }
        )

    def create_config(self) -> None:
        """Create initial configuration and persist it."""
        # Nothing extra to compute here; save current config
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

    # --- Embeddings acquisition ---
    def _check_for_embeddings(self) -> bool:
        emb_files = [
            "train_embeddings.parquet",
            "val_embeddings.parquet",
            "test_embeddings.parquet",
        ]
        return all((self.output_dir / f).exists() for f in emb_files)

    def _dataset_to_embeddings_with_label(
        self, model: Presto, ds: WorldCerealTrainingDataset
    ) -> pd.DataFrame:
        model.eval()
        df = get_training_df(
            ds,
            model,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            time_explicit=False,
        )
        return df

    def _compute_embeddings(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Loading raw data files...")
        train_df = pd.read_parquet(self.data_dir / "train_df.parquet")
        val_df = pd.read_parquet(self.data_dir / "val_df.parquet")
        test_df = pd.read_parquet(self.data_dir / "test_df.parquet")

        orig_classes = sorted(train_df["finetune_class"].unique())

        # downstream_class = finetune_class by default
        for df in (train_df, val_df, test_df):
            df["downstream_class"] = df["finetune_class"]

        num_timesteps = 12 if self.timestep_freq == "month" else 36
        presto_model = Presto()
        presto_model = load_presto_weights(presto_model, self.presto_model_path).to(
            device
        )

        masking_config = SensorMaskingConfig(
            enable=True,
            s1_full_dropout_prob=0.05,
            s1_timestep_dropout_prob=0.1,
            s2_cloud_timestep_prob=0.1,
            s2_cloud_block_prob=0.05,
            s2_cloud_block_min=2,
            s2_cloud_block_max=3,
            meteo_timestep_dropout_prob=0.03,
            dem_dropout_prob=0.01,
        )

        trn_ds = WorldCerealTrainingDataset(
            train_df,
            num_timesteps=num_timesteps,
            timestep_freq=self.timestep_freq,
            task_type="multiclass",
            num_outputs=len(orig_classes),
            augment=True,
            masking_config=masking_config,
            repeats=5,
        )
        val_ds = WorldCerealTrainingDataset(
            val_df,
            num_timesteps=num_timesteps,
            timestep_freq=self.timestep_freq,
            task_type="multiclass",
            num_outputs=len(orig_classes),
            augment=False,
            masking_config=SensorMaskingConfig(enable=False),
            repeats=1,
        )
        test_ds = WorldCerealTrainingDataset(
            test_df,
            num_timesteps=num_timesteps,
            timestep_freq=self.timestep_freq,
            task_type="multiclass",
            num_outputs=len(orig_classes),
            augment=False,
            masking_config=SensorMaskingConfig(enable=False),
            repeats=1,
        )

        logger.info("Computing embeddings...")
        trn_df = self._dataset_to_embeddings_with_label(presto_model, trn_ds)
        val_df = self._dataset_to_embeddings_with_label(presto_model, val_ds)
        tst_df = self._dataset_to_embeddings_with_label(presto_model, test_ds)

        logger.info("Saving computed embeddings...")
        trn_df.to_parquet(self.output_dir / "train_embeddings.parquet")
        val_df.to_parquet(self.output_dir / "val_embeddings.parquet")
        tst_df.to_parquet(self.output_dir / "test_embeddings.parquet")
        return trn_df, val_df, tst_df

    def _load_embeddings(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Loading pre-computed embeddings...")
        trn_df_emb = pd.read_parquet(self.output_dir / "train_embeddings.parquet")
        val_df_emb = pd.read_parquet(self.output_dir / "val_embeddings.parquet")
        tst_df_emb = pd.read_parquet(self.output_dir / "test_embeddings.parquet")

        logger.info("Merging and re-splitting by ref_id...")
        df = pd.concat([trn_df_emb, val_df_emb, tst_df_emb], ignore_index=True)

        # Optional ref_id filtering to match CatBoost script behavior
        if self.ref_id_to_keep_path is not None:
            try:
                with open(self.ref_id_to_keep_path, "r") as f:
                    ref_ids_json = json.load(f)
                ref_id_to_keep = list(ref_ids_json.keys())
                logger.info(f"Filtering by ref_id_to_keep: {len(ref_id_to_keep)} ids")
                df = df[df["ref_id"].isin(ref_id_to_keep)]
                logger.info(f"Size after ref_id filtering: {len(df)}")
            except Exception as e:
                logger.warning(
                    f"Failed to load ref_id_to_keep at {self.ref_id_to_keep_path}: {e}"
                )

        unique_ref_ids = df["ref_id"].unique()
        from sklearn.model_selection import train_test_split

        train_ids, val_ids = train_test_split(
            unique_ref_ids, test_size=0.3, random_state=self.seed
        )
        val_ids, test_ids = train_test_split(
            val_ids, test_size=0.5, random_state=self.seed
        )
        trn_df = df[df["ref_id"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["ref_id"].isin(val_ids)].reset_index(drop=True)
        tst_df = df[df["ref_id"].isin(test_ids)].reset_index(drop=True)
        return trn_df, val_df, tst_df

    def _get_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self._check_for_embeddings():
            logger.info("Found pre-computed embeddings, loading them...")
            return self._load_embeddings()
        else:
            logger.info("No pre-computed embeddings found, computing them ...")
            return self._compute_embeddings()

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

    def get_balanced_sampler(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
    ) -> "WeightedRandomSampler":
        # extract the sampling class (strings or ints)
        bc_vals = df[self.sampling_class].values

        logger.info("Computing class weights ...")
        class_weights = get_class_weights(
            bc_vals,
            self.balancing_method,
            clip_range=self.clip_range,
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

    def train(self) -> nn.Module:
        # Save base config
        self.create_config()

        trn_df, val_df, tst_df = self._get_training_data()

        # Column "downstream_class" is actually "finetune_class" at this point
        trn_df = trn_df.rename(columns={"downstream_class": "finetune_class"})
        val_df = val_df.rename(columns={"downstream_class": "finetune_class"})
        tst_df = tst_df.rename(columns={"downstream_class": "finetune_class"})

        # Save class list
        self.classes_list = sorted(trn_df["finetune_class"].unique())
        logger.info(f"Classes after mapping: {self.classes_list}")

        # Remove samples to be ignored
        trn_df = trn_df[trn_df["finetune_class"] != "remove"]
        val_df = val_df[val_df["finetune_class"] != "remove"]
        tst_df = tst_df[tst_df["finetune_class"] != "remove"]

        # Update class list after removing samples
        self.classes_list = sorted(trn_df["finetune_class"].unique())
        logger.info(f"Final classes: {self.classes_list}")

        # Apply downstream class mapping (default to identity mapping if not specified)
        if self.downstream_classes is not None:
            logger.info(f"Applying downstream class mapping: {self.downstream_classes}")

            # Check that all finetune classes are covered in the mapping
            missing_classes = set(self.classes_list) - set(
                self.downstream_classes.keys()
            )
            if missing_classes:
                raise ValueError(
                    f"Downstream mapping missing for classes: {missing_classes}"
                )

            # Apply mapping to all dataframes
            trn_df["downstream_class"] = trn_df["finetune_class"].map(
                self.downstream_classes
            )
            val_df["downstream_class"] = val_df["finetune_class"].map(
                self.downstream_classes
            )
            tst_df["downstream_class"] = tst_df["finetune_class"].map(
                self.downstream_classes
            )

            # Update classes list to downstream classes
            self.classes_list = sorted(trn_df["downstream_class"].unique())
            logger.info(f"Classes after downstream mapping: {self.classes_list}")

            # Set the target column for training
            self.target_column = "downstream_class"
        else:
            # Default case: create identity mapping for finetune_classes
            logger.info(
                "No downstream_classes specified, using finetune_classes directly"
            )
            self.downstream_classes = {cls: cls for cls in self.classes_list}
            trn_df["downstream_class"] = trn_df["finetune_class"]
            val_df["downstream_class"] = val_df["finetune_class"]
            tst_df["downstream_class"] = tst_df["finetune_class"]
            logger.info(f"Using classes: {self.classes_list}")

            # Set the target column for training
            self.target_column = "downstream_class"

        # Determine if binary classification based on final classes
        if len(self.classes_list) == 2:
            self.is_binary = True
            logger.info(
                f"Binary classification detected with classes: {self.classes_list}"
            )

            # If binary classification with "other" class, ensure proper ordering
            if "other" in self.classes_list:
                target_class = [cls for cls in self.classes_list if cls != "other"][0]
                # Ensure "other" is first (index 0) and target class is second (index 1)
                self.classes_list = ["other", target_class]
                logger.info(
                    f"Binary classes reordered: {self.classes_list} (other=0, {target_class}=1)"
                )

                # Save target class name for reference
                self.target_class_name = target_class
        else:
            self.is_binary = False
            logger.info(
                f"Multiclass classification with {len(self.classes_list)} classes: {self.classes_list}"
            )

        in_dim, num_classes = self._prepare_data(trn_df, val_df, tst_df)
        # Update config with dimensions and classes
        self.update_config_after_prepare(in_dim, num_classes)

        # Build datasets and loaders
        train_ds = EmbeddingsDataset(
            trn_df, feat_prefix="presto_ft_", label_col="label"
        )
        val_ds = EmbeddingsDataset(val_df, feat_prefix="presto_ft_", label_col="label")
        test_ds = EmbeddingsDataset(tst_df, feat_prefix="presto_ft_", label_col="label")
        # Sampler for class balancing + quality weighting
        if self.use_balancing:
            sampler = self.get_balanced_sampler(trn_df, normalize=True)
            train_loader = DataLoader(
                train_ds, batch_size=1024, shuffle=False, sampler=sampler, num_workers=0
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=1024, shuffle=True, num_workers=0
            )
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)

        device_ = device
        model = self._build_model(in_dim, num_classes).to(device_)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        best_val = float("inf")
        best_state = None
        best_epoch = 0
        epochs_no_improve = 0
        stopped_early = False

        logger.info("Starting training...")
        for epoch in (pbar := tqdm(range(1, self.epochs + 1), desc="Training")):
            tr_loss = self._train_epoch(model, train_loader, optimizer, device_)
            val_loss, val_preds, val_labels = self._eval_epoch(
                model, val_loader, device_
            )
            scheduler.step(val_loss)
            if val_loss < (best_val - self.early_stopping_min_delta):
                best_val = val_loss
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
                    logger.info(
                        f"Early stopping triggered at epoch {epoch:03d} "
                        f"(no improvement for {epochs_no_improve} epochs)"
                    )
                    stopped_early = True
                    break
            logger.info(
                f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} best_val={best_val:.4f}"
            )

            description = (
                f"Epoch {epoch + 1}/{self.epochs + 1} | "
                f"Train Loss: {tr_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Best Loss: {best_val:.4f}"
            )

            description += (
                " (improved)"
                if epochs_no_improve == 0
                else f" (no improvement for {epochs_no_improve} epochs)"
            )

            pbar.set_description(description)
            pbar.set_postfix(lr=scheduler.get_last_lr()[0])

        if best_state is not None:
            model.load_state_dict(best_state)

        # Save training outcome to config
        self.update_config_after_training(
            best_val_loss=best_val,
            best_epoch=best_epoch,
            epochs_trained=epoch,
            stopped_early=stopped_early,
        )

        # Save model
        self.save_model(model)

        # Evaluate on test
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch classification head on Presto embeddings"
    )
    parser.add_argument(
        "--head_type",
        choices=["linear", "mlp"],
        default="linear",
        help="Head architecture",
    )
    parser.add_argument(
        "--presto_model_path",
        type=str,
        required=False,
        help="Path to fine-tuned Presto model (for on-the-fly embeddings)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with train_df/val_df/test_df or embeddings parquet files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to store outputs"
    )
    parser.add_argument("--timestep_freq", choices=["month", "dekad"], default="month")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=6,
        help="Number of epochs with no val improvement to tolerate before stopping (0 disables)",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum decrease in val loss to qualify as improvement",
    )
    parser.add_argument("--use_balancing", action="store_true")
    parser.add_argument(
        "--sampling_class",
        type=str,
        default="downstream_class",
        help="Column to balance by (typically 'downstream_class')",
    )
    parser.add_argument(
        "--balancing_method",
        type=str,
        choices=["log", "inverse", "sqrt"],
        default="log",
        help="Formula for class balance weights",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        nargs=2,
        default=(0.3, 5.0),
        metavar=("LOW", "HIGH"),
        help="Clip range for final per-sample weights",
    )
    parser.add_argument(
        "--quality_col",
        type=str,
        default=None,
        help="Optional column with per-sample quality weights",
    )

    return parser.parse_args()


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


def main() -> None:
    # Debug-friendly manual config (set True to bypass argparse)
    USE_MANUAL_CONFIG = True

    if USE_MANUAL_CONFIG:

        class ManualArgs:
            def __init__(self):
                self.head_type = "linear"
                self.presto_model_path = "/projects/worldcereal/models/presto-prometheo-landcover-MulticlassWithCroplandAuxBCELoss-labelsmoothing=0.05-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-masking=enabled-run=202510301004/presto-prometheo-landcover-MulticlassWithCroplandAuxBCELoss-labelsmoothing=0.05-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-masking=enabled-run=202510301004_encoder.pt"
                self.data_dir = "/projects/worldcereal/models/presto-prometheo-landcover-MulticlassWithCroplandAuxBCELoss-labelsmoothing=0.05-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-masking=enabled-run=202510301004/"
                self.output_dir = (
                    "/projects/worldcereal/models/downstream/LANDCOVER10_PyTorch/"
                )
                self.modelversion = "010_prestorun=202510301004"
                self.detector = "cropland"
                # self.downstream_classes = None
                self.downstream_classes = {
                    "temporary_crops": "cropland",
                    "temporary_grasses": "other",
                    "bare_sparsely_vegetated": "other",
                    "permanent_crops": "other",
                    "grasslands": "other",
                    "wetlands": "other",
                    "shrubland": "other",
                    "trees": "other",
                    "built_up": "other",
                    "water": "other",
                }
                self.timestep_freq = "month"
                self.batch_size = 1024
                self.num_workers = 8
                self.hidden_dim = 256
                self.dropout = 0.2
                self.lr = 1e-3
                self.epochs = 25
                self.seed = 42
                self.ref_id_to_keep_path = "/home/vito/vtrichtk/git/worldcereal-classification/ref_id_to_keep.json"
                self.use_balancing = True
                self.sampling_class = "downstream_class"
                self.balancing_method = "log"
                self.clip_range = (0.3, 5.0)
                self.early_stopping_patience = 6
                self.early_stopping_min_delta = 0.0

        args = ManualArgs()
        logger.info("Using manual configuration for debug mode (Torch head)")
    else:
        args = parse_args()  # type: ignore
        logger.info("Using command line arguments (Torch head)")

        # Parse downstream_classes JSON string if provided
        if args.downstream_classes is not None:
            try:
                args.downstream_classes = json.loads(args.downstream_classes)
                logger.info(f"Parsed downstream_classes: {args.downstream_classes}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for downstream_classes: {e}")

    plt.switch_backend("Agg")

    trainer = TorchHeadTrainer(
        head_type=args.head_type,
        presto_model_path=args.presto_model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        timestep_freq=args.timestep_freq,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        modelversion=args.modelversion,
        detector=args.detector,
        downstream_classes=args.downstream_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        ref_id_to_keep_path=getattr(args, "ref_id_to_keep_path", None)
        if hasattr(args, "ref_id_to_keep_path")
        else getattr(args, "ref_id_to_keep_path", None),
        use_balancing=getattr(args, "use_balancing", False),
        sampling_class=getattr(args, "sampling_class", "downstream_class"),
        balancing_method=getattr(args, "balancing_method", "log"),
        clip_range=getattr(args, "clip_range", (0.3, 5.0)),
        quality_col="quality_score_ct"
        if args.detector == "croptype"
        else "quality_score_lc",
        early_stopping_patience=getattr(args, "early_stopping_patience", 6),
        early_stopping_min_delta=getattr(args, "early_stopping_min_delta", 0.0),
    )

    trainer.train()
    logger.success("Torch head training completed successfully!")


if __name__ == "__main__":
    main()
