#!/usr/bin/env python3
"""Train a CatBoost classifier on presto embeddings.

This script can work with either:
1. Raw parquet files (train_df.parquet, val_df.parquet, test_df.parquet) - computes embeddings on-the-fly
2. Pre-computed embedding files - skips embedding computation

The script uses the Trainer class pattern from the old version with modern evaluation
and on-the-fly embedding computation from the new version.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from loguru import logger
from prometheo.models import Presto
from prometheo.models.presto.wrapper import load_presto_weights
from prometheo.predictors import Predictors
from prometheo.utils import device
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from worldcereal.train.datasets import (
    WorldCerealLabelledDataset,
    get_class_weights,
)
from worldcereal.utils.refdata import map_classes


class PrestoEmbeddingTrainer:
    """Trainer class for CatBoost models on Presto embeddings."""

    def __init__(
        self,
        presto_model_path: str,
        data_dir: str,
        output_dir: str,
        finetune_classes: str = "LANDCOVER10",
        timestep_freq: str = "month",
        batch_size: int = 1024,
        num_workers: int = 8,
        modelversion: str = "001",
        detector: str = "cropland",
        downstream_classes: Optional[dict] = None,
    ):
        self.presto_model_path = Path(presto_model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.finetune_classes = finetune_classes
        self.timestep_freq = timestep_freq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modelversion = modelversion
        self.detector = detector
        self.downstream_classes = downstream_classes

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

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.sink = logger.add(
            self.output_dir / "logfile.log",
            level="DEBUG",
        )

        # Initialize config
        self.config: Dict[str, Any] = {}

    def _check_for_embeddings(self) -> bool:
        """Check if pre-computed embeddings exist."""
        emb_files = [
            "train_embeddings.parquet",
            "val_embeddings.parquet",
            "test_embeddings.parquet",
        ]
        return all((self.output_dir / f).exists() for f in emb_files)

    def _dataloader_to_encodings_and_ids(
        self, model: Presto, dl: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate embeddings and sample IDs from a dataloader."""
        encs: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        model.eval()
        if hasattr(model, "head"):
            model.head = None
        with torch.no_grad():
            for predictors, sample_ids in tqdm(
                dl, desc="Computing Embeddings", leave=False
            ):
                if not isinstance(predictors, Predictors):
                    predictors = Predictors(**predictors)
                out = model(predictors)
                encs.append(out.cpu().numpy())
                if isinstance(sample_ids, torch.Tensor):
                    ids.append(sample_ids.cpu().numpy())
                else:
                    ids.append(np.asarray(sample_ids))
        enc_np = np.concatenate(encs, axis=0)
        id_np = np.concatenate([i.reshape(-1) for i in ids], axis=0)
        return enc_np, id_np

    def _emb_to_df(
        self, embeddings: np.ndarray, ids: np.ndarray, original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert embeddings and IDs to DataFrame."""
        df = pd.DataFrame(np.squeeze(embeddings))
        df.columns = [f"emb_{i}" for i in range(df.shape[1])]
        df["sample_id"] = ids
        df["ewoc_code"] = df["sample_id"].map(
            original_df.set_index("sample_id")["ewoc_code"].to_dict()
        )
        # Also map finetune_class from original dataframe
        df["finetune_class"] = df["sample_id"].map(
            original_df.set_index("sample_id")["finetune_class"].to_dict()
        )
        return df

    def _compute_embeddings(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute embeddings from raw parquet files."""
        logger.info("Loading raw data files...")
        train_df = pd.read_parquet(self.data_dir / "train_df.parquet")
        val_df = pd.read_parquet(self.data_dir / "val_df.parquet")
        test_df = pd.read_parquet(self.data_dir / "test_df.parquet")

        orig_classes = sorted(train_df["finetune_class"].unique())

        # Load Presto model
        logger.info(f"Loading Presto model from {self.presto_model_path}...")
        num_timesteps = 12 if self.timestep_freq == "month" else 36
        presto_model = Presto(num_outputs=len(orig_classes), regression=False)
        presto_model = load_presto_weights(presto_model, self.presto_model_path).to(
            device
        )

        # Create datasets and dataloaders
        logger.info("Creating datasets...")
        trn_ds = WorldCerealLabelledDataset(
            train_df,
            num_timesteps=num_timesteps,
            timestep_freq=self.timestep_freq,
            task_type="multiclass",
            num_outputs=len(orig_classes),
            classes_list=orig_classes,
            augment=True,
            return_sample_id=True,
        )
        val_ds = WorldCerealLabelledDataset(
            val_df,
            num_timesteps=num_timesteps,
            timestep_freq=self.timestep_freq,
            task_type="multiclass",
            num_outputs=len(orig_classes),
            classes_list=orig_classes,
            return_sample_id=True,
        )
        test_ds = WorldCerealLabelledDataset(
            test_df,
            num_timesteps=num_timesteps,
            timestep_freq=self.timestep_freq,
            task_type="multiclass",
            num_outputs=len(orig_classes),
            classes_list=orig_classes,
            return_sample_id=True,
        )

        trn_dl = DataLoader(
            trn_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Compute embeddings
        logger.info("Computing embeddings...")
        trn_emb, trn_ids = self._dataloader_to_encodings_and_ids(presto_model, trn_dl)
        val_emb, val_ids = self._dataloader_to_encodings_and_ids(presto_model, val_dl)
        tst_emb, tst_ids = self._dataloader_to_encodings_and_ids(presto_model, test_dl)

        logger.info("Converting embeddings to DataFrames...")
        trn_df_emb = self._emb_to_df(trn_emb, trn_ids, train_df)
        val_df_emb = self._emb_to_df(val_emb, val_ids, val_df)
        tst_df_emb = self._emb_to_df(tst_emb, tst_ids, test_df)

        # Save embeddings for future use
        logger.info("Saving computed embeddings...")
        trn_df_emb.to_parquet(self.output_dir / "train_embeddings.parquet")
        val_df_emb.to_parquet(self.output_dir / "val_embeddings.parquet")
        tst_df_emb.to_parquet(self.output_dir / "test_embeddings.parquet")

        return trn_df_emb, val_df_emb, tst_df_emb

    def _load_embeddings(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load pre-computed embeddings."""
        logger.info("Loading pre-computed embeddings...")
        trn_df_emb = pd.read_parquet(self.output_dir / "train_embeddings.parquet")
        val_df_emb = pd.read_parquet(self.output_dir / "val_embeddings.parquet")
        tst_df_emb = pd.read_parquet(self.output_dir / "test_embeddings.parquet")

        return trn_df_emb, val_df_emb, tst_df_emb

    def _get_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get training data - either compute or load embeddings."""
        if self._check_for_embeddings():
            logger.info("Found pre-computed embeddings, loading them...")
            return self._load_embeddings()
        else:
            logger.info("No pre-computed embeddings found, computing them ...")
            return self._compute_embeddings()

    def _setup_model(self) -> CatBoostClassifier:
        """Setup the CatBoost model."""
        logger.info("Setting up CatBoost model...")

        # Determine loss function and eval metric based on binary/multiclass mode
        if self.is_binary:
            loss_function = "Logloss"
            eval_metric = "F1"
        else:
            loss_function = "MultiClass"
            eval_metric = "MultiClass"

        model = CatBoostClassifier(
            iterations=6000,
            depth=6,
            learning_rate=0.05,
            loss_function=loss_function,
            eval_metric=eval_metric,
            early_stopping_rounds=25,
            task_type="GPU" if torch.cuda.is_available() else "CPU",
            devices="0" if torch.cuda.is_available() else None,
            thread_count=4,
            random_state=42,
            l2_leaf_reg=3,
            verbose=100,
            class_names=self.classes_list,
            train_dir=self.output_dir,
        )

        # Save model parameters to config
        model_params = model.get_params()
        model_params["train_dir"] = str(model_params["train_dir"])
        self.config["model_params"] = model_params
        self.save_config()
        logger.info(f"Model parameters: {model_params}")

        return model

    def _prepare_training_data(
        self, trn_df: pd.DataFrame, val_df: pd.DataFrame, tst_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare training data with weights and feature selection."""
        # Get feature columns
        feat_cols = [c for c in trn_df.columns if c.startswith("emb_")]
        self.feat_cols = feat_cols

        # Calculate class weights
        logger.info("Calculating class weights...")
        class_weights = get_class_weights(
            trn_df[self.target_column].values,
            method="log",
            clip_range=(0.2, 10),
            normalize=True,
        )

        # Apply sample weights
        sample_weights = np.ones_like(
            trn_df[self.target_column].values, dtype=np.float32
        )
        for k, v in class_weights.items():
            sample_weights[trn_df[self.target_column].values == k] = v
        trn_df["weight"] = sample_weights
        val_df["weight"] = 1.0  # Validation weights are uniform
        tst_df["weight"] = 1.0  # Test weights are uniform

        # Add label column using target column
        trn_df["label"] = trn_df[self.target_column]
        val_df["label"] = val_df[self.target_column]
        tst_df["label"] = tst_df[self.target_column]

        # Save class information to config
        self.config["classes"] = {
            str(i): str(cls) for i, cls in enumerate(self.classes_list)
        }
        self.config["class_weights"] = {
            str(k): float(v) for k, v in class_weights.items()
        }
        self.save_config()

        return trn_df, val_df, tst_df

    def _setup_datapools(
        self, cal_data: pd.DataFrame, val_data: pd.DataFrame
    ) -> tuple[Pool, Pool]:
        """Setup CatBoost data pools."""
        calibration_pool = Pool(
            data=cal_data[self.feat_cols],
            label=cal_data["label"],
            weight=cal_data["weight"],
        )
        eval_pool = Pool(
            data=val_data[self.feat_cols],
            label=val_data["label"],
            weight=val_data["weight"],
        )
        return calibration_pool, eval_pool

    def train(self) -> CatBoostClassifier:
        """Train the CatBoost model."""
        # Get training data
        trn_df, val_df, tst_df = self._get_training_data()

        # Map classes
        logger.info("Mapping classes...")
        trn_df = map_classes(trn_df, finetune_classes=self.finetune_classes)
        val_df = map_classes(val_df, finetune_classes=self.finetune_classes)
        tst_df = map_classes(tst_df, finetune_classes=self.finetune_classes)

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

        # Prepare data
        trn_df, val_df, tst_df = self._prepare_training_data(trn_df, val_df, tst_df)

        # Update config with final class information
        self.config["final_classes"] = self.classes_list
        if hasattr(self, "target_class_name"):
            self.config["target_class_name"] = self.target_class_name
        self.save_config()

        # Save processed data
        logger.info("Saving processed data...")
        trn_df.to_parquet(self.output_dir / "processed_calibration_df.parquet")
        val_df.to_parquet(self.output_dir / "processed_validation_df.parquet")
        tst_df.to_parquet(self.output_dir / "processed_test_df.parquet")

        # Setup model
        model = self._setup_model()

        # Setup data pools
        calibration_pool, eval_pool = self._setup_datapools(trn_df, val_df)

        # Train model
        logger.info("Starting training...")
        model.fit(
            calibration_pool,
            eval_set=eval_pool,
        )

        # Save model
        self.save_model(model)

        # Evaluate model
        self.evaluate(model, tst_df)

        # Plot feature importance
        self._plot_feature_importance(model)

        return model

    def save_model(self, model: CatBoostClassifier) -> None:
        """Save model in both CBM and ONNX formats."""
        modelname = f"PrestoDownstreamCatBoost_{self.detector}_v{self.modelversion}"

        # Save as CBM
        cbm_path = self.output_dir / f"{modelname}.cbm"
        model.save_model(cbm_path)
        logger.info(f"Model saved as CBM: {cbm_path}")

        # Save as ONNX
        onnx_path = self.output_dir / f"{modelname}.onnx"
        model.save_model(
            str(onnx_path),
            format="onnx",
            export_parameters={
                "onnx_domain": "ai.catboost",
                "onnx_model_version": 1,
                "onnx_doc_string": f"Default {self.detector} model using CatBoost",
                "onnx_graph_name": f"CatBoostModel_for_{self.detector}",
            },
        )
        logger.info(f"Model saved as ONNX: {onnx_path}")

    def evaluate(self, model: CatBoostClassifier, test_df: pd.DataFrame) -> dict:
        """Evaluate the model on test data."""
        logger.info("Evaluating model...")

        # Get predictions
        preds = model.predict(test_df[self.feat_cols])
        true_labels = test_df["label"].values

        # Classification report
        report = classification_report(
            true_labels, preds, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(self.output_dir / "classification_report.csv")

        # Confusion matrices
        self._plot_confusion_matrices(true_labels, preds)

        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, preds)

        # Save metrics
        with open(self.output_dir / "metrics.txt", "w") as f:
            f.write("Test results:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
                logger.info(f"{key} = {value}")

        return metrics

    def _plot_confusion_matrices(
        self, true_labels: np.ndarray, preds: np.ndarray
    ) -> None:
        """Plot confusion matrices (absolute and normalized)."""
        fig_size = max(6, len(self.classes_list) * 0.45)

        # Absolute confusion matrix
        cm = ConfusionMatrixDisplay.from_predictions(
            true_labels, preds, xticks_rotation="vertical"
        )
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        cm.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)  # type: ignore
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        plt.tight_layout()
        plt.savefig(self.output_dir / "CM_abs.png")
        plt.close(fig)

        # Normalized confusion matrix
        cm_norm = ConfusionMatrixDisplay.from_predictions(
            true_labels, preds, normalize="true", xticks_rotation="vertical"
        )
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        cm_norm.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)  # type: ignore
        for text in ax.texts:
            val = float(text.get_text())
            text.set_text(f"{val:.2f}")
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        plt.tight_layout()
        plt.savefig(self.output_dir / "CM_norm.png")
        plt.close(fig)

    def _calculate_metrics(self, true_labels: np.ndarray, preds: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        metrics = {}

        if len(self.classes_list) == 2:
            # Binary classification - use the second class as positive label
            pos_label = self.classes_list[1]

            metrics["OA"] = round(accuracy_score(true_labels, preds), 3)
            metrics["F1"] = round(f1_score(true_labels, preds, pos_label=pos_label), 3)
            metrics["Precision"] = round(
                precision_score(true_labels, preds, pos_label=pos_label), 3
            )
            metrics["Recall"] = round(
                recall_score(true_labels, preds, pos_label=pos_label), 3
            )
        else:
            # Multiclass classification
            metrics["OA"] = round(accuracy_score(true_labels, preds), 3)
            metrics["F1"] = round(f1_score(true_labels, preds, average="macro"), 3)
            metrics["Precision"] = round(
                precision_score(true_labels, preds, average="macro"), 3
            )
            metrics["Recall"] = round(
                recall_score(true_labels, preds, average="macro"), 3
            )

        return metrics

    def _plot_feature_importance(self, model: CatBoostClassifier) -> None:
        """Plot feature importance."""
        logger.info("Plotting feature importance...")
        ft_imp = model.get_feature_importance()
        sorting = np.argsort(np.array(ft_imp))[::-1]

        f, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.bar(np.array(self.feat_cols)[sorting], np.array(ft_imp)[sorting])
        ax.set_xticklabels(np.array(self.feat_cols)[sorting], rotation=90)
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png")
        plt.close(f)

    def create_config(self) -> None:
        """Create initial configuration."""
        self.config = {
            "presto_model_path": str(self.presto_model_path),
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "finetune_classes": self.finetune_classes,
            "timestep_freq": self.timestep_freq,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "modelversion": self.modelversion,
            "detector": self.detector,
            "downstream_classes": self.downstream_classes,
            "is_binary": self.is_binary,
        }

        self.save_config()

    def save_config(self) -> None:
        """Save configuration to JSON file."""
        config_path = self.output_dir / "config.json"

        # Convert any numpy types to Python native types
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CatBoost on Presto embeddings")
    parser.add_argument(
        "--presto_model_path",
        type=str,
        required=True,
        help="Path to fine-tuned Presto model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train_df.parquet, val_df.parquet and test_df.parquet OR pre-computed embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store CatBoost models and reports",
    )
    parser.add_argument(
        "--finetune_classes",
        type=str,
        default="LANDCOVER10",
        help="Class mapping scheme to use",
    )
    parser.add_argument(
        "--timestep_freq",
        choices=["month", "dekad"],
        default="month",
        help="Temporal frequency for time series",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--modelversion", type=str, default="001", help="Model version identifier"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="cropland",
        help="Type of detector (cropland, croptype, etc.)",
    )
    parser.add_argument(
        "--downstream_classes",
        type=str,
        default=None,
        help='JSON string mapping finetune_classes to downstream classes. Example: \'{"class1": "target", "class2": "non_target"}\'. If not specified, finetune_classes are used directly. If resulting classes are binary, binary mode is automatically enabled.',
    )
    return parser.parse_args()


def main() -> None:
    """Main training function."""

    # =============================================================================
    # MANUAL CONFIGURATION FOR DEBUG MODE
    # Set USE_MANUAL_CONFIG = True to bypass argparser and use manual settings
    # =============================================================================
    USE_MANUAL_CONFIG = True

    if USE_MANUAL_CONFIG:
        # Manual configuration - edit these values as needed
        class ManualArgs:
            def __init__(self):
                self.presto_model_path = "/projects/worldcereal/models/presto-prometheo-landcover-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-run=202507170930/presto-prometheo-landcover-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-run=202507170930.pt"
                self.data_dir = "/projects/worldcereal/models/presto-prometheo-landcover-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-run=202507170930/"
                self.output_dir = "./CROPLAND2"
                self.finetune_classes = "LANDCOVER10"
                self.timestep_freq = "month"
                self.batch_size = 1024
                self.num_workers = 8
                self.modelversion = "001-debug"
                self.detector = "temporary-crops"
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

        args = ManualArgs()
        logger.info("Using manual configuration for debug mode")
    else:
        args = parse_args()  # type: ignore
        logger.info("Using command line arguments")

        # Parse downstream_classes JSON string if provided
        if args.downstream_classes is not None:
            try:
                args.downstream_classes = json.loads(args.downstream_classes)
                logger.info(f"Parsed downstream_classes: {args.downstream_classes}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for downstream_classes: {e}")

    # Plot without display
    plt.switch_backend("Agg")

    # Create trainer
    trainer = PrestoEmbeddingTrainer(
        presto_model_path=args.presto_model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        finetune_classes=args.finetune_classes,
        timestep_freq=args.timestep_freq,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        modelversion=args.modelversion,
        detector=args.detector,
        downstream_classes=args.downstream_classes,
    )

    # Create initial config
    trainer.create_config()

    # Train model
    trainer.train()

    logger.success("Training completed successfully!")


if __name__ == "__main__":
    main()
