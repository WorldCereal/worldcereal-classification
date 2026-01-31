#!/usr/bin/env python3
"""Train a PyTorch classification head on Presto embeddings for seasonal inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from worldcereal.train import GLOBAL_SEASON_IDS
from worldcereal.train.data import (
    compute_seasonal_embeddings_from_splits,
    dataset_to_embeddings,
)
from worldcereal.train.datasets import SensorMaskingConfig, WorldCerealTrainingDataset
from worldcereal.train.downstream import TorchTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch head for the seasonal WorldCereal model."
    )
    parser.add_argument(
        "--head-task",
        choices=["croptype", "landcover"],
        default="croptype",
        help="Task for the downstream head.",
    )
    parser.add_argument(
        "--head-type",
        choices=["linear", "mlp"],
        default="linear",
        help="Head architecture.",
    )
    parser.add_argument(
        "--season-id",
        type=str,
        default=None,
        help="Season identifier for seasonal head training (e.g., tc-s1).",
    )
    parser.add_argument(
        "--season-calendar-mode",
        choices=["auto", "calendar", "custom", "off"],
        default="calendar",
        help="Season calendar strategy for deriving masks.",
    )
    parser.add_argument(
        "--presto-model-path",
        type=str,
        required=False,
        help="Path or URL to the Presto model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        help="Directory with train_df/val_df/test_df parquet files.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        required=False,
        help="Parquet file containing precomputed embeddings with split column.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to store outputs."
    )
    parser.add_argument("--timestep-freq", choices=["month", "dekad"], default="month")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modelversion", type=str, default="")
    parser.add_argument("--detector", type=str, default=None)
    parser.add_argument(
        "--downstream-classes",
        type=str,
        default=None,
        help="JSON mapping defining downstream class remapping.",
    )
    parser.add_argument(
        "--cropland-class-names",
        type=str,
        default=None,
        help="JSON list of cropland class names for gating.",
    )
    parser.add_argument("--use-balancing", action="store_true")
    parser.add_argument(
        "--sampling-class",
        type=str,
        default="downstream_class",
        help="Column to balance by.",
    )
    parser.add_argument(
        "--balancing-method",
        type=str,
        choices=["log", "inverse", "sqrt"],
        default="log",
        help="Formula for class balance weights.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        nargs=2,
        default=(0.3, 5.0),
        metavar=("LOW", "HIGH"),
        help="Clip range for final per-sample weights.",
    )
    parser.add_argument(
        "--quality-col",
        type=str,
        default=None,
        help="Optional column with per-sample quality weights.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
        help="Number of epochs with no val improvement to tolerate before stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum decrease in val loss to qualify as improvement.",
    )
    return parser.parse_args()


def _load_split_dataframes(data_dir: str) -> Dict[str, pd.DataFrame]:
    data_path = Path(data_dir)
    splits = {
        "train": pd.read_parquet(data_path / "train_df.parquet"),
        "val": pd.read_parquet(data_path / "val_df.parquet"),
        "test": pd.read_parquet(data_path / "test_df.parquet"),
    }
    for split_df in splits.values():
        if "downstream_class" not in split_df.columns and "finetune_class" in split_df.columns:
            split_df["downstream_class"] = split_df["finetune_class"]
    return splits


def _compute_embeddings_from_splits(
    splits: Dict[str, pd.DataFrame],
    presto_model_path: str,
    timestep_freq: str,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    num_timesteps = 12 if timestep_freq == "month" else 36
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

    def _build_dataset(df: pd.DataFrame, augment: bool) -> WorldCerealTrainingDataset:
        return WorldCerealTrainingDataset(
            df,
            num_timesteps=num_timesteps,
            timestep_freq=timestep_freq,
            task_type="multiclass",
            augment=augment,
            masking_config=masking_config if augment else SensorMaskingConfig(enable=False),
            repeats=3 if augment else 1,
        )

    train_ds = _build_dataset(splits["train"], augment=True)
    val_ds = _build_dataset(splits["val"], augment=False)
    test_ds = _build_dataset(splits["test"], augment=False)

    from prometheo.models import Presto
    from prometheo.models.presto.wrapper import load_presto_weights
    from prometheo.utils import device

    presto_model = Presto()
    presto_model = load_presto_weights(presto_model, presto_model_path).to(device)

    train_embeddings = dataset_to_embeddings(
        train_ds, presto_model, batch_size=batch_size, num_workers=num_workers
    )
    val_embeddings = dataset_to_embeddings(
        val_ds, presto_model, batch_size=batch_size, num_workers=num_workers
    )
    test_embeddings = dataset_to_embeddings(
        test_ds, presto_model, batch_size=batch_size, num_workers=num_workers
    )

    train_embeddings["split"] = "train"
    val_embeddings["split"] = "val"
    test_embeddings["split"] = "test"
    return (
        pd.concat([train_embeddings, val_embeddings, test_embeddings])
        .reset_index(drop=True)
        .copy()
    )


def _parse_optional_json(value: Optional[str], expected: str) -> Optional[Any]:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {expected}: {exc}") from exc


def main() -> None:
    USE_MANUAL_CONFIG = False

    if USE_MANUAL_CONFIG:

        class ManualArgs:
            def __init__(self):
                self.head_task = "croptype"
                self.head_type = "linear"
                self.season_id = "tc-s1"
                self.season_calendar_mode = "calendar"
                self.presto_model_path = "/path/to/presto_encoder.pt"
                self.data_dir = "/path/to/parquet_splits"
                self.embeddings_path = None
                self.output_dir = "./downstream_classifier"
                self.timestep_freq = "month"
                self.batch_size = 1024
                self.num_workers = 8
                self.hidden_dim = 256
                self.dropout = 0.2
                self.lr = 1e-3
                self.epochs = 10
                self.seed = 42
                self.modelversion = ""
                self.detector = None
                self.downstream_classes = None
                self.cropland_class_names = None
                self.use_balancing = True
                self.sampling_class = "downstream_class"
                self.balancing_method = "log"
                self.clip_range = (0.3, 5.0)
                self.quality_col = None
                self.early_stopping_patience = 6
                self.early_stopping_min_delta = 0.0

        args = ManualArgs()
        logger.info("Using manual configuration for debug mode (Torch head)")
    else:
        args = parse_args()
        logger.info("Using command line arguments (Torch head)")
    plt.switch_backend("Agg")

    if args.season_id is not None and args.season_id not in GLOBAL_SEASON_IDS:
        raise ValueError(
            f"Season id {args.season_id!r} must be one of {list(GLOBAL_SEASON_IDS)}."
        )

    downstream_classes = None
    if isinstance(args.downstream_classes, str):
        downstream_classes = _parse_optional_json(
            args.downstream_classes, "downstream-classes"
        )
    else:
        downstream_classes = args.downstream_classes

    cropland_class_names = None
    if isinstance(args.cropland_class_names, str):
        cropland_class_names = _parse_optional_json(
            args.cropland_class_names, "cropland-class-names"
        )
    else:
        cropland_class_names = args.cropland_class_names

    embeddings_df = None
    if args.embeddings_path:
        logger.info("Loading embeddings dataframe from %s", args.embeddings_path)
        embeddings_df = pd.read_parquet(args.embeddings_path)
    else:
        if args.data_dir is None or args.presto_model_path is None:
            raise ValueError(
                "Either --embeddings-path or both --data-dir and --presto-model-path must be provided."
            )
        splits = _load_split_dataframes(args.data_dir)

        if args.season_id:
            logger.info(
                "Computing seasonal embeddings for season %s using %s",
                args.season_id,
                args.presto_model_path,
            )
            embeddings_df = compute_seasonal_embeddings_from_splits(
                splits["train"],
                splits["val"],
                splits["test"],
                args.presto_model_path,
                season_id=args.season_id,
                timestep_freq=args.timestep_freq,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                season_calendar_mode=args.season_calendar_mode,
            )
        else:
            logger.info(
                "Computing pooled embeddings using %s", args.presto_model_path
            )
            embeddings_df = _compute_embeddings_from_splits(
                splits,
                presto_model_path=args.presto_model_path,
                timestep_freq=args.timestep_freq,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

    trainer = TorchTrainer(
        embeddings_df,
        head_type=args.head_type,
        head_task=args.head_task,
        output_dir=args.output_dir,
        modelversion=args.modelversion,
        detector=args.detector,
        downstream_classes=downstream_classes,
        cropland_class_names=cropland_class_names,
        season_id=args.season_id,
        presto_model_path=args.presto_model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        use_balancing=args.use_balancing,
        balancing_label=args.sampling_class,
        balancing_method=args.balancing_method,
        weights_clip_range=args.clip_range,
        quality_col=args.quality_col,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    trainer.train()
    logger.success("Torch head training completed successfully!")


if __name__ == "__main__":
    main()
