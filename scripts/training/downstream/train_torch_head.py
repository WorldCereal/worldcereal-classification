#!/usr/bin/env python3
"""Train a PyTorch classification head on Presto embeddings for seasonal inference.

# Option 1: Use single dataframe with automatic splitting
python train_torch_head.py \
  --single-dataframe /path/to/all_data.parquet \
  --val-split 0.2 --test-split 0.15 \
  --head-task croptype --season-id tc-s1 \
  --output-dir ./output

# Option 2: Use single dataframe with predefined split column
python train_torch_head.py \
  --single-dataframe /path/to/data_with_splits.parquet \
  --split-column my_split_col \
  --head-task croptype --season-id tc-s1 \
  --output-dir ./output

# Option 3: Existing behavior with separate files
python train_torch_head.py \
  --data-dir /path/to/splits/ \
  --head-task croptype --season-id tc-s1 \
  --output-dir ./output



"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from worldcereal.train.backbone import resolve_seasonal_encoder
from worldcereal.train.data import (
    compute_embeddings_from_splits,
    spatial_train_val_test_split,
)
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
        "--season-windows",
        type=str,
        default=None,
        help='JSON mapping of season_id to [start_date, end_date], e.g. \'{"tc-s1": ["2021-01-01", "2021-06-30"]}\'. '
        "Dates should be in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--presto-model-path",
        type=str,
        required=False,
        help="Path or URL to the Presto model checkpoint. Defaults to the packaged seasonal artifact when omitted.",
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
        "--single-dataframe",
        type=str,
        required=False,
        help="Parquet file containing all training data (will be split into train/val/test).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation set fraction when using --single-dataframe.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Test set fraction when using --single-dataframe.",
    )
    parser.add_argument(
        "--split-stratify-col",
        type=str,
        default="downstream_class",
        help="Column to stratify by when splitting --single-dataframe.",
    )
    parser.add_argument(
        "--split-column",
        type=str,
        default="split",
        help="Column name for predefined splits in --single-dataframe.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to store outputs."
    )
    parser.add_argument("--timestep-freq", choices=["month", "dekad"], default="month")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=50)
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
    args = parser.parse_args()
    if args.head_task == "croptype" and not args.season_id:
        parser.error("--season-id is required when training croptype heads.")
    return args


def _load_split_dataframes(data_dir: str) -> Dict[str, pd.DataFrame]:
    data_path = Path(data_dir)
    splits = {
        "train": pd.read_parquet(data_path / "train_df.parquet"),
        "val": pd.read_parquet(data_path / "val_df.parquet"),
        "test": pd.read_parquet(data_path / "test_df.parquet"),
    }
    for split_df in splits.values():
        if (
            "downstream_class" not in split_df.columns
            and "finetune_class" in split_df.columns
        ):
            split_df["downstream_class"] = split_df["finetune_class"]
    return splits


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
                self.head_task = "landcover"
                self.head_type = "linear"
                self.season_id = None
                self.season_calendar_mode = None
                self.season_windows = None
                self.presto_model_path = None
                self.data_dir = None
                self.embeddings_path = None
                self.single_dataframe = None
                self.val_split = 0.15
                self.test_split = 0.20
                self.split_stratify_col = "downstream_class"
                self.split_column = "split"
                self.output_dir = "."
                self.timestep_freq = "month"
                self.batch_size = 1024
                self.num_workers = 8
                self.hidden_dim = 256
                self.dropout = 0.2
                self.lr = 1e-2
                self.epochs = 50
                self.seed = 42
                self.modelversion = ""
                self.detector = "cropland"
                self.downstream_classes = {
                    "temporary_crops": "cropland",
                    "temporary_grasses": "other",
                    "permanent_crops": "cropland",
                    "grasslands": "other",
                    "wetlands": "other",
                    "bare_sparsely_vegetated": "other",
                    "shrubland": "other",
                    "trees": "other",
                    "built_up": "other",
                    "water": "other",
                }
                self.cropland_class_names = ["cropland"]
                self.use_balancing = True
                self.sampling_class = "downstream_class"
                self.balancing_method = "log"
                self.clip_range = None
                self.early_stopping_patience = 6
                self.early_stopping_min_delta = 0.0

        args: Any = ManualArgs()
        logger.info("Using manual configuration for debug mode (Torch head)")
    else:
        args = parse_args()
        logger.info("Using command line arguments (Torch head)")
    plt.switch_backend("Agg")

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

    presto_checkpoint = args.presto_model_path or resolve_seasonal_encoder()[0]

    # Parse season_windows if provided
    season_windows = None
    if args.season_windows:
        season_windows = _parse_optional_json(args.season_windows, "season-windows")
        if season_windows and not isinstance(season_windows, dict):
            raise ValueError("season_windows must be a JSON object/dict")

    embeddings_df = None
    if args.embeddings_path:
        logger.info(f"Loading embeddings dataframe from {args.embeddings_path}")
        embeddings_df = pd.read_parquet(args.embeddings_path)
    else:
        # Get splits from either single_dataframe or data_dir
        if args.single_dataframe:
            logger.info(f"Loading single dataframe from {args.single_dataframe}")
            full_df = pd.read_parquet(args.single_dataframe)
            if "anomaly_flag" in full_df.columns:
                logger.info("Dropping samples with anomaly_flag==candidate")
                full_df = full_df[~(full_df["anomaly_flag"] == "candidate")].copy()

            # Ensure downstream_class column exists
            if (
                "downstream_class" not in full_df.columns
                and "finetune_class" in full_df.columns
            ):
                full_df["downstream_class"] = full_df["finetune_class"]

            # Use existing train_val_test_split function
            train_df, val_df, test_df = spatial_train_val_test_split(
                full_df,
                split_column=args.split_column,
                val_size=args.val_split,
                test_size=args.test_split,
                seed=args.seed,
                stratify_label=args.split_stratify_col,
                min_samples_per_class=10,
                bin_size_degrees=1.0,
            )
            splits = {"train": train_df, "val": val_df, "test": test_df}
        elif args.data_dir:
            splits = _load_split_dataframes(args.data_dir)
        else:
            raise ValueError(
                "Either --embeddings-path, --single-dataframe, or --data-dir must be provided."
            )

        # Compute embeddings from splits using unified function
        # season_id=None triggers global pooling for landcover
        if args.season_id:
            logger.info(
                f"Computing seasonal embeddings for season {args.season_id} using {presto_checkpoint}"
            )
        else:
            logger.info(
                f"Computing globally pooled embeddings for landcover using {presto_checkpoint}"
            )

        embeddings_df = compute_embeddings_from_splits(
            splits["train"],
            splits["val"],
            splits["test"],
            presto_checkpoint,
            season_id=args.season_id,
            timestep_freq=args.timestep_freq,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            season_calendar_mode=args.season_calendar_mode,
            season_windows=season_windows,
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
        presto_model_path=presto_checkpoint,
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
        quality_col="quality_score_ct"
        if args.detector == "croptype"
        else "quality_score_lc",
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    trainer.train()
    logger.success("Torch head training completed successfully!")


if __name__ == "__main__":
    main()
