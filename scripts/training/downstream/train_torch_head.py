#!/usr/bin/env python3
"""Train a PyTorch classification head (Linear or small MLP) on Presto embeddings.

This complements `train_catboost.py` by using a neural head instead of CatBoost.
It supports either computing embeddings on-the-fly or loading precomputed embeddings
and benchmarks accuracy/F1 against the CatBoost approach.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from prometheo.utils import device
from sklearn.model_selection import split_training_dataframe

from worldcereal.train.data import dataset_to_embeddings

from worldcereal.train.downstream import TorchTrainer


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
        required=False,
        help="Directory with train_df/val_df/test_df parquet files (required when embeddings are computed on the fly)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=False,
        help="Optional single parquet file containing embeddings + metadata for splitting",
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
    parser.add_argument(
        "--ref_id_to_keep_path",
        type=str,
        default=None,
        help="Optional JSON file listing ref_ids to retain before training",
    )
    parser.add_argument(
        "--split_by_ref_id",
        action="store_true",
        help="Rebuild train/val/test splits based on unique ref_id values",
    )
    parser.add_argument(
        "--split_column",
        type=str,
        default="split",
        help="Column name that stores split labels inside the embeddings dataframe (if available)",
    )
    parser.add_argument(
        "--train_split_label",
        type=str,
        default="train",
        help="Label identifying training rows inside split_column",
    )
    parser.add_argument(
        "--val_split_label",
        type=str,
        default="val",
        help="Label identifying validation rows inside split_column (optional)",
    )
    parser.add_argument(
        "--test_split_label",
        type=str,
        default="test",
        help="Label identifying test rows inside split_column",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation when deriving splits from a single dataframe",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing when no split column is available",
    )
    parser.add_argument(
        "--stratify_column",
        type=str,
        default="downstream_class",
        help="Column to stratify on when creating splits from scratch",
    )

    return parser.parse_args()


EMBEDDING_FILES = {
    "train": "train_embeddings.parquet",
    "val": "val_embeddings.parquet",
    "test": "test_embeddings.parquet",
}


def cached_embeddings_exist(output_dir: str) -> bool:
    out = Path(output_dir)
    return all((out / fname).exists() for fname in EMBEDDING_FILES.values())


def load_cached_embedding_splits(output_dir: str) -> Dict[str, pd.DataFrame]:
    out = Path(output_dir)
    logger.info("Loading cached embeddings from %s", out)
    return {
        split: pd.read_parquet(out / fname) for split, fname in EMBEDDING_FILES.items()
    }


def compute_embeddings_from_data_dir(
    data_dir: str,
    presto_model_path: str,
    timestep_freq: str,
    batch_size: int,
    num_workers: int,
    output_dir: str,
) -> Dict[str, pd.DataFrame]:
    if data_dir is None or presto_model_path is None:
        raise ValueError(
            "Both data_dir and presto_model_path are required to compute embeddings."
        )

    data_path = Path(data_dir)
    logger.info("Loading raw parquet splits from %s", data_path)
    train_df = pd.read_parquet(data_path / "train_df.parquet")
    val_df = pd.read_parquet(data_path / "val_df.parquet")
    test_df = pd.read_parquet(data_path / "test_df.parquet")

    orig_classes = sorted(train_df["finetune_class"].unique())
    for df in (train_df, val_df, test_df):
        df["downstream_class"] = df["finetune_class"]






    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trn_df.to_parquet(out / EMBEDDING_FILES["train"])
    val_df.to_parquet(out / EMBEDDING_FILES["val"])
    tst_df.to_parquet(out / EMBEDDING_FILES["test"])

    return {"train": trn_df, "val": val_df, "test": tst_df}


def combine_with_split_labels(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    split_sizes = {}
    for split_name, df in splits.items():
        if df is None or df.empty:
            continue
        tmp = df.copy().reset_index(drop=True)
        tmp["split"] = split_name
        frames.append(tmp)
        split_sizes[split_name] = len(tmp)
    if not frames:
        raise ValueError("No samples available to train on after preparing splits.")
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Prepared dataframe with split column: train=%d val=%d test=%d",
        split_sizes.get("train", 0),
        split_sizes.get("val", 0),
        split_sizes.get("test", 0),
    )
    return combined


def filter_by_ref_ids(df: pd.DataFrame, ref_id_path: Optional[str]) -> pd.DataFrame:
    if not ref_id_path:
        return df
    ref_path = Path(ref_id_path)
    if not ref_path.exists():
        logger.warning("ref_id file %s not found; skipping filtering.", ref_path)
        return df
    with open(ref_path, "r") as f:
        ref_content = json.load(f)
    if isinstance(ref_content, dict):
        ref_ids = list(ref_content.keys())
    else:
        ref_ids = list(ref_content)
    filtered = df[df["ref_id"].isin(ref_ids)].reset_index(drop=True)
    logger.info(
        "Filtered dataframe to %d rows using %d ref_ids", len(filtered), len(ref_ids)
    )
    return filtered


def split_dataframe_by_ref_id(
    df: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> pd.DataFrame:
    if "ref_id" not in df.columns:
        logger.warning(
            "ref_id column missing; cannot split by ref_id. Returning original dataframe."
        )
        return df
    if train_fraction <= 0 or train_fraction >= 1:
        raise ValueError("train_fraction must be in (0, 1).")
    if val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("val_fraction must keep train+val < 1.")

    unique_ids = df["ref_id"].unique()
    if len(unique_ids) < 3:
        logger.warning(
            "Not enough unique ref_ids (%d) to resplit; leaving original splits.",
            len(unique_ids),
        )
        return df

    train_ids, temp_ids = train_test_split(
        unique_ids, test_size=1 - train_fraction, random_state=seed
    )
    val_ratio = val_fraction / (1 - train_fraction)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=1 - val_ratio, random_state=seed
    )

    reassigned = df.copy().reset_index(drop=True)
    reassigned["split"] = "test"
    reassigned.loc[reassigned["ref_id"].isin(train_ids), "split"] = "train"
    reassigned.loc[reassigned["ref_id"].isin(val_ids), "split"] = "val"
    logger.info(
        "Ref_id-based split sizes | train=%d val=%d test=%d",
        (reassigned["split"] == "train").sum(),
        (reassigned["split"] == "val").sum(),
        (reassigned["split"] == "test").sum(),
    )
    return reassigned


def main() -> None:
    # Debug-friendly manual config (set True to bypass argparse)
    USE_MANUAL_CONFIG = True

    if USE_MANUAL_CONFIG:

        class ManualArgs:
            def __init__(self):
                self.head_type = "linear"
                self.presto_model_path = "/projects/worldcereal/models/presto-prometheo-landcover-MulticlassWithCroplandAuxBCELoss-labelsmoothing=0.05-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-masking=enabled-run=202510301004/presto-prometheo-landcover-MulticlassWithCroplandAuxBCELoss-labelsmoothing=0.05-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-masking=enabled-run=202510301004_encoder.pt"
                self.data_dir = "/projects/worldcereal/models/presto-prometheo-landcover-MulticlassWithCroplandAuxBCELoss-labelsmoothing=0.05-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-masking=enabled-run=202510301004/"
                self.embeddings_path = None
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
                self.split_column = "split"
                self.train_split_label = "train"
                self.val_split_label = "val"
                self.test_split_label = "test"
                self.val_fraction = 0.2
                self.test_fraction = 0.2
                self.stratify_column = "downstream_class"

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

    embeddings_df = None
    if getattr(args, "embeddings_path", None):
        logger.info(f"Loading embeddings dataframe from {args.embeddings_path}")
        embeddings_df = pd.read_parquet(args.embeddings_path)

    if args.data_dir is None and embeddings_df is None:
        logger.info(
            "No data_dir or embeddings_path provided; will look for cached embeddings inside the output directory."
        )

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

    trainer = TorchTrainer(
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
        embeddings_df=embeddings_df,
        split_column=getattr(args, "split_column", None),
        train_split_label=getattr(args, "train_split_label", "train"),
        val_split_label=getattr(args, "val_split_label", "val"),
        test_split_label=getattr(args, "test_split_label", "test"),
        val_fraction=getattr(args, "val_fraction", 0.2),
        test_fraction=getattr(args, "test_fraction", 0.2),
        stratify_col=getattr(args, "stratify_column", "downstream_class"),
        ref_id_to_keep_path=getattr(args, "ref_id_to_keep_path", None)
        if hasattr(args, "ref_id_to_keep_path")
        else getattr(args, "ref_id_to_keep_path", None),
        use_balancing=getattr(args, "use_balancing", False),
        balancing_label=getattr(args, "sampling_class", "downstream_class"),
        balancing_method=getattr(args, "balancing_method", "log"),
        weights_clip_range=getattr(args, "clip_range", (0.3, 5.0)),
        quality_col="quality_score_ct"
        if args.detector == "croptype"
        else "quality_score_lc",
        early_stopping_patience=getattr(args, "early_stopping_patience", 6),
        early_stopping_min_delta=getattr(args, "early_stopping_min_delta", 0.0),
    )

    trainer.train()
    logger.success("Torch head training completed successfully!")
