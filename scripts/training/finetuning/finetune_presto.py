#!/usr/bin/env python3
import argparse
import json
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import (Any, Dict, List, Literal, Optional, Sequence, Tuple, Union,
                    cast)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib.figure import Figure
from prometheo.finetune import Hyperparams
from prometheo.models import Presto
from prometheo.models.presto import param_groups_lrd
from prometheo.predictors import NODATAVALUE
from prometheo.utils import DEFAULT_SEED, device, initialize_logging
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from worldcereal.train import GLOBAL_SEASON_IDS
from worldcereal.train.backbone import checkpoint_fingerprint
from worldcereal.train.data import (collate_fn, get_training_dfs_from_parquet,
                                    remove_small_classes)
from worldcereal.train.datasets import SensorMaskingConfig
from worldcereal.train.finetuning_utils import (SeasonalMultiTaskLoss,
                                                evaluate_finetuned_model,
                                                prepare_training_datasets,
                                                run_finetuning)
from worldcereal.train.seasonal_head import (SeasonalFinetuningHead,
                                             WorldCerealSeasonalModel)
from worldcereal.utils.refdata import get_class_mappings

CLASS_MAPPINGS = get_class_mappings(source="sharepoint")


def _path_to_str(path: Optional[Path]) -> Optional[str]:
    return str(path) if path is not None else None


def _object_to_dict(obj):
    if obj is None:
        return None
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except TypeError:
        pass
    if hasattr(obj, "__dict__"):
        return {
            key: value for key, value in vars(obj).items() if not key.startswith("_")
        }
    return obj


def _unique_preserve_order(values):
    """Return unique items from iterable while preserving their first occurrence."""

    seen = set()
    ordered = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _is_ignore_label(value) -> bool:
    try:
        return str(value).strip().lower() == "ignore"
    except Exception:  # noqa: BLE001
        return False


def _filter_ignore_labels(values):
    """Drop class labels named 'ignore' (case-insensitive) while preserving order."""
    return [
        value for value in values if value is not None and not _is_ignore_label(value)
    ]


def _drop_outliers(
    df: pd.DataFrame,
    split_name: str,
    outlier_col: str = "LC10_anomaly_flag",
    drop_level: Literal[
        "drop_candidate", "drop_suspect", "drop_flagged"
    ] = "drop_candidate",
) -> pd.DataFrame:

    if outlier_col not in df.columns:
        logger.warning(
            f"Outlier drop requested but '{outlier_col}' column is missing in {split_name} split."
        )
        return df

    if drop_level == "drop_candidate":
        outliers = df[df[outlier_col] == "candidate"]["sample_id"].tolist()
    elif drop_level == "drop_suspect":
        outliers = df[df[outlier_col].isin(["candidate", "suspect"])][
            "sample_id"
        ].tolist()
    elif drop_level == "drop_flagged":
        outliers = df[df[outlier_col].isin(["candidate", "suspect", "flagged"])][
            "sample_id"
        ].tolist()
    else:
        raise ValueError(
            f"Invalid drop_level '{drop_level}'; must be one of ['drop_candidate', 'drop_suspect', 'drop_flagged']"
        )

    if len(outliers) > 0:
        logger.warning(
            f"Dropping {len(outliers)} samples from {split_name} split "
            f"with outlier categories <= '{drop_level}'"
        )
        df = df[~df["sample_id"].isin(outliers)].copy()
    else:
        logger.info(
            f"No samples dropped from {split_name} split for outlier level '{drop_level}'."
        )
    return df


def _series_from_column(
    df: pd.DataFrame, column: Optional[str], default: float = 1.0
) -> pd.Series:
    if column and column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _combine_quality_outlier(
    quality: pd.Series,
    outlier: pd.Series,
    quality_weight: float,
    outlier_weight: float,
) -> pd.Series:
    # Auto-detect 0–100 scale and rescale to 0–1
    if quality.max() > 1.0:
        quality = quality / 100.0
    if outlier.max() > 1.0:
        outlier = outlier / 100.0
    quality_clipped = quality.clip(0.0, 1.0)
    outlier_clipped = outlier.clip(0.0, 1.0)
    denom = float(quality_weight + outlier_weight)
    if denom <= 0.0:
        return outlier_clipped
    combined = (
        quality_clipped * quality_weight + outlier_clipped * outlier_weight
    ) / denom
    return combined.clip(0.0, 1.0)


def _attach_sample_weights(
    df: pd.DataFrame,
    split_name: str,
    quality_score_col: str,
    outlier_score_col: str,
    quality_weight: float,
    outlier_weight: float,
    output_col: str,
) -> pd.DataFrame:
    quality_score = _series_from_column(df, quality_score_col, default=1.0)
    outlier_score = _series_from_column(df, outlier_score_col, default=1.0)
    base_weight = _combine_quality_outlier(
        quality_score, outlier_score, quality_weight, outlier_weight
    )
    combined = base_weight.astype(float)
    updated = df.copy()
    updated[output_col] = combined

    stats = {
        "min": float(combined.min()),
        "max": float(combined.max()),
        "mean": float(combined.mean()),
    }
    logger.info(f"{split_name} {output_col} stats: {stats}")
    return updated


def _build_head_manifest(
    *,
    experiment_name: str,
    timestamp: str,
    timestep_freq: Literal["month", "dekad"],
    time_explicit: bool,
    landcover_key: str,
    landcover_classes: Sequence[str],
    croptype_key: str,
    croptype_classes: Sequence[str],
    cropland_class_names: Sequence[str],
    seasonal_head_dropout: float,
    config_filename: str,
    manifest_filename: str,
) -> Dict[str, Any]:
    """Describe the seasonal heads so downstream tooling can replace them safely."""

    landcover_labels = [str(cls) for cls in landcover_classes]
    croptype_labels = [str(cls) for cls in croptype_classes]
    cropland_labels = [str(cls) for cls in cropland_class_names]

    checkpoint_name = f"{experiment_name}.pt"
    encoder_checkpoint_name = f"{experiment_name}_encoder.pt"

    landcover_head = {
        "name": "landcover",
        "task": "landcover",
        "task_type": "multiclass",
        "num_classes": len(landcover_labels),
        "classes_key": landcover_key,
        "class_names": landcover_labels,
        "label_column": "landcover_label",
        "logits_attr": "global_logits",
        "state_dict_prefix": "head.landcover_head",
        "replacement_contract": {
            "input_tensor": "global_embedding",
            "output_attr": "global_logits",
            "expects_time_dimension": False,
        },
    }

    croptype_head = {
        "name": "croptype",
        "task": "croptype",
        "task_type": "multiclass",
        "num_classes": len(croptype_labels),
        "classes_key": croptype_key,
        "class_names": croptype_labels,
        "label_column": "croptype_label",
        "logits_attr": "season_logits",
        "state_dict_prefix": "head.crop_head",
        "replacement_contract": {
            "input_tensor": "season_embeddings",
            "output_attr": "season_logits",
            "expects_season_masks": True,
        },
        "gating": {
            "enabled": bool(cropland_labels),
            "cropland_classes": cropland_labels,
        },
    }

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": timestamp,
        "experiment": {
            "name": experiment_name,
            "timestep_freq": timestep_freq,
            "time_explicit": time_explicit,
        },
        "backbone": {
            "head_dropout": seasonal_head_dropout,
        },
        "heads": [landcover_head, croptype_head],
        "artifacts": {
            "config": config_filename,
            "head_manifest": manifest_filename,
            "checkpoints": {
                "full": checkpoint_name,
                "encoder_only": encoder_checkpoint_name,
            },
            "packages": {
                "full": checkpoint_name.replace(".pt", ".zip"),
            },
        },
    }

    return manifest


def _package_model_checkpoints(
    full_checkpoint_path: Path,
    encoder_checkpoint_path: Path,
    *,
    manifest_path: Path,
    run_config_path: Path,
) -> None:
    """Bundle full and encoder-only checkpoints with manifest/config into a single zip."""

    # Verify both checkpoints exist
    if not full_checkpoint_path.exists():
        logger.warning(
            f"Full checkpoint {full_checkpoint_path} not found; skipping packaging"
        )
        return
    if not encoder_checkpoint_path.exists():
        logger.warning(
            f"Encoder checkpoint {encoder_checkpoint_path} not found; skipping packaging"
        )
        return
    if not manifest_path.exists():
        logger.warning(
            f"Head manifest missing at {manifest_path}; cannot package checkpoints"
        )
        return

    zip_path = full_checkpoint_path.with_suffix(".zip")

    try:
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(full_checkpoint_path, arcname=full_checkpoint_path.name)
            zf.write(encoder_checkpoint_path, arcname=encoder_checkpoint_path.name)
            zf.write(manifest_path, arcname="config.json")
            if run_config_path.exists():
                zf.write(run_config_path, arcname=run_config_path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to package checkpoints: {exc}")
        return

    logger.info(
        f"Packaged {full_checkpoint_path.name} and {encoder_checkpoint_path.name} into {zip_path}"
    )


def _annotate_dual_task_labels(
    df: pd.DataFrame,
    *,
    split_name: str,
    landcover_key: str,
    croptype_key: str,
    cropland_class_names: List[str],
) -> pd.DataFrame:
    """Attach landcover/croptype labels and task routing metadata to a split dataframe."""

    if df.empty:
        logger.warning(f"{split_name} split is empty before seasonal label annotation")
        return df

    updated = df.copy()
    landcover_map = {
        int(code): label for code, label in CLASS_MAPPINGS[landcover_key].items()
    }
    updated["landcover_label"] = updated["ewoc_code"].map(landcover_map)

    # explicitly remove ignore class
    missing_landcover = updated["landcover_label"].isna()
    ignore_landcover = updated["landcover_label"] == "ignore"

    if missing_landcover.any() | ignore_landcover.any():
        logger.warning(
            f"Removing {int(missing_landcover.sum())} samples from {split_name} split without '{landcover_key}' landcover mapping and {int(ignore_landcover.sum())} samples with 'ignore' landcover label"
        )
        updated = updated.loc[~missing_landcover & ~ignore_landcover].copy()
        if updated.empty:
            raise ValueError(
                f"No samples remain in {split_name} split after applying landcover mapping {landcover_key}."
            )

    croptype_map = {
        int(code): label for code, label in CLASS_MAPPINGS[croptype_key].items()
    }
    updated["croptype_label"] = updated["ewoc_code"].map(croptype_map)

    has_croptype_label = (updated["croptype_label"].notna()) & (
        updated["croptype_label"] != "ignore"
    )
    updated["label_task"] = np.where(
        has_croptype_label,
        "croptype",
        "landcover",
    )

    return updated


def _validate_seasonal_task_labels(
    df: pd.DataFrame, *, split_name: str, strict: bool = True
) -> None:
    """Ensure each seasonal task has at least two unique labels when present."""

    if df.empty:
        logger.warning(
            f"{split_name} split is empty; skipping seasonal task diversity validation",
        )
        return

    landcover_values = sorted(
        {str(label) for label in df["landcover_label"].dropna().unique()}
    )
    croptype_values = sorted(
        {
            str(label)
            for label in df.loc[df["label_task"] == "croptype", "croptype_label"]
            .dropna()
            .unique()
        }
    )

    def _handle_violation(task_name: str, values: List[str]) -> None:
        msg = (
            f"{split_name} split has {len(values)} unique {task_name} label(s): {values or ['<none>']}. "
            "Seasonal dual-task training requires at least two distinct labels per task."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if len(landcover_values) < 2:
        _handle_violation("landcover", landcover_values)
    if len(croptype_values) < 2:
        _handle_violation("croptype", croptype_values)


def _filter_temporally_invalid_rows(
    df: pd.DataFrame,
    *,
    split_name: str,
    num_timesteps: int,
    augment: bool,
    artifact_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Drop rows that cannot produce valid timestep windows in dataset sampling.

    This pre-check mirrors the center-point constraints in
    `WorldCerealDataset._get_center_point()` to avoid runtime DataLoader crashes
    such as `ValueError: low >= high`.
    """

    required_cols = {"available_timesteps", "valid_position"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        logger.warning(
            f"{split_name}: cannot run temporal pre-check, missing columns: {sorted(missing_cols)}"
        )
        return df

    if df.empty:
        return df

    checked = df.copy()
    checked["_available_timesteps"] = pd.to_numeric(
        checked["available_timesteps"], errors="coerce"
    )
    checked["_valid_position"] = pd.to_numeric(
        checked["valid_position"], errors="coerce"
    )

    available = checked["_available_timesteps"]
    valid_pos = checked["_valid_position"]
    half = num_timesteps // 2

    base_valid = (
        available.notna()
        & valid_pos.notna()
        & (available >= num_timesteps)
        & (valid_pos >= 0)
        & (valid_pos < available)
    )

    if augment:
        # Mirrors _get_center_point for non-ssl branch where np.random.randint is used.
        min_center = np.maximum(half, valid_pos + 1 - half)
        max_center = np.minimum(available - half, valid_pos - 1 + half)
        jitter_valid = (available == num_timesteps) | (min_center <= max_center)
        valid_mask = base_valid & jitter_valid
    else:
        valid_mask = base_valid

    invalid_mask = ~valid_mask
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return df

    rejected = checked.loc[
        invalid_mask,
        [
            col
            for col in [
                "sample_id",
                "ref_id",
                "start_date",
                "end_date",
                "valid_time",
                "available_timesteps",
                "valid_position",
            ]
            if col in checked.columns
        ],
    ].copy()

    logger.warning(
        f"{split_name}: dropping {invalid_count} temporally invalid sample(s) before dataset creation."
    )
    preview_cols = [
        col
        for col in ["sample_id", "available_timesteps", "valid_position"]
        if col in rejected.columns
    ]
    if preview_cols:
        logger.warning(
            f"{split_name}: first invalid samples:\n{rejected[preview_cols].head(10).to_string(index=False)}"
        )

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        reject_path = artifact_dir / f"invalid_temporal_samples_{split_name}.csv"
        rejected.to_csv(reject_path, index=False)
        logger.warning(
            f"{split_name}: wrote rejected temporal samples to {reject_path}"
        )

    filtered = checked.loc[valid_mask].drop(
        columns=["_available_timesteps", "_valid_position"], errors="ignore"
    )
    if filtered.empty:
        raise ValueError(
            f"{split_name}: no samples remain after temporal pre-check filtering."
        )
    return filtered


def get_parquet_file_list(timestep_freq: Literal["month", "dekad"] = "month"):
    if timestep_freq == "month":
        parquet_files = list(
            Path(
                "/projects/TAP/worldcereal/data/worldcereal_all_extractions_with_anomalies.parquet"
                # "/projects/TAP/worldcereal/data/worldcereal_all_extractions.parquet"
            ).rglob("*.parquet")
        )
    elif timestep_freq == "dekad":
        raise NotImplementedError(
            "Dekad parquet files are not yet implemented. Please use 'month' timestep frequency."
        )
    else:
        raise ValueError(
            f"timestep_freq {timestep_freq} is not supported. Supported values are 'month' and 'dekad'."
        )

    return parquet_files


def main(args):
    """Main function to run the finetuning process."""
    # ------------------------------------------
    # Parameter settings (can become argparser)
    # ------------------------------------------

    experiment_tag = args.experiment_tag
    base_output_dir = args.base_output_dir
    timestep_freq = args.timestep_freq  # "month" or "dekad"
    max_timesteps_trim = args.max_timesteps_trim  # "auto", int or tuple of string dates
    use_valid_time = args.use_valid_time

    # Path to the training data
    if args.parquet_files:
        parquet_files = args.parquet_files
    else:
        parquet_files = get_parquet_file_list(timestep_freq)
    val_samples_file = args.val_samples_file  # If None, random split is used
    test_samples_file = args.test_samples_file  # If None, random split is used
    ignore_samples_file = args.ignore_samples_file  # If None, no samples are ignored

    # Most popular maps: LANDCOVER10, CROPTYPE9, CROPTYPE0, CROPLAND2
    initial_mapping = args.initial_mapping
    augment = args.augment
    time_explicit = args.time_explicit
    enable_masking = args.enable_masking
    debug = args.debug
    use_class_balancing = (
        args.use_class_balancing
    )  # If True, weight samples by class frequency
    cropland_class_names = _filter_ignore_labels(
        [
            cls.strip()
            for cls in args.landcover_cropland_classes.split(",")
            if cls.strip()
        ]
    )
    if not cropland_class_names:
        cropland_class_names = ["temporary_crops"]

    # ± timesteps to jitter true label pos, for time_explicit only; will only be set for training
    label_jitter = args.label_jitter

    # ± timesteps to expand around label pos (true or moved), for time_explicit only; will only be set for training
    label_window = args.label_window

    # Minimum fraction of season slots required inside the training window for season supervision.
    # Val/test always use 1.0 (full coverage). With augmentation the window shifts randomly
    # so a lower threshold prevents spurious loss of croptype supervision signal.
    train_min_season_coverage: float = args.train_min_season_coverage

    # Season IDs for crop-type supervision (defaults to GLOBAL_SEASON_IDS)
    season_ids: Optional[Tuple[str, ...]] = (
        tuple(args.season_ids) if args.season_ids else None
    )
    logger.info(
        f"Season IDs: {season_ids or GLOBAL_SEASON_IDS}"
        + (" (default)" if season_ids is None else " (custom)")
    )

    # Presto freezing settings
    freeze_layers = None
    unfreeze_epoch = None
    if args.head_only_training > 0:
        freeze_layers = ["encoder"]
        unfreeze_epoch = args.head_only_training
        logger.info(
            f"Training head only for the first {args.head_only_training} epoch(s)"
        )

    # Masking parameters
    masking_config = SensorMaskingConfig(
        enable=enable_masking,
        s1_full_dropout_prob=0.05,
        s1_timestep_dropout_prob=0.05,
        s2_cloud_timestep_prob=0.1,
        s2_cloud_block_prob=0.05,
        s2_cloud_block_min=2,
        s2_cloud_block_max=3 if timestep_freq == "month" else 9,
        meteo_timestep_dropout_prob=0.05,
        dem_dropout_prob=0.01,
        seed=DEFAULT_SEED,
    )

    # Experiment signature
    timestamp_ind = datetime.now().strftime("%Y%m%d%H%M")

    # Update experiment name to include masking info
    if masking_config.enable:
        masking_info = "enabled"
    else:
        masking_info = "disabled"

    experiment_name = f"presto-prometheo-{experiment_tag}-{timestep_freq}-augment={augment}-balance={use_class_balancing}-timeexplicit={time_explicit}-masking={masking_info}-run={timestamp_ind}"
    output_dir = f"{base_output_dir}/{experiment_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tensorboard_dir: Optional[Path] = None
    if args.log_tensorboard:
        tensorboard_dir = Path(output_dir) / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorBoard logs will be stored at {tensorboard_dir}")

    def _export_eval_artifacts(
        eval_results: pd.DataFrame,
        cm: Figure,
        cm_norm: Figure,
        class_names: Optional[List[str]],
        *,
        suffix: Optional[str] = None,
        artifact_dir: Path,
    ) -> None:
        """Persist confusion matrices and metrics CSV for a given evaluation split.

        Parameters
        ----------
        eval_results : pd.DataFrame
            Tabular metrics as returned by ``evaluate_finetuned_model``.
        cm : matplotlib.figure.Figure
            Confusion-matrix plot using raw counts.
        cm_norm : matplotlib.figure.Figure
            Normalized confusion-matrix plot (per-class percentages).
        class_names : Optional[List[str]]
            Explicit label ordering to use for filenames/logging; falls back
            to the classes present in ``eval_results``.
        suffix : Optional[str], default None
            Optional suffix injected in the filenames so train/val/test
            exports do not overwrite each other.
        artifact_dir : Path
            Destination folder where artifacts are persisted.
        """
        label_set = class_names or [
            cls
            for cls in eval_results["class"].tolist()
            if cls not in {"accuracy", "macro avg", "weighted avg"}
        ]
        if not label_set:
            label_set = ["class"]
        logger.debug("Saving confusion matrices with %d label entries", len(label_set))

        suffix_part = f"_{suffix}" if suffix else ""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        cm_path = artifact_dir / f"CM_{experiment_name}{suffix_part}.png"
        cm_norm_path = artifact_dir / f"CM_{experiment_name}{suffix_part}_norm.png"

        cm.savefig(cm_path, bbox_inches="tight")
        plt.close(cm)
        cm_norm.savefig(cm_norm_path, bbox_inches="tight")
        plt.close(cm_norm)

        eval_results.to_csv(
            artifact_dir / f"results_{experiment_name}{suffix_part}.csv",
            index=False,
        )

    def _format_macro_summary(results_df: pd.DataFrame) -> str:
        macro_row = results_df[results_df["class"] == "macro avg"]
        if macro_row.empty:
            return "macro metrics unavailable"
        macro = macro_row.iloc[0]
        return (
            f"macro F1={macro['f1-score']:.3f}, precision={macro['precision']:.3f}, "
            f"recall={macro['recall']:.3f}"
        )

    # setup path for processed wide parquet file so that it can be reused across experiments
    if args.explicit_training_dataframe:
        wide_parquet_output_path = Path(args.explicit_training_dataframe)
    elif not debug:
        # wide_parquet_output_path = Path(
        #     "/projects/worldcereal/data/cached_wide_merged/merged_305_wide.parquet"
        # )
        wide_parquet_output_path = None
    else:
        wide_parquet_output_path = None

    # Training parameters
    pretrained_model_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
    epochs = 100
    batch_size = args.batch_size
    patience = 20
    num_workers = 8

    # ------------------------------------------

    # Setup logging
    initialize_logging(
        log_file=Path(output_dir) / "logs" / f"{experiment_name}.log",
        level="INFO",
        console_filter_keyword="PROGRESS",
    )

    # Get the paths to train/val/test dataframe parquet files
    train_df_path = Path(output_dir) / "train_df.parquet"
    val_df_path = Path(output_dir) / "val_df.parquet"
    test_df_path = Path(output_dir) / "test_df.parquet"

    # Get / load the train/val/test dataframes
    if (
        train_df_path.exists()
        and val_df_path.exists()
        and test_df_path.exists()
        and not debug
    ):
        logger.info("Loading existing train/val/test DataFrames from parquet files.")
        train_df = pd.read_parquet(train_df_path)
        val_df = pd.read_parquet(val_df_path)
        test_df = pd.read_parquet(test_df_path)
    else:
        logger.info("Generating train/val/test DataFrames from source parquet files.")
        train_df, val_df, test_df = get_training_dfs_from_parquet(
            parquet_files,
            wide_parquet_output_path=wide_parquet_output_path,
            timestep_freq=timestep_freq,
            max_timesteps_trim=max_timesteps_trim,
            use_valid_time=use_valid_time,
            finetune_classes=initial_mapping,
            class_mappings=CLASS_MAPPINGS,
            val_samples_file=val_samples_file,
            test_samples_file=test_samples_file,
            ignore_samples_file=ignore_samples_file,
            region_filter=args.finetune_regions,
            debug=debug,
            overwrite=False,
        )
        logger.info("Saving train, val, and test DataFrames to parquet files ...")
        # train_df.to_parquet(train_df_path)
        # val_df.to_parquet(val_df_path)
        # test_df.to_parquet(test_df_path)

    if "drop" in args.outlier_mode:
        train_df = _drop_outliers(
            train_df,
            split_name="train",
            drop_level=args.outlier_mode,
        )
        val_df = _drop_outliers(
            val_df,
            split_name="val",
            drop_level=args.outlier_mode,
        )
        test_df = _drop_outliers(
            test_df,
            split_name="test",
            drop_level=args.outlier_mode,
        )

    landcover_key = args.landcover_classes_key
    croptype_key = args.croptype_classes_key
    if landcover_key not in CLASS_MAPPINGS:
        raise ValueError(
            f"Unknown landcover_classes_key '{landcover_key}'. Available keys: {list(CLASS_MAPPINGS)}"
        )
    if croptype_key not in CLASS_MAPPINGS:
        raise ValueError(
            f"Unknown croptype_classes_key '{croptype_key}'. Available keys: {list(CLASS_MAPPINGS)}"
        )

    annotated_splits = {}
    for split_name, df in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        annotated_splits[split_name] = _annotate_dual_task_labels(
            df,
            split_name=split_name,
            landcover_key=landcover_key,
            croptype_key=croptype_key,
            cropland_class_names=cropland_class_names,
        )

    train_df = annotated_splits["train"]
    val_df = annotated_splits["val"]
    test_df = annotated_splits["test"]

    num_timesteps = 12 if timestep_freq == "month" else 36
    temporal_rejects_dir = Path(output_dir) / "logs"
    train_df = _filter_temporally_invalid_rows(
        train_df,
        split_name="train",
        num_timesteps=num_timesteps,
        augment=augment,
        artifact_dir=temporal_rejects_dir,
    )
    val_df = _filter_temporally_invalid_rows(
        val_df,
        split_name="val",
        num_timesteps=num_timesteps,
        augment=False,
        artifact_dir=temporal_rejects_dir,
    )
    test_df = _filter_temporally_invalid_rows(
        test_df,
        split_name="test",
        num_timesteps=num_timesteps,
        augment=False,
        artifact_dir=temporal_rejects_dir,
    )

    # if debug:
    #     train_df = train_df.sample(n=100000, random_state=DEFAULT_SEED).reset_index(
    #         drop=True
    #     )
    #     val_df = val_df.sample(n=20000, random_state=DEFAULT_SEED).reset_index(
    #         drop=True
    #     )
    #     test_df = test_df.sample(n=20000, random_state=DEFAULT_SEED).reset_index(
    #         drop=True
    #     )
    #     logger.info("Debug mode: reduced dataset sizes.")

    _validate_seasonal_task_labels(train_df, split_name="train", strict=True)
    _validate_seasonal_task_labels(val_df, split_name="val", strict=False)
    _validate_seasonal_task_labels(test_df, split_name="test", strict=False)

    logger.info(f"Number of training samples: {len(train_df)}")
    logger.info(f"Number of validation samples: {len(val_df)}")
    logger.info(f"Number of test samples: {len(test_df)}")

    quality_weight = args.sample_weight_quality
    outlier_weight = args.sample_weight_outlier

    # Remove small classes from train_df; make sure to keep the same classes in val/test for consistency
    train_df, removed_lc_classes = remove_small_classes(
        train_df,
        min_samples=args.min_samples_per_class,
        class_column="landcover_label",
    )
    train_df, removed_ct_classes = remove_small_classes(
        train_df,
        min_samples=args.min_samples_per_class,
        class_column="croptype_label",
    )
    if len(removed_lc_classes) > 0:
        logger.warning(
            f"Removing {val_df['landcover_label'].isin(removed_lc_classes).sum()} validation and {test_df['landcover_label'].isin(removed_lc_classes).sum()} test samples with landcover classes {removed_lc_classes} removed from training split"
        )
    if len(removed_ct_classes) > 0:
        logger.warning(
            f"Removing {val_df['croptype_label'].isin(removed_ct_classes).sum()} validation and {test_df['croptype_label'].isin(removed_ct_classes).sum()} test samples with croptype classes {removed_ct_classes} removed from training split"
        )
    val_df = val_df[~val_df["landcover_label"].isin(removed_lc_classes)].copy()
    test_df = test_df[~test_df["landcover_label"].isin(removed_lc_classes)].copy()
    val_df = val_df[~val_df["croptype_label"].isin(removed_ct_classes)].copy()
    test_df = test_df[~test_df["croptype_label"].isin(removed_ct_classes)].copy()

    train_df = _attach_sample_weights(
        train_df,
        split_name="train",
        quality_score_col="quality_score_lc",
        outlier_score_col="LC10_confidence_nonoutlier",
        quality_weight=quality_weight,
        outlier_weight=outlier_weight,
        output_col="sample_weight_lc",
    )
    val_df = _attach_sample_weights(
        val_df,
        split_name="val",
        quality_score_col="quality_score_lc",
        outlier_score_col="LC10_confidence_nonoutlier",
        quality_weight=quality_weight,
        outlier_weight=outlier_weight,
        output_col="sample_weight_lc",
    )
    test_df = _attach_sample_weights(
        test_df,
        split_name="test",
        quality_score_col="quality_score_lc",
        outlier_score_col="LC10_confidence_nonoutlier",
        quality_weight=quality_weight,
        outlier_weight=outlier_weight,
        output_col="sample_weight_lc",
    )

    train_df = _attach_sample_weights(
        train_df,
        split_name="train",
        quality_score_col="quality_score_ct",
        outlier_score_col="CT25_confidence_nonoutlier",
        quality_weight=quality_weight,
        outlier_weight=outlier_weight,
        output_col="sample_weight_ct",
    )
    val_df = _attach_sample_weights(
        val_df,
        split_name="val",
        quality_score_col="quality_score_ct",
        outlier_score_col="CT25_confidence_nonoutlier",
        quality_weight=quality_weight,
        outlier_weight=outlier_weight,
        output_col="sample_weight_ct",
    )
    test_df = _attach_sample_weights(
        test_df,
        split_name="test",
        quality_score_col="quality_score_ct",
        outlier_score_col="CT25_confidence_nonoutlier",
        quality_weight=quality_weight,
        outlier_weight=outlier_weight,
        output_col="sample_weight_ct",
    )

    # Write the processed dataframes after all manipulations have already been done
    train_df.to_parquet(train_df_path)
    val_df.to_parquet(val_df_path)
    test_df.to_parquet(test_df_path)

    # Use type casting to specify to mypy that task_type is a valid Literal value
    task_type_literal: Literal["binary", "multiclass"] = "multiclass"  # type: ignore

    # Construct training and validation datasets with masking parameters
    train_ds, val_ds, test_ds = prepare_training_datasets(
        train_df,
        val_df,
        test_df,
        num_timesteps=num_timesteps,
        timestep_freq=timestep_freq,
        augment=augment,
        time_explicit=time_explicit,
        emit_label_tensor=False,  # Supervise through attributes only
        task_type=task_type_literal,
        masking_config=masking_config,
        label_jitter=label_jitter,
        label_window=label_window,
        train_min_season_coverage=train_min_season_coverage,
        season_ids=season_ids,
    )

    # Construct the finetuning model based on the pretrained model
    # Start from the full taxonomy ordering, then restrict to classes
    # actually present in training data.  This guarantees:
    #   1. Head output dim == number of effective classes (no wasted logits).
    #   2. Loss indices stay consistent with head outputs.
    #   3. Confusion matrices only show classes the model can predict.
    landcover_classes_raw = _unique_preserve_order(
        CLASS_MAPPINGS[landcover_key].values()
    )
    landcover_classes_full = _filter_ignore_labels(landcover_classes_raw)

    croptype_classes_raw = _unique_preserve_order(CLASS_MAPPINGS[croptype_key].values())
    croptype_classes_full = _filter_ignore_labels(croptype_classes_raw)

    # Effective classes = those present in training split (preserving taxonomy order)
    train_lc_labels = set(
        str(label)
        for label in train_df["landcover_label"].dropna().unique()
        if not _is_ignore_label(label)
    )
    train_ct_labels = set(
        str(label)
        for label in train_df.loc[
            train_df["label_task"] == "croptype", "croptype_label"
        ]
        .dropna()
        .unique()
        if not _is_ignore_label(label)
    )
    landcover_classes = [
        cls for cls in landcover_classes_full if cls in train_lc_labels
    ]
    croptype_classes = [
        cls for cls in croptype_classes_full if cls in train_ct_labels
    ]

    if len(landcover_classes) < len(landcover_classes_full):
        dropped_lc = set(landcover_classes_full) - set(landcover_classes)
        logger.info(
            f"Narrowed landcover head from {len(landcover_classes_full)} to "
            f"{len(landcover_classes)} classes (dropped {dropped_lc})"
        )
    if len(croptype_classes) < len(croptype_classes_full):
        dropped_ct = set(croptype_classes_full) - set(croptype_classes)
        logger.info(
            f"Narrowed croptype head from {len(croptype_classes_full)} to "
            f"{len(croptype_classes)} classes (dropped {dropped_ct})"
        )

    if not landcover_classes or not croptype_classes:
        raise ValueError(
            "Both landcover and croptype class mappings must contain at least one class "
            "after restricting to training data."
        )

    # Keep cropland gate names consistent with the effective landcover head
    cropland_class_names = [
        name for name in cropland_class_names if name in landcover_classes
    ]
    if not cropland_class_names:
        logger.warning(
            "No cropland gate class names remain after narrowing to effective "
            "landcover classes; cropland gating will be disabled."
        )

    backbone = Presto(pretrained_model_path=pretrained_model_path)
    seasonal_head = SeasonalFinetuningHead(
        embedding_dim=backbone.encoder.embedding_size,
        landcover_num_outputs=len(landcover_classes),
        crop_num_outputs=len(croptype_classes),
        dropout=args.seasonal_head_dropout,
    )
    model = WorldCerealSeasonalModel(
        backbone=backbone,
        head=seasonal_head,
    ).to(device)
    sample_weight_mapping = {
        "landcover": "sample_weight_lc",
        "croptype": "sample_weight_ct",
    }

    loss_fn = SeasonalMultiTaskLoss(
        landcover_classes=landcover_classes,
        croptype_classes=croptype_classes,
        ignore_index=NODATAVALUE,
        landcover_weight=args.seasonal_loss_landcover_weight,
        croptype_weight=args.seasonal_loss_croptype_weight,
        task_sample_weight_attrs=sample_weight_mapping,
        cropland_class_names=cropland_class_names,
    )
    logger.info(
        "Using seasonal head with landcover key {} ({} classes) and croptype key {} ({} classes)",
        landcover_key,
        len(landcover_classes),
        croptype_key,
        len(croptype_classes),
    )
    logger.info("Using loss function: {}", loss_fn)

    def _evaluate_and_export(
        model_to_eval: torch.nn.Module,
        *,
        suffix_prefix: Optional[str],
        artifact_dir: Path,
    ) -> None:
        split_label = suffix_prefix or "FINAL"
        seasonal_results = evaluate_finetuned_model(
            model_to_eval,
            test_ds,
            num_workers,
            batch_size,
            time_explicit=time_explicit,
            seasonal_landcover_classes=landcover_classes,
            seasonal_croptype_classes=croptype_classes,
            cropland_class_names=cropland_class_names,
        )

        if not isinstance(seasonal_results, dict):
            raise TypeError(
                "Seasonal evaluation is expected to return a dictionary of task artifacts"
            )

        seasonal_results = cast(Dict[str, Dict[str, Any]], seasonal_results)

        for task_name, artifacts in seasonal_results.items():
            suffix = "_".join(part for part in (suffix_prefix, task_name) if part)
            results_df = artifacts["results"]
            _export_eval_artifacts(
                results_df,
                artifacts["cm"],
                artifacts["cm_norm"],
                artifacts.get("classes"),
                suffix=suffix,
                artifact_dir=artifact_dir,
            )

            summary = _format_macro_summary(results_df)
            logger.info(
                f"{task_name.capitalize()} summary ({split_label}) on test set: {summary}",
            )

            if task_name == "croptype" and artifacts.get("gate_rejections"):
                logger.info(
                    f"Croptype gate rejections ({split_label}): {artifacts['gate_rejections']}",
                )

    # Set the parameters
    hyperparams = Hyperparams(
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        num_workers=num_workers,
    )

    # ----------------------------
    # Definining the optimizer and scheduler
    # ----------------------------

    head_lr = args.head_learning_rate
    full_lr = args.full_learning_rate
    decay_gamma = args.lr_gamma
    warmup_epochs = max(0, args.post_unfreeze_warmup_epochs)
    warmup_start_factor = max(0.0, min(args.post_unfreeze_warmup_start_factor, 1.0))
    if head_lr <= 0 or full_lr <= 0:
        raise ValueError("Learning rates must be positive numbers.")
    drop_factor = full_lr / head_lr
    if drop_factor <= 0:
        raise ValueError("full_learning_rate must be greater than 0.")
    parameters = param_groups_lrd(model)
    optimizer = AdamW(parameters, lr=head_lr)
    for group in optimizer.param_groups:
        group["initial_lr"] = head_lr

    stage_defs = []

    head_only_epochs = unfreeze_epoch or 0
    if head_only_epochs > 0:
        logger.info(
            f"Using constant LR {head_lr:.2e} for the first {head_only_epochs} epoch(s) while the encoder is frozen",
        )
        head_sched = lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=head_only_epochs,
        )
        stage_defs.append((head_sched, head_only_epochs))

    # Stage to drop LR (and optionally warm up after unfreezing)
    if warmup_epochs > 0 and warmup_start_factor <= 0:
        logger.warning(
            "post_unfreeze_warmup_start_factor <= 0 supplied; defaulting to 0.1"
        )
        warmup_start_factor = 0.1
    if warmup_epochs == 0:
        warmup_start_factor = 1.0
    logger.info(
        f"LR schedule: head_lr={head_lr:.2e}, full_lr={full_lr:.2e}, head_only_epochs={unfreeze_epoch or 0}, warmup_epochs={warmup_epochs}, warmup_start_factor={warmup_start_factor:.2f}, gamma={decay_gamma:.3f}",
    )
    drop_stage_iters = warmup_epochs if warmup_epochs > 0 else 1
    start_factor = drop_factor * (warmup_start_factor if warmup_epochs > 0 else 1.0)
    drop_stage = lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=drop_factor,
        total_iters=drop_stage_iters,
    )
    stage_defs.append((drop_stage, drop_stage_iters))

    # Exponential decay once full finetuning is underway
    exp_stage = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: drop_factor * (decay_gamma**step),
    )
    stage_defs.append((exp_stage, None))

    scheduler_stages = [stage for stage, _ in stage_defs]
    milestones = []
    cumulative = 0
    for _, duration in stage_defs[:-1]:
        cumulative += duration or 0
        milestones.append(cumulative)
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=scheduler_stages,
        milestones=milestones,
    )

    # Setup dataloaders
    generator = torch.Generator()
    generator.manual_seed(DEFAULT_SEED)

    balancing_clip: Optional[Tuple[float, float]] = None
    if args.balancing_clip_max > args.balancing_clip_min:
        balancing_clip = (args.balancing_clip_min, args.balancing_clip_max)

    # DualHeadBatchSampler guarantees every batch has exactly 50 % LC-assigned
    # and 50 % CT-assigned samples.  Task-level 50/50 split is always enforced.
    # use_class_balancing controls whether sampling probabilities are weighted by
    # the class distribution (True) or uniform across the pool (False, method="none").
    # Spatial density down-weighting is always applied when spatial_bin_size_deg is
    # set, independently of class balancing.
    _effective_class_method = (
        args.class_balancing_method if use_class_balancing else "none"
    )
    train_batch_sampler = train_ds.get_dual_head_batch_sampler(
        batch_size=hyperparams.batch_size,
        class_weight_method=_effective_class_method,
        clip_range=balancing_clip,
        spatial_bin_size_degrees=args.spatial_bin_size_deg,
        spatial_weight_method=args.spatial_balancing_method,
        generator=generator,
    )

    train_dl = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=hyperparams.num_workers,
        collate_fn=collate_fn,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=hyperparams.num_workers,
        collate_fn=collate_fn,
    )

    intermediate_eval_dir = Path(output_dir) / "intermediate_evals"
    intermediate_eval_dir.mkdir(parents=True, exist_ok=True)

    def _on_validation_improved(
        epoch_idx: int, best_model_snapshot: torch.nn.Module, best_val_loss: float
    ) -> None:
        _evaluate_and_export(
            best_model_snapshot,
            suffix_prefix=f"epoch{epoch_idx:03d}",
            artifact_dir=intermediate_eval_dir,
        )

    config_path = Path(output_dir) / "run_config.json"
    manifest_path = Path(output_dir) / "head_manifest.json"
    head_manifest = _build_head_manifest(
        experiment_name=experiment_name,
        timestamp=timestamp_ind,
        timestep_freq=timestep_freq,
        time_explicit=time_explicit,
        landcover_key=landcover_key,
        landcover_classes=landcover_classes,
        croptype_key=croptype_key,
        croptype_classes=croptype_classes,
        cropland_class_names=cropland_class_names,
        seasonal_head_dropout=args.seasonal_head_dropout,
        config_filename=config_path.name,
        manifest_filename=manifest_path.name,
    )
    args_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    masking_payload = _object_to_dict(masking_config)
    hyperparams_payload = {
        "max_epochs": hyperparams.max_epochs,
        "batch_size": hyperparams.batch_size,
        "patience": hyperparams.patience,
        "num_workers": hyperparams.num_workers,
    }
    scheduler_payload = {
        "decay_gamma": decay_gamma,
        "warmup_epochs": warmup_epochs,
        "warmup_start_factor": warmup_start_factor,
        "drop_factor": drop_factor,
        "head_only_epochs": head_only_epochs,
    }
    optimizer_payload = {
        "head_learning_rate": head_lr,
        "full_learning_rate": full_lr,
    }
    balancing_payload = {
        "sampler": "DualHeadBatchSampler",
        "use_class_balancing": use_class_balancing,
        "class_weight_method": _effective_class_method,
        "clip_range": list(balancing_clip) if balancing_clip else None,
        "spatial_enabled": args.spatial_bin_size_deg is not None,
        "spatial_bin_size_deg": args.spatial_bin_size_deg,
        "spatial_balancing_method": args.spatial_balancing_method,
    }
    data_sources_payload = {
        "parquet_files": [str(path) for path in parquet_files],
        "wide_parquet_output_path": _path_to_str(wide_parquet_output_path),
        "val_samples_file": args.val_samples_file,
        "test_samples_file": args.test_samples_file,
        "ignore_samples_file": args.ignore_samples_file,
        "train_df_cache": _path_to_str(train_df_path),
        "val_df_cache": _path_to_str(val_df_path),
        "test_df_cache": _path_to_str(test_df_path),
    }
    dataset_payload = {
        "timestep_freq": timestep_freq,
        "num_timesteps": 12 if timestep_freq == "month" else 36,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "augment": augment,
        "time_explicit": time_explicit,
        "label_jitter": label_jitter,
        "label_window": label_window,
        "season_ids": list(season_ids) if season_ids else list(GLOBAL_SEASON_IDS),
        "masking": masking_payload,
    }
    classes_payload = {
        "landcover": {
            "key": landcover_key,
            "count": len(landcover_classes),
            "labels": landcover_classes,
        },
        "croptype": {
            "key": croptype_key,
            "count": len(croptype_classes),
            "labels": croptype_classes,
        },
    }
    model_payload = {
        "pretrained_model_path": pretrained_model_path,
        "seasonal_head_dropout": args.seasonal_head_dropout,
        "freeze_layers": freeze_layers,
        "unfreeze_epoch": unfreeze_epoch,
    }
    loss_payload = {
        "landcover_weight": args.seasonal_loss_landcover_weight,
        "croptype_weight": args.seasonal_loss_croptype_weight,
        "cropland_class_names": cropland_class_names,
        "sample_weight_strategy": args.sample_weight_strategy,
        "sample_weight_mapping": sample_weight_mapping,
    }
    run_config = {
        "timestamp": timestamp_ind,
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "tensorboard_dir": _path_to_str(tensorboard_dir),
        "args": args_payload,
        "seed": DEFAULT_SEED,
        "data_sources": data_sources_payload,
        "dataset": dataset_payload,
        "classes": classes_payload,
        "model": model_payload,
        "loss": loss_payload,
        "training": {
            "hyperparams": hyperparams_payload,
            "optimizer": optimizer_payload,
            "scheduler": scheduler_payload,
            "label_jitter": label_jitter,
            "label_window": label_window,
            "val_loss_ema_alpha": args.val_loss_ema_alpha,
        },
        "balancing": balancing_payload,
    }
    # Run the finetuning
    logger.info("Starting finetuning...")
    finetuned_model = run_finetuning(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        experiment_name=experiment_name,
        output_dir=output_dir,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparams=hyperparams,
        setup_logging=False,  # Already setup logging
        freeze_layers=freeze_layers,
        unfreeze_epoch=unfreeze_epoch,
        on_validation_improved=_on_validation_improved,
        tensorboard_logdir=tensorboard_dir,
        val_loss_ema_alpha=args.val_loss_ema_alpha,
    )

    seasonal_checkpoint_path = Path(output_dir) / f"{experiment_name}.pt"
    if not seasonal_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected seasonal checkpoint not found at {seasonal_checkpoint_path}."
        )

    # Compute fingerprint from encoder-only checkpoint (saved by run_finetuning)
    encoder_checkpoint_path = Path(output_dir) / f"{experiment_name}_encoder.pt"
    backbone_fingerprint = checkpoint_fingerprint(encoder_checkpoint_path)
    head_manifest["backbone"]["fingerprint"] = backbone_fingerprint
    run_config["head_manifest"] = head_manifest

    def _write_json(payload: Dict[str, Any], path: Path, label: str) -> None:
        try:
            with path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to save {label}: {exc}")
        else:
            logger.info(f"Saved {label} to {path}")

    _write_json(run_config, config_path, "run configuration")
    _write_json(head_manifest, manifest_path, "head manifest")

    _evaluate_and_export(
        finetuned_model,
        suffix_prefix=None,
        artifact_dir=Path(output_dir),
    )

    _package_model_checkpoints(
        Path(output_dir) / f"{experiment_name}.pt",
        Path(output_dir) / f"{experiment_name}_encoder.pt",
        manifest_path=manifest_path,
        run_config_path=config_path,
    )

    spatial_enabled = args.run_spatial_inference or bool(
        args.spatial_inference_patches_dir
    )
    if spatial_enabled:
        if not args.spatial_inference_patches_dir:
            logger.warning(
                "Spatial inference requested but no --spatial_inference_patches_dir provided; skipping."
            )
        else:
            from spatial_inference import run_spatial_inference

            continents_raw = args.spatial_inference_continents or "all"
            continents: Union[str, List[str]] = continents_raw
            if isinstance(continents_raw, str):
                tokens = [
                    token.strip()
                    for token in continents_raw.split(",")
                    if token.strip()
                ]
                if tokens and not any(token.lower() == "all" for token in tokens):
                    continents = tokens

            logger.info("Starting post-finetuning spatial inference...")
            run_spatial_inference(
                model_dir=Path(output_dir),
                patches_dir=Path(args.spatial_inference_patches_dir),
                continents=continents,
                output_dir=Path(output_dir) / "inference_patches",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

    logger.info("Finetuning completed!")


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Train in-season crop type model")

    def auto_or_int(value):
        if value == "auto":
            return value
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "max_timesteps_trim must be 'auto' or an integer."
            )

    # General setup
    parser.add_argument("--experiment_tag", type=str, default="")
    parser.add_argument(
        "--timestep_freq", type=str, choices=["month", "dekad"], default="month"
    )
    parser.add_argument(
        "--max_timesteps_trim",
        type=auto_or_int,
        default="auto",
        help='Maximum number of timesteps to retain after trimming. Can be "auto" or an integer.',
    )
    parser.add_argument(
        "--use_valid_time",
        type=bool,
        default=True,
        help="Whether to use the 'valid_time' column for processing timesteps.",
    )
    parser.add_argument(
        "--landcover_classes_key",
        type=str,
        default="LANDCOVER10",
        help="Class mapping key used for landcover targets in the dual-head configuration.",
    )
    parser.add_argument(
        "--croptype_classes_key",
        type=str,
        default="CROPTYPE2",
        help="Class mapping key used for crop-type targets in the dual-head configuration.",
    )
    parser.add_argument(
        "--seasonal_head_dropout",
        type=float,
        default=0.0,
        help="Dropout rate applied inside the seasonal head before projection layers.",
    )
    parser.add_argument(
        "--seasonal_loss_landcover_weight",
        type=float,
        default=1.0,
        help="Relative weight for the landcover branch when using the seasonal loss.",
    )
    parser.add_argument(
        "--seasonal_loss_croptype_weight",
        type=float,
        default=1.0,
        help="Relative weight for the crop-type branch when using the seasonal loss.",
    )
    parser.add_argument(
        "--landcover_cropland_classes",
        type=str,
        default="temporary_crops",
        help="Comma-separated list of landcover class names treated as cropland during gating.",
    )

    # Data paths
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default=None,
        help="Base directory for output files. If not set, a default location will be used.",
    )
    parser.add_argument(
        "--val_samples_file",
        type=str,
        default=None,
        help="Path to a CSV with val sample IDs. If not set, a random split will be used.",
    )
    parser.add_argument(
        "--ignore_samples_file",
        type=str,
        default=None,
        help="Path to a CSV with ignore sample IDs. If not set, a random split will be used.",
    )
    parser.add_argument(
        "--test_samples_file",
        type=str,
        default=None,
        help="Path to a CSV with test sample IDs. If not set, a random split will be used.",
    )
    parser.add_argument(
        "--parquet_files",
        type=str,
        nargs="*",
        default=None,
        help="Explicit list of parquet files to use for training. If not set, uses get_parquet_file_list based on timestep_freq.",
    )
    parser.add_argument(
        "--explicit_training_dataframe",
        type=str,
        default=None,
        help="Path to cache the merged wide parquet file for reuse across experiments. If not set, uses default location in non-debug mode.",
    )
    parser.add_argument(
        "--finetune_regions",
        type=str,
        default=None,
        help=(
            "Comma-separated list of region names to keep for finetuning "
            "Use 'all' or leave unset to keep all samples."
            "Possible regions: Micronesia, Eastern Asia, Western Europe, Southern Europe,"
            "South America, Central America, Caribbean, Northern Africa,"
            "Western Africa, Northern Europe, Central Asia,"
            "Middle Africa, Western Asia, Eastern Europe,"
            "Eastern Africa, South-Eastern Asia, Polynesia,"
            "Northern America, Melanesia, None, Southern Asia,"
            "Australia and New Zealand, Southern Africa"
        ),
    )

    # Task setup
    parser.add_argument("--initial_mapping", type=str, default="LANDCOVER10")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--time_explicit", action="store_true")
    parser.add_argument("--enable_masking", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--use_class_balancing",
        action="store_true",
        help="Weight sampler draws by class frequency within each task pool.",
    )
    parser.add_argument(
        "--log_tensorboard",
        action="store_true",
        help="Write TensorBoard event files with train/val metrics inside the experiment folder.",
    )
    parser.add_argument(
        "--task_balancing_method",
        type=str,
        default="balanced",
        choices=["balanced", "log", "effective", "none"],
        help="Strategy for balancing tasks when building the sampler.",
    )
    parser.add_argument(
        "--class_balancing_method",
        type=str,
        default="balanced",
        choices=["balanced", "log", "effective", "none"],
        help="Strategy for balancing classes within each task.",
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=100,
        help="Minimum number of samples required for a class to be included in training. Classes with fewer samples will be removed. Decision is taken based on the training set, but the same classes will be removed from val and test for consistency.",
    )
    parser.add_argument(
        "--balancing_clip_min",
        type=float,
        default=0.1,
        help="Lower bound applied to sampler weights when balancing is enabled.",
    )
    parser.add_argument(
        "--balancing_clip_max",
        type=float,
        default=10.0,
        help="Upper bound applied to sampler weights when balancing is enabled.",
    )
    parser.add_argument(
        "--sample_weight_strategy",
        type=str,
        default="quality",
        choices=["none", "quality"],
        help=(
            "Enable sample-level weighting using quality/confidence scores. "
            "Outlier weighting is always applied when confidence_nonoutlier columns are available."
        ),
    )
    parser.add_argument(
        "--sample_weight_quality",
        type=float,
        default=1.0,
        help="Relative weight for quality scores when combining with outlier scores.",
    )
    parser.add_argument(
        "--sample_weight_outlier",
        type=float,
        default=1.0,
        help="Relative weight for outlier scores when combining with quality scores.",
    )
    parser.add_argument(
        "--outlier_mode",
        type=str,
        default="keep",
        choices=["keep", "drop_candidate", "drop_suspect", "drop_flagged"],
        help="Keep all samples or drop outliers based on nested anomaly_flag categories. "
        "E.g., if 'drop_suspect' is chosen, both 'suspect' and 'candidate' categories will be dropped.",
    )
    parser.add_argument(
        "--spatial_group_column",
        type=str,
        default=None,
        help="Name of a dataframe column containing spatial group IDs for balancing (e.g., tile IDs).",
    )
    parser.add_argument(
        "--spatial_bin_size_deg",
        type=float,
        default=None,
        help="Size in degrees of latitude/longitude bins when spatial_group_column is not provided.",
    )
    parser.add_argument(
        "--spatial_balancing_method",
        type=str,
        default="log",
        choices=["balanced", "log", "effective", "none"],
        help="Strategy for down/up-weighting spatial groups when balancing is enabled.",
    )
    parser.add_argument(
        "--head_only_training",
        type=int,
        default=0,
        help="Freeze encoder weights for this many initial epochs (0 disables freezing)",
    )
    parser.add_argument(
        "--head_learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate to use while training the head with the encoder frozen.",
    )
    parser.add_argument(
        "--full_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate target once the encoder is unfrozen.",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.99,
        help="Multiplicative decay applied each epoch after the warmup stage.",
    )
    parser.add_argument(
        "--post_unfreeze_warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs to linearly ramp from a reduced LR to the full finetuning LR after unfreezing.",
    )
    parser.add_argument(
        "--post_unfreeze_warmup_start_factor",
        type=float,
        default=0.1,
        help="Relative factor (0-1] of the full finetuning LR to start from during the post-unfreeze warmup stage.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size for training."
    )

    # Label timing (for time_explicit only)
    parser.add_argument("--label_jitter", type=int, default=0)
    parser.add_argument("--label_window", type=int, default=0)
    parser.add_argument(
        "--val_loss_ema_alpha",
        type=float,
        default=0.0,
        help=(
            "Exponential moving average alpha for smoothing val loss used in early stopping "
            "and best-model selection. 0.0 disables smoothing (raw val loss is used directly). "
            "Suggested: 0.3 (new epoch gets 30%% weight, running EMA gets 70%%)."
        ),
    )

    # Season coverage threshold for the training split.
    # Val/test always enforce full coverage (1.0). During training with augmentation
    # the timestamp window can shift so a season is only partially covered;
    # lowering this threshold prevents losing croptype supervision in those cases.
    parser.add_argument(
        "--train_min_season_coverage",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of a season's composite slots that must fall inside "
            "the selected 12-timestamp window for the season to contribute to "
            "crop-type supervision in the training split. "
            "Val/test always use 1.0 (full coverage required). "
            "Default: 0.5."
        ),
    )

    # Season selection
    parser.add_argument(
        "--season_ids",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Season IDs for crop-type supervision (e.g. 'tc-s1 tc-s2' or 'annual'). "
            "Defaults to GLOBAL_SEASON_IDS (tc-s1, tc-s2). "
            "Use 'annual' for a single annual season from the crop calendar."
        ),
    )

    # Optional post-finetuning spatial inference
    parser.add_argument(
        "--run_spatial_inference",
        action="store_true",
        help="Run local spatial inference on patch .nc files at the end of finetuning.",
    )
    parser.add_argument(
        "--spatial_inference_patches_dir",
        type=str,
        default=None,
        help="Root directory containing continent subfolders with .nc patches.",
    )
    parser.add_argument(
        "--spatial_inference_continents",
        type=str,
        default="all",
        help=(
            "Continent selection for spatial inference: 'all' or CSV list (e.g. Africa,Europe)."
        ),
    )

    # Parse the arguments
    args = parser.parse_args(arg_list)

    return args


if __name__ == "__main__":
    # manual_args = [
    #     "--experiment_tag",
    #     "debug-run",
    #     "--timestep_freq",
    #     "month",
    #     "--enable_masking",
    #     "--time_explicit",
    #     "--label_jitter",
    #     "1",
    #     "--augment",
    #     "--initial_mapping",
    #     "LANDCOVER10",  # CROPTYPE28
    #     "--use_class_balancing",
    #     "--spatial_bin_size_deg",
    #     "5.0",
    #     "--head_only_training",
    #     "10",
    #     "--log_tensorboard",
    #     "--debug",
    # ]
    manual_args = None

    args = parse_args(manual_args)
    main(args)
