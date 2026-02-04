#!/usr/bin/env python3
import argparse
import json
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

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

from worldcereal.train.backbone import checkpoint_fingerprint
from worldcereal.train.data import collate_fn, get_training_dfs_from_parquet
from worldcereal.train.datasets import SensorMaskingConfig
from worldcereal.train.finetuning_utils import (
    SeasonalMultiTaskLoss,
    evaluate_finetuned_model,
    prepare_training_datasets,
    run_finetuning,
)
from worldcereal.train.seasonal_head import (
    SeasonalFinetuningHead,
    WorldCerealSeasonalModel,
)
from worldcereal.utils.refdata import get_class_mappings

CLASS_MAPPINGS = get_class_mappings()


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


def _zip_checkpoint_with_config(
    checkpoint_path: Path,
    *,
    manifest_path: Path,
    run_config_path: Path,
) -> Optional[Path]:
    """Bundle a checkpoint with its manifest/config so inference can recover metadata."""

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint {checkpoint_path} not found; skipping packaging")
        return None
    if not manifest_path.exists():
        logger.warning(
            f"Head manifest missing at {manifest_path}; cannot package {checkpoint_path.name}"
        )
        return None

    zip_path = checkpoint_path.with_suffix(".zip")
    try:
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(checkpoint_path, arcname=checkpoint_path.name)
            zf.write(manifest_path, arcname="config.json")
            if run_config_path.exists():
                zf.write(run_config_path, arcname=run_config_path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to package {checkpoint_path.name}: {exc}")
        return None

    logger.info(f"Packaged {checkpoint_path.name} with manifest into {zip_path}")
    return zip_path


def _package_model_checkpoints(
    checkpoint_paths: Sequence[Path],
    *,
    manifest_path: Path,
    run_config_path: Path,
) -> None:
    """Create zip artifacts for each checkpoint using the shared manifest/config."""

    for checkpoint_path in checkpoint_paths:
        _zip_checkpoint_with_config(
            checkpoint_path,
            manifest_path=manifest_path,
            run_config_path=run_config_path,
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
    missing_landcover = updated["landcover_label"].isna()
    if missing_landcover.any():
        removed = int(missing_landcover.sum())
        logger.warning(
            f"Removing {removed} samples from {split_name} split without '{landcover_key}' landcover mapping"
        )
        updated = updated.loc[~missing_landcover].copy()
        if updated.empty:
            raise ValueError(
                f"No samples remain in {split_name} split after applying landcover mapping {landcover_key}."
            )

    croptype_map = {
        int(code): label for code, label in CLASS_MAPPINGS[croptype_key].items()
    }
    updated["croptype_label"] = updated["ewoc_code"].map(croptype_map)

    has_croptype_label = updated["croptype_label"].notna()
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


def get_parquet_file_list(timestep_freq: Literal["month", "dekad"] = "month"):
    if timestep_freq == "month":
        parquet_files = list(
            Path(
                "/projects/TAP/worldcereal/data/worldcereal_all_extractions.parquet"
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
    timestep_freq = args.timestep_freq  # "month" or "dekad"
    max_timesteps_trim = args.max_timesteps_trim  # "auto", int or tuple of string dates
    use_valid_time = args.use_valid_time

    # Path to the training data
    parquet_files = get_parquet_file_list(timestep_freq)
    val_samples_file = args.val_samples_file  # If None, random split is used

    # Most popular maps: LANDCOVER10, CROPTYPE9, CROPTYPE0, CROPLAND2
    initial_mapping = args.initial_mapping
    augment = args.augment
    time_explicit = args.time_explicit
    enable_masking = args.enable_masking
    debug = args.debug
    use_balancing = args.use_balancing  # If True, use class balancing for training
    cropland_class_names = [
        cls.strip() for cls in args.landcover_cropland_classes.split(",") if cls.strip()
    ]
    if not cropland_class_names:
        cropland_class_names = ["temporary_crops"]

    # ± timesteps to jitter true label pos, for time_explicit only; will only be set for training
    label_jitter = args.label_jitter

    # ± timesteps to expand around label pos (true or moved), for time_explicit only; will only be set for training
    label_window = args.label_window

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

    experiment_name = f"presto-prometheo-{experiment_tag}-{timestep_freq}-augment={augment}-balance={use_balancing}-timeexplicit={time_explicit}-masking={masking_info}-run={timestamp_ind}"
    output_dir = f"/projects/worldcereal/models/{experiment_name}"
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
    if not debug:
        wide_parquet_output_path = Path(
            "/projects/worldcereal/data/cached_wide_merged305/merged_305_wide.parquet"
        )
    else:
        wide_parquet_output_path = None
        # wide_parquet_output_path = Path(
        # "/projects/worldcereal/data/cached_wide_merged305/merged_305_wide.parquet"
        # )

    # Training parameters
    pretrained_model_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
    epochs = 50
    batch_size = 4096
    patience = 8
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
    # TODO: handle outliers
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
            debug=debug,
            overwrite=False,
        )
        logger.info("Saving train, val, and test DataFrames to parquet files ...")
        train_df.to_parquet(train_df_path)
        val_df.to_parquet(val_df_path)
        test_df.to_parquet(test_df_path)

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

    # Use type casting to specify to mypy that task_type is a valid Literal value
    task_type_literal: Literal["binary", "multiclass"] = "multiclass"  # type: ignore

    # Construct training and validation datasets with masking parameters
    train_ds, val_ds, test_ds = prepare_training_datasets(
        train_df,
        val_df,
        test_df,
        num_timesteps=12 if timestep_freq == "month" else 36,
        timestep_freq=timestep_freq,
        augment=augment,
        time_explicit=time_explicit,
        emit_label_tensor=False,  # Supervise through attributes only
        task_type=task_type_literal,
        masking_config=masking_config,
        label_jitter=label_jitter,
        label_window=label_window,
    )

    # Construct the finetuning model based on the pretrained model
    landcover_classes = _unique_preserve_order(CLASS_MAPPINGS[landcover_key].values())
    croptype_classes = _unique_preserve_order(CLASS_MAPPINGS[croptype_key].values())
    if not landcover_classes or not croptype_classes:
        raise ValueError(
            "Both landcover and croptype class mappings must contain at least one class."
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
    class_column_map = {
        "landcover": "landcover_label",
        "croptype": "croptype_label",
    }
    sample_weight_mapping: Optional[dict[str, str]] = None
    if args.sample_weight_strategy == "quality":
        sample_weight_mapping = {
            "landcover": "quality_score_lc",
            "croptype": "quality_score_ct",
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

    train_sampler = None
    if use_balancing:
        train_sampler = train_ds.get_task_balanced_sampler(
            task_weight_method=args.task_balancing_method,
            class_weight_method=args.class_balancing_method,
            class_column_map=class_column_map,
            clip_range=balancing_clip,
            spatial_group_column=args.spatial_group_column,
            spatial_bin_size_degrees=args.spatial_bin_size_deg,
            spatial_weight_method=args.spatial_balancing_method,
            generator=generator,
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=hyperparams.batch_size,
        shuffle=True if not use_balancing else None,
        sampler=train_sampler,
        generator=generator if not use_balancing else None,
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
        "enabled": use_balancing,
        "task_weight_method": args.task_balancing_method,
        "class_weight_method": args.class_balancing_method,
        "class_column_map": class_column_map,
        "clip_range": list(balancing_clip) if balancing_clip else None,
        "spatial_group_column": args.spatial_group_column,
        "spatial_bin_size_deg": args.spatial_bin_size_deg,
        "spatial_balancing_method": args.spatial_balancing_method,
    }
    data_sources_payload = {
        "parquet_files": [str(path) for path in parquet_files],
        "wide_parquet_output_path": _path_to_str(wide_parquet_output_path),
        "val_samples_file": args.val_samples_file,
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

    checkpoints_to_package = [
        Path(output_dir) / f"{experiment_name}.pt",
        Path(output_dir) / f"{experiment_name}_encoder.pt",
    ]
    _package_model_checkpoints(
        checkpoints_to_package,
        manifest_path=manifest_path,
        run_config_path=config_path,
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
        default="CROPTYPE28",
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
        "--val_samples_file",
        type=str,
        default=None,
        help="Path to a CSV with val sample IDs. If not set, a random split will be used.",
    )

    # Task setup
    parser.add_argument("--initial_mapping", type=str, default="LANDCOVER10")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--time_explicit", action="store_true")
    parser.add_argument("--enable_masking", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_balancing", action="store_true")
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
        "--balancing_clip_min",
        type=float,
        default=0.3,
        help="Lower bound applied to sampler weights when balancing is enabled.",
    )
    parser.add_argument(
        "--balancing_clip_max",
        type=float,
        default=5.0,
        help="Upper bound applied to sampler weights when balancing is enabled.",
    )
    parser.add_argument(
        "--sample_weight_strategy",
        type=str,
        default="quality",
        choices=["none", "quality"],
        help="Enable sample-level weighting (currently supports using quality scores).",
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

    # Label timing (for time_explicit only)
    parser.add_argument("--label_jitter", type=int, default=0)
    parser.add_argument("--label_window", type=int, default=0)

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
    #     "--use_balancing",
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
