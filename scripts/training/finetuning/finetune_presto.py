#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

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
        logger.warning("%s split is empty before seasonal label annotation", split_name)
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
            "Removing %d samples from %s split without '%s' landcover mapping",
            removed,
            split_name,
            landcover_key,
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

    cropland_targets = set(cropland_class_names)
    is_cropland = updated["landcover_label"].isin(cropland_targets)
    has_croptype_label = updated["croptype_label"].notna()
    updated["label_task"] = np.where(
        is_cropland & has_croptype_label,
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
            "%s split is empty; skipping seasonal task diversity validation",
            split_name,
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
    finetune_classes = args.finetune_classes
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

    experiment_name = f"presto-prometheo-{experiment_tag}-{timestep_freq}-{finetune_classes}-augment={augment}-balance={use_balancing}-timeexplicit={time_explicit}-masking={masking_info}-run={timestamp_ind}"
    output_dir = f"/projects/worldcereal/models/{experiment_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _export_eval_artifacts(
        eval_results: pd.DataFrame,
        cm: Figure,
        cm_norm: Figure,
        class_names: Optional[List[str]],
        *,
        suffix: Optional[str] = None,
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
        cm_path = Path(output_dir) / f"CM_{experiment_name}{suffix_part}.png"
        cm_norm_path = Path(output_dir) / f"CM_{experiment_name}{suffix_part}_norm.png"

        cm.savefig(cm_path, bbox_inches="tight")
        plt.close(cm)
        cm_norm.savefig(cm_norm_path, bbox_inches="tight")
        plt.close(cm_norm)

        eval_results.to_csv(
            Path(output_dir) / f"results_{experiment_name}{suffix_part}.csv",
            index=False,
        )

    # setup path for processed wide parquet file so that it can be reused across experiments
    if not debug:
        wide_parquet_output_path = Path(
            f"/projects/TAP/worldcereal/data/cached_wide_parquets/worldcereal_all_extractions_wide_{timestep_freq}_{finetune_classes}.parquet"
        )
    else:
        # wide_parquet_output_path = None
        wide_parquet_output_path = Path(
            "/projects/worldcereal/data/cached_wide_merged305/merged_305_wide.parquet"
        )

    # Training parameters
    pretrained_model_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
    epochs = 25
    batch_size = 2024
    patience = 6
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
    # TODO: alternative pathway without mapping the classes early (so classes are being removed unwanted)
    # TODO: build-in confidence score handling for both tasks
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
            finetune_classes=finetune_classes,
            class_mappings=CLASS_MAPPINGS,
            val_samples_file=val_samples_file,
            debug=debug,
            overwrite=False,
        )
        logger.info("Saving train, val, and test DataFrames to parquet files ...")
        train_df.to_parquet(train_df_path)
        val_df.to_parquet(val_df_path)
        test_df.to_parquet(test_df_path)

    landcover_key = args.landcover_classes_key or "LANDCOVER10"
    croptype_key = args.croptype_classes_key or finetune_classes
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

    if debug:
        train_df = train_df.sample(n=10000, random_state=DEFAULT_SEED).reset_index(
            drop=True
        )
        val_df = val_df.sample(n=2000, random_state=DEFAULT_SEED).reset_index(drop=True)
        test_df = test_df.sample(n=2000, random_state=DEFAULT_SEED).reset_index(
            drop=True
        )
        logger.info("Debug mode: reduced dataset sizes.")

    _validate_seasonal_task_labels(train_df, split_name="train", strict=True)
    _validate_seasonal_task_labels(val_df, split_name="val", strict=False)
    _validate_seasonal_task_labels(test_df, split_name="test", strict=False)

    mapping_values = _unique_preserve_order(CLASS_MAPPINGS[finetune_classes].values())
    present_classes = {
        str(xx) for xx in train_df["finetune_class"].dropna().unique().tolist()
    }
    classes_list = [xx for xx in mapping_values if xx in present_classes]
    logger.info(f"classes_list: {classes_list}")
    num_classes = train_df["finetune_class"].nunique()
    if num_classes == 2:
        task_type = "binary"
        num_outputs = 1
    elif num_classes > 2:
        task_type = "multiclass"
        num_outputs = num_classes
    else:
        raise ValueError(
            f"Number of classes {num_classes} is not supported. "
            f"Dataset contains the following classes: {train_df.finetune_class.unique()}."
        )
    logger.info(f"Task type: {task_type}, num_outputs: {num_outputs}")
    logger.info(f"Number of training samples: {len(train_df)}")
    logger.info(f"Number of validation samples: {len(val_df)}")
    logger.info(f"Number of test samples: {len(test_df)}")

    # Use type casting to specify to mypy that task_type is a valid Literal value
    task_type_literal: Literal["binary", "multiclass"] = task_type  # type: ignore

    # Construct training and validation datasets with masking parameters
    train_ds, val_ds, test_ds = prepare_training_datasets(
        train_df,
        val_df,
        test_df,
        num_timesteps=12 if timestep_freq == "month" else 36,
        timestep_freq=timestep_freq,
        augment=augment,
        time_explicit=time_explicit,
        task_type=task_type_literal,
        num_outputs=num_outputs,
        classes_list=classes_list,
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
    loss_fn = SeasonalMultiTaskLoss(
        landcover_classes=landcover_classes,
        croptype_classes=croptype_classes,
        ignore_index=NODATAVALUE,
        landcover_weight=args.seasonal_loss_landcover_weight,
        croptype_weight=args.seasonal_loss_croptype_weight,
    )
    logger.info(
        "Using seasonal head with landcover key {} ({} classes) and croptype key {} ({} classes)",
        landcover_key,
        len(landcover_classes),
        croptype_key,
        len(croptype_classes),
    )
    logger.info("Using loss function: {}", loss_fn)

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

    # Define optimizer
    head_lr = 1e-2
    full_lr = 1e-4

    parameters = param_groups_lrd(model)
    optimizer = AdamW(parameters, lr=head_lr)
    for group in optimizer.param_groups:
        group["initial_lr"] = head_lr  # required by some schedulers.

    # Phase 1: constant LR = base_lr
    const_sched = lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,  # keep LR = base_lr
        total_iters=unfreeze_epoch
        if unfreeze_epoch is not None
        else 0,  # number of epochs before unfreeze
    )

    # Phase 2: exponential decay but starting from a reduced LR
    for group in optimizer.param_groups:
        group["lr"] = (
            full_lr  # Manually scale the optimizer LR before building the second scheduler
        )
    exp_sched = lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99,
    )

    # Create the scheduler
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[const_sched, exp_sched],
        milestones=[unfreeze_epoch if unfreeze_epoch is not None else 0],
    )

    # Setup dataloaders
    generator = torch.Generator()
    generator.manual_seed(DEFAULT_SEED)

    train_dl = DataLoader(
        train_ds,
        batch_size=hyperparams.batch_size,
        shuffle=True if not use_balancing else None,
        sampler=train_ds.get_balanced_sampler(
            generator=generator,
            sampling_class="finetune_class",
            method="log",
            clip_range=(0.3, 5),
        )
        if use_balancing
        else None,
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
    )

    logger.info("Evaluating seasonal head outputs...")
    seasonal_results = evaluate_finetuned_model(
        finetuned_model,
        test_ds,
        num_workers,
        batch_size,
        time_explicit=time_explicit,
        seasonal_landcover_classes=landcover_classes,
        seasonal_croptype_classes=croptype_classes,
        cropland_class_names=cropland_class_names,
    )
    for task_name, artifacts in seasonal_results.items():
        results_df = artifacts["results"]
        cm = artifacts["cm"]
        cm_norm = artifacts["cm_norm"]
        task_classes = artifacts.get("classes")
        _export_eval_artifacts(
            results_df,
            cm,
            cm_norm,
            task_classes,
            suffix=task_name,
        )
        logger.info("%s evaluation results:", task_name.capitalize())
        logger.info("\n" + results_df.to_string(index=False))
        if task_name == "croptype" and artifacts.get("gate_rejections"):
            logger.info(
                "Croptype predictions skipped due to landcover gate: %d",
                artifacts["gate_rejections"],
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
        default=None,
        help="Class mapping key used for landcover targets in the dual-head configuration.",
    )
    parser.add_argument(
        "--croptype_classes_key",
        type=str,
        default="CROPTYPE27",
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
    parser.add_argument("--finetune_classes", type=str, default="LANDCOVER10")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--time_explicit", action="store_true")
    parser.add_argument("--enable_masking", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_balancing", action="store_true")
    parser.add_argument(
        "--head_only_training",
        type=int,
        default=0,
        help="Freeze encoder weights for this many initial epochs (0 disables freezing)",
    )

    # Label timing (for time_explicit only)
    parser.add_argument("--label_jitter", type=int, default=0)
    parser.add_argument("--label_window", type=int, default=0)

    # Parse the arguments
    args = parser.parse_args(arg_list)

    return args


if __name__ == "__main__":
    manual_args = [
        "--experiment_tag",
        "debug-run",
        "--timestep_freq",
        "month",
        "--enable_masking",
        "--time_explicit",
        "--label_jitter",
        "1",
        "--augment",
        "--finetune_classes",
        "LANDCOVER10",  # CROPTYPE27
        "--use_balancing",
        "--head_only_training",
        "5",
        "--debug",
    ]
    # manual_args = None

    args = parse_args(manual_args)
    main(args)
