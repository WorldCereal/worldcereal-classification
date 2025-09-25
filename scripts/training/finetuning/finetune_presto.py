#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
from loguru import logger
from prometheo.finetune import Hyperparams
from prometheo.models.presto import param_groups_lrd
from prometheo.utils import DEFAULT_SEED, device, initialize_logging
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

# from worldcereal_in_season.datasets import MaskingStrategy
from worldcereal.train.data import get_training_dfs_from_parquet
from worldcereal.train.finetuning_utils import (
    evaluate_finetuned_model,
    prepare_training_datasets,
    run_finetuning,
)
from worldcereal.train.models import build_worldcereal_presto
from worldcereal.utils.refdata import get_class_mappings

CLASS_MAPPINGS = get_class_mappings()


def get_parquet_file_list(timestep_freq: Literal["month", "dekad"] = "month"):
    if timestep_freq == "month":
        parquet_files = list(
            Path(
                "/projects/TAP/worldcereal/data/WORLDCEREAL_ALL_EXTRACTIONS/worldcereal_all_extractions.parquet"
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

    # Path to the training data
    parquet_files = get_parquet_file_list(timestep_freq)
    val_samples_file = args.val_samples_file  # If None, random split is used

    # Most popular maps: LANDCOVER14, CROPTYPE9, CROPTYPE0, CROPLAND2
    finetune_classes = args.finetune_classes
    augment = args.augment
    time_explicit = args.time_explicit
    debug = args.debug
    use_balancing = args.use_balancing  # If True, use class balancing for training
    temporal_attention = args.temporal_attention

    # ± timesteps to jitter true label pos, for time_explicit only; will only be set for training
    label_jitter = args.label_jitter

    # ± timesteps to expand around label pos (true or moved), for time_explicit only; will only be set for training
    raw_label_window = args.label_window
    time_kernel = args.time_kernel
    time_kernel_bandwidth = args.time_kernel_bandwidth
    if temporal_attention and not time_explicit:
        logger.info(
            "Temporal attention requires time-explicit supervision; enabling time_explicit mode"
        )
        time_explicit = True

    if temporal_attention and time_kernel == "delta":
        logger.info(
            "Temporal attention requires temporal kernel weighting; switching to 'gaussian' kernel"
        )
        time_kernel = "gaussian"

    if temporal_attention and raw_label_window == 0:
        raw_label_window = 1
        logger.info(
            "Temporal attention → setting label_window=1 to provide supervision window"
        )

    temporal_kernel_active = time_explicit and time_kernel != "delta"
    needs_time_weights = temporal_kernel_active or temporal_attention
    apply_temporal_weights = temporal_kernel_active

    if not apply_temporal_weights and raw_label_window != 0:
        logger.info(
            "Disabling label_window expansion (set to 0) because temporal loss weighting is off"
        )
        raw_label_window = 0

    if needs_time_weights and time_kernel != "delta" and time_kernel_bandwidth is None:
        time_kernel_bandwidth = max(
            1.0, float(raw_label_window) if raw_label_window > 0 else 1.0
        )
        logger.info(
            "Setting default time_kernel_bandwidth={} for kernel '{}'",
            time_kernel_bandwidth,
            time_kernel,
        )

    freeze_layers = None
    unfreeze_epoch = None
    if args.freeze_encoder_epochs > 0:
        freeze_layers = ["encoder"]
        unfreeze_epoch = args.freeze_encoder_epochs
        logger.info(
            f"Freezing encoder for the first {args.freeze_encoder_epochs} epoch(s)"
        )

    # Experiment signature
    timestamp_ind = datetime.now().strftime("%Y%m%d%H%M")

    experiment_name = f"presto-prometheo-{experiment_tag}-{timestep_freq}-{finetune_classes}-augment={augment}-balance={use_balancing}-timeexplicit={time_explicit}-run={timestamp_ind}"
    output_dir = f"/projects/worldcereal/models/{experiment_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Training parameters
    pretrained_model_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
    epochs = 100
    batch_size = 1024
    patience = 6
    num_workers = 8

    # ------------------------------------------

    # Setup logging
    initialize_logging(
        log_file=Path(output_dir) / "logs" / f"{experiment_name}.log",
        level="INFO",
        console_filter_keyword="PROGRESS",
    )

    # Get the train/val/test dataframes
    train_df, val_df, test_df = get_training_dfs_from_parquet(
        parquet_files,
        timestep_freq=timestep_freq,
        finetune_classes=finetune_classes,
        class_mappings=CLASS_MAPPINGS,
        val_samples_file=val_samples_file,
        debug=debug,
    )

    logger.warning("Still applying a patch here ...")
    train_df = train_df[train_df["available_timesteps"] >= 12]
    val_df = val_df[val_df["available_timesteps"] >= 12]
    test_df = test_df[test_df["available_timesteps"] >= 12]

    logger.info("Saving train, val, and test DataFrames to parquet files ...")
    train_df.to_parquet(Path(output_dir) / "train_df.parquet")
    val_df.to_parquet(Path(output_dir) / "val_df.parquet")
    test_df.to_parquet(Path(output_dir) / "test_df.parquet")

    classes_list = list(sorted(set(CLASS_MAPPINGS[finetune_classes].values())))
    classes_list = [
        xx for xx in classes_list if xx in train_df["finetune_class"].unique()
    ]
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

    # Construct training and validation datasets
    label_window = raw_label_window if raw_label_window > 0 else None

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
        label_jitter=label_jitter,
        label_window=label_window,
        return_time_weights=needs_time_weights,
        time_kernel=time_kernel,
        time_kernel_bandwidth=time_kernel_bandwidth,
    )

    if temporal_attention:
        logger.info(
            "Temporal attention head enabled; using temporal priors as soft bias"
        )

    if apply_temporal_weights:
        logger.info(
            f"Temporal kernel weighting enabled (kernel={time_kernel}, bandwidth={time_kernel_bandwidth})"
        )
    elif needs_time_weights:
        logger.info("Temporal priors provided to the model (without loss weighting)")

    # Construct the finetuning model based on the pretrained model
    model = build_worldcereal_presto(
        num_outputs=num_outputs,
        regression=False,
        pretrained_model_path=pretrained_model_path,
        temporal_attention=temporal_attention,
    ).to(device)

    # Set the parameters
    hyperparams = Hyperparams(
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        num_workers=num_workers,
    )
    parameters = param_groups_lrd(model)
    optimizer = AdamW(parameters, lr=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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
            clip_range=(0.2, 10),
        )
        if use_balancing
        else None,
        generator=generator if not use_balancing else None,
        num_workers=hyperparams.num_workers,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=hyperparams.num_workers,
    )

    # Run the finetuning
    logger.info("Starting finetuning...")
    finetuned_model = run_finetuning(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        experiment_name=experiment_name,
        output_dir=output_dir,
        task_type=task_type,
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparams=hyperparams,
        setup_logging=False,
        apply_temporal_weights=apply_temporal_weights,
        attention_entropy_weight=0.01,
        freeze_layers=freeze_layers,
        unfreeze_epoch=unfreeze_epoch,
    )

    # Evaluate the finetuned model
    logger.info("Evaluating the finetuned model...")
    eval_results, confusionmatrix, confusionmatrix_norm = evaluate_finetuned_model(
        finetuned_model,
        test_ds,
        num_workers,
        batch_size,
        time_explicit=time_explicit,
        classes_list=classes_list,
    )

    # Adjust figure size based on label length
    max_label_length = max(len(label) for label in classes_list)
    per_label_size = 0.45  # Width/height in inches per label
    label_length_factor = 0.1  # Additional size per character in the longest label

    # Define minimum and maximum limits if desired
    min_size = 6
    max_size = 30

    # Compute figure size dynamically
    fig_size = min(
        max(
            len(classes_list) * per_label_size + max_label_length * label_length_factor,
            min_size,
        ),
        max_size,
    )

    _, ax = plt.subplots(figsize=(fig_size, fig_size))
    confusionmatrix.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / f"CM_{experiment_name}.png"))
    plt.close()

    _, ax = plt.subplots(figsize=(fig_size, fig_size))
    confusionmatrix_norm.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    # Format the text annotations: keep 2 decimal places
    for text in ax.texts:
        val = float(text.get_text())
        text.set_text(f"{val:.2f}")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / f"CM_{experiment_name}_norm.png"))
    plt.close()

    eval_results.to_csv(
        Path(output_dir) / f"results_{experiment_name}.csv", index=False
    )
    logger.info("Evaluation results:")
    logger.info("\n" + eval_results.to_string(index=False))

    logger.info("Finetuning completed!")


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Train in-season crop type model")

    # General setup
    parser.add_argument("--experiment_tag", type=str, default="")
    parser.add_argument(
        "--timestep_freq", type=str, choices=["month", "dekad"], default="month"
    )

    # Data paths
    parser.add_argument(
        "--val_samples_file",
        type=str,
        default=None,
        help="Path to a CSV with val sample IDs. If not set, a random split will be used.",
    )

    # Task setup
    parser.add_argument("--finetune_classes", type=str, default="LANDCOVER14")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--time_explicit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_balancing", action="store_true")
    parser.add_argument("--temporal_attention", action="store_true")
    parser.add_argument(
        "--freeze_encoder_epochs",
        type=int,
        default=0,
        help="Freeze encoder weights for this many initial epochs (0 disables freezing)",
    )
    parser.add_argument(
        "--time_kernel",
        type=str,
        choices=["delta", "gaussian", "triangular"],
        default="delta",
    )
    parser.add_argument("--time_kernel_bandwidth", type=float, default=None)

    # Label timing (for time_explicit only)
    parser.add_argument("--label_jitter", type=int, default=0)
    parser.add_argument("--label_window", type=int, default=0)

    # parser.add_argument(
    #     type=str,
    #     choices=["none", "fixed", "random"],
    #     default="random",

    # parser.add_argument(
    #     type=str,
    #     choices=["none", "fixed", "random"],
    #     default="fixed",

    args = parser.parse_args(arg_list)

    return args


if __name__ == "__main__":
    # manual_args = [
    #     "--experiment_tag",
    #     "debug-run",
    #     "--timestep_freq",
    #     "month",
    #     "--time_explicit",
    #     "--label_jitter",
    #     "1",
    #     "--augment",
    #     "--finetune_classes",
    #     "CROPTYPE20",  # LANDCOVER14
    #     "--use_balancing",
    #     "--debug",

    # ]
    manual_args = None

    args = parse_args(manual_args)
    main(args)
