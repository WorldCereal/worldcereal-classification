#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
from loguru import logger
from prometheo.finetune import Hyperparams, run_finetuning
from prometheo.models import Presto
from prometheo.models.presto import param_groups_lrd
from prometheo.predictors import NODATAVALUE
from prometheo.utils import DEFAULT_SEED, device, initialize_logging
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

# from worldcereal_in_season.datasets import MaskingStrategy
from worldcereal.train.data import get_training_dfs_from_parquet
from worldcereal.train.finetuning_utils import (
    evaluate_finetuned_model,
    prepare_training_datasets,
)
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
    max_timesteps_trim = args.max_timesteps_trim  # "auto", int or tuple of string dates
    use_valid_time = args.use_valid_time

    # Path to the training data
    parquet_files = get_parquet_file_list(timestep_freq)
    val_samples_file = args.val_samples_file  # If None, random split is used

    # Most popular maps: LANDCOVER14, CROPTYPE9, CROPTYPE0, CROPLAND2
    finetune_classes = args.finetune_classes
    augment = args.augment
    time_explicit = args.time_explicit
    debug = args.debug
    use_balancing = args.use_balancing  # If True, use class balancing for training

    # ± timesteps to jitter true label pos, for time_explicit only; will only be set for training
    label_jitter = args.label_jitter

    # ± timesteps to expand around label pos (true or moved), for time_explicit only; will only be set for training
    label_window = args.label_window

    # # In-season masking parameters
    # masking_strategy_train = args.masking_strategy_train
    # masking_strategy_val = args.masking_strategy_val

    # Experiment signature
    timestamp_ind = datetime.now().strftime("%Y%m%d%H%M")

    # # Update experiment name to include masking info
    # if masking_strategy_train.mode == "random":
    #     masking_info = f"random-masked-from-{masking_strategy_train.from_position}"
    # elif masking_strategy_train.mode == "fixed":
    #     masking_info = f"masked-from-{masking_strategy_train.from_position}"
    # else:
    #     masking_info = "no-masking"

    experiment_name = f"presto-prometheo-{experiment_tag}-{timestep_freq}-{finetune_classes}-augment={augment}-balance={use_balancing}-timeexplicit={time_explicit}-run={timestamp_ind}"
    output_dir = f"/projects/worldcereal/models/{experiment_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # setup path for processed wide parquet file so that it can be reused across experiments
    wide_parquet_output_path = Path("/projects/worldcereal/merged_parquets_wide.parquet")

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
        # masking_strategy_train=masking_strategy_train,
        # masking_strategy_val=masking_strategy_val,
        label_jitter=label_jitter,
        label_window=label_window,
    )

    # Construct the finetuning model based on the pretrained model
    model = Presto(
        num_outputs=num_outputs,
        regression=False,
        pretrained_model_path=pretrained_model_path,
    ).to(device)

    # Define the loss function based on the task type
    if task_type == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass":
        loss_fn = nn.CrossEntropyLoss(ignore_index=NODATAVALUE)
    else:
        raise ValueError(
            f"Task type {task_type} is not supported. "
            f"Supported task types are 'binary' and 'multiclass'."
        )

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
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparams=hyperparams,
        setup_logging=False,  # Already setup logging
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
    parser.add_argument(
        "--max_timesteps_trim",
        type=str,
        default="auto",
        help='Maximum number of timesteps to retain after trimming. Can be "auto", an integer, or a tuple of string dates.',
    )
    parser.add_argument(
        "--use_valid_time",
        action="store_true",
        help="Whether to use the 'valid_time' column for processing timesteps.",
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

    # Label timing (for time_explicit only)
    parser.add_argument("--label_jitter", type=int, default=0)
    parser.add_argument("--label_window", type=int, default=0)

    # # Masking strategy
    # parser.add_argument(
    #     "--masking_train_mode",
    #     type=str,
    #     choices=["none", "fixed", "random"],
    #     default="random",
    # )
    # parser.add_argument("--masking_train_from", type=int, default=5)

    # parser.add_argument(
    #     "--masking_val_mode",
    #     type=str,
    #     choices=["none", "fixed", "random"],
    #     default="fixed",
    # )
    # parser.add_argument("--masking_val_from", type=int, default=6)

    args = parser.parse_args(arg_list)

    # # Compose masking strategy objects
    # args.masking_strategy_train = MaskingStrategy(
    #     mode=args.masking_train_mode, from_position=args.masking_train_from
    # )
    # args.masking_strategy_val = MaskingStrategy(
    #     mode=args.masking_val_mode, from_position=args.masking_val_from
    # )

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
    #     # "--masking_train_mode",
    #     # "random",
    #     # "--masking_train_from",
    #     # "15",
    #     # "--masking_val_mode",
    #     # "fixed",
    #     # "--masking_val_from",
    #     # "18",
    # ]
    manual_args = None

    args = parse_args(manual_args)
    main(args)
