from glob import glob
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from presto.utils import prep_dataframe, process_parquet
from prometheo.datasets import WorldCerealLabelledDataset
from prometheo.finetune import Hyperparams, run_finetuning
from prometheo.models.presto import param_groups_lrd
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper
from prometheo.predictors import NODATAVALUE
from prometheo.utils import initialize_logging
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from worldcereal.utils.refdata import split_df


def prepare_training_df(
    parquet_file: Union[Path, str],
    val_samples_file: Optional[Union[Path, str]] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Reading dataset")
    files = sorted(glob(f"{parquet_file}/**/*.parquet"))

    if debug:
        logger.warning("Debug mode is enabled. Max 3 files will be read.")
        files = files[:3]

    df_list = []
    for f in tqdm(files):
        _data = pd.read_parquet(f, engine="fastparquet")
        _ref_id = f.split("/")[-2].split("=")[-1]
        _data["ref_id"] = _ref_id
        _data_pivot = process_parquet(_data)
        _data_pivot.reset_index(inplace=True)
        df_list.append(_data_pivot)
    df = pd.concat(df_list)
    df = df.fillna(NODATAVALUE)
    del df_list

    df = prep_dataframe(df, filter_function=None, dekadal=False).reset_index()

    if val_samples_file is not None:
        logger.info(f"Controlled train/test split based on: {val_samples_file}")
        val_samples_df = pd.read_csv(val_samples_file)
        train_df, test_df = split_df(
            df, val_sample_ids=val_samples_df.sample_id.tolist()
        )
    else:
        logger.info("Random train/test split ...")
        train_df, test_df = split_df(df, val_size=0.2)
    train_df, val_df = split_df(train_df, val_size=0.2)

    return train_df, val_df, test_df


def prepare_training_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    augment: bool = True,
    time_explicit: bool = False,
) -> Tuple[
    WorldCerealLabelledDataset, WorldCerealLabelledDataset, WorldCerealLabelledDataset
]:
    train_ds = WorldCerealLabelledDataset(
        train_df,
        task_type="binary",
        num_outputs=1,
        time_explicit=time_explicit,
        augment=augment,
    )
    val_ds = WorldCerealLabelledDataset(
        val_df,
        task_type="binary",
        num_outputs=1,
        time_explicit=time_explicit,
        augment=False,
    )
    test_ds = WorldCerealLabelledDataset(
        test_df,
        task_type="binary",
        num_outputs=1,
        time_explicit=time_explicit,
        augment=False,
    )
    return train_ds, val_ds, test_ds


def evaluate_finetuned_model(
    finetuned_model: PretrainedPrestoWrapper,
    test_ds: WorldCerealLabelledDataset,
    num_workers: int,
    batch_size: int,
    task_type: Literal["cropland", "croptype"] = "cropland",
):
    assert task_type in ["cropland", "croptype"]

    # Put model in eval mode
    finetuned_model.eval()

    # Construct the dataloader
    val_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,  # keep as False!
        num_workers=num_workers,
    )
    assert isinstance(val_dl.sampler, torch.utils.data.SequentialSampler)

    # Run the model on the test set
    all_preds, all_targets = [], []

    for batch in val_dl:
        with torch.no_grad():
            preds = finetuned_model(batch)

            # Presto head does not contain an activation. Need to apply it here.
            if task_type == "cropland":
                preds = nn.functional.sigmoid(preds).cpu().numpy()
            else:
                preds = nn.functional.softmax(preds, dim=-1).cpu().numpy()

            # Flatten predictions and targets
            preds = preds.flatten()
            targets = batch.label.float().numpy().flatten()

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if task_type == "cropland":
        # metrics_agg = "binary"
        croptype_list = ["not_crop", "crop"]
        all_targets = np.array(
            ["crop" if xx > 0.5 else "not_crop" for xx in all_targets]
        )
        all_preds = np.array(["crop" if xx > 0.5 else "not_crop" for xx in all_preds])
    else:
        # metrics_agg = "macro"
        raise NotImplementedError("Croptype evaluation not implemented yet")

    results = classification_report(
        all_targets,
        all_preds,
        labels=croptype_list,
        output_dict=True,
        zero_division=0,
    )

    results_df = pd.DataFrame(results).transpose().reset_index()
    results_df.columns = pd.Index(
        ["class", "precision", "recall", "f1-score", "support"]
    )

    return results_df


if __name__ == "__main__":
    # ------------------------------------------
    # Parameter settings (can become argparser)
    # ------------------------------------------

    # Path to the training data
    parquet_file = "/home/vito/vtrichtk/projects/worldcereal/data/worldcereal_training_data_monthly.parquet/worldcereal_training_data.parquet"
    val_samples_file = "/home/vito/vtrichtk/git/worldcereal-classification/scripts/training/finetuning/cropland_random_generalization_test_split_samples.csv"
    # val_samples_file = None  # If None, random split is used

    # Experiment settings
    experiment_name = "presto-prometheo-cropland-finetune-run-1"
    output_dir = "./presto-prometheo-cropland-finetune-run-1"
    debug = False
    augment = True
    time_explicit = False
    # balance = True  # Not implemented yet in the dataset
    # train_masking = 0.25  # Not implemented yet in the dataset
    # task_type = "cropland" # Not implemented yet in the dataset
    # finetune_classes = "CROPTYPE0"  # Not implemented yet in the dataset

    # Training parameters
    pretrained_model_path = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96_corrected-mask.pt"
    epochs = 100
    batch_size = 512
    patience = 5
    num_workers = 16

    # ------------------------------------------

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    initialize_logging(
        log_file=Path(output_dir) / "logs" / f"{experiment_name}.log",
        level="INFO",
        console_filter_keyword="PROGRESS",
    )

    # Get the train/val/test dataframes
    train_df, val_df, test_df = prepare_training_df(
        parquet_file, val_samples_file, debug=debug
    )

    # Construct training and validation datasets
    train_ds, val_ds, test_ds = prepare_training_datasets(
        train_df, val_df, test_df, augment=augment, time_explicit=time_explicit
    )

    # Construct the finetuning model based on the pretrained model
    model = PretrainedPrestoWrapper(
        num_outputs=1,
        regression=False,
        pretrained_model_path=pretrained_model_path,
    )

    # Define the loss function: with logits because no activation on output layer
    # is applied in Presto.
    loss_fn = nn.BCEWithLogitsLoss()

    # Set the parameters
    hyperparams = Hyperparams(
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        num_workers=num_workers,
    )
    parameters = param_groups_lrd(model)
    optimizer = AdamW(parameters, lr=hyperparams.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Run the finetuning
    logger.info("Starting finetuning...")
    finetuned_model = run_finetuning(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
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
    eval_results = evaluate_finetuned_model(
        finetuned_model, test_ds, num_workers, batch_size
    )
    eval_results.to_csv(
        Path(output_dir) / f"results_{experiment_name}.csv", index=False
    )
    logger.info("Evaluation results:")
    logger.info("\n" + eval_results.to_string(index=False))
    logger.info("Finetuning completed!")
