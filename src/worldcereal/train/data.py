from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from loguru import logger
from prometheo.models import Presto
from prometheo.predictors import Predictors
from prometheo.utils import device
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from worldcereal.train.datasets import WorldCerealDataset
from worldcereal.utils.refdata import get_class_mappings, map_classes, split_df
from worldcereal.utils.timeseries import process_parquet


class WorldCerealTrainingDataset(WorldCerealDataset):
    def __getitem__(self, idx):
        # Get the sample
        sample = super().__getitem__(idx)
        row = self.dataframe.iloc[idx, :]

        attrs = [
            "lat",
            "lon",
            "ref_id",
            "sample_id",
            "downstream_class",
        ]

        attrs = [attr for attr in attrs if attr in row.index]

        return sample, row[attrs].to_dict()


def collate_fn(batch: Sequence[Tuple[Predictors, dict]]):
    # we assume that the same values are consistently None
    collated_dict = default_collate([i.as_dict(ignore_nones=True) for i, _ in batch])
    collated_attrs = default_collate([attrs for _, attrs in batch])
    return Predictors(**collated_dict), collated_attrs


def get_training_df(
    dataset: WorldCerealTrainingDataset,
    presto_model: Presto,
    batch_size: int = 2048,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Function to extract Presto embeddings, targets and relevant
    auxiliary attributes from a dataset.

    Parameters
    ----------
    dataset : WorldCerealTrainingDataset
        dataset to extract embeddings from
    presto_model : Presto
        presto model to use for extracting embeddings
    batch_size : int, optional
        by default 2048
    num_workers : int, optional
        number of workers to use in DataLoader, by default 0

    Returns
    -------
    pd.DataFrame
        training dataframe that can be used for training downstream classifier
    """

    # Make sure model is in eval mode and moved to the correct device
    presto_model.eval().to(device)

    # Create dataloader
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Initialize final dataframe
    final_df = None

    # Iterate through dataloader to consume all samples
    for predictors, attrs in tqdm(dl):
        # Compute Presto embeddings
        with torch.no_grad():
            encodings = presto_model(predictors).cpu().numpy().reshape((-1, 128))

        # Convert to dataframe
        attrs = pd.DataFrame.from_dict(attrs)
        encodings = pd.DataFrame(
            encodings, columns=[f"presto_ft_{i}" for i in range(encodings.shape[1])]
        )
        result = pd.concat([encodings, attrs], axis=1)

        # Append to final dataframe
        final_df = result if final_df is None else pd.concat([final_df, result])

    return final_df


def get_training_dfs_from_parquet(
    parquet_files: Union[Union[Path, str], List[Union[Path, str]]],
    timestep_freq: Literal["month", "dekad"] = "month",
    finetune_classes: str = "CROPLAND2",
    class_mappings: Dict[str, Dict[str, str]] = get_class_mappings(),
    val_samples_file: Optional[Union[Path, str]] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare training, validation, and test DataFrames from parquet files for presto model fine-tuning.

    This function reads parquet files containing time series data, processes them into a wide format,
    maps the classes according to the specified fine-tuning target, and splits the data into train,
    validation, and test sets.

    Parameters
    ----------
    parquet_files : List[Union[Path, str]]
        List of local paths to parquet files.
    timestep_freq : str, default="month"
        Frequency of timesteps. Can be "month" or "dekad".
    finetune_classes (str):
        The set of fine-tuning classes to use from CLASS_MAPPINGS.
        This should be one of the keys in CLASS_MAPPINGS.
        Most popular maps: "LANDCOVER14", "CROPTYPE9", "CROPTYPE0", "CROPLAND2".
        Defaults to "CROPLAND2".
    class_mappings (dict, optional):
            Dictionary containing the mapping of original class codes to new class labels.
    val_samples_file : Optional[Union[Path, str]], default=None
        Path to a CSV file containing sample IDs for controlled validation set selection.
        If provided, the test set will be constructed using these sample IDs.
        If None, a random train/test split will be performed.
    debug : bool, default=False
        If True, a maximum of one file will be processed for quick testing.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three DataFrames:
        - train_df: DataFrame with training samples
        - val_df: DataFrame with validation samples
        - test_df: DataFrame with test samples
    """
    logger.info("Reading dataset")

    if isinstance(parquet_files, (str, Path)):
        # If a single file is provided, convert it to a list
        parquet_files = [parquet_files]

    if debug:
        # select 1st file in debug mode
        parquet_files = parquet_files[:1]
        logger.warning("Debug mode is enabled.")

    df = None
    for f in parquet_files:
        logger.info(f"Processing {f}")
        _data = pd.read_parquet(f, engine="fastparquet")
        _data = _data[_data["sample_id"].notnull()]
        _data["ewoc_code"] = _data["ewoc_code"].astype(int)

        for tcol in ["valid_time", "start_time", "end_time", "timestamp"]:
            if tcol in _data.columns:
                _data[tcol] = pd.to_datetime(_data[tcol], utc=True)
                _data[tcol] = _data[tcol].dt.tz_localize(None)

        _data_pivot = process_parquet(_data, freq=timestep_freq)
        _data_pivot.reset_index(inplace=True)
        df = _data_pivot if df is None else pd.concat([df, _data_pivot])

    df = map_classes(df, finetune_classes, class_mappings=class_mappings)

    # Remove classes with too few samples for stratification
    # For stratified split, each class must have at least 2 samples for test split, and at least 2 for val split.
    # We'll use a minimum of 5 per class for safety.
    min_samples_per_split = 5
    class_counts = df["finetune_class"].value_counts()
    minor_classes = class_counts[class_counts < min_samples_per_split].index.tolist()
    if minor_classes:
        logger.warning(
            f"The following classes have fewer than {min_samples_per_split} samples and will be removed for stratified splitting: {minor_classes}. "
            f"Samples removed: {df[df['finetune_class'].isin(minor_classes)].shape[0]}"
        )
        df = df[~df["finetune_class"].isin(minor_classes)].copy()
        # After removal, check again for any classes with too few samples
        class_counts = df["finetune_class"].value_counts()
        if (class_counts < min_samples_per_split).any():
            logger.error(
                "Some classes still have too few samples after removal. Consider increasing your dataset or lowering min_samples_per_split."
            )

    if val_samples_file is not None:
        logger.info(f"Controlled train/test split based on: {val_samples_file}")
        val_samples_df = pd.read_csv(val_samples_file)
        trainval_df, test_df = split_df(
            df, val_sample_ids=val_samples_df.sample_id.tolist()
        )
    else:
        logger.info("Random train/test split ...")
        # train_df, test_df = split_df(df, val_size=0.2)
        # TO DO: add possibility of per-class stratification to original split_df function
        trainval_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["finetune_class"]
        )

    # train_df, val_df = split_df(train_df, val_size=0.2)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.2,
        random_state=42,
        stratify=trainval_df["finetune_class"],
    )

    return train_df, val_df, test_df
