import gc
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
import torch
from loguru import logger
from prometheo.models import Presto
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import NODATAVALUE, Predictors
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
        timestep_positions, valid_position = self.get_timestep_positions(row)
        valid_position = valid_position - timestep_positions[0]
        attrs = [
            "lat",
            "lon",
            "ref_id",
            "sample_id",
            "downstream_class",
            "valid_time",
        ]

        attrs = [attr for attr in attrs if attr in row.index]
        attrs = row[attrs].to_dict()
        attrs["valid_position"] = valid_position

        return sample, attrs


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
    time_explicit: bool = False,
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
    time_explicit : bool, optional
        Switch from globally pooled sequence embeddings to
        valid timestep embeddings, by default False

    Returns
    -------
    pd.DataFrame
        training dataframe that can be used for training downstream classifier
    """

    if time_explicit:
        logger.info("Computing time-explicit Presto embeddings ...")
    else:
        logger.info("Computing time-agnostic Presto embeddings ...")

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
            if not time_explicit:
                encodings = presto_model(predictors).cpu().numpy().reshape((-1, 128))
            else:
                encodings = (
                    presto_model(predictors, eval_pooling=PoolingMethods.TIME)
                    .cpu()
                    .numpy()
                    .reshape((-1, dataset.num_timesteps, 128))
                )
                # Cut out the correct position in the time series
                encodings = encodings[
                    np.arange(encodings.shape[0]), attrs["valid_position"], :
                ]

        # Convert to dataframe
        attrs = pd.DataFrame.from_dict(attrs)
        encodings = pd.DataFrame(
            encodings, columns=[f"presto_ft_{i}" for i in range(encodings.shape[1])]
        )
        result = pd.concat([encodings, attrs], axis=1)

        # Append to final dataframe
        final_df = result if final_df is None else pd.concat([final_df, result])

    return final_df


def remove_small_classes(df, min_samples):
    # Remove classes with too few samples for stratification
    # For stratified split, each class must have at least 2 samples for test split, and at least 2 for val split.
    # By default we'll use a minimum of 5 per class for safety.

    class_counts = df["finetune_class"].value_counts()
    minor_classes = class_counts[class_counts < min_samples].index.tolist()
    if minor_classes:
        logger.warning(
            f"The following classes have fewer than {min_samples} samples and will be removed for stratified splitting: {minor_classes}. "
            f"Samples removed: {df[df['finetune_class'].isin(minor_classes)].shape[0]}"
        )
        df = df[~df["finetune_class"].isin(minor_classes)].copy()
        # After removal, check again for any classes with too few samples
        class_counts = df["finetune_class"].value_counts()
        if (class_counts < min_samples).any():
            logger.error(
                "Some classes still have too few samples after removal. Consider increasing your dataset or lowering min_samples_per_split."
            )
    return df


def get_training_dfs_from_parquet(
    parquet_files: Union[Union[Path, str], List[Union[Path, str]]],
    timestep_freq: Literal["month", "dekad"] = "month",
    max_timesteps_trim: Union[str, int, tuple] = "auto",
    use_valid_time: bool = True,
    finetune_classes: str = "CROPLAND2",
    class_mappings: Dict[str, Dict[str, str]] = get_class_mappings(),
    val_samples_file: Optional[Union[Path, str]] = None,
    test_samples_file: Optional[Union[Path, str]] = None,
    debug: bool = False,
    overwrite: bool = False,
    wide_parquet_output_path: Optional[Union[Path, str]] = None,
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
    max_timesteps_trim : Union[str, int, tuple], default="auto"
        Maximum number of timesteps to retain after trimming.
        If "auto", it will be determined based on the timestep_freq and MIN_EDGE_BUFFER.
    use_valid_time : bool, default=True
        Whether to use the 'valid_time' column for processing timesteps.
        If True, centering and filtering of samples will be based on 'valid_time'.
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
    overwrite : bool, default=False
        If True, overwrite existing wide parquet file.
    wide_parquet_output_path : Union[Path, str], optional
        Path to save the processed wide-format parquet file.
        If None, a temporary file is created, used internally, and deleted before return.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three DataFrames:
        - train_df: DataFrame with training samples
        - val_df: DataFrame with validation samples
        - test_df: DataFrame with test samples
    """
    logger.info("Reading dataset")

    is_tempfile = False  # <-- track if we must clean up
    if wide_parquet_output_path is None:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            suffix=".parquet", prefix="wide_", delete=False
        )
        wide_parquet_output_path = Path(tmp.name)
        tmp.close()
        is_tempfile = True
        logger.warning(
            f"No wide_parquet_output_path provided; using temporary file: {wide_parquet_output_path}"
        )
    else:
        wide_parquet_output_path = Path(wide_parquet_output_path)
        logger.info(
            f"Using provided wide parquet output path: {wide_parquet_output_path}"
        )

    if isinstance(parquet_files, (str, Path)):
        # If a single file is provided, convert it to a list
        parquet_files = [parquet_files]

    if debug:
        # select 1st file in debug mode
        parquet_files = parquet_files[:1]
        logger.warning("Debug mode is enabled.")

    db = duckdb.connect()
    db.sql("INSTALL spatial")
    db.load_extension("spatial")

    STRING_COLS = [
        "sample_id",
        "timestamp",
        "h3_l3_cell",
        "valid_time",
        "start_date",
        "end_date",
    ]
    INT_COLS = [
        "extract",
        "quality_score_ct",
        "quality_score_lc",
        "ewoc_code",
        "S2-L2A-B02",
        "S2-L2A-B03",
        "S2-L2A-B04",
        "S2-L2A-B05",
        "S2-L2A-B06",
        "S2-L2A-B07",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B11",
        "S2-L2A-B12",
        "S1-SIGMA0-VH",
        "S1-SIGMA0-VV",
        "slope",
        "elevation",
        "AGERA5-PRECIP",
        "AGERA5-TMEAN",
    ]
    FLOAT_COLS = ["lon", "lat"]
    REQUIRED_COLS = STRING_COLS + INT_COLS + FLOAT_COLS

    if overwrite or is_tempfile or not wide_parquet_output_path.exists():
        logger.info(
            f"Creating wide parquet file at {wide_parquet_output_path} (overwrite={overwrite})"
        )
        db_path = Path(f"{str(wide_parquet_output_path).split('.')[0]}.duckdb")
        table_name = "merged_parquets_wide"
        wide_parquet_output_path.unlink(missing_ok=True)
        db_path.unlink(missing_ok=True)

        con = duckdb.connect(db_path)
        con.execute("PRAGMA memory_limit='4GB'")
        # If the DB existed and table might be present, ensure a clean start:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        initialized = False
        for f in tqdm(parquet_files, desc="Processing long parquet files"):
            _data = pd.read_parquet(f, engine="fastparquet")
            _ref_id = Path(f).stem
            _data["ref_id"] = _ref_id
            _data = _data[REQUIRED_COLS]
            _data_pivot = process_parquet(
                _data,
                freq=timestep_freq,
                use_valid_time=use_valid_time,
                max_timesteps_trim=max_timesteps_trim,
            )
            _data_pivot = _data_pivot.reset_index()
            _data_pivot = _data_pivot.fillna(NODATAVALUE)
            con.register("pivot_batch", _data_pivot)
            if not initialized:
                con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM pivot_batch")
                initialized = True
            else:
                con.execute(
                    f"INSERT INTO {table_name} BY NAME SELECT * FROM pivot_batch"
                )
            con.unregister("pivot_batch")
            # --- force flush to disk & keep WAL small ---
            con.execute(
                "CHECKPOINT"
            )  # forces a checkpoint so data is on disk, WAL rotated
            # --- drop big Python objects ASAP ---
            del _data_pivot, _data
            gc.collect()

        # write a single Parquet file
        con.execute(f"""
            COPY (SELECT * FROM {table_name})
            TO '{wide_parquet_output_path}'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)

    # Load the merged Parquet
    df = pd.read_parquet(wide_parquet_output_path)

    df = map_classes(df, finetune_classes, class_mappings=class_mappings)

    # Remove classes with too few samples for stratification
    df = remove_small_classes(df, min_samples=10)

    if test_samples_file is not None:
        logger.info(
            f"Controlled `train/val` vs `test` split based on: {test_samples_file}"
        )
        test_samples_df = pd.read_csv(test_samples_file)
        trainval_df, test_df = split_df(
            df, val_sample_ids=test_samples_df.sample_id.tolist()
        )
    else:
        logger.info("Random `train/val` vs `test` split ...")
        # train_df, test_df = split_df(df, val_size=0.2)
        # TO DO: add possibility of per-class stratification to original split_df function
        trainval_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["finetune_class"]
        )

    # train_df, val_df = split_df(train_df, val_size=0.2)
    # Remove classes with too few samples for stratification, now on trainval_df
    trainval_df = remove_small_classes(trainval_df, min_samples=5)

    if val_samples_file is not None:
        logger.info(f"Controlled `train` vs `val` split based on: {val_samples_file}")
        val_samples_df = pd.read_csv(val_samples_file)
        train_df, val_df = split_df(
            trainval_df, val_sample_ids=val_samples_df.sample_id.tolist()
        )
    else:
        logger.info("Random `train` vs `val` split ...")
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.2,
            random_state=42,
            stratify=trainval_df["finetune_class"],
        )

    if test_samples_file:
        # With controlled test set it's possible that either
        # the test set has unique classes not present in training
        # So we need to remove those classes in its totality
        train_classes = set(train_df["finetune_class"].unique())
        val_classes = set(val_df["finetune_class"].unique())
        test_classes = set(test_df["finetune_class"].unique())
        nontrainval_classes = test_classes - (train_classes | val_classes)

        test_df = test_df[~test_df["finetune_class"].isin(nontrainval_classes)]

        if len(nontrainval_classes) > 0:
            logger.warning(
                "Removed classes from test set because they "
                f"do not occur train/val: {nontrainval_classes}"
            )

    # Cleanup temporary files if created
    if is_tempfile:
        try:
            wide_parquet_output_path.unlink(missing_ok=True)
            db_path = Path(f"{str(wide_parquet_output_path).split('.')[0]}.duckdb")
            db_path.unlink(missing_ok=True)
            logger.info(
                f"Deleted temporary wide parquet file: {wide_parquet_output_path}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to delete temporary file {wide_parquet_output_path}: {e}"
            )

    return train_df, val_df, test_df
