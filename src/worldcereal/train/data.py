import gc
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from loguru import logger
from prometheo.models import Presto
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import NODATAVALUE, Predictors
from prometheo.utils import device
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from worldcereal.train.datasets import WorldCerealTrainingDataset
from worldcereal.utils.refdata import get_class_mappings, map_classes, split_df
from worldcereal.utils.timeseries import process_parquet

_ATTR_KEYS_ALLOW_PARTIAL_NONE = {
    "landcover_label",
    "croptype_label",
    "label_task",
}


def collate_fn(batch: Sequence[Tuple[Predictors, dict]]):
    predictor_dicts = [item.as_dict(ignore_nones=True) for item, _ in batch]
    collated_dict = default_collate(predictor_dicts)
    predictors = Predictors(**collated_dict)

    attrs_list = [attrs for _, attrs in batch]
    collated_attrs = _collate_attrs(attrs_list)

    return predictors, collated_attrs


def _collate_attrs(attrs_list: Sequence[dict]) -> dict:
    if not attrs_list:
        return {}

    collated: Dict[str, Any] = {}
    all_keys = set().union(*(attrs.keys() for attrs in attrs_list))
    for key in all_keys:
        values = [attrs.get(key) for attrs in attrs_list]

        if key in {"season_masks", "in_seasons"}:
            if all(v is None for v in values):
                collated[key] = None
                continue

            first = next(v for v in values if v is not None)
            filler = (
                np.ones_like(first, dtype=bool)
                if key == "season_masks"
                else np.zeros_like(first, dtype=bool)
            )
            stacked = [
                np.asarray(v, dtype=bool) if v is not None else filler for v in values
            ]
            collated[key] = np.stack(stacked, axis=0)
            continue

        if all(v is None for v in values):
            collated[key] = None
            continue

        if any(v is None for v in values):
            if key in _ATTR_KEYS_ALLOW_PARTIAL_NONE:
                # Keep per-sample values so downstream helpers can handle missing labels.
                collated[key] = values
                continue
            missing_indices = [i for i, v in enumerate(values) if v is None]
            raise ValueError(
                f"_collate_attrs received None values for key '{key}' at indices {missing_indices}"
            )

        collated[key] = default_collate(values)

    return collated


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
        attrs_frame = {
            k: v for k, v in attrs.items() if k not in ("season_masks", "in_seasons")
        }
        attrs_df = pd.DataFrame.from_dict(attrs_frame)
        encodings = pd.DataFrame(
            encodings, columns=[f"presto_ft_{i}" for i in range(encodings.shape[1])]
        )
        result = pd.concat([encodings, attrs_df], axis=1)

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


def duckdb_type_from_series(s: pd.Series) -> str:
    """
    Infer a DuckDB column type from a pandas Series.
    We'll be conservative:
    - ints -> BIGINT
    - floats -> DOUBLE
    - bool -> BOOLEAN
    - everything else -> TEXT
    """
    if pd.api.types.is_integer_dtype(s):
        return "BIGINT"
    if pd.api.types.is_float_dtype(s):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(s):
        return "BOOLEAN"
    # timestamps: let DuckDB infer TIMESTAMP if we detect datetime64
    if pd.api.types.is_datetime64_any_dtype(s):
        return "TIMESTAMP"
    return "TEXT"


def get_table_columns(con, table_name):
    """Return current column names (ordered) from the DuckDB table."""
    info_df = con.execute(f"PRAGMA table_info('{table_name}')").df()
    return list(info_df["name"])


def add_missing_columns_to_table(con, table_name, batch_df, current_cols, nodata_value):
    """
    For each column in batch_df that is NOT yet in table_name:
    ALTER TABLE ... ADD COLUMN that_col <type> DEFAULT NODATAVALUE (or NULL).
    If the column looks numeric/bool/datetime, we add that sql type.
    If we add a numeric column with DEFAULT NODATAVALUE, older rows get the fill.
    """

    new_cols = [c for c in batch_df.columns if c not in current_cols]

    for col in new_cols:
        col_type = duckdb_type_from_series(batch_df[col])

        # Decide default value depending on type.
        # For numeric types we can use NODATAVALUE.
        # For non-numeric types, use NULL default so we don't shove NODATAVALUE into text/timestamps.
        if col_type in ("BIGINT", "DOUBLE"):
            default_expr = str(nodata_value)
        elif col_type == "BOOLEAN":
            default_expr = "FALSE"
        elif col_type == "TIMESTAMP":
            default_expr = "NULL"
        else:
            default_expr = "NULL"

        # Quote column name if it has weird chars like '-' :
        quoted_col = f'"{col}"'

        alter_sql = (
            f"ALTER TABLE {table_name} "
            f"ADD COLUMN {quoted_col} {col_type} DEFAULT {default_expr};"
        )
        con.execute(alter_sql)

    # Return updated list of columns from the table after ALTERs
    return get_table_columns(con, table_name)


def align_batch_to_table_columns(batch_df, table_cols, nodata_value):
    """
    Ensure batch_df has exactly all columns in table_cols.
    - Add missing cols with nodata_value (or NaN for non-numeric).
    - Reorder to table_cols.
    """

    # Add any columns that exist in table_cols but not in batch_df
    missing_for_batch = [c for c in table_cols if c not in batch_df.columns]
    for col in missing_for_batch:
        batch_df[col] = nodata_value

    # Reorder columns to match table
    batch_df = batch_df[table_cols]
    return batch_df


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
        # select first 3 files in debug mode
        parquet_files = parquet_files[:3]
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
        "ref_id",
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
                # Make sure next batch aligns with table schema:
                # 1. Get current columns from table
                table_cols = get_table_columns(con, table_name)
                # 2. If batch has NEW columns, ALTER TABLE to add them w/ defaults
                table_cols = add_missing_columns_to_table(
                    con,
                    table_name,
                    _data_pivot,
                    table_cols,
                    NODATAVALUE,
                )
                # 3. Align batch df to table columns (add any columns the table has that batch doesn't)
                _data_pivot = align_batch_to_table_columns(
                    _data_pivot,
                    table_cols,
                    NODATAVALUE,
                )
                # 4. Register and insert
                con.register("pivot_batch", _data_pivot)
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

        if not initialized:
            raise RuntimeError(
                "No parquet files were processed; the wide table was never created. "
                "Ensure the `parquet_files` list is not empty and points to readable files."
            )

        # write a single Parquet file
        con.execute(
            f"""
            COPY (SELECT * FROM {table_name})
            TO '{wide_parquet_output_path}'
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
        )

    # Load the merged Parquet -> use pyarrow for efficient chunked reading
    logger.info(f"Loading wide parquet file from {wide_parquet_output_path} ...")
    pf = pq.ParquetFile(wide_parquet_output_path)
    parts = []
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i)  # arrow Table (usually lower overhead than pandas)
        parts.append(table.to_pandas())  # convert one chunk at a time
    df = pd.concat(parts, ignore_index=True)

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
