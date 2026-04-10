"""Utility functions to generate Presto embeddings and train CatBoost classifiers.

This module provides a light-weight pipeline to:

1. Align raw reference data extractions to a user defined season (``TemporalContext``).
2. Generate 128‑D Presto embeddings (either globally pooled or time‑explicit at the valid
     timestep) ready for downstream ML.
3. Train and evaluate a CatBoost classifier (multiclass crop type or binary cropland).

Notes
-----
The embedding dimensionality is currently fixed at 128 (``presto_ft_0`` .. ``presto_ft_127``).
If the upstream Presto model changes dimensionality this file should be updated accordingly.
"""

from pathlib import Path
from typing import Any, Literal, Optional, Union

import pandas as pd
from loguru import logger
from openeo_gfmap import TemporalContext
from prometheo.utils import DEFAULT_SEED
from tabulate import tabulate

from worldcereal.train.backbone import (
    build_presto_backbone,
    checkpoint_fingerprint,
    resolve_seasonal_encoder,
)
from worldcereal.train.datasets import SensorMaskingConfig
from worldcereal.utils.refdata import process_extractions_df


def get_input(label):
    """Prompt user for a short identifier without spaces.

    Parameters
    ----------
    label : str
        Human readable description of what is being named (e.g. 'model').

    Returns
    -------
    str
        User provided identifier guaranteed not to contain whitespace.
    """
    while True:
        modelname = input(f"Enter a short name for your {label} (don't use spaces): ")
        if " " not in modelname:
            return modelname
        print("Invalid input. Please enter a name without spaces.")


def _load_presto_encoder(custom_presto_url: Optional[str] = None):
    """Load the Presto encoder using the same helper as the training scripts."""

    if custom_presto_url:
        weights_path = custom_presto_url
        if str(weights_path).startswith(("http://", "https://")):
            fingerprint = "custom-url"
        else:
            try:
                fingerprint = checkpoint_fingerprint(weights_path)
            except (FileNotFoundError, OSError):
                logger.warning(
                    f"Could not compute fingerprint for {weights_path}; falling back to 'custom'"
                )
                fingerprint = "custom"
    else:
        weights_path, fingerprint = resolve_seasonal_encoder()

    logger.info(f"Loading Presto encoder (fingerprint={fingerprint})")
    presto_model = build_presto_backbone(checkpoint_path=weights_path)
    return presto_model, fingerprint


def align_extractions_to_season(
    df: pd.DataFrame,
    season: Optional[TemporalContext] = None,
    freq: Literal["month", "dekad"] = "month",
    valid_time_buffer: int = 0,
    season_window: Optional[TemporalContext] = None,
) -> pd.DataFrame:
    """Align raw extraction rows to a target season and enrich with labels.

    When processing_period (season) is provided, samples must have complete satellite
    coverage for that 12-month window. When omitted (None), only season_window filtering
    is applied, allowing samples with partial temporal coverage.

    Samples are removed if:
    - (If season provided) They lack satellite coverage for the full processing period
    - Their valid_time falls outside the season window
    - (If season provided) Their valid_time is too close to processing period edges

    Output additions
    ----------------
    The returned DataFrame preserves original attributes and adds:

    * ``year``: integer year parsed from ``ref_id`` (first token before underscore).
    * ``label_full``: human readable crop / land cover label.
    * ``sampling_label``: label variant used for stratified splitting.
    * All samples have exactly 12 monthly (or 36 dekadal) timesteps
    TODO: the above does not seem to be enforced if buffer > 0

    Parameters
    ----------
    df : pandas.DataFrame
        Raw extraction rows containing at minimum ``ref_id``, ``ewoc_code``, ``timestamp``,
        and ``valid_time``.
    season : TemporalContext, optional
        Processing period (12 consecutive months). Required for trimming samples to exactly
        12 timesteps (needed for embedding computation). When None, no trimming occurs.
    freq : {'month', 'dekad'}, default='month'
        Resampling / alignment frequency controlling internal temporal aggregation.
    valid_time_buffer : int, default=0
        Buffer (in months for monthly freq) allowing valid_time closer to processing period edges.
    season_window : TemporalContext, optional
        Campaign window for valid_time filtering (typically the slider selection).
        When provided, only samples with valid_time inside this window are retained.

    Returns
    -------
    pandas.DataFrame
        Aligned samples with additional ``year``, ``label_full`` and ``sampling_label``
        columns. All samples have exactly ``num_timesteps`` observations (12 for monthly,
        36 for dekadal). A warning is logged if fewer than two unique ``ewoc_code`` values
        remain (training feasibility issue).

    Notes
    -----
    Call :func:`compute_seasonal_presto_embeddings` next to generate embedding features,
    then :func:`train_seasonal_torch_head` to train a model on them.
    """
    from worldcereal.utils.legend import ewoc_code_to_label

    # Align the samples with the season of interest
    df = process_extractions_df(
        df,
        season,
        freq,
        valid_time_buffer,
        season_window=season_window,
    )

    # Report on contents of the resulting dataframe here
    logger.info(
        f"Samples originating from {df['ref_id'].nunique()} unique reference datasets."
    )

    logger.info("Distribution of samples across years:")
    # extract year from ref_id
    df["year"] = df["ref_id"].str.split("_").str[0].astype(int)
    logger.info(
        "\n"
        + tabulate(
            df["year"].value_counts().reset_index(),
            headers=["Year", "Count"],
            tablefmt="psql",
            showindex=False,
        )
    )

    # Get crop statistics
    ncroptypes = df["ewoc_code"].nunique()
    logger.info(f"Number of crop types remaining: {ncroptypes}")
    if ncroptypes <= 1:
        error_msg = (
            "Not enough crop types found in the remaining data to train a model, cannot continue with model training."
            " Consider adjusting the season window to retain more samples."
        )
        raise RuntimeError(error_msg)

    # Enrich resulting dataframe with full and sampling string labels
    df["label_full"] = ewoc_code_to_label(df["ewoc_code"], label_type="full")
    df["sampling_label"] = ewoc_code_to_label(df["ewoc_code"], label_type="sampling")

    return df.reset_index()


def compute_seasonal_presto_embeddings(
    df: pd.DataFrame,
    *,
    season_id: str,
    batch_size: int = 256,
    task_type: str = "croptype",
    augment: bool = False,
    mask_on_training: bool = True,
    repeats: int = 3,
    custom_presto_url: Optional[str] = None,
    season_calendar_mode: Literal["auto", "calendar", "custom", "off"] = "calendar",
    season_window: Optional[TemporalContext] = None,
    min_season_coverage: float = 1.0,
    use_spatial_split: bool = True,
    bin_size_degrees: float = 0.25,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    """Compute Presto embeddings pooled over a single season selection.

    When ``season_window`` is provided, the supplied ``TemporalContext`` defines the
    pooling window so custom campaign identifiers can be used without relying on the
    global seasonality lookup.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with temporally aligned samples.
    season_id : str
        Identifier for the season to use for pooling.
    batch_size : int, default=256
        Inference batch size.
    task_type : {'croptype', 'cropland'}, default='croptype'
        Selects pretrained weights and multiclass vs binary configuration.
    augment : bool, default=True
        Whether to apply temporal jitter augmentation.
    mask_on_training : bool, default=True
        Whether to apply sensor masking augmentations on the training set.
    repeats : int, default=3
        Number of times to repeat each sample in the training set.
    custom_presto_url : str, optional
        URL to custom Presto model weights.
    season_calendar_mode : {'auto', 'calendar', 'custom', 'off'}, default='calendar'
        Season calendar resolution mode.
    season_window : TemporalContext, optional
        Custom temporal window for pooling.
    min_season_coverage : float, optional
        Minimum fraction of a season's composite slots that must fall inside
        the selected timestep window for the season mask to be enabled.
        1.0 (default) enforces full coverage.
    use_spatial_split : bool, default=False
        If True, uses spatial binning to split data and avoid spatial leakage.
        Requires 'lat' and 'lon' columns in the dataframe.
    bin_size_degrees : float, default=0.25
        Size of spatial bins in degrees (used when use_spatial_split=True).
    val_size : float, default=0.15
        Fraction of data (or spatial bins) to use for validation.
    test_size : float, default=0.15
        Fraction of data (or spatial bins) to use for testing.

    Returns
    -------
    pandas.DataFrame
        Dataframe with Presto embeddings and a 'split' column indicating
        train/val/test assignment.
    """

    from worldcereal.train.data import (
        dataset_to_embeddings,
        spatial_train_val_test_split,
        train_val_test_split,
    )
    from worldcereal.train.datasets import WorldCerealTrainingDataset

    if task_type not in {"croptype", "cropland"}:
        raise ValueError(
            f"Unknown task type: `{task_type}`. Only 'croptype' and 'cropland' are supported."
        )

    presto_model, presto_fingerprint = _load_presto_encoder(custom_presto_url)
    logger.info(f"Presto encoder fingerprint: {presto_fingerprint}")

    # Use existing splitting utilities from worldcereal.train.data
    if use_spatial_split:
        samples_train, samples_val, samples_test = spatial_train_val_test_split(
            df,
            val_size=val_size,
            test_size=test_size,
            seed=DEFAULT_SEED,
            bin_size_degrees=bin_size_degrees,
            stratify_label="downstream_class",
        )
    else:
        samples_train, samples_val, samples_test = train_val_test_split(
            df,
            val_size=val_size,
            test_size=test_size,
            seed=DEFAULT_SEED,
            stratify_label="downstream_class",
        )

    if mask_on_training:
        masking_config = SensorMaskingConfig(
            enable=True,
            s1_full_dropout_prob=0.05,
            s1_timestep_dropout_prob=0.1,
            s2_cloud_timestep_prob=0.1,
            s2_cloud_block_prob=0.05,
            s2_cloud_block_min=2,
            s2_cloud_block_max=3,
            meteo_timestep_dropout_prob=0.03,
            dem_dropout_prob=0.01,
        )
    else:
        masking_config = SensorMaskingConfig(enable=False)

    season_windows = None
    if season_window is not None:
        start_ts = pd.Timestamp(season_window.start_date).to_pydatetime()
        end_ts = pd.Timestamp(season_window.end_date).to_pydatetime()
        season_windows = {season_id: (start_ts, end_ts)}
    effective_mode = (
        "custom"
        if season_windows and season_calendar_mode == "calendar"
        else season_calendar_mode
    )

    def _build_dataset(
        frame: pd.DataFrame,
        augment_flag: bool,
        masking_config: Optional[SensorMaskingConfig] = None,
    ) -> WorldCerealTrainingDataset:
        return WorldCerealTrainingDataset(
            frame.reset_index(),
            task_type="multiclass" if task_type == "croptype" else "binary",
            augment=augment_flag,
            masking_config=masking_config,
            repeats=repeats if (augment_flag or masking_config) else 1,
            season_ids=[season_id],
            season_calendar_mode=effective_mode,
            season_windows=season_windows,
            min_season_coverage=min_season_coverage,
        )

    train_ds = _build_dataset(
        samples_train, augment_flag=augment, masking_config=masking_config
    )
    val_ds = _build_dataset(samples_val, augment_flag=False)
    test_ds = _build_dataset(samples_test, augment_flag=False)

    df_train = dataset_to_embeddings(
        train_ds, presto_model, batch_size=batch_size, season_index=0
    )
    df_val = dataset_to_embeddings(
        val_ds, presto_model, batch_size=batch_size, season_index=0
    )
    df_test = dataset_to_embeddings(
        test_ds, presto_model, batch_size=batch_size, season_index=0
    )

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    return pd.concat([df_train, df_val, df_test]).reset_index(drop=True)


def train_seasonal_torch_head(
    training_dataframe: pd.DataFrame,
    *,
    season_id: str,
    head_task: Literal["croptype", "landcover"] = "croptype",
    output_dir: Union[str, Path] = "./downstream_classifier",
    num_workers: int = 0,
    disable_progressbar: bool = True,
    **trainer_kwargs: Any,
):
    """Train a torch head compatible with the seasonal model bundle."""

    from worldcereal.train.downstream import TorchTrainer

    trainer = TorchTrainer(
        training_dataframe,
        head_task=head_task,
        output_dir=output_dir,
        season_id=season_id,
        num_workers=num_workers,
        disable_progressbar=disable_progressbar,
        **trainer_kwargs,
    )
    return trainer.train()
