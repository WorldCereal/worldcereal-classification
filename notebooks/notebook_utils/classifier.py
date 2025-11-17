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

from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger
from openeo_gfmap import TemporalContext
from prometheo.utils import DEFAULT_SEED
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from worldcereal.parameters import CropLandParameters, CropTypeParameters
from worldcereal.train.datasets import MIN_EDGE_BUFFER, SensorMaskingConfig
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


def align_extractions_to_season(
    df: pd.DataFrame,
    season: TemporalContext,
    freq: Literal["month", "dekad"] = "month",
    valid_time_buffer: int = MIN_EDGE_BUFFER,
) -> pd.DataFrame:
    """Align raw extraction rows to a target season and enrich with labels.

    This function performs the temporal / metadata alignment step. After alignment
    it reports basic dataset diagnostics and adds helper label columns.

    Output additions
    ----------------
    The returned DataFrame preserves original attributes and adds:

    * ``year``: integer year parsed from ``ref_id`` (first token before underscore).
    * ``label_full``: human readable crop / land cover label.
    * ``sampling_label``: label variant used for stratified splitting.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw extraction rows containing at minimum ``ref_id`` and ``ewoc_code``.
    season : TemporalContext
        Target temporal context defining the modelling season.
    freq : {'month', 'dekad'}, default='month'
        Resampling / alignment frequency controlling internal temporal aggregation.
    valid_time_buffer : int, default=MIN_EDGE_BUFFER
        Safety buffer in frequency units between original valid time and season edges.

    Returns
    -------
    pandas.DataFrame
        Aligned samples with additional ``year``, ``label_full`` and ``sampling_label``
        columns. A warning is logged if fewer than two unique ``ewoc_code`` values
        remain (training feasibility issue).

    Notes
    -----
    Call :func:`compute_presto_embeddings` next to generate embedding features, then
    :func:`train_classifier` to train a model on them.
    """
    from worldcereal.utils.legend import ewoc_code_to_label

    # Align the samples with the season of interest
    df = process_extractions_df(df, season, freq, valid_time_buffer)

    # Report on contents of the resulting dataframe here
    logger.info(
        f"Samples originating from {df['ref_id'].nunique()} unique reference datasets."
    )

    logger.info("Distribution of samples across years:")
    # extract year from ref_id
    df["year"] = df["ref_id"].str.split("_").str[0].astype(int)
    logger.info(f"\n{df.year.value_counts()}")

    # Get crop statistics
    ncroptypes = df["ewoc_code"].nunique()
    logger.info(f"Number of crop types remaining: {ncroptypes}")
    if ncroptypes <= 1:
        logger.warning(
            "Not enough crop types found in the remaining data to train a model, cannot continue with model training!"
        )

    # Enrich resulting dataframe with full and sampling string labels
    df["label_full"] = ewoc_code_to_label(df["ewoc_code"], label_type="full")
    df["sampling_label"] = ewoc_code_to_label(df["ewoc_code"], label_type="sampling")

    return df.reset_index()


def compute_presto_embeddings(
    df: pd.DataFrame,
    batch_size: int = 256,
    task_type: str = "croptype",
    augment: bool = True,
    mask_on_training: bool = True,
    repeats: int = 3,
    time_explicit: bool = False,
    custom_presto_url: Optional[str] = None,
) -> pd.DataFrame:
    """Run pretrained *Presto* model to compute 128‑D embeddings for each sample.

    Parameters
    ----------
    df : pandas.DataFrame
        Temporally aligned dataframe (see :func:`compute_training_features`).
    batch_size : int, default=256
        Inference batch size.
    task_type : {'croptype', 'cropland'}, default='croptype'
        Selects pretrained weights and multiclass vs binary configuration.
    augment : bool, default=True
        Whether the underlying dataset applies temporal jitter.
    mask_on_training : bool, default=True
        Whether to apply sensor masking augmentations on the training set.
    repeats : int, default=3
        Number of times to repeat each sample in the training set (temporal / masking
        augmentation leverage). Should be >=1; values >1 increase effective dataset size.
    time_explicit : bool, default=False
        When ``True`` selects the embedding at ``valid_position`` instead of a pooled
        sequence representation.
    custom_presto_url : str, optional
        If provided, this URL overrides the default Presto model used to compute embeddings.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with columns ``presto_ft_0`` .. ``presto_ft_127`` appended.

    Raises
    ------
    ValueError
        If ``task_type`` is not one of the supported values.

    Notes
    -----
    This function *does not* modify the target column: it only adds features. A temporary
    column named ``downstream_class`` must be present in ``df``. This function does not
    rename it; later training uses that column directly.
    """
    from prometheo.models import Presto
    from prometheo.models.presto.wrapper import load_presto_weights

    from worldcereal.train.data import get_training_df
    from worldcereal.train.datasets import WorldCerealTrainingDataset

    # Determine Presto model URL
    if custom_presto_url is not None:
        presto_model_url = custom_presto_url
    elif task_type == "croptype":
        presto_model_url = CropTypeParameters().feature_parameters.presto_model_url
    elif task_type == "cropland":
        presto_model_url = CropLandParameters().feature_parameters.presto_model_url
    else:
        raise ValueError(
            (
                f"Unknown task type: `{task_type}` and no `custom_presto_url`"
                " given -> cannot infer Presto model"
            )
        )

    # Load pretrained Presto model
    logger.info(f"Presto URL: {presto_model_url}")
    presto_model = Presto()
    presto_model = load_presto_weights(presto_model, presto_model_url)

    # Split dataframe in cal/val
    try:
        samples_train, samples_test = train_test_split(
            df,
            test_size=0.2,
            random_state=DEFAULT_SEED,
            stratify=df["sampling_label"],
        )
    except ValueError:
        logger.warning(
            "Stratified train/test split failed (not enough samples per class),"
            " proceeding with random split."
        )
        samples_train, samples_test = train_test_split(
            df,
            test_size=0.2,
            random_state=DEFAULT_SEED,
        )

    # Initialize datasets
    samples_train, samples_test = (
        samples_train.reset_index(),
        samples_test.reset_index(),
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

    # Augmentations and repeats only on training set
    ds_train = WorldCerealTrainingDataset(
        samples_train,
        task_type="multiclass" if task_type == "croptype" else "binary",
        augment=augment,
        masking_config=masking_config,
        repeats=repeats,
    )

    # No augmentations on test set
    ds_test = WorldCerealTrainingDataset(
        samples_test,
        task_type="multiclass" if task_type == "croptype" else "binary",
        augment=False,
        masking_config=SensorMaskingConfig(enable=False),
    )

    # Compute embeddings
    logger.info("Computing Presto embeddings on train set ...")
    df_train = get_training_df(
        ds_train,
        presto_model,
        batch_size=batch_size,
        time_explicit=time_explicit,
    )
    logger.info("Computing Presto embeddings on test set ...")
    df_test = get_training_df(
        ds_test,
        presto_model,
        batch_size=batch_size,
        time_explicit=time_explicit,
    )

    # Merging train and test embeddings
    df_train["split"] = "train"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    logger.info("Done.")

    return df


def train_classifier(
    training_dataframe: pd.DataFrame,
    class_names: Optional[List[str]] = None,
    balance_classes: bool = False,
    show_confusion_matrix: Optional[Literal["absolute", "relative"]] = "relative",
) -> Tuple[CatBoostClassifier, Union[str | dict], np.ndarray]:
    """Fit and evaluate a CatBoost classifier on Presto embeddings.

    Parameters
    ----------
    training_dataframe : pandas.DataFrame
        DataFrame containing feature columns ``presto_ft_0``..``presto_ft_127``,
        a target column named ``downstream_class`` and a ``split`` column with values
        ``'train'`` and ``'test'``.
    class_names : list of str, optional
        Explicit class ordering passed to CatBoost. If ``None`` the unique labels in
        the training split are used.
    balance_classes : bool, default=False
        When ``True`` compute inverse-frequency class weights and pass them as sample
        weights to CatBoost.
    show_confusion_matrix : {'absolute', 'relative', None}, default='relative'
        Display a confusion matrix after training. ``'relative'`` normalizes per true row.

    Returns
    -------
    (CatBoostClassifier, dict | str, numpy.ndarray)
        ``(model, report, cm)`` where:

        * ``model``: trained ``CatBoostClassifier`` instance.
        * ``report``: string classification report (also logged) OR dict depending on
          internal downstream consumption (maintained for backwards compatibility).
        * ``cm``: raw (absolute counts) confusion matrix ``shape=(n_classes, n_classes)``.

    Raises
    ------
    ValueError
        If fewer than 2 unique classes are available for training.

    Notes
    -----
    Embedding semantics depend on how they were produced:
    * Time pooled (``time_explicit=False``): represents the whole (possibly jittered) window.
    * Time explicit (``time_explicit=True``): represents the state at the valid timestep.
    """

    # Split into train and test set
    samples_train = training_dataframe[
        training_dataframe["split"] == "train"
    ].reset_index()
    samples_test = training_dataframe[
        training_dataframe["split"] == "test"
    ].reset_index()

    # Define loss function and eval metric
    if np.unique(samples_train["downstream_class"]).shape[0] < 2:
        raise ValueError("Not enough classes to train a classifier.")
    elif np.unique(samples_train["downstream_class"]).shape[0] > 2:
        eval_metric = "MultiClass"
        loss_function = "MultiClass"
    else:
        eval_metric = "Logloss"
        loss_function = "Logloss"

    # Compute sample weights
    if balance_classes:
        logger.info("Computing class weights ...")
        class_weights = np.round(
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(samples_train["downstream_class"]),
                y=samples_train["downstream_class"],
            ),
            3,
        )
        class_weights = {
            k: v
            for k, v in zip(np.unique(samples_train["downstream_class"]), class_weights)
        }
        logger.info(f"Class weights: {class_weights}")

        sample_weights = np.ones((len(samples_train["downstream_class"]),))
        sample_weights_val = np.ones((len(samples_test["downstream_class"]),))
        for k, v in class_weights.items():
            sample_weights[samples_train["downstream_class"] == k] = v
            sample_weights_val[samples_test["downstream_class"] == k] = v
        samples_train["weight"] = sample_weights
        samples_test["weight"] = sample_weights_val
    else:
        samples_train["weight"] = 1
        samples_test["weight"] = 1

    # Define classifier
    custom_downstream_model = CatBoostClassifier(
        iterations=2000,  # Not too high to avoid too large model size
        depth=5,
        learning_rate=0.15,
        early_stopping_rounds=20,
        l2_leaf_reg=3,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_state=DEFAULT_SEED,
        verbose=25,
        class_names=(
            class_names
            if class_names is not None
            else np.unique(samples_train["downstream_class"])
        ),
    )

    # Setup dataset Pool
    bands = [f"presto_ft_{i}" for i in range(128)]
    calibration_data = Pool(
        data=samples_train[bands],
        label=samples_train["downstream_class"],
        weight=samples_train["weight"],
    )
    eval_data = Pool(
        data=samples_test[bands],
        label=samples_test["downstream_class"],
        weight=samples_test["weight"],
    )

    # Train classifier
    logger.info("Training CatBoost classifier ...")
    custom_downstream_model.fit(
        calibration_data,
        eval_set=eval_data,
    )

    # Make predictions
    report, cm, _ = apply_classifier(
        samples_test,
        custom_downstream_model,
        show_confusion_matrix=show_confusion_matrix,
    )

    return custom_downstream_model, report, cm


def apply_classifier(
    df: pd.DataFrame,
    model: CatBoostClassifier,
    show_confusion_matrix: Optional[Literal["absolute", "relative"]] = None,
    print_report: bool = True,
    target_attribute: str = "downstream_class",
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Apply a trained CatBoost classifier and optionally visualize its performance.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a target column (``target_attribute``) and feature columns
        ``presto_ft_0`` .. ``presto_ft_127``.
    model : CatBoostClassifier
        Fitted model returned by :func:`train_classifier`.
    show_confusion_matrix : {'absolute', 'relative', None}, default=None
        If ``'absolute'`` displays raw counts; if ``'relative'`` normalizes each row
        (true class) to sum to 1. ``None`` disables the plot.
    print_report : bool, default=True
        Whether to log & print the human readable classification report.
    target_attribute : str, default='downstream_class'
        Column name holding ground truth labels in ``df``.

    Returns
    -------
    tuple
        ``(report_dict, cm, pred)`` where:

        * ``report_dict``: dict form of :func:`sklearn.metrics.classification_report`.
        * ``cm``: raw confusion matrix (absolute counts) as ``numpy.ndarray``.
        * ``pred``: 1‑D numpy array of predicted class labels.

    Notes
    -----
    Probabilities are not currently returned; they can be obtained via
    ``model.predict_proba(df[bands])`` if needed.
    """

    # Make predictions
    bands = [f"presto_ft_{i}" for i in range(128)]
    pred = model.predict(df[bands]).flatten()

    # Classification report
    report_dict = classification_report(df[target_attribute], pred, output_dict=True)
    if print_report:
        report = classification_report(df[target_attribute], pred)
        logger.info("Classification report:")
        print(report)

    # Confusion matrix
    cm = confusion_matrix(df[target_attribute], pred)

    # Show confusion matrix if requested
    if show_confusion_matrix is not None:
        assert show_confusion_matrix in ["absolute", "relative"]

        # Get list of unique labels
        pred_labels = np.unique(pred)
        true_labels = np.unique(df[target_attribute])
        labels = sorted(np.unique(np.concatenate((pred_labels, true_labels))))

        if show_confusion_matrix == "relative":
            # normalize CM
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)
        else:
            cm_normalized = cm

        font_size = 18
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized, display_labels=labels
        )
        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=True,
            values_format=".2f",
            xticks_rotation="vertical",
        )
        for text in ax.texts:
            text.set_fontsize(font_size - 8)

        ax.set_xlabel("Predicted label", fontsize=font_size - 4)
        ax.set_ylabel("True label", fontsize=font_size - 4)
        ax.tick_params(axis="both", which="major", labelsize=font_size - 4)
        fig.suptitle(f"Confusion Matrix ({show_confusion_matrix.capitalize()})")
        plt.tight_layout()
        plt.show()

    return report_dict, cm, pred
