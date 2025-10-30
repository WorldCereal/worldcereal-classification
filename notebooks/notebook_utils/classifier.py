"""Utility functions to generate Presto embeddings and train CatBoost classifiers.

This module provides a light-weight, notebook friendly pipeline to:

1. Align raw reference data extractions to a user defined season (``TemporalContext``).
2. Generate 128‑D Presto embeddings (either globally pooled or time‑explicit at the valid
     timestep) ready for downstream ML.
3. Train and evaluate a CatBoost classifier (multiclass crop type or binary cropland).

Design principles
-----------------
* Keep dependencies minimal inside the notebook environment.
* Preserve original metadata columns in the returned DataFrames whenever possible.
* Make temporal behaviour (augmentation vs time explicit embedding) explicit in the
    docstrings so that downstream interpretation of model features is unambiguous.

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
from worldcereal.train.datasets import MIN_EDGE_BUFFER
from worldcereal.utils.refdata import process_extractions_df


def get_input(label):
    """Get user input as short string without spaces."""
    while True:
        modelname = input(f"Enter a short name for your {label} (don't use spaces): ")
        if " " not in modelname:
            return modelname
        print("Invalid input. Please enter a name without spaces.")


def compute_training_features(
    df: pd.DataFrame,
    season: TemporalContext,
    freq: Literal["month", "dekad"] = "month",
    valid_time_buffer: int = MIN_EDGE_BUFFER,
    batch_size: int = 256,
    task_type: str = "croptype",
    augment: bool = True,
    time_explicit: bool = True,
    custom_presto_url: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a training dataframe with Presto embeddings and labels.

    The pipeline performs three steps: (1) temporal alignment of extraction rows to the
    user provided ``season`` (with a safety buffer), (2) embedding inference with a
    pretrained *Presto* model (optionally using temporal augmentation), and (3) light
    harmonisation / renaming of the target column.

    Temporal representation modes
    -----------------------------
    ``time_explicit=False``
        One 128‑D embedding is produced per sample via internal global pooling over the
        (possibly jittered) temporal slice. Augmentation changes which timesteps
        contribute to the pooled summary.

    ``time_explicit=True`` (default)
        No global pooling: the embedding at the *valid* timestep (``valid_position``)
        is selected after alignment / augmentation. There is still one 128‑D vector per
        sample but it represents the state at the valid time instead of an aggregate.

    Augmentation interaction
    ------------------------
    ``augment=True`` introduces horizontal (temporal) jitter. For pooled embeddings this
    changes the window being summarised. For time explicit mode the absolute real-world
    valid timestep stays the same; only its relative index inside the extracted window may shift.

    Output schema
    -------------
    The returned dataframe preserves original metadata and adds the following columns:

    * ``presto_ft_0`` .. ``presto_ft_127`` : float32 embedding features.
    * ``ewoc_code`` : final class label (copied / renamed from temporary ``downstream_class``).
    * ``year`` : year parsed from ``ref_id`` (added for simple diagnostics).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw extraction rows containing at minimum ``ewoc_code`` and ``ref_id``.
    season : TemporalContext
        Target temporal context defining the modelling season.
    freq : {'month', 'dekad'}, default='month'
        Resampling / alignment frequency.
    valid_time_buffer : int, default=MIN_EDGE_BUFFER
        Minimum distance (in time units compatible with ``freq``) required between the
        sample's original valid time and the edges of ``season``.
    batch_size : int, default=256
        Batch size used during embedding inference.
    task_type : {'croptype', 'cropland'}, default='croptype'
        Determines which pretrained Presto weights to load and multiclass vs binary mode.
    augment : bool, default=True
        Enable temporal jitter data augmentation.
    time_explicit : bool, default=True
        Switch from globally pooled sequence embeddings to valid timestep embeddings.
    custom_presto_url : str, optional
        If provided, this URL overrides the default Presto model used to compute embeddings.

    Returns
    -------
    pandas.DataFrame
        Aligned samples with embedding columns and ``ewoc_code`` label. A warning is
        logged if fewer than 2 unique classes remain.

    Raises
    ------
    ValueError
        If ``task_type`` is invalid (propagated from downstream functions).

    Notes
    -----
    This function does not perform train/test splitting; it only produces features.
    Use :func:`train_classifier` for modelling.
    """
    from worldcereal.utils.legend import ewoc_code_to_label

    # Align the samples with the season of interest
    df = process_extractions_df(df, season, freq, valid_time_buffer)

    # Create an attribute "downstream_class" that is a copy of "ewoc_code"
    # for compatibility with presto computation
    df["downstream_class"] = df["ewoc_code"].copy()

    # Now compute the Presto embeddings
    df = compute_presto_embeddings(
        df,
        batch_size=batch_size,
        task_type=task_type,
        augment=augment,
        time_explicit=time_explicit,
        custom_presto_url=custom_presto_url,
    )

    # Report on contents of the resulting dataframe here
    logger.info(
        f"Samples originating from {df['ref_id'].nunique()} unique reference datasets."
    )

    logger.info("Distribution of samples across years:")
    # extract year from ref_id
    df["year"] = df["ref_id"].str.split("_").str[0].astype(int)
    logger.info(df.year.value_counts())

    # Rename downstream_class to ewoc_code
    df.rename(columns={"downstream_class": "ewoc_code"}, inplace=True)
    ncroptypes = df["ewoc_code"].nunique()
    logger.info(f"Number of crop types remaining: {ncroptypes}")
    if ncroptypes <= 1:
        logger.warning(
            "Not enough crop types found in the remaining data to train a model, cannot continue with model training!"
        )

    # Enrich resulting dataframe with full and sampling string labels
    df["label_full"] = ewoc_code_to_label(df["ewoc_code"], label_type="full")
    df["sampling_label"] = ewoc_code_to_label(df["ewoc_code"], label_type="sampling")

    return df


def compute_presto_embeddings(
    df: pd.DataFrame,
    batch_size: int = 256,
    task_type: str = "croptype",
    augment: bool = True,
    time_explicit: bool = True,
    custom_presto_url: Optional[str] = None,
) -> pd.DataFrame:
    """Run pretrained *Presto* model to attach 128‑D embeddings to each sample.

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
    time_explicit : bool, default=True
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
    column named ``downstream_class`` is expected by the dataset wrapper prior to renaming
    in :func:`compute_training_features`.
    """
    from prometheo.models import Presto
    from prometheo.models.presto.wrapper import load_presto_weights

    from worldcereal.train.data import WorldCerealTrainingDataset, get_training_df

    # Determine Presto model URL
    if custom_presto_url is not None:
        presto_model_url = custom_presto_url
    elif task_type == "croptype":
        presto_model_url = CropTypeParameters().feature_parameters.presto_model_url
    elif task_type == "cropland":
        presto_model_url = CropLandParameters().feature_parameters.presto_model_url
    else:
        raise ValueError((f"Unknown task type: `{task_type}` and no `custom_presto_url`"
                          " given -> cannot infer Presto model"))

    # Load pretrained Presto model
    logger.info(f"Presto URL: {presto_model_url}")
    presto_model = Presto()
    presto_model = load_presto_weights(presto_model, presto_model_url)

    # Initialize dataset
    df = df.reset_index()
    ds = WorldCerealTrainingDataset(
        df,
        task_type="multiclass" if task_type == "croptype" else "binary",
        augment=augment,
    )
    logger.info("Computing Presto embeddings ...")
    df = get_training_df(
        ds,
        presto_model,
        batch_size=batch_size,
        time_explicit=time_explicit,
    )

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
        DataFrame containing feature columns ``presto_ft_0``..``presto_ft_127`` and
        a target column named ``downstream_class``.
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
    tuple
        ``(model, report, cm)`` where:

        * ``model`` is the trained ``CatBoostClassifier``.
        * ``report`` is the string output of :func:`sklearn.metrics.classification_report`.
        * ``cm`` is the raw (non-normalized) confusion matrix ``numpy.ndarray``.

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
    logger.info("Split train/test ...")
    samples_train, samples_test = train_test_split(
        training_dataframe,
        test_size=0.2,
        random_state=DEFAULT_SEED,
        stratify=training_dataframe["downstream_class"],
    )

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
        depth=8,
        early_stopping_rounds=20,
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
) -> pd.DataFrame:
    """Method to apply a trained CatBoostClassifier to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe containing the features to apply the model on
    model : CatBoostClassifier
        trained CatBoost model
    show_confusion_matrix : Optional[Literal["absolute", "relative"]], optional
        if 'absolute', the confusion matrix is shown as absolute values,
        if 'relative', the confusion matrix is shown as relative values,
        if None, no confusion matrix is shown,
        by default None
    print_report : bool, optional
        if True, the classification report is printed to the console,
        by default True
    target_attribute : str, optional
        name of the attribute in the dataframe containing the true class labels,
        by default "downstream_class"

    Returns
    -------
    pd.DataFrame
        dataframe with additional columns "predicted_class" and "predicted_proba"
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
