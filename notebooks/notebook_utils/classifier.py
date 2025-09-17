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
from worldcereal.utils.refdata import process_extractions_df
from worldcereal.utils.timeseries import MIN_EDGE_BUFFER


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
) -> pd.DataFrame:
    """Compute features for training a crop classification model.
    This function processes the time series in the input dataframe to align
    them with the specified temporal context (season) and computes
    Presto embeddings based on the extracted time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the training data.
    season : TemporalContext
        Temporal context defining the season of interest.
    freq : Literal["month", "dekad"], optional
        Frequency of the data, by default "month".
    valid_time_buffer : int, optional
        Buffer in months to apply when aligning available extractions
        with user-defined temporal extent.
        Determines how close we allow the true valid_time of the sample
        to be to the edge of the processing period, by default MIN_EDGE_BUFFER.
    batch_size : int, optional
        Batch size for processing, by default 256.
    task_type : str, optional
        Type of task (e.g., "croptype"), by default "croptype".
    augment : bool, optional
        If True, temporal jittering is enabled, by default True.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the computed 128 Presto embeddings,
        along with ewoc_code (crop type label).
    """

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

    return df


def compute_presto_embeddings(
    df: pd.DataFrame,
    batch_size: int = 256,
    task_type: str = "croptype",
    augment: bool = True,
) -> pd.DataFrame:
    """Method to generate a training dataframe with Presto embeddings for downstream Catboost training.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe with required input features for Presto
    batch_size : int, optional
        by default 256
    task_type : str, optional
        cropland or croptype task, by default "croptype"
    augment : bool, optional
        if True, temporal jittering is enabled, by default True

    Returns
    -------
    pd.DataFrame
        output training dataframe for downstream training

    Raises
    ------
    ValueError
        if an unknown tasktype is specified
    """
    from prometheo.models import Presto
    from prometheo.models.presto.wrapper import load_presto_weights

    from worldcereal.train.data import WorldCerealTrainingDataset, get_training_df

    if task_type == "croptype":
        presto_model_url = CropTypeParameters().feature_parameters.presto_model_url
    elif task_type == "cropland":
        presto_model_url = CropLandParameters().feature_parameters.presto_model_url
    else:
        raise ValueError(f"Unknown task type: {task_type}")

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
    )

    logger.info("Done.")

    return df


def train_classifier(
    training_dataframe: pd.DataFrame,
    class_names: Optional[List[str]] = None,
    balance_classes: bool = False,
    show_confusion_matrix: Optional[Literal["absolute", "relative"]] = "relative",
) -> Tuple[CatBoostClassifier, Union[str | dict], np.ndarray]:
    """Method to train a custom CatBoostClassifier on a training dataframe.

    Parameters
    ----------
    training_dataframe : pd.DataFrame
        training dataframe containing inputs and targets
    class_names : Optional[List[str]], optional
        class names to use, by default None
    balance_classes : bool, optional
        if True, class weights are used during training to balance the classes, by default False
    show_confusion_matrix : Optional[Literal["absolute", "relative"]], optional
        if 'absolute', the confusion matrix is shown as absolute values,
        if 'relative', the confusion matrix is shown as relative values,
        if None, no confusion matrix is shown,
        by default 'relative'

    Returns
    -------
    Tuple[CatBoostClassifier, Union[str | dict], np.ndarray]
        The trained CatBoost model, the classification report, and the confusion matrix

    Raises
    ------
    ValueError
        When not enough classes are present in the training dataframe to train a model
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
