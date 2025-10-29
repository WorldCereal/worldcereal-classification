from copy import deepcopy
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from prometheo.finetune import Hyperparams
from prometheo.finetune import _setup as _prometheo_setup
from prometheo.predictors import NODATAVALUE, Predictors
from prometheo.utils import device, seed_everything
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from worldcereal.train.datasets import SensorMaskingConfig, WorldCerealLabelledDataset


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=1,
        gamma=2.0,
        reduction="mean",
        ignore_index: Optional[int] = None,
        label_smoothing: float = 0.0,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MulticlassWithCroplandAuxBCELoss(nn.Module):
    """
    Combines standard multiclass CrossEntropy with an auxiliary binary cropland loss.

    From logits z (shape [..., C]):
        z_pos = logsumexp(z[k] for k in pos_classes)
        z_neg = logsumexp(z[k] for k not in pos_classes)
        z_bin = z_pos - z_neg
        p_bin = sigmoid(z_bin)

    Total loss = ce_weight * CE + aux_weight * BCEWithLogits(z_bin, y_bin)

    y_bin = 1 if target in pos_classes else 0 (ignored if target == ignore_index).

    """

    def __init__(
        self,
        pos_class_indices: List[int],
        ce_weight: float = 1.0,
        aux_weight: float = 0.3,
        ignore_index: Optional[int] = None,
        pos_weight: Optional[float] = None,  # for BCE class imbalance
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.pos_class_indices = sorted(pos_class_indices)
        self.ce_weight = ce_weight
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.pos_weight = (
            torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None
            else None
        )

    def _binary_logit(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (..., C)
        returns: (...,) binary cropland logit
        """
        C = logits.shape[-1]
        device = logits.device
        pos_mask = torch.zeros(C, dtype=torch.bool, device=device)
        pos_mask[self.pos_class_indices] = True
        neg_mask = ~pos_mask

        z_pos = torch.logsumexp(logits[..., pos_mask], dim=-1)
        z_neg = torch.logsumexp(logits[..., neg_mask], dim=-1)
        return z_pos - z_neg  # binary logit

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: shape (B, C) or (B, T, C)
        targets: shape (B,) or (B, T,) with class indices
        """
        is_time = logits.dim() == 3  # (B, T, C)

        if is_time:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
        else:
            logits_flat = logits
            targets_flat = targets

        # Multiclass CE
        ce_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )

        # Build binary targets (ignore positions with ignore_index)
        with torch.no_grad():
            valid_mask = (
                targets_flat != self.ignore_index
                if self.ignore_index is not None
                else torch.ones_like(targets_flat, dtype=torch.bool)
            )
            y_bin = torch.zeros_like(targets_flat, dtype=torch.float32)
            pos_set = set(self.pos_class_indices)
            pos_mask = torch.tensor(
                [t.item() in pos_set for t in targets_flat],
                dtype=torch.bool,
                device=targets_flat.device,
            )
            y_bin[pos_mask & valid_mask] = 1.0

        # Compute binary logit only on valid positions
        z_bin_all = self._binary_logit(logits_flat)  # shape (B*T,) or (B,)
        z_bin = z_bin_all[valid_mask]
        y_bin_valid = y_bin[valid_mask]

        if z_bin.numel() == 0:
            bce_loss = torch.tensor(0.0, device=logits.device)
        else:
            bce_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
                if self.pos_weight is not None
                else None
            )
            bce_loss = bce_loss_fn(z_bin, y_bin_valid)

        total_loss = self.ce_weight * ce_loss + self.aux_weight * bce_loss

        return total_loss


def prepare_training_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_timesteps: int = 12,
    timestep_freq: str = "month",
    augment: bool = True,
    time_explicit: bool = False,
    task_type: Literal["binary", "multiclass"] = "binary",
    num_outputs: int = 1,
    classes_list: Optional[List[str]] = None,
    masking_config: Optional[SensorMaskingConfig] = None,
    label_jitter=0,
    label_window=0,
) -> Tuple[
    WorldCerealLabelledDataset, WorldCerealLabelledDataset, WorldCerealLabelledDataset
]:
    """
    Prepare training, validation, and test datasets for model training.

    This function creates WorldCerealLabelledDataset instances from provided dataframes.

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame containing training data.
    val_df : pd.DataFrame
        DataFrame containing validation data.
    test_df : pd.DataFrame
        DataFrame containing test data.
    num_timesteps : int, default=12
        Number of timesteps to use for each sample.
    timestep_freq : str, default="month"
        Frequency of timesteps. Can be "month" or "dekad".
    augment : bool, default=True
        Whether to apply data augmentation to the training dataset.
    time_explicit : bool, default=False
        Whether to use explicit time features.
    task_type : Literal["binary", "multiclass"], default="binary"
        Type of classification task.
    num_outputs : int, default=1
        Number of output classes.
    classes_list : Optional[List[str]], default=None
        List of class names. If None, an empty list is used. Required for multiclass task.
    masking_config : Optional[SensorMaskingConfig], default=None
        Configuration for sensor masking during training and validation.
    label_jitter : int, default=0
        Jittering true position of label(s). If 0, no jittering is applied.
    label_window : int, default=0
        Expanding true label in the neighboring window. If 0, no windowing is applied.

    Returns
    -------
    Tuple[InSeasonLabelledDataset, InSeasonLabelledDataset, InSeasonLabelledDataset]
        Tuple containing training, validation, and test datasets.
    """
    train_ds = WorldCerealLabelledDataset(
        train_df,
        num_timesteps=num_timesteps,
        timestep_freq=timestep_freq,
        task_type=task_type,
        num_outputs=num_outputs,
        time_explicit=time_explicit,
        classes_list=classes_list if classes_list is not None else [],
        augment=augment,
        masking_config=masking_config,
        label_jitter=label_jitter,
        label_window=label_window,
    )
    val_ds = WorldCerealLabelledDataset(
        val_df,
        num_timesteps=num_timesteps,
        timestep_freq=timestep_freq,
        task_type=task_type,
        num_outputs=num_outputs,
        time_explicit=time_explicit,
        classes_list=classes_list if classes_list is not None else [],
        augment=False,  # No augmentation for validation
        masking_config=None,  # No masking for validation
        label_jitter=0,  # No jittering for validation
        label_window=0,  # No windowing for validation
    )
    test_ds = WorldCerealLabelledDataset(
        test_df,
        num_timesteps=num_timesteps,
        timestep_freq=timestep_freq,
        task_type=task_type,
        num_outputs=num_outputs,
        time_explicit=time_explicit,
        classes_list=classes_list if classes_list is not None else [],
        augment=False,  # No augmentation for testing
        masking_config=None,  # No masking for testing
        label_jitter=0,  # No jittering for testing
        label_window=0,  # No windowing for testing
    )
    return train_ds, val_ds, test_ds


def evaluate_finetuned_model(
    finetuned_model,
    test_ds: WorldCerealLabelledDataset,
    num_workers: int,
    batch_size: int,
    time_explicit: bool = False,
    classes_list: Optional[List[str]] = None,
    return_uncertainty: bool = False,
):
    """
    Evaluates a fine-tuned Presto model on a test dataset and calculates performance metrics.
    This function runs the provided model through the test dataset and computes classification
    metrics including precision, recall, F1-score, and support for each class.

    Parameters
    ----------
    finetuned_model : PretrainedPrestoWrapper
        The fine-tuned Presto model to evaluate.
    test_ds : InSeasonLabelledDataset
        The test dataset containing samples and ground truth labels.
    num_workers : int
        Number of worker processes for the DataLoader.
    batch_size : int
        Batch size for model evaluation.
    time_explicit : bool, default=False
        Whether to handle time-explicit predictions by only evaluating valid timesteps. Defaults to False.
    classes_list : Optional[List[str]], default=None
        List of class names for multiclass classification.
        Used to map numeric indices to class names in the output.
        Defaults to None. Required for multiclass task.

    Returns
    -------
        pd.DataFrame: A DataFrame containing classification metrics (precision, recall, F1-score, support)
                     for each class, with class names as rows and metrics as columns.

    Raises:
    -------
    ValueError : If the task type in the test dataset is not supported (must be 'binary' or 'multiclass').
    """
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report
    from torch.utils.data import DataLoader

    # storage for full distributions if we need entropy
    all_probs_full: list[np.ndarray] = [] if return_uncertainty else []

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
    all_probs = []
    all_preds = []
    all_targets = []

    for batch in val_dl:
        with torch.no_grad():
            # batch may already be a Predictors or a dict collated by DataLoader
            if isinstance(batch, dict):
                batch = Predictors(**batch)

            model_output = finetuned_model(batch)
            targets = batch.label.cpu().numpy().astype(int)

            if test_ds.task_type == "binary":
                probs = torch.sigmoid(model_output).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            elif test_ds.task_type == "multiclass":
                probs_all = (
                    torch.softmax(model_output, dim=-1).cpu().numpy()
                )  # shape (B,T,C)

                preds = np.argmax(probs_all, axis=-1, keepdims=True)
                probs = np.max(probs_all, axis=-1, keepdims=True)

                preds = preds[targets != NODATAVALUE]
                probs = probs[targets != NODATAVALUE]
                probs_all = probs_all[(targets != NODATAVALUE)[..., -1], :]
                targets = targets[targets != NODATAVALUE]

                if return_uncertainty:
                    # flatten batch×timesteps into (N,C)
                    if all_probs_full is not None:
                        all_probs_full.append(
                            probs_all.reshape(-1, probs_all.shape[-1])
                        )
            else:
                raise ValueError(f"Unsupported task type: {test_ds.task_type}")

            # Handle time-explicit predictions by filtering to valid timesteps only
            if time_explicit:
                # Create a mask that identifies where targets are valid (not NODATAVALUE)
                valid_mask = targets != NODATAVALUE

                # Flatten everything with masks to keep only valid predictions
                for i in range(targets.shape[0]):
                    sample_valid_mask = valid_mask[i].flatten()
                    if np.any(sample_valid_mask):
                        # Only include samples that have at least one valid target
                        sample_probs = probs[i].flatten()[sample_valid_mask]
                        sample_preds = preds[i].flatten()[sample_valid_mask]
                        sample_targets = targets[i].flatten()[sample_valid_mask]

                        all_probs.append(sample_probs)
                        all_preds.append(sample_preds)
                        all_targets.append(sample_targets)
            else:
                # For non-time-explicit, just flatten and add everything
                all_probs.append(probs.flatten())
                all_preds.append(preds.flatten())
                all_targets.append(targets.flatten())

    if time_explicit:
        all_probs = np.concatenate(all_probs) if all_probs else np.array([])
        all_preds = np.concatenate(all_preds) if all_preds else np.array([])
        all_targets = np.concatenate(all_targets) if all_targets else np.array([])
    else:
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

    # Map numeric indices to class names if necessary
    if test_ds.task_type == "multiclass" and classes_list:
        all_targets_classes = np.array(
            [classes_list[x] if x != NODATAVALUE else "unknown" for x in all_targets]
        )
        all_preds_classes = np.array([classes_list[x] for x in all_preds])

        # Remove any "unknown" targets before classification report
        valid_indices = all_targets_classes != "unknown"
        all_targets = list(all_targets_classes[valid_indices])
        all_preds = list(all_preds_classes[valid_indices])
        if len(all_probs) > 0:
            all_probs_array = np.array(all_probs)[valid_indices]
            all_probs = list(all_probs_array)
        else:
            all_probs = []
    elif test_ds.task_type == "binary":
        # For binary classification, convert to class names
        all_targets = ["crop" if x > 0.5 else "not_crop" for x in all_targets]
        all_preds = list(
            np.array(["crop" if x > 0.5 else "not_crop" for x in all_preds])
        )
        classes_to_use = ["not_crop", "crop"]
    else:
        # Just use the classes as is
        classes_to_use = classes_list if classes_list is not None else []

    results = classification_report(
        all_targets,
        all_preds,
        labels=classes_to_use if test_ds.task_type == "binary" else None,
        output_dict=True,
        zero_division=0,
    )

    cm = ConfusionMatrixDisplay.from_predictions(
        all_targets,
        all_preds,
        xticks_rotation="vertical",
        labels=classes_to_use if test_ds.task_type == "binary" else None,
    )
    cm_norm = ConfusionMatrixDisplay.from_predictions(
        all_targets,
        all_preds,
        xticks_rotation="vertical",
        normalize="true",
        labels=classes_to_use if test_ds.task_type == "binary" else None,
    )

    results_df = pd.DataFrame(results).transpose().reset_index()
    results_df.columns = pd.Index(
        ["class", "precision", "recall", "f1-score", "support"]
    )
    # compute average predictive entropy if requested
    if return_uncertainty:
        from scipy.stats import entropy

        pf = (
            np.concatenate(all_probs_full, axis=0) if all_probs_full else np.empty((0,))
        )  # shape (N, C)
        ent = entropy(pf.T) if pf.size > 0 else np.array([])  # length N
        results_df["avg_entropy"] = ent.mean() if ent.size > 0 else np.nan

    return results_df, cm, cm_norm


def run_finetuning(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    experiment_name: str,
    output_dir: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    hyperparams: Hyperparams,
    loss_fn: torch.nn.Module,
    *,
    setup_logging: bool = False,
    freeze_layers: Optional[List[str]] = None,
    unfreeze_epoch: Optional[int] = None,
):
    """Perform the training loop for fine-tuning a model.

    Parameters
    ----------

    Returns
    -------
    torch.nn.Module
        The trained model.
    """

    output_dir = Path(output_dir)
    _prometheo_setup(output_dir, experiment_name, setup_logging)
    seed_everything()

    train_loss = []
    val_loss = []
    best_loss = None
    best_model_dict = None
    epochs_since_improvement = 0

    # Track layers that were originally frozen
    originally_frozen_layers = set()

    # Freeze specified layers initially
    if freeze_layers:
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                if not param.requires_grad:
                    originally_frozen_layers.add(name)
                param.requires_grad = False
                logger.info(f"Freezing layer: {name}")

    for epoch in (pbar := tqdm(range(hyperparams.max_epochs), desc="Finetuning")):
        model.train()

        # Unfreezing logic
        if freeze_layers and epoch == unfreeze_epoch:
            for name, param in model.named_parameters():
                if name not in originally_frozen_layers and any(
                    layer in name for layer in freeze_layers
                ):
                    param.requires_grad = True
                    logger.info(f"Unfreezing layer: {name}")

        epoch_train_loss = 0.0

        for batch in tqdm(train_dl, desc="Training", leave=False):
            optimizer.zero_grad()
            preds = model(batch)
            targets = batch.label.to(device)

            if preds.dim() > 1 and preds.size(-1) > 1:
                # multiclass case: targets should be class indices
                # predictions are multiclass logits
                targets = targets.long().squeeze(axis=-1)
            else:
                # binary or regression case
                targets = targets.float()

            # Compute loss
            loss = loss_fn(
                preds[targets != NODATAVALUE], targets[targets != NODATAVALUE]
            )

            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(epoch_train_loss / len(train_dl))

        model.eval()
        all_preds, all_y = [], []

        model.eval()
        all_preds, all_y = [], []

        for batch in val_dl:
            with torch.no_grad():
                preds = model(batch)
                targets = batch.label.to(device)

                if preds.dim() > 1 and preds.size(-1) > 1:
                    # multiclass case: targets should be class indices
                    # predictions are multiclass logits
                    targets = targets.long().squeeze(axis=-1)
                else:
                    # binary or regression case
                    targets = targets.float()

                preds = preds[targets != NODATAVALUE]
                targets = targets[targets != NODATAVALUE]
                all_preds.append(preds)
                all_y.append(targets)

        val_preds = torch.cat(all_preds)
        val_targets = torch.cat(all_y)
        current_val_loss = loss_fn(val_preds, val_targets).item()
        val_loss.append(current_val_loss)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_val_loss)
        else:
            scheduler.step()

        if best_loss is None:
            best_loss = val_loss[-1]
            best_model_dict = deepcopy(model.state_dict())
        else:
            if val_loss[-1] < best_loss:
                best_loss = val_loss[-1]
                best_model_dict = deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= hyperparams.patience:
                    logger.info("Early stopping!")
                    break

        description = (
            f"Epoch {epoch + 1}/{hyperparams.max_epochs} | "
            f"Train Loss: {train_loss[-1]:.4f} | "
            f"Val Loss: {current_val_loss:.4f} | "
            f"Best Loss: {best_loss:.4f}"
        )

        if epochs_since_improvement > 0:
            description += f" (no improvement for {epochs_since_improvement} epochs)"
        else:
            description += " (improved)"

        pbar.set_description(description)
        pbar.set_postfix(lr=scheduler.get_last_lr()[0])
        logger.info(
            f"PROGRESS after Epoch {epoch + 1}/{hyperparams.max_epochs}: {description}"
        )  # Only log to file if console filters on "PROGRESS"

    assert best_model_dict is not None

    model.load_state_dict(best_model_dict)
    model.eval()

    return model
