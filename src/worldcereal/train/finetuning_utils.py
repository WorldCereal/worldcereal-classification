from copy import deepcopy
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from prometheo.finetune import Hyperparams
from prometheo.finetune import _setup as _prometheo_setup
from prometheo.predictors import NODATAVALUE, Predictors
from prometheo.utils import device, seed_everything
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from worldcereal.train.datasets import WorldCerealLabelledDataset


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
    label_jitter=0,
    label_window: Optional[int] = None,
    return_time_weights: bool = False,
    time_kernel: Literal["delta", "gaussian", "triangular"] = "delta",
    time_kernel_bandwidth: Optional[float] = None,
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
    label_jitter : int, default=0
        Jittering true position of label(s). If 0, no jittering is applied.
    label_window : Optional[int], default=None
        Radius of the supervised window. ``None`` lets the dataset decide
        (default 0 for delta kernels, ≥1 for soft kernels). Provide a non-negative
        integer to override the inferred value.
    return_time_weights : bool, default=False
        When True, the training dataset yields temporal kernel weights alongside the predictors.
    time_kernel : Literal["delta", "gaussian", "triangular"], default="delta"
        Shape of the temporal weighting kernel applied around the valid timestep.
    time_kernel_bandwidth : float, optional
        Kernel bandwidth in timesteps. Used as the Gaussian sigma or triangular half-width.

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
        label_jitter=label_jitter,
        label_window=label_window,
        return_time_weights=return_time_weights,
        time_kernel=time_kernel,
        time_kernel_bandwidth=time_kernel_bandwidth,
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
        label_jitter=0,  # No jittering for validation
        label_window=0,  # No windowing for validation
        return_time_weights=False,
        time_kernel=time_kernel,
        time_kernel_bandwidth=time_kernel_bandwidth,
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
        label_jitter=0,  # No jittering for testing
        label_window=0,  # No windowing for testing
        return_time_weights=False,
        time_kernel=time_kernel,
        time_kernel_bandwidth=time_kernel_bandwidth,
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


def _split_batch_with_temporal_weights(batch):
    """Unpack a DataLoader batch into Predictors and optional temporal weights."""

    weights = None
    extra = None

    if isinstance(batch, Predictors):
        predictors = batch
    elif isinstance(batch, dict):
        predictors = Predictors(**batch)
    elif isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Received empty batch tuple")
        predictors = batch[0]
        if len(batch) >= 2:
            weights = batch[1]
        if len(batch) >= 3:
            extra = batch[2]
        if isinstance(predictors, dict):
            predictors = Predictors(**predictors)
    else:
        predictors = batch

    if isinstance(predictors, dict):
        predictors = Predictors(**predictors)

    return predictors, weights, extra


def _extract_valid_time_mask(targets: torch.Tensor) -> Optional[torch.Tensor]:
    """Reduce label tensor to a [B, T] mask of supervised timesteps."""

    if targets.numel() == 0:
        return None

    mask = targets != NODATAVALUE
    if mask.dim() >= 2:
        mask = mask.any(dim=-1)

    while mask.dim() > 2:
        mask = mask.any(dim=1)

    if mask.dim() != 2:
        return None
    return mask


def _save_attention_visualizations(
    attention: torch.Tensor,
    prior: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    epoch: int,
    output_dir: Path,
    max_samples: int = 4,
) -> None:
    """Save heatmap snapshots of attention (and priors) for qualitative monitoring."""

    if attention.numel() == 0:
        return

    attention_dir = Path(output_dir) / "attention_maps"
    attention_dir.mkdir(parents=True, exist_ok=True)

    att = attention.detach().cpu().numpy().reshape(-1, attention.shape[-1])
    if att.shape[0] == 0:
        return

    num_samples = min(max_samples, att.shape[0])
    time_steps = att.shape[-1]

    prior_array = None
    if prior is not None and prior.numel() > 0:
        p = prior.detach().cpu().numpy()
        if p.ndim == 5 and p.shape[-1] == 1:
            p = np.squeeze(p, axis=-1)
        p = p.reshape(-1, p.shape[-1])
        prior_array = p
        num_samples = min(num_samples, prior_array.shape[0])

    valid_array = None
    if valid_mask is not None:
        valid_array = valid_mask.detach().cpu().numpy()
        num_samples = min(num_samples, valid_array.shape[0])

    centre_positions: list[Optional[int]] = []
    if prior_array is not None:
        centre_positions = prior_array.argmax(axis=1).tolist()
    elif valid_array is not None:
        for row in valid_array:
            idxs = np.flatnonzero(row > 0)
            if idxs.size > 0:
                centre_positions.append(int(np.round(idxs.mean())))
            else:
                centre_positions.append(None)

    rows = 2 if prior_array is not None else 1
    fig, axes = plt.subplots(
        rows, num_samples, figsize=(3 * num_samples, 3 * rows), sharey=True
    )
    if rows == 1:
        axes = np.atleast_1d(axes)

    for idx in range(num_samples):
        sample_att = att[idx][None, :]
        ax = axes[idx] if rows == 1 else axes[0, idx]
        im = ax.imshow(sample_att, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Sample {idx}")
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, time_steps - 1, min(time_steps, 5)).astype(int))
        if idx == num_samples - 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if centre_positions:
            centre = centre_positions[idx] if idx < len(centre_positions) else None
            if centre is not None:
                ax.axvline(centre, color="white", linestyle="--", linewidth=1.0)

        if valid_array is not None:
            valid_indices = np.where(valid_array[idx] > 0)[0]
            for pos in valid_indices:
                ax.axvline(pos, color="red", linestyle="--", linewidth=1.0)

        if prior_array is not None:
            sample_prior = prior_array[idx][None, :]
            pax = axes[1, idx]
            im2 = pax.imshow(
                sample_prior, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0
            )
            pax.set_title(f"Prior {idx}")
            pax.set_yticks([])
            pax.set_xticks(
                np.linspace(0, time_steps - 1, min(time_steps, 5)).astype(int)
            )
            if idx == num_samples - 1:
                fig.colorbar(im2, ax=pax, fraction=0.046, pad=0.04)

            if centre_positions:
                centre = centre_positions[idx] if idx < len(centre_positions) else None
                if centre is not None:
                    pax.axvline(centre, color="white", linestyle="--", linewidth=1.0)

            if valid_array is not None:
                valid_indices = np.where(valid_array[idx] > 0)[0]
                for pos in valid_indices:
                    pax.axvline(pos, color="red", linestyle="--", linewidth=1.0)

    for ax in np.array(axes).flatten():
        ax.set_xlabel("Timestep")

    plt.tight_layout()
    fig_path = attention_dir / f"epoch_{epoch:03d}.png"
    fig.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved attention snapshot to {fig_path}")


def _prepare_weight_tensor(weights, reference, device_):
    if weights is None:
        return torch.ones_like(reference, dtype=torch.float32, device=device_)
    if isinstance(weights, np.ndarray):
        weights_tensor = torch.from_numpy(weights)
    else:
        weights_tensor = torch.as_tensor(weights)
    return weights_tensor.to(device=device_, dtype=torch.float32)


def _compute_temporal_loss(preds, targets, weights):
    weights = weights.to(targets.device)

    if preds.dim() > 1 and preds.size(-1) > 1:
        targets = targets.long().squeeze(-1)
        weights = weights.squeeze(-1)
        mask = targets != NODATAVALUE
        if not mask.any():
            zero = torch.tensor(0.0, device=targets.device)
            metrics = {
                key: torch.tensor(0.0, device=targets.device)
                for key in [
                    "weight_sum",
                    "weighted_idx_sum",
                    "weighted_sq_idx_sum",
                    "support_sum",
                    "support_count",
                ]
            }
            return zero, zero, metrics
        loss_terms = F.cross_entropy(
            preds[mask],
            targets[mask],
            reduction="none",
        )
        weight_vals = weights[mask]
    else:
        targets = targets.float()
        mask = targets != NODATAVALUE
        if not mask.any():
            zero = torch.tensor(0.0, device=targets.device)
            metrics = {
                key: torch.tensor(0.0, device=targets.device)
                for key in [
                    "weight_sum",
                    "weighted_idx_sum",
                    "weighted_sq_idx_sum",
                    "support_sum",
                    "support_count",
                ]
            }
            return zero, zero, metrics
        logits = preds[mask]
        target_vals = targets[mask]
        weight_vals = weights[mask]
        loss_terms = F.binary_cross_entropy_with_logits(
            logits,
            target_vals,
            reduction="none",
        )

    weighted_sum = (loss_terms * weight_vals).sum()
    total_weight = weight_vals.sum()

    w_monitor = weights
    if w_monitor.dim() > 0 and w_monitor.shape[-1] == 1:
        w_monitor = w_monitor.squeeze(-1)
    if w_monitor.dim() == 1:
        w_monitor = w_monitor.unsqueeze(0)
    else:
        w_monitor = w_monitor.reshape(-1, w_monitor.shape[-1])

    idx = torch.arange(w_monitor.shape[-1], device=weights.device, dtype=torch.float32)
    weighted_idx_sum = (w_monitor * idx).sum(dim=-1)
    weighted_sq_idx_sum = (w_monitor * (idx**2)).sum(dim=-1)
    weight_sum_seq = w_monitor.sum(dim=-1)
    support_seq = (w_monitor > 0).sum(dim=-1).float()
    valid = weight_sum_seq > 0
    metrics = {
        "weight_sum": weight_sum_seq[valid].sum()
        if valid.any()
        else torch.tensor(0.0, device=weights.device),
        "weighted_idx_sum": weighted_idx_sum[valid].sum()
        if valid.any()
        else torch.tensor(0.0, device=weights.device),
        "weighted_sq_idx_sum": weighted_sq_idx_sum[valid].sum()
        if valid.any()
        else torch.tensor(0.0, device=weights.device),
        "support_sum": support_seq[valid].sum()
        if valid.any()
        else torch.tensor(0.0, device=weights.device),
        "support_count": valid.sum().float(),
    }

    return weighted_sum, total_weight, metrics


def run_finetuning(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    experiment_name: str,
    output_dir: str | Path,
    task_type: Literal["binary", "multiclass"],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    hyperparams: Hyperparams,
    *,
    apply_temporal_weights: bool = True,
    setup_logging: bool = False,
    freeze_layers: Optional[List[str]] = None,
    unfreeze_epoch: Optional[int] = None,
    visualize_attention_every: Optional[int] = 1,
    attention_entropy_weight: float = 0.0,
):
    """Fine-tune a Presto model with optional temporally weighted supervision.

    Args:
        apply_temporal_weights: When ``True`` the temporal kernel weights are
            folded into the loss; when ``False`` the priors are still passed to
            the model (e.g. for attention MIL) but the loss defaults to
            uniform-in-time weighting.
        visualize_attention_every: Plot attention snapshots every N epochs
            (set to ``None`` or ``0`` to disable).
        attention_entropy_weight: Strength of entropy regularisation on temporal
            attention (0 disables it).
    """

    output_dir = Path(output_dir)
    _prometheo_setup(output_dir, experiment_name, setup_logging)
    seed_everything()

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_loss: Optional[float] = None
    best_state_dict = None
    epochs_since_improvement = 0

    originally_frozen = set()
    if freeze_layers:
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                if not param.requires_grad:
                    originally_frozen.add(name)
                param.requires_grad = False
                logger.info(f"Freezing layer: {name}")

    for epoch in (pbar := tqdm(range(hyperparams.max_epochs), desc="Finetuning")):
        model.train()

        if freeze_layers and unfreeze_epoch is not None and epoch == unfreeze_epoch:
            for name, param in model.named_parameters():
                if name not in originally_frozen and any(
                    layer in name for layer in freeze_layers
                ):
                    param.requires_grad = True
                    logger.info(f"Unfreezing layer: {name}")

        train_weight_sum = torch.tensor(0.0, device=device)
        train_loss_sum = torch.tensor(0.0, device=device)
        train_monitor = {
            "weight_sum": 0.0,
            "weighted_idx_sum": 0.0,
            "weighted_sq_idx_sum": 0.0,
            "support_sum": 0.0,
            "support_count": 0.0,
        }

        for batch in tqdm(train_dl, desc="Training", leave=False):
            optimizer.zero_grad()
            predictors, weights, _ = _split_batch_with_temporal_weights(batch)
            targets = predictors.label.to(device)

            prior_tensor = (
                _prepare_weight_tensor(weights, targets, device)
                if weights is not None
                else None
            )
            loss_weights = (
                prior_tensor
                if (apply_temporal_weights and prior_tensor is not None)
                else _prepare_weight_tensor(None, targets, device)
            )

            preds = model(predictors, time_weights=prior_tensor)

            loss_sum, weight_sum, metrics = _compute_temporal_loss(
                preds, targets, loss_weights
            )

            if prior_tensor is not None and not apply_temporal_weights:
                _, _, metrics_prior = _compute_temporal_loss(
                    preds.detach(), targets, prior_tensor
                )
                metrics = metrics_prior
            if weight_sum <= 0:
                continue

            if attention_entropy_weight > 0:
                raw_attn: Optional[torch.Tensor] = None
                head = getattr(model, "classification_head", None)
                if head is not None and hasattr(head, "_last_attention_raw"):
                    raw_attr = getattr(head, "_last_attention_raw")
                    if isinstance(raw_attr, torch.Tensor):
                        raw_attn = raw_attr

                if raw_attn is not None:
                    entropy = (
                        -(raw_attn.clamp_min(1e-8) * raw_attn.clamp_min(1e-8).log())
                        .sum(dim=-1)
                        .mean()
                    )
                    loss_sum = (
                        loss_sum - attention_entropy_weight * entropy * weight_sum
                    )
                    if head is not None:
                        setattr(head, "_last_attention_raw", None)

            loss = loss_sum / weight_sum
            loss.backward()
            optimizer.step()

            train_weight_sum += weight_sum.detach()
            train_loss_sum += loss_sum.detach()
            for key, value in metrics.items():
                train_monitor[key] += float(value.detach().cpu())

        if train_weight_sum > 0:
            train_epoch_loss = (train_loss_sum / train_weight_sum).item()
        else:
            train_epoch_loss = 0.0
        train_loss_history.append(train_epoch_loss)

        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        val_weight_sum = torch.tensor(0.0, device=device)
        val_monitor = {
            "weight_sum": 0.0,
            "weighted_idx_sum": 0.0,
            "weighted_sq_idx_sum": 0.0,
            "support_sum": 0.0,
            "support_count": 0.0,
        }

        attention_snapshot: Optional[torch.Tensor] = None
        prior_snapshot: Optional[torch.Tensor] = None
        valid_mask_snapshot: Optional[torch.Tensor] = None
        capture_attention = (
            visualize_attention_every is not None
            and visualize_attention_every > 0
            and (epoch % visualize_attention_every == 0)
        )

        with torch.no_grad():
            for batch in val_dl:
                predictors, weights, _ = _split_batch_with_temporal_weights(batch)
                targets = predictors.label.to(device)

                prior_tensor = (
                    _prepare_weight_tensor(weights, targets, device)
                    if weights is not None
                    else None
                )
                loss_weights = (
                    prior_tensor
                    if (apply_temporal_weights and prior_tensor is not None)
                    else _prepare_weight_tensor(None, targets, device)
                )

                preds = model(predictors, time_weights=prior_tensor)
                loss_sum, weight_sum, metrics = _compute_temporal_loss(
                    preds, targets, loss_weights
                )

                if prior_tensor is not None and not apply_temporal_weights:
                    _, _, metrics_prior = _compute_temporal_loss(
                        preds.detach(), targets, prior_tensor
                    )
                    metrics = metrics_prior
                val_loss_sum += loss_sum
                val_weight_sum += weight_sum
                for key, value in metrics.items():
                    val_monitor[key] += float(value.detach().cpu())

                if capture_attention and attention_snapshot is None:
                    head = getattr(model, "classification_head", None)
                    attn_snapshot = (
                        getattr(head, "_last_attention", None)
                        if head is not None
                        else None
                    )
                    if attn_snapshot is not None:
                        attention_snapshot = attn_snapshot.detach().cpu()
                        if prior_tensor is not None:
                            prior_snapshot = prior_tensor.detach().cpu()
                        valid_mask_tensor = _extract_valid_time_mask(targets)
                        if valid_mask_tensor is not None:
                            valid_mask_snapshot = valid_mask_tensor.detach().cpu()

        if val_weight_sum > 0:
            current_val_loss = (val_loss_sum / val_weight_sum).item()
        else:
            current_val_loss = 0.0
        val_loss_history.append(current_val_loss)

        if capture_attention and attention_snapshot is not None:
            _save_attention_visualizations(
                attention_snapshot,
                prior_snapshot,
                valid_mask_snapshot,
                epoch,
                output_dir,
            )

        def _log_temporal_monitor(prefix: str, stats: dict):
            if stats["weight_sum"] > 0 and stats["support_count"] > 0:
                mean_idx = stats["weighted_idx_sum"] / stats["weight_sum"]
                variance = max(
                    stats["weighted_sq_idx_sum"] / stats["weight_sum"] - mean_idx**2,
                    0.0,
                )
                std_idx = variance**0.5
                avg_support = stats["support_sum"] / stats["support_count"]
                msg = f"{prefix} temporal weights → mean_idx={mean_idx:.2f}, std_idx={std_idx:.2f}, avg_support={avg_support:.2f}"
                # logger.debug(msg)
                tqdm.write(msg)

        _log_temporal_monitor("Train", train_monitor)
        _log_temporal_monitor("Val", val_monitor)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_val_loss)
        else:
            scheduler.step()

        if best_loss is None or current_val_loss < best_loss:
            best_loss = current_val_loss
            best_state_dict = deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= hyperparams.patience:
                logger.info("Early stopping")
                break

        description = (
            f"Epoch {epoch + 1}/{hyperparams.max_epochs} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {current_val_loss:.4f} | "
            f"Best Loss: {best_loss:.4f}"
        )
        if epochs_since_improvement > 0:
            description += f" (no improvement for {epochs_since_improvement} epochs)"
        else:
            description += " (improved)"
        pbar.set_description(description)
        if hasattr(scheduler, "get_last_lr"):
            pbar.set_postfix(lr=scheduler.get_last_lr()[0])
        logger.info(
            f"PROGRESS after Epoch {epoch + 1}/{hyperparams.max_epochs}: {description}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()
    return model


# Backwards compatibility: keep the old entry point name available.
run_temporal_kernel_finetuning = run_finetuning
