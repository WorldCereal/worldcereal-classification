from copy import deepcopy
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from matplotlib.figure import Figure
from prometheo.finetune import Hyperparams
from prometheo.finetune import _setup as _prometheo_setup
from prometheo.predictors import NODATAVALUE, Predictors
from prometheo.utils import device, seed_everything
from seaborn import heatmap
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from worldcereal.train.data import collate_fn
from worldcereal.train.datasets import (
    SensorMaskingConfig,
    WorldCerealLabelledDataset,
    _is_missing_value,
)
from worldcereal.train.seasonal_head import SeasonalHeadOutput


def build_confusion_matrix_figure(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Optional[Sequence[str]] = None,
    normalize: bool = True,
    title: Optional[str] = None,
) -> Figure:
    """Render a confusion matrix heatmap."""

    if labels is not None:
        label_order = list(labels)
    else:
        combined = list(y_true) + list(y_pred)
        label_order = list(dict.fromkeys(combined)) if combined else []
    if not label_order:
        label_order = ["n/a"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_order,
        normalize="true" if normalize else None,
    )

    def _annot(value: float) -> str:
        if normalize:
            return f"{100 * value:.1f}%" if value > 0 else ""
        return f"{int(value)}" if value > 0 else ""

    max_label_length = max(len(str(label)) for label in label_order)
    base = max(len(label_order) * 0.7 + max_label_length * 0.15, 6)
    fig = plt.figure(figsize=(max(10, base), max(9, base - 1)))
    if title:
        plt.title(title)
    data = 100 * cm if normalize else cm
    annotations = np.asarray([_annot(x) for x in cm.flatten()]).reshape(cm.shape)
    ax = fig.add_subplot(111)
    heatmap(
        data,
        vmin=0,
        vmax=100 if normalize else None,
        annot=annotations,
        fmt="",
        xticklabels=label_order,
        yticklabels=label_order,
        linewidths=0.01,
        square=True,
        ax=ax,
    )
    ax.set_xticklabels(label_order, rotation=90)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    plt.tight_layout()
    return fig


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=1,
        gamma=2.0,
        reduction="mean",
        ignore_index: Optional[int] = -100,
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
        ignore_index: int = -100,
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


class SeasonalMultiTaskLoss(nn.Module):
    """Compute landcover and crop-type losses from SeasonalHeadOutput."""

    def __init__(
        self,
        landcover_classes: List[str],
        croptype_classes: List[str],
        *,
        ignore_index: int = NODATAVALUE,
        landcover_weight: float = 1.0,
        croptype_weight: float = 1.0,
        landcover_task_name: str = "landcover",
        croptype_task_name: str = "croptype",
    ) -> None:
        super().__init__()
        if not landcover_classes:
            raise ValueError("landcover_classes cannot be empty for seasonal loss")
        if not croptype_classes:
            raise ValueError("croptype_classes cannot be empty for seasonal loss")

        self.landcover_classes = landcover_classes
        self.croptype_classes = croptype_classes
        self.landcover_weight = landcover_weight
        self.croptype_weight = croptype_weight
        self.landcover_task_name = landcover_task_name
        self.croptype_task_name = croptype_task_name

        self._lc_to_idx = {name: idx for idx, name in enumerate(landcover_classes)}
        self._ct_to_idx = {name: idx for idx, name in enumerate(croptype_classes)}

        self._lc_loss = nn.CrossEntropyLoss()
        self._ct_loss = nn.CrossEntropyLoss()
        self._ignore_index = ignore_index

    def forward(
        self,
        output: SeasonalHeadOutput,
        predictors: Predictors,
        attrs: dict,
    ) -> torch.Tensor:
        batch_size = output.global_embedding.shape[0]
        device = output.global_embedding.device

        if attrs is None:
            raise ValueError("SeasonalMultiTaskLoss requires attrs from the DataLoader")

        landcover_labels = _ensure_list(
            attrs.get("landcover_label"), batch_size, fill=None
        )
        croptype_labels = _ensure_list(
            attrs.get("croptype_label"), batch_size, fill=None
        )
        label_tasks = _ensure_list(attrs.get("label_task"), batch_size, fill=None)

        tasks: List[str] = []
        for idx in range(batch_size):
            if label_tasks[idx] is not None:
                tasks.append(str(label_tasks[idx]))
            elif croptype_labels[idx] is not None and not _is_missing_value(
                croptype_labels[idx]
            ):
                tasks.append(self.croptype_task_name)
            else:
                tasks.append(self.landcover_task_name)

        loss = torch.zeros(1, device=device, dtype=torch.float32)

        # Landcover pathway
        landcover_indices = [
            i for i, task in enumerate(tasks) if task == self.landcover_task_name
        ]
        if landcover_indices:
            if output.global_logits is None:
                raise ValueError(
                    "Seasonal head missing global logits for landcover supervision"
                )
            lc_logits = output.global_logits[landcover_indices]
            lc_targets: List[int] = []
            valid_idx: List[int] = []
            for batch_idx, sample_idx in enumerate(landcover_indices):
                class_name = landcover_labels[sample_idx]
                if class_name is None or _is_missing_value(class_name):
                    continue
                mapped = self._lc_to_idx.get(str(class_name))
                if mapped is None:
                    continue
                lc_targets.append(mapped)
                valid_idx.append(batch_idx)

            if lc_targets:
                logits = lc_logits[valid_idx]
                targets = torch.tensor(lc_targets, device=device, dtype=torch.long)
                loss = loss + self.landcover_weight * self._lc_loss(logits, targets)

        # Crop-type pathway
        croptype_indices = [
            i for i, task in enumerate(tasks) if task == self.croptype_task_name
        ]
        if croptype_indices:
            if output.season_logits is None:
                raise ValueError(
                    "Seasonal head missing season logits for crop-type supervision"
                )

            season_selection = _select_representative_season(
                output, attrs, croptype_indices, allow_multiple=True
            )

            selected_logits: List[torch.Tensor] = []
            selected_targets: List[int] = []
            for local_idx, sample_idx in enumerate(croptype_indices):
                class_name = croptype_labels[sample_idx]
                if class_name is None or _is_missing_value(class_name):
                    continue
                mapped = self._ct_to_idx.get(str(class_name))
                if mapped is None:
                    continue

                seasons = season_selection[local_idx]
                if not seasons:
                    continue

                season_ids = torch.as_tensor(
                    seasons, device=output.season_logits.device, dtype=torch.long
                )
                logits = output.season_logits[sample_idx, season_ids, :]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                selected_logits.append(logits)
                selected_targets.extend([mapped] * logits.shape[0])

            if selected_logits:
                logits_tensor = torch.cat(selected_logits, dim=0)
                targets = torch.tensor(
                    selected_targets, device=device, dtype=torch.long
                )
                loss = loss + self.croptype_weight * self._ct_loss(
                    logits_tensor, targets
                )

        return loss


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
    *,
    seasonal_landcover_classes: Optional[List[str]] = None,
    seasonal_croptype_classes: Optional[List[str]] = None,
    cropland_class_names: Optional[Sequence[str]] = None,
) -> Union[dict, Tuple[pd.DataFrame, Figure, Figure]]:
    """Evaluate a fine-tuned model on a labelled dataset and report metrics.

    Parameters
    ----------
    finetuned_model : torch.nn.Module
        Model to evaluate; can emit logits or ``SeasonalHeadOutput``.
    test_ds : WorldCerealLabelledDataset
        Dataset containing samples and ground-truth labels.
    num_workers : int
        Number of workers for the evaluation ``DataLoader``.
    batch_size : int
        Batch size used during evaluation.
    time_explicit : bool, default False
        Whether the labels/logits are time-explicit, in which case only valid
        timesteps are scored.
    classes_list : Optional[List[str]]
        Mapping from class index to class name for multiclass/binary outputs.
    return_uncertainty : bool, default False
        When True, also compute average predictive entropy from the stored
        probability distributions.
    seasonal_landcover_classes : Optional[List[str]]
        Required when the model returns ``SeasonalHeadOutput``; names for the
        landcover logits.
    seasonal_croptype_classes : Optional[List[str]]
        Required when the model returns ``SeasonalHeadOutput``; names for the
        crop-type logits.
    cropland_class_names : Optional[Sequence[str]]
        Optional list of class names that should trigger cropland gating during
        seasonal evaluation.

    Returns
    -------
    Union[dict, Tuple[pd.DataFrame, Figure, Figure]]
        Seasonal heads return a dictionary with per-task reports and confusion
        matrices. Standard heads return ``(results_df, cm, cm_norm)`` similar to
        ``sklearn.metrics.classification_report`` where each confusion matrix is a
        Matplotlib ``Figure`` instance.

    Raises
    ------
    ValueError
        If ``test_ds.task_type`` is unsupported or if seasonal outputs are
        encountered without the necessary class lists.
    """

    def _compute_metrics_from_records(
        records: List[dict], label_order: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, Figure, Figure]:
        columns = ["class", "precision", "recall", "f1-score", "support"]
        if not records:
            df = pd.DataFrame(columns=columns)
            labels = label_order or ["n/a"]
            cm = build_confusion_matrix_figure([], [], labels=labels, normalize=False)
            cm_norm = build_confusion_matrix_figure(
                [], [], labels=labels, normalize=True
            )
            return df, cm, cm_norm

        y_true = [rec["target_class"] for rec in records]
        y_pred = [rec["pred_class"] for rec in records]
        labels = label_order
        results = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        df = pd.DataFrame(results).transpose().reset_index()
        df.columns = pd.Index(columns)
        cm = build_confusion_matrix_figure(
            y_true,
            y_pred,
            labels=labels,
            normalize=False,
        )
        cm_norm = build_confusion_matrix_figure(
            y_true,
            y_pred,
            labels=labels,
            normalize=True,
        )
        return df, cm, cm_norm

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
        collate_fn=collate_fn,
    )
    assert isinstance(val_dl.sampler, torch.utils.data.SequentialSampler)

    # Run the model on the test set
    all_probs = []
    all_preds = []
    all_targets = []
    seasonal_mode = False
    seasonal_records = {
        "landcover": [],
        "croptype": [],
        "croptype_gate_rejections": 0,
    }

    for batch in val_dl:
        predictors, attrs = _unpack_predictor_batch(batch)
        with torch.no_grad():
            model_output = _forward_with_optional_attrs(
                finetuned_model, predictors, attrs
            )
            if isinstance(model_output, SeasonalHeadOutput):
                if (
                    seasonal_landcover_classes is None
                    or seasonal_croptype_classes is None
                ):
                    raise ValueError(
                        "Seasonal evaluation requires landcover and crop-type class lists"
                    )
                batch_summary = summarize_seasonal_predictions(
                    model_output,
                    attrs or {},
                    seasonal_landcover_classes,
                    seasonal_croptype_classes,
                    cropland_class_names=cropland_class_names,
                )
                seasonal_records["landcover"].extend(batch_summary["landcover"])
                seasonal_records["croptype"].extend(batch_summary["croptype"])
                seasonal_records["croptype_gate_rejections"] += batch_summary[
                    "croptype_gate_rejections"
                ]
                seasonal_mode = True
                continue
            targets = predictors.label.cpu().numpy().astype(int)

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

    if seasonal_mode:
        landcover_df, landcover_cm, landcover_cm_norm = _compute_metrics_from_records(
            seasonal_records["landcover"], seasonal_landcover_classes
        )
        croptype_df, croptype_cm, croptype_cm_norm = _compute_metrics_from_records(
            seasonal_records["croptype"], seasonal_croptype_classes
        )
        gate_rejections = seasonal_records["croptype_gate_rejections"]
        if gate_rejections:
            rejection_row = pd.DataFrame(
                [
                    {
                        "class": "croptype_gate_rejections",
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1-score": np.nan,
                        "support": gate_rejections,
                    }
                ]
            )
            croptype_df = pd.concat([croptype_df, rejection_row], ignore_index=True)

        return {
            "landcover": {
                "results": landcover_df,
                "cm": landcover_cm,
                "cm_norm": landcover_cm_norm,
                "classes": seasonal_landcover_classes,
                "num_samples": len(seasonal_records["landcover"]),
            },
            "croptype": {
                "results": croptype_df,
                "cm": croptype_cm,
                "cm_norm": croptype_cm_norm,
                "classes": seasonal_croptype_classes,
                "num_samples": len(seasonal_records["croptype"]),
                "gate_rejections": gate_rejections,
            },
        }

    if time_explicit:
        all_probs = np.concatenate(all_probs) if all_probs else np.array([])
        all_preds = np.concatenate(all_preds) if all_preds else np.array([])
        all_targets = np.concatenate(all_targets) if all_targets else np.array([])
    else:
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

    classes_to_use: Optional[List[str]] = None
    label_order: Optional[List[str]] = None

    # Map numeric indices to class names if necessary
    if test_ds.task_type == "multiclass" and classes_list:
        label_order = [str(cls) for cls in classes_list]
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
        label_order = classes_to_use
    else:
        label_order = [str(cls) for cls in classes_list] if classes_list else None

    results = classification_report(
        all_targets,
        all_preds,
        labels=classes_to_use if test_ds.task_type == "binary" else None,
        output_dict=True,
        zero_division=0,
    )

    cm = build_confusion_matrix_figure(
        all_targets,
        all_preds,
        labels=label_order,
        normalize=False,
    )
    cm_norm = build_confusion_matrix_figure(
        all_targets,
        all_preds,
        labels=label_order,
        normalize=True,
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


def compute_validation_metrics(
    val_preds: torch.Tensor, val_targets: torch.Tensor, task_type: Optional[str]
) -> tuple[dict, str]:
    """Compute standard classification metrics for validation set.

    Parameters
    ----------
    val_preds : torch.Tensor
        Concatenated model predictions (logits) for all validation samples with ignored labels removed.
        Shape (N,) for binary or (N, C) for multiclass.
    val_targets : torch.Tensor
        Concatenated ground truth targets with ignored labels removed. Shape (N,).
    task_type : Optional[str]
        Either 'binary', 'multiclass' or None. If not recognized metrics are skipped.

    Returns
    -------
    metrics_dict : dict
        Dictionary containing accuracy, macro_f1, weighted_f1 (keys absent if computation failed).
    metrics_str : str
        Pre-formatted string for console logging.

    Notes
    -----
    This helper is designed for easy future expansion (e.g., adding per-class F1, confusion matrix
    serialization, binary-specific metrics like precision/recall for the positive class, or figure logging).
    """
    metrics: dict[str, float] = {}
    metrics_str = ""
    if task_type not in {"binary", "multiclass"}:
        return metrics, metrics_str

    from sklearn.metrics import accuracy_score, f1_score

    if val_targets.numel() == 0:
        logger.warning(
            "Empty validation targets encountered; skipping metrics computation."
        )
        return metrics, metrics_str

    if task_type == "binary":
        probs = torch.sigmoid(val_preds).detach().cpu().numpy()
        preds_np = (probs > 0.5).astype(int)
        targets_np = val_targets.detach().cpu().numpy().astype(int)
    else:  # multiclass
        preds_np = torch.argmax(val_preds, dim=-1).detach().cpu().numpy().astype(int)
        targets_np = val_targets.detach().cpu().numpy().astype(int)

    try:
        acc = accuracy_score(targets_np, preds_np)
        f1_macro = f1_score(targets_np, preds_np, average="macro", zero_division=0)
        f1_weighted = f1_score(
            targets_np, preds_np, average="weighted", zero_division=0
        )
        metrics.update(
            {
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }
        )
        metrics_str = f" | Acc: {acc:.4f} | F1(macro): {f1_macro:.4f} | F1(weighted): {f1_weighted:.4f}"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed computing validation metrics: {e}")
    return metrics, metrics_str


def _unpack_predictor_batch(batch: Any) -> Tuple[Predictors, dict]:
    """Normalize DataLoader outputs so callers always receive Predictors + attrs."""

    if isinstance(batch, Predictors):
        return batch, {}

    if isinstance(batch, tuple):
        if len(batch) == 2 and isinstance(batch[0], Predictors):
            attrs = batch[1] if isinstance(batch[1], dict) else {}
            return batch[0], attrs
        if len(batch) == 1:
            return _unpack_predictor_batch(batch[0])

    if isinstance(batch, list) and batch:
        return _unpack_predictor_batch(tuple(batch))

    if isinstance(batch, dict):
        return Predictors(**batch), {}

    raise TypeError(f"Unsupported batch type from DataLoader: {type(batch)}")


def _forward_with_optional_attrs(
    model: torch.nn.Module, predictors: Predictors, attrs: Optional[dict]
):
    """Attempt to call model with attrs; gracefully fallback when unsupported."""

    if attrs:
        try:
            return model(predictors, attrs=attrs)
        except TypeError as exc:  # model might not accept attrs
            if "unexpected keyword argument 'attrs'" not in str(exc):
                raise
    return model(predictors)


def _ensure_list(value, expected_len: int, fill=None) -> List:
    """Normalize assorted attr containers (list/np/tensor) into Python lists."""

    if value is None:
        return [fill] * expected_len

    if isinstance(value, list):
        result = value
    elif isinstance(value, tuple):
        result = list(value)
    elif isinstance(value, np.ndarray):
        result = value.tolist()
    elif torch.is_tensor(value):
        result = value.detach().cpu().tolist()
    else:
        result = list(value)

    if len(result) != expected_len:
        raise ValueError(
            f"Attribute list has length {len(result)}, expected {expected_len}"
        )
    return result


def _select_representative_season(
    output: SeasonalHeadOutput,
    attrs: dict,
    sample_indices: List[int],
    *,
    allow_multiple: bool = True,
) -> Union[torch.Tensor, List[List[int]]]:
    """Choose one or more season indices per sample using metadata cues.

    When ``allow_multiple`` is True the function returns a ``List[List[int]]`` where
    each inner list enumerates all seasons whose masks overlap the label date. When
    False it falls back to a single representative per sample and returns a
    tensor of indices, matching the previous behavior expected by evaluation utilities.
    """

    season_masks = output.season_masks
    if not torch.is_tensor(season_masks):
        season_masks = torch.as_tensor(season_masks, dtype=torch.bool)
    season_masks = season_masks.to(device=output.global_embedding.device)

    in_seasons = attrs.get("in_seasons")
    if in_seasons is not None:
        in_seasons_tensor = torch.as_tensor(
            in_seasons, device=season_masks.device, dtype=torch.bool
        )
        if in_seasons_tensor.dim() == 1:
            in_seasons_tensor = in_seasons_tensor.unsqueeze(0)
    else:
        in_seasons_tensor = None

    valid_position = attrs.get("valid_position")
    if valid_position is None:
        raise ValueError(
            "valid_position must be present in attrs for seasonal operations"
        )
    valid_position_tensor = torch.as_tensor(
        valid_position, device=season_masks.device, dtype=torch.long
    )

    num_timesteps = season_masks.shape[-1]
    sample_ids = attrs.get("sample_id")
    sample_id_list = (
        _ensure_list(sample_ids, season_masks.shape[0], fill=None)
        if sample_ids is not None
        else None
    )

    selections: List[List[int]] = []
    for sample_idx in sample_indices:
        candidate_indices: List[int] = []
        if in_seasons_tensor is not None:
            season_flags = in_seasons_tensor[sample_idx].flatten()
            if season_flags.any():
                candidate_indices.extend(
                    torch.nonzero(season_flags, as_tuple=False).view(-1).tolist()
                )

        if not candidate_indices:
            vp = int(valid_position_tensor[sample_idx].item())
            vp = max(0, min(vp, num_timesteps - 1))
            mask = season_masks[sample_idx, :, vp]
            mask = mask.flatten()
            if mask.any():
                candidate_indices.extend(
                    torch.nonzero(mask, as_tuple=False).view(-1).tolist()
                )

        if not candidate_indices:
            # sample_id = (
            #     sample_id_list[sample_idx] if sample_id_list is not None else None
            # )
            # logger.warning(
            #     "Skipping sample without representative season"
            #     + (f" (sample_id={sample_id})" if sample_id is not None else "")
            # )
            selections.append([])
            continue

        selections.append(candidate_indices)

    if allow_multiple:
        return selections

    first_indices = [indices[0] for indices in selections if indices]
    if len(first_indices) != len(selections):
        logger.warning(
            "Dropping samples without representative season while returning tensor"
        )
    return torch.tensor(first_indices, device=season_masks.device, dtype=torch.long)


def summarize_seasonal_predictions(
    output: SeasonalHeadOutput,
    attrs: dict,
    landcover_classes: List[str],
    croptype_classes: List[str],
    *,
    cropland_class_names: Optional[Sequence[str]] = None,
    landcover_task_name: str = "landcover",
    croptype_task_name: str = "croptype",
) -> dict:
    """Convert seasonal logits into per-branch classification records.

    Returns dictionaries for landcover and crop-type predictions so that
    evaluation and inference code can independently consume whichever branch
    they need. Crop-type predictions are automatically gated by the landcover
    head: if the predicted landcover class is not part of ``cropland_class_names``
    (when provided), the crop-type prediction is marked invalid and excluded.
    """

    batch_size = output.global_embedding.shape[0]
    landcover_labels = _ensure_list(attrs.get("landcover_label"), batch_size, fill=None)
    croptype_labels = _ensure_list(attrs.get("croptype_label"), batch_size, fill=None)
    label_tasks = _ensure_list(attrs.get("label_task"), batch_size, fill=None)

    tasks: List[str] = []
    for idx in range(batch_size):
        label_task = label_tasks[idx]
        if label_task is not None and not _is_missing_value(label_task):
            tasks.append(str(label_task))
        elif croptype_labels[idx] is not None and not _is_missing_value(
            croptype_labels[idx]
        ):
            tasks.append(croptype_task_name)
        else:
            tasks.append(landcover_task_name)

    landcover_records: List[dict] = []
    croptype_records: List[dict] = []
    croptype_gate_rejections = 0

    lc_probs: Optional[torch.Tensor] = None
    if output.global_logits is not None:
        lc_probs = torch.softmax(output.global_logits, dim=-1)
    elif landcover_task_name in tasks:
        raise ValueError("Landcover supervision requested but global logits are None")

    landcover_pred_names: List[Optional[str]] = []
    if lc_probs is not None:
        for idx in range(batch_size):
            pred_idx = int(torch.argmax(lc_probs[idx]).item())
            landcover_pred_names.append(landcover_classes[pred_idx])
    else:
        landcover_pred_names = [None] * batch_size

    landcover_indices = [
        idx for idx, task in enumerate(tasks) if task == landcover_task_name
    ]
    if lc_probs is not None:
        for sample_idx in landcover_indices:
            target_name = landcover_labels[sample_idx]
            if target_name is None or _is_missing_value(target_name):
                continue
            if str(target_name) not in landcover_classes:
                continue
            probs = lc_probs[sample_idx]
            pred_idx = int(torch.argmax(probs).item())
            landcover_records.append(
                {
                    "pred_class": landcover_classes[pred_idx],
                    "target_class": str(target_name),
                    "prob": float(torch.max(probs).item()),
                }
            )

    cropland_set = {
        str(name)
        for name in (cropland_class_names or [])
        if not _is_missing_value(name)
    }
    croptype_indices = [
        idx for idx, task in enumerate(tasks) if task == croptype_task_name
    ]
    if croptype_indices:
        if output.season_logits is None:
            raise ValueError(
                "Seasonal head missing season logits for crop-type supervision"
            )
        season_selection = _select_representative_season(
            output, attrs, croptype_indices, allow_multiple=True
        )
        for local_idx, sample_idx in enumerate(croptype_indices):
            target_name = croptype_labels[sample_idx]
            if _is_missing_value(target_name):
                continue
            if str(target_name) not in croptype_classes:
                continue

            landcover_target = landcover_labels[sample_idx]
            has_landcover_label = not _is_missing_value(landcover_target)
            is_cropland = not cropland_set or (
                has_landcover_label and str(landcover_target) in cropland_set
            )
            if not is_cropland:
                croptype_gate_rejections += 1
                continue

            seasons = season_selection[local_idx]
            if not seasons:
                continue

            season_ids = torch.as_tensor(
                seasons, device=output.season_logits.device, dtype=torch.long
            )
            logits = output.season_logits[sample_idx, season_ids, :]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            probs = torch.softmax(logits, dim=-1).mean(dim=0)
            pred_idx = int(torch.argmax(probs).item())
            croptype_records.append(
                {
                    "pred_class": croptype_classes[pred_idx],
                    "target_class": str(target_name),
                    "prob": float(torch.max(probs).item()),
                }
            )

    return {
        "landcover": landcover_records,
        "croptype": croptype_records,
        "croptype_gate_rejections": croptype_gate_rejections,
    }


def _compute_loss(
    loss_fn: torch.nn.Module,
    preds,
    predictors: Predictors,
    attrs: Optional[dict],
):
    """Route to the appropriate loss implementation and return masked outputs."""

    if isinstance(preds, SeasonalHeadOutput):
        loss = loss_fn(preds, predictors, attrs or {})
        return loss, None, None

    targets = predictors.label.to(device)
    if preds.dim() > 1 and preds.size(-1) > 1:
        targets = targets.long().squeeze(axis=-1)
    else:
        targets = targets.float()

    mask = targets != NODATAVALUE
    if not torch.any(mask):
        loss = torch.zeros(1, device=preds.device, dtype=preds.dtype)
        return loss, None, None

    masked_preds = preds[mask]
    masked_targets = targets[mask]
    loss = loss_fn(masked_preds, masked_targets)
    return loss, masked_preds.detach(), masked_targets.detach()


def run_finetuning(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    experiment_name: str,
    output_dir: Union[Path, str],
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

    # Define checkpoint paths
    best_ckpt_path = Path(output_dir) / f"{experiment_name}.pt"
    best_encoder_ckpt_path = Path(output_dir) / f"{experiment_name}_encoder.pt"

    def _save_best(epoch_idx: int, model: torch.nn.Module, best_val: float):
        """Persist best full-model checkpoint and encoder-only variant (head=None)."""
        # Save full model
        try:
            torch.save(model.state_dict(), best_ckpt_path)
            logger.debug(
                f"Saved best checkpoint (val_loss={best_val:.4f}) to {best_ckpt_path}"
            )
        except Exception as e:
            logger.warning(f"Failed saving best full-model checkpoint: {e}")

        # Save encoder-only by deepcopy + removing head
        if hasattr(model, "head"):
            try:
                model.head = None
                torch.save(model.state_dict(), best_encoder_ckpt_path)
                logger.debug(
                    f"Saved best encoder-only checkpoint to {best_encoder_ckpt_path}"
                )
            except Exception as e:
                logger.warning(f"Failed saving encoder-only checkpoint: {e}")
        else:
            logger.debug(
                "Model has no 'head' attribute; skipping encoder-only checkpoint."
            )

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
            predictors, attrs = _unpack_predictor_batch(batch)
            optimizer.zero_grad()
            preds = _forward_with_optional_attrs(model, predictors, attrs)
            loss, _, _ = _compute_loss(loss_fn, preds, predictors, attrs)

            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(epoch_train_loss / len(train_dl))

        model.eval()
        weighted_loss_sum = 0.0
        weighted_count = 0.0
        fallback_loss_sum = 0.0
        fallback_batches = 0
        val_pred_chunks: List[torch.Tensor] = []
        val_target_chunks: List[torch.Tensor] = []

        for batch in val_dl:
            predictors, attrs = _unpack_predictor_batch(batch)
            with torch.no_grad():
                preds = _forward_with_optional_attrs(model, predictors, attrs)
                loss_value, flat_preds, flat_targets = _compute_loss(
                    loss_fn, preds, predictors, attrs
                )
                if flat_preds is not None and flat_targets is not None:
                    chunk_weight = float(flat_targets.numel())
                    weighted_loss_sum += loss_value.item() * chunk_weight
                    weighted_count += chunk_weight
                    val_pred_chunks.append(flat_preds)
                    val_target_chunks.append(flat_targets)
                else:
                    fallback_loss_sum += loss_value.item()
                    fallback_batches += 1

        if weighted_count > 0:
            current_val_loss = weighted_loss_sum / weighted_count
            val_preds = torch.cat(val_pred_chunks)
            val_targets = torch.cat(val_target_chunks)
        elif fallback_batches > 0:
            current_val_loss = fallback_loss_sum / fallback_batches
            val_preds = None
            val_targets = None
        else:
            current_val_loss = 0.0
            val_preds = None
            val_targets = None
        val_loss.append(current_val_loss)

        metrics_str = ""
        if val_preds is not None and val_targets is not None:
            task_type = getattr(val_dl.dataset, "task_type", None)
            _, metrics_str = compute_validation_metrics(
                val_preds, val_targets, task_type
            )

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_val_loss)
        else:
            scheduler.step()

        improved = best_loss is None or current_val_loss < best_loss
        if improved:
            best_loss = current_val_loss
            best_model = deepcopy(model)
            best_model_dict = model.state_dict()
            epochs_since_improvement = 0
            _save_best(epoch + 1, best_model, best_loss)
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
        ) + (metrics_str if "metrics_str" in locals() else "")

        description += (
            " (improved)"
            if epochs_since_improvement == 0
            else f" (no improvement for {epochs_since_improvement} epochs)"
        )

        pbar.set_description(description)
        pbar.set_postfix(lr=scheduler.get_last_lr()[0])
        logger.info(
            f"PROGRESS after Epoch {epoch + 1}/{hyperparams.max_epochs}: {description}"
        )  # Only log to file if console filters on "PROGRESS"

    assert best_model_dict is not None

    model.load_state_dict(best_model_dict)
    model.eval()

    return model
