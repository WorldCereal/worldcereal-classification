import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc,assignment]
from tqdm.auto import tqdm

from worldcereal.train.data import collate_fn
from worldcereal.train.datasets import (
    SensorMaskingConfig,
    WorldCerealLabelledDataset,
    _is_missing_value,
)
from worldcereal.train.seasonal_head import SeasonalHeadOutput

ValidationImprovementCallback = Callable[[int, torch.nn.Module, float], None]


def _compute_metrics_from_records(
    records: List[dict], label_order: Optional[Sequence[str]]
) -> Tuple[pd.DataFrame, Figure, Figure]:
    columns = ["class", "precision", "recall", "f1-score", "support"]
    labels: Optional[List[str]]
    if not records:
        df = pd.DataFrame(columns=columns)
        labels = list(label_order) if label_order else ["n/a"]
        cm = build_confusion_matrix_figure([], [], labels=labels, normalize=False)
        cm_norm = build_confusion_matrix_figure([], [], labels=labels, normalize=True)
        return df, cm, cm_norm

    y_true = [rec["target_class"] for rec in records]
    y_pred = [rec["pred_class"] for rec in records]
    labels = list(label_order) if label_order else None
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


def _records_to_scalar_metrics(records: List[dict]) -> dict[str, float]:
    if not records:
        return {}

    from sklearn.metrics import accuracy_score, f1_score

    y_true = [rec["target_class"] for rec in records]
    y_pred = [rec["pred_class"] for rec in records]

    try:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "support": float(len(records)),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed computing seasonal metrics: {exc}")
        return {}


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
        task_sample_weight_attrs: Optional[Mapping[str, str]] = None,
        sample_weight_clip: Optional[Tuple[float, float]] = None,
        sample_weight_default: float = 1.0,
        cropland_class_names: Optional[Sequence[str]] = None,
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
        self._task_sample_weight_attrs = dict(task_sample_weight_attrs or {})
        self._sample_weight_clip = sample_weight_clip
        self._sample_weight_default = float(sample_weight_default)
        self.cropland_class_names = (
            list(cropland_class_names) if cropland_class_names is not None else []
        )

        self._lc_to_idx = {name: idx for idx, name in enumerate(landcover_classes)}
        self._ct_to_idx = {name: idx for idx, name in enumerate(croptype_classes)}

        self._ignore_index = ignore_index
        self._last_task_losses: dict[str, dict[str, float]] = {}
        self._last_croptype_supervision: dict[str, float] = {}

    @property
    def last_task_losses(self) -> Mapping[str, dict[str, float]]:
        return self._last_task_losses

    @property
    def last_croptype_supervision(self) -> Mapping[str, float]:
        return self._last_croptype_supervision

    def _task_weights_for(
        self,
        attrs: dict,
        sample_indices: Sequence[int],
        *,
        task_name: str,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        attr_key = self._task_sample_weight_attrs.get(task_name)
        if attr_key is None or not sample_indices:
            return None

        attr_values = _ensure_list(attrs.get(attr_key), batch_size, fill=None)
        weights: List[float] = []
        for idx in sample_indices:
            value = attr_values[idx]
            if value is None or _is_missing_value(value):
                numeric = self._sample_weight_default
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = self._sample_weight_default
            weights.append(numeric)

        weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        if self._sample_weight_clip is not None:
            weight_tensor = torch.clamp(
                weight_tensor,
                min=self._sample_weight_clip[0],
                max=self._sample_weight_clip[1],
            )
        return torch.clamp(weight_tensor, min=1e-6)

    @staticmethod
    def _reduce_loss(
        per_sample_losses: torch.Tensor,
        sample_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if per_sample_losses.numel() == 0:
            return torch.zeros(1, device=per_sample_losses.device, dtype=torch.float32)
        if sample_weights is None:
            return per_sample_losses.mean()
        total = torch.sum(sample_weights)
        if total <= 0:
            return per_sample_losses.mean()
        return torch.sum(per_sample_losses * sample_weights) / torch.clamp(
            total, min=1e-6
        )

    def forward(
        self,
        output: SeasonalHeadOutput,
        predictors: Predictors,
        attrs: dict,
    ) -> torch.Tensor:
        self._last_task_losses = {}
        self._last_croptype_supervision = {}
        batch_size = output.global_embedding.shape[0]
        device = output.global_embedding.device

        if attrs is None:
            raise ValueError("SeasonalMultiTaskLoss requires attrs from the DataLoader")

        landcover_labels: List[Optional[Any]] = _ensure_list(
            attrs.get("landcover_label"), batch_size, fill=None
        )
        croptype_labels: List[Optional[Any]] = _ensure_list(
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

        # Landcover pathway (supervise whenever a valid label exists)
        landcover_indices = [
            idx
            for idx in range(batch_size)
            if landcover_labels[idx] is not None
            and not _is_missing_value(landcover_labels[idx])
            and str(landcover_labels[idx]) in self._lc_to_idx
        ]
        landcover_weights_full = self._task_weights_for(
            attrs,
            landcover_indices,
            task_name=self.landcover_task_name,
            batch_size=batch_size,
            device=device,
        )
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
                per_sample_losses = F.cross_entropy(
                    logits,
                    targets,
                    reduction="none",
                )
                lc_sample_weights = None
                if landcover_weights_full is not None:
                    weight_idx = torch.tensor(
                        valid_idx,
                        device=device,
                        dtype=torch.long,
                    )
                    lc_sample_weights = landcover_weights_full[weight_idx]
                branch_loss = self._reduce_loss(per_sample_losses, lc_sample_weights)
                loss = loss + self.landcover_weight * branch_loss

                effective_weight = (
                    float(torch.sum(lc_sample_weights).item())
                    if lc_sample_weights is not None
                    else float(len(lc_targets))
                )
                scaled_loss = branch_loss * self.landcover_weight
                self._last_task_losses[self.landcover_task_name] = {
                    "raw_loss": float(branch_loss.detach().item()),
                    "scaled_loss": float(scaled_loss.detach().item()),
                    "weight": max(effective_weight, 1e-8),
                }

        # Crop-type pathway
        croptype_indices = [
            i for i, task in enumerate(tasks) if task == self.croptype_task_name
        ]
        croptype_weights_full = self._task_weights_for(
            attrs,
            croptype_indices,
            task_name=self.croptype_task_name,
            batch_size=batch_size,
            device=device,
        )
        eligible_croptype_samples = 0
        missing_representative_season = 0
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
            selection_weights: List[float] = []
            for local_idx, sample_idx in enumerate(croptype_indices):
                class_name = croptype_labels[sample_idx]
                if class_name is None or _is_missing_value(class_name):
                    continue
                mapped = self._ct_to_idx.get(str(class_name))
                if mapped is None:
                    continue

                eligible_croptype_samples += 1
                seasons = season_selection[local_idx]
                if not seasons:
                    missing_representative_season += 1
                    continue

                season_ids = torch.as_tensor(
                    seasons, device=output.season_logits.device, dtype=torch.long
                )
                logits = output.season_logits[sample_idx, season_ids, :]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                selected_logits.append(logits)
                selected_targets.extend([mapped] * logits.shape[0])
                if croptype_weights_full is not None:
                    weight_value = float(croptype_weights_full[local_idx].item())
                    selection_weights.extend([weight_value] * logits.shape[0])

            if selected_logits:
                logits_tensor = torch.cat(selected_logits, dim=0)
                targets = torch.tensor(
                    selected_targets, device=device, dtype=torch.long
                )
                per_sample_losses = F.cross_entropy(
                    logits_tensor,
                    targets,
                    reduction="none",
                )
                ct_sample_weights = None
                if croptype_weights_full is not None and selection_weights:
                    ct_sample_weights = torch.tensor(
                        selection_weights,
                        device=device,
                        dtype=torch.float32,
                    )
                branch_loss = self._reduce_loss(per_sample_losses, ct_sample_weights)
                loss = loss + self.croptype_weight * branch_loss

                effective_weight = (
                    float(torch.sum(ct_sample_weights).item())
                    if ct_sample_weights is not None
                    else float(len(selected_targets))
                )
                scaled_loss = branch_loss * self.croptype_weight
                self._last_task_losses[self.croptype_task_name] = {
                    "raw_loss": float(branch_loss.detach().item()),
                    "scaled_loss": float(scaled_loss.detach().item()),
                    "weight": max(effective_weight, 1e-8),
                }

        supervised_samples = max(
            eligible_croptype_samples - missing_representative_season, 0
        )
        stats: dict[str, float] = {
            "eligible_samples": float(eligible_croptype_samples),
            "missing_representative_season": float(missing_representative_season),
            "supervised_samples": float(supervised_samples),
        }
        if eligible_croptype_samples > 0:
            stats["missing_fraction"] = (
                missing_representative_season / eligible_croptype_samples
            )
        self._last_croptype_supervision = stats

        return loss


def prepare_training_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_timesteps: int = 12,
    timestep_freq: str = "month",
    augment: bool = True,
    time_explicit: bool = False,
    emit_label_tensor: bool = True,
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
    emit_label_tensor : bool, default=True
        Whether to emit the label tensor in the dataset samples.
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
        emit_label_tensor=emit_label_tensor,
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
        emit_label_tensor=emit_label_tensor,
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
        emit_label_tensor=emit_label_tensor,
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
    prob_batches: List[np.ndarray] = []
    pred_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    seasonal_mode = False
    seasonal_landcover_records: List[dict[str, Any]] = []
    seasonal_croptype_records: List[dict[str, Any]] = []
    croptype_gate_rejections = 0

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
                seasonal_landcover_records.extend(batch_summary["landcover"])
                seasonal_croptype_records.extend(batch_summary["croptype"])
                croptype_gate_rejections += batch_summary["croptype_gate_rejections"]
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

                        prob_batches.append(sample_probs)
                        pred_batches.append(sample_preds)
                        target_batches.append(sample_targets)
            else:
                # For non-time-explicit, just flatten and add everything
                prob_batches.append(probs.flatten())
                pred_batches.append(preds.flatten())
                target_batches.append(targets.flatten())

    if seasonal_mode:
        landcover_df, landcover_cm, landcover_cm_norm = _compute_metrics_from_records(
            seasonal_landcover_records, seasonal_landcover_classes
        )
        croptype_df, croptype_cm, croptype_cm_norm = _compute_metrics_from_records(
            seasonal_croptype_records, seasonal_croptype_classes
        )
        if croptype_gate_rejections:
            rejection_row = pd.DataFrame(
                [
                    {
                        "class": "croptype_gate_rejections",
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1-score": np.nan,
                        "support": croptype_gate_rejections,
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
                "num_samples": len(seasonal_landcover_records),
            },
            "croptype": {
                "results": croptype_df,
                "cm": croptype_cm,
                "cm_norm": croptype_cm_norm,
                "classes": seasonal_croptype_classes,
                "num_samples": len(seasonal_croptype_records),
                "gate_rejections": croptype_gate_rejections,
            },
        }

    if time_explicit:
        all_preds_array = np.concatenate(pred_batches) if pred_batches else np.array([])
        all_targets_array = (
            np.concatenate(target_batches) if target_batches else np.array([])
        )
    else:
        all_preds_array = np.concatenate(pred_batches)
        all_targets_array = np.concatenate(target_batches)

    classes_to_use: Optional[List[str]] = None
    label_order: Optional[List[str]] = None

    # Map numeric indices to class names if necessary
    all_targets: List[Any]
    all_preds: List[Any]

    if test_ds.task_type == "multiclass" and classes_list:
        label_order = [str(cls) for cls in classes_list]
        all_targets_classes = np.array(
            [
                classes_list[x] if x != NODATAVALUE else "unknown"
                for x in all_targets_array
            ]
        )
        all_preds_classes = np.array([classes_list[x] for x in all_preds_array])

        # Remove any "unknown" targets before classification report
        valid_indices = all_targets_classes != "unknown"
        all_targets = all_targets_classes[valid_indices].tolist()
        all_preds = all_preds_classes[valid_indices].tolist()

    elif test_ds.task_type == "binary":
        # For binary classification, convert to class names
        all_targets = [
            "crop" if x > 0.5 else "not_crop" for x in all_targets_array.tolist()
        ]
        all_preds = [
            "crop" if x > 0.5 else "not_crop" for x in all_preds_array.tolist()
        ]
        classes_to_use = ["not_crop", "crop"]
        label_order = classes_to_use
    else:
        label_order = [str(cls) for cls in classes_list] if classes_list else None
        all_targets = all_targets_array.tolist()
        all_preds = all_preds_array.tolist()

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
    enforce_cropland_gate: bool = False,
) -> dict:
    """Convert seasonal logits into per-branch classification records.

    Returns dictionaries for landcover and crop-type predictions so that
    evaluation and inference code can independently consume whichever branch
    they need. When ``enforce_cropland_gate`` is True, crop-type predictions are
    only emitted if either the predicted or labelled landcover class belongs to
    ``cropland_class_names``.
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

    has_landcover_targets = any(
        label is not None and not _is_missing_value(label) for label in landcover_labels
    )

    lc_probs: Optional[torch.Tensor] = None
    if output.global_logits is not None:
        lc_probs = torch.softmax(output.global_logits, dim=-1)
    elif has_landcover_targets:
        raise ValueError("Landcover supervision requested but global logits are None")

    landcover_pred_names: List[Optional[str]] = []
    if lc_probs is not None:
        for idx in range(batch_size):
            pred_idx = int(torch.argmax(lc_probs[idx]).item())
            landcover_pred_names.append(landcover_classes[pred_idx])
    else:
        landcover_pred_names = [None] * batch_size

    landcover_indices = [
        idx
        for idx in range(batch_size)
        if landcover_labels[idx] is not None
        and not _is_missing_value(landcover_labels[idx])
        and str(landcover_labels[idx]) in landcover_classes
    ]
    if lc_probs is not None:
        for sample_idx in landcover_indices:
            target_name = landcover_labels[sample_idx]
            probs = lc_probs[sample_idx]
            pred_idx = int(torch.argmax(probs).item())
            landcover_records.append(
                {
                    "pred_class": landcover_classes[pred_idx],
                    "target_class": str(target_name),
                    "prob": float(torch.max(probs).item()),
                }
            )

    source_cropland_names = (
        list(cropland_class_names) if cropland_class_names is not None else []
    )
    cropland_set = {
        str(name) for name in source_cropland_names if not _is_missing_value(name)
    }
    gating_enabled = enforce_cropland_gate and bool(cropland_set)
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

            if gating_enabled:
                landcover_target = landcover_labels[sample_idx]
                pred_landcover = landcover_pred_names[sample_idx]
                has_landcover_label = not _is_missing_value(landcover_target)

                pred_is_cropland = (
                    pred_landcover is not None and pred_landcover in cropland_set
                )
                label_is_cropland = (
                    has_landcover_label and str(landcover_target) in cropland_set
                )

                if not (pred_is_cropland or label_is_cropland):
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
    on_validation_improved: Optional[ValidationImprovementCallback] = None,
    tensorboard_logdir: Optional[Union[Path, str]] = None,
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

    tb_writer: Optional[Any] = None
    if tensorboard_logdir:
        if SummaryWriter is None:
            logger.warning(
                "TensorBoard logging requested but torch.utils.tensorboard is unavailable. "
                "Install the 'tensorboard' package to enable logging."
            )
        else:
            tb_path = Path(tensorboard_logdir)
            tb_path.mkdir(parents=True, exist_ok=True)
            try:
                tb_writer = SummaryWriter(log_dir=str(tb_path))
                logger.info(f"TensorBoard logging enabled at {tb_path}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to initialize TensorBoard writer: {exc}")
                tb_writer = None

    metrics_history_path = Path(output_dir) / f"{experiment_name}_val_history.jsonl"

    def _make_task_tracker() -> defaultdict:
        return defaultdict(lambda: {"total": 0.0, "weight": 0.0})

    def _accumulate_task_tracker(
        tracker: defaultdict,
        task_losses: Optional[Mapping[str, Mapping[str, float]]],
    ) -> None:
        if not task_losses:
            return
        for task_name, stats in task_losses.items():
            scaled_loss = stats.get("scaled_loss")
            if scaled_loss is None:
                scaled_loss = stats.get("raw_loss")
            weight = stats.get("weight", 0.0)
            if scaled_loss is None or weight <= 0.0:
                continue
            tracker[task_name]["total"] += float(scaled_loss) * float(weight)
            tracker[task_name]["weight"] += float(weight)

    def _finalize_task_tracker(tracker: defaultdict) -> dict[str, float]:
        result: dict[str, float] = {}
        for task_name, payload in tracker.items():
            weight = payload.get("weight", 0.0)
            if weight <= 0.0:
                continue
            result[task_name] = payload["total"] / max(weight, 1e-8)
        return result

    def _make_croptype_supervision_tracker() -> dict[str, float]:
        return {"eligible": 0.0, "missing": 0.0}

    def _accumulate_croptype_supervision_tracker(
        tracker: Optional[dict[str, float]],
        stats: Optional[Mapping[str, float]],
    ) -> None:
        if tracker is None or not stats:
            return
        tracker["eligible"] += float(stats.get("eligible_samples", 0.0))
        tracker["missing"] += float(stats.get("missing_representative_season", 0.0))

    def _finalize_croptype_supervision_tracker(
        tracker: Optional[dict[str, float]],
    ) -> dict[str, float]:
        if tracker is None:
            return {}
        eligible = tracker.get("eligible", 0.0)
        missing = tracker.get("missing", 0.0)
        supervised = max(eligible - missing, 0.0)
        fraction = missing / eligible if eligible > 0 else 0.0
        return {
            "eligible_samples": eligible,
            "missing_representative_season": missing,
            "supervised_samples": supervised,
            "missing_fraction": fraction,
        }

    # Track layers that were originally frozen
    originally_frozen_layers = set()

    track_croptype_supervision = isinstance(loss_fn, SeasonalMultiTaskLoss)

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
                encoder_only = deepcopy(model)
                encoder_only.head = None  # type: ignore[assignment]
                torch.save(encoder_only.state_dict(), best_encoder_ckpt_path)
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
        train_task_tracker = _make_task_tracker()
        train_croptype_supervision = (
            _make_croptype_supervision_tracker() if track_croptype_supervision else None
        )

        for batch in tqdm(train_dl, desc="Training", leave=False):
            predictors, attrs = _unpack_predictor_batch(batch)
            optimizer.zero_grad()
            preds = _forward_with_optional_attrs(model, predictors, attrs)
            loss, _, _ = _compute_loss(loss_fn, preds, predictors, attrs)

            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            _accumulate_task_tracker(
                train_task_tracker, getattr(loss_fn, "last_task_losses", None)
            )
            _accumulate_croptype_supervision_tracker(
                train_croptype_supervision,
                getattr(loss_fn, "last_croptype_supervision", None),
            )

        train_loss.append(epoch_train_loss / len(train_dl))

        model.eval()
        val_task_tracker = _make_task_tracker()
        val_croptype_supervision = (
            _make_croptype_supervision_tracker() if track_croptype_supervision else None
        )
        weighted_loss_sum = 0.0
        weighted_count = 0.0
        fallback_loss_sum = 0.0
        fallback_batches = 0
        val_pred_chunks: List[torch.Tensor] = []
        val_target_chunks: List[torch.Tensor] = []
        seasonal_loss = loss_fn if isinstance(loss_fn, SeasonalMultiTaskLoss) else None
        seasonal_metrics_supported = seasonal_loss is not None
        seasonal_landcover_records: List[dict[str, Any]] = []
        seasonal_croptype_records: List[dict[str, Any]] = []
        seasonal_gate_rejections = 0

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

                _accumulate_task_tracker(
                    val_task_tracker, getattr(loss_fn, "last_task_losses", None)
                )
                _accumulate_croptype_supervision_tracker(
                    val_croptype_supervision,
                    getattr(loss_fn, "last_croptype_supervision", None),
                )

                if seasonal_loss is not None and isinstance(preds, SeasonalHeadOutput):
                    try:
                        batch_summary = summarize_seasonal_predictions(
                            preds,
                            attrs or {},
                            seasonal_loss.landcover_classes,
                            seasonal_loss.croptype_classes,
                            cropland_class_names=getattr(
                                seasonal_loss, "cropland_class_names", None
                            ),
                            landcover_task_name=getattr(
                                seasonal_loss, "landcover_task_name", "landcover"
                            ),
                            croptype_task_name=getattr(
                                seasonal_loss, "croptype_task_name", "croptype"
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            f"Failed summarizing seasonal predictions during validation: {exc}"
                        )
                    else:
                        seasonal_landcover_records.extend(batch_summary["landcover"])
                        seasonal_croptype_records.extend(batch_summary["croptype"])
                        seasonal_gate_rejections += batch_summary[
                            "croptype_gate_rejections"
                        ]

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

        train_task_loss_avgs = _finalize_task_tracker(train_task_tracker)
        val_task_loss_avgs = _finalize_task_tracker(val_task_tracker)
        train_croptype_supervision_stats = _finalize_croptype_supervision_tracker(
            train_croptype_supervision
        )
        val_croptype_supervision_stats = _finalize_croptype_supervision_tracker(
            val_croptype_supervision
        )
        croptype_supervision_summary = None
        if track_croptype_supervision:
            croptype_supervision_summary = {
                "train": train_croptype_supervision_stats,
                "val": val_croptype_supervision_stats,
            }

        metrics_dict: dict[str, float] = {}
        if val_preds is not None and val_targets is not None:
            task_type = getattr(val_dl.dataset, "task_type", None)
            metrics_dict, _ = compute_validation_metrics(
                val_preds, val_targets, task_type
            )

        seasonal_metrics_flat: dict[str, float] = {}
        if seasonal_metrics_supported:
            lc_metrics = _records_to_scalar_metrics(seasonal_landcover_records)
            if lc_metrics:
                for key, value in lc_metrics.items():
                    seasonal_metrics_flat[f"landcover/{key}"] = value
            ct_metrics = _records_to_scalar_metrics(seasonal_croptype_records)
            if ct_metrics:
                for key, value in ct_metrics.items():
                    seasonal_metrics_flat[f"croptype/{key}"] = value

            if seasonal_gate_rejections:
                total_attempts = seasonal_gate_rejections + len(
                    seasonal_croptype_records
                )
                seasonal_metrics_flat["croptype/gate_rejections"] = float(
                    seasonal_gate_rejections
                )
                if total_attempts > 0:
                    seasonal_metrics_flat["croptype/gate_rejection_rate"] = (
                        seasonal_gate_rejections / total_attempts
                    )

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_val_loss)
        else:
            scheduler.step()

        combined_metrics = dict(metrics_dict)
        combined_metrics.update(seasonal_metrics_flat)
        seasonal_metric_groups: dict[str, dict[str, float]] = {}
        if seasonal_metrics_flat:
            grouped: dict[str, dict[str, float]] = {}
            for metric_name, metric_value in seasonal_metrics_flat.items():
                parts = metric_name.split("/", 1)
                if len(parts) != 2:
                    continue
                branch, leaf = parts
                grouped.setdefault(leaf, {})[branch] = metric_value
            seasonal_metric_groups = grouped

        if tb_writer is not None:
            global_step = epoch + 1
            tb_writer.add_scalars(
                "loss",
                {
                    "train": train_loss[-1],
                    "val": current_val_loss,
                },
                global_step,
            )
            tb_writer.add_scalar(
                "learning_rate", scheduler.get_last_lr()[0], global_step
            )
            tb_writer.add_scalar(
                "patience/epochs_since_improvement",
                epochs_since_improvement,
                global_step,
            )
            task_names = sorted(
                set(train_task_loss_avgs.keys()) | set(val_task_loss_avgs.keys())
            )
            for task_name in task_names:
                payload = {}
                if task_name in train_task_loss_avgs:
                    payload["train"] = train_task_loss_avgs[task_name]
                if task_name in val_task_loss_avgs:
                    payload["val"] = val_task_loss_avgs[task_name]
                if payload:
                    tb_writer.add_scalars(f"loss/{task_name}", payload, global_step)

            if croptype_supervision_summary is not None:
                train_sup = croptype_supervision_summary.get("train", {})
                val_sup = croptype_supervision_summary.get("val", {})
                frac_payload = {}
                if train_sup.get("eligible_samples", 0.0) > 0:
                    frac_payload["train"] = train_sup.get("missing_fraction", 0.0)
                if val_sup.get("eligible_samples", 0.0) > 0:
                    frac_payload["val"] = val_sup.get("missing_fraction", 0.0)
                if frac_payload:
                    tb_writer.add_scalars(
                        "croptype/missing_season_fraction", frac_payload, global_step
                    )

                count_payload = {}
                if train_sup.get("eligible_samples", 0.0) > 0:
                    count_payload["train"] = train_sup.get(
                        "missing_representative_season", 0.0
                    )
                if val_sup.get("eligible_samples", 0.0) > 0:
                    count_payload["val"] = val_sup.get(
                        "missing_representative_season", 0.0
                    )
                if count_payload:
                    tb_writer.add_scalars(
                        "croptype/missing_season_count", count_payload, global_step
                    )

            logged_suffixes = set()
            for metric_name, metric_value in combined_metrics.items():
                if metric_name in seasonal_metrics_flat:
                    continue
                tb_writer.add_scalar(
                    f"metrics/{metric_name}", metric_value, global_step
                )
                logged_suffixes.add(metric_name)
            for suffix, series in seasonal_metric_groups.items():
                tb_writer.add_scalars(f"metrics/{suffix}", series, global_step)
            tb_writer.flush()

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss[-1],
            "val_loss": current_val_loss,
            "task_losses": {
                "train": train_task_loss_avgs,
                "val": val_task_loss_avgs,
            },
            "metrics": combined_metrics,
        }
        if croptype_supervision_summary is not None:
            history_entry["croptype_supervision"] = croptype_supervision_summary
        try:
            with metrics_history_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(history_entry) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to append validation metrics history at epoch {epoch + 1}: {exc}"
            )

        improved = best_loss is None or current_val_loss < best_loss
        if improved:
            best_loss = current_val_loss
            best_model_dict = deepcopy(model.state_dict())
            epochs_since_improvement = 0

            validation_context = {
                "epoch": epoch + 1,
                "train_loss": train_loss[-1],
                "val_loss": current_val_loss,
                "task_losses": {
                    "train": train_task_loss_avgs,
                    "val": val_task_loss_avgs,
                },
                "metrics": combined_metrics,
            }
            if croptype_supervision_summary is not None:
                validation_context["croptype_supervision"] = (
                    croptype_supervision_summary
                )
            setattr(model, "_last_validation_context", validation_context)

            if on_validation_improved is not None:
                try:
                    on_validation_improved(epoch + 1, model, best_loss)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Validation-improvement callback failed at epoch {epoch + 1}: {exc}"
                    )

            checkpoint_model = deepcopy(model)
            _save_best(epoch + 1, checkpoint_model, best_loss)
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

    if tb_writer is not None:
        tb_writer.close()

    return model
