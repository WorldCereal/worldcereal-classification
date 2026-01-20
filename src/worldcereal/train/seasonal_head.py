"""Season-aware head utilities for Presto fine-tuning.

This module centralizes the logic required to pool time-explicit Presto
embeddings into (a) a single global representation used for landcover
classification and (b) per-season representations that can drive
season-specific crop-type heads. All logic that deviates from the original
Prometheo FinetuningHead lives here so that upstream repos remain unchanged.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from prometheo.models.pooling import PoolingMethods
from prometheo.models.presto.single_file_presto import FinetuningHead
from prometheo.models.presto.wrapper import PretrainedPrestoWrapper
from prometheo.predictors import Predictors
from torch import nn

Tensor = torch.Tensor


@dataclass
class SeasonalHeadOutput:
    """Container for the logits and intermediate embeddings.

    Attributes
    ----------
    global_logits:
        Optional logits corresponding to the globally pooled embedding.
    season_logits:
        Optional logits with shape ``[B, S, num_crop_outputs]`` representing
        per-season predictions. ``None`` when the season head is disabled.
    global_embedding:
        The pooled global embedding with shape ``[B, D]``.
    season_embeddings:
        Per-season embeddings with shape ``[B, S, D]``.
    season_masks:
        Boolean mask ``[B, S, T]`` that was used for pooling. Retained so that
        callers computing custom losses can reuse it without having to keep a
        parallel copy.
    """

    global_logits: Optional[Tensor]
    season_logits: Optional[Tensor]
    global_embedding: Tensor
    season_embeddings: Tensor
    season_masks: Tensor


class SeasonalFinetuningHead(nn.Module):
    """Head that emits global and per-season logits from time embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        landcover_num_outputs: Optional[int] = None,
        crop_num_outputs: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        """Setup the seasonal finetuning head.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the input embeddings.
        landcover_num_outputs : Optional[int], optional
            Number of landcover output classes, by default None
        crop_num_outputs : Optional[int], optional
            Number of crop output classes, by default None
        dropout : float, optional
            Dropout rate applied to the pooled embeddings, by default 0.0
        Raises
        ------
        ValueError
            Raised if neither landcover_num_outputs nor crop_num_outputs is provided.
        """
        super().__init__()
        if landcover_num_outputs is None and crop_num_outputs is None:
            raise ValueError(
                "At least one of landcover_num_outputs/crop_num_outputs must be provided"
            )

        self.embedding_dim = embedding_dim
        self.landcover_head = (
            FinetuningHead(
                hidden_size=embedding_dim,
                num_outputs=landcover_num_outputs,
                regression=False,
            )
            if landcover_num_outputs is not None
            else None
        )
        self.crop_head = (
            FinetuningHead(
                hidden_size=embedding_dim,
                num_outputs=crop_num_outputs,
                regression=False,
            )
            if crop_num_outputs is not None
            else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        time_embeddings: Tensor,
        season_masks: Tensor,
        *,
        global_mask: Optional[Tensor] = None,
    ) -> SeasonalHeadOutput:
        """Pool embeddings and project them to logits.

        Parameters
        ----------
        time_embeddings:
            Tensor shaped ``[B, T, D]`` (time-explicit Presto embeddings).
        season_masks:
            Boolean tensor ``[B, S, T]`` describing which timesteps belong to
            each season. The number of seasons ``S`` can vary between
            dataloaders as it is inferred per batch.
        global_mask:
            Optional boolean tensor ``[B, T]`` that specifies which timesteps
            to include when pooling the global embedding. When omitted, all
            timesteps in the sequence contribute equally to the global mean.
        """

        if season_masks.ndim != 3:
            raise ValueError(
                f"season_masks must be [B, S, T]; got shape {tuple(season_masks.shape)}"
            )

        device = time_embeddings.device
        batch_size, timesteps, emb_dim = time_embeddings.shape
        if emb_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, received {emb_dim}"
            )

        season_masks = season_masks.to(device=device, dtype=torch.bool)
        if season_masks.shape[0] != batch_size or season_masks.shape[2] != timesteps:
            raise ValueError(
                "season mask batch/time dims must match embeddings: "
                f"got {season_masks.shape} vs {(batch_size, timesteps)}"
            )

        # Global pooling for landcover
        if global_mask is not None:
            global_mask_float = global_mask.to(
                device=device, dtype=time_embeddings.dtype
            )
        else:
            global_mask_float = torch.ones(
                (batch_size, timesteps),
                device=device,
                dtype=time_embeddings.dtype,
            )
        global_weights = global_mask_float.sum(dim=-1, keepdim=True)
        zero_global = global_weights == 0
        if torch.any(zero_global):
            global_mask_float = global_mask_float.clone()
            global_mask_float[zero_global.squeeze(-1)] = 1.0
            global_weights = global_mask_float.sum(dim=-1, keepdim=True)

        global_z = (global_mask_float.unsqueeze(-1) * time_embeddings).sum(
            dim=-2
        ) / torch.clamp(global_weights, min=1e-6)
        global_z = self.dropout(global_z)

        # Seasonal pooling for crops
        season_mask_float = season_masks.to(device=device, dtype=time_embeddings.dtype)
        season_weights = season_mask_float.sum(dim=-1, keepdim=True)
        zero_weight = season_weights == 0
        if torch.any(zero_weight):
            season_mask_float = season_mask_float.clone()
            season_mask_float[zero_weight.squeeze(-1)] = 1.0
            season_weights = season_mask_float.sum(dim=-1, keepdim=True)

        season_z = (season_mask_float.unsqueeze(-1) * time_embeddings.unsqueeze(1)).sum(
            dim=-2
        ) / torch.clamp(season_weights, min=1e-6)
        season_z = self.dropout(season_z)

        # Project to logits for landcover and crops
        lc_logits = self.landcover_head(global_z) if self.landcover_head else None
        crop_logits: Optional[Tensor]
        if self.crop_head:
            flat = season_z.reshape(-1, self.embedding_dim)
            crop_logits = self.crop_head(flat).reshape(
                batch_size, season_z.shape[1], -1
            )
        else:
            crop_logits = None

        return SeasonalHeadOutput(
            global_logits=lc_logits,
            season_logits=crop_logits,
            global_embedding=global_z,
            season_embeddings=season_z,
            season_masks=season_masks,
        )


class WorldCerealSeasonalModel(nn.Module):
    """Adapter that wires a Presto encoder to the seasonal head."""

    def __init__(
        self,
        backbone: PretrainedPrestoWrapper,
        head: SeasonalFinetuningHead,
        *,
        season_mask_key: str = "season_masks",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.season_mask_key = season_mask_key
        self.encoder = backbone.encoder

    def forward(
        self, predictors: Predictors, attrs: Optional[dict] = None
    ) -> SeasonalHeadOutput:
        if attrs is None or self.season_mask_key not in attrs:
            raise ValueError("Seasonal model expects attrs to contain 'season_masks'.")

        # Extract time embeddings from Presto
        time_embeddings = self.backbone(predictors, eval_pooling=PoolingMethods.TIME)
        flat_embeddings = self._flatten_embeddings(time_embeddings)
        season_masks = attrs[self.season_mask_key]
        if isinstance(season_masks, np.ndarray):
            season_masks = torch.from_numpy(season_masks)
        return self.head(flat_embeddings, season_masks)

    @staticmethod
    def _flatten_embeddings(embeddings: Tensor) -> Tensor:
        if embeddings.dim() == 5:
            b, h, w, t, d = embeddings.shape
            return embeddings.view(b * h * w, t, d)
        if embeddings.dim() == 4:
            b, t, d, _ = embeddings.shape
            return embeddings.view(b, t, d)
        if embeddings.dim() == 3:
            return embeddings
        raise ValueError(
            f"Unexpected Presto encoder output shape: {tuple(embeddings.shape)}"
        )
