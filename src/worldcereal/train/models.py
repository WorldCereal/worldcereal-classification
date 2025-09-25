from __future__ import annotations

from typing import Optional, Union

import os
import torch
from einops import rearrange
from torch import nn

from prometheo.models.presto.single_file_presto import FinetuningHead
from prometheo.models.presto.wrapper import (
    PoolingMethods,
    PretrainedPrestoWrapper,
    dataset_to_model,
    to_torchtensor,
)
from prometheo.predictors import Predictors


class TemporalAttentionHead(nn.Module):
    """Attention-based multiple-instance head with optional temporal priors."""

    def __init__(
        self,
        hidden_size: int,
        num_outputs: int,
        regression: bool,
        use_prior: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if regression:
            raise ValueError("TemporalAttentionHead currently supports classification only")
        self.use_prior = use_prior
        self.eps = eps

        self.att_proj = nn.Linear(hidden_size, hidden_size)
        self.att_act = nn.Tanh()
        self.att_score = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_outputs)
        self.register_buffer("_last_attention", None, persistent=False)
        self._last_attention_raw: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        time_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                "TemporalAttentionHead expects input of shape [B, H, W, T, D], got "
                f"{tuple(x.shape)}"
            )

        scores = self.att_score(self.att_act(self.att_proj(x))).squeeze(-1)

        if self.use_prior and time_weights is not None:
            if time_weights.dim() == 5:
                prior = torch.log(time_weights.squeeze(-1).clamp_min(self.eps))
            else:
                prior = torch.log(time_weights.clamp_min(self.eps))
            scores = scores + prior

        alpha = torch.softmax(scores, dim=-1)
        self._last_attention = alpha.detach()
        self._last_attention_raw = alpha

        pooled = torch.sum(alpha.unsqueeze(-1) * x, dim=-2)
        logits = self.classifier(pooled)  # [B, H, W, num_outputs]
        logits = logits.unsqueeze(-2).expand(-1, -1, -1, x.shape[-2], -1)
        return logits


class WorldCerealPretrainedPresto(PretrainedPrestoWrapper):
    """Presto wrapper that supports swapping in custom heads within this repo."""

    def __init__(
        self,
        *,
        num_outputs: int,
        regression: bool,
        pretrained_model_path: Optional[Union[str, os.PathLike]] = None,
        temporal_attention: bool = False,
        attention_use_prior: bool = True,
    ) -> None:
        if regression and temporal_attention:
            raise ValueError("Temporal attention currently supports classification only")

        super().__init__(
            num_outputs=None,
            regression=None,
            pretrained_model_path=pretrained_model_path,
        )

        self.head = None  # ensure base wrapper does not apply its own head
        hidden_size = self.encoder.embedding_size
        self._regression = regression
        self._temporal_attention = temporal_attention
        self._default_pooling = PoolingMethods.TIME if temporal_attention else PoolingMethods.GLOBAL

        if temporal_attention:
            self.classification_head: nn.Module = TemporalAttentionHead(
                hidden_size=hidden_size,
                num_outputs=num_outputs,
                regression=regression,
                use_prior=attention_use_prior,
            )
        else:
            self.classification_head = FinetuningHead(
                hidden_size=hidden_size,
                num_outputs=num_outputs,
                regression=regression,
            )

    def forward(
        self,
        x: Predictors,
        eval_pooling: Union[PoolingMethods, None] = PoolingMethods.GLOBAL,
        time_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s1_s2_era5_srtm, mask, dynamic_world, latlon, timestamps, h, w = (
            dataset_to_model(x)
        )

        inferred_pooling: Union[PoolingMethods, None] = eval_pooling

        if x.label is not None:
            if x.label.shape[3] == dynamic_world.shape[1]:
                inferred_pooling = PoolingMethods.TIME
            else:
                if x.label.shape[1] != 1:
                    raise ValueError(f"Unexpected label shape {x.label.shape}")
                inferred_pooling = PoolingMethods.GLOBAL

        if inferred_pooling is None:
            inferred_pooling = self._default_pooling


        if x.timestamps is None:
            raise ValueError("Presto requires input timestamps")

        model_device = self.encoder.pos_embed.device
        embeddings = self.encoder(
            x=to_torchtensor(s1_s2_era5_srtm, device=model_device).float(),
            dynamic_world=to_torchtensor(dynamic_world, device=model_device).long(),
            latlons=to_torchtensor(latlon, device=model_device).float(),
            mask=to_torchtensor(mask, device=model_device).long(),
            month=to_torchtensor(timestamps[:, :, 1] - 1, device=model_device),
            eval_pooling=inferred_pooling.value if inferred_pooling is not None else None,
        )

        if inferred_pooling == PoolingMethods.GLOBAL:
            b = int(embeddings.shape[0] / (h * w))
            embeddings = rearrange(embeddings, "(b h w) d -> b h w d", b=b, h=h, w=w)
            embeddings = torch.unsqueeze(embeddings, 3)
        elif inferred_pooling == PoolingMethods.TIME:
            b = int(embeddings.shape[0] / (h * w))
            embeddings = rearrange(
                embeddings, "(b h w) t d -> b h w t d", b=b, h=h, w=w
            )
        else:
            if (h != 1) or (w != 1):
                raise ValueError("h w != 1 unsupported for SSL")

        if self._temporal_attention:
            logits = self.classification_head(embeddings, time_weights=time_weights)
        else:
            logits = self.classification_head(embeddings)
        return logits


def build_worldcereal_presto(
    *,
    num_outputs: int,
    regression: bool,
    pretrained_model_path: Optional[Union[str, os.PathLike]] = None,
    temporal_attention: bool = False,
    attention_use_prior: bool = True,
) -> WorldCerealPretrainedPresto:
    """Factory to construct a Presto wrapper with optional attention head."""

    return WorldCerealPretrainedPresto(
        num_outputs=num_outputs,
        regression=regression,
        pretrained_model_path=pretrained_model_path,
        temporal_attention=temporal_attention,
        attention_use_prior=attention_use_prior,
    )
