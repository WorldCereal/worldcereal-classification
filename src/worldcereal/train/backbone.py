"""Helpers for resolving and loading the seasonal Presto backbone."""

from __future__ import annotations

import hashlib
import shutil
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

from loguru import logger

from worldcereal.openeo.inference import DEFAULT_CACHE_ROOT, load_model_artifact
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_MODEL_URL


def checkpoint_fingerprint(path: Union[str, Path]) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@lru_cache(maxsize=4)
def resolve_seasonal_encoder(
    seasonal_model_url: Optional[str] = None,
) -> Tuple[str, str]:
    """Return the resolved encoder checkpoint path and fingerprint from a seasonal model."""

    model_url = seasonal_model_url or DEFAULT_SEASONAL_MODEL_URL
    artifact = load_model_artifact(model_url)
    manifest_backbone = artifact.manifest.get("backbone") or {}
    checkpoints_entry = artifact.manifest.get("artifacts", {}).get("checkpoints", {})
    checkpoint_rel = checkpoints_entry.get("encoder_only")
    if not checkpoint_rel:
        raise ValueError(
            f"Seasonal model manifest at {model_url} must include an encoder_only checkpoint entry."
        )

    checkpoint_path = _resolve_checkpoint_path(checkpoint_rel, artifact.extract_dir)

    if checkpoint_path:
        fingerprint = manifest_backbone.get("fingerprint") or checkpoint_fingerprint(
            checkpoint_path
        )
        return str(checkpoint_path), fingerprint

    raise FileNotFoundError(
        f"Backbone checkpoint '{checkpoint_rel}' not found for seasonal artifact at {artifact.extract_dir}."
    )


def build_presto_backbone(
    *,
    checkpoint_path: Optional[str] = None,
    seasonal_model_url: Optional[str] = None,
):
    """Instantiate a Presto wrapper with seasonal weights pre-loaded."""

    from prometheo.models import Presto
    from prometheo.models.presto.wrapper import load_presto_weights

    if checkpoint_path:
        ckpt = checkpoint_path
    else:
        ckpt, _ = resolve_seasonal_encoder(seasonal_model_url)

    # Construct the model
    model = Presto()

    # Load the weights and be strict to avoid silent issues
    model = load_presto_weights(model, ckpt, strict=True)

    return model


def _resolve_checkpoint_path(reference: str, extract_dir: Path) -> Optional[Path]:
    """Resolve a checkpoint path from manifest reference, downloading if needed."""

    # Try relative to the extracted artifact contents first.
    candidate = Path(extract_dir) / reference
    if candidate.exists():
        return candidate

    # Accept absolute filesystem paths.
    candidate = Path(reference)
    if candidate.exists():
        return candidate

    parsed = urllib.parse.urlparse(reference)
    if parsed.scheme in {"http", "https"}:
        return _download_checkpoint(reference)

    return None


def _download_checkpoint(reference: str) -> Path:
    cache_dir = DEFAULT_CACHE_ROOT / "backbones"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha256(reference.encode("utf-8")).hexdigest()[:16]
    filename = Path(urllib.parse.urlparse(reference).path).name or "presto.pt"
    target = cache_dir / f"{slug}_{filename}"
    if target.exists():
        return target

    logger.info("Downloading seasonal backbone checkpoint from {}", reference)
    with urllib.request.urlopen(reference) as response, open(target, "wb") as handle:  # nosec: B310
        shutil.copyfileobj(response, handle)
    return target
