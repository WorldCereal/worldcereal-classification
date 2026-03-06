"""Utilities around models for the WorldCereal package."""

import hashlib
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
)

from loguru import logger

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "worldcereal" / "models"


# ---------------------------------------------------------------------------
# Artifact loading utilities
# ---------------------------------------------------------------------------


@dataclass
class ModelArtifact:
    source: str
    zip_path: Path
    extract_dir: Path
    manifest: Dict[str, Any]
    run_config: Optional[Dict[str, Any]]
    checkpoint_path: Path


def ensure_cache_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "downloads").mkdir(exist_ok=True)
    (root / "extracted").mkdir(exist_ok=True)
    return root


def _hash_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _download_artifact(source: str, cache_root: Path) -> Path:
    parsed = urllib.parse.urlparse(source)
    downloads_dir = cache_root / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    if parsed.scheme in {"http", "https"}:
        slug = _hash_source(source)
        target = downloads_dir / f"{slug}.zip"
        if target.exists():
            return target
        logger.info(f"Downloading seasonal model artifact from {source}")
        with (
            urllib.request.urlopen(source) as resp,
            open(target, "wb") as fh,
        ):  # nosec: B310
            shutil.copyfileobj(resp, fh)
        return target
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {source}")
    return path


def _extract_artifact(zip_path: Path, cache_root: Path) -> Path:
    slug = (
        zip_path.stem
        if zip_path.parent == cache_root / "downloads"
        else _hash_source(str(zip_path))
    )
    extract_dir = cache_root / "extracted" / slug
    if extract_dir.exists():
        return extract_dir

    tmp_dir = Path(tempfile.mkdtemp(dir=cache_root / "extracted"))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        tmp_dir.rename(extract_dir)
        return extract_dir
    except Exception:  # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_checkpoint_path(
    manifest: Mapping[str, Any], extract_dir: Path, priority: Sequence[str]
) -> Path:
    artifacts = manifest.get("artifacts", {})
    checkpoints = artifacts.get("checkpoints", {})
    for key in priority:
        candidate = checkpoints.get(key)
        if candidate:
            candidate_path = extract_dir / candidate
            if candidate_path.exists():
                return candidate_path
    pt_files = list(extract_dir.glob("*.pt"))
    if len(pt_files) == 1:
        return pt_files[0]
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint found in {extract_dir}")
    raise FileNotFoundError(
        "Multiple .pt files found; manifest must declare the checkpoint name explicitly"
    )


def load_model_artifact(
    source: str | Path,
    cache_root: Optional[Path] = None,
    encoder_only: bool = False,
) -> ModelArtifact:
    """Download, extract and load a model artifact package."""
    cache_root = ensure_cache_dir(cache_root or DEFAULT_CACHE_ROOT)
    zip_path = _download_artifact(str(source), cache_root)
    extract_dir = _extract_artifact(zip_path, cache_root)

    manifest = _load_json(extract_dir / "config.json")
    run_config = (
        _load_json(extract_dir / "run_config.json")
        if (extract_dir / "run_config.json").exists()
        else None
    )
    if encoder_only:
        priority = ("encoder_only", "model")
    else:
        priority = ("full", "model")
    checkpoint = resolve_checkpoint_path(manifest, extract_dir, priority=priority)

    return ModelArtifact(
        source=str(source),
        zip_path=zip_path,
        extract_dir=extract_dir,
        manifest=manifest,
        run_config=run_config,
        checkpoint_path=checkpoint,
    )
