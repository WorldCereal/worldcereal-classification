"""Utilities around models for the WorldCereal package."""

import hashlib
import json
import logging
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


def _get_url(url: str) -> bytes:

    # We use a dual approach for artifact downloading:
    # 1. Try httpx with HTTP/2 support, as it is the only approach working in Terrascope notebooks.
    # 2. Fall back to urllib for environments where httpx is not available or fails (inside openEO backend).
    try:
        import httpx

        with httpx.Client(http2=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content
    except Exception:  # noqa: BLE001
        with urllib.request.urlopen(url) as resp:  # nosec: B310
            return resp.read()

try:
    from loguru import logger
except ImportError:
    # loguru not available, use standard logging
    logger = logging.getLogger(__name__)  # type: ignore

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
        target.write_bytes(_get_url(source))
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


SUPPORTED_MASKABLE_MODALITIES = {"s1", "s2", "meteo", "dem"}


def create_masked_seasonal_artifact(
    base_source: str | Path,
    excluded_modalities: Sequence[str],
    output_dir: str | Path,
    output_name: Optional[str] = None,
    cache_root: Optional[Path] = None,
) -> Path:
    """Repackage a seasonal model artifact with sensor-disable flags baked in.

    A downstream head trained with `excluded_modalities` is only ever
    consistent at inference if the seasonal model suite it is deployed
    with actually skips loading those same sensors. Since the backbone
    itself is shared/frozen, this derives a new zip (same encoder,
    checkpoints and manifest as `base_source`) whose `run_config.json`
    records `excluded_modalities` as `disable_*` flags, so
    `worldcereal.job._get_disabled_modalities` (and the inference engine's
    own masking) correctly skip those sensors for this specific model
    suite. Any embedded "landcover" head is dropped from the manifest,
    since it was trained with the full sensor set and is not guaranteed
    compatible with the excluded modalities.

    Parameters
    ----------
    base_source : str | Path
        URL or local path to the base seasonal model artifact to derive from.
    excluded_modalities : Sequence[str]
        Modalities to mark as disabled. Supported values are `s1`, `s2`,
        `meteo` and `dem`.
    output_dir : str | Path
        Directory in which the derived zip is written.
    output_name : Optional[str]
        Base name (without extension) for the derived zip. Defaults to a
        name derived from `base_source` and the excluded modalities.
    cache_root : Optional[Path]
        Cache root used to resolve/download `base_source`.

    Returns
    -------
    Path
        Path to the newly created zip archive.
    """
    excluded = sorted({str(modality).lower() for modality in excluded_modalities})
    invalid = set(excluded) - SUPPORTED_MASKABLE_MODALITIES
    if invalid:
        raise ValueError(f"Unsupported excluded modalities: {', '.join(sorted(invalid))}")
    if {"s1", "s2"}.issubset(excluded):
        raise ValueError("S1 and S2 cannot both be excluded from a model suite.")

    base_artifact = load_model_artifact(str(base_source), cache_root=cache_root)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(dir=output_dir))
    try:
        shutil.copytree(base_artifact.extract_dir, work_dir, dirs_exist_ok=True)

        manifest_path = work_dir / "config.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        heads = manifest.get("heads", [])
        dropped = [head.get("name", head.get("task")) for head in heads if head.get("task") == "landcover"]
        manifest["heads"] = [head for head in heads if head.get("task") != "landcover"]
        if dropped:
            logger.info(
                f"Dropping embedded landcover head(s) {dropped} from derived seasonal "
                "model artifact: not guaranteed compatible with excluded modalities."
            )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        run_config_path = work_dir / "run_config.json"
        run_config = (
            json.loads(run_config_path.read_text(encoding="utf-8"))
            if run_config_path.exists()
            else {}
        )
        args = dict(run_config.get("args") or {})
        for modality in SUPPORTED_MASKABLE_MODALITIES:
            args[f"disable_{modality}"] = bool(
                args.get(f"disable_{modality}", False) or modality in excluded
            )
        run_config["args"] = args
        run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        name = output_name or (
            f"{Path(str(base_source)).stem}_excl-{'-'.join(excluded) or 'none'}"
        )
        zip_path = output_dir / f"{name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in work_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(work_dir))
        return zip_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
