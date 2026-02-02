#!/usr/bin/env python3
"""Seasonal inference utilities for WorldCereal Presto models."""

import datetime
import hashlib
import json
import logging
import random
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

try:  # Python 3.10+
    from typing import TypeAlias
except ImportError:  # pragma: no cover - fallback for older runtimes
    from typing_extensions import TypeAlias  # type: ignore[misc, assignment]

import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import convolve, zoom

try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            level = record.levelname
            logger.opt(depth=6).log(level, record.getMessage())

    # Replace existing handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(InterceptHandler())

except ImportError:
    # loguru not available, use standard logging
    logger = logging.getLogger(__name__)  # type: ignore


from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prometheo.predictors import Predictors

    from worldcereal.train.seasonal_head import WorldCerealSeasonalModel

    try:
        from torch import Tensor as _TorchTensorType
        from torch import device as _TorchDeviceType
    except Exception:  # pragma: no cover - stubs not available
        _TorchTensorType = Any  # type: ignore[assignment]
        _TorchDeviceType = Any  # type: ignore[assignment]
else:  # pragma: no cover - runtime avoids importing torch eagerly
    _TorchTensorType = Any  # type: ignore[assignment]
    _TorchDeviceType = Any  # type: ignore[assignment]

TorchTensor: TypeAlias = _TorchTensorType
TorchDevice: TypeAlias = _TorchDeviceType


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "PyTorch is required for seasonal inference. When running inside openEO, "
            "call _require_openeo_runtime() first so feature dependencies are available."
        ) from exc
    return torch


def _seasonal_workflow_presets():
    from worldcereal.openeo import parameters as _seasonal_parameters

    return (
        _seasonal_parameters.DEFAULT_SEASONAL_WORKFLOW_PRESET,
        _seasonal_parameters.SEASONAL_WORKFLOW_PRESETS,
    )


# ---------------------------------------------------------------------------
# Constants shared with legacy inference pipeline
# ---------------------------------------------------------------------------

GFMAP_BAND_MAPPING = {
    "S2-L2A-B02": "B2",
    "S2-L2A-B03": "B3",
    "S2-L2A-B04": "B4",
    "S2-L2A-B05": "B5",
    "S2-L2A-B06": "B6",
    "S2-L2A-B07": "B7",
    "S2-L2A-B08": "B8",
    "S2-L2A-B8A": "B8A",
    "S2-L2A-B11": "B11",
    "S2-L2A-B12": "B12",
    "S1-SIGMA0-VH": "VH",
    "S1-SIGMA0-VV": "VV",
    "AGERA5-TMEAN": "temperature_2m",
    "AGERA5-PRECIP": "total_precipitation",
}

S1_INPUT_BANDS = ["S1-SIGMA0-VV", "S1-SIGMA0-VH"]
NODATA_VALUE = 65535
NOCROP_VALUE = 254

POSTPROCESSING_EXCLUDED_VALUES = [NOCROP_VALUE, 255, 65535]
POSTPROCESSING_NODATA = 255
DEFAULT_POSTPROCESS_METHOD = "majority_vote"


@dataclass
class PostprocessOptions:
    enabled: bool = False
    method: Optional[str] = None
    kernel_size: int = 5

    def resolved_method(self) -> Optional[str]:
        if not self.enabled:
            return None
        return self.method or DEFAULT_POSTPROCESS_METHOD


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "worldcereal" / "models"


SeasonDateLike = Union[str, datetime.date, datetime.datetime, np.datetime64]
SeasonWindowValue = Union[
    Tuple[SeasonDateLike, SeasonDateLike],
    Sequence[Tuple[SeasonDateLike, SeasonDateLike]],
]


# ---------------------------------------------------------------------------
# DEM helpers reused from the legacy pipeline
# ---------------------------------------------------------------------------


class SlopeCalculator:
    """Utility that computes slope layers from DEM inputs."""

    @staticmethod
    def compute(resolution: float, dem: np.ndarray) -> np.ndarray:
        prepared = SlopeCalculator._prepare_dem_array(dem)
        downsampled = SlopeCalculator._downsample_to_20m(prepared, resolution)
        gradient = SlopeCalculator._compute_slope_gradient(downsampled)
        return SlopeCalculator._upsample_to_original(gradient, dem.shape, resolution)

    @staticmethod
    def _prepare_dem_array(dem: np.ndarray) -> np.ndarray:
        dem_arr = dem.astype(np.float32)
        dem_arr[dem_arr == NODATA_VALUE] = np.nan
        return SlopeCalculator._fill_nans(dem_arr)

    @staticmethod
    def _fill_nans(dem_arr: np.ndarray, max_iter: int = 2) -> np.ndarray:
        if max_iter == 0 or not np.any(np.isnan(dem_arr)):
            return dem_arr

        mask = np.isnan(dem_arr)
        roll_params = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(roll_params)

        for shift_x, shift_y in roll_params:
            rolled = np.roll(dem_arr, shift_x, axis=0)
            rolled = np.roll(rolled, shift_y, axis=1)
            dem_arr[mask] = rolled[mask]

        return SlopeCalculator._fill_nans(dem_arr, max_iter - 1)

    @staticmethod
    def _downsample_to_20m(dem_arr: np.ndarray, resolution: float) -> np.ndarray:
        factor = int(20 / resolution)
        if factor < 1 or factor % 2 != 0:
            raise ValueError(
                f"Unsupported resolution for slope computation: {resolution}"
            )

        x_size, y_size = dem_arr.shape
        pad_x = (factor - (x_size % factor)) % factor
        pad_y = (factor - (y_size % factor)) % factor
        padded = np.pad(dem_arr, ((0, pad_x), (0, pad_y)), mode="reflect")

        reshaped = padded.reshape(
            (x_size + pad_x) // factor, factor, (y_size + pad_y) // factor, factor
        )
        return np.nanmean(reshaped, axis=(1, 3))

    @staticmethod
    def _compute_slope_gradient(dem: np.ndarray) -> np.ndarray:
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8.0 * 20)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8.0 * 20)

        dx = convolve(dem, kernel_x, mode="nearest")
        dy = convolve(dem, kernel_y, mode="nearest")
        gradient = np.sqrt(dx**2 + dy**2)
        return np.arctan(gradient) * (180 / np.pi)

    @staticmethod
    def _upsample_to_original(
        slope: np.ndarray, original_shape: Tuple[int, int], resolution: float
    ) -> np.ndarray:
        factor = int(20 / resolution)
        slope_upsampled = zoom(slope, zoom=factor, order=1)

        if original_shape[0] % 2 != 0:
            slope_upsampled = slope_upsampled[:-1, :]
        if original_shape[1] % 2 != 0:
            slope_upsampled = slope_upsampled[:, :-1]

        return slope_upsampled.astype(np.uint16)


class CoordinateTransformer:
    """Minimal helpers for resolution estimation and coordinate transforms."""

    @staticmethod
    def get_resolution(arr: xr.DataArray, epsg: int) -> float:
        if epsg == 4326:
            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            pts = [
                transformer.transform(arr.x.values[i], arr.y.values[0])
                for i in range(2)
            ]
            return abs(pts[1][0] - pts[0][0])
        return abs(float(arr.x.values[1] - arr.x.values[0]))


class DataPreprocessor:
    """Apply harmonization/rescaling expected by the predictor builder."""

    @staticmethod
    def rescale_s1_backscatter(arr: xr.DataArray) -> xr.DataArray:
        present = [b for b in S1_INPUT_BANDS if b in arr.bands.values]
        if not present:
            return arr
        if not np.issubdtype(arr.dtype, np.floating):
            # Allow negative dB values to be written back safely
            arr = arr.astype("float32")
        s1 = arr.sel(bands=present).astype(np.float32)
        data = s1.values
        nodata_mask = data == NODATA_VALUE
        valid_mask = ~nodata_mask
        scaled = np.full_like(data, NODATA_VALUE, dtype=np.float32)
        if np.any(valid_mask):
            DataPreprocessor._validate_s1_data(data[valid_mask])
            power = 20.0 * np.log10(data[valid_mask]) - 83.0
            power = np.power(10, power / 10.0)
            power[~np.isfinite(power)] = np.nan
            scaled[valid_mask] = 10.0 * np.log10(power)
        arr.loc[{"bands": present}] = scaled
        return arr

    @staticmethod
    def _validate_s1_data(data: np.ndarray) -> None:
        if data.min() < 1 or data.max() > NODATA_VALUE:
            raise ValueError(
                "S1 data expected as uint16 in range 1-65535 before rescaling."
            )

    @staticmethod
    def add_slope_band(arr: xr.DataArray, epsg: int) -> xr.DataArray:
        if "slope" in arr.bands.values:
            return arr
        if "COP-DEM" not in arr.bands.values:
            logger.warning("DEM band missing; slope band cannot be created.")
            return arr
        resolution = CoordinateTransformer.get_resolution(arr.isel(t=0), epsg)
        dem = arr.sel(bands="COP-DEM").isel(t=0).values
        slope = SlopeCalculator.compute(resolution, dem)
        slope_da = (
            xr.DataArray(
                slope[None, :, :],
                dims=("bands", "y", "x"),
                coords={"bands": ["slope"], "y": arr.y, "x": arr.x},
            )
            .expand_dims({"t": arr.t})
            .astype("float32")
        )
        return xr.concat([arr.astype("float32"), slope_da], dim="bands")


# ---------------------------------------------------------------------------
# Postprocessing helpers
# ---------------------------------------------------------------------------


def _gather_probabilities_for_labels(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_value_to_index: Mapping[int, int],
) -> np.ndarray:
    """Pick per-pixel probabilities corresponding to the provided labels."""

    result = np.zeros(labels.shape, dtype=np.float32)
    for class_value, prob_index in class_value_to_index.items():
        mask = labels == class_value
        if not np.any(mask):
            continue
        result[mask] = probabilities[prob_index][mask]
    return result


def _majority_vote_labels(
    labels: np.ndarray,
    *,
    kernel_size: int,
    excluded_values: Sequence[int],
) -> np.ndarray:
    from scipy.signal import convolve2d

    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1 for majority vote")
    if kernel_size == 1:
        return labels
    if kernel_size > 25:
        raise ValueError("kernel_size cannot exceed 25 for majority vote")

    valid_mask = ~np.isin(labels, excluded_values)
    if not np.any(valid_mask):
        return labels

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint16)
    unique_labels = sorted(np.unique(labels[valid_mask]))
    counts = np.zeros((len(unique_labels), *labels.shape), dtype=np.uint32)

    for idx, class_value in enumerate(unique_labels):
        class_mask = (labels == class_value).astype(np.uint8)
        counts[idx] = convolve2d(class_mask, kernel, mode="same", boundary="symm")

    winning_indices = np.argmax(counts, axis=0)
    new_labels = labels.copy()
    for idx, class_value in enumerate(unique_labels):
        vote_mask = (winning_indices == idx) & valid_mask
        if np.any(vote_mask):
            new_labels[vote_mask] = class_value
    return new_labels


def _smooth_probability_cube(
    probabilities: np.ndarray, excluded_mask: np.ndarray
) -> np.ndarray:
    from scipy.signal import convolve2d

    kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]], dtype=np.float32)
    kernel_sum = float(kernel.sum())
    smoothed = np.zeros_like(probabilities, dtype=np.float32)
    for idx in range(probabilities.shape[0]):
        smoothed[idx] = (
            convolve2d(probabilities[idx], kernel, mode="same", boundary="symm")
            / kernel_sum
        )
        smoothed[idx][excluded_mask] = 0.0
    norm = smoothed.sum(axis=0, keepdims=True)
    norm[norm == 0.0] = 1.0
    return smoothed / norm


def _labels_from_probability_cube(
    probabilities: np.ndarray,
    class_value_to_index: Mapping[int, int],
    base_labels: np.ndarray,
    excluded_mask: np.ndarray,
) -> np.ndarray:
    best_idx = np.argmax(probabilities, axis=0)
    new_labels = base_labels.copy()
    for class_value, prob_index in class_value_to_index.items():
        vote_mask = (best_idx == prob_index) & ~excluded_mask
        if np.any(vote_mask):
            new_labels[vote_mask] = class_value
    return new_labels


def _run_postprocess(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_value_to_index: Mapping[int, int],
    options: PostprocessOptions,
    excluded_values: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    method = options.resolved_method()
    excluded_mask = np.isin(labels, excluded_values)
    prob_cube = probabilities.astype(np.float32, copy=False)

    if method is None:
        updated_labels = labels
    elif method == "majority_vote":
        updated_labels = _majority_vote_labels(
            labels,
            kernel_size=options.kernel_size,
            excluded_values=excluded_values,
        )
    elif method == "smooth_probabilities":
        smoothed = _smooth_probability_cube(prob_cube, excluded_mask)
        prob_cube = smoothed
        updated_labels = _labels_from_probability_cube(
            smoothed,
            class_value_to_index,
            base_labels=labels,
            excluded_mask=excluded_mask,
        )
    else:
        raise ValueError(f"Unknown postprocess method '{method}'")

    updated_probabilities = _gather_probabilities_for_labels(
        prob_cube, updated_labels, class_value_to_index
    )
    return updated_labels, updated_probabilities, prob_cube


# ---------------------------------------------------------------------------
# Artifact loading utilities
# ---------------------------------------------------------------------------


@dataclass
class HeadSpec:
    task: str
    class_names: List[str]
    cropland_classes: List[str]
    gating_enabled: bool
    head_type: str = "linear"
    hidden_dim: int = 256
    dropout: float = 0.0

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass
class ModelArtifact:
    source: str
    zip_path: Path
    extract_dir: Path
    manifest: Dict[str, Any]
    run_config: Optional[Dict[str, Any]]
    checkpoint_path: Path


def _ensure_cache_dir(root: Path) -> Path:
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
        with urllib.request.urlopen(source) as resp, open(target, "wb") as fh:  # nosec: B310
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


def _resolve_checkpoint_path(
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
    source: str | Path, cache_root: Optional[Path] = None
) -> ModelArtifact:
    cache_root = _ensure_cache_dir(cache_root or DEFAULT_CACHE_ROOT)
    zip_path = _download_artifact(str(source), cache_root)
    extract_dir = _extract_artifact(zip_path, cache_root)
    manifest = _load_json(extract_dir / "config.json")
    run_config = None
    run_config_path = extract_dir / "run_config.json"
    if run_config_path.exists():
        run_config = _load_json(run_config_path)
    checkpoint = _resolve_checkpoint_path(
        manifest, extract_dir, priority=("full", "model")
    )
    return ModelArtifact(
        source=str(source),
        zip_path=zip_path,
        extract_dir=extract_dir,
        manifest=manifest,
        run_config=run_config,
        checkpoint_path=checkpoint,
    )


def _select_head_spec(heads: Iterable[Mapping[str, Any]], task: str) -> HeadSpec:
    for head in heads:
        if head.get("task") == task:
            class_names = [str(cls) for cls in head.get("class_names", [])]
            if not class_names:
                raise ValueError(f"Head '{task}' missing class_names in manifest")
            gating_cfg = head.get("gating", {})
            cropland_classes = [
                str(cls) for cls in gating_cfg.get("cropland_classes", [])
            ]
            return HeadSpec(
                task=task,
                class_names=class_names,
                cropland_classes=cropland_classes,
                gating_enabled=bool(gating_cfg.get("enabled", False)),
                head_type=str(head.get("head_type", "linear")),
                hidden_dim=int(head.get("hidden_dim", 256)),
                dropout=float(head.get("dropout", 0.0)),
            )
    raise ValueError(f"Manifest does not define a '{task}' head")


# ---------------------------------------------------------------------------
# Seasonal model bundle (loads backbone and optional replacement heads)
# ---------------------------------------------------------------------------


class SeasonalModelBundle:
    """Convenience wrapper that owns the seasonal model and metadata.

    Custom head overrides are validated to ensure every head uses a compatible
    Presto backbone checkpoint so embeddings stay aligned across tasks.
    """

    def __init__(
        self,
        base_artifact: ModelArtifact,
        *,
        landcover_head_zip: str | Path | None = None,
        croptype_head_zip: str | Path | None = None,
        cache_root: Optional[Path] = None,
        device: Union[str, TorchDevice] = "cpu",
        enable_croptype_head: bool = True,
        enable_cropland_head: bool = True,
    ) -> None:
        torch = _lazy_import_torch()
        self.device = torch.device(device)
        self.base_artifact = base_artifact
        self.cache_root = _ensure_cache_dir(cache_root or DEFAULT_CACHE_ROOT)
        self._croptype_head_enabled = enable_croptype_head
        self._cropland_head_enabled = enable_cropland_head

        if not (self._croptype_head_enabled or self._cropland_head_enabled):
            raise ValueError(
                "Seasonal model bundle requires at least one head to be enabled."
            )

        heads = base_artifact.manifest.get("heads", [])
        self.landcover_spec = _select_head_spec(heads, task="landcover")
        self.croptype_spec = _select_head_spec(heads, task="croptype")
        self.cropland_gate_classes: List[str] = []
        self._base_backbone_checkpoint = (
            base_artifact.manifest.get("backbone", {}) or {}
        ).get("pretrained_checkpoint")
        self._custom_head_backbone_checkpoints: Dict[str, Optional[str]] = {}

        dropout = base_artifact.manifest.get("backbone", {}).get("head_dropout", 0.0)
        self.model = self._build_model(dropout)

        if landcover_head_zip:
            if not self._cropland_head_enabled:
                logger.info(
                    "Cropland head disabled; ignoring custom landcover head override."
                )
            else:
                self._apply_custom_head(
                    task="landcover",
                    source=landcover_head_zip,
                    priority=("head", "full", "model"),
                )
        if enable_croptype_head and croptype_head_zip:
            self._apply_custom_head(
                task="croptype",
                source=croptype_head_zip,
                priority=("head", "full", "model"),
            )
        if croptype_head_zip and not enable_croptype_head:
            logger.info(
                "Croptype head override provided but croptype head disabled; override ignored."
            )
        if not self._cropland_head_enabled:
            logger.info(
                "Cropland head disabled by configuration; cropland outputs will be skipped."
            )
        self._update_cropland_gate()

    def _build_model(self, dropout: float) -> "WorldCerealSeasonalModel":
        torch = _lazy_import_torch()
        from prometheo.models import Presto

        from worldcereal.train.seasonal_head import (
            SeasonalFinetuningHead,
            WorldCerealSeasonalModel,
        )

        backbone = Presto()
        head = SeasonalFinetuningHead(
            embedding_dim=backbone.encoder.embedding_size,
            landcover_num_outputs=(
                self.landcover_spec.num_classes if self._cropland_head_enabled else None
            ),
            crop_num_outputs=(
                self.croptype_spec.num_classes if self._croptype_head_enabled else None
            ),
            dropout=dropout,
            landcover_head_type=self.landcover_spec.head_type,
            croptype_head_type=self.croptype_spec.head_type,
            landcover_hidden_dim=self.landcover_spec.hidden_dim,
            croptype_hidden_dim=self.croptype_spec.hidden_dim,
        )
        model = WorldCerealSeasonalModel(backbone=backbone, head=head)
        state_dict = torch.load(
            self.base_artifact.checkpoint_path, map_location=self.device
        )
        if not self._croptype_head_enabled:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("head.crop_head")
            }
        if not self._cropland_head_enabled:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("head.landcover_head")
            }
        model.load_state_dict(state_dict)
        return model.to(self.device).eval()

    def _apply_custom_head(
        self,
        *,
        task: str,
        source: str | Path,
        priority: Sequence[str],
    ) -> None:
        torch = _lazy_import_torch()
        artifact = load_model_artifact(source, cache_root=self.cache_root)
        backbone_override = (artifact.manifest.get("backbone", {}) or {}).get(
            "pretrained_checkpoint"
        )
        self._validate_backbone_override(task, backbone_override)
        checkpoint = _resolve_checkpoint_path(
            artifact.manifest, artifact.extract_dir, priority
        )
        state_dict = torch.load(checkpoint, map_location=self.device)
        head_spec = _select_head_spec(artifact.manifest.get("heads", []), task)
        if head_spec.head_type != "linear":
            self.model.head.replace_head(
                task=task,
                num_outputs=head_spec.num_classes,
                head_type=head_spec.head_type,
                hidden_dim=head_spec.hidden_dim,
                dropout=head_spec.dropout,
            )
        if task == "landcover":
            if not self._cropland_head_enabled:
                logger.info(
                    "Cropland head disabled; ignoring custom landcover head override."
                )
                return
            module = self.model.head.landcover_head
            if module is None:
                raise ValueError(
                    "Base model missing landcover head; cannot apply override"
                )
            missing, unexpected = module.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logger.warning(
                    f"Custom landcover head state dict mismatch. Missing={missing}, unexpected={unexpected}"
                )
            self.landcover_spec = head_spec
        elif task == "croptype":
            module = self.model.head.crop_head
            if module is None:
                raise ValueError(
                    "Base model missing croptype head; cannot apply override"
                )
            missing, unexpected = module.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logger.warning(
                    f"Custom croptype head state dict mismatch. Missing={missing}, unexpected={unexpected}"
                )
            self.croptype_spec = head_spec
        else:
            raise ValueError(f"Unknown head task '{task}'")
        self._update_cropland_gate()

    def _validate_backbone_override(
        self, task: str, override_checkpoint: Optional[str]
    ) -> None:
        if override_checkpoint:
            if (
                self._base_backbone_checkpoint
                and override_checkpoint != self._base_backbone_checkpoint
            ):
                raise ValueError(
                    "Custom head backbone checkpoint does not match seasonal model "
                    f"backbone ({override_checkpoint} != {self._base_backbone_checkpoint})."
                )
            for (
                other_task,
                other_checkpoint,
            ) in self._custom_head_backbone_checkpoints.items():
                if other_checkpoint and other_checkpoint != override_checkpoint:
                    raise ValueError(
                        "Custom head backbone checkpoints must match across tasks "
                        f"({task}={override_checkpoint} != {other_task}={other_checkpoint})."
                    )
            self._custom_head_backbone_checkpoints[task] = override_checkpoint
            return

        if self._base_backbone_checkpoint:
            return

        for (
            other_task,
            other_checkpoint,
        ) in self._custom_head_backbone_checkpoints.items():
            if other_checkpoint:
                raise ValueError(
                    "Custom head backbone checkpoint missing for task "
                    f"'{task}', but '{other_task}' specifies '{other_checkpoint}'."
                )
    def _update_cropland_gate(self) -> None:
        if not self._cropland_head_enabled:
            self.cropland_gate_classes = []
            logger.info("Cropland head disabled; cropland gating unavailable.")
            return
        if self.landcover_spec.cropland_classes:
            self.cropland_gate_classes = list(self.landcover_spec.cropland_classes)
        elif self.croptype_spec.cropland_classes:
            self.cropland_gate_classes = list(self.croptype_spec.cropland_classes)
        else:
            self.cropland_gate_classes = []
        logger.info(
            f"Cropland gating enabled for classes: {self.cropland_gate_classes}"
        )


# ---------------------------------------------------------------------------
# Season mask helpers
# ---------------------------------------------------------------------------


def _coerce_datetime64(value: SeasonDateLike) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    if isinstance(value, datetime.datetime):
        return np.datetime64(value.date(), "D")
    if isinstance(value, datetime.date):
        return np.datetime64(value, "D")
    if isinstance(value, str):
        try:
            return np.datetime64(value, "D")
        except ValueError as exc:  # pragma: no cover - bad user input
            raise ValueError(f"Could not parse season window date '{value}'.") from exc
    raise TypeError(f"Unsupported season date type: {type(value)!r}")


def _ensure_datetime64_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return arr.astype("datetime64[D]")
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]")
    try:
        return arr.astype("datetime64[D]")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Season mask construction requires datetime-like 't' coordinates."
        ) from exc


def _normalize_season_windows_input(
    season_windows: Optional[Mapping[str, SeasonWindowValue]],
) -> Dict[str, List[Tuple[np.datetime64, np.datetime64]]]:
    if not season_windows:
        return {}

    normalized: Dict[str, List[Tuple[np.datetime64, np.datetime64]]] = {}
    for season_id, raw_value in season_windows.items():
        if raw_value is None:
            raise ValueError(f"Season '{season_id}' requires at least one window")

        if isinstance(raw_value, tuple):
            candidate_pairs: List[Sequence[SeasonDateLike]] = [raw_value]
        elif isinstance(raw_value, list):
            if raw_value and not isinstance(raw_value[0], (tuple, list)):
                candidate_pairs = [tuple(raw_value)]
            else:
                candidate_pairs = raw_value  # type: ignore[assignment]
        else:
            raise TypeError(
                f"Season '{season_id}' windows must be tuples or lists, got {type(raw_value)!r}"
            )

        normalized_pairs: List[Tuple[np.datetime64, np.datetime64]] = []
        for pair in candidate_pairs:  # type: ignore[arg-type]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    f"Season '{season_id}' windows must be (start, end) tuples, got {pair!r}"
                )
            start_raw, end_raw = pair
            start_dt = _coerce_datetime64(start_raw)
            end_dt = _coerce_datetime64(end_raw)
            if end_dt < start_dt:
                raise ValueError(
                    f"Season '{season_id}' window end {end_dt} is before start {start_dt}."
                )
            normalized_pairs.append((start_dt, end_dt))

        if not normalized_pairs:
            raise ValueError(f"Season '{season_id}' windows cannot be empty")
        normalized[season_id] = normalized_pairs

    return normalized


def _build_masks_from_windows(
    timestamps: np.ndarray,
    season_ids: Sequence[str],
    windows: Mapping[str, List[Tuple[np.datetime64, np.datetime64]]],
    batch_size: int,
    composite_frequency: Optional[Literal["month", "dekad"]] = "month",
) -> np.ndarray:
    ts_arr = np.asarray(timestamps).astype("datetime64[D]")
    num_timesteps = ts_arr.shape[0]
    coverage_start = ts_arr.min()
    coverage_end = ts_arr.max()
    align_fn = None
    if composite_frequency in {"month", "dekad"}:
        from worldcereal.train.seasonal import align_to_composite_window

        align_fn = align_to_composite_window
    base = np.zeros((len(season_ids), num_timesteps), dtype=bool)
    for idx, season_id in enumerate(season_ids):
        season_windows = windows.get(season_id)
        if not season_windows:
            raise ValueError(
                f"No season windows supplied for '{season_id}' but it was requested."
            )
        mask = np.zeros(num_timesteps, dtype=bool)
        for start, end in season_windows:
            start_aligned = start
            end_aligned = end
            if align_fn is not None and composite_frequency is not None:
                start_aligned = align_fn(start, composite_frequency)
                end_aligned = align_fn(end, composite_frequency)
            if start_aligned < coverage_start or end_aligned > coverage_end:
                raise ValueError(
                    f"Season window for '{season_id}' extends beyond available timestamps "
                    f"({coverage_start} to {coverage_end})."
                )
            mask |= (ts_arr >= start_aligned) & (ts_arr <= end_aligned)
        if not mask.any():
            raise ValueError(
                f"Season '{season_id}' window does not overlap any available timesteps."
            )
        base[idx] = mask
    return np.repeat(base[None, ...], batch_size, axis=0)


def _build_uniform_masks(
    batch_size: int, num_timesteps: int, num_seasons: int
) -> np.ndarray:
    base = np.ones((num_seasons, num_timesteps), dtype=bool)
    return np.repeat(base[None, ...], batch_size, axis=0)


def _normalize_provided_masks(
    masks: np.ndarray,
    batch_size: int,
    num_timesteps: int,
) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 2:
        arr = np.repeat(arr[None, ...], batch_size, axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] not in {1, batch_size}:
            raise ValueError(
                "Provided season masks must have shape [S, T] or [B, S, T]."
            )
        if arr.shape[0] == 1 and batch_size > 1:
            arr = np.repeat(arr, batch_size, axis=0)
    else:
        raise ValueError("Season masks must be 2D or 3D arrays.")

    if arr.shape[0] != batch_size or arr.shape[2] != num_timesteps:
        raise ValueError(
            "Season masks have incompatible batch or timestep dimensions for this cube."
        )
    return arr.astype(bool, copy=False)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class SeasonalInferenceEngine:
    """High-level orchestrator that runs seasonal inference on xarray cubes."""

    def __init__(
        self,
        *,
        seasonal_model_zip: str | Path,
        landcover_head_zip: str | Path | None = None,
        croptype_head_zip: str | Path | None = None,
        cache_root: Optional[Path] = None,
        device: Union[str, TorchDevice] = "cpu",
        season_ids: Optional[Sequence[str]] = None,
        season_windows: Optional[Mapping[str, SeasonWindowValue]] = None,
        batch_size: int = 2048,
        season_composite_frequency: Optional[Literal["month", "dekad"]] = "month",
        export_class_probabilities: bool = False,
        enable_croptype_head: bool = True,
        enable_cropland_head: bool = True,
        cropland_postprocess: Optional[Mapping[str, Any]] = None,
        croptype_postprocess: Optional[Mapping[str, Any]] = None,
    ) -> None:
        base_artifact = load_model_artifact(seasonal_model_zip, cache_root=cache_root)
        self.bundle = SeasonalModelBundle(
            base_artifact,
            landcover_head_zip=landcover_head_zip,
            croptype_head_zip=croptype_head_zip,
            cache_root=cache_root,
            device=device,
            enable_croptype_head=enable_croptype_head,
            enable_cropland_head=enable_cropland_head,
        )
        torch = _lazy_import_torch()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._season_composite_frequency = season_composite_frequency
        self._export_class_probabilities = export_class_probabilities
        self._croptype_enabled = enable_croptype_head
        self._cropland_enabled = enable_cropland_head
        self._cropland_postprocess = _build_postprocess_options(cropland_postprocess)
        self._croptype_postprocess = _build_postprocess_options(croptype_postprocess)
        from worldcereal.train import GLOBAL_SEASON_IDS

        if not self._croptype_enabled:
            logger.info(
                "Croptype head disabled by configuration; seasonal outputs will not be emitted."
            )
        if not self._cropland_enabled:
            logger.info(
                "Cropland head disabled by configuration; cropland outputs will not be emitted."
            )

        default_ids = list(GLOBAL_SEASON_IDS)
        if season_ids is not None and len(season_ids) == 0:
            raise ValueError("season_ids cannot be empty when provided")
        self._default_season_ids = list(season_ids) if season_ids else default_ids
        self._default_season_windows = _normalize_season_windows_input(season_windows)
        logger.info(
            f"SeasonalInferenceEngine initialized (device={self.device}, batch_size={self.batch_size}, "
            f"default_seasons={self._default_season_ids}, export_probs={self._export_class_probabilities}, "
            f"croptype_enabled={self._croptype_enabled}, cropland_enabled={self._cropland_enabled}, "
            f"cropland_postprocess={self._cropland_postprocess.enabled}, "
            f"croptype_postprocess={self._croptype_postprocess.enabled})"
        )

    def infer(
        self,
        arr: xr.DataArray,
        epsg: int,
        *,
        enforce_cropland_gate: bool = True,
        season_windows: Optional[Mapping[str, SeasonWindowValue]] = None,
        season_masks: Optional[np.ndarray] = None,
        season_ids: Optional[Sequence[str]] = None,
    ) -> xr.Dataset:
        dims_summary = {dim: size for dim, size in zip(arr.dims, arr.shape)}
        logger.info(
            f"Seasonal inference request received (epsg={epsg}, enforce_gate={enforce_cropland_gate}, "
            f"dims={dims_summary})"
        )
        prepped = self._prepare_array(arr, epsg)
        from worldcereal.train.predictors import generate_predictor

        prepped_summary = {dim: size for dim, size in zip(prepped.dims, prepped.shape)}
        logger.debug(
            f"Prepared inference cube (dims={prepped_summary}, dtype={prepped.dtype})"
        )
        predictors = generate_predictor(prepped.transpose("bands", "t", "x", "y"), epsg)
        num_samples = getattr(predictors, "B", None)
        num_timesteps = getattr(predictors, "T", None)
        logger.info(
            f"Predictors ready (samples={num_samples}, timesteps={num_timesteps}, batch_size={self.batch_size})"
        )
        mask_array, active_season_ids = self._resolve_season_masks(
            timestamps=prepped.t.values,
            batch_size=predictors.B,
            season_windows=season_windows,
            season_masks=season_masks,
            season_ids=season_ids,
        )
        logger.info(
            f"Season masks resolved for {active_season_ids} (shape={mask_array.shape})"
        )
        outputs = self._run_batches(predictors, mask_array)
        logger.info(
            f"Batch inference complete; formatting outputs for {len(active_season_ids)} seasons"
        )

        dataset = self._format_outputs(
            arr=prepped,
            outputs=outputs,
            season_ids=active_season_ids,
            enforce_cropland_gate=enforce_cropland_gate,
        )

        return _dataset_to_multiband_array(dataset)

    def _prepare_array(self, arr: xr.DataArray, epsg: int) -> xr.DataArray:
        if "bands" not in arr.dims:
            raise ValueError("Input DataArray must expose a 'bands' dimension")
        reordered = arr.transpose("bands", "t", "y", "x")
        reordered = DataPreprocessor.rescale_s1_backscatter(reordered)
        renamed_bands = [
            GFMAP_BAND_MAPPING.get(str(b), str(b)) for b in reordered.bands.values
        ]
        reordered = reordered.assign_coords(bands=renamed_bands)
        reordered = reordered.transpose("bands", "t", "x", "y")
        reordered = DataPreprocessor.add_slope_band(reordered, epsg)
        return reordered.fillna(NODATA_VALUE).astype(np.float32)

    def _resolve_season_masks(
        self,
        *,
        timestamps: np.ndarray,
        batch_size: int,
        season_windows: Optional[Mapping[str, SeasonWindowValue]],
        season_masks: Optional[np.ndarray],
        season_ids: Optional[Sequence[str]],
    ) -> Tuple[np.ndarray, List[str]]:
        ts_days = _ensure_datetime64_array(timestamps)
        num_timesteps = ts_days.shape[0]
        if num_timesteps == 0:
            raise ValueError(
                "Input array must expose at least one timestep for seasonal inference"
            )

        mask_array: Optional[np.ndarray] = None
        if season_masks is not None:
            mask_array = _normalize_provided_masks(
                season_masks, batch_size, num_timesteps
            )

        normalized_windows = (
            _normalize_season_windows_input(season_windows)
            if season_windows is not None
            else self._default_season_windows
        )

        mask_season_count = mask_array.shape[1] if mask_array is not None else None
        active_ids = self._resolve_season_ids(
            override=season_ids,
            windows=normalized_windows,
            mask_season_count=mask_season_count,
        )

        if mask_array is not None:
            if mask_array.shape[1] != len(active_ids):
                raise ValueError(
                    "Provided season masks do not match the requested season identifiers."
                )
            return mask_array, active_ids

        if normalized_windows:
            mask_array = _build_masks_from_windows(
                ts_days,
                active_ids,
                normalized_windows,
                batch_size,
                composite_frequency=self._season_composite_frequency,
            )
            return mask_array, active_ids

        if len(active_ids) > 1 and self._croptype_enabled:
            raise ValueError(
                "Season masks/windows are required to evaluate multiple seasons."
                " Provide `season_windows` or `season_masks` to avoid uniform coverage."
            )

        if len(active_ids) > 1 and not self._croptype_enabled:
            logger.info(
                "Croptype head disabled; evaluating %d seasons with uniform full-coverage masks.",
                len(active_ids),
            )
        else:
            log_fn = logger.warning if self._croptype_enabled else logger.info
            log_fn(
                "No season windows or masks provided; treating season '%s' as full-year coverage",
                active_ids[0],
            )
        mask_array = _build_uniform_masks(batch_size, num_timesteps, len(active_ids))
        return mask_array, active_ids

    def _resolve_season_ids(
        self,
        *,
        override: Optional[Sequence[str]],
        windows: Dict[str, List[Tuple[np.datetime64, np.datetime64]]],
        mask_season_count: Optional[int],
    ) -> List[str]:
        if override is not None:
            if len(override) == 0:
                raise ValueError("season_ids override cannot be empty")
            return list(override)
        if windows:
            return list(windows.keys())
        if mask_season_count is not None:
            if len(self._default_season_ids) != mask_season_count:
                raise ValueError(
                    "Provided season masks require explicit season_ids when their count"
                    " differs from the engine defaults."
                )
            return list(self._default_season_ids)
        if not self._default_season_ids:
            raise ValueError(
                "Seasonal inference requires at least one season identifier"
            )
        return list(self._default_season_ids)

    def _run_batches(
        self, predictors: "Predictors", season_masks: np.ndarray
    ) -> Tuple[Optional[TorchTensor], Optional[TorchTensor]]:
        torch = _lazy_import_torch()
        from prometheo.predictors import Predictors, to_torchtensor

        landcover_logits: List[TorchTensor] = []
        croptype_logits: List[TorchTensor] = []

        total_samples = getattr(predictors, "B", 0)
        estimated_batches = (
            (total_samples + self.batch_size - 1) // self.batch_size
            if total_samples
            else None
        )
        if total_samples:
            logger.info(
                f"Running seasonal heads on {total_samples} samples (~{estimated_batches or 1} batches)"
            )
        processed_batches = 0
        start = 0
        for processed_batches, batch in enumerate(
            predictors.as_batches(self.batch_size), start=1
        ):
            batch_size = batch.B
            if estimated_batches:
                logger.debug(
                    f"Processing predictor batch {processed_batches}/{estimated_batches} (size={batch_size})"
                )
            batch_dict = {
                field: to_torchtensor(getattr(batch, field), device=self.device)
                for field in batch._fields
                if getattr(batch, field) is not None
            }
            batch_predictors = Predictors(**batch_dict)
            mask_tensor = torch.as_tensor(
                season_masks[start : start + batch_size],
                device=self.device,
                dtype=torch.bool,
            )
            start += batch_size
            with torch.inference_mode():
                output = self.bundle.model(
                    batch_predictors, attrs={"season_masks": mask_tensor}
                )
            if output.global_logits is not None:
                landcover_logits.append(output.global_logits.detach().cpu())
            if output.season_logits is not None:
                croptype_logits.append(output.season_logits.detach().cpu())

        logger.info(
            f"Finished running {processed_batches} predictor batches (landcover={len(landcover_logits)}, "
            f"croptype={len(croptype_logits)})"
        )

        lc_pieces = [t for t in landcover_logits if t.numel() > 0]
        ct_pieces = [t for t in croptype_logits if t.numel() > 0]
        lc_tensor = torch.cat(lc_pieces, dim=0) if lc_pieces else None
        ct_tensor = torch.cat(ct_pieces, dim=0) if ct_pieces else None
        return lc_tensor, ct_tensor

    def _format_outputs(
        self,
        *,
        arr: xr.DataArray,
        outputs: Tuple[Optional[TorchTensor], Optional[TorchTensor]],
        season_ids: Sequence[str],
        enforce_cropland_gate: bool,
    ) -> xr.Dataset:
        torch = _lazy_import_torch()
        height = arr.sizes["y"]
        width = arr.sizes["x"]
        landcover_logits, croptype_logits = outputs

        band_layers: List[Tuple[str, np.ndarray]] = []
        cropland_mask_bool: Optional[np.ndarray] = None

        def _register_band(name: str, values: np.ndarray) -> None:
            if values.shape != (height, width):
                raise ValueError(
                    f"Band '{name}' has incompatible shape {values.shape}; expected {(height, width)}."
                )
            band_layers.append((name, values.astype(np.uint8, copy=False)))

        if landcover_logits is not None and landcover_logits.numel() > 0:
            probs = torch.softmax(landcover_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            prob_cube = (
                probs.detach()
                .cpu()
                .numpy()
                .reshape(height, width, self.bundle.landcover_spec.num_classes)
            )
            prob_cube = np.transpose(prob_cube, (2, 0, 1))
            preds_np = preds.numpy().reshape(height, width)

            landcover_classes = list(self.bundle.landcover_spec.class_names)
            cropland_gate_labels = list(self.bundle.cropland_gate_classes)
            class_index = {name: idx for idx, name in enumerate(landcover_classes)}
            cropland_label_order = [
                name for name in cropland_gate_labels if name in class_index
            ]
            missing_cropland = [
                name for name in cropland_gate_labels if name not in class_index
            ]
            if missing_cropland:
                logger.warning(
                    "Cropland classes %s missing from landcover outputs; treating them as non-cropland.",
                    missing_cropland,
                )
            cropland_indices = [class_index[name] for name in cropland_label_order]
            cropland_index_set = set(cropland_indices)
            other_indices = [
                idx
                for idx in range(len(landcover_classes))
                if idx not in cropland_index_set
            ]

            cropland_prob_other = (
                prob_cube[other_indices].sum(axis=0)
                if other_indices
                else np.zeros((height, width), dtype=np.float32)
            )
            cropland_prob_crops = (
                prob_cube[cropland_indices].sum(axis=0)
                if cropland_indices
                else np.zeros((height, width), dtype=np.float32)
            )
            raw_cropland_prob_cube = np.stack(
                [cropland_prob_other, cropland_prob_crops], axis=0
            )
            cropland_prob_cube = raw_cropland_prob_cube.copy()

            if cropland_gate_labels:
                if cropland_indices:
                    cropland_mask_bool = np.isin(preds_np, cropland_indices)
                else:
                    logger.warning(
                        "Configured cropland classes do not match landcover outputs; defaulting to all pixels as cropland."
                    )
                    cropland_mask_bool = np.ones_like(preds_np, dtype=bool)
            else:
                logger.warning(
                    "Cropland classes unavailable; defaulting to all pixels as cropland."
                )
                cropland_mask_bool = np.ones_like(preds_np, dtype=bool)

            cropland_labels_uint8 = cropland_mask_bool.astype(np.uint8)
            cropland_method = self._cropland_postprocess.resolved_method()
            if cropland_method:
                logger.info(
                    f"Applying {cropland_method} postprocess to cropland mask (kernel_size={self._cropland_postprocess.kernel_size})"
                )
            (
                cropland_labels_uint8,
                _cropland_probability_np,
                cropland_prob_cube,
            ) = _run_postprocess(
                cropland_labels_uint8,
                cropland_prob_cube,
                class_value_to_index={0: 0, 1: 1},
                options=self._cropland_postprocess,
                excluded_values=(),
            )
            cropland_mask_bool = cropland_labels_uint8.astype(bool)
            cropland_probability_uint8 = _probabilities_to_uint8(
                raw_cropland_prob_cube[1]
            )
            probability_other_uint8 = _probabilities_to_uint8(raw_cropland_prob_cube[0])

            _register_band("cropland_classification", cropland_labels_uint8)
            _register_band("probability_cropland", cropland_probability_uint8)

            if self._export_class_probabilities:
                if cropland_indices:
                    per_class_probs = prob_cube[cropland_indices]
                    for idx, label in enumerate(cropland_label_order):
                        _register_band(
                            f"probability_{label}",
                            _probabilities_to_uint8(per_class_probs[idx]),
                        )
                _register_band("probability_other", probability_other_uint8)
        else:
            if self._cropland_enabled:
                logger.warning(
                    "Landcover head missing; cropland mask cannot be derived."
                )
            else:
                logger.info(
                    "Cropland head disabled; skipping cropland classification outputs."
                )

        if croptype_logits is not None and croptype_logits.numel() > 0:
            probs = torch.softmax(croptype_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            num_seasons = preds.shape[1]
            preds_np = preds.numpy().reshape(height, width, num_seasons)
            prob_np = (
                probs.detach()
                .cpu()
                .numpy()
                .reshape(
                    height, width, num_seasons, self.bundle.croptype_spec.num_classes
                )
            )
            gate_applicable = (
                enforce_cropland_gate
                and self._cropland_enabled
                and cropland_mask_bool is not None
            )
            if gate_applicable:
                assert cropland_mask_bool is not None, (
                    "Cropland mask required when gating is enabled"
                )
                gate = cropland_mask_bool[:, :, None]
                preds_np = np.where(gate, preds_np, NOCROP_VALUE)

            prob_cube = np.transpose(prob_np, (2, 3, 0, 1))  # season, class, y, x
            if gate_applicable:
                assert cropland_mask_bool is not None, (
                    "Cropland mask required when gating is enabled"
                )
                gating = cropland_mask_bool[None, None, :, :]
                prob_cube = np.where(gating, prob_cube, 0.0)
            class_value_to_index = {
                idx: idx for idx in range(self.bundle.croptype_spec.num_classes)
            }

            processed_labels: List[np.ndarray] = []
            processed_probabilities: List[np.ndarray] = []
            processed_probability_cubes: List[np.ndarray] = []
            croptype_method = self._croptype_postprocess.resolved_method()
            if croptype_method:
                logger.info(
                    f"Applying {croptype_method} postprocess to croptype logits (kernel_size={self._croptype_postprocess.kernel_size}, seasons={num_seasons})"
                )
            for season_idx in range(num_seasons):
                season_labels = preds_np[:, :, season_idx].astype(np.uint16, copy=True)
                season_prob_cube = prob_cube[season_idx]
                (
                    season_labels,
                    season_probabilities,
                    season_prob_cube_processed,
                ) = _run_postprocess(
                    season_labels,
                    season_prob_cube,
                    class_value_to_index=class_value_to_index,
                    options=self._croptype_postprocess,
                    excluded_values=POSTPROCESSING_EXCLUDED_VALUES,
                )
                processed_labels.append(season_labels)
                processed_probabilities.append(season_probabilities)
                processed_probability_cubes.append(season_prob_cube_processed)

            preds_stack = np.stack(processed_labels, axis=0)
            _ensure_uint8_range(preds_stack, name="croptype_classification")
            preds_uint8 = preds_stack.astype(np.uint8, copy=False)
            conf_uint8 = _probabilities_to_uint8(
                np.stack(processed_probabilities, axis=0)
            )
            if gate_applicable and cropland_mask_bool is not None:
                gate = cropland_mask_bool[None, :, :]
                sentinel_uint8 = np.uint8(NOCROP_VALUE)
                conf_uint8 = np.where(gate, conf_uint8, sentinel_uint8)

            season_labels = list(season_ids[:num_seasons])
            for idx, season_id in enumerate(season_labels):
                band_name = f"croptype_classification:{season_id}"
                _register_band(band_name, preds_uint8[idx])
            for idx, season_id in enumerate(season_labels):
                band_name = f"croptype_probability:{season_id}"
                _register_band(band_name, conf_uint8[idx])

            if self._export_class_probabilities:
                prob_stack = np.stack(processed_probability_cubes, axis=0)
                prob_uint8 = _probabilities_to_uint8(prob_stack)
                if gate_applicable and cropland_mask_bool is not None:
                    gate = cropland_mask_bool[None, None, :, :]
                    sentinel_uint8 = np.uint8(NOCROP_VALUE)
                    prob_uint8 = np.where(gate, prob_uint8, sentinel_uint8)
                for season_idx, season_id in enumerate(season_labels):
                    for class_idx, class_name in enumerate(
                        self.bundle.croptype_spec.class_names
                    ):
                        layer_name = f"croptype_probability:{season_id}:{class_name}"
                        _register_band(layer_name, prob_uint8[season_idx, class_idx])

        elif self._croptype_enabled:
            logger.warning("Croptype head missing; skipping seasonal crop outputs.")
        else:
            logger.info(
                "Croptype outputs skipped because the croptype head is disabled."
            )

        ordered_vars: OrderedDict[str, xr.DataArray] = OrderedDict()
        for name, values in band_layers:
            ordered_vars[name] = xr.DataArray(
                values,
                dims=("y", "x"),
                coords={"y": arr.y, "x": arr.x},
            )
        return xr.Dataset(ordered_vars)


# ---------------------------------------------------------------------------
# Convenience wrapper for ad-hoc usage
# ---------------------------------------------------------------------------


def run_seasonal_workflow(
    arr: xr.DataArray,
    epsg: int,
    *,
    seasonal_model_zip: str | Path,
    landcover_head_zip: str | Path | None = None,
    croptype_head_zip: str | Path | None = None,
    enforce_cropland_gate: bool = True,
    cache_root: Optional[Path] = None,
    device: Union[str, TorchDevice] = "cpu",
    batch_size: int = 256,
    season_ids: Optional[Sequence[str]] = None,
    season_windows: Optional[Mapping[str, SeasonWindowValue]] = None,
    season_masks: Optional[np.ndarray] = None,
    season_composite_frequency: Optional[Literal["month", "dekad"]] = "month",
    export_class_probabilities: bool = False,
    enable_croptype_head: bool = True,
    enable_cropland_head: bool = True,
    cropland_postprocess: Optional[Mapping[str, Any]] = None,
    croptype_postprocess: Optional[Mapping[str, Any]] = None,
) -> xr.DataArray:
    """Run the full seasonal workflow and return a multi-band array."""

    engine = SeasonalInferenceEngine(
        seasonal_model_zip=seasonal_model_zip,
        landcover_head_zip=landcover_head_zip,
        croptype_head_zip=croptype_head_zip,
        cache_root=cache_root,
        device=device,
        batch_size=batch_size,
        season_ids=season_ids,
        season_windows=season_windows,
        season_composite_frequency=season_composite_frequency,
        export_class_probabilities=export_class_probabilities,
        enable_croptype_head=enable_croptype_head,
        enable_cropland_head=enable_cropland_head,
        cropland_postprocess=cropland_postprocess,
        croptype_postprocess=croptype_postprocess,
    )
    datacube = engine.infer(
        arr,
        epsg,
        enforce_cropland_gate=enforce_cropland_gate,
        season_ids=season_ids,
        season_windows=season_windows,
        season_masks=season_masks,
    )
    return datacube


# ---------------------------------------------------------------------------
# openEO UDF integration hooks
# ---------------------------------------------------------------------------


def _require_openeo_runtime() -> None:
    sys.path.insert(0, "feature_deps")
    sys.path.insert(0, "worldcereallib")
    sys.path.insert(0, "prometheolib")

    try:
        import prometheo
        import torch

        import worldcereal

        logger.debug(f"Loading worldcereal from {worldcereal.__file__}")
        logger.debug(f"Loading prometheo from {prometheo.__file__}")
        logger.debug(f"Loading torch from {torch.__file__}")
    except ImportError as exc:
        raise ImportError(
            "openEO UDF seasonal inference requires the worldcereal, prometheo, and loguru packages."
        ) from exc


def _infer_udf_epsg(udf_data: "UdfData") -> int:
    proj = getattr(udf_data, "proj", None)
    if proj and "EPSG" in proj:
        return int(proj["EPSG"])
    raise ValueError("EPSG code not found in UDF projection metadata")


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    raise TypeError(f"Cannot interpret boolean value from type {type(value)!r}")


def _build_postprocess_options(
    spec: Optional[Union[PostprocessOptions, Mapping[str, Any]]],
) -> PostprocessOptions:
    if spec is None:
        return PostprocessOptions()
    if isinstance(spec, PostprocessOptions):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError(
            "Postprocess options must be provided as a mapping or PostprocessOptions instance"
        )

    enabled = _as_bool(spec.get("enabled"), False)
    method = spec.get("method")
    kernel_value = spec.get("kernel_size", 5)
    try:
        kernel_size = int(kernel_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"kernel_size must be an integer, got {kernel_value!r}"
        ) from exc

    return PostprocessOptions(enabled=enabled, method=method, kernel_size=kernel_size)


def _normalize_season_id_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [piece.strip() for piece in value.replace(";", ",").split(",")]
        result = [piece for piece in parts if piece]
        return result or None
    if isinstance(value, Sequence):
        collected: List[str] = []
        for item in value:
            if isinstance(item, str):
                collected.extend(
                    piece.strip()
                    for piece in item.replace(";", ",").split(",")
                    if piece.strip()
                )
            else:
                collected.append(str(item))
        return collected or None
    raise TypeError("season_ids must be a string or sequence of strings")


def _resolve_effective_season_ids(context: Mapping[str, Any]) -> List[str]:
    override = _normalize_season_id_list(
        context.get("season_ids") or context.get("season_id")
    )
    if override:
        return list(override)

    config_block = context.get("workflow_config")
    if isinstance(config_block, Mapping):
        season_section = config_block.get("season") or {}
        override = _normalize_season_id_list(season_section.get("season_ids"))
        if override:
            return list(override)

    workflow_block = context.get("seasonal_workflow")
    if isinstance(workflow_block, Mapping):
        season_section = workflow_block.get("season") or {}
        override = _normalize_season_id_list(season_section.get("season_ids"))
        if override:
            return list(override)

    preset_name = context.get("parameters") or context.get("preset")
    try:
        preset, _ = _select_workflow_preset(preset_name)
        preset_ids = preset.get("season", {}).get("season_ids")
        override = _normalize_season_id_list(preset_ids)
        if override:
            return list(override)
    except ValueError:
        logger.warning(
            "Unknown seasonal workflow preset '%s' in metadata context", preset_name
        )

    from worldcereal.train import GLOBAL_SEASON_IDS

    return list(GLOBAL_SEASON_IDS)


def _coerce_window_pair(value: Any) -> Tuple[SeasonDateLike, SeasonDateLike]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(
        "Season windows must be expressed as (start, end) pairs or ID:START:END strings"
    )


def _parse_udf_season_windows(
    value: Any,
) -> Optional[Dict[str, Tuple[SeasonDateLike, SeasonDateLike]]]:
    if value is None:
        return None
    windows: Dict[str, Tuple[SeasonDateLike, SeasonDateLike]] = {}

    def _add_window(spec_id: Any, start: SeasonDateLike, end: SeasonDateLike) -> None:
        name = str(spec_id).strip()
        if not name:
            raise ValueError("Season window identifiers must be non-empty")
        windows[name] = (start, end)

    if isinstance(value, Mapping):
        for key, raw in value.items():
            start, end = _coerce_window_pair(raw)
            _add_window(key, start, end)
        return windows

    entries: Sequence[Any]
    if isinstance(value, str):
        entries = [value]
    elif isinstance(value, Sequence):
        entries = value
    else:
        raise TypeError(
            "season_windows must be a mapping, list/tuple, or ID:START:END string"
        )

    for entry in entries:
        if isinstance(entry, str):
            parts = [piece.strip() for piece in entry.split(":")]
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid season window specification '{entry}'. Expected ID:START:END"
                )
            _add_window(parts[0], parts[1], parts[2])
        elif isinstance(entry, (list, tuple)) and len(entry) == 3:
            season_id, start, end = entry
            _add_window(season_id, start, end)
        else:
            raise ValueError(
                "Season window list entries must be ID:START:END strings or (id,start,end) tuples"
            )
    return windows


def _normalize_udf_season_masks(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim < 2:
        raise ValueError("season_masks must have at least two dimensions (S, T)")
    return arr


def _probabilities_to_uint8(array: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.clip(array, 0.0, 1.0) * 100.0)
    return scaled.astype(np.uint8)


def _ensure_uint8_range(values: np.ndarray, *, name: str) -> None:
    if values.size == 0:
        return
    min_val = values.min()
    max_val = values.max()
    if min_val < 0 or max_val > 255:
        raise ValueError(
            f"{name} contains values outside the uint8 range (min={min_val}, max={max_val})."
        )


def _dataset_to_multiband_array(dataset: xr.Dataset) -> xr.DataArray:
    if not dataset.data_vars:
        raise ValueError("Seasonal workflow produced an empty dataset")

    band_arrays: List[xr.DataArray] = []
    for band_name, data_array in dataset.data_vars.items():
        if data_array.dims != ("y", "x"):
            raise ValueError(
                f"Band '{band_name}' must expose ('y', 'x') dimensions; got {data_array.dims}."
            )
        layer = (
            data_array.astype(np.uint8, copy=False)
            .expand_dims("bands")
            .assign_coords(bands=[band_name])
        )
        band_arrays.append(layer)

    stacked = xr.concat(band_arrays, dim="bands").astype(np.uint8, copy=False)
    return stacked.transpose("bands", "y", "x")


def _expected_udf_band_labels(
    season_ids: Sequence[str],
    *,
    export_class_probabilities: bool = False,
    cropland_gate_classes: Optional[Sequence[str]] = None,
    croptype_classes: Optional[Sequence[str]] = None,
    croptype_enabled: bool = True,
    cropland_enabled: bool = True,
) -> List[str]:
    labels: List[str] = []
    if cropland_enabled:
        labels.extend(
            [
                "cropland_classification",
                "probability_cropland",
            ]
        )
    if croptype_enabled:
        for season_id in season_ids:
            labels.append(f"croptype_classification:{season_id}")
        for season_id in season_ids:
            labels.append(f"croptype_probability:{season_id}")
    if export_class_probabilities:
        if cropland_enabled:
            gate_classes = list(cropland_gate_classes) if cropland_gate_classes else []
            if gate_classes:
                labels.extend([f"probability_{cls}" for cls in gate_classes])
            labels.append("probability_other")
        if croptype_enabled:
            ct_classes = list(croptype_classes) if croptype_classes else []
            if ct_classes:
                for season_id in season_ids:
                    labels.extend(
                        [
                            f"croptype_probability:{season_id}:{cls}"
                            for cls in ct_classes
                        ]
                    )
            else:
                for season_id in season_ids:
                    labels.append(f"croptype_probability:{season_id}")
    return labels


def _select_workflow_preset(name: Any) -> Tuple[Dict[str, Dict[str, Any]], str]:
    default_preset, preset_map = _seasonal_workflow_presets()
    preset_key = str(name).strip() if isinstance(name, str) and name.strip() else name
    if not preset_key:
        preset_key = default_preset
    preset = preset_map.get(str(preset_key))
    if preset is None:
        available = ", ".join(sorted(preset_map)) or "<none>"
        raise ValueError(
            f"Unknown seasonal workflow preset '{preset_key}'. Available: {available}"
        )
    preset_copy = deepcopy(preset)
    _normalize_workflow_section_names(preset_copy)
    return preset_copy, str(preset_key)


def _normalize_workflow_section_names(workflow_cfg: Dict[str, Any]) -> None:
    workflow_cfg.setdefault("season", {})
    workflow_cfg.setdefault("postprocess", {})


def _merge_workflow_sections(
    base: Dict[str, Dict[str, Any]], overrides: Mapping[str, Any]
) -> None:
    for section, values in overrides.items():
        if section not in base:
            raise ValueError(
                f"Unsupported seasonal_workflow section '{section}'. "
                "Expected sections: model, runtime, season, postprocess"
            )
        if isinstance(values, Mapping):
            base_section = base[section]
            if not isinstance(base_section, dict):
                raise TypeError(
                    f"Preset section '{section}' cannot accept nested overrides"
                )
            for key, value in values.items():
                base_section[key] = value
        else:
            base[section] = values


def _finalize_workflow_config(workflow_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(workflow_cfg, Mapping):
        raise TypeError("workflow_config must be provided as a mapping")
    workflow_dict = {key: value for key, value in workflow_cfg.items()}
    _normalize_workflow_section_names(workflow_dict)

    model_cfg_raw = workflow_dict.get("model", {})
    runtime_cfg_raw = workflow_dict.get("runtime", {})
    season_cfg_raw = workflow_dict.get("season", {}) or {}
    postprocess_cfg_raw = workflow_dict.get("postprocess", {}) or {}

    if not isinstance(model_cfg_raw, Mapping):
        raise TypeError("workflow 'model' section must be a mapping")
    if not isinstance(runtime_cfg_raw, Mapping):
        raise TypeError("workflow 'runtime' section must be a mapping")
    if not isinstance(season_cfg_raw, Mapping):
        raise TypeError("workflow 'season' section must be a mapping")
    if not isinstance(postprocess_cfg_raw, Mapping):
        raise TypeError("workflow 'postprocess' section must be a mapping")

    model_cfg = dict(model_cfg_raw)
    runtime_cfg = dict(runtime_cfg_raw)
    season_cfg = dict(season_cfg_raw)
    postprocess_cfg = dict(postprocess_cfg_raw)

    model_source = model_cfg.get("seasonal_model_zip")
    if not model_source:
        raise ValueError(
            "seasonal_workflow configuration does not define a 'seasonal_model_zip'"
        )

    cache_root_raw = runtime_cfg.get("cache_root")
    cache_root = Path(cache_root_raw) if cache_root_raw else None

    batch_size_value = runtime_cfg.get("batch_size", 2048)
    try:
        batch_size_int = int(batch_size_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"batch_size must be an integer, got {batch_size_value!r}"
        ) from exc
    device = runtime_cfg.get("device", "cpu")

    export_probs_value = season_cfg.get("export_class_probabilities")
    export_class_probabilities = _as_bool(export_probs_value, False)
    enable_croptype_head_value = model_cfg.get("enable_croptype_head")
    enable_croptype_head = _as_bool(enable_croptype_head_value, True)
    enable_cropland_head_value = model_cfg.get("enable_cropland_head")
    enable_cropland_head = _as_bool(enable_cropland_head_value, True)

    season_ids = _normalize_season_id_list(season_cfg.get("season_ids"))
    season_windows = _parse_udf_season_windows(season_cfg.get("season_windows"))
    season_masks = _normalize_udf_season_masks(season_cfg.get("season_masks"))
    enforce_gate = _as_bool(season_cfg.get("enforce_cropland_gate"), True)
    composite_frequency_raw = season_cfg.get("composite_frequency", "month")
    if composite_frequency_raw not in (None, "month", "dekad"):
        raise ValueError("Season composite frequency must be 'month', 'dekad', or None")
    composite_frequency = (
        str(composite_frequency_raw)
        if composite_frequency_raw in {"month", "dekad"}
        else None
    )

    def _normalize_postprocess_entry(value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("postprocess entries must be provided as mappings")
        return dict(value)

    cropland_post_cfg = _normalize_postprocess_entry(postprocess_cfg.get("cropland"))
    croptype_post_cfg = _normalize_postprocess_entry(postprocess_cfg.get("croptype"))

    if not (enable_croptype_head or enable_cropland_head):
        raise ValueError(
            "Seasonal workflow configuration must enable at least one head."
        )

    return {
        "seasonal_model_zip": model_source,
        "landcover_head_zip": model_cfg.get("landcover_head_zip"),
        "croptype_head_zip": model_cfg.get("croptype_head_zip"),
        "enforce_cropland_gate": enforce_gate,
        "cache_root": cache_root,
        "device": device,
        "batch_size": batch_size_int,
        "season_ids": season_ids,
        "season_windows": season_windows,
        "season_masks": season_masks,
        "season_composite_frequency": composite_frequency or "month",
        "export_class_probabilities": export_class_probabilities,
        "enable_croptype_head": enable_croptype_head,
        "enable_cropland_head": enable_cropland_head,
        "cropland_postprocess": cropland_post_cfg,
        "croptype_postprocess": croptype_post_cfg,
    }


def _extract_udf_configuration(context: Mapping[str, Any]) -> Dict[str, Any]:
    config_block = context.get("workflow_config")
    if config_block is not None:
        if not isinstance(config_block, Mapping):
            raise TypeError("workflow_config overrides must be provided as a mapping")
        return _finalize_workflow_config(dict(config_block))

    preset_name = context.get("parameters") or context.get("preset")
    workflow_cfg, _ = _select_workflow_preset(preset_name)
    _normalize_workflow_section_names(workflow_cfg)
    workflow_cfg.setdefault("model", {})
    workflow_cfg.setdefault("runtime", {})
    workflow_cfg.setdefault("season", {})
    workflow_cfg.setdefault("postprocess", {})

    workflow_overrides = context.get("seasonal_workflow")
    if workflow_overrides is not None:
        if not isinstance(workflow_overrides, Mapping):
            raise TypeError("seasonal_workflow overrides must be provided as a mapping")
        _merge_workflow_sections(workflow_cfg, workflow_overrides)

    direct_model_source = (
        context.get("seasonal_model_zip")
        or context.get("seasonal_zip")
        or context.get("seasonal_model_url")
    )
    if direct_model_source:
        workflow_cfg["model"]["seasonal_model_zip"] = direct_model_source
    for head_key in ("landcover_head_zip", "croptype_head_zip"):
        if head_key in context:
            workflow_cfg["model"][head_key] = context.get(head_key)

    if "enable_croptype_head" in context:
        workflow_cfg["model"]["enable_croptype_head"] = _as_bool(
            context.get("enable_croptype_head"), True
        )
    disable_ctx = None
    if "disable_croptype_head" in context:
        disable_ctx = context.get("disable_croptype_head")
    elif "disable_croptype" in context:
        disable_ctx = context.get("disable_croptype")
    if disable_ctx is not None and _as_bool(disable_ctx, False):
        workflow_cfg["model"]["enable_croptype_head"] = False

    if "enable_cropland_head" in context:
        workflow_cfg["model"]["enable_cropland_head"] = _as_bool(
            context.get("enable_cropland_head"), True
        )
    disable_cropland_ctx = context.get("disable_cropland_head")
    if disable_cropland_ctx is not None and _as_bool(disable_cropland_ctx, False):
        workflow_cfg["model"]["enable_cropland_head"] = False

    croptype_only_ctx = context.get("croptype_only")
    if croptype_only_ctx is not None and _as_bool(croptype_only_ctx, False):
        workflow_cfg["model"]["enable_cropland_head"] = False
        workflow_cfg["model"]["enable_croptype_head"] = True
        workflow_cfg["season"]["enforce_cropland_gate"] = False

    cache_root_override = context.get("cache_root") or context.get("cache_dir")
    if cache_root_override is not None:
        workflow_cfg["runtime"]["cache_root"] = cache_root_override
    if "device" in context:
        workflow_cfg["runtime"]["device"] = context.get("device")
    if "batch_size" in context:
        workflow_cfg["runtime"]["batch_size"] = context.get("batch_size")

    season_id_override = context.get("season_ids") or context.get("season_id")
    if season_id_override is not None:
        workflow_cfg["season"]["season_ids"] = season_id_override
    windows_override = context.get("season_windows") or context.get("season_window")
    if windows_override is not None:
        workflow_cfg["season"]["season_windows"] = windows_override
    masks_override = context.get("season_masks") or context.get("season_mask")
    if masks_override is not None:
        workflow_cfg["season"]["season_masks"] = masks_override
    composite_override = context.get("season_composite_frequency") or context.get(
        "composite_frequency"
    )
    if composite_override is not None:
        workflow_cfg["season"]["composite_frequency"] = composite_override

    if "cropland_postprocess" in context:
        workflow_cfg["postprocess"]["cropland"] = context.get("cropland_postprocess")
    if "croptype_postprocess" in context:
        workflow_cfg["postprocess"]["croptype"] = context.get("croptype_postprocess")

    if "enforce_cropland_gate" in context:
        workflow_cfg["season"]["enforce_cropland_gate"] = _as_bool(
            context.get("enforce_cropland_gate"), True
        )
    if "disable_cropland_gate" in context:
        disable_flag = _as_bool(context.get("disable_cropland_gate"), False)
        if disable_flag:
            workflow_cfg["season"]["enforce_cropland_gate"] = False

    if "export_class_probabilities" in context:
        workflow_cfg["season"]["export_class_probabilities"] = context.get(
            "export_class_probabilities"
        )

    return _finalize_workflow_config(workflow_cfg)


def apply_udf_data(udf_data: UdfData) -> UdfData:
    """openEO entry point that wraps the seasonal inference workflow."""

    _require_openeo_runtime()
    if not udf_data.datacube_list:
        raise ValueError("UDF input does not contain any data cubes")

    context = udf_data.user_context or {}
    config = _extract_udf_configuration(context)
    epsg = _infer_udf_epsg(udf_data)

    input_array = udf_data.datacube_list[0].get_array()
    try:
        prepared = input_array.transpose("bands", "t", "y", "x")
    except ValueError as exc:  # pragma: no cover - guard unexpected layouts
        raise ValueError(
            "Input cube must expose dimensions ('bands', 't', 'y', 'x') for seasonal inference"
        ) from exc

    datacube = run_seasonal_workflow(arr=prepared, epsg=epsg, **config)
    udf_data.datacube_list = [XarrayDataCube(datacube)]
    return udf_data


def apply_metadata(metadata: Any, context: Optional[Mapping[str, Any]]) -> Any:
    """openEO metadata hook that keeps band labels in sync with the workflow outputs."""

    _require_openeo_runtime()

    try:
        context_map = dict(context or {})
        season_ids = _resolve_effective_season_ids(context_map)
        export_probs = False
        cropland_gate_classes: Optional[Sequence[str]] = None
        croptype_classes: Optional[Sequence[str]] = None
        croptype_enabled = True
        cropland_enabled = True

        try:
            config = _extract_udf_configuration(context_map)
            export_probs = config.get("export_class_probabilities", False)
            croptype_enabled = config.get("enable_croptype_head", True)
            cropland_enabled = config.get("enable_cropland_head", True)
            if export_probs:
                artifact = load_model_artifact(
                    config["seasonal_model_zip"], cache_root=config.get("cache_root")
                )
                heads = artifact.manifest.get("heads", [])
                if cropland_enabled:
                    landcover_spec = _select_head_spec(heads, "landcover")
                    cropland_gate_classes = landcover_spec.cropland_classes
                if croptype_enabled:
                    croptype_classes = _select_head_spec(heads, "croptype").class_names
        except Exception as exc:
            logger.warning(f"Metadata configuration fallback: {exc}")

        labels = _expected_udf_band_labels(
            season_ids,
            export_class_probabilities=export_probs,
            cropland_gate_classes=cropland_gate_classes,
            croptype_classes=croptype_classes,
            croptype_enabled=croptype_enabled,
            cropland_enabled=cropland_enabled,
        )
        return metadata.rename_labels(dimension="bands", target=labels)
    except Exception as exc:  # pragma: no cover - metadata best-effort
        logger.warning(f"apply_metadata fallback: {exc}")
        return metadata
