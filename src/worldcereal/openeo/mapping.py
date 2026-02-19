"""Mapping helpers for cropland, croptype and embeddings products.

Band naming conventions produced by the UDF (`inference.py`):

Single workflow (only cropland OR only croptype parameters passed to UDF):
    classification, probability, probability_<class>
    If save_intermediate: raw_<band> duplicates (e.g. raw_classification)

Combined workflow (croptype with cropland masking: both `croptype_params` &
`cropland_params` passed):
    croptype_<band>, cropland_<band>
    If save_intermediate: croptype_raw_<band>, cropland_raw_<band>
        Example: croptype_classification -> croptype_raw_classification

Important: Raw bands in the combined workflow do NOT duplicate the base prefix;
they simply replace the leading product prefix with <product>_raw_.

Simplification: We ignore any *save_intermediate* flags. If raw bands are
present we save them; the UDF only emits them when intermediate results were
requested upstream.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import openeo
from openeo import DataCube
from openeo_gfmap import TemporalContext

from worldcereal.openeo.inference import (
    _resolve_effective_season_ids,
    apply_metadata,
)
from worldcereal.parameters import (
    EmbeddingsParameters,
    WorldCerealProductType,
)

NEIGHBORHOOD_SPEC = dict(
    size=[
        {"dimension": "x", "unit": "px", "value": 128},
        {"dimension": "y", "unit": "px", "value": 128},
    ],
    overlap=[
        {"dimension": "x", "unit": "px", "value": 0},
        {"dimension": "y", "unit": "px", "value": 0},
    ],
)


def _normalize_band_label(band: str) -> str:
    if band.startswith("croptype_"):
        band = band.replace("croptype_", "")
    return band.replace(":", "_")


def _extract_band_season_id(band: str) -> Optional[str]:
    if not band.startswith("croptype_"):
        return None
    parts = band.split(":")
    if len(parts) < 2:
        return None
    return parts[1]


def _season_order_from_bands(band_names: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for band in band_names:
        season_id = _extract_band_season_id(band)
        if season_id and season_id not in seen:
            seen.add(season_id)
            ordered.append(season_id)
    return ordered


def _season_band_label(band: str, season_id: str) -> str:
    if not band.startswith("croptype_"):
        return band.replace(":", "_")
    trimmed = band[len("croptype_") :]
    parts = trimmed.split(":")
    cleaned: List[str] = []
    for part in parts:
        if part == season_id:
            continue
        if part:
            cleaned.append(part)
    return "_".join(cleaned) if cleaned else trimmed.replace(":", "_")


def _sanitize_filename_fragment(value: Any) -> str:
    """Best-effort cleanup so window labels are filesystem-friendly."""

    text = str(value).strip()
    if not text:
        return "window"
    for char in ("/", "\\", ":"):
        text = text.replace(char, "-")
    return text.replace(" ", "")


def _normalize_window_entry(value: Any) -> Optional[Tuple[str, str]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        start = value.get("start") or value.get("begin")
        end = value.get("end") or value.get("stop")
        if start is not None and end is not None:
            return (str(start), str(end))
        return None
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and value[0] is not None and value[1] is not None:
            return (str(value[0]), str(value[1]))
        return None
    if isinstance(value, str):
        parts = [piece.strip() for piece in value.split(":")]
        if len(parts) == 2:
            return (parts[0], parts[1])
    return None


def _season_windows_from_context(
    workflow_context: Mapping[str, Any],
) -> Dict[str, Tuple[str, str]]:
    windows: Dict[str, Tuple[str, str]] = {}

    def _collect(block: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(block, Mapping):
            return
        season_block = block.get("season")
        if not isinstance(season_block, Mapping):
            return
        raw = season_block.get("season_windows")
        if not isinstance(raw, Mapping):
            return
        for season_id, window in raw.items():
            normalized = _normalize_window_entry(window)
            if normalized:
                windows[str(season_id)] = normalized

    _collect(workflow_context.get("workflow_config"))
    _collect(workflow_context.get("seasonal_workflow"))

    direct = workflow_context.get("season_windows")
    if isinstance(direct, Mapping):
        for season_id, window in direct.items():
            normalized = _normalize_window_entry(window)
            if normalized:
                windows[str(season_id)] = normalized

    return windows


def _run_udf(inputs: DataCube, udf: openeo.UDF) -> DataCube:
    return inputs.apply_neighborhood(process=udf, **NEIGHBORHOOD_SPEC)


def _reduce_temporal_mean(cube: DataCube) -> DataCube:
    return cube.reduce_dimension(dimension="t", reducer="mean")


def _filename_prefix(
    product: WorldCerealProductType, temporal: TemporalContext, raw: bool = False
) -> str:
    suffix = "-raw" if raw else ""
    return f"{product.value}{suffix}_{temporal.start_date}_{temporal.end_date}"


def _save_result(cube: DataCube, prefix: str) -> DataCube:
    return cube.save_result(format="GTiff", options={"filename_prefix": prefix})


def _workflow_inference_cube(
    inputs: DataCube, workflow_context: Mapping[str, Any]
) -> DataCube:
    inference_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "inference.py",
        context=workflow_context,
    )
    cube = _run_udf(inputs, inference_udf)
    cube.metadata = apply_metadata(cube.metadata, workflow_context)
    cube = _reduce_temporal_mean(cube)
    return cube.linear_scale_range(0, 254, 0, 254)


def _cropland_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    workflow_context: Mapping[str, Any],
) -> List[DataCube]:
    """Produce cropland product bands using the configured seasonal workflow."""

    cropland_context: Dict[str, Any] = deepcopy(dict(workflow_context))
    cropland_context["disable_croptype_head"] = True
    workflow_cfg = cropland_context.setdefault("workflow_config", {})
    if isinstance(workflow_cfg, dict):
        model_cfg = workflow_cfg.setdefault("model", {})
        if isinstance(model_cfg, dict):
            model_cfg["enable_croptype_head"] = False
    cube = _workflow_inference_cube(inputs, cropland_context)
    band_names = cube.metadata.band_names
    available_bands = [band for band in band_names if not band.startswith("croptype")]
    if not available_bands:
        raise ValueError(
            "Seasonal UDF did not emit cropland bands; expected cropland outputs."
        )
    filtered = cube.filter_bands(available_bands)
    return [
        _save_result(
            filtered,
            _filename_prefix(WorldCerealProductType.CROPLAND, temporal_extent),
        )
    ]


def _croptype_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    workflow_context: Mapping[str, Any],
) -> List[DataCube]:
    """Produce per-season crop-type products alongside a dedicated cropland product."""

    cube = _workflow_inference_cube(inputs, workflow_context)
    band_names = cube.metadata.band_names
    classification_bands = [
        band for band in band_names if band.startswith("croptype_classification")
    ]
    if not classification_bands:
        raise ValueError(
            "Seasonal UDF did not emit croptype classification bands; cannot build product."
        )

    configured_seasons: List[str] = []
    try:
        configured_seasons = _resolve_effective_season_ids(workflow_context)
    except (
        Exception
    ):  # pragma: no cover - defensive, should not trigger in normal flows
        configured_seasons = []
    available_seasons = _season_order_from_bands(band_names)
    season_order: List[str] = [
        season_id for season_id in configured_seasons if season_id in available_seasons
    ]
    if not season_order:
        season_order = available_seasons
    else:
        for season_id in available_seasons:
            if season_id not in season_order:
                season_order.append(season_id)

    season_windows = _season_windows_from_context(workflow_context)
    season_results: List[DataCube] = []
    for season_id in season_order:
        season_bands = [
            band
            for band in band_names
            if band.startswith("croptype_")
            and _extract_band_season_id(band) == season_id
        ]
        if not season_bands:
            continue
        normalized_labels = [
            _season_band_label(band, season_id) for band in season_bands
        ]
        season_cube = cube.filter_bands(season_bands).rename_labels(
            dimension="bands",
            target=normalized_labels,
        )
        window = season_windows.get(season_id)
        window_suffix = ""
        if window:
            start_label = _sanitize_filename_fragment(window[0])
            end_label = _sanitize_filename_fragment(window[1])
            window_suffix = f"_[{start_label}_{end_label}]"
        season_results.append(
            _save_result(
                season_cube,
                f"{_filename_prefix(WorldCerealProductType.CROPTYPE, temporal_extent)}_{season_id}{window_suffix}",
            )
        )

    if not season_results:
        raise ValueError(
            "Seasonal UDF did not emit croptype bands that could be assigned to a season."
        )

    cropland_bands = [band for band in band_names if not band.startswith("croptype")]

    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered_cropland = []
    for band in cropland_bands:
        if band not in seen:
            seen.add(band)
            ordered_cropland.append(band)

    cropland_results: List[DataCube] = []
    if ordered_cropland:
        cropland_cube = cube.filter_bands(ordered_cropland)
        cropland_results.append(
            _save_result(
                cropland_cube,
                _filename_prefix(WorldCerealProductType.CROPLAND, temporal_extent),
            )
        )

    return season_results + cropland_results


def _embeddings_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,  # temporal extent unused but kept for signature consistency
    embeddings_parameters: EmbeddingsParameters,
    scale_uint16: bool = True,
) -> DataCube:
    """Produce embeddings map using Prometheo feature extractor."""

    feature_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "feature_extractor.py",
        context=embeddings_parameters.feature_parameters.model_dump(),
    )
    embeddings = _run_udf(inputs, feature_udf)
    embeddings = _reduce_temporal_mean(embeddings)

    if scale_uint16:
        OFFSET = -6
        SCALE = 0.0002
        embeddings = (embeddings - OFFSET) / SCALE
        embeddings = embeddings.linear_scale_range(0, 65534, 0, 65534)

    return embeddings
