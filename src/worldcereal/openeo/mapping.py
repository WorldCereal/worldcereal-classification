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
    product: WorldCerealProductType | str,
    temporal: TemporalContext,
    raw: bool = False,
) -> str:
    stem = (
        product.value if isinstance(product, WorldCerealProductType) else str(product)
    )
    suffix = "-raw" if raw else ""
    return f"{stem}{suffix}_{temporal.start_date}_{temporal.end_date}"


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
    return cube


def _dedupe_bands(bands: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for band in bands:
        if band not in seen:
            seen.add(band)
            ordered.append(band)
    return ordered


def _split_auxiliary_bands(
    band_names: Sequence[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split UDF outputs into classification, ndvi, embedding dims and embedding-scale groups."""

    ndvi = [band for band in band_names if band.startswith("ndvi:")]
    embedding_dims = [
        band for band in band_names if band.startswith("global_embedding:dim_")
    ]
    emb_scale = [band for band in band_names if band == "global_embedding:scale"]
    classification = [
        band
        for band in band_names
        if band not in set(ndvi)
        and band not in set(embedding_dims)
        and band not in set(emb_scale)
    ]
    return (
        _dedupe_bands(classification),
        _dedupe_bands(ndvi),
        _dedupe_bands(embedding_dims),
        _dedupe_bands(emb_scale),
    )


def _is_supported_workflow_band(band: str) -> bool:
    if band in {
        "cropland_classification",
        "probability_cropland",
        "probability_other",
        "global_embedding:scale",
    }:
        return True
    if band.startswith("croptype_classification:"):
        return True
    if band.startswith("croptype_probability:"):
        return True
    if band.startswith("ndvi:"):
        return True
    if band.startswith("global_embedding:dim_"):
        return True
    return False


def _embeddings_export_requested(workflow_context: Mapping[str, Any]) -> bool:
    def _resolve_in_block(block: Optional[Mapping[str, Any]]) -> Optional[bool]:
        if not isinstance(block, Mapping):
            return None
        model = block.get("model")
        if not isinstance(model, Mapping):
            return None
        value = model.get("export_embeddings")
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, str)):
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        return None

    for key in ("workflow_config", "seasonal_workflow"):
        resolved = _resolve_in_block(workflow_context.get(key))
        if resolved is not None:
            return resolved

    direct = workflow_context.get("export_embeddings")
    if isinstance(direct, bool):
        return direct
    if isinstance(direct, (int, str)):
        return str(direct).strip().lower() in {"1", "true", "yes", "on"}
    return False


def _merge_products_requested(workflow_context: Mapping[str, Any]) -> bool:
    season = workflow_context.get("workflow_config", workflow_context).get("season", {})
    value = season.get("merge_classification_products")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return False


def _validate_band_contract(
    band_names: Sequence[str], *, require_embeddings_bundle: bool = False
) -> None:
    unknown = [band for band in band_names if not _is_supported_workflow_band(band)]
    if unknown:
        unknown_sorted = ", ".join(sorted(set(unknown)))
        raise ValueError(
            "Seasonal UDF emitted unsupported output bands: "
            f"{unknown_sorted}. Update mapping contract before exporting."
        )

    embedding_dims = [
        band for band in band_names if band.startswith("global_embedding:dim_")
    ]
    has_embedding_dims = bool(embedding_dims)
    has_embedding_scale = "global_embedding:scale" in band_names

    if has_embedding_dims != has_embedding_scale:
        raise ValueError(
            "Embeddings outputs must include both quantized dimensions and global_embedding:scale."
        )

    if require_embeddings_bundle and not (has_embedding_dims and has_embedding_scale):
        raise ValueError(
            "export_embeddings=True but embeddings bundle is incomplete or missing from UDF outputs."
        )


def _save_optional_scaled_result(
    cube: DataCube,
    bands: Sequence[str],
    prefix: str,
    *,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> Optional[DataCube]:
    if not bands:
        return None
    selected = cube.filter_bands(list(bands))
    scaled = selected.linear_scale_range(in_min, in_max, out_min, out_max)
    return _save_result(scaled, prefix)


def _save_optional_result(
    cube: DataCube,
    bands: Sequence[str],
    prefix: str,
) -> Optional[DataCube]:
    if not bands:
        return None
    return _save_result(cube.filter_bands(list(bands)), prefix)


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
    _validate_band_contract(
        band_names,
        require_embeddings_bundle=_embeddings_export_requested(cropland_context),
    )
    classification_bands, ndvi_bands, embedding_dim_bands, emb_scale_bands = (
        _split_auxiliary_bands(
            [band for band in band_names if not band.startswith("croptype_")]
        )
    )
    if not classification_bands:
        raise ValueError(
            "Seasonal UDF did not emit cropland bands; expected cropland outputs."
        )

    outputs: List[DataCube] = []
    cropland_main = _save_optional_scaled_result(
        cube,
        classification_bands,
        _filename_prefix(WorldCerealProductType.CROPLAND, temporal_extent),
        in_min=0,
        in_max=254,
        out_min=0,
        out_max=254,
    )
    if cropland_main is not None:
        outputs.append(cropland_main)

    ndvi_output = _save_optional_scaled_result(
        cube,
        ndvi_bands,
        _filename_prefix("ndvi", temporal_extent),
        in_min=0,
        in_max=254,
        out_min=0,
        out_max=254,
    )
    if ndvi_output is not None:
        outputs.append(ndvi_output)

    embeddings_output = _save_optional_scaled_result(
        cube,
        embedding_dim_bands,
        _filename_prefix("embeddings", temporal_extent),
        in_min=0,
        in_max=254,
        out_min=0,
        out_max=254,
    )
    if embeddings_output is not None:
        outputs.append(embeddings_output)

    emb_scale_output = _save_optional_result(
        cube,
        emb_scale_bands,
        _filename_prefix("embeddings-scale", temporal_extent),
    )
    if emb_scale_output is not None:
        outputs.append(emb_scale_output)

    return outputs


def _croptype_map(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    workflow_context: Mapping[str, Any],
) -> List[DataCube]:
    """Produce per-season crop-type products alongside a dedicated cropland product."""

    cube = _workflow_inference_cube(inputs, workflow_context)
    band_names = cube.metadata.band_names
    _validate_band_contract(
        band_names,
        require_embeddings_bundle=_embeddings_export_requested(workflow_context),
    )
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

    merge_products = _merge_products_requested(workflow_context)
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
        if not merge_products:
            season_cube = cube.filter_bands(season_bands).rename_labels(
                dimension="bands",
                target=normalized_labels,
            )
            season_cube = season_cube.linear_scale_range(0, 254, 0, 254)
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

    if not season_order:
        raise ValueError(
            "Seasonal UDF did not emit croptype bands that could be assigned to a season."
        )

    if merge_products:
        all_croptype_bands = [
            band for band in band_names if band.startswith("croptype_")
        ]
        if not all_croptype_bands:
            raise ValueError(
                "Seasonal UDF did not emit croptype bands; cannot build merged product."
            )
        cropland_bands_for_merge, _, _, _ = _split_auxiliary_bands(
            [band for band in band_names if not band.startswith("croptype_")]
        )
        # merged_bands = all cropland outputs (classification + probabilities)
        #              + all croptype outputs (classification + probabilities, all seasons)
        merged_bands = cropland_bands_for_merge + all_croptype_bands
        merged_cube = cube.filter_bands(merged_bands).linear_scale_range(0, 254, 0, 254)
        season_results.append(
            _save_result(
                merged_cube,
                _filename_prefix("cropland-croptype", temporal_extent),
            )
        )

    cropland_bands = [band for band in band_names if not band.startswith("croptype_")]
    (
        cropland_classification_bands,
        ndvi_bands,
        embedding_dim_bands,
        emb_scale_bands,
    ) = _split_auxiliary_bands(cropland_bands)

    additional_results: List[DataCube] = []
    if not merge_products:
        cropland_output = _save_optional_scaled_result(
            cube,
            cropland_classification_bands,
            _filename_prefix(WorldCerealProductType.CROPLAND, temporal_extent),
            in_min=0,
            in_max=254,
            out_min=0,
            out_max=254,
        )
        if cropland_output is not None:
            additional_results.append(cropland_output)

    ndvi_output = _save_optional_scaled_result(
        cube,
        ndvi_bands,
        _filename_prefix("ndvi", temporal_extent),
        in_min=0,
        in_max=254,
        out_min=0,
        out_max=254,
    )
    if ndvi_output is not None:
        additional_results.append(ndvi_output)

    embeddings_output = _save_optional_scaled_result(
        cube,
        embedding_dim_bands,
        _filename_prefix("embeddings", temporal_extent),
        in_min=0,
        in_max=254,
        out_min=0,
        out_max=254,
    )
    if embeddings_output is not None:
        additional_results.append(embeddings_output)

    emb_scale_output = _save_optional_result(
        cube,
        emb_scale_bands,
        _filename_prefix("embeddings-scale", temporal_extent),
    )
    if emb_scale_output is not None:
        additional_results.append(emb_scale_output)

    return season_results + additional_results


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
