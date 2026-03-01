"""Executing inference jobs on the OpenEO backend.

Inference entry points now live in the unified job manager and CLI helpers.

"""

import datetime as dt
import json
from copy import deepcopy
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import openeo
import pandas as pd
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import BACKEND_CONNECTIONS

from worldcereal.openeo.inference import (
    _merge_workflow_sections,
    _select_workflow_preset,
    load_model_artifact,
)
from worldcereal.openeo.mapping import _cropland_map, _croptype_map, _embeddings_map
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs
from worldcereal.openeo.workflow_config import WorldCerealWorkflowConfig
from worldcereal.parameters import EmbeddingsParameters, WorldCerealProductType

FEATURE_DEPS_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/torch_deps_python311.zip"
PROMETHEO_WHL_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/prometheo-0.0.3-py3-none-any.whl"
WORLDCEREAL_WHL_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/worldcereal/worldcereal-2.5.0-py3-none-any.whl"
DEFAULT_INFERENCE_JOB_OPTIONS = {
    "driver-memory": "4g",
    "executor-memory": "2g",
    "executor-memoryOverhead": "3g",
    "executor-request-cores": "1800m",
    "max-executors": 20,
    "python-memory": "disable",
    "soft-errors": 0.1,
    "udf-dependency-archives": [
        f"{FEATURE_DEPS_URL}#feature_deps",
        f"{PROMETHEO_WHL_URL}#prometheolib",
        f"{WORLDCEREAL_WHL_URL}#worldcereallib",
    ],
}
DEFAULT_INPUTS_JOB_OPTIONS = {
    "driver-memory": "4g",
    "executor-memory": "2g",
    "executor-memoryOverhead": "1g",
    "python-memory": "4g",
    "soft-errors": 0.1,
    "max-executors": 10,
}

SeasonWindowMapping = Dict[str, Tuple[str, str]]
ManifestDict = Mapping[str, Any]
ClassLUT = Dict[str, int]


class WorldCerealTask(str, Enum):
    INPUTS = "inputs"
    EMBEDDINGS = "embeddings"
    INFERENCE = "inference"


def _normalize_season_id_list(value: Optional[Union[str, Sequence[Any]]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        tokens = [
            piece.strip()
            for piece in value.replace(";", ",").split(",")
            if piece.strip()
        ]
        return tokens
    if isinstance(value, Sequence):
        collected: List[str] = []
        for element in value:
            if isinstance(element, str):
                collected.extend(_normalize_season_id_list(element))
            elif element is not None:
                collected.append(str(element))
        return collected
    return [str(value)]


def _deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _try_parse_jsonish(value: str) -> Optional[Any]:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _parse_window_string(value: str) -> Optional[Mapping[str, Tuple[str, str]]]:
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    mapping: Dict[str, Tuple[str, str]] = {}
    for entry in entries:
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) == 3:
            mapping[parts[0]] = (parts[1], parts[2])
    return mapping or None


def _as_window_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        parsed = _try_parse_jsonish(value)
        if isinstance(parsed, Mapping):
            return parsed
        parsed = _parse_window_string(value)
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _datetime64_to_str(value: np.datetime64) -> str:
    return value.astype("datetime64[D]").astype(str)


def _to_datetime64(value: Any) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    if isinstance(value, (dt.date, dt.datetime)):
        return np.datetime64(value.strftime("%Y-%m-%d"), "D")
    text = str(value).strip()
    try:
        return np.datetime64(text, "D")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unable to parse date '{value}'") from exc


def _coerce_iso_date(value: Any) -> str:
    return _datetime64_to_str(_to_datetime64(value))


def _coerce_temporal_bounds(
    temporal_extent: TemporalContext,
) -> Tuple[np.datetime64, np.datetime64]:
    start = _to_datetime64(temporal_extent.start_date)
    end = _to_datetime64(temporal_extent.end_date)
    if end < start:
        raise ValueError("Temporal extent end date precedes start date.")
    return start, end


def _coerce_window_pair(value: Any) -> Tuple[str, str]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    elif isinstance(value, Mapping):
        start = value.get("start") or value.get("begin")
        end = value.get("end") or value.get("stop")
    elif isinstance(value, str):
        parts = [piece.strip() for piece in value.split(":")]
        if len(parts) != 2:
            raise ValueError(
                "Season window strings must follow 'START:END' format when used standalone."
            )
        start, end = parts
    else:
        raise TypeError(f"Unsupported season window specification: {value!r}")
    if start is None or end is None:
        raise ValueError("Season windows require both start and end dates.")
    return _coerce_iso_date(start), _coerce_iso_date(end)


def _normalize_season_windows(
    raw_windows: Mapping[str, Any], *, expected_ids: Optional[Sequence[str]] = None
) -> SeasonWindowMapping:
    normalized: SeasonWindowMapping = {}
    keys_to_process: Sequence[str]
    if expected_ids:
        keys_to_process = [str(season_id) for season_id in expected_ids]
    else:
        keys_to_process = [str(key) for key in raw_windows.keys()]

    for season_id in keys_to_process:
        if season_id not in raw_windows:
            if expected_ids:
                raise ValueError(f"Season window missing for '{season_id}'.")
            continue
        normalized[season_id] = _coerce_window_pair(raw_windows[season_id])

    if not expected_ids:
        for key, value in raw_windows.items():
            key_str = str(key)
            normalized.setdefault(key_str, _coerce_window_pair(value))
    return normalized


def _season_ids_from_row(row: Optional[pd.Series]) -> List[str]:
    if row is None:
        return []
    for key in ("season_ids", "season_id"):
        if key in row and pd.notna(row[key]):
            return _normalize_season_id_list(row[key])
    return []


def _season_windows_from_row(
    row: Optional[pd.Series], season_ids: Sequence[str]
) -> Optional[SeasonWindowMapping]:
    if row is None:
        return None
    for key in ("season_windows", "season_window"):
        if key in row and pd.notna(row[key]):
            mapping = _as_window_mapping(row[key])
            if mapping:
                expected = season_ids if season_ids else None
                return _normalize_season_windows(mapping, expected_ids=expected)
    return None


def _workflow_sections_from_config(
    workflow_config: Optional[WorldCerealWorkflowConfig],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not workflow_config:
        return None
    config_dict = workflow_config.to_dict()
    if not config_dict:
        return None
    return deepcopy(config_dict)


def _apply_row_overrides(
    workflow_cfg: Dict[str, Dict[str, Any]], row: Optional[pd.Series]
) -> None:
    if row is None:
        return
    season_cfg = workflow_cfg.setdefault("season", {})
    model_cfg = workflow_cfg.setdefault("model", {})
    row_ids = _deduplicate_preserve_order(_season_ids_from_row(row))
    if row_ids:
        season_cfg["season_ids"] = row_ids
    row_windows = _season_windows_from_row(row, row_ids)
    if row_windows:
        merged_windows = dict(season_cfg.get("season_windows") or {})
        merged_windows.update(row_windows)
        season_cfg["season_windows"] = merged_windows

    for key in ("seasonal_model_zip", "landcover_head_zip", "croptype_head_zip"):
        if key in row and pd.notna(row[key]):
            model_cfg[key] = row[key]


def _compose_workflow_sections(
    preset: str, *override_blocks: Optional[Mapping[str, Dict[str, Any]]]
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    workflow_cfg, preset_name = _select_workflow_preset(preset)
    workflow_cfg.setdefault("model", {})
    workflow_cfg.setdefault("runtime", {})
    workflow_cfg.setdefault("season", {})
    workflow_cfg.setdefault("postprocess", {})
    for overrides in override_blocks:
        if overrides:
            _merge_workflow_sections(workflow_cfg, overrides)
    return workflow_cfg, preset_name


def _finalize_season_requirements(
    workflow_cfg: Dict[str, Dict[str, Any]],
    *,
    temporal_extent: TemporalContext,
) -> None:
    season_cfg = workflow_cfg.setdefault("season", {})
    season_ids = [str(season_id) for season_id in season_cfg.get("season_ids", [])]
    if not season_ids:
        raise ValueError(
            "Seasonal workflow configuration requires at least one season identifier."
        )
    season_cfg["season_ids"] = season_ids

    raw_windows = season_cfg.get("season_windows")
    normalized_windows: Optional[SeasonWindowMapping] = None
    if raw_windows:
        normalized_windows = _normalize_season_windows(raw_windows, expected_ids=None)
        missing = [sid for sid in season_ids if sid not in normalized_windows]
        if missing:
            raise ValueError(
                "Season workflow configuration is missing windows for: "
                + ", ".join(missing)
            )
    elif len(season_ids) == 1:
        start, end = _coerce_temporal_bounds(temporal_extent)
        normalized_windows = {
            season_ids[0]: (
                _datetime64_to_str(start),
                _datetime64_to_str(end),
            )
        }
    else:
        raise ValueError(
            "Provide explicit season windows when configuring multiple season identifiers."
        )

    season_cfg["season_windows"] = normalized_windows


def _build_workflow_context(
    *,
    preset: str,
    temporal_extent: TemporalContext,
    override_blocks: Sequence[Optional[Mapping[str, Dict[str, Any]]]] = (),
    row: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    workflow_cfg, _ = _compose_workflow_sections(preset, *override_blocks)
    _apply_row_overrides(workflow_cfg, row)
    _finalize_season_requirements(workflow_cfg, temporal_extent=temporal_extent)
    return {"workflow_config": workflow_cfg}


def _resolve_model_config(
    preset: str, overrides: Optional[Mapping[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    workflow_cfg, _ = _compose_workflow_sections(preset, overrides)
    model_cfg = workflow_cfg.get("model", {})
    if not isinstance(model_cfg, dict) or not model_cfg.get("seasonal_model_zip"):
        raise ValueError(
            "Seasonal workflow configuration is missing 'seasonal_model_zip'."
        )
    return model_cfg


@lru_cache(maxsize=8)
def _get_artifact_manifest(source: str) -> ManifestDict:
    artifact = load_model_artifact(source)
    return deepcopy(artifact.manifest)


def _lut_from_manifest(manifest: ManifestDict, task: str) -> ClassLUT:
    heads = manifest.get("heads", [])
    for head in heads:
        if head.get("task") == task:
            class_names = head.get("class_names") or []
            if not class_names:
                raise ValueError(
                    f"Head '{task}' missing class_names in manifest from seasonal model"
                )
            return {name: idx for idx, name in enumerate(class_names)}
    raise ValueError(f"Manifest does not define a '{task}' head")


def create_inference_process_graph(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    connection: Optional[openeo.Connection] = None,
    seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
    workflow_config: Optional[WorldCerealWorkflowConfig] = None,
    row: Optional[pd.Series] = None,
) -> List[openeo.DataCube]:
    """Wrapper function that creates the inference openEO process graph.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    product_type : WorldCerealProductType, optional
        product describer, by default WorldCerealProductType.CROPLAND
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]]
        Sentinel-1 orbit state to use for the inference. If not provided,
        the orbit state will be dynamically determined based on the spatial extent.
    out_format : str, optional
        Output format, by default "GTiff"
    backend_context : BackendContext
        backend to run the job on, by default CDSE.
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.
    target_epsg: Optional[int] = None
        EPSG code to use for the output products. If not provided, the
        default EPSG will be used.
    seasonal_preset: str
        Name of the seasonal workflow preset to use. Defaults to
        `DEFAULT_SEASONAL_WORKFLOW_PRESET`.
    workflow_config: Optional[WorldCerealWorkflowConfig]
        Typed helper that describes overrides using Python objects instead of raw
        dictionaries. When omitted, the preset values are used as-is.
    row: Optional[pd.Series]
        Optional production-grid row providing additional metadata (season ids
        or windows). Only required when invoking the helper inside the
        job manager inference runner.
    connection: Optional[openeo.Connection] = None,
        Optional OpenEO connection to use. If not provided, a new connection
        will be created based on the backend_context.

    Returns
    -------
    List[openeo.DataCube]
        A list with one or more result objects or a list of DataCube objects, representing the inference
        process graph. This object can be used to execute the job on the OpenEO backend.
        The result will be a DataCube with the classification results.

    Raises
    ------
    ValueError
        if the product is not supported
    ValueError
        if the out_format is not supported
    """
    if product_type not in WorldCerealProductType:
        raise ValueError(f"Product {product_type.value} not supported.")

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    inputs = _build_inputs_cube(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        s1_orbit_state=s1_orbit_state,
        backend_context=backend_context,
        tile_size=tile_size,
        target_epsg=target_epsg,
        connection=connection,
    )

    config_overrides = _workflow_sections_from_config(workflow_config)
    workflow_context = _build_workflow_context(
        preset=seasonal_preset,
        temporal_extent=temporal_extent,
        override_blocks=(config_overrides,),
        row=row,
    )

    # Construct the feature extraction and model inference pipeline
    if product_type == WorldCerealProductType.CROPLAND:
        results = _cropland_map(
            inputs,
            temporal_extent,
            workflow_context=workflow_context,
        )

    elif product_type == WorldCerealProductType.CROPTYPE:
        # Generate crop type map with optional cropland masking
        results = _croptype_map(
            inputs,
            temporal_extent,
            workflow_context=workflow_context,
        )

    return results


def create_embeddings_process_graph(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    embeddings_parameters: EmbeddingsParameters = EmbeddingsParameters(),
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    scale_uint16: bool = True,
    connection: Optional[openeo.Connection] = None,
) -> openeo.DataCube:
    """Create an OpenEO process graph for generating embeddings.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        Spatial extent of the map.
    temporal_extent : TemporalContext
        Temporal range to consider.
    embeddings_parameters : EmbeddingsParameters, optional
        Parameters for the embeddings product inference pipeline, by default EmbeddingsParameters().
    s1_orbit_state : Optional[Literal["ASCENDING", "DESCENDING"]], optional
        Sentinel-1 orbit state to use for the inference. If not provided, the orbit state will be dynamically determined based on the spatial extent, by default None.
    out_format : str, optional
        Output format, by default "GTiff".
    backend_context : BackendContext, optional
        Backend to run the job on, by default BackendContext(Backend.CDSE).
    tile_size : Optional[int], optional
        Tile size to use for the data loading in OpenEO, by default 128.
    target_epsg : Optional[int], optional
        EPSG code to use for the output products. If not provided, the default EPSG will be used.
    scale_uint16 : bool, optional
        Whether to scale the embeddings to uint16 for memory optimization, by default True.
    connection : Optional[openeo.Connection], optional
        Existing OpenEO connection to reuse. If omitted, a new connection is created.

    Returns
    -------
    openeo.DataCube
        DataCube object representing the embeddings process graph. This object can be used to execute the job on the OpenEO backend. The result will be a DataCube with the embeddings.

    Raises
    ------
    ValueError
        If the output format is not supported.
    """

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    inputs = _build_inputs_cube(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        s1_orbit_state=s1_orbit_state,
        backend_context=backend_context,
        tile_size=tile_size,
        target_epsg=target_epsg,
        connection=connection,
    )

    embeddings = _embeddings_map(
        inputs,
        temporal_extent,
        embeddings_parameters=embeddings_parameters,
        scale_uint16=scale_uint16,
    )

    # Save the final result
    embeddings = embeddings.save_result(
        format=out_format,
        options=dict(
            filename_prefix=f"WorldCereal_Embeddings_{temporal_extent.start_date}_{temporal_extent.end_date}",
        ),
    )

    return embeddings


def create_inputs_process_graph(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: str = "NetCDF",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    compositing_window: Literal["month", "dekad"] = "month",
    connection: Optional[openeo.Connection] = None,
) -> openeo.DataCube:
    """Wrapper function that creates the inputs openEO process graph.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]]
        Sentinel-1 orbit state to use for the inference. If not provided,
        the orbit state will be dynamically determined based on the spatial extent.
    out_format : str, optional
        Output format, by default "NetCDF"
    backend_context : BackendContext
        backend to run the job on, by default CDSE.
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.
    target_epsg: Optional[int] = None
        EPSG code to use for the output products. If not provided, the
        default EPSG will be used.
    compositing_window: Literal["month", "dekad"]
        Compositing window to use for the data loading in OpenEO, by default
        "month".
    connection : Optional[openeo.Connection], optional
        Existing OpenEO connection to reuse. If omitted, a new connection is created.

    Returns
    -------
    openeo.DataCube
        DataCube object representing the inputs process graph.
        This object can be used to execute the job on the OpenEO backend.
        The result will be a DataCube with the preprocessed inputs.

    Raises
    ------
    ValueError
        if the out_format is not supported
    """

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    inputs = _build_inputs_cube(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        s1_orbit_state=s1_orbit_state,
        backend_context=backend_context,
        tile_size=tile_size,
        target_epsg=target_epsg,
        compositing_window=compositing_window,
        connection=connection,
    )

    # Save the final result
    inputs = inputs.save_result(
        format=out_format,
        options=dict(
            filename_prefix=f"preprocessed-inputs_{temporal_extent.start_date}_{temporal_extent.end_date}",
        ),
    )

    return inputs


def create_worldcereal_process_graph(
    task: WorldCerealTask,
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: Optional[str] = None,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    connection: Optional[openeo.Connection] = None,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    embeddings_parameters: EmbeddingsParameters = EmbeddingsParameters(),
    scale_uint16: bool = True,
    compositing_window: Literal["month", "dekad"] = "month",
    seasonal_preset: str = DEFAULT_SEASONAL_WORKFLOW_PRESET,
    workflow_config: Optional[WorldCerealWorkflowConfig] = None,
    row: Optional[pd.Series] = None,
) -> Union[openeo.DataCube, List[openeo.DataCube]]:
    """Create a WorldCereal process graph based on the requested task."""

    task = WorldCerealTask(task)

    if task == WorldCerealTask.INPUTS:
        return create_inputs_process_graph(
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            s1_orbit_state=s1_orbit_state,
            out_format=out_format or "NetCDF",
            backend_context=backend_context,
            tile_size=tile_size,
            target_epsg=target_epsg,
            compositing_window=compositing_window,
            connection=connection,
        )

    if task == WorldCerealTask.EMBEDDINGS:
        return create_embeddings_process_graph(
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            embeddings_parameters=embeddings_parameters,
            s1_orbit_state=s1_orbit_state,
            out_format=out_format or "GTiff",
            backend_context=backend_context,
            tile_size=tile_size,
            target_epsg=target_epsg,
            scale_uint16=scale_uint16,
            connection=connection,
        )

    if task == WorldCerealTask.INFERENCE:
        return create_inference_process_graph(
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            product_type=product_type,
            s1_orbit_state=s1_orbit_state,
            out_format=out_format or "GTiff",
            backend_context=backend_context,
            tile_size=tile_size,
            target_epsg=target_epsg,
            connection=connection,
            seasonal_preset=seasonal_preset,
            workflow_config=workflow_config,
            row=row,
        )

    raise ValueError(f"Unsupported task: {task}")


def _build_inputs_cube(
    *,
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    compositing_window: Optional[Literal["month", "dekad"]] = "month",
    connection: Optional[openeo.Connection] = None,
) -> openeo.DataCube:
    if connection is None:
        connection = BACKEND_CONNECTIONS[backend_context.backend]()

    inputs = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
        compositing_window=compositing_window,
    )

    return inputs.filter_bbox(dict(spatial_extent))
