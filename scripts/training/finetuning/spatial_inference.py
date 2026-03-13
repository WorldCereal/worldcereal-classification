#!/usr/bin/env python3
"""Run local spatial inference on NetCDF patch directories and export panel PNGs.

This module provides:
1) an importable API function (`run_spatial_inference`) that can be called
   from finetuning scripts;
2) a CLI entrypoint for inference-only runs on existing model folders.

The inference execution path reuses the notebook helper in
``notebooks/notebook_utils/local_inference.py`` to keep parity with notebook
local inference behaviour.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import path
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from openeo_gfmap import BoundingBoxExtent
from pyproj import CRS, Transformer
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from worldcereal.seasons import get_season_dates_for_extent

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_INFERENCE_PATH = REPO_ROOT / "notebooks" / "notebook_utils" / "local_inference.py"
WORLD_BOUNDS_PATH = (
    REPO_ROOT
    / "src"
    / "worldcereal"
    / "data"
    / "world-administrative-boundaries"
    / "world-administrative-boundaries.geoparquet"
)
CLASS_MAPPINGS_PATH = (
    REPO_ROOT
    / "src"
    / "worldcereal"
    / "data"
    / "croptype_mappings"
    / "class_mappings.json"
)
DEFAULT_SEASON_IDS = ("tc-s1", "tc-s2")

# -- Season-ID ↔ head-key helpers ------------------------------------------
_SEASON_TO_HEAD: Dict[str, str] = {
    "tc-s1": "CROPTYPE_S1",
    "tc-s2": "CROPTYPE_S2",
    "tc-annual": "CROPTYPE_ANNUAL",
}
_HEAD_TO_SEASON: Dict[str, str] = {v: k for k, v in _SEASON_TO_HEAD.items()}


def _season_id_to_head_key(season_id: str) -> str:
    """Map a season ID to the corresponding head key for display/indexing."""
    return _SEASON_TO_HEAD.get(
        season_id,
        f"CROPTYPE_{season_id.upper().replace('-', '_')}",
    )


def _head_key_to_season_id(head_key: str) -> Optional[str]:
    """Map a head key back to its season ID (``None`` if not a croptype head)."""
    return _HEAD_TO_SEASON.get(head_key)


# -- GT variable helpers per head key --------------------------------------
_HEAD_GT_CONFIG: Dict[str, Dict[str, Any]] = {
    "LANDCOVER": {
        "candidates": [
            "WORLDCEREAL_LANDCOVER_GT",
            "LANDCOVER10_GT",
            "landcover_gt",
            "landcover",
        ],
        "gt_var": "WORLDCEREAL_LANDCOVER_GT",
    },
    "CROPTYPE_S1": {
        "candidates": ["WORLDCEREAL_SEASON1_GT", "worldcereal_season1_gt"],
        "gt_var": "WORLDCEREAL_SEASON1_GT",
    },
    "CROPTYPE_S2": {
        "candidates": ["WORLDCEREAL_SEASON2_GT", "worldcereal_season2_gt"],
        "gt_var": "WORLDCEREAL_SEASON2_GT",
    },
    "CROPTYPE_ANNUAL": {
        # Annual season: use season-1 GT as best available proxy.
        "candidates": [
            "WORLDCEREAL_SEASON1_GT",
            "worldcereal_season1_gt",
            "WORLDCEREAL_SEASON2_GT",
            "worldcereal_season2_gt",
        ],
        "gt_var": "WORLDCEREAL_SEASON1_GT",
    },
}


NO_CROP_VALUE = 254
DEBUG_CROP_SIZE = 256
PANEL_TITLE_FONTSIZE = 40
LEGEND_FONTSIZE = 26
PANEL_TITLE_PAD = 6
MAIN_TITLE_FONTSIZE = PANEL_TITLE_FONTSIZE + 10
MODEL_TITLE_FONTSIZE = int(MAIN_TITLE_FONTSIZE / 2)
MODEL_TITLE_Y_OFFSET = 0.03
SUPTITLE_Y = 0.97
SUBPLOT_TOP = 0.90
SUBPLOT_LEFT = 0.02
SUBPLOT_RIGHT = 0.98
SUBPLOT_BOTTOM = 0.02
SUBPLOT_HSPACE = 0.12
SUBPLOT_WSPACE = 0.05


@dataclass
class HeadSpec:
    key: str
    display_name: str
    class_names: List[str]
    class_colors: Dict[int, Tuple[float, float, float, float]]


@dataclass(frozen=True)
class DebugCropLabel:
    label: str
    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int


def _load_local_inference_module():
    """Load notebook local inference helpers via file-path import.

    TODO: promote notebook local inference code to a src/ package module and
    import it directly from there.
    """

    if not LOCAL_INFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find local inference module at {LOCAL_INFERENCE_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "worldcereal_notebook_local_inference", LOCAL_INFERENCE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {LOCAL_INFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_continents(
    continents: Union[str, Sequence[str], None]
) -> Union[str, List[str]]:
    if continents is None:
        return "all"
    if isinstance(continents, str):
        raw = [continents]
    else:
        raw = list(continents)

    parsed: List[str] = []
    for item in raw:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                parsed.append(token)
    if not parsed:
        return "all"
    if any(token.lower() == "all" for token in parsed):
        return "all"
    return sorted(set(parsed), key=lambda x: x.lower())


def list_patches(
    patches_dir: Union[str, Path],
    continents: Union[str, Sequence[str]] = "all",
) -> List[Tuple[str, Path]]:
    """Discover patch files with stable deterministic ordering."""

    root = Path(patches_dir)
    if not root.exists():
        raise FileNotFoundError(f"Patches directory not found: {root}")

    selection = _normalize_continents(continents)

    found: List[Tuple[str, Path]] = []
    if selection == "all":
        for continent_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            for nc_path in sorted(continent_dir.rglob("*.nc")):
                found.append((continent_dir.name, nc_path))
        if len(found) == 0:
            fallback_continent = root.name if root.name else "patches"
            found = [
                (fallback_continent, Path(p))
                for p in glob.glob(str(root / "*.nc"))
            ]
    else:
        assert isinstance(selection, list)
        for continent in selection:
            continent_dir = root / continent
            if not continent_dir.exists() or not continent_dir.is_dir():
                logger.warning(f"Continent folder missing, skipping: {continent_dir}")
                continue
            for nc_path in sorted(continent_dir.rglob("*.nc")):
                found.append((continent, nc_path))

    found.sort(key=lambda item: (str(item[0]).lower(), item[1].name.lower(), str(item[1])))
    return found


def _apply_debug_crop(
    ds: xr.Dataset,
    *,
    enabled: bool,
    crop_size: int = DEBUG_CROP_SIZE,
    rng: Optional[np.random.Generator] = None,
    label: Optional[str] = None,
) -> Tuple[xr.Dataset, Optional[DebugCropLabel]]:
    if not enabled:
        return ds, None
    if "x" not in ds.dims or "y" not in ds.dims:
        raise ValueError("Debug crop requires x/y dimensions on the patch dataset.")

    x_size = int(ds.sizes["x"])
    y_size = int(ds.sizes["y"])
    if x_size < crop_size or y_size < crop_size:
        label_suffix = f" for {label}" if label else ""
        logger.warning(
            f"Debug crop requested{label_suffix} but patch is smaller than "
            f"{crop_size}x{crop_size} (x={x_size}, y={y_size}). Using full patch."
        )
        full_label = f"full_{x_size}x{y_size}"
        return ds, DebugCropLabel(
            label=full_label,
            x0=0,
            y0=0,
            x1=x_size,
            y1=y_size,
            width=x_size,
            height=y_size,
        )

    if rng is None:
        rng = np.random.default_rng()
    x0 = int(rng.integers(0, x_size - crop_size + 1))
    y0 = int(rng.integers(0, y_size - crop_size + 1))
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    label_suffix = f" for {label}" if label else ""
    logger.info(
        f"Debug crop{label_suffix}: size={crop_size}x{crop_size}, x[{x0}:{x1}], y[{y0}:{y1}]"
    )
    crop_label = f"crop{crop_size}_x{x0}_y{y0}"
    return (
        ds.isel(x=slice(x0, x1), y=slice(y0, y1)),
        DebugCropLabel(
            label=crop_label,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            width=crop_size,
            height=crop_size,
        ),
    )


def _find_time_dim(ds: xr.Dataset) -> str:
    preferred = ["t", "time", "valid_time", "date", "datetime"]
    for name in preferred:
        if name in ds.dims:
            return name
    for dim in ds.dims:
        coord = ds.coords.get(dim)
        if coord is not None and np.issubdtype(coord.dtype, np.datetime64):
            return dim
    raise ValueError("Could not detect time dimension in patch dataset.")


def _pick_var(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.data_vars:
            return name
        if name in ds.coords:
            return name
    return None


def _compute_rgb_ndvi(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    time_dim = _find_time_dim(ds)

    red_name = _pick_var(ds, ["S2-L2A-B04", "B04", "B4", "red"])
    green_name = _pick_var(ds, ["S2-L2A-B03", "B03", "B3", "green"])
    blue_name = _pick_var(ds, ["S2-L2A-B02", "B02", "B2", "blue"])
    nir_name = _pick_var(ds, ["S2-L2A-B08", "B08", "B8", "nir"])

    if red_name is None or green_name is None or blue_name is None:
        raise ValueError(
            "Could not derive RGB from patch; expected S2 red/green/blue bands."
        )
    if nir_name is None:
        raise ValueError("Could not derive NDVI from patch; expected a NIR band.")

    red = ds[red_name].astype("float32")
    green = ds[green_name].astype("float32")
    blue = ds[blue_name].astype("float32")
    nir = ds[nir_name].astype("float32")

    rgb = xr.concat([red, green, blue], dim="channel").mean(dim=time_dim, skipna=True)
    rgb_np = np.moveaxis(rgb.values, 0, -1)
    scale = 10000.0 if np.nanmax(rgb_np) > 1.5 else 1.0
    rgb_np = np.clip(rgb_np / scale, 0.0, 1.0)
    valid = np.isfinite(rgb_np)
    if np.any(valid):
        p2, p98 = np.percentile(rgb_np[valid], [2, 98])
        if p98 > p2:
            rgb_np = np.clip((rgb_np - p2) / (p98 - p2), 0.0, 1.0)
    rgb_np = np.power(rgb_np, 1.0 / 2.2)

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi_mean = np.clip(ndvi.mean(dim=time_dim, skipna=True).values, -1.0, 1.0)
    return rgb_np, ndvi_mean


def _get_patch_extent(ds: xr.Dataset) -> BoundingBoxExtent:
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Patch is missing x/y coordinates required for spatial season lookup.")

    west = float(np.min(ds.coords["x"].values))
    east = float(np.max(ds.coords["x"].values))
    south = float(np.min(ds.coords["y"].values))
    north = float(np.max(ds.coords["y"].values))

    epsg = 4326
    if "crs" in ds and "spatial_ref" in ds["crs"].attrs:
        try:
            parsed_epsg = CRS.from_wkt(ds["crs"].attrs["spatial_ref"]).to_epsg()
            if parsed_epsg is not None:
                epsg = int(parsed_epsg)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to parse patch CRS WKT; falling back to EPSG:4326 ({exc})")

    return BoundingBoxExtent(west=west, south=south, east=east, north=north, epsg=epsg)


def _get_patch_time_range(ds: xr.Dataset) -> Tuple[pd.Timestamp, pd.Timestamp]:
    time_dim = _find_time_dim(ds)
    tvals = pd.to_datetime(ds[time_dim].values)
    if len(tvals) == 0:
        raise ValueError("Patch contains no timestamps.")
    return pd.Timestamp(tvals.min()), pd.Timestamp(tvals.max())


def _overlap_days(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> int:
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    if end < start:
        return 0
    return int((end - start).days) + 1


def _derive_season_windows_for_patch(
    ds: xr.Dataset,
    season_ids: Sequence[str],
) -> Dict[str, Tuple[str, str]]:
    """Infer season windows for this patch and fit them to available patch dates."""

    if not season_ids:
        return {}

    extent = _get_patch_extent(ds)
    data_start, data_end = _get_patch_time_range(ds)
    candidate_years = list(range(data_start.year - 1, data_end.year + 2))

    season_windows: Dict[str, Tuple[str, str]] = {}
    for season_id in season_ids:
        best_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        best_overlap = -1
        best_year_distance = 10**9

        for year in candidate_years:
            try:
                tc = get_season_dates_for_extent(extent, year=year, season=season_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    f"Season lookup failed for {season_id} in year {year}: {exc}"
                )
                continue

            start = pd.to_datetime(tc.start_date)
            end = pd.to_datetime(tc.end_date)
            overlap = _overlap_days(start, end, data_start, data_end)
            year_distance = abs(year - data_end.year)
            if (
                overlap > best_overlap
                or (overlap == best_overlap and year_distance < best_year_distance)
            ):
                best_overlap = overlap
                best_year_distance = year_distance
                best_window = (start, end)

        if best_window is None:
            logger.warning(
                f"No crop-calendar season window could be inferred for {season_id}; using patch time range instead."
            )
            start, end = data_start, data_end
        else:
            start, end = best_window
            if best_overlap <= 0:
                logger.warning(
                    f"Season {season_id} has no overlap with patch timestamps; clamping to patch range."
                )
                start, end = data_start, data_end

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        clamped_start = max(start_ts, data_start)
        clamped_end = min(end_ts, data_end)
        if clamped_end < clamped_start:
            logger.warning(
                f"Season {season_id} window is invalid after clamping; using patch range."
            )
            clamped_start, clamped_end = data_start, data_end
        elif clamped_start != start_ts or clamped_end != end_ts:
            logger.info(
                f"Clamping season {season_id} window to patch range "
                f"({data_start:%Y-%m-%d} to {data_end:%Y-%m-%d})."
            )

        season_windows[season_id] = (
            clamped_start.strftime("%Y-%m-%d"),
            clamped_end.strftime("%Y-%m-%d"),
        )

    return season_windows


def _extract_fill_values(da: xr.DataArray) -> List[float]:
    fill_values: List[float] = []
    for key in ("_FillValue", "missing_value", "nodata", "no_data"):
        value = da.attrs.get(key)
        if value is None:
            continue
        try:
            fill_values.append(float(value))
        except (TypeError, ValueError):
            pass
    return fill_values


def _to_2d_array(da: xr.DataArray, target_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(da.squeeze().values)
    if arr.shape == target_shape:
        return arr
    if arr.ndim == 2 and arr.T.shape == target_shape:
        return arr.T
    raise ValueError(
        f"GT shape mismatch; expected {target_shape} but got {arr.shape} for {da.name}."
    )


def get_gt_for_row(ds: xr.Dataset, row_key: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Get ground-truth array per output row, if present and interpretable.

    *row_key* can be an uppercase head key (``"CROPTYPE_S1"``, ``"CROPTYPE_ANNUAL"``,
    ``"LANDCOVER"``) or a legacy lowercase key (``"croptype_s1"`` etc.).
    """
    # Normalise to uppercase head key for config lookup.
    lookup_key = row_key.upper()

    if lookup_key not in _HEAD_GT_CONFIG:
        # Try legacy lowercase conversions.
        _LOWER_TO_HEAD = {
            "croptype_s1": "CROPTYPE_S1",
            "croptype_s2": "CROPTYPE_S2",
            "landcover": "LANDCOVER",
        }
        lookup_key = _LOWER_TO_HEAD.get(row_key, lookup_key)

    cfg = _HEAD_GT_CONFIG.get(lookup_key)
    if cfg is None:
        return None
    candidates: List[str] = cfg["candidates"]

    for var in candidates:
        if var not in ds:
            continue
        try:
            return _to_2d_array(ds[var], target_shape)
        except ValueError:
            continue

    return None


def _valid_mask_for_gt(gt: np.ndarray, fill_values: Iterable[float]) -> np.ndarray:
    mask = np.isfinite(gt)
    for fv in fill_values:
        mask &= gt != fv
    mask &= gt >= 0
    mask &= gt != 255
    return mask


def _compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    fill_values: Iterable[float] = (),
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    valid = _valid_mask_for_gt(gt, fill_values)
    valid &= np.isfinite(pred)
    valid &= (gt < num_classes)

    if not np.any(valid):
        return None, None, None

    gt_v = gt[valid].astype(np.int64)
    pred_v = pred[valid].astype(np.int64)

    acc = float(accuracy_score(gt_v, pred_v))
    macro = float(
        f1_score(gt_v, pred_v, average="macro", labels=np.arange(num_classes), zero_division=0)
    )
    per_class = f1_score(
        gt_v,
        pred_v,
        average=None,
        labels=np.arange(num_classes),
        zero_division=0,
    )
    return acc, macro, per_class


def _build_color_map(class_names: Sequence[str]) -> Dict[int, Tuple[float, float, float, float]]:
    """Deterministic class colors with fixed ordering by class index."""

    if len(class_names) == 2:
        return {
            0: (186 / 255, 186 / 255, 186 / 255, 1.0),
            1: (224 / 255, 24 / 255, 28 / 255, 1.0),
        }

    def _distinct_palette(n: int) -> List[Tuple[float, float, float, float]]:
        import colorsys

        palette: List[Tuple[float, float, float, float]] = []
        golden_ratio = 0.618033988749895
        hue = 0.0
        for _ in range(n):
            hue = (hue + golden_ratio) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
            palette.append((r, g, b, 1.0))
        return palette

    palette = _distinct_palette(len(class_names))

    special_colors = {
        "sunflower": (0.97, 0.78, 0.19, 1.0),
        "no_temporary_crop": (0.25, 0.25, 0.25, 1.0),
        "wheat": (0.99, 0.80, 0.20, 1.0),
        "grass_fodder_crops": (0.17, 0.55, 0.30, 1.0),
        "sorghum": (0.76, 0.20, 0.12, 1.0),
        "herb_spice_medicinal": (0.55, 0.20, 0.74, 1.0),
        "maize": (0.96, 0.62, 0.16, 1.0),
        "fibre_crops": (0.10, 0.47, 0.85, 1.0),
    }

    colors: Dict[int, Tuple[float, float, float, float]] = {}
    for idx, name in enumerate(class_names):
        key = str(name).strip().lower()
        if key in special_colors:
            colors[idx] = special_colors[key]
        else:
            colors[idx] = palette[idx % len(palette)]
    return colors


def _load_class_mappings_from_sharepoint() -> Dict[str, Dict[str, str]]:
    try:
        from worldcereal.utils.sharepoint import (  # noqa: WPS433
            build_class_mappings, get_excel_from_sharepoint)

        legend = get_excel_from_sharepoint(
            site_url="https://vitoresearch.sharepoint.com/sites/21717-ccn-world-cereal",
            file_server_relative_url="Research and Development/Legend/WorldCereal_LC_CT_legend_v2_class_mappings.xlsx",
            sheet_name=0,
        )
        legend["ewoc_code"] = legend["ewoc_code"].str.replace("-", "").astype(int)
        return build_class_mappings(legend)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to load class mappings from SharePoint: {exc}")
        return {}


@lru_cache(maxsize=8)
def _load_mapping_for_key(mapping_key: str) -> Dict[int, str]:
    """Load a single mapping dict for the requested key from known sources."""

    sources: List[Tuple[str, Path]] = [
        ("repo", CLASS_MAPPINGS_PATH),
        ("root", REPO_ROOT.parent / "class_mappings.json"),
    ]

    for label, path in sources:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to read class mappings from {path}: {exc}")
            continue
        mapping = payload.get(mapping_key)
        if not mapping:
            continue
        logger.info(f"Loaded class mapping for {mapping_key} from {label} path: {path}")
        coerced: Dict[int, str] = {}
        for code_str, label_str in mapping.items():
            if label_str is None:
                continue
            try:
                code_int = int(code_str)
            except (TypeError, ValueError):
                continue
            coerced[code_int] = str(label_str)
        return coerced

    sharepoint = _load_class_mappings_from_sharepoint()
    if mapping_key in sharepoint:
        logger.info(f"Loaded class mapping for {mapping_key} from SharePoint.")
        coerced = {}
        for code_str, label_str in sharepoint[mapping_key].items():
            if label_str is None:
                continue
            try:
                code_int = int(code_str)
            except (TypeError, ValueError):
                continue
            coerced[code_int] = str(label_str)
        return coerced

    raise ValueError(
        f"Class mapping for {mapping_key} not found in local paths or SharePoint."
    )

def _map_gt_to_indices(
    gt: np.ndarray,
    class_names: Sequence[str],
    mapping_key: str,
) -> np.ndarray:
    if gt.size == 0:
        return gt.astype(np.int32, copy=False)
    if not class_names:
        logger.warning(
            f"GT mapping skipped for {mapping_key}: empty class list from model metadata."
        )
        return np.full(gt.shape, -1, dtype=np.int32)

    try:
        max_val = np.nanmax(gt)
        min_val = np.nanmin(gt)
        if min_val >= 0 and max_val < len(class_names):
            return gt.astype(np.int32, copy=False)
    except Exception:  # noqa: BLE001
        pass

    mapping = _load_mapping_for_key(mapping_key)
    if not mapping:
        logger.warning(
            f"GT mapping skipped for {mapping_key}: no mapping entries found."
        )
        return np.full(gt.shape, -1, dtype=np.int32)

    class_index = {str(name): idx for idx, name in enumerate(class_names)}
    code_to_idx: Dict[int, int] = {}
    missing_labels: List[str] = []
    for code, label in mapping.items():
        idx = class_index.get(str(label))
        if idx is None:
            missing_labels.append(str(label))
            continue
        code_to_idx[int(code)] = idx

    if not code_to_idx:
        sample = ", ".join(sorted(set(missing_labels))[:8])
        logger.warning(
            f"GT mapping for {mapping_key} has no labels present in model class list. "
            f"Missing labels sample: [{sample}]"
        )
        return np.full(gt.shape, -1, dtype=np.int32)

    out = np.full(gt.shape, -1, dtype=np.int32)
    valid = np.isfinite(gt)
    if not np.any(valid):
        return out

    gt_int = np.where(valid, gt, 0).astype(np.int64, copy=False)
    unique_codes, counts = np.unique(gt_int[valid], return_counts=True)
    mapped_count = 0
    unmapped: List[Tuple[int, int]] = []
    for code, count in zip(unique_codes, counts):
        idx = code_to_idx.get(int(code))
        if idx is None:
            unmapped.append((int(code), int(count)))
            continue
        out[gt_int == code] = idx
        mapped_count += int(count)

    total_valid = int(valid.sum())
    unmapped_count = total_valid - mapped_count
    if mapped_count == 0:
        top = sorted(unmapped, key=lambda item: item[1], reverse=True)[:6]
        logger.warning(
            f"GT mapping for {mapping_key} mapped 0/{total_valid} pixels. "
            f"Top unmapped codes: {top}"
        )
    elif unmapped_count > 0:
        top = sorted(unmapped, key=lambda item: item[1], reverse=True)[:6]
        logger.info(
            f"GT mapping for {mapping_key} mapped {mapped_count}/{total_valid} pixels; "
            f"{unmapped_count} unmapped. Top unmapped codes: {top}"
        )

    return out


def _log_gt_value_counts(ds: xr.Dataset, band_name: str, *, max_items: int = 10) -> None:
    if band_name not in ds:
        return
    try:
        arr = np.asarray(ds[band_name].values)
        if arr.size == 0:
            logger.warning(f"GT band {band_name}: empty array")
            return
        valid = np.isfinite(arr)
        valid_count = int(valid.sum())
        if valid_count == 0:
            logger.warning(f"GT band {band_name}: no finite values")
            return
        vals, counts = np.unique(arr[valid], return_counts=True)
        order = np.argsort(counts)[::-1]
        vals = vals[order]
        counts = counts[order]
        top = [
            f"{vals[i]}:{counts[i]}" for i in range(min(max_items, len(vals)))
        ]
        logger.warning(
            f"GT band {band_name}: shape={arr.shape} finite={valid_count} "
            f"unique={len(vals)} top={top}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to summarize GT band {band_name}: {exc}")


def _load_model_metadata(model_dir: Path) -> Dict[str, Any]:
    run_config_path = model_dir / "run_config.json"
    manifest_path = model_dir / "head_manifest.json"

    run_config: Dict[str, Any] = {}
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {"run_config": run_config, "manifest": manifest}


def _resolve_model_zip(model_dir: Path, metadata: Mapping[str, Any]) -> Path:
    manifest = metadata.get("manifest", {})
    zip_name = (
        manifest.get("artifacts", {})
        .get("packages", {})
        .get("full")
    )
    if zip_name:
        zip_path = model_dir / str(zip_name)
        if zip_path.exists():
            return zip_path

    zips = sorted(model_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No model zip found in {model_dir}")
    return zips[0]


def _infer_model_signature(
    metadata: Mapping[str, Any],
    model_zip: Path,
    model_dir: Optional[Path] = None,
) -> str:
    manifest = metadata.get("manifest", {}) if isinstance(metadata, Mapping) else {}
    run_config = metadata.get("run_config", {}) if isinstance(metadata, Mapping) else {}
    signature = run_config.get("experiment_name")

    return str(signature)


def _resolve_season_ids_from_metadata(
    metadata: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Read ``season_ids`` from the model's run_config; fall back to *DEFAULT_SEASON_IDS*."""
    run_config = metadata.get("run_config", {})
    if isinstance(run_config, dict):
        dataset_cfg = run_config.get("dataset", {})
        if isinstance(dataset_cfg, dict):
            sids = dataset_cfg.get("season_ids")
            if sids and isinstance(sids, (list, tuple)):
                return tuple(str(s) for s in sids)
    return DEFAULT_SEASON_IDS


def _build_head_specs(
    metadata: Mapping[str, Any],
    season_ids: Optional[Sequence[str]] = None,
) -> Dict[str, HeadSpec]:
    run_config = metadata.get("run_config", {})
    classes_cfg = run_config.get("classes", {}) if isinstance(run_config, dict) else {}
    manifest = metadata.get("manifest", {}) if isinstance(metadata, Mapping) else {}

    croptype_classes = classes_cfg.get("croptype", {}).get("labels")
    landcover_classes = classes_cfg.get("landcover", {}).get("labels")

    if not croptype_classes or not landcover_classes:
        heads = manifest.get("heads", []) if isinstance(manifest, dict) else []
        for head in heads:
            class_names = head.get("class_names") if isinstance(head, dict) else None
            if not class_names:
                continue
            task = str(head.get("task") or head.get("name") or "").lower()
            if not croptype_classes and "crop" in task:
                croptype_classes = class_names
            if not landcover_classes and "landcover" in task:
                landcover_classes = class_names

    if not croptype_classes:
        raise ValueError(
            "No croptype class names found in run_config.json or head_manifest.json."
        )

    croptype_classes = [str(x) for x in croptype_classes]
    landcover_classes = [str(x) for x in landcover_classes] if landcover_classes else []

    # Local inference emits a binary cropland head; enforce canonical labels.
    lc_lower = [name.strip().lower() for name in landcover_classes]
    if lc_lower != ["no_cropland", "cropland"]:
        logger.warning(
            "Overriding landcover class labels %s -> ['no_cropland', 'cropland'] "
            "because local inference emits a binary cropland head.",
            landcover_classes,
        )
        landcover_classes = ["no_cropland", "cropland"]

    if season_ids is None:
        season_ids = _resolve_season_ids_from_metadata(metadata)

    def _summarize(labels: Sequence[str]) -> str:
        preview = ", ".join(labels[:5])
        suffix = "..." if len(labels) > 5 else ""
        return f"{len(labels)} labels [{preview}{suffix}]"

    logger.info(
        "Building head specs with croptype=%s, landcover=%s, seasons=%s.",
        _summarize(croptype_classes),
        _summarize(landcover_classes),
        list(season_ids),
    )

    specs: Dict[str, HeadSpec] = {
        "LANDCOVER": HeadSpec(
            key="LANDCOVER",
            display_name="LANDCOVER",
            class_names=[str(x) for x in landcover_classes],
            class_colors=_build_color_map(landcover_classes),
        ),
    }
    for sid in season_ids:
        head_key = _season_id_to_head_key(sid)
        display = f"CROPTYPE ({sid})"
        specs[head_key] = HeadSpec(
            key=head_key,
            display_name=display,
            class_names=[str(x) for x in croptype_classes],
            class_colors=_build_color_map(croptype_classes),
        )
    return specs


def _season_for_row(row_key: str) -> str:
    """Map a head/row key to its season ID."""
    sid = _head_key_to_season_id(row_key)
    if sid is not None:
        return sid
    # Legacy fallback for unknown keys
    return DEFAULT_SEASON_IDS[0] if row_key == "CROPTYPE_S1" else DEFAULT_SEASON_IDS[1]


def _extract_head_outputs(
    result_ds: xr.Dataset,
    head_key: str,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if head_key == "LANDCOVER":
        pred = result_ds["cropland_classification"].values.astype(np.int32)
        p_crop = result_ds.get("probability_cropland")
        p_other = result_ds.get("probability_other")
        if p_crop is None or p_other is None:
            raise KeyError("Expected cropland probability bands in inference output.")
        cube = np.stack(
            [p_other.values.astype(np.float32), p_crop.values.astype(np.float32)],
            axis=-1,
        )
        if np.nanmax(cube) > 1.0:
            cube = cube / 255.0
    else:
        season_id = _season_for_row(head_key)
        pred_name = f"croptype_classification:{season_id}"
        if pred_name not in result_ds:
            raise KeyError(f"Missing expected band '{pred_name}' in inference output.")
        pred = result_ds[pred_name].values.astype(np.int32)

        class_probs: List[np.ndarray] = []
        for class_name in class_names:
            class_name_str = str(class_name)
            band_name = f"croptype_probability:{season_id}:{class_name_str}"
            if band_name not in result_ds:
                break
            class_probs.append(result_ds[band_name].values.astype(np.float32))

        if len(class_probs) == len(class_names):
            cube = np.stack(class_probs, axis=-1)
            if np.nanmax(cube) > 1.0:
                cube = cube / 255.0
        else:
            # Fallback if per-class probabilities are unavailable.
            prob_name = f"croptype_probability:{season_id}"
            if prob_name not in result_ds:
                raise KeyError(
                    f"Missing expected probability output '{prob_name}' for {head_key}."
                )
            win = result_ds[prob_name].values.astype(np.float32)
            if np.nanmax(win) > 1.0:
                win = win / 255.0
            second = np.zeros_like(win, dtype=np.float32)
            return pred, np.clip(win, 0.0, 1.0), second

    win = np.max(cube, axis=-1)
    if cube.shape[-1] > 1:
        second = np.partition(cube, -2, axis=-1)[..., -2]
    else:
        second = np.zeros_like(win)
    return pred, np.clip(win, 0.0, 1.0), np.clip(second, 0.0, 1.0)


def _plot_categorical(
    ax,
    data: np.ndarray,
    head: HeadSpec,
    per_class_f1: Optional[np.ndarray] = None,
    *,
    force_legend: bool = False,
    empty_label: Optional[str] = None,
    fit_legend: bool = False,
) -> None:
    n_classes = len(head.class_names)
    masked = np.ma.masked_where((data < 0) | (data >= n_classes), data)
    cmap = ListedColormap([head.class_colors[i] for i in range(n_classes)])
    norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5, 1.0), cmap.N)
    ax.imshow(masked, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    present_classes = (
        sorted({int(v) for v in np.unique(masked.compressed())})
        if masked.count() > 0
        else []
    )
    legend_handles: List[Patch] = []
    for idx in present_classes:
        label = head.class_names[idx]
        if per_class_f1 is not None and idx < len(per_class_f1):
            label = f"{label} ({per_class_f1[idx]:.2f})"
        legend_handles.append(Patch(facecolor=head.class_colors[idx], edgecolor="black", label=label))

    if not legend_handles and empty_label:
        legend_handles.append(Patch(facecolor="none", edgecolor="none", label=empty_label))

    if legend_handles or force_legend:
        base_ncol = min(4, max(1, len(legend_handles) // 6 + 1))
        legend_kwargs = dict(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            frameon=True,
            framealpha=0.85,
            facecolor="white",
            edgecolor="none",
            columnspacing=0.7 if fit_legend else 0.8,
            handlelength=0.9 if fit_legend else 1.0,
            borderpad=0.3 if fit_legend else 0.4,
            # title="Classes",
            title_fontsize=LEGEND_FONTSIZE,
        )

        def _draw_legend(fontsize: int, ncol: int):
            return ax.legend(
                fontsize=fontsize,
                ncol=ncol,
                **legend_kwargs,
            )

        if not fit_legend:
            _draw_legend(LEGEND_FONTSIZE, base_ncol)
            return

        fig = ax.figure
        try:
            fig.canvas.draw()
        except Exception:  # noqa: BLE001
            _draw_legend(LEGEND_FONTSIZE, base_ncol)
            return

        ax_box = ax.get_window_extent()
        min_fontsize = max(10, int(LEGEND_FONTSIZE * 0.6))

        for ncol in range(base_ncol, 0, -1):
            for fontsize in range(LEGEND_FONTSIZE, min_fontsize - 1, -2):
                legend = _draw_legend(fontsize, ncol)
                try:
                    fig.canvas.draw()
                    legend_box = legend.get_window_extent()
                except Exception:  # noqa: BLE001
                    return
                if (
                    legend_box.width <= ax_box.width
                    and legend_box.height <= ax_box.height
                ):
                    return
                legend.remove()

        _draw_legend(min_fontsize, 1)


def _infer_country(ds: xr.Dataset, patch_name: str) -> Optional[str]:
    candidates = [
        "country",
        "COUNTRY",
        "country_name",
        "COUNTRY_NAME",
        "admin0",
        "ADMIN0",
        "adm0_name",
        "ADM0_NAME",
        "iso3",
        "ISO3",
        "iso2",
        "ISO2",
    ]
    for key in candidates:
        value = ds.attrs.get(key)
        if value:
            return str(value)
    for key in candidates:
        if key in ds:
            try:
                values = np.asarray(ds[key].values).ravel()
                if values.size == 1:
                    return str(values[0])
            except Exception:  # noqa: BLE001
                continue

    inferred = _infer_country_from_centroid(ds)
    if inferred:
        return inferred

    tokens = patch_name.replace("-", "_").split("_")
    for token in tokens:
        if len(token) in {2, 3} and token.isalpha():
            return token.upper()
    return None


@lru_cache(maxsize=1)
def _load_world_bounds():
    try:
        import geopandas as gpd
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Geopandas unavailable for country lookup: {exc}")
        return None

    if not WORLD_BOUNDS_PATH.exists():
        logger.warning(f"World boundaries file not found: {WORLD_BOUNDS_PATH}")
        return None

    gdf = gpd.read_parquet(WORLD_BOUNDS_PATH)
    keep_cols = [col for col in ("iso3", "name", "region", "geometry") if col in gdf.columns]
    gdf = gdf[keep_cols]
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _infer_country_from_centroid(ds: xr.Dataset) -> Optional[str]:
    world_df = _load_world_bounds()
    if world_df is None:
        return None

    try:
        extent = _get_patch_extent(ds)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to derive patch extent for country lookup: {exc}")
        return None

    center_x = 0.5 * (extent.west + extent.east)
    center_y = 0.5 * (extent.south + extent.north)
    lon, lat = center_x, center_y
    if extent.epsg != 4326:
        try:
            transformer = Transformer.from_crs(extent.epsg, 4326, always_xy=True)
            lon, lat = transformer.transform(center_x, center_y)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to project patch centroid to EPSG:4326: {exc}")
            return None

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        point_geom = Point(lon, lat)
        point = gpd.GeoDataFrame(geometry=[point_geom], crs="EPSG:4326")
        try:
            joined = gpd.sjoin(point, world_df, how="left", predicate="within")
        except Exception:  # noqa: BLE001
            joined = gpd.GeoDataFrame()
        if joined.empty or joined.iloc[0].isna().all():
            try:
                joined = gpd.sjoin_nearest(
                    point.to_crs("EPSG:3857"),
                    world_df.to_crs("EPSG:3857"),
                    how="left",
                )
            except Exception:  # noqa: BLE001
                joined = gpd.GeoDataFrame()
        if not joined.empty:
            name = joined.iloc[0].get("name") or joined.iloc[0].get("iso3")
            return str(name) if name else None

        # Fallback without spatial index: iterate geometries.
        for _, row in world_df.iterrows():
            geom = row.get("geometry")
            if geom is None:
                continue
            if geom.intersects(point_geom):
                name = row.get("name") or row.get("iso3")
                return str(name) if name else None
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to infer country from centroid: {exc}")
        return None


def _infer_title(
    ds: xr.Dataset,
    patch_name: str,
    *,
    country: Optional[str] = None,
    crop_details: Optional[str] = None,
) -> str:
    year = ds.attrs.get("year") or ds.attrs.get("YEAR")

    # Fallback from filename patterns
    fallback_tokens = patch_name.replace("-", "_").split("_")
    if year is None:
        for token in fallback_tokens:
            if token.isdigit() and len(token) == 4:
                year = token
                break

    year_str = str(year) if year is not None else "unknown-year"
    country_str = country or _infer_country(ds, patch_name) or "unknown-country"
    crop_str = crop_details or "full"
    return f"{country_str} | {year_str} | {patch_name} | {crop_str}"


def _format_season_months(
    season_windows: Optional[Mapping[str, Tuple[str, str]]],
    season_id: str,
) -> Optional[str]:
    if not season_windows or season_id not in season_windows:
        return None
    try:
        start_raw, end_raw = season_windows[season_id]
        start = pd.to_datetime(start_raw)
        end = pd.to_datetime(end_raw)
        start_label = start.strftime("%b")
        end_label = end.strftime("%b")
        return start_label if start_label == end_label else f"{start_label} - {end_label}"
    except Exception:  # noqa: BLE001
        return None


def _render_patch_figure(
    *,
    ds: xr.Dataset,
    patch_name: str,
    output_png: Path,
    outputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    head_specs: Dict[str, HeadSpec],
    enabled_rows: Sequence[str],
    country: Optional[str] = None,
    crop_details: Optional[str] = None,
    model_signature: Optional[str] = None,
    season_windows: Optional[Mapping[str, Tuple[str, str]]] = None,
    ct_mapping_key: str = "CROPTYPE25",
) -> None:
    rgb, ndvi = _compute_rgb_ndvi(ds)

    # Build row layout dynamically from head_specs (LANDCOVER first, then croptype heads).
    row_layout: List[str] = []
    if "LANDCOVER" in head_specs:
        row_layout.append("LANDCOVER")
    for key in head_specs:
        if key.startswith("CROPTYPE"):
            row_layout.append(key)

    n_rows = 1 + len(row_layout)  # 1 overview row + head rows
    row_height = 42 / 4  # keep per-row height identical to original 4-row layout
    fig_height = row_height * n_rows

    fig, axes = plt.subplots(n_rows, 3, figsize=(30, fig_height), constrained_layout=False)
    # Ensure axes is 2-D even when n_rows == 1.
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title_line = _infer_title(
        ds,
        patch_name,
        country=country,
        crop_details=crop_details,
    )
    fig.suptitle(
        title_line,
        fontsize=MAIN_TITLE_FONTSIZE,
        y=SUPTITLE_Y,
    )
    if model_signature:
        fig.text(
            0.5,
            SUPTITLE_Y - MODEL_TITLE_Y_OFFSET,
            f"Model: {model_signature}",
            ha="center",
            va="top",
            fontsize=MODEL_TITLE_FONTSIZE,
        )
    fig.subplots_adjust(
        top=SUBPLOT_TOP,
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM,
        hspace=SUBPLOT_HSPACE,
        wspace=SUBPLOT_WSPACE,
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(
        "Mean RGB", fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD
    )
    axes[0, 0].axis("off")

    ndvi_im = axes[0, 1].imshow(ndvi, cmap="RdYlGn", vmin=-0.8, vmax=0.8)
    axes[0, 1].set_title(
        "Mean NDVI", fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD
    )
    axes[0, 1].axis("off")
    fig.colorbar(ndvi_im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Use the first croptype head for the reference GT panel.
    first_ct_key = next((k for k in row_layout if k.startswith("CROPTYPE")), None)
    if first_ct_key is not None:
        gt_context = get_gt_for_row(ds, first_ct_key, ndvi.shape)
        if gt_context is not None:
            gt_context = _map_gt_to_indices(
                gt_context,
                head_specs[first_ct_key].class_names,
                mapping_key=ct_mapping_key,
            )
        context_head = head_specs[first_ct_key]
        gt_label = _season_for_row(first_ct_key)
    else:
        gt_context = None
        context_head = next(iter(head_specs.values()))
        gt_label = None

    if gt_context is None:
        empty = np.full(ndvi.shape, -1, dtype=np.int32)
        _plot_categorical(
            axes[0, 2],
            empty,
            context_head,
            force_legend=True,
            empty_label="No GT",
            fit_legend=True,
        )
        axes[0, 2].set_title(
            "Reference", fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD
        )
    else:
        _plot_categorical(
            axes[0, 2],
            gt_context.astype(np.int32),
            context_head,
            fit_legend=True,
        )
        ref_suffix = f" ({gt_label} GT)" if gt_label else ""
        axes[0, 2].set_title(
            f"Reference{ref_suffix}", fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD
        )

    cropland_mask = None
    if "LANDCOVER" in enabled_rows and "LANDCOVER" in outputs:
        cropland_pred = outputs["LANDCOVER"][0]
        if cropland_pred is not None and cropland_pred.size > 0:
            cropland_mask = (cropland_pred != 1) | (cropland_pred == NO_CROP_VALUE)

    for row_idx, row_key in enumerate(row_layout, start=1):
        pred_ax = axes[row_idx, 0]
        win_ax = axes[row_idx, 1]
        second_ax = axes[row_idx, 2]

        if row_key not in enabled_rows:
            pred_ax.text(0.5, 0.5, "Disabled", ha="center", va="center")
            pred_ax.axis("off")
            win_ax.axis("off")
            second_ax.axis("off")
            continue

        pred, win, second = outputs[row_key]
        head = head_specs[row_key]

        # Fetch GT for this row via the config-driven lookup.
        gt_arr = get_gt_for_row(ds, row_key, pred.shape)
        if gt_arr is not None and row_key.startswith("CROPTYPE"):
            gt_arr = _map_gt_to_indices(
                gt_arr,
                head.class_names,
                mapping_key=ct_mapping_key,
            )

        per_class_f1: Optional[np.ndarray] = None
        acc: Optional[float] = None
        macro: Optional[float] = None
        if gt_arr is not None:
            fill_values: List[float] = []
            gt_cfg = _HEAD_GT_CONFIG.get(row_key, {})
            gt_var = gt_cfg.get("gt_var")
            if gt_var and gt_var in ds:
                fill_values = _extract_fill_values(ds[gt_var])
            acc, macro, per_class_f1 = _compute_metrics(
                pred,
                gt_arr,
                num_classes=len(head.class_names),
                fill_values=fill_values,
            )

        fit_legend = row_key.startswith("CROPTYPE")
        _plot_categorical(
            pred_ax,
            pred,
            head,
            per_class_f1=per_class_f1,
            fit_legend=fit_legend,
        )

        if row_key.startswith("LANDCOVER"):
            row_title = "LC output"
        else:
            season_id = _season_for_row(row_key)
            season_months = _format_season_months(season_windows, season_id)
            season_suffix = f" ({season_months})" if season_months else ""
            row_title = f"CT {head.display_name}{season_suffix}"

        if acc is not None and macro is not None:
            row_title += f" — Acc={acc:.3f} MacroF1={macro:.3f}"
        title_loc = "left" if row_key.startswith("CROPTYPE") else "center"
        pred_ax.set_title(
            row_title,
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=PANEL_TITLE_PAD,
            loc=title_loc,
        )

        win_data = win
        second_data = second
        if row_key.startswith("CROPTYPE") and cropland_mask is not None:
            win_data = np.ma.masked_where(cropland_mask, win_data)
            second_data = np.ma.masked_where(cropland_mask, second_data)

        prob_min = 0.05
        prob_max = 0.7
        win_im = win_ax.imshow(
            win_data, cmap="RdYlGn", vmin=prob_min, vmax=prob_max
        )
        if row_key.startswith("LANDCOVER"):
            win_ax.set_title(
                "Winning probability", fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD
            )
        else:
            win_ax.set_title("")
        win_ax.axis("off")
        fig.colorbar(win_im, ax=win_ax, fraction=0.046, pad=0.04)

        if row_key.startswith("LANDCOVER"):
            second_im = second_ax.imshow(
                second_data, cmap="RdYlGn", vmin=prob_min, vmax=prob_max
            )
            second_ax.set_title(
                "Second-best probability",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=PANEL_TITLE_PAD,
            )
            second_ax.axis("off")
            fig.colorbar(second_im, ax=second_ax, fraction=0.046, pad=0.04)
        else:
            if gt_arr is None:
                empty = np.full(pred.shape, -1, dtype=np.int32)
                _plot_categorical(
                    second_ax,
                    empty,
                    head,
                    force_legend=True,
                    empty_label="No GT",
                    fit_legend=fit_legend,
                )
            else:
                _plot_categorical(
                    second_ax,
                    gt_arr.astype(np.int32),
                    head,
                    fit_legend=fit_legend,
                )
            second_ax.set_title(
                "GT",
                fontsize=PANEL_TITLE_FONTSIZE,
                pad=PANEL_TITLE_PAD,
            )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _normalize_heads(
    heads: Optional[Sequence[str]],
    valid_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    """Parse ``--heads`` CLI values into valid head keys.

    *valid_keys* should come from ``head_specs.keys()`` so it adapts to
    the model's actual season configuration.
    """
    default = list(valid_keys) if valid_keys else ["LANDCOVER", "CROPTYPE_S1", "CROPTYPE_S2"]
    if not heads:
        return default
    parsed: List[str] = []
    for item in heads:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                parsed.append(token.upper())
    allowed = set(default)
    selected = [token for token in parsed if token in allowed]
    return selected if selected else default


def _resolve_device(device: Optional[str]) -> str:
    if device and str(device).strip():
        return str(device).strip()

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass

    return "cpu"


def run_spatial_inference(
    model_dir: Union[Path, str],
    patches_dir: Union[Path, str],
    continents: Union[List[str], str] = "all",
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    heads: Optional[List[str]] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True,
    debug: bool = False,
    debug_seed: Optional[int] = None,
    ct_mapping_key: str = "CROPTYPE25",
) -> Path:
    """Run local spatial inference over patch files and write 4x3 panel PNGs.

    Returns
    -------
    Path
        Output directory containing generated per-patch PNGs.
    """

    model_dir = Path(model_dir)
    patches_dir = Path(patches_dir)
    if output_dir is None:
        output_dir = model_dir / "inference_patches"

    metadata = _load_model_metadata(model_dir)
    model_zip = _resolve_model_zip(model_dir, metadata)
    model_signature = _infer_model_signature(metadata, model_zip, model_dir=model_dir)
    season_ids = _resolve_season_ids_from_metadata(metadata)
    head_specs = _build_head_specs(metadata, season_ids=season_ids)
    selected_heads = _normalize_heads(heads, valid_keys=list(head_specs.keys()))

    # Derive the season IDs that are actually enabled for this run.
    model_season_ids: List[str] = []
    for hk in selected_heads:
        sid = _head_key_to_season_id(hk)
        if sid is not None:
            model_season_ids.append(sid)
    if not model_season_ids:
        # Fallback: use whatever the model was trained with.
        model_season_ids = list(season_ids)

    logger.info(
        "Resolved model seasons: %s  (head keys: %s)",
        model_season_ids,
        [_season_id_to_head_key(s) for s in model_season_ids],
    )

    patch_items = list_patches(patches_dir, continents)
    if limit is not None and limit > 0:
        patch_items = patch_items[:limit]
    if not patch_items:
        logger.warning("No patches found for inference.")
        return output_dir

    local_inference = _load_local_inference_module()
    run_local = local_inference.run_seasonal_inference

    device_name = _resolve_device(device)
    logger.info(
        f"Spatial inference starting with model={model_zip.name}, patches={len(patch_items)}, "
        f"device={device_name}, debug_crop={'on' if debug else 'off'}"
    )
    if debug:
        seed_msg = f" (seed={debug_seed})" if debug_seed is not None else ""
        logger.info(
            "Debug mode enabled: running inference on a random "
            f"{DEBUG_CROP_SIZE}x{DEBUG_CROP_SIZE} crop per patch{seed_msg}."
        )
    rng = np.random.default_rng(debug_seed) if debug else None

    iterator = (
        tqdm(patch_items, desc="Spatial inference", unit="patch")
        if show_progress
        else patch_items
    )

    processed = 0
    skipped = 0

    for continent, nc_path in iterator:
        out_png = output_dir / continent / f"{nc_path.stem}.png"
        if not debug and out_png.exists() and not overwrite:
            logger.info(f"Skipping existing output: {out_png}")
            skipped += 1
            if show_progress and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(f"processed={processed} skipped={skipped}")
            continue

        logger.info(f"Running patch: {nc_path}")
        ds = xr.open_dataset(nc_path, mask_and_scale=True)
        ds, debug_crop = _apply_debug_crop(ds, enabled=debug, rng=rng, label=nc_path.stem)
        # Temporary GT diagnostics (after cropping, before any transformations)
        _log_gt_value_counts(ds, "WORLDCEREAL_SEASON1_GT")
        _log_gt_value_counts(ds, "WORLDCEREAL_SEASON2_GT")
        if debug_crop is not None:
            out_png = output_dir / continent / f"{nc_path.stem}__{debug_crop.label}.png"
            if out_png.exists() and not overwrite:
                logger.info(f"Skipping existing output: {out_png}")
                skipped += 1
                ds.close()
                if show_progress and hasattr(iterator, "set_postfix_str"):
                    iterator.set_postfix_str(f"processed={processed} skipped={skipped}")
                continue

        enable_cropland = "LANDCOVER" in selected_heads
        enable_croptype = any(
            _head_key_to_season_id(key) is not None for key in selected_heads
        )
        season_ids_for_run = model_season_ids if enable_croptype else []

        season_windows = (
            _derive_season_windows_for_patch(ds, season_ids_for_run)
            if enable_croptype
            else None
        )
        result_ds = run_local(
            ds,
            seasonal_model_zip=model_zip,
            season_ids=season_ids_for_run,
            season_windows=season_windows,
            export_class_probabilities=True,
            enable_cropland_head=enable_cropland,
            enable_croptype_head=enable_croptype,
            as_dataset=True,
            device=device_name,
        )
    
        if isinstance(result_ds, xr.DataArray):
            raise TypeError("Expected dataset output from run_seasonal_inference.")

        outputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for key, spec in head_specs.items():
            try:
                pred, win, second = _extract_head_outputs(
                    result_ds,
                    key,
                    class_names=spec.class_names,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to extract {key} outputs for {nc_path.name}: {exc}")
                pred = np.zeros((ds.dims.get("y", 1), ds.dims.get("x", 1)), dtype=np.int32)
                win = np.zeros_like(pred, dtype=np.float32)
                second = np.zeros_like(pred, dtype=np.float32)
            outputs[key] = (pred, win, second)

        _render_patch_figure(
            ds=ds,
            patch_name=nc_path.stem,
            output_png=out_png,
            outputs=outputs,
            head_specs=head_specs,
            enabled_rows=selected_heads,
            country=_infer_country(ds, nc_path.stem),
            crop_details=debug_crop.label if debug_crop is not None else None,
            model_signature=model_signature,
            season_windows=season_windows,
            ct_mapping_key=ct_mapping_key,
        )
        ds.close()
        processed += 1
        if show_progress and hasattr(iterator, "set_postfix_str"):
            iterator.set_postfix_str(f"processed={processed} skipped={skipped}")
        logger.info(f"Saved panel: {out_png}")

    if show_progress and hasattr(iterator, "close"):
        iterator.close()

    logger.info(f"Spatial inference complete. Outputs in {output_dir}")
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local spatial inference on patch .nc files and export panel PNGs."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--patches_dir", type=str, required=True)
    parser.add_argument(
        "--continents",
        action="append",
        default=["all"],
        help=(
            "Continent selection. Supports repeated flags or CSV values, e.g. "
            "--continents Africa --continents Europe or --continents Africa,Europe. "
            "Use 'all' to scan all first-level continent folders."
        ),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help=(
            "Comma-separated list of heads to render. Valid keys depend on "
            "the model's season configuration: LANDCOVER, CROPTYPE_S1, "
            "CROPTYPE_S2, CROPTYPE_ANNUAL. Defaults to all heads the model "
            "was trained with."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode by running inference on a random 256x256 crop.",
    )
    parser.add_argument(
        "--debug_seed",
        type=int,
        default=None,
        help="Optional RNG seed to make debug crop selection deterministic.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--ct_mapping_key",
        type=str,
        default="CROPTYPE25",
        help="Key for the crop type mapping to use.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    heads = args.heads.split(",") if args.heads else None
    run_spatial_inference(
        model_dir=Path(args.model_dir),
        patches_dir=Path(args.patches_dir),
        continents=args.continents,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        device=args.device,
        heads=heads,
        limit=args.limit,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
        debug=args.debug,
        debug_seed=args.debug_seed,
        ct_mapping_key=args.ct_mapping_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
