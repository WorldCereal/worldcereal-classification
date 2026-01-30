#!/usr/bin/env python3
"""Local seasonal inference runner for cropland + croptype mapping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
import xarray as xr

from worldcereal.openeo.inference import SeasonalInferenceEngine
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_MODEL_URL

DEFAULT_INPUT_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/presto/localtestdata/"
    "local_presto_inputs.nc"
)
DEFAULT_INPUT_PATH = Path.cwd() / "presto_test_inputs.nc"


def parse_args(arg_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seasonal cropland/croptype inference locally"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to NetCDF input cube (defaults to bundled sample)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "seasonal_inference.nc",
        help="Output NetCDF path",
    )
    parser.add_argument(
        "--epsg", type=int, default=32631, help="EPSG code of the input cube"
    )
    parser.add_argument(
        "--seasonal-zip",
        type=str,
        default=DEFAULT_SEASONAL_MODEL_URL,
        help="Seasonal model zip (URL or local path)",
    )
    parser.add_argument(
        "--landcover-head-zip",
        type=str,
        default=None,
        help="Optional replacement zip for the landcover head",
    )
    parser.add_argument(
        "--croptype-head-zip",
        type=str,
        default=None,
        help="Optional replacement zip for the croptype head",
    )
    parser.add_argument(
        "--disable-gating",
        action="store_true",
        help="Disable cropland gating for croptype outputs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for inference"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory for downloaded artifacts",
    )
    parser.add_argument(
        "--season-id",
        action="append",
        help="Override season identifiers (default: tc-s1, tc-s2)",
    )
    parser.add_argument(
        "--season-window",
        action="append",
        metavar="ID:START:END",
        help="Define a manual season window using YYYY-MM-DD dates (can be passed multiple times)",
    )
    parser.add_argument(
        "--sample-url",
        type=str,
        default=DEFAULT_INPUT_URL,
        help="URL for the fallback sample cube",
    )
    parser.add_argument(
        "--export-class-probabilities",
        dest="export_class_probabilities",
        action="store_true",
        help="Also export per-class probability bands for landcover and croptype outputs",
    )
    parser.add_argument(
        "--cropland-postprocess",
        action="store_true",
        help="Enable cropland postprocessing (majority vote or smoothing)",
    )
    parser.add_argument(
        "--cropland-method",
        type=str,
        choices=("majority_vote", "smooth_probabilities"),
        help="Cropland postprocess method",
    )
    parser.add_argument(
        "--cropland-kernel-size",
        type=int,
        help="Kernel size for cropland postprocess",
    )
    parser.add_argument(
        "--croptype-postprocess",
        action="store_true",
        help="Enable croptype postprocessing (majority vote or smoothing)",
    )
    parser.add_argument(
        "--croptype-method",
        type=str,
        choices=("majority_vote", "smooth_probabilities"),
        help="Croptype postprocess method",
    )
    parser.add_argument(
        "--croptype-kernel-size",
        type=int,
        help="Kernel size for croptype postprocess",
    )
    return parser.parse_args(arg_list)


def parse_manual_windows(entries: Optional[list[str]]) -> Dict[str, Tuple[str, str]]:
    """Parse repeated --season-window arguments into a mapping."""

    windows: Dict[str, Tuple[str, str]] = {}
    if not entries:
        return windows
    for spec in entries:
        parts = [piece.strip() for piece in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid season window '{spec}'. Expected format ID:START:END"
            )
        season_id, start, end = parts
        if not season_id:
            raise ValueError("Season window must include a non-empty identifier")
        windows[season_id] = (start, end)
    return windows


def download_sample(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    print(f"Downloading test cube from {url} -> {target}")
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(target, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
    return target


def load_input_cube(
    path: Optional[Path], sample_url: str
) -> tuple[xr.DataArray, xr.Dataset]:
    dataset_path = path
    if dataset_path is None:
        dataset_path = download_sample(sample_url, DEFAULT_INPUT_PATH)
    ds = xr.open_dataset(dataset_path)
    arr = ds.drop_vars("crs", errors="ignore").astype("uint16").to_array(dim="bands")
    return arr, ds


def attach_crs_metadata(
    result: xr.Dataset | xr.DataArray, template: xr.Dataset
) -> xr.Dataset | xr.DataArray:
    crs_var = template.get("crs")
    coords = {}
    if "x" in template.coords:
        coords["x"] = template.coords["x"]
    if "y" in template.coords:
        coords["y"] = template.coords["y"]
    out = result.assign_coords(**coords) if coords else result

    if crs_var is None:
        return out

    crs_name = "spatial_ref"
    crs_data = xr.DataArray(0, attrs=crs_var.attrs)

    if isinstance(out, xr.Dataset):
        out[crs_name] = crs_data
        for name in out.data_vars:
            out[name].attrs.setdefault("grid_mapping", crs_name)
        return out

    out = out.assign_coords({crs_name: crs_data})
    out.attrs.setdefault("grid_mapping", crs_name)
    return out


def build_postprocess_spec(
    *,
    enabled: bool,
    method: Optional[str],
    kernel_size: Optional[int],
) -> Optional[Dict[str, object]]:
    requested = enabled or method is not None or kernel_size is not None
    if not requested:
        return None

    spec: Dict[str, object] = {
        "enabled": enabled or method is not None or kernel_size is not None
    }
    if method:
        spec["method"] = method
    if kernel_size is not None:
        spec["kernel_size"] = kernel_size
    return spec


def main() -> None:
    default_arg_list = [
        "--season-window",
        "tc-s1:2020-12-01:2021-07-31",
        "--season-window",
        "tc-s2:2021-04-01:2021-10-31",
        "--cropland-postprocess",
        "--croptype-postprocess",
        # "--export-class-probabilities",
    ]
    arg_list = None if len(sys.argv) > 1 else default_arg_list
    args = parse_args(arg_list)

    try:
        arr, template = load_input_cube(args.input, args.sample_url)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load input cube: {exc}", file=sys.stderr)
        sys.exit(1)

    season_ids = args.season_id if args.season_id else None
    try:
        season_windows = parse_manual_windows(args.season_window)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    cropland_postprocess = build_postprocess_spec(
        enabled=args.cropland_postprocess,
        method=args.cropland_method,
        kernel_size=args.cropland_kernel_size,
    )
    croptype_postprocess = build_postprocess_spec(
        enabled=args.croptype_postprocess,
        method=args.croptype_method,
        kernel_size=args.croptype_kernel_size,
    )

    engine = SeasonalInferenceEngine(
        seasonal_model_zip=args.seasonal_zip,
        landcover_head_zip=args.landcover_head_zip,
        croptype_head_zip=args.croptype_head_zip,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        season_ids=season_ids,
        export_class_probabilities=args.export_class_probabilities,
        cropland_postprocess=cropland_postprocess,
        croptype_postprocess=croptype_postprocess,
    )

    result = engine.infer(
        arr,
        epsg=args.epsg,
        enforce_cropland_gate=not args.disable_gating,
        season_windows=season_windows or None,
        season_ids=season_ids,
    )
    result = attach_crs_metadata(result, template)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(args.output)
    print(f"Seasonal inference saved to {args.output}")


if __name__ == "__main__":
    main()
