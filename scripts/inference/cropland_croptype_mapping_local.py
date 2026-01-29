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
        "--keep-class-probabilities",
        action="store_true",
        help="Also export per-class probability bands for landcover and croptype outputs",
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


def attach_crs_metadata(result: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    crs_var = template.get("crs")
    out = result.assign_coords(x=template.coords.get("x"), y=template.coords.get("y"))
    if crs_var is not None:
        crs_name = "spatial_ref"
        out[crs_name] = xr.DataArray(0, attrs=crs_var.attrs)
        for name in out.data_vars:
            out[name].attrs.setdefault("grid_mapping", crs_name)
    return out


def main() -> None:
    args = parse_args(
        [
            "--season-window",
            "tc-s1:2020-12-01:2021-07-31",
            "--season-window",
            "tc-s2:2021-04-01:2021-10-31",
            "--keep-class-probabilities",
        ]
    )

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

    engine = SeasonalInferenceEngine(
        seasonal_model_zip=args.seasonal_zip,
        landcover_head_zip=args.landcover_head_zip,
        croptype_head_zip=args.croptype_head_zip,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        season_ids=season_ids,
        keep_class_probabilities=args.keep_class_probabilities,
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
