#!/usr/bin/env python3
"""Export cropland-mapping process graphs without submitting them.

This is an advanced workflow for users who want to inspect, edit, or manually
submit the generated OpenEO process graphs on another backend. The script reuses
WorldCereal's job-manager workflow configuration and tile splitting, but stops
before backend submission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
from openeo_gfmap import Backend, BackendContext, TemporalContext
from shapely.geometry import box

from worldcereal.job import WorldCerealTask
from worldcereal.jobmanager import WorldCerealJobManager
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_WORKFLOW_PRESET
from worldcereal.openeo.workflow_config import build_config_from_params
from worldcereal.parameters import WorldCerealProductType

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--bbox must be a comma-separated list: minx,miny,maxx,maxy"
        )
    return tuple(parts)  # type: ignore[return-value]


def export_cropland_graph(args: argparse.Namespace) -> Path:
    minx, miny, maxx, maxy = _parse_bbox(args.bbox)
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [box(minx, miny, maxx, maxy)],
            "tile_name": ["export_tile_0"],
            "start_date": [args.start_date],
            "end_date": [args.end_date],
        },
        crs=f"EPSG:{args.bbox_epsg}",
    )

    backend_context = BackendContext(Backend[args.backend.upper()])
    workflow_config = build_config_from_params(
        seasonal_model_zip=args.seasonal_model_zip,
        landcover_head_zip=args.landcover_head_zip,
        enable_cropland_head=True,
        enable_croptype_head=False,
        mask_cropland=False,
        enable_cropland_postprocess=args.enable_cropland_postprocess,
        cropland_postprocess_method=args.cropland_postprocess_method,
        cropland_postprocess_kernel_size=args.cropland_postprocess_kernel_size,
        export_embeddings=args.export_embeddings,
    )

    manager = WorldCerealJobManager(
        output_dir=args.output_dir,
        task=WorldCerealTask.CLASSIFICATION,
        aoi_gdf=gdf,
        backend_context=backend_context,
        grid_size=args.grid_size,
        temporal_extent=TemporalContext(start_date=args.start_date, end_date=args.end_date),
    )

    export_dir = manager.export_graphs(
        product_type=WorldCerealProductType.CROPLAND,
        target_epsg=args.target_epsg,
        s1_orbit_state=args.s1_orbit_state,
        compositing_window=args.compositing_window,
        seasonal_preset=args.seasonal_preset,
        workflow_config=workflow_config,
        export_dir=Path(args.output_dir) / "graphs",
    )
    return export_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and serialize a WorldCereal cropland mapping process graph "
            "without submitting it to a backend."
        )
    )
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Bounding box as minx,miny,maxx,maxy.",
    )
    parser.add_argument(
        "--bbox_epsg",
        type=int,
        default=4326,
        help="EPSG code for the supplied bounding box. Default: 4326.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--seasonal_model_zip",
        type=str,
        required=True,
        help="Path or URL to the seasonal model artifact.",
    )
    parser.add_argument(
        "--landcover_head_zip",
        type=str,
        default=None,
        help="Optional landcover head artifact.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the exported JSON graph should be written.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="Tile size for splitting the AOI before graph export. Default: 20.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="CDSE",
        choices=[backend.name for backend in Backend],
        help="Backend label used for graph generation metadata. Default: CDSE.",
    )
    parser.add_argument(
        "--seasonal_preset",
        type=str,
        default=DEFAULT_SEASONAL_WORKFLOW_PRESET,
        help="Seasonal workflow preset to use.",
    )
    parser.add_argument(
        "--s1_orbit_state",
        type=str,
        choices=["ASCENDING", "DESCENDING"],
        default=None,
        help="Optional Sentinel-1 orbit direction.",
    )
    parser.add_argument(
        "--target_epsg",
        type=int,
        default=None,
        help="Optional EPSG for the target projection.",
    )
    parser.add_argument(
        "--compositing_window",
        type=str,
        choices=["month", "dekad"],
        default="month",
        help="Temporal compositing window. Default: month.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=128,
        help="Tile size for OpenEO data loading. Default: 128.",
    )
    parser.add_argument(
        "--s2_tile",
        type=str,
        default=None,
        help="Optional Sentinel-2 tile ID to constrain loading.",
    )
    parser.add_argument(
        "--enable_cropland_postprocess",
        action="store_true",
        help="Enable cropland post-processing in the model config.",
    )
    parser.add_argument(
        "--cropland_postprocess_method",
        type=str,
        default="majority_vote",
        help="Cropland post-process method. Default: majority_vote.",
    )
    parser.add_argument(
        "--cropland_postprocess_kernel_size",
        type=int,
        default=3,
        help="Kernel size for cropland post-process. Default: 3.",
    )
    parser.add_argument(
        "--export_embeddings",
        action="store_true",
        help="Include the optional embeddings output in the workflow config.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_path = export_cropland_graph(args)
    print(f"Exported cropland mapping process graph to: {out_path}")
    print("This is a saved process graph only; it was not submitted to the backend.")


if __name__ == "__main__":
    main()
