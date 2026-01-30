"""Cropland mapping inference script, demonstrating the use of the GFMAP, Presto and WorldCereal classifiers in a first inference pipeline."""

from loguru import logger
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import Backend, BackendContext

from worldcereal.job import WorldCerealProductType, generate_map
from worldcereal.openeo.workflow_config import (
    ModelSection,
    SeasonSection,
    WorldCerealWorkflowConfig,
)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     prog="WC - Crop Mapping Inference",
    #     description="Crop Mapping inference using GFMAP, Presto and WorldCereal classifiers",
    # )

    # parser.add_argument("minx", type=float, help="Minimum X coordinate (west)")
    # parser.add_argument("miny", type=float, help="Minimum Y coordinate (south)")
    # parser.add_argument("maxx", type=float, help="Maximum X coordinate (east)")
    # parser.add_argument("maxy", type=float, help="Maximum Y coordinate (north)")
    # parser.add_argument(
    #     "start_date", type=str, help="Starting date for data extraction."
    # )
    # parser.add_argument("end_date", type=str, help="Ending date for data extraction.")
    # parser.add_argument(
    #     "product",
    #     type=str,
    #     help="Product to generate. One of ['cropland', 'croptype']",
    # )
    # parser.add_argument(
    #     "output_path",
    #     type=Path,
    #     help="Path to folder where to save the resulting GeoTiff.",
    # )
    # parser.add_argument(
    #     "--epsg",
    #     type=int,
    #     default=4326,
    #     help="EPSG code of the input `minx`, `miny`, `maxx`, `maxy` parameters.",
    # )
    # parser.add_argument(
    #     "--class-probabilities",
    #     action="store_true",
    #     help="Output per-class probabilities in the resulting product",
    # )
    # args = parser.parse_args()

    # minx = args.minx
    # miny = args.miny
    # maxx = args.maxx
    # maxy = args.maxy
    # epsg = args.epsg

    # start_date = args.start_date
    # end_date = args.end_date

    # product = args.product
    # export_class_probabilities = args.class_probabilities
    # output_dir = args.output_path

    minx, miny, maxx, maxy = (664000, 5611134, 665000, 5612134)  # Small test
    # minx, miny, maxx, maxy = (664000, 5611134, 684000, 5631134)  # Large test
    # minx, miny, maxx, maxy = (634000, 5601134, 684000, 5651134)  # Very large test
    epsg = 32631
    start_date = "2020-11-01"
    end_date = "2021-10-31"
    product = "croptype"
    output_dir = "."
    export_class_probabilities = True
    enable_cropland_postprocess = True
    enable_croptype_postprocess = True

    season_windows = {
        "tc-s1": ("2020-12-01", "2021-07-31"),
        "tc-s2": ("2021-04-01", "2021-10-31"),
    }
    season_ids = list(season_windows.keys())

    spatial_extent = BoundingBoxExtent(minx, miny, maxx, maxy, epsg)
    temporal_extent = TemporalContext(start_date, end_date)

    backend_context = BackendContext(Backend.CDSE)

    # Replace with the URI of your custom croptype head archive when overriding the preset model.
    croptype_head_uri = None  # e.g. "abfs://models/custom_croptype_head.zip"

    postprocess_config = {}
    if enable_cropland_postprocess:
        postprocess_config["cropland"] = {"enabled": True}
    if enable_croptype_postprocess:
        postprocess_config["croptype"] = {"enabled": True}

    workflow_cfg = WorldCerealWorkflowConfig(
        model=(
            ModelSection(
                croptype_head_zip=croptype_head_uri,
                enable_croptype_head=True,
            )
            if croptype_head_uri
            else None
        ),
        season=SeasonSection(
            export_class_probabilities=export_class_probabilities,
            season_ids=season_ids,
            season_windows=season_windows,
        ),
        postprocess=postprocess_config or None,
    )

    job_results = generate_map(
        spatial_extent,
        temporal_extent,
        output_dir=output_dir,
        product_type=WorldCerealProductType(product),
        out_format="GTiff",
        backend_context=backend_context,
        workflow_config=workflow_cfg,
    )
    logger.success(f"Job finished:\n\t{job_results}")
