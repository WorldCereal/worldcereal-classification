"""
Script that generates the worldcereal_crop_type UDP from the source code.
"""

import json

import openeo
from openeo.api.process import Parameter
from openeo.rest.udp import build_process_dict
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import Backend, BackendContext, cdse_connection

from worldcereal.job import (
    DEFAULT_INFERENCE_JOB_OPTIONS,
    WorldCerealProductType,
    create_inference_process_graph,
)
from worldcereal.openeo.workflow_config import (
    ModelSection,
    SeasonSection,
    WorldCerealWorkflowConfig,
)




# -------------------------------------------------------------------------------
# PROCESS GRAPH CREATION
# -------------------------------------------------------------------------------


def create_process_graph_with_parameters() -> openeo.MultiResult:
    # Set bbox
    # minx, miny, maxx, maxy = (664000, 5611134, 665000, 5612134)  # Small test
    # minx, miny, maxx, maxy = (664000, 5611134, 684000, 5631134)  # Large test
    minx, miny, maxx, maxy = (634000, 5601134, 684000, 5651134)  # Very large test

    # EPSG in which bbox is defined.
    epsg = 32631

    # Manual spec of temporal extent and season windows for demonstration purposes.
    start_date = "2020-11-01"
    end_date = "2021-10-31"
    season_windows = {
        "tc-s1": ("2020-12-01", "2021-07-31"),
        "tc-s2": ("2021-04-01", "2021-10-31"),
    }
    season_ids = list(
        season_windows.keys()
    )  # CAN WE ADOPT UDF TO NOT NEED THIS AND DERIVE FROM SEASON WINDOWS?

    export_class_probabilities = True
    enable_cropland_postprocess = (
        True  # what is the cost because we postprocess anyway?
    )
    enable_croptype_postprocess = (
        True  # what is the cost because we postprocess anyway?
    )
    export_embeddings = False
    export_ndvi = False
    merge_classification_products = True  # Keeps all classification products together

    # -------------------------------------------------------------------------------

    # Global production always runs as CROPTYPE product
    product = WorldCerealProductType.CROPTYPE

    # Get a connection to the OpenEO backend
    connection = cdse_connection()

    spatial_extent = BoundingBoxExtent(minx, miny, maxx, maxy, epsg)
    temporal_extent = TemporalContext(start_date, end_date)

    backend_context = BackendContext(Backend.CDSE)

    postprocess_config = {}
    if enable_cropland_postprocess:
        postprocess_config["cropland"] = {
            "enabled": True,
            "method": "majority_vote",
            "kernel_size": 5,
        }
    if enable_croptype_postprocess:
        postprocess_config["croptype"] = {
            "enabled": True,
            "method": "majority_vote",
            "kernel_size": 5,
        }

    workflow_cfg = WorldCerealWorkflowConfig(
        model=ModelSection(
            export_embeddings=export_embeddings,
            export_ndvi=export_ndvi,
        ),
        season=SeasonSection(
            export_class_probabilities=export_class_probabilities,
            season_ids=season_ids,
            season_windows=season_windows,
            merge_classification_products=merge_classification_products,
        ),
        postprocess=postprocess_config or None,
    )

    # Create the process graph
    results = create_inference_process_graph(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        product_type=product,
        backend_context=backend_context,
        connection=connection,
        workflow_config=workflow_cfg,
    )

    results = openeo.MultiResult(results)

    return results



# -------------------------------------------------------------------------------
# CREATE AND WRITE THE UDP
# -------------------------------------------------------------------------------


def main():

    results = create_process_graph_with_parameters()

    default_job_options = DEFAULT_INFERENCE_JOB_OPTIONS.copy()

    default_job_options.update(
        {
            "driver-memory": "8g",
            "executor-memory": "3g",
            "executor-memoryOverhead": "3500m"
        }
    )

    job = results.create_job(
        title="WorldCereal inference benchmark - 2 CPU threads, 2 interop threads, less overheadmemory",
        job_options=default_job_options,
    )

    job.start_and_wait()





if __name__ == "__main__":
    main()