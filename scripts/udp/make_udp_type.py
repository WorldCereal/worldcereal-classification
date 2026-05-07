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
from worldcereal.openeo.parameters import (
    DEFAULT_POSTPROCESS_SECTION,
    DEFAULT_SEASONAL_MODEL_URL,
)
from worldcereal.openeo.workflow_config import (
    ModelSection,
    SeasonSection,
    WorldCerealWorkflowConfig,
)

default_job_options = DEFAULT_INFERENCE_JOB_OPTIONS.copy()

CROPLAND_DEFAULTS = DEFAULT_POSTPROCESS_SECTION["cropland"]
CROPTYPE_DEFAULTS = DEFAULT_POSTPROCESS_SECTION["croptype"]

# -------------------------------------------------------------------------------
# PROCESS GRAPH CREATION
# -------------------------------------------------------------------------------


def create_process_graph_with_parameters() -> openeo.MultiResult:
    # Set bbox
    minx, miny, maxx, maxy = (664000, 5611134, 665000, 5612134)  # Small test
    # minx, miny, maxx, maxy = (664000, 5611134, 684000, 5631134)  # Large test
    # minx, miny, maxx, maxy = (634000, 5601134, 684000, 5651134)  # Very large test

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
            "method": CROPLAND_DEFAULTS["method"],
            "kernel_size": CROPLAND_DEFAULTS["kernel_size"],
        }
    if enable_croptype_postprocess:
        postprocess_config["croptype"] = {
            "enabled": True,
            "method": CROPTYPE_DEFAULTS["method"],
            "kernel_size": CROPTYPE_DEFAULTS["kernel_size"],
        }

    workflow_cfg = WorldCerealWorkflowConfig(
        model=ModelSection(
            export_embeddings=export_embeddings,
            export_ndvi=export_ndvi,
            enable_cropland_head=True,
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
# FUNCTIONS TO PARAMETRIZE THE PROCESS GRAPH
# -------------------------------------------------------------------------------


def replace_temporal_extent(obj, target_temporal_extent=["2020-11-01", "2021-10-31"]):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "temporal_extent" and value == target_temporal_extent:
                obj[key] = {"from_parameter": "temporal_extent"}
            else:
                replace_temporal_extent(value, target_temporal_extent)
    elif isinstance(obj, list):
        for item in obj:
            replace_temporal_extent(item, target_temporal_extent)


def replace_spatial_extent(
    obj,
    target_spatial_extent={
        "west": 664000,
        "east": 665000,
        "north": 5612134,
        "south": 5611134,
        "crs": "EPSG:32631",
    },
):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "extent" and value == target_spatial_extent:
                obj[key] = {"from_parameter": "spatial_extent"}
            else:
                replace_spatial_extent(value, target_spatial_extent)
    elif isinstance(obj, list):
        for item in obj:
            replace_spatial_extent(item, target_spatial_extent)


def replace_orbit_state(obj, target_orbit_state="DESCENDING"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "y" and value == target_orbit_state:
                obj[key] = {"from_parameter": "orbit_state"}
            else:
                replace_orbit_state(value, target_orbit_state)
    elif isinstance(obj, list):
        for item in obj:
            replace_orbit_state(item, target_orbit_state)


def replace_model_url(
    obj,
    target_model_url="https://s3.waw3-1.cloudferro.com/project_dependencies/worldcereal/presto-prometheo-dualtask-SeasonalMultiTaskLoss-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202601240103.zip",
):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "seasonal_model_zip" and value == target_model_url:
                obj[key] = {"from_parameter": "model_url"}
            else:
                replace_model_url(value, target_model_url)
    elif isinstance(obj, list):
        for item in obj:
            replace_model_url(item, target_model_url)


def replace_season_ids(obj):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == "season_ids":
                obj[key] = {"from_parameter": "season_ids"}
            else:
                replace_season_ids(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            replace_season_ids(item)


def replace_season_windows(obj):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == "season_windows":
                obj[key] = {"from_parameter": "season_windows"}
            else:
                replace_season_windows(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            replace_season_windows(item)

def replace_postprocess_method_cropland(obj, target_postprocess_method=CROPLAND_DEFAULTS["method"]):
    if isinstance(obj, dict):
        if "postprocess" in obj:
            cropland = obj["postprocess"].get("cropland")
            if isinstance(cropland, dict) and cropland.get("method") == target_postprocess_method:
                cropland["method"] = {"from_parameter": "postprocess_method_cropland"}
        for value in obj.values():
            replace_postprocess_method_cropland(value, target_postprocess_method)
    elif isinstance(obj, list):
        for item in obj:
            replace_postprocess_method_cropland(item, target_postprocess_method)

def replace_kernel_size_cropland(obj, target_kernel_size=CROPLAND_DEFAULTS["kernel_size"]):
    if isinstance(obj, dict):
        if "postprocess" in obj:
            cropland = obj["postprocess"].get("cropland")
            if isinstance(cropland, dict) and cropland.get("kernel_size") == target_kernel_size:
                cropland["kernel_size"] = {"from_parameter": "postprocess_kernel_size_cropland"}
        for value in obj.values():
            replace_kernel_size_cropland(value, target_kernel_size)
    elif isinstance(obj, list):
        for item in obj:
            replace_kernel_size_cropland(item, target_kernel_size)


def replace_postprocess_method_croptype(obj, target_postprocess_method=CROPTYPE_DEFAULTS["method"]):
    if isinstance(obj, dict):
        if "postprocess" in obj:
            croptype = obj["postprocess"].get("croptype")
            if isinstance(croptype, dict) and croptype.get("method") == target_postprocess_method:
                croptype["method"] = {"from_parameter": "postprocess_method_croptype"}
        for value in obj.values():
            replace_postprocess_method_croptype(value, target_postprocess_method)
    elif isinstance(obj, list):
        for item in obj:
            replace_postprocess_method_croptype(item, target_postprocess_method)

def replace_kernel_size_croptype(obj, target_kernel_size=CROPTYPE_DEFAULTS["kernel_size"]):
    if isinstance(obj, dict):
        if "postprocess" in obj:
            croptype = obj["postprocess"].get("croptype")
            if isinstance(croptype, dict) and croptype.get("kernel_size") == target_kernel_size:
                croptype["kernel_size"] = {"from_parameter": "postprocess_kernel_size_croptype"}
        for value in obj.values():
            replace_kernel_size_croptype(value, target_kernel_size)
    elif isinstance(obj, list):
        for item in obj:
            replace_kernel_size_croptype(item, target_kernel_size)


def replace_seasonal_head_zip(
    obj,
    target_seasonal_head_zip=DEFAULT_SEASONAL_MODEL_URL,
):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "seasonal_model_zip" and value == target_seasonal_head_zip:
                obj[key] = {"from_parameter": "seasonal_head_zip"}
            else:
                replace_seasonal_head_zip(value, target_seasonal_head_zip)
    elif isinstance(obj, list):
        for item in obj:
            replace_seasonal_head_zip(item, target_seasonal_head_zip)

def replace_landcover_head_zip(
    obj,
    target_landcover_head_zip=None,
):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "landcover_head_zip" and value == target_landcover_head_zip:
                obj[key] = {"from_parameter": "landcover_head_zip"}
            else:
                replace_landcover_head_zip(value, target_landcover_head_zip)
    elif isinstance(obj, list):
        for item in obj:
            replace_landcover_head_zip(item, target_landcover_head_zip)

def replace_mask_cropland(
    obj,
    target_mask_cropland=True,
):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "mask_cropland" and value == target_mask_cropland:
                obj[key] = {"from_parameter": "mask_cropland"}
            else:
                replace_mask_cropland(value, target_mask_cropland)
    elif isinstance(obj, list):
        for item in obj:
            replace_mask_cropland(item, target_mask_cropland)

def replace_enable_cropland_head(
    obj,
    target_enable_cropland_head=True,
):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "enable_cropland_head" and value == target_enable_cropland_head:
                obj[key] = {"from_parameter": "enable_cropland_head"}
            else:
                replace_enable_cropland_head(value, target_enable_cropland_head)
    elif isinstance(obj, list):
        for item in obj:
            replace_enable_cropland_head(item, target_enable_cropland_head)

def remove_filter_bands(obj: dict):
    obj["process_graph"].pop("filterbands1", None)
    obj["process_graph"]["apply5"]["arguments"]["data"]["from_node"] = "reducedimension3"




# -------------------------------------------------------------------------------
# CREATE AND WRITE THE UDP
# -------------------------------------------------------------------------------


def main():

    results = create_process_graph_with_parameters()

    spatial_extent_param = Parameter.spatial_extent()

    temporal_extent_param = Parameter.temporal_interval()

    orbit_state_param = Parameter.string(
        "orbit_state",
        description="Sentinel-1 orbit state.",
        values=["ASCENDING", "DESCENDING"],
    )
    postprocess_method_cropland_param = Parameter.string(
        "postprocess_method_cropland",
        description="Method used for postprocessing cropland.",
        values=["majority_vote", "smooth_probabilities"],
        optional=True,
        default=CROPLAND_DEFAULTS["method"],
    )
    postprocess_kernel_size_cropland_param = Parameter(
        "postprocess_kernel_size_cropland",
        description="Size of the kernel used for postprocessing cropland. Only used if postprocess_method_cropland is set to majority_vote.",
        schema={
            "type": "integer",
            "maximum": 25,
            "allOf": [{"not": {"multipleOf": 2}}],
        },
        optional=True,
        default=CROPLAND_DEFAULTS["kernel_size"],
    )

    postprocess_method_croptype_param = Parameter.string(
        "postprocess_method_croptype",
        description="Method used for postprocessing crop type.",
        values=["majority_vote", "smooth_probabilities"],
        optional=True,
        default=CROPTYPE_DEFAULTS["method"],
    )
    postprocess_kernel_size_croptype_param = Parameter(
        "postprocess_kernel_size_croptype",
        description="Size of the kernel used for postprocessing crop type. Only used if postprocess_method_croptype is set to majority_vote.",
        schema={
            "type": "integer",
            "maximum": 25,
            "allOf": [{"not": {"multipleOf": 2}}],
        },
        optional=True,
        default=CROPTYPE_DEFAULTS["kernel_size"],
    )

    season_ids_param = Parameter(
        "season_ids",
        description="List of season ids to run the inference for. Must match the keys of the season_windows parameter.",
        schema={
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {"type": "string"},
        },
    )
    season_windows_param = Parameter(
        "season_windows",
        description="Season windows for each season",
        schema={
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "format": "date"},
            },
            "minProperties": 1,
            "maxProperties": 2,
        },
    )

    seasonal_model_zip_param = Parameter.string(
        "seasonal_model_zip",
        description="Model to be used for feature computation. Also used for classification if no custom landcover_head_zip is provided.",
        optional=True,
        default=DEFAULT_SEASONAL_MODEL_URL,
    )

    landcover_head_zip_param = Parameter.string(
        "landcover_head_zip",
        description="Custom land cover classification head to be applied.",
        optional=True,
        default=None
    )

    enable_cropland_head_param = Parameter.boolean(
        "enable_cropland_head",
        description="Whether to generate a cropland product or not.",
        optional=True,
        default=True,
    )

    mask_cropland_param = Parameter.boolean(
        "mask_cropland",
        description="Whether to mask non-cropland areas from the crop type product based on the generated cropland product.",
        optional=True,
        default=True,
    )



    spec = build_process_dict(
        process_id="worldcereal_crop_type",
        process_graph=results,
        parameters=[
            spatial_extent_param,
            temporal_extent_param,
            orbit_state_param,
            season_ids_param,
            season_windows_param,
            postprocess_method_cropland_param,
            postprocess_kernel_size_cropland_param,
            postprocess_method_croptype_param,
            postprocess_kernel_size_croptype_param,
            seasonal_model_zip_param,
            landcover_head_zip_param,
            enable_cropland_head_param,
            mask_cropland_param,
        ],
        default_job_options=default_job_options,
    )

    replace_temporal_extent(spec)
    replace_spatial_extent(spec)
    replace_orbit_state(spec)
    replace_season_ids(spec)
    replace_season_windows(spec)
    replace_postprocess_method_cropland(spec)
    replace_kernel_size_cropland(spec)
    replace_postprocess_method_croptype(spec)
    replace_kernel_size_croptype(spec)
    replace_seasonal_head_zip(spec)
    replace_landcover_head_zip(spec)
    replace_mask_cropland(spec)
    replace_enable_cropland_head(spec)
    remove_filter_bands(spec)

    path_to_udp_json = "./worldcereal_crop_type.json"
    with open(path_to_udp_json, "w") as f:
        json.dump(spec, f, indent=2)


if __name__ == "__main__":
    main()
