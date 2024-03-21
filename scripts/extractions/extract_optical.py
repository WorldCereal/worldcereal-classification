"""Extract S2 data using OpenEO-GFMAP package."""
import argparse
import json
from datetime import datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import List

import geojson
import geopandas as gpd
import openeo
import pandas as pd
import pystac
import xarray as xr
from extract_sar import (
    _buffer_geometry,
    _filter_extract_true,
    _get_job_nb_polygons,
    _pipeline_log,
    _setup_logger,
    _upload_geoparquet_artifactory,
    create_job_dataframe,
    generate_output_path,
)
from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.backend import cdse_staging_connection
from openeo_gfmap.fetching.s2 import build_sentinel2_l2a_extractor
from openeo_gfmap.manager import _log
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import (
    _append_h3_index,
    _load_s2_grid,
    split_job_s2grid,
)
from pyproj import CRS
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

AUXILIARY = pystac.extensions.item_assets.AssetDefinition(
    {
        "title": "ground truth data",
        "description": "This asset contains the crop type codes.",
        "type": "application/x-netcdf",
        "roles": ["data"],
        "proj:shape": [64, 64],
        "raster:bands": [
            {"name": "CROPTYPE", "data_type": "uint16", "bits_per_sample": 16}
        ],
    }
)


def create_datacube_optical(
    row: pd.Series,
    connection: openeo.DataCube,
    provider=None,
    connection_provider=None,
    executor_memory: str = "5G",
    executor_memory_overhead: str = "2G",
) -> gpd.GeoDataFrame:
    start_date = row.start_date
    end_date = row.end_date
    temporal_context = TemporalContext(start_date, end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)

    # Filter the geometry to the rows with the extract only flag
    geometry = _filter_extract_true(geometry)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Performs a buffer of 64 px around the geometry
    geometry_df = _buffer_geometry(geometry)
    spatial_extent_url = _upload_geoparquet_artifactory(geometry_df, row.name)

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    fetch_type = FetchType.POLYGON
    bands_to_download = [
        "S2-L2A-B01",
        "S2-L2A-B02",
        "S2-L2A-B03",
        "S2-L2A-B04",
        "S2-L2A-B05",
        "S2-L2A-B06",
        "S2-L2A-B07",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B09",
        "S2-L2A-B11",
        "S2-L2A-B12",
        "S2-L2A-SCL",
    ]

    # Extract the SCL collection only
    sub_collection = connection.load_collection(
        collection_id="SENTINEL2_L2A",
        bands=["SCL"],
        temporal_extent=[start_date, end_date],
        properties={
            "eo:cloud_cover": lambda val: val <= 95.0,
            "tileId": lambda val: val == row.s2_tile,
        },
    )

    # Compute the SCL dilation mask
    scl_dilated_mask = sub_collection.process(
        "to_scl_dilation_mask",
        data=sub_collection,
        scl_band_name="SCL",
        kernel1_size=17,  # 17px dilation on a 20m layer
        kernel2_size=77,  # 77px dilation on a 20m layer
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3,
    ).rename_labels("bands", ["S2-L2A-SCL_DILATED_MASK"])

    # Compute the distance to cloud and add it to the cube
    distance_to_cloud = sub_collection.apply_neighborhood(
        process=openeo.UDF.from_file(
            Path(__file__).parent / "udf_distance_to_cloud.py"
        ),
        size=[
            {"dimension": "x", "unit": "px", "value": 256},
            {"dimension": "y", "unit": "px", "value": 256},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 16},
            {"dimension": "y", "unit": "px", "value": 16},
        ],
    ).rename_labels("bands", ["S2-L2A-DISTANCE-TO-CLOUD"])

    additional_masks = scl_dilated_mask.merge_cubes(distance_to_cloud)

    # Create the job to extract S2
    extraction_parameters = {
        "target_resolution": None,  # Disable target resolution
        "load_collection": {
            "eo:cloud_cover": lambda val: val <= 95.0,
            "tileId": lambda val: val == row.s2_tile,
        },
        "pre_merge": additional_masks,  # Add an additional mask computed from the SCL layer
        "pre_mask": scl_dilated_mask,
    }
    extractor = build_sentinel2_l2a_extractor(
        backend_context,
        bands=bands_to_download,
        fetch_type=fetch_type.POLYGON,
        **extraction_parameters,
    )

    cube = extractor.get_cube(connection, spatial_extent_url, temporal_context)

    cube = cube.linear_scale_range(0, 65534, 0, 65534)

    # Get the h3index to use in the tile
    s2_tile = row.s2_tile
    valid_date = geometry.features[0].properties["valid_date"]

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_polygons = _get_job_nb_polygons(row)
    _log.debug("Number of polygons to extract %s", number_polygons)

    job_options = {
        "executor-memory": executor_memory,
        "executor-memoryOverhead": executor_memory_overhead,
    }

    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_S2_{s2_tile}_{valid_date}",
        sample_by_feature=True,
        job_options=job_options,
    )


def add_item_asset(related_item: pystac.Item, path: Path):
    asset = AUXILIARY.create_asset(href=path.as_posix())
    related_item.add_asset("auxiliary", asset)


def post_job_action(
    job_items: List[pystac.Item], row: pd.Series, parameters: dict = {}
) -> list:
    base_gpd = gpd.GeoDataFrame.from_features(json.loads(row.geometry)).set_crs(
        epsg=4326
    )
    assert len(base_gpd[base_gpd.extract]) == len(
        job_items
    ), "The number of result paths should be the same as the number of geometries"

    extracted_gpd = base_gpd[base_gpd.extract].reset_index(drop=True)
    # In this case we want to burn the metadata in a new file in the same folder as the S2 product
    for idx, item in enumerate(job_items):
        if "sample_id" in extracted_gpd.columns:
            sample_id = extracted_gpd.iloc[idx].sample_id
        else:
            sample_id = extracted_gpd.iloc[idx].sampleID

        ref_id = extracted_gpd.iloc[idx].ref_id
        confidence = extracted_gpd.iloc[idx].confidence
        valid_date = extracted_gpd.iloc[idx].valid_date
        h3index = extracted_gpd.iloc[idx].h3index
        s2_tile = row.s2_tile

        item_asset_path = Path(list(item.assets.values())[0].href)
        # Read information from the item file (could also read it from the item object metadata)
        result_ds = xr.open_dataset(item_asset_path)

        # Add some metadata to the result_df netcdf file
        result_ds.attrs.update(
            {
                "start_date": row.start_date,
                "end_date": row.end_date,
                "valid_date": valid_date,
                "GFMAP_version": version("openeo_gfmap"),
                "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": f"Sentinel2 L2A observations for sample: {sample_id}, unprocessed.",
                "title": f"Sentinel2 L2A - {sample_id}",
                "sample_id": sample_id,
                "ref_id": ref_id,
                "spatial_resolution": "10m",
                "s2_tile": s2_tile,
                "h3index": h3index,
            }
        )
        result_ds.to_netcdf(item_asset_path)

        target_crs = CRS.from_wkt(result_ds.crs.attrs["crs_wkt"])

        # Get the surrounding polygons around our extracted center geometry to rastetize them
        bounds = (
            result_ds.x.min().item(),
            result_ds.y.min().item(),
            result_ds.x.max().item(),
            result_ds.y.max().item(),
        )
        bbox = box(*bounds)
        surround_gpd = base_gpd.to_crs(target_crs).clip(bbox)

        # Burn the polygon croptypes
        transform = from_bounds(*bounds, result_ds.x.size, result_ds.y.size)
        croptype_shapes = list(zip(surround_gpd.geometry, surround_gpd.croptype_label))

        fill_value = 0
        croptype = rasterize(
            croptype_shapes,
            out_shape=(result_ds.y.size, result_ds.x.size),
            transform=transform,
            all_touched=False,
            fill=fill_value,
            default_value=0,
            dtype="int64",
        )

        # Create the attributes to add to the metadata
        attributes = {
            "ref_id": ref_id,
            "sample_id": sample_id,
            "confidence": str(confidence),
            "valid_date": valid_date,
            "_FillValue": fill_value,
            "Conventions": "CF-1.9",
            "GFMAP_version": version("openeo_gfmap"),
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": f"Contains rasterized WorldCereal labels for sample: {sample_id}.",
            "title": f"WORLDCEREAL Auxiliary file for sample: {sample_id}",
            "spatial_resolution": "10m",
        }

        aux_dataset = xr.Dataset(
            {"LABEL": (("y", "x"), croptype), "crs": result_ds["crs"]},
            coords={"y": result_ds.y, "x": result_ds.x},
            attrs=attributes,
        )
        # Required to map the 'crs' layer as geo-reference for the 'LABEL' layer.
        aux_dataset["LABEL"].attrs["grid_mapping"] = "crs"

        # Save the metadata in the same folder as the S2 product
        metadata_path = (
            item_asset_path.parent
            / f"WORLDCEREAL-10m_{sample_id}_{target_crs.to_epsg()}_{valid_date}.nc"
        )
        aux_dataset.to_netcdf(metadata_path, format="NETCDF4", engine="h5netcdf")
        aux_dataset.close()

        # Adds this metadata as a new asset
        add_item_asset(item, metadata_path)

    return job_items


if __name__ == "__main__":
    _setup_logger()
    from extract_sar import _pipeline_log

    parser = argparse.ArgumentParser(
        description="S2 samples extraction with OpenEO-GFMAP package."
    )
    parser.add_argument(
        "output_path", type=Path, help="Path where to save the extraction results."
    )
    parser.add_argument(
        "input_df",
        type=str,
        help="Path or URL to the input dataframe for the training data.",
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=500,
        help="Maximum number of locations to extract per job.",
    )
    parser.add_argument(
        "--memory", type=str, default="3G", help="Memory to allocate for the executor."
    )
    parser.add_argument(
        "--memory-overhead",
        type=str,
        default="3G",
        help="Memory overhead to allocate for the executor.",
    )

    args = parser.parse_args()

    tracking_df_path = Path(args.output_path) / "job_tracking.csv"

    # Load the input dataframe
    _pipeline_log.info("Loading input dataframe from %s.", args.input_df)

    input_df = gpd.read_file(args.input_df)
    input_df = _append_h3_index(input_df, grid_resolution=3)

    split_dfs = split_job_s2grid(input_df, max_points=args.max_locations)
    split_dfs = [df for df in split_dfs if df.extract.any()]

    job_df = create_job_dataframe(Backend.CDSE_STAGING, split_dfs, prefix="S2-L2A-10m")

    _pipeline_log.warning(
        "Sub-sampling the job dataframe for testing. Remove this for production."
    )
    job_df = job_df.iloc[[0]]

    # Setup the memory parameters for the job creator.
    create_datacube_optical = partial(
        create_datacube_optical,
        executor_memory=args.memory,
        executor_memory_overhead=args.memory_overhead,
    )

    # Setup the s2 grid for the output path generation function
    generate_output_path = partial(
        generate_output_path,
        s2_grid=_load_s2_grid(),
    )

    manager = GFMAPJobManager(
        output_dir=args.output_path,
        output_path_generator=generate_output_path,
        post_job_action=post_job_action,
        collection_id="SENTINEL2-EXTRACTION",
        collection_description="Sentinel-2 and Auxiliary data extraction example.",
        poll_sleep=60,
        n_threads=2,
        post_job_params={},
    )

    manager.add_backend(
        Backend.CDSE_STAGING.value, cdse_staging_connection, parallel_jobs=6
    )

    _pipeline_log.info("Launching the jobs from the manager.")
    manager.run_jobs(job_df, create_datacube_optical, tracking_df_path)

    manager.create_stac(asset_definitions={"auxiliary": AUXILIARY})
