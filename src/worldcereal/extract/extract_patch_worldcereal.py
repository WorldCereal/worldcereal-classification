"""Extract WorldCereal preprocessed inputs using OpenEO-GFMAP package."""

import json
import shutil
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Tuple, Union

import geojson
import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
import pystac
import xarray as xr
from openeo_gfmap import (
    Backend,
    BackendContext,
    BoundingBoxExtent,
    FetchType,
    TemporalContext,
)
from openeo_gfmap.manager.job_splitters import load_s2_grid
from openeo_gfmap.utils.catalogue import select_s1_orbitstate_vvvh
from pyproj import CRS
from tqdm import tqdm

from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.preprocessing import (
    precomposited_datacube_METEO,
    worldcereal_preprocessed_inputs,
)
from worldcereal.utils.geoloader import load_reproject

from worldcereal.extract.extract_common import (  # isort: skip
    get_job_nb_polygons,  # isort: skip
    pipeline_log,  # isort: skip
    upload_geoparquet_artifactory,  # isort: skip
)


WORLDCEREAL_BEGIN_DATE = datetime(2017, 1, 1)


def create_job_dataframe_patch_worldcereal(
    backend: Backend,
    split_jobs: List[gpd.GeoDataFrame],
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    rows = []
    for job in tqdm(split_jobs):
        # Compute the average in the valid date and make a buffer of 1.5 year around
        min_time = job.valid_time.min()
        max_time = job.valid_time.max()

        # Compute the average in the valid date and make a buffer of 1.5 year around
        # 9 months before and after the valid time
        start_date = (min_time - pd.Timedelta(days=275)).to_pydatetime()
        end_date = (max_time + pd.Timedelta(days=275)).to_pydatetime()

        # Impose limits due to the data availability
        start_date = max(start_date, WORLDCEREAL_BEGIN_DATE)
        end_date = min(end_date, datetime.now())

        # We need to start 1st of the month and end on the last day of the month
        start_date = start_date.replace(day=1)
        end_date = end_date + pd.offsets.MonthEnd(0)

        # Convert dates to string format
        start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime(
            "%Y-%m-%d"
        )

        s2_tile = job.tile.iloc[0]  # Job dataframes are split depending on the

        # Get job bounds
        geometry_bbox = job.to_crs(epsg=4326).total_bounds

        # Buffer if the geometry is a point
        if geometry_bbox[0] == geometry_bbox[2]:
            geometry_bbox = (
                geometry_bbox[0] - 0.0001,
                geometry_bbox[1],
                geometry_bbox[2] + 0.0001,
                geometry_bbox[3],
            )
        if geometry_bbox[1] == geometry_bbox[3]:
            geometry_bbox = (
                geometry_bbox[0],
                geometry_bbox[1] - 0.0001,
                geometry_bbox[2],
                geometry_bbox[3] + 0.0001,
            )

        # Find best orbit state
        orbit_state = select_s1_orbitstate_vvvh(
            backend=BackendContext(backend),
            spatial_extent=BoundingBoxExtent(*geometry_bbox),
            temporal_extent=TemporalContext(start_date, end_date),
        )

        # Set back the valid_time in the geometry as string
        job["valid_time"] = job.valid_time.dt.strftime("%Y-%m-%d")

        variables = {
            "backend_name": backend.value,
            "out_prefix": "WORLDCEREAL-INPUTS-10m",
            "out_extension": ".nc",
            "start_date": start_date,
            "end_date": end_date,
            "s2_tile": s2_tile,
            "geometry": job.to_json(),
        }
        variables.update({"orbit_state": orbit_state})

        rows.append(pd.Series(variables))

    return pd.DataFrame(rows)


def create_job_patch_worldcereal(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    executor_memory: str,
    python_memory: str,
    max_executors: int,
) -> openeo.BatchJob:
    """Creates an OpenEO BatchJob from the given row information."""

    # Load the temporal extent
    start_date = row.start_date
    end_date = row.end_date
    temporal_context = TemporalContext(start_date, end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)

    # Get the S1 orbit state
    orbit_state = row.orbit_state

    # Transfor to a GeoDataFrame
    def _to_gdf(geometries: geojson.FeatureCollection) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame.from_features(geometries).set_crs(epsg=4326)
        utm = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(utm)

        return gdf

    geometry_df = _to_gdf(geometry)
    spatial_extent_url = upload_geoparquet_artifactory(
        geometry_df, row.name, collection="worldcereal"
    )

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    # Get S2 tile extent
    s2_grid = load_s2_grid()
    geojson_features = (
        s2_grid.set_index("tile").loc[[row.s2_tile]].geometry.__geo_interface__
    )
    s2_tile_extent = geojson.GeoJSON(
        {"type": "FeatureCollection", "features": geojson_features["features"]}
    )

    # Create the job to extract preprocessed worldcereal inputs
    # Disable default meteo fetching
    cube = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent_url,
        temporal_extent=temporal_context,
        fetch_type=FetchType.POLYGON,
        s1_orbit_state=orbit_state,
        tile_size=128,
        disable_meteo=True,
    )

    # Get precomposited meteo cube using s2 tile extent
    meteo_cube = precomposited_datacube_METEO(
        connection=connection,
        temporal_extent=temporal_context,
        spatial_extent=s2_tile_extent,
    )

    # Join the meteo cube with the other inputs
    cube = cube.merge_cubes(meteo_cube)

    # Additional values to generate the BatchJob name
    s2_tile = row.s2_tile
    valid_time = geometry.features[0].properties["valid_time"]

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_polygons = get_job_nb_polygons(row)
    pipeline_log.debug("Number of polygons to extract %s", number_polygons)

    job_options = {
        "executor-memory": executor_memory,
        "python-memory": python_memory,
        "soft-errors": "true",
        "max_executors": max_executors,
    }
    return cube.create_job(
        out_format="NetCDF",
        title=f"GFMAP_Extraction_WORLDCEREAL_{s2_tile}_{valid_time}",
        sample_by_feature=True,
        job_options=job_options,
        feature_id_property="sample_id",
    )


def get_worldcereal_product(
    worldcereal_product: str, bounds: Tuple[int, int, int, int], epsg: int
) -> np.ndarray:
    """Method to load a WorldCereal product from a VRT file based
    on given bounds and EPSG.

    Parameters
    ----------
    worldcereal_product : str
        path to the VRT file of the WorldCereal product
    bounds : tuple
        bounds to load the data
    epsg : int
        EPSG in which bounds are expressed

    Returns
    -------
    np.ndarray
        resulting loaded WorldCereal product patch
    """
    vrt_file = f"/vitodata/worldcereal_data/MAP-v3/2021/VRTs/ESA_WORLDCEREAL_{worldcereal_product}_V1.vrt"

    data = load_reproject(vrt_file, bounds, epsg)
    data[np.isnan(data)] = 255
    data = data.astype(np.uint8)

    return data


def postprocess_extracted_file(
    item_asset_path: Union[Path, str], new_attributes: Dict[str, Union[str, int]]
) -> None:
    """Method to postprocess extracted NetCDF file. Will first generate new temporary
    file in current working directory and then move it to the original file.

    Parameters
    ----------
    item_asset_path : Union[Path, str]
        path to the original NetCDF file
    new_attributes : Dict[str, Union[str, int]]
        dictionary with new attributes to be added to the NetCDF file
    """

    GFMAP_BAND_MAPPING = {
        "S2-L2A-B02": "B2",
        "S2-L2A-B03": "B3",
        "S2-L2A-B04": "B4",
        "S2-L2A-B05": "B5",
        "S2-L2A-B06": "B6",
        "S2-L2A-B07": "B7",
        "S2-L2A-B08": "B8",
        "S2-L2A-B8A": "B8A",
        "S2-L2A-B11": "B11",
        "S2-L2A-B12": "B12",
        "S1-SIGMA0-VH": "VH",
        "S1-SIGMA0-VV": "VV",
        "AGERA5-TMEAN": "temperature_2m",
        "AGERA5-PRECIP": "total_precipitation",
    }

    # Open original file
    pipeline_log.info("Postprocessing file %s", item_asset_path)
    ds = xr.open_dataset(item_asset_path)

    # Strip borders for no-data artefacts of filter_spatial()
    x_crop = int(ds.x.shape[0] * 0.05)
    y_crop = int(ds.y.shape[0] * 0.05)
    ds = ds.isel(x=slice(x_crop, -x_crop), y=slice(y_crop, -y_crop))

    # Add WorldCereal phase I products
    bounds = ds.rio.bounds()
    epsg = CRS.from_wkt(ds.crs.attrs["crs_wkt"]).to_epsg()
    products = [
        "TEMPORARYCROPS",
        "WINTERCEREALS",
        "SPRINGCEREALS",
        "MAIZE-MAIN",
        "MAIZE-SECOND",
    ]
    for product in products:
        data = get_worldcereal_product(product, bounds, epsg)
        ds[f"WORLDCEREAL_{product}_2021"] = (("y", "x"), data)
        ds[f"WORLDCEREAL_{product}_2021"].encoding["grid_mapping"] = "crs"

    # Map to presto band names
    new_band_names = {b: GFMAP_BAND_MAPPING.get(b, b) for b in ds if b != "crs"}
    ds = ds.rename(new_band_names)

    # Compute slope if it's not present yet
    if "slope" not in ds:
        pipeline_log.info("Computing slope")
        slope = (
            PrestoFeatureExtractor()
            .compute_slope(
                ds.isel(t=1)[["elevation"]].to_array(dim="bands"), resolution=10
            )
            .sel(bands="slope", drop=True)
            .assign_attrs({"grid_mapping": "crs"})
        )
        ds["slope"] = slope

    # Make DEM 2D
    da = ds["elevation"].isel(t=0, drop=True).assign_attrs({"grid_mapping": "crs"})
    ds["elevation"] = da

    # Update attributes
    pipeline_log.info("Updating attributes")
    ds = ds.assign_attrs(new_attributes)

    # Save to temporary file
    tempfile = f"./{Path(item_asset_path).name}"
    pipeline_log.info("Saving to new temporary file")
    ds.to_netcdf(tempfile)

    # Move to output
    pipeline_log.info("Moving to original file")
    shutil.move(tempfile, item_asset_path)


def post_job_action_patch_worldcereal(
    job_items: List[pystac.Item],
    row: pd.Series,
    extract_value: int,
    description: str,
    title: str,
    spatial_resolution: str,
) -> list:
    """From the job items, extract the metadata and save it in a netcdf file."""
    base_gpd = gpd.GeoDataFrame.from_features(json.loads(row.geometry)).set_crs(
        epsg=4326
    )
    if len(base_gpd[base_gpd.extract == extract_value]) != len(job_items):
        pipeline_log.warning(
            "Different amount of geometries in the job output items and the "
            "input geometry. Job items #: %s, Input geometries #: %s",
            len(job_items),
            len(base_gpd[base_gpd.extract == extract_value]),
        )

    extracted_gpd = base_gpd[base_gpd.extract == extract_value].reset_index(drop=True)
    # In this case we want to burn the metadata in a new file in the same folder as the S2 product
    for item in job_items:
        item_id = item.id.replace(".nc", "").replace("openEO_", "")
        sample_id_column_name = "sample_id"

        geometry_information = extracted_gpd.loc[
            extracted_gpd[sample_id_column_name] == item_id
        ].squeeze()

        if len(geometry_information) == 0:
            pipeline_log.warning(
                "No geometry found for the sample_id %s in the input geometry.",
                item_id,
            )
            continue

        sample_id = geometry_information[sample_id_column_name]
        valid_time = geometry_information.valid_time
        s2_tile = row.s2_tile

        item_asset_path = Path(list(item.assets.values())[0].href)

        # Define some metadata to the result_df netcdf file
        new_attributes = {
            "start_date": row.start_date,
            "end_date": row.end_date,
            "valid_time": valid_time,
            "GFMAP_version": version("openeo_gfmap"),
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "title": title,
            "sample_id": sample_id,
            "spatial_resolution": spatial_resolution,
            "s2_tile": s2_tile,
            "_FillValue": 65535,  # No data value for uint16
        }

        # Postprocess file
        postprocess_extracted_file(item_asset_path, new_attributes)

    return job_items


def generate_output_path_patch_worldcereal(
    root_folder: Path,
    job_index: int,
    row: pd.Series,
    asset_id: str,
    s2_grid: gpd.GeoDataFrame,
):
    """Generate the output path for the extracted data, from a base path and
    the row information.
    """
    sample_id = asset_id.replace(".nc", "").replace("openEO_", "")

    s2_tile_id = row.s2_tile
    epsg = s2_grid[s2_grid.tile == s2_tile_id].iloc[0].epsg

    return (
        root_folder
        / f"{row.out_prefix}_{sample_id}_{epsg}_{row.start_date}_{row.end_date}{row.out_extension}"
    )
