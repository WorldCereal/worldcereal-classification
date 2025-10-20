import copy
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import openeo
import pandas as pd
import pystac_client
from loguru import logger
from openeo.processes import ProcessBuilder, eq, if_
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.preprocessing.compositing import mean_compositing, median_compositing
from openeo_gfmap.preprocessing.sar import (
    compress_backscatter_uint16,
    decompress_backscatter_uint16,
)
from openeo_gfmap.utils.catalogue import UncoveredS1Exception, select_s1_orbitstate_vvvh
from pandas.core.dtypes.dtypes import CategoricalDtype
from shapely.geometry import MultiPolygon, shape

from worldcereal.extract.point_worldcereal import REQUIRED_ATTRIBUTES
from worldcereal.extract.utils import S2_GRID, upload_geoparquet_artifactory
from worldcereal.rdm_api import RdmInteraction
from worldcereal.rdm_api.rdm_interaction import RDM_DEFAULT_COLUMNS
from worldcereal.utils.refdata import gdf_to_points

STAC_ENDPOINT_S1 = (
    "https://stac.openeo.vito.be/collections/worldcereal_sentinel_1_patch_extractions"
)

STAC_ENDPOINT_S2 = (
    "https://stac.openeo.vito.be/collections/worldcereal_sentinel_2_patch_extractions"
)

STAC_ENDPOINT_METEO_TERRASCOPE = (
    "https://stac.openeo.vito.be/collections/agera5_monthly_terrascope"
)

STAC_ENDPOINT_SLOPE_TERRASCOPE = (
    "https://stac.openeo.vito.be/collections/COPERNICUS30_DEM_SLOPE_TERRASCOPE"
)

# Due to a bug on openEO side (https://github.com/Open-EO/openeo-geopyspark-driver/issues/1153)
# We have to provide here ALL bands in ALPHABETICAL order!
S2_BANDS = [
    "S2-L2A-B01",
    "S2-L2A-B02",
    "S2-L2A-B03",
    "S2-L2A-B04",
    "S2-L2A-B05",
    "S2-L2A-B06",
    "S2-L2A-B07",
    "S2-L2A-B08",
    "S2-L2A-B09",
    "S2-L2A-B11",
    "S2-L2A-B12",
    "S2-L2A-B8A",
    "S2-L2A-DISTANCE-TO-CLOUD",
    "S2-L2A-SCL",
    "S2-L2A-SCL_DILATED_MASK",
]

S2_BANDS_SELECTED = [
    "S2-L2A-B02",
    "S2-L2A-B03",
    "S2-L2A-B04",
    "S2-L2A-B05",
    "S2-L2A-B06",
    "S2-L2A-B07",
    "S2-L2A-B08",
    "S2-L2A-B8A",
    "S2-L2A-B11",
    "S2-L2A-B12",
    "S2-L2A-SCL_DILATED_MASK",
]


def label_points_centroid(
    gdf: gpd.GeoDataFrame, epsg: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Sample points from the centroid of the input GeoDataFrame.
    """
    if epsg is not None:
        gdf = gdf.to_crs(epsg=epsg)

    gdf["geometry"] = gdf.centroid

    if epsg is not None:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def get_label_points(
    row: pd.Series,
    ground_truth_file: Optional[Union[Path, str]] = None,
    only_flagged_samples: bool = False,
) -> gpd.GeoDataFrame:
    """
    Retrieve label points for a given row from STAC collections and RDM API.

    Parameters
    ----------
    row : pd.Series
        The row containing ref_id, epsg, start_date, and end_date.
    ground_truth_file : Optional[Union[Path, str]], optional
        The path to the ground truth file. If provided, this file will
        be queried for getting the ground truth. If not, the RDM will
        be used for the query.
    only_flagged_samples : bool, optional
        If True, only samples with extract flag >0 will be retrieved, no collateral samples.
        (This is useful for very large and dense datasets like USDA).

    Returns
    -------
    gpd.GeoDataFrame, bool
        A tuple containing:
        - gpd.GeoDataFrame: The sampled GeoDataFrame with label points.
        - bool: A flag indicating whether S1 extraction is disabled
                (True if no S1 sample_ids found).

    """

    # Find all items (i.e. patches) corresponding to the given ref_id and epsg
    stac_query = {
        "ref_id": {"eq": row["ref_id"]},
        "proj:epsg": {"eq": int(row["epsg"])},
    }
    client = pystac_client.Client.open("https://stac.openeo.vito.be/")

    search_s1 = client.search(
        collections=["worldcereal_sentinel_1_patch_extractions"], query=stac_query
    )
    search_s2 = client.search(
        collections=["worldcereal_sentinel_2_patch_extractions"], query=stac_query
    )

    logger.info("Querying S1/S2 STAC collections ...")
    items_s1 = {
        item.properties["sample_id"]: shape(item.geometry).buffer(1e-9)
        for item in search_s1.items()
    }
    items_s2 = {
        item.properties["sample_id"]: shape(item.geometry).buffer(1e-9)
        for item in search_s2.items()
    }

    # Find sample_ids which are present in either S2 or both STAC collections
    common_sample_ids = set(items_s1.keys()).intersection(set(items_s2.keys()))
    if len(common_sample_ids) == 0:
        logger.warning(
            "No common sample_ids found in S1 and S2 STAC collections. "
            "S1 extraction will be disabled from process graph!."
        )
        disable_s1 = True
    else:
        logger.info(f"Found {len(common_sample_ids)} common sample_ids in S1 and S2")
        disable_s1 = False

    s2_only_sample_ids = set(items_s2.keys()).difference(common_sample_ids)
    logger.info(f"Found {len(s2_only_sample_ids)} S2-only sample_ids")
    selected_sample_ids = common_sample_ids.union(s2_only_sample_ids)

    if len(selected_sample_ids) == 0:
        raise ValueError(
            "No sample_ids found in S1 or S2 STAC collections. "
            "Please check the extractions for this ref_id and epsg."
        )
    logger.info(f"Total selected sample_ids for extraction: {len(selected_sample_ids)}")

    # Items with the same sample_id will also have the same geometry
    polygons = [items_s2[sample_id] for sample_id in selected_sample_ids]

    multi_polygon = MultiPolygon(polygons)

    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)

    # From the RDM API, we also want other 'collateral' geometries from the same ref_id.
    gdf = RdmInteraction().get_samples(
        ref_ids=[row["ref_id"]],
        spatial_extent=multi_polygon,
        temporal_extent=temporal_extent,
        include_private=True,
        ground_truth_file=ground_truth_file,
        subset=only_flagged_samples,
    )

    sampled_gdf = gdf_to_points(gdf)

    return sampled_gdf, disable_s1


def generate_output_path_patch_to_point_worldcereal(
    root_folder: Path,
    geometry_index: int,
    row: pd.Series,
    asset_id: Optional[str] = None,
) -> Path:
    """
    Generate the output path for point extractions.

    Parameters
    ----------
    root_folder : Path
        Root folder where the output parquet file will be saved.
    geometry_index : int
        Index of the geometry. Always 0 for point extractions.
    row : pd.Series
        The current job row from the job manager.
    asset_id : str, optional
        Asset ID for compatibility with the job manager. Not used.

    Returns
    -------
    Path
        Path to the output parquet file.
    """

    epsg = row.epsg

    # Create the subfolder to store the output
    subfolder = root_folder / str(epsg)
    subfolder.mkdir(parents=True, exist_ok=True)

    # we may have multiple output files per s2_tile_id and need
    # a unique name so we use the job ID
    output_file = f"WORLDCEREAL_{root_folder.name}_{row.start_date}_{row.end_date}_{epsg}_{row.id}{row.out_extension}"

    return subfolder / output_file


def create_job_dataframe_patch_to_point_worldcereal(
    ref_id, ground_truth_file=None, only_flagged_samples: bool = False
):
    """
    Create a job dataframe for patch-to-point extractions.

    This function queries the STAC catalog to retrieve unique EPSG codes and temporal extents
    for the given `ref_id`. It also identifies ground truth samples and prepares the job
    dataframe for further processing.

    Parameters
    ----------
    ref_id : str
        Reference ID for the extraction.
    ground_truth_file : str, optional
        Path to a ground truth file. If not provided, the function queries RDM for ground truth.
    only_flagged_samples : bool, optional
        If True, only samples with extract flag >0 will be retrieved, no collateral samples.
        (This is useful for very large and dense datasets like USDA).

    Returns
    -------
    pd.DataFrame
        A dataframe containing job configurations for each EPSG zone.
    """

    client = pystac_client.Client.open("https://stac.openeo.vito.be/")

    stac_query = {
        "ref_id": {"eq": ref_id},
    }

    search = client.search(
        collections=["worldcereal_sentinel_2_patch_extractions"],
        query=stac_query,
    )

    # Get a list of EPSG codes that occur for this ref_id as we need
    # to run jobs per UTM zone.
    logger.info(f"Creating job dataframe for: {ref_id}")
    logger.info("Looking for unique EPSG codes in STAC collection ...")
    epsg_codes = {}
    for item in search.items():
        epsg = int(item.properties["proj:epsg"])

        if epsg not in epsg_codes and epsg != 4038:
            logger.debug(f"Found EPSG: {epsg}")

            epsg_codes[epsg] = {
                "start_date": pd.to_datetime(item.properties["start_date"]),
                "end_date": pd.to_datetime(item.properties["end_date"]),
            }
        elif epsg != 4038:
            current_start_date = pd.to_datetime(item.properties["start_date"])
            current_end_date = pd.to_datetime(item.properties["end_date"])
            if current_start_date < epsg_codes[epsg]["start_date"]:
                epsg_codes[epsg]["start_date"] = current_start_date
            if current_end_date > epsg_codes[epsg]["end_date"]:
                epsg_codes[epsg]["end_date"] = current_end_date

    # Initialize job dataframe for patch to point
    rows = []

    logger.info(f"Found {len(epsg_codes)} unique EPSG codes in STAC collection.")

    for epsg in epsg_codes.keys():
        # We assume identical start and end date for the entire ref_id
        start_date = epsg_codes[epsg]["start_date"]
        end_date = epsg_codes[epsg]["end_date"]

        # Clamp end_date to last day of the last complete month to avoid compositing an incomplete month
        last_complete_month_end = pd.Timestamp.today().normalize().replace(
            day=1
        ) - pd.Timedelta(days=1)
        end_date = min(pd.Timestamp(end_date), last_complete_month_end)

        # ensure start date is 1st day of month, end date is last day of month
        start_date = start_date.replace(day=1)
        end_date = end_date.replace(day=1) + pd.offsets.MonthEnd(0)

        # Convert dates to string format
        start_date, end_date = (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        variables = {
            "backend_name": "terrascope",
            "out_prefix": "patch-to-point",
            "out_extension": ".geoparquet",
            "start_date": start_date,
            "end_date": end_date,
            "ref_id": ref_id,
            "ground_truth_file": ground_truth_file,
            "epsg": epsg,
            "geometry_url": None,
        }
        rows.append(pd.Series(variables))

    job_df = pd.DataFrame(rows)
    job_df["geometry_url"] = job_df["geometry_url"].astype("string")

    # Now find matching ground truth by querying RDM
    for ix, row in job_df.iterrows():
        logger.info(f"Processing EPSG {row.epsg} for REF_ID {row.ref_id}")

        # Get the ground truth in the patches
        # Note that we can work around RDM by specifically providing a ground truth file
        logger.info("Finding ground truth samples ...")
        gdf, disable_s1 = get_label_points(
            row,
            ground_truth_file=row["ground_truth_file"],
            only_flagged_samples=only_flagged_samples,
        )
        gdf["ref_id"] = (
            row.ref_id
        )  # Overwrite due to current bug in automatic assignment

        if gdf.empty:
            logger.warning(f"No samples found for {row.epsg} and {row.ref_id}")
            continue
        else:
            logger.info(f"Found {len(gdf)} samples for {row.epsg} and {row.ref_id}")

        # Keep essential attributes only
        gdf = gdf[RDM_DEFAULT_COLUMNS]

        if not disable_s1:
            # Determine S1 orbit; very small buffer to cover cases with < 3 samples
            try:
                job_df.loc[ix, "orbit_state"] = select_s1_orbitstate_vvvh(
                    BackendContext(Backend.CDSE),
                    BoundingBoxExtent(
                        *gdf.to_crs(epsg=3857).buffer(1).to_crs(epsg=4326).total_bounds
                    ),
                    TemporalContext(row.start_date, row.end_date),
                )
            except UncoveredS1Exception:
                logger.warning(
                    f"No S1 orbit state found for {row.epsg} and {row.ref_id}. "
                    "This will result in no S1 data being extracted."
                )
                job_df.loc[ix, "orbit_state"] = "DESCENDING"  # Just a placeholder
        else:
            # Disabling S1 which cannot be automatically handled yet by openEO
            job_df.loc[ix, "orbit_state"] = None

        # Determine S2 tiles
        logger.info("Finding S2 tiles ...")
        original_crs = gdf.crs
        gdf = gdf.to_crs(epsg=3857)
        gdf["centroid"] = gdf.geometry.centroid

        gdf = gpd.sjoin(
            gdf.set_geometry("centroid"),
            S2_GRID[["tile", "geometry"]].to_crs(epsg=3857),
            predicate="intersects",
        ).drop(columns=["index_right", "centroid"])
        gdf = gdf.set_geometry("geometry").to_crs(original_crs)

        # Set back the valid_time in the geometry as string
        gdf["valid_time"] = gdf.valid_time.dt.strftime("%Y-%m-%d")

        # Add other attributes we want to keep in the result
        logger.info(f"Determined start and end date: {row.start_date} - {row.end_date}")
        gdf["start_date"] = row.start_date
        gdf["end_date"] = row.end_date
        gdf["lat"] = gdf.geometry.y
        gdf["lon"] = gdf.geometry.x

        # Reset index for certain openEO compatibility
        gdf = gdf.reset_index(drop=True)

        # Upload the geoparquet file to Artifactory
        logger.info("Deploying geoparquet file to Artifactory ...")
        # url = upload_geoparquet_s3("cdse", gdf, ref_id, collection=f"{row.epsg}")
        url = upload_geoparquet_artifactory(gdf, ref_id, collection=f"{row.epsg}")

        # Get sample points from RDM
        job_df.loc[ix, "geometry_url"] = url

    # Remove rows without geometry URL as indication for jobs to skip
    job_df = job_df[job_df["geometry_url"].notna()]

    return job_df


def create_job_patch_to_point_worldcereal(
    row: pd.Series,
    connection: openeo.Connection,
    provider,
    connection_provider,
    job_options: dict,
    period="month",
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Assume row has the following fields: backend, start_date, end_date, epsg, ref_id and geometry_url

    # s1_orbit_state = row.get(
    #     "orbit_state", "DESCENDING"
    # )  # default to DESCENDING, same as for inference workflow

    # Currently, empty datacubes for NetCDF collections are not supported
    # so we have to manually take care of the no-S1 case.
    s1_orbit_state = row.get("orbit_state") if not row.isnull()["orbit_state"] else None

    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)

    # Get preprocessed cube from patch extractions
    logger.info(f"Creating cube with compositing window: {period}")
    cube = worldcereal_preprocessed_inputs_from_patches(
        connection,
        temporal_extent=temporal_extent,
        ref_id=row["ref_id"],
        epsg=int(row["epsg"]),
        s1_orbit_state=s1_orbit_state,
        period=period,
    )

    # Do spatial aggregation
    point_geometries = connection.load_url(
        url=str(row["geometry_url"]), format="Parquet"
    )
    cube = cube.aggregate_spatial(geometries=point_geometries, reducer="mean")

    return cube.create_job(
        title=f"WorldCereal patch-to-point extraction for: {row['ref_id']} and epsg: {row['epsg']} (period: {period})",
        out_format="Parquet",
        job_options=job_options,
    )


def post_job_action_point_worldcereal(parquet_file):
    """
    Perform post-processing on the extracted parquet file.

    This function cleans and validates the extracted data, removes invalid samples,
    and ensures the data conforms to the required schema.

    Parameters
    ----------
    parquet_file : str or Path
        Path to the parquet file to be processed.

    Returns
    -------
    None
    """

    logger.info(f"Running post-job action for: {parquet_file}")
    gdf = gpd.read_parquet(parquet_file)

    # Convert the dates to datetime format
    gdf["timestamp"] = pd.to_datetime(gdf["date"])
    gdf.drop(columns=["date"], inplace=True)

    if "S1-SIGMA0-VH" not in gdf.columns or "S1-SIGMA0-VV" not in gdf.columns:
        logger.warning(
            "S1 bands not found in the extracted data. "
            "This probably due to disabling of S1 in patch-to-point."
            " Filling with nodata values."
        )
        gdf["S1-SIGMA0-VH"] = 65535
        gdf["S1-SIGMA0-VV"] = 65535

    # Convert band dtype to uint16 (temporary fix)
    # TODO: remove this step when the issue is fixed on the OpenEO backend
    bands = [
        "S2-L2A-B02",
        "S2-L2A-B03",
        "S2-L2A-B04",
        "S2-L2A-B05",
        "S2-L2A-B06",
        "S2-L2A-B07",
        "S2-L2A-B08",
        "S2-L2A-B8A",
        "S2-L2A-B11",
        "S2-L2A-B12",
        "S1-SIGMA0-VH",
        "S1-SIGMA0-VV",
        "elevation",
        "slope",
        "AGERA5-PRECIP",
        "AGERA5-TMEAN",
    ]
    gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

    # Remove samples where S1 and S2 are completely nodata
    cols = [c for c in gdf.columns if "S2" in c or "S1" in c]
    orig_sample_nr = len(gdf["sample_id"].unique())
    nodata_rows = (gdf[cols] == 65535).all(axis=1)
    all_nodata_per_sample = (
        gdf.assign(nodata=nodata_rows).groupby("sample_id")["nodata"].all()
    )
    valid_sample_ids = all_nodata_per_sample[~all_nodata_per_sample].index
    removed_samples = orig_sample_nr - len(valid_sample_ids)
    if removed_samples > 0:
        logger.warning(
            f"Removed {removed_samples} samples with all S1 and S2 bands as nodata."
        )
        gdf = gdf[gdf["sample_id"].isin(valid_sample_ids)]

    # Do some checks and perform corrections
    assert (
        len(gdf["ref_id"].unique()) == 1
    ), f"There are multiple ref_ids in the dataframe: {gdf['ref_id'].unique()}"
    ref_id = gdf["ref_id"].iloc[0]
    year = int(ref_id.split("_")[0])
    gdf["year"] = year

    # Make sure we remove the timezone information from the timestamp
    gdf["timestamp"] = gdf["timestamp"].dt.tz_localize(None)

    # Select required attributes and cast to dtypes
    required_attributes = copy.deepcopy(REQUIRED_ATTRIBUTES)
    required_attributes["ref_id"] = CategoricalDtype(categories=[ref_id], ordered=False)
    gdf = gdf[required_attributes.keys()]
    gdf = gdf.astype(required_attributes)

    gdf.to_parquet(parquet_file, index=False)


def worldcereal_preprocessed_inputs_from_patches(
    connection,
    temporal_extent,
    ref_id: str,
    epsg: int,
    s1_orbit_state: Optional[str] = None,
    period: Optional[str] = "month",
):
    assert period in ["month", "dekad"], "period must be either 'month' or 'dekad'"

    # TODO: move preprocessing to separate functions 'preprocess_cube_x(cube: openeo.DataCube) -> openeo.DataCube' which will be the same across the different extraction workflows
    s1_stac_property_filter = {
        "ref_id": lambda x: eq(x, ref_id),
        "proj:epsg": lambda x: eq(x, epsg),
        "sat:orbit_state": lambda x: eq(x, s1_orbit_state),
    }

    s2_stac_property_filter = {
        "ref_id": lambda x: eq(x, ref_id),
        "proj:epsg": lambda x: eq(x, epsg),
    }

    if s1_orbit_state is not None:
        s1_raw = connection.load_stac(
            url=STAC_ENDPOINT_S1,
            properties=s1_stac_property_filter,
            temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
            bands=["S1-SIGMA0-VH", "S1-SIGMA0-VV"],
        )
        s1_raw.result_node().update_arguments(featureflags={"allow_empty_cube": True})
        s1 = decompress_backscatter_uint16(backend_context=None, cube=s1_raw)
        s1 = mean_compositing(s1, period=period)
        s1 = compress_backscatter_uint16(backend_context=None, cube=s1)
    else:
        logger.warning("No S1 orbit state provided, S1 extraction will be disabled.")

    s2_raw = connection.load_stac(
        url=STAC_ENDPOINT_S2,
        properties=s2_stac_property_filter,
        temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
        bands=S2_BANDS,
    ).filter_bands(S2_BANDS_SELECTED)

    def optimized_mask(input: ProcessBuilder):
        """
        To be used as a callback to apply_dimension on the band dimension.
        It's an optimized way of masking, if the mask is already present in the cube.
        """
        mask_band = input.array_element(label="S2-L2A-SCL_DILATED_MASK")
        return if_(mask_band != 1, input)

    s2 = s2_raw.apply_dimension(dimension="bands", process=optimized_mask)
    s2 = median_compositing(s2, period=period)
    s2 = s2.filter_bands(S2_BANDS_SELECTED[:-1])
    s2 = s2.linear_scale_range(0, 65534, 0, 65534)

    dem_raw = connection.load_collection("COPERNICUS_30", bands=["DEM"])
    dem_raw = dem_raw.resample_spatial(
        resolution=10.0, projection=epsg, method="bilinear"
    )
    dem = dem_raw.min_time()
    dem = dem.rename_labels(dimension="bands", target=["elevation"], source=["DEM"])

    slope = connection.load_stac(
        STAC_ENDPOINT_SLOPE_TERRASCOPE,
        bands=["Slope"],
    ).rename_labels(dimension="bands", target=["slope"])
    slope = slope.resample_spatial(resolution=10.0, projection=epsg, method="bilinear")
    # Client fix for CDSE, the openeo client might be unsynchronized with
    # the backend.
    if "t" not in slope.metadata.dimension_names():
        slope.metadata = slope.metadata.add_dimension("t", "2020-01-01", "temporal")
    slope = slope.min_time()

    copernicus = slope.merge_cubes(dem)
    copernicus = copernicus.linear_scale_range(0, 65534, 0, 65534)

    if period == "month":
        # Load precomposited monthly meteo data
        meteo_raw = connection.load_stac(
            url=STAC_ENDPOINT_METEO_TERRASCOPE,
            temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
            bands=["temperature-mean", "precipitation-flux"],
        )
    elif period == "dekad":
        # Load precomposited dekadal meteo data
        meteo_raw = connection.load_stac(
            url="https://stac.openeo.vito.be/collections/agera5_dekad_terrascope",
            temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
            bands=["temperature-mean", "precipitation-flux"],
        )

    meteo = meteo_raw.resample_spatial(
        resolution=10.0, projection=epsg, method="bilinear"
    )
    meteo = meteo.rename_labels(
        dimension="bands",
        target=["AGERA5-TMEAN", "AGERA5-PRECIP"],
    )

    if s1_orbit_state is None:
        # If no S1 orbit state is provided, we disable S1
        cube = s2
    else:
        cube = s2.merge_cubes(s1)
    cube = cube.merge_cubes(meteo)
    cube = cube.merge_cubes(copernicus)

    return cube
