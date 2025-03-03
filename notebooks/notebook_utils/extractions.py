import logging
from pathlib import Path
from typing import List, Optional

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from openeo_gfmap.manager.job_splitters import load_s2_grid
from shapely.geometry import Polygon

from worldcereal.extract.common import load_dataframe
from worldcereal.rdm_api import RdmInteraction
from worldcereal.utils.refdata import gdf_to_points

logging.getLogger("rasterio").setLevel(logging.ERROR)

RDM_ATTRIBUTES = [
    "sample_id",
    "ewoc_code",
    "valid_time",
    "irrigation_status",
    "extract",
    "quality_score_lc",
    "quality_score_ct",
    "h3_l3_cell",
    "geometry",
]


def prepare_samples_dataframe(
    input_df: str,
    ref_id: str,
    subset: bool = False,
) -> gpd.GeoDataFrame:
    """Prepare the samples dataframe for the point extractions.

    Parameters
    ----------
    input_df : str
        path to the input dataframe containing the geometries
        for which extractions need to be done
    ref_id : str
        The collection ID.
    subset : bool, optional
        whether to subset the input dataframe to reduce sample size, by default False

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the samples
    """

    gdf = load_dataframe(Path(input_df))
    if not subset:
        # Set extract attribute to 1 for all samples
        gdf["extract"] = 1

    # Keep essential attributes only
    gdf = gdf[RDM_ATTRIBUTES]

    # Add ref_id attribute
    gdf["ref_id"] = ref_id

    # Convert geometries to points
    gdf = gdf_to_points(gdf)

    return gdf


def query_extractions(
    bbox_poly: Polygon,
    buffer: int = 250000,
    include_public: bool = True,
    private_extraction_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Method to query the WorldCereal global database of pre-extracted input data for a given area.
    You can query both public and private extractions.
    Parameters
    ----------
    bbox_poly : Polygon
        bounding box of the area to make the query for. Expected to be in WGS84 coordinates.
    buffer : int, optional
        buffer (in meters) to apply to the requested area, by default 250000
    include_public: bool, optional
        include public extractions in the query, by default True
    private_extraction_dir: Path, optional
        path to the directory containing private extractions, by default None,
        meaning no private extractions will be included.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the extractions matching the request.
    """

    # Prepare the spatial extent
    logger.info(f"Applying a buffer of {int(buffer/1000)} km to the selected area ...")

    bbox_poly = (
        gpd.GeoSeries(bbox_poly, crs="EPSG:4326")
        .to_crs(epsg=3785)
        .buffer(buffer, cap_style="square", join_style="mitre")
        .to_crs(epsg=4326)[0]
    )

    xmin, ymin, xmax, ymax = bbox_poly.bounds
    bbox_poly = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    # Query the public extractions
    if include_public:
        public_df = _query_public_extractions(bbox_poly)
    else:
        public_df = pd.DataFrame()

    # Query the private extractions
    if private_extraction_dir:
        private_df = _query_private_extractions(private_extraction_dir, bbox_poly)
    else:
        private_df = pd.DataFrame()

    # Combine the public and private extractions
    dfs = [public_df, private_df]
    dfs = [df for df in dfs if df.empty is False]
    if not dfs:
        logger.error(
            "No extractions found that intersect with the selected area, cannot continue."
        )
        raise ValueError(
            "No extractions found  that intersect with the selected area, cannot continue."
        )
    df_raw = pd.concat(dfs, ignore_index=True)

    # Check if the data contains only one class
    if df_raw["ewoc_code"].nunique() == 1:
        logger.error(
            f"Queried data contains only one class: {df_raw['ewoc_code'].unique()[0]}. Cannot train a model with only one class."
        )
        raise ValueError(
            "Queried data contains only one class. Cannot train a model with only one class."
        )

    return df_raw


def _query_public_extractions(
    bbox_poly: Polygon,
) -> pd.DataFrame:
    """Method to query the WorldCereal public database of pre-extracted input data for a given area.
    Parameters
    ----------
    bbox_poly : Polygon
        bounding box of the area to make the query for. Expected to be in WGS84 coordinates.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the extractions matching the request.
    """

    # First we query the RDM to get a list of overlapping datasets
    rdm = RdmInteraction()
    collections = rdm.get_collections(geometry=bbox_poly)
    ref_ids = [collection.id for collection in collections]

    if not ref_ids:
        logger.warning(
            "No public datasets found in the WorldCereal Reference Data Module that intersect with the selected area."
        )
        return pd.DataFrame()

    logger.info(
        f"Found {len(ref_ids)} public datasets in WorldCereal Reference Data Module that intersect with the selected area."
    )
    # Query the public extraction
    logger.info(
        "Querying WorldCereal public extractions database (this can take a while) ..."
    )

    # TODO: construct the query...
    main_query = ""

    # Execute the query
    db = duckdb.connect()
    db.sql("INSTALL spatial")
    db.load_extension("spatial")

    public_df_raw = db.sql(main_query).df()

    if public_df_raw.empty:
        logger.warning(
            "No samples from the WorldCereal public extractions database fall into the selected area."
        )

    return public_df_raw


def _query_private_extractions(
    private_extraction_dir: Path,
    bbox_poly: Polygon,
) -> pd.DataFrame:
    """Method to query private pre-extracted input data for a given area.
    Parameters
    ----------
    private_extraction_dir: Path
        path to a local directory containing private extractions
    bbox_poly : Polygon
        bounding box of the area to make the query for. Expected to be in WGS84 coordinates.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the extractions matching the request.
    """

    # First we query the RDM to get a list of overlapping datasets
    rdm = RdmInteraction()
    collections = rdm.get_collections(
        geometry=bbox_poly, include_private=True, include_public=False
    )
    ref_ids = [collection.id for collection in collections]

    if not ref_ids:
        logger.warning(
            "No private datasets found in the WorldCereal Reference Data Module that intersect with the selected area."
        )
        return pd.DataFrame()

    logger.info(
        f"Found {len(ref_ids)} private datasets in WorldCereal Reference Data Module that intersect with the selected area."
    )

    # Check which S2 tiles are intersecting with the selected area
    s2_grid = load_s2_grid()
    s2_tiles = s2_grid[s2_grid.intersects(bbox_poly)]
    s2_tile_ids = s2_tiles["tile"].values

    # Now locate the associated parquet files in the private extraction directory
    parquet_files = []
    for ref_id in ref_ids:
        search_dir = private_extraction_dir / ref_id
        tile_dirs = [search_dir / tile_id for tile_id in s2_tile_ids]
        for tile_dir in tile_dirs:
            if tile_dir.exists():
                parquet_files.extend(
                    list(tile_dir.glob("**/point_extractions.geoparquet"))
                )

    if not parquet_files:
        logger.warning("No private extractions found in the selected area.")
        return pd.DataFrame()

    # Load the DuckDB spatial extension
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")

    # Convert the Shapely polygon to WKT
    query_polygon_wkt = bbox_poly.wkt

    # Construct the query
    query = f"""
    SELECT *
    FROM read_parquet({parquet_files})
    WHERE ST_Intersects(ST_GeomFromWKT(geometry), ST_GeomFromText('{query_polygon_wkt}'))
    """

    # Execute the query and convert results to a GeoDataFrame
    result = conn.execute(query).fetchdf()
    gdf = gpd.GeoDataFrame(result, geometry=gpd.GeoSeries.from_wkt(result["geometry"]))

    return gdf


def load_point_extractions(infolder: Path) -> gpd.GeoDataFrame:
    """Load all point extractions from the given folder.
    Parameters
    ----------
    infolder : Path
        path containing extractions for a given collection
    Returns
    -------
    GeoPandas GeoDataFrame
        GeoDataFrame containing all point extractions,
        organized in long format
        (each row represents a single timestep for a single sample)
    """

    dfs = []
    # Get all subfolders
    tiles = [x for x in infolder.iterdir() if x.is_dir()]
    for tile in tiles:
        batches = [x for x in tile.iterdir() if x.is_dir()]
        for batch in batches:
            infile = batch / "point_extractions.geoparquet"
            if infile.exists():
                dfs.append(gpd.read_parquet(infile))

    return pd.concat(dfs)


def visualize_timeseries(
    gdf: gpd.GeoDataFrame,
    outfile: Optional[Path] = None,
    variable: str = "NDVI",
    sample_ids: Optional[List] = None,
):
    """Function to visaulize the timeseries for one variable and one or mulitple samples
    from an extractions GeoDataFrame.
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing all point extractions,
        result from load_point_extractions. Should at least contain a
        column "timestamp" and the variable to visualize.
    outfile : Path, optional
        path to a file to store the visualization, by default None
    variable : str, optional
        the variable within the dataframe to visualize.
        Either the variable should be part of the geodataframe, or should be "NDVI",
        by default "NDVI"
    sample_ids : List, optional
        sample ids for which the time series needs to be visualized,
        by default None meaning all samples will be visualized
    """

    if sample_ids is None:
        sample_ids = gdf["sample_id"].unique()

    fig, ax = plt.subplots()
    for sample_id in sample_ids:
        sample = gdf[gdf["sample_id"] == sample_id]
        sample = sample.sort_values("timestamp")

        if variable == "NDVI":
            sample[variable] = (sample["S2-L2A-B08"] - sample["S2-L2A-B04"]) / (
                sample["S2-L2A-B08"] + sample["S2-L2A-B04"]
            )

        if variable not in sample.columns:
            print(f"Variable {variable} not found in the dataframe")
            return

        ax.plot(sample["timestamp"], sample[variable], label=sample_id)

    plt.xlabel("Date")
    plt.ylabel(variable)
    plt.xticks(rotation=45)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

    if outfile is not None:
        plt.savefig(outfile)
    return
