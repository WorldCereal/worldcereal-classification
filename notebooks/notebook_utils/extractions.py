import logging
import textwrap
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipyleaflet import GeoJSON, Map, WidgetControl, basemaps
from loguru import logger
from openeo_gfmap.manager.job_splitters import load_s2_grid
from prometheo.utils import DEFAULT_SEED
from shapely.geometry import Polygon, box
from tabulate import tabulate
from tqdm import tqdm

from worldcereal.openeo.preprocessing import WORLDCEREAL_BANDS
from worldcereal.rdm_api.rdm_interaction import RDM_DEFAULT_COLUMNS
from worldcereal.utils.legend import ewoc_code_to_label
from worldcereal.utils.refdata import (
    gdf_to_points,
    query_private_extractions,
    query_public_extractions,
)

logging.getLogger("rasterio").setLevel(logging.ERROR)


NODATAVALUE = 65535


def load_dataframe(df_path: Path) -> gpd.GeoDataFrame:
    """Load the input dataframe from the given path.

    Parameters
    ----------
    df_path : Path
        path to the input dataframe
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the
        input dataframe
    """

    if df_path.name.endswith("parquet"):
        return gpd.read_parquet(df_path)
    else:
        return gpd.read_file(df_path)


def prepare_samples_dataframe(
    samples_df_path: Union[str, Path],
    ref_id: str,
) -> gpd.GeoDataFrame:
    """Prepare the samples dataframe for the point extractions.

    Parameters
    ----------
    samples_df_path : Union[str, Path]
        The path to the input dataframe containing the geometries
        for which extractions need to be done
    ref_id : str
        The collection ID.

    Returns
    -------
    gpd.GeoDataFrame
        Processed GeoDataFrame containing the samples

    Raises
    ------
    ValueError
        If the input dataframe is missing essential attributes
    """

    gdf = load_dataframe(Path(samples_df_path))

    # Add ref_id attribute
    gdf["ref_id"] = ref_id

    # Check presence of essential attributes
    missing_attributes = set(RDM_DEFAULT_COLUMNS) - set(gdf.columns)
    if len(missing_attributes) > 0:
        raise ValueError(
            f"Missing essential attributes in the input dataframe: {missing_attributes}"
        )

    # Keep essential attributes only
    gdf = gdf[RDM_DEFAULT_COLUMNS]

    # Convert geometries to points
    logger.info(f"Loaded {len(gdf)} samples from {samples_df_path}")
    gdf = gdf_to_points(gdf)
    if len(gdf) == 0:
        logger.warning("No valid samples left in your dataset.")
    if len(gdf) > 1000:
        logger.warning(
            "More than 1000 samples in your dataset. Extractions will likely consume a considerable amount of credits."
        )

    # Check how many S2 tiles are involved
    gdf = gdf.to_crs(epsg=4326)
    s2_grid = load_s2_grid()
    gdf = gpd.sjoin(
        gdf,
        s2_grid[["tile", "geometry"]],
        predicate="intersects",
    ).drop(columns=["index_right"])
    n_tiles = len(gdf["tile"].unique())
    logger.info(f"Samples cover {n_tiles} S2 tiles.")
    if n_tiles > 50:
        logger.warning(
            "The number of S2 tiles is high. Extractions will take a while..."
        )
    # drop tile attribute again
    gdf = gdf.drop(columns=["tile"])

    return gdf


def _apply_band_scaling(array: np.array, bandname: str) -> np.array:
    """Apply scaling to the band values based on the band name.
    Parameters
    ----------
    array : np.array
        array containing the band values
    bandname : str
        name of the band
    Returns
    -------
    np.array
        array containing the scaled band values
    Raises
    ------
    ValueError
        If the band is not supported
    """

    idx_valid = array != NODATAVALUE
    array = array.astype(np.float32)

    # No scaling for DEM bands
    if bandname in WORLDCEREAL_BANDS["DEM"]:
        pass
    # Divide by 10000 for S2 bands
    elif bandname in WORLDCEREAL_BANDS["SENTINEL2"]:
        array[idx_valid] = array[idx_valid] / 10000
    # Convert to dB for S1 bands
    elif bandname in WORLDCEREAL_BANDS["SENTINEL1"]:
        idx_valid = idx_valid & (array > 0)
        array[idx_valid] = 20 * np.log10(array[idx_valid]) - 83
    # Scale meteo bands by factor 100
    elif bandname in WORLDCEREAL_BANDS["METEO"]:
        array[idx_valid] = array[idx_valid] / 100
    else:
        raise ValueError(f"Unsupported band name {bandname}")

    return array


def load_point_extractions(
    extractions_dir: Path,
    subset=False,
) -> gpd.GeoDataFrame:
    """Load point extractions from the given folder.

    Parameters
    ----------
    extractions_dir : Path
        path containing extractions for a given collection
    subset : bool, optional
        whether to subset the data to reduce the size, by default False

    Returns
    -------
    GeoPandas GeoDataFrame
        GeoDataFrame containing all point extractions,
        organized in long format
        (each row represents a single timestep for a single sample)
    """

    # Look for all extractions in the given folder
    infiles = list(Path(extractions_dir).glob("**/*.geoparquet"))
    # Get rid of merged geoparquet
    infiles = [f for f in infiles if not Path(f).is_dir()]

    if len(infiles) == 0:
        raise FileNotFoundError(f"No point extractions found in {extractions_dir}")
    logger.info(f"Found {len(infiles)} geoparquet files in {extractions_dir}")

    if subset:
        # only load first file
        gdf = gpd.read_parquet(infiles[0])
    else:
        # load all files
        gdf = gpd.read_parquet(infiles)

    return gdf


def get_band_statistics(
    extractions_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Get the band statistics for the point extractions.
    Parameters
    ----------
    extractions_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the point extractions in LONG format.
    Returns
    -------
    pd.DataFrame
        DataFrame containing the band statistics
    """

    # Check if the extractions dataframe is empty
    if extractions_gdf.empty:
        raise ValueError("The extractions dataframe is empty.")

    # Get the band statistics
    band_stats = {}
    for source, bands in WORLDCEREAL_BANDS.items():
        for bandname in bands:
            if bandname in extractions_gdf.columns:
                # count percentage of nodata values
                nodata_count = (extractions_gdf[bandname] == NODATAVALUE).sum()
                nodata_percentage = nodata_count / len(extractions_gdf) * 100
                # Treat missing bands
                if nodata_percentage == 100:
                    logger.warning(
                        f"Band {bandname} has no valid data in the extractions dataframe."
                    )
                    band_stats[bandname] = {
                        "%_nodata": "100.00",
                        "min": "N/A",
                        "max": "N/A",
                        "mean": "N/A",
                        "std": "N/A",
                    }
                    continue
                # Apply scaling
                scaled_values = _apply_band_scaling(
                    extractions_gdf[bandname].values, bandname
                )
                # discard nodata values
                scaled_values = scaled_values[scaled_values != NODATAVALUE]
                # get the statistics and only show 2 decimals
                band_stats[bandname] = {
                    "%_nodata": f"{nodata_percentage:.2f}",
                    "min": f"{np.min(scaled_values):.2f}",
                    "max": f"{np.max(scaled_values):.2f}",
                    "mean": f"{np.mean(scaled_values):.2f}",
                    "std": f"{np.std(scaled_values):.2f}",
                }

    # Convert to DataFrame
    stats_df = pd.DataFrame(band_stats).T
    print("-------------------------------------")
    print("Band statistics:")
    print(tabulate(stats_df, headers="keys", tablefmt="psql", showindex=True))

    return stats_df


def visualize_timeseries(
    extractions_gdf: gpd.GeoDataFrame,
    nsamples: int = 5,
    band: str = "NDVI",
    outfile: Optional[Path] = None,
    sample_ids: Optional[List] = None,
    random_seed: Optional[int] = None,
    crop_label_attr: Optional[str] = "label_full",
) -> None:
    """Function to visaulize the timeseries for one band and random or specific samples
    from an extractions dataframe.

    Parameters
    ----------
    extractions_gdf : gpd.GeoDataFrame
        geodataframe containing extractions in LONG format.
    nsamples : int, optional
        number of random samples to visualize, by default 5.
        Gets overruled if sample_ids are specified.
    band : str, optional
        the band to visualize, by default NDVI.
    outfile : Path, optional
        path to a file to store the visualization, by default None
    sample_ids : List, optional
        sample ids for which the time series needs to be visualized,
        by default None meaning a random subset will be visualized
    random_seed : Optional[int], optional
        Seed for reproducible random sampling of sample_ids when sample_ids is None.
        If None, sampling will be non-deterministic.
    crop_label_attr : Optional[str], optional
        Attribute name of the data containing the crop type labels, by default 'label_full'.

    Returns
    -------
    None
    """

    # Check if the extractions dataframe is empty
    if extractions_gdf.empty:
        raise ValueError("The extractions dataframe is empty.")

    # Check whether we have a valid band name
    supported_bands = [
        value for k, values in WORLDCEREAL_BANDS.items() for value in values
    ]
    supported_bands.append("NDVI")
    if band not in supported_bands:
        raise ValueError(f"Band {band} not found in the extractions dataframe")

    # Check whether we have sufficient data
    available_samples = extractions_gdf["sample_id"].nunique()
    if available_samples < nsamples:
        logger.warning(
            f"Not enough samples in the dataframe to visualize {nsamples} samples. "
            f"Visualizing {available_samples} samples instead."
        )
        nsamples = available_samples

    # Sample the data
    if sample_ids is None:
        sample_ids = extractions_gdf["sample_id"].unique()
        rng = np.random.default_rng(random_seed)
        selected_ids = rng.choice(sample_ids, nsamples, replace=False)
    else:
        selected_ids = sample_ids

    fig, ax = plt.subplots(figsize=(12, 6))

    for sample_id in selected_ids:
        sample = extractions_gdf[extractions_gdf["sample_id"] == sample_id]
        sample = sample.sort_values("timestamp")
        if crop_label_attr is not None:
            crop_label = sample[crop_label_attr].values[0]
        else:
            crop_label = ""

        # Prepare the data to be shown
        if band == "NDVI":
            nir = sample.loc[:, "S2-L2A-B08"].values.astype(np.float32)
            nir[nir == NODATAVALUE] = np.nan
            red = sample.loc[:, "S2-L2A-B04"].values.astype(np.float32)
            red[red == NODATAVALUE] = np.nan
            values = (nir - red) / (nir + red)
        else:
            # scale values
            values = sample.loc[:, band].values
            values = _apply_band_scaling(values, band)
            # get rid of nodata values
            values[values == NODATAVALUE] = np.nan

        # plot
        ax.plot(
            sample["timestamp"],
            values,
            marker="o",
            linestyle="-",
            label=f"{crop_label} ({sample_id})",
        )

    plt.xlabel("Date")
    plt.ylabel(band)
    plt.xticks(rotation=90)
    # put legend underneath the plot
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.subplots_adjust(right=0.75)  # Ensures enough space for the legend
    # add gridlines
    plt.grid()
    plt.tight_layout()
    plt.show()

    if outfile is not None:
        plt.savefig(outfile)
    return


def query_extractions(
    bbox_poly: Optional[Polygon] = None,
    buffer: int = 250000,
    filter_cropland: bool = True,
    include_public: bool = True,
    private_parquet_path: Optional[Path] = None,
    ref_ids: Optional[List[str]] = None,
    crop_types: Optional[List[int]] = None,
    query_collateral_samples: bool = True,
) -> pd.DataFrame:
    """Wrapper function to query both public and private extractions.

    This function queries extraction data (both public and private) using one of three modes:
    1. Spatial discovery: Query by spatial intersection with a polygon (bbox_poly only)
    2. Dataset-specific: Query specific datasets directly (ref_ids only)
    3. Combined filtering: Query specific datasets within a spatial area (both bbox_poly and ref_ids)

    IMPORTANT: You must specify at least one of 'ref_ids' or 'bbox_poly'.

    Parameters
    ----------
    bbox_poly : Optional[Polygon], default=None
        Polygon representing the area of interest in EPSG:4326. Can be used alone for spatial
        discovery or combined with ref_ids to spatially filter specific datasets.
    buffer : int, default=250000
        Buffer to add to the bounding box in meters, by default 250000 (250 km).
        Only used when bbox_poly is provided.
    filter_cropland : bool, default=True
        Whether to filter out non-cropland samples, by default True
    include_public : bool, default=True
        Whether to include public extractions, by default True
    private_parquet_path : Optional[Path], default=None
        Path to a parquet file containing private extractions, by default None
    ref_ids : Optional[List[str]], default=None
        List of specific reference dataset IDs to query. Can be used alone for
        dataset-specific queries or combined with bbox_poly to spatially filter specific datasets.
    crop_types : Optional[List[int]], default=None
        List of crop types to filter on, by default None
        If None, all crop types are included.
    query_collateral_samples : bool, default=True
        Whether to include collateral samples in the query.
        Collateral samples are those samples that were not specifically marked for extraction,
        but fell into the vicinity of such samples during the extraction process. While using
        collateral samples will result in significant increase in amount of samples available for training,
        it will also shift the distribution of the classes.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted samples.

    Raises
    ------
    ValueError
        If neither ref_ids nor bbox_poly is specified.

    Examples
    --------
    # Mode 1: Query datasets in a spatial area (auto-discovery)
    >>> from shapely.geometry import Polygon
    >>> my_area = Polygon([...])  # your area of interest
    >>> data = query_extractions(
    ...     bbox_poly=my_area,
    ...     buffer=50000,  # 50km buffer
    ...     include_public=True,
    ...     private_parquet_path="./extractions/"
    ... )

    # Mode 2: Query specific known datasets (no spatial filtering)
    >>> data = query_extractions(
    ...     ref_ids=["dataset1", "dataset2"],
    ...     include_public=True,
    ...     private_parquet_path="./extractions/"
    ... )

    # Mode 3: Query specific datasets within a spatial area (combined filtering)
    >>> data = query_extractions(
    ...     ref_ids=["dataset1", "dataset2", "dataset3"],
    ...     bbox_poly=my_area,
    ...     buffer=25000,  # 25km buffer
    ...     include_public=True
    ... )
    """

    # Enforce that user must specify at least one of ref_ids or bbox_poly
    if ref_ids is None and bbox_poly is None:
        raise ValueError(
            "You must specify at least one of 'ref_ids' (list of dataset names) OR 'bbox_poly' (spatial area of interest). "
            "Cannot proceed without knowing which datasets to query or which area to query."
        )

    results = []
    extraction_summary = []

    # Query public extractions
    if include_public:
        logger.info("Querying public extractions...")
        public_df = query_public_extractions(
            bbox_poly=bbox_poly,
            buffer=buffer,
            filter_cropland=filter_cropland,
            crop_types=crop_types,
            query_collateral_samples=query_collateral_samples,
            ref_ids=ref_ids,
        )

        if len(public_df) > 0:
            results.append(public_df)

            # Collect dataset information for summary table
            for ref_id in sorted(public_df["ref_id"].unique()):
                ref_data = public_df[public_df["ref_id"] == ref_id]
                extraction_summary.append(
                    {
                        "Source": "Public",
                        "Dataset": ref_id,
                        "Samples": ref_data["sample_id"].nunique(),
                        "Crop Types": ref_data["ewoc_code"].nunique(),
                    }
                )

    # Query private extractions
    if private_parquet_path is not None:
        logger.info("Querying private extractions...")
        private_df = query_private_extractions(
            str(private_parquet_path),
            bbox_poly=bbox_poly,
            filter_cropland=filter_cropland,
            buffer=buffer,
            crop_types=crop_types,
            ref_ids=ref_ids,
        )

        if len(private_df) > 0:
            results.append(private_df)

            # Collect dataset information for summary table
            for ref_id in sorted(private_df["ref_id"].unique()):
                ref_data = private_df[private_df["ref_id"] == ref_id]
                extraction_summary.append(
                    {
                        "Source": "Private",
                        "Dataset": ref_id,
                        "Samples": ref_data["sample_id"].nunique(),
                        "Crop Types": ref_data["ewoc_code"].nunique(),
                    }
                )

    # Now we merge the results together in one dataframe
    if len(results) > 1:
        merged_df = pd.concat(results)
    elif len(results) == 1:
        merged_df = results[0]
    else:
        raise ValueError("No extractions found for the specified criteria.")

    # Translate ewoc codes to labels
    merged_df["label_full"] = ewoc_code_to_label(
        merged_df["ewoc_code"], label_type="full"
    )
    merged_df["sampling_label"] = ewoc_code_to_label(
        merged_df["ewoc_code"], label_type="sampling"
    )

    # Create and display beautified summary
    print("\n" + "=" * 80)
    print("QUERY EXTRACTIONS SUMMARY")
    print("=" * 80)

    if extraction_summary:
        DEFAULT_COL_WIDTH = 40
        summary_df = pd.DataFrame(extraction_summary)
        # Truncate textual columns to a fixed width to avoid overly wide tables
        for col in summary_df.columns:
            if summary_df[col].dtype == object:
                summary_df[col] = (
                    summary_df[col]
                    .astype(str)
                    .apply(
                        lambda s: (
                            (s[: DEFAULT_COL_WIDTH - 1] + "…")
                            if len(s) > DEFAULT_COL_WIDTH
                            else s
                        )
                    )
                )
        print("\nDatasets Retrieved:")
        print(
            tabulate(
                summary_df,
                headers="keys",
                tablefmt="grid",
                showindex=False,
                maxcolwidths=[DEFAULT_COL_WIDTH] * summary_df.shape[1],
            )
        )

        # Overall statistics
        total_samples = merged_df["sample_id"].nunique()
        total_datasets = len(extraction_summary)
        total_crop_types = merged_df["ewoc_code"].nunique()
        unique_crop_groups = sorted(merged_df["sampling_label"].unique())

        # Build full (untruncated) crop groups list and wrap over multiple lines for readability
        crop_groups_full = ", ".join(unique_crop_groups)
        # Wrap at 40 characters per line (matches default column width) but do NOT truncate
        wrapped_crop_groups = "\n".join(textwrap.wrap(crop_groups_full, width=40))
        stats_table = [
            ["Total Samples", f"{total_samples:,}"],
            ["Total Datasets", f"{total_datasets}"],
            ["Unique Crop Types", f"{total_crop_types}"],
            ["Crop Groups", wrapped_crop_groups],
        ]

        print("\nOverall Statistics:")
        print(
            tabulate(
                stats_table,
                headers=["Metric", "Value"],
                tablefmt="grid",
                maxcolwidths=[None, 40],
            )
        )

    print("=" * 80 + "\n")

    if total_crop_types <= 1:
        logger.warning(
            "Not enough crop types found in the extracted data to train a model. "
            "Expand your area of interest or add more reference data."
        )

    # Explictily drop column "feature_index" if it exists
    if "feature_index" in merged_df.columns:
        merged_df = merged_df.drop(columns=["feature_index"])

    return merged_df


def retrieve_extractions_extent(
    include_crop_types: bool = True,
) -> tuple[gpd.GeoDataFrame, Map]:
    """Function to retrieve the extents of publicly available WorldCereal reference datasets
    with satellite extractions.
    Parameters
    ----------
    include_crop_types : bool, optional
        Whether to only include datasets with crop type information, by default True
        Returns
        -------
        tuple[gpd.GeoDataFrame, Map]
                A tuple containing:
                - GeoDataFrame with the extents of publicly available WorldCereal reference datasets
                    with satellite extractions.
                - An ipyleaflet Map widget showing those extents.
    """
    # Download the file holding all extents of publicly available WorldCereal reference datasets with satellite extractions
    local_file = Path("./download/worldcereal_public_extractions_extent.parquet")
    local_file.parent.mkdir(parents=True, exist_ok=True)
    url = "https://s3.waw3-1.cloudferro.com/swift/v1/geoparquet/worldcereal_public_extractions_extent.parquet"
    urllib.request.urlretrieve(url, local_file)

    # Read with geopandas
    gdf = gpd.read_parquet(local_file)
    # Some environments may return a tuple (data, metadata)
    if isinstance(gdf, tuple):
        gdf = gdf[0]
    # Ensure we have a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" not in gdf.columns:
            raise ValueError(
                "Public extractions extent parquet lacks a 'geometry' column."
            )
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    # Ignore global extents
    gdf = gdf[~gdf.ref_id.str.contains("GLO")]
    # Optionally filter on crop types
    if include_crop_types:
        # Drop datasets with "_100" or "_101" in the ref_id
        gdf = gdf[~gdf.ref_id.str.contains("_100|_101")]

    # Visualization of the map with hover functionality using ipyleaflet
    extent_gdf = gdf.to_crs(epsg=4326)
    info = widgets.HTML(value="<b>ref_id:</b> hover over a dataset")
    info_control = WidgetControl(widget=info, position="topright")

    extent_map = Map(
        basemap=basemaps.CartoDB.Positron,
        center=(0, 0),
        zoom=1,
        scroll_wheel_zoom=True,
    )

    geojson = GeoJSON(
        data=extent_gdf.__geo_interface__,
        style={
            "color": "#3112bb",
            "weight": 1,
            "fillColor": "#3b82f6",
            "fillOpacity": 0.15,
        },
        hover_style={
            "color": "#2563eb",
            "weight": 2,
            "fillOpacity": 0.25,
        },
    )

    def handle_hover(feature, **kwargs):
        ref_id = feature.get("properties", {}).get("ref_id", "N/A")
        info.value = f"<b>ref_id:</b> {ref_id}"

    def handle_mouseout(**kwargs):
        info.value = "<b>ref_id:</b> hover over a dataset"

    geojson.on_hover(handle_hover)
    geojson.on_mouseout(handle_mouseout)

    extent_map.add_control(info_control)
    extent_map.add_layer(geojson)

    return gdf, extent_map


def generate_extractions_extent(
    parquet_folder: Path,
) -> gpd.GeoDataFrame:
    """Function to generate the extents of WorldCereal reference datasets
    with satellite extractions.
    Parameters
    ----------
    parquet_folder : Path
        Path to the folder containing all parquet files with extractions.
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the extents of publicly available WorldCereal reference datasets
        with satellite extractions.
    """

    # Look for all parquet files in the given folder
    parquet_files = list(parquet_folder.glob("**/*.parquet"))
    # Get rid of merged geoparquet
    parquet_files = [f for f in parquet_files if not Path(f).is_dir()]

    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {parquet_folder}")
    logger.info(f"Found {len(parquet_files)} parquet files in {parquet_folder}")

    # Initialize empty geodataframe to hold the extents
    ref_id_extent = gpd.GeoDataFrame(columns=["ref_id", "geometry"], crs="EPSG:4326")

    for infile in tqdm(parquet_files):
        gdf = gpd.read_parquet(infile)
        ref_id = gdf.ref_id.iloc[0]
        assert ref_id in str(infile)
        bbox = box(*gdf.total_bounds)

        ref_id_extent = pd.concat(
            [
                pd.DataFrame([[ref_id, bbox]], columns=ref_id_extent.columns),
                ref_id_extent,
            ],
            ignore_index=True,
        )

    # Transform back to geodataframe
    ref_id_extent = gpd.GeoDataFrame(ref_id_extent, crs="EPSG:4326")

    return ref_id_extent


def find_extractions_in_area(
    extent_gdf: gpd.GeoDataFrame,
    bbox_poly: Polygon,
    include_crop_types: bool = True,
) -> List[str]:
    """Find extraction datasets that overlap with a given spatial extent.

    Parameters
    ----------
    extent_gdf : gpd.GeoDataFrame
        GeoDataFrame containing the extents of available extraction datasets.
        Should have 'ref_id' and 'geometry' columns.
    bbox_poly : Polygon
        Polygon representing the area of interest to search for overlapping datasets.
    include_crop_types : bool, optional
        Whether to only include datasets with crop type information, by default True.
        This filters out datasets with "_100" or "_101" in the ref_id.

    Returns
    -------
    List[str]
        List of ref_ids (dataset names) that spatially overlap with the given bbox_poly.
    """

    # Make a copy to avoid modifying the original
    gdf = extent_gdf.copy()

    # Optionally filter on crop types
    if include_crop_types:
        # Drop datasets with "_100" or "_101" in the ref_id
        gdf = gdf[~gdf.ref_id.str.contains("_100|_101")]

    # Ensure both geometries are in the same CRS
    if gdf.crs != "EPSG:4326":
        logger.info(f"Reprojecting extent_gdf from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    # Create a GeoDataFrame for the bbox_poly for spatial operations
    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_poly], crs="EPSG:4326")

    # Find intersecting datasets
    intersecting = gpd.sjoin(
        gdf[["ref_id", "geometry"]], bbox_gdf, predicate="intersects", how="inner"
    )

    # Get unique ref_ids
    ref_ids_list = intersecting["ref_id"].unique().tolist()

    logger.info(
        f"Found {len(ref_ids_list)} datasets overlapping with the specified area."
    )

    return ref_ids_list


def map_classes(
    df: pd.DataFrame,
    class_mappings_csv: Path,
    target_col: str = "finetune_class",
) -> pd.DataFrame:

    # Get class mappings from csv
    if class_mappings_csv.exists():
        class_mappings = pd.read_csv(class_mappings_csv, sep=";", header=0)
        # Remove rows with missing values in target_col
        class_mappings = class_mappings.dropna(subset=[target_col])
        # Convert ewoc_code to int64
        class_mappings["ewoc_code"] = (
            class_mappings["ewoc_code"].str.replace("-", "").astype(np.int64)
        )

        # Convert to dictionary
        class_mappings = dict(
            zip(class_mappings["ewoc_code"], class_mappings[target_col])
        )
    else:
        raise FileNotFoundError(f"Class mappings file not found: {class_mappings_csv}")

    # Map classes
    df.loc[:, target_col] = df["ewoc_code"].map(
        {int(k): v for k, v in class_mappings.items()}
    )
    # Remove rows with missing target_col values after mapping
    df = df.dropna(subset=[target_col])

    return df


def _classify_ref_ids(
    ref_ids: List[str], include_public: bool, private_parquet_path: Optional[Path]
) -> tuple[List[str], List[str]]:
    """
    Helper function to classify provided ref_ids as public vs private.

    Parameters
    ----------
    ref_ids : List[str]
        List of ref_ids to classify
    include_public : bool
        Whether to check for public datasets
    private_parquet_path : Optional[Path]
        Path to private extractions

    Returns
    -------
    tuple[List[str], List[str]]
        Lists of (public_ref_ids, private_ref_ids) from the provided ref_ids
    """
    available_public_ref_ids = []
    available_private_ref_ids = []

    if include_public:
        try:
            logger.info(
                "Checking which datasets are available in public extractions..."
            )
            public_extent_gdf, _ = retrieve_extractions_extent(include_crop_types=True)
            all_public_ref_ids = set(public_extent_gdf["ref_id"].unique())
            available_public_ref_ids = [
                ref_id for ref_id in ref_ids if ref_id in all_public_ref_ids
            ]
            logger.info(
                f"Found {len(available_public_ref_ids)} datasets available in public data: {sorted(available_public_ref_ids)}"
            )
        except Exception as e:
            logger.warning(f"Could not determine public datasets: {e}")

    if private_parquet_path is not None:
        try:
            logger.info(
                "Checking which datasets are available in private extractions..."
            )
            private_extent_gdf = generate_extractions_extent(
                parquet_folder=private_parquet_path
            )
            all_private_ref_ids = set(private_extent_gdf["ref_id"].unique())
            available_private_ref_ids = [
                ref_id for ref_id in ref_ids if ref_id in all_private_ref_ids
            ]
            logger.info(
                f"Found {len(available_private_ref_ids)} datasets available in private data: {sorted(available_private_ref_ids)}"
            )
        except Exception as e:
            logger.warning(f"Could not determine private datasets: {e}")

    return available_public_ref_ids, available_private_ref_ids


def _process_dataset_sampling(
    dataset_data: pd.DataFrame,
    ref_id: str,
    crop_types: Optional[List[int]],
    sample_size: int,
    class_mappings_csv: Optional[Path],
    random_state: int,
    sampled_dfs: List[pd.DataFrame],
    sampling_summary: List[dict],
) -> None:
    """
    Helper function to process sampling for a single dataset.

    Parameters
    ----------
    dataset_data : pd.DataFrame
        The extraction data for a single dataset
    ref_id : str
        The reference dataset ID
    crop_types : Optional[List[int]]
        List of crop types to sample, or None to use all available
    sample_size : int
        Number of samples per crop type
    class_mappings_csv : Optional[Path]
        Path to class mappings file
    random_state : int
        Random state for sampling
    sampled_dfs : List[pd.DataFrame]
        List to append sampled data to (modified in place)
    sampling_summary : List[dict]
        List to append sampling summary to (modified in place)
    """
    if dataset_data.empty:
        logger.warning(f"No data found for dataset: {ref_id}")
        if crop_types is not None:
            for crop_type in crop_types:
                sampling_summary.append(
                    {
                        "ref_id": ref_id,
                        "crop_type": crop_type,
                        "available": 0,
                        "sampled": 0,
                    }
                )
        return

    # Map classes if mapping is provided
    if class_mappings_csv is not None:
        target_col = "finetune_class"
        dataset_data = map_classes(
            dataset_data,
            class_mappings_csv=class_mappings_csv,
            target_col=target_col,
        )
    else:
        target_col = "ewoc_code"

    # If crop_types not specified, get them from this dataset
    if crop_types is None:
        dataset_crop_types = dataset_data[target_col].unique().tolist()
        logger.info(
            f"Using crop types from dataset {ref_id}: {len(dataset_crop_types)} types"
        )
    else:
        dataset_crop_types = crop_types

    # Get a list of unique sample id's with crop type information
    available_data = dataset_data.drop_duplicates(subset=["sample_id"])

    # Process each crop type for this dataset
    for crop_type in dataset_crop_types:
        # Filter data for this specific crop type
        subset = available_data[available_data[target_col] == crop_type]

        if subset.empty:
            sampling_summary.append(
                {
                    "ref_id": ref_id,
                    "crop_type": crop_type,
                    "available": 0,
                    "sampled": 0,
                }
            )
            continue

        # Sample the requested number (or all available if fewer)
        n_available = len(subset)
        n_to_sample = min(sample_size, n_available)

        # Perform sampling
        sampled_ids = subset.sample(
            n=n_to_sample, random_state=random_state
        ).sample_id.tolist()

        # Get the full time series data for the sampled IDs
        sampled_subset = dataset_data[dataset_data.sample_id.isin(sampled_ids)].copy()

        sampled_dfs.append(sampled_subset)
        sampling_summary.append(
            {
                "ref_id": ref_id,
                "crop_type": crop_type,
                "available": n_available,
                "sampled": n_to_sample,
            }
        )


def sample_extractions(
    bbox_poly: Optional[Polygon] = None,
    buffer: int = 250000,
    filter_cropland: bool = True,
    include_public: bool = True,
    private_parquet_path: Optional[Path] = None,
    ref_ids: Optional[List[str]] = None,
    crop_types: Optional[List[int]] = None,
    sample_size: int = 100,
    class_mappings_csv: Optional[Path] = None,
    query_collateral_samples: bool = True,
    random_state: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Sample a specified number of samples for each ref_id and crop_type combination from both public and private extractions.

    This function queries extraction data (both public and private) dataset by dataset and samples a fixed number of samples
    for each combination of reference dataset (ref_id) and crop type (ewoc_code).
    This approach is more memory-efficient than loading all data at once and more efficient than querying entire regions
    when specific datasets are targeted.

    IMPORTANT: You must specify at least one of 'ref_ids' or 'bbox_poly'. Three modes are supported:
    1. Dataset-specific: Use ref_ids only when you know specific dataset names to sample from
    2. Spatial discovery: Use bbox_poly only to automatically discover and sample from all datasets
       that spatially intersect with your area of interest (uses extent-based discovery for efficiency)
    3. Combined filtering: Use both ref_ids and bbox_poly to sample from specific datasets within
       a spatial area (most efficient when you want specific datasets in a region)

    Parameters
    ----------
    bbox_poly : Optional[Polygon], default=None
        Polygon representing the area of interest. Can be used alone for spatial discovery
        or combined with ref_ids to filter specific datasets within the spatial area.
    buffer : int, default=250000
        Buffer to add to the bounding box in meters, by default 250000 (250 km).
        Only used when bbox_poly is specified.
    filter_cropland : bool, default=True
        Whether to filter for temporary cropland samples only (WorldCereal codes 1100000000-1115000000,
        excluding fallow classes). Should be True when using data for croptype classification.
    include_public : bool, default=True
        Whether to include public extractions.
    private_parquet_path : Optional[Path], default=None
        Path to a parquet file containing private extractions.
    ref_ids : Optional[List[str]], default=None
        List of specific reference dataset IDs to sample from. Can be used alone for
        dataset-specific sampling or combined with bbox_poly to spatially filter specific datasets.
    crop_types : Optional[List[int]], default=None
        List of crop type codes (ewoc_codes) to sample. If None, all available crop types will be used.
    sample_size : int, default=100
        Number of samples to extract for each ref_id and crop_type combination.
        If a combination has fewer samples than requested, all available samples will be returned.
    class_mappings_csv : Optional[Path], default=None
        Path to CSV file containing class mappings for remapping crop types.
    query_collateral_samples : bool, default=True
        Whether to include collateral samples in the query.
    random_state : int, default=DEFAULT_SEED
        Random state for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        A GeoPandas DataFrame containing the sampled data with columns indicating
        the ref_id, ewoc_code, and other extraction attributes.

    Raises
    ------
    ValueError
        If neither ref_ids nor bbox_poly is specified.

    Examples
    --------
    # Mode 1: Sample from specific known datasets (no spatial filtering)
    >>> sampled_data = sample_extractions(
    ...     ref_ids=["dataset1", "dataset2"],
    ...     include_public=True,
    ...     private_parquet_path="./extractions/",
    ...     crop_types=[1101000010, 1102000010],  # wheat, maize
    ...     sample_size=100
    ... )

    # Mode 2: Sample from datasets in a spatial area (auto-discovery)
    >>> from shapely.geometry import Polygon
    >>> my_area = Polygon([...])  # your area of interest
    >>> sampled_data = sample_extractions(
    ...     bbox_poly=my_area,
    ...     buffer=50000,  # 50km buffer
    ...     include_public=True,
    ...     private_parquet_path="./extractions/",
    ...     sample_size=50
    ... )

    # Mode 3: Sample from specific datasets within a spatial area (combined filtering)
    >>> sampled_data = sample_extractions(
    ...     ref_ids=["dataset1", "dataset2", "dataset3"],
    ...     bbox_poly=my_area,
    ...     buffer=25000,  # 25km buffer
    ...     include_public=True,
    ...     sample_size=75
    ... )
    >>> print(sampled_data.groupby(['ref_id', 'ewoc_code']).size())
    """

    logger.info("######## Sampling extractions...")

    # Enforce that user must specify at least one of ref_ids or bbox_poly
    if ref_ids is None and bbox_poly is None:
        raise ValueError(
            "You must specify at least one of 'ref_ids' (list of dataset names) OR 'bbox_poly' (spatial area of interest). "
            "Cannot proceed without knowing which datasets to sample from or which area to query."
        )

    # Determine which ref_ids are available in public vs private sources
    available_public_ref_ids = []
    available_private_ref_ids = []

    if ref_ids is None:
        # Mode 1: Spatial discovery - find all datasets that intersect with the provided area
        logger.info(
            "No ref_ids specified. Discovering datasets that intersect with the provided area..."
        )

        # Find public datasets that intersect with the area
        if include_public:
            try:
                logger.info("Retrieving public extractions extent...")
                public_extent_gdf, _ = retrieve_extractions_extent(
                    include_crop_types=True
                )
                available_public_ref_ids = find_extractions_in_area(
                    extent_gdf=public_extent_gdf,
                    bbox_poly=bbox_poly,
                    include_crop_types=True,
                )
            except Exception as e:
                logger.warning(f"Could not retrieve public extractions extent: {e}")

        # Find private datasets that intersect with the area
        if private_parquet_path is not None:
            try:
                logger.info("Generating private extractions extent...")
                private_extent_gdf = generate_extractions_extent(
                    parquet_folder=private_parquet_path
                )
                available_private_ref_ids = find_extractions_in_area(
                    extent_gdf=private_extent_gdf,
                    bbox_poly=bbox_poly,
                    include_crop_types=True,
                )
            except Exception as e:
                logger.warning(f"Could not generate private extractions extent: {e}")

        if not available_public_ref_ids and not available_private_ref_ids:
            logger.warning("No datasets found that intersect with the specified area.")
            return pd.DataFrame()

        logger.info(
            f"Total datasets found in area: {len(available_public_ref_ids) + len(available_private_ref_ids)}"
        )

    elif ref_ids is not None and bbox_poly is not None:
        # Mode 2: Combined filtering - filter provided ref_ids by spatial intersection
        logger.info(
            f"Filtering {len(ref_ids)} provided datasets by spatial intersection..."
        )

        # Get extent data and filter by provided ref_ids, then find spatial intersections
        public_ref_ids = []
        private_ref_ids = []

        if include_public:
            try:
                logger.info("Checking spatial intersection for public datasets...")
                public_extent_gdf, _ = retrieve_extractions_extent(
                    include_crop_types=True
                )
                public_extent_gdf = public_extent_gdf[
                    public_extent_gdf["ref_id"].isin(ref_ids)
                ]
                public_ref_ids = find_extractions_in_area(
                    extent_gdf=public_extent_gdf,
                    bbox_poly=bbox_poly,
                    include_crop_types=True,
                )
                logger.info(
                    f"Found {len(public_ref_ids)} public datasets from the provided list that intersect with the area"
                )
            except Exception as e:
                logger.warning(
                    f"Could not check public datasets spatial intersection: {e}"
                )

        if private_parquet_path is not None:
            try:
                logger.info("Checking spatial intersection for private datasets...")
                private_extent_gdf = generate_extractions_extent(
                    parquet_folder=private_parquet_path
                )
                private_extent_gdf = private_extent_gdf[
                    private_extent_gdf["ref_id"].isin(ref_ids)
                ]
                private_ref_ids = find_extractions_in_area(
                    extent_gdf=private_extent_gdf,
                    bbox_poly=bbox_poly,
                    include_crop_types=True,
                )
                logger.info(
                    f"Found {len(private_ref_ids)} private datasets from the provided list that intersect with the area"
                )
            except Exception as e:
                logger.warning(
                    f"Could not check private datasets spatial intersection: {e}"
                )

        available_public_ref_ids = public_ref_ids
        available_private_ref_ids = private_ref_ids

        if not available_public_ref_ids and not available_private_ref_ids:
            logger.warning(
                f"None of the provided datasets ({ref_ids}) intersect with the specified area."
            )
            return pd.DataFrame()

    else:
        # Mode 3: Dataset-specific - user provided ref_ids without spatial filtering
        logger.info(
            f"Determining which of the {len(ref_ids)} provided datasets are public vs private..."
        )
        available_public_ref_ids, available_private_ref_ids = _classify_ref_ids(
            ref_ids, include_public, private_parquet_path
        )

        if not available_public_ref_ids and not available_private_ref_ids:
            logger.warning(
                f"None of the provided datasets ({ref_ids}) were found in the available data sources."
            )
            return pd.DataFrame()

    # If crop_types not specified, we'll determine them from each dataset individually
    if crop_types is None:
        logger.info(
            "No crop_types specified. Will determine available crop types from each dataset individually"
        )

    # Prepare outputs
    sampled_dfs = []
    sampling_summary = []

    # Apply spatial filtering when bbox_poly is provided
    spatial_filter = bbox_poly
    spatial_buffer = buffer if spatial_filter is not None else 0

    # Prepare list of crop types for the query
    if class_mappings_csv is not None:
        crop_types_for_query = None  # Query all, will map later
    else:
        crop_types_for_query = crop_types

    # Process public datasets
    for ref_id in available_public_ref_ids:
        logger.info(f"Processing public dataset: {ref_id}")

        public_df = query_public_extractions(
            bbox_poly=spatial_filter,
            buffer=spatial_buffer,
            filter_cropland=filter_cropland,
            crop_types=crop_types_for_query,
            query_collateral_samples=query_collateral_samples,
            ref_ids=[ref_id],  # Filter for this specific ref_id at query level
        )

        _process_dataset_sampling(
            dataset_data=public_df,
            ref_id=ref_id,
            crop_types=crop_types,
            sample_size=sample_size,
            class_mappings_csv=class_mappings_csv,
            random_state=random_state,
            sampled_dfs=sampled_dfs,
            sampling_summary=sampling_summary,
        )

    # Process private datasets
    for ref_id in available_private_ref_ids:
        logger.info(f"Processing private dataset: {ref_id}")

        private_df = query_private_extractions(
            str(private_parquet_path),
            bbox_poly=spatial_filter,
            filter_cropland=filter_cropland,
            buffer=spatial_buffer,
            crop_types=crop_types_for_query,
            ref_ids=[ref_id],  # Filter for this specific ref_id
        )

        _process_dataset_sampling(
            dataset_data=private_df,
            ref_id=ref_id,
            crop_types=crop_types,
            sample_size=sample_size,
            class_mappings_csv=class_mappings_csv,
            random_state=random_state,
            sampled_dfs=sampled_dfs,
            sampling_summary=sampling_summary,
        )

    if not sampled_dfs:
        logger.warning(
            "No samples could be extracted for any of the requested combinations."
        )
        return pd.DataFrame()

    # Combine all sampled data
    result = pd.concat(sampled_dfs, ignore_index=True)

    # Create and display beautified sampling summary
    summary_df = pd.DataFrame(sampling_summary)
    total_sampled = summary_df["sampled"].sum()

    print("\n" + "=" * 80)
    print("SAMPLE EXTRACTIONS SUMMARY")
    print("=" * 80)

    if not summary_df.empty:
        # Prepare detailed sampling table
        display_summary = []
        all_processed_ref_ids = available_public_ref_ids + available_private_ref_ids

        for ref_id in all_processed_ref_ids:
            ref_summary = summary_df[summary_df["ref_id"] == ref_id]
            if not ref_summary.empty:
                # Determine source type
                source = "Public" if ref_id in available_public_ref_ids else "Private"

                # Calculate totals for this dataset
                total_available = ref_summary["available"].sum()
                total_sampled_ref = ref_summary["sampled"].sum()
                crop_types_with_data = (ref_summary["sampled"] > 0).sum()
                total_crop_types = len(ref_summary)

                display_summary.append(
                    {
                        "Source": source,
                        "Dataset": ref_id,
                        "Available Samples": f"{total_available:,}",
                        "Sampled": f"{total_sampled_ref:,}",
                        "Crop Types": f"{crop_types_with_data}/{total_crop_types}",
                    }
                )

        print("\nSampling Results by Dataset:")
        print(
            tabulate(display_summary, headers="keys", tablefmt="grid", showindex=False)
        )

        # Overall statistics
        total_datasets = len(all_processed_ref_ids)
        total_available = summary_df["available"].sum()
        unique_crop_types = len(
            summary_df[summary_df["sampled"] > 0]["crop_type"].unique()
        )
        sampling_efficiency = (
            (total_sampled / total_available * 100) if total_available > 0 else 0
        )

        stats_table = [
            ["Total Datasets Processed", f"{total_datasets}"],
            ["Total Available Samples", f"{total_available:,}"],
            ["Total Sampled", f"{total_sampled:,}"],
            ["Unique Crop Types Sampled", f"{unique_crop_types}"],
            ["Sampling Efficiency", f"{sampling_efficiency:.1f}%"],
        ]

        print("\nOverall Sampling Statistics:")
        print(tabulate(stats_table, headers=["Metric", "Value"], tablefmt="grid"))

    print("=" * 80 + "\n")

    return result


def report_extractions_content(df: pd.DataFrame, crop_attribute: str) -> None:
    """
    Report the content of a dataframe containing private extractions.

    This function prints a summary of the number of unique samples, reference datasets,
    and crop types present in the provided dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        A GeoPandas DataFrame containing private extraction data with columns
        'sample_id', 'ref_id', and 'ewoc_code'.

    Returns
    -------
    None
        This function prints the summary to the console.

    Examples
    --------
    >>> report_extractions_content(sampled_data)
    Total unique samples: 1500
    Unique reference datasets: 3
    Unique crop types: 5
    """

    if df.empty:
        logger.warning("The provided dataframe is empty.")
        return

    # Report total unique samples
    df = df[["sample_id", "valid_time", "ref_id", crop_attribute]]
    df = df.drop_duplicates(subset=["sample_id"])
    logger.info(f"Total unique samples: {len(df)}")

    # Report ref_id composition
    unique_ref_ids = df["ref_id"].nunique()
    logger.info(f"Unique reference datasets: {unique_ref_ids}")
    logger.info(df["ref_id"].value_counts())

    # check crop types available
    logger.info(df[crop_attribute].value_counts())
