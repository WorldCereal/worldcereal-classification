import logging
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from openeo_gfmap.manager.job_splitters import load_s2_grid

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

BANDS = {
    "S2-L2A": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"],
    "S1-SIGMA0": ["VV", "VH"],
    "DEM": ["slope", "elevation"],
    "AGERA5": ["PRECIP", "TMEAN"],
}

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

    # Check presence of essential attributes
    missing_attributes = set(RDM_ATTRIBUTES) - set(gdf.columns)
    if len(missing_attributes) > 0:
        raise ValueError(
            f"Missing essential attributes in the input dataframe: {missing_attributes}"
        )

    # Keep essential attributes only
    gdf = gdf[RDM_ATTRIBUTES]

    # Add ref_id attribute
    gdf["ref_id"] = ref_id

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
    if bandname in BANDS["DEM"]:
        pass
    # Divide by 10000 for S2 bands
    elif bandname.startswith("S2-L2A"):
        array[idx_valid] = array[idx_valid] / 10000
    # Convert to dB for S1 bands
    elif bandname.startswith("S1-SIGMA0"):
        idx_valid = idx_valid & (array > 0)
        array[idx_valid] = 20 * np.log10(array[idx_valid]) - 83
    # Scale meteo bands by factor 100
    elif bandname.startswith("AGERA5"):
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

    # Get the band statistics
    band_stats = {}
    for sensor, bands in BANDS.items():
        for band in bands:
            if sensor == "DEM":
                bandname = band
            else:
                bandname = f"{sensor}-{band}"
            if bandname in extractions_gdf.columns:
                # count percentage of nodata values
                nodata_count = (extractions_gdf[bandname] == NODATAVALUE).sum()
                nodata_percentage = nodata_count / len(extractions_gdf) * 100
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

    print(
        f"""
        -------------------------------------
        Band statistics:
        -------------------------------------
        {stats_df.to_string(index=True, header=True)}

        """
    )

    return stats_df


def visualize_timeseries(
    extractions_gdf: gpd.GeoDataFrame,
    nsamples: int = 5,
    band: str = "NDVI",
    outfile: Optional[Path] = None,
    sample_ids: Optional[List] = None,
):
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
    """

    # Check whether we have a valid band name
    supported_bands = [
        f"{k}-{value}" for k, values in BANDS.items() for value in values
    ]
    supported_bands.append("NDVI")
    if band not in supported_bands:
        raise ValueError(f"Band {band} not found in the extractions dataframe")

    # Sample the data
    if sample_ids is None:
        sample_ids = extractions_gdf["sample_id"].unique()
        selected_ids = np.random.choice(sample_ids, nsamples, replace=False)
    else:
        selected_ids = sample_ids

    fig, ax = plt.subplots(figsize=(12, 6))

    for sample_id in selected_ids:
        sample = extractions_gdf[extractions_gdf["sample_id"] == sample_id]
        sample = sample.sort_values("timestamp")

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
        ax.plot(sample["timestamp"], values, marker="o", linestyle="-", label=sample_id)

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
