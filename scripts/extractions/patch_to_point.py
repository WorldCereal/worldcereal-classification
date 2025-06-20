import argparse
import json
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
import pystac_client
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import MultiBackendJobManager
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.utils.catalogue import UncoveredS1Exception, select_s1_orbitstate_vvvh
from pandas.core.dtypes.dtypes import CategoricalDtype

from worldcereal.extract.patch_to_point_worldcereal import (
    create_job_patch_to_point_worldcereal,
    get_label_points,
)
from worldcereal.extract.utils import S2_GRID, upload_geoparquet_artifactory
from worldcereal.rdm_api.rdm_interaction import RDM_DEFAULT_COLUMNS


class PatchToPointJobManager(MultiBackendJobManager):
    def on_job_done(self, job: BatchJob, row):
        logger.info(f"Job {job.job_id} completed")
        output_file = generate_output_path_point_worldcereal(self._root_dir, 0, row)
        job.get_results().download_file(target=output_file, name="timeseries.parquet")

        job_metadata = job.describe()
        metadata_path = output_file.parent / f"job_{job.job_id}.json"
        self.ensure_job_dir_exists(job.job_id)

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(job_metadata, f, ensure_ascii=False)

        post_job_action(output_file)
        logger.success("Job completed")


def create_job_dataframe(ref_id, ground_truth_file=None):
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
            if current_start_date > epsg_codes[epsg]["start_date"]:
                epsg_codes[epsg]["start_date"] = current_start_date
            if current_end_date < epsg_codes[epsg]["end_date"]:
                epsg_codes[epsg]["end_date"] = current_end_date

    # Initialize job dataframe for patch to point
    rows = []

    logger.info(f"Found {len(epsg_codes)} unique EPSG codes in STAC collection.")

    for epsg in epsg_codes.keys():
        # We assume identical start and end date for the entire ref_id
        start_date = epsg_codes[epsg]["start_date"]
        end_date = epsg_codes[epsg]["end_date"]

        # ensure start date is 1st day of month, end date is last day of month
        # Start a month later and end a month earlier to ensure the extractions cover this.
        start_date = (start_date + pd.Timedelta(days=31)).replace(day=1)
        end_date = (
            end_date.replace(day=1) - pd.Timedelta(days=31) + pd.offsets.MonthEnd(0)
        )

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
        gdf = get_label_points(row, ground_truth_file=row["ground_truth_file"])
        gdf["ref_id"] = (
            row.ref_id
        )  # Overwrite due to current back in automatic assignment

        if gdf.empty:
            logger.warning(f"No samples found for {row.epsg} and {row.ref_id}")
            continue
        else:
            logger.info(f"Found {len(gdf)} samples for {row.epsg} and {row.ref_id}")

        # Keep essential attributes only
        gdf = gdf[RDM_DEFAULT_COLUMNS]

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


def post_job_action(parquet_file):
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
    ref_id = gdf["ref_id"][0]
    year = int(ref_id.split("_")[0])
    gdf["year"] = year

    # Make sure we remove the timezone information from the timestamp
    gdf["timestamp"] = gdf["timestamp"].dt.tz_localize(None)

    required_attributes = {
        "feature_index": np.int64,
        "sample_id": str,
        "ref_id": CategoricalDtype(categories=[ref_id], ordered=False),
        "timestamp": "datetime64[ns]",
        "S2-L2A-B02": np.uint16,
        "S2-L2A-B03": np.uint16,
        "S2-L2A-B04": np.uint16,
        "S2-L2A-B05": np.uint16,
        "S2-L2A-B06": np.uint16,
        "S2-L2A-B07": np.uint16,
        "S2-L2A-B08": np.uint16,
        "S2-L2A-B8A": np.uint16,
        "S2-L2A-B11": np.uint16,
        "S2-L2A-B12": np.uint16,
        "S1-SIGMA0-VH": np.uint16,
        "S1-SIGMA0-VV": np.uint16,
        "slope": np.uint16,
        "elevation": np.uint16,
        "AGERA5-PRECIP": np.uint16,
        "AGERA5-TMEAN": np.uint16,
        "lon": np.float64,
        "lat": np.float64,
        "geometry": "geometry",
        "tile": str,
        "h3_l3_cell": str,
        "start_date": str,
        "end_date": str,
        "year": np.int64,
        "valid_time": str,
        "ewoc_code": np.int64,
        "irrigation_status": np.int64,
        "quality_score_lc": np.int64,
        "quality_score_ct": np.int64,
        "extract": np.int64,
    }

    # Select required attributes and cast to dtypes
    gdf = gdf[required_attributes.keys()]
    gdf = gdf.astype(required_attributes)

    gdf.to_parquet(parquet_file, index=False)


def generate_output_path_point_worldcereal(
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


def merge_individual_parquet_files(
    parquet_files: List[Union[Path, str]],
) -> gpd.GeoDataFrame:
    """
    Merge individual parquet files into a single GeoDataFrame.

    Parameters
    ----------
    parquet_files : list of Union[Path, str]
        List of paths to individual parquet files.

    Returns
    -------
    gpd.GeoDataFrame
        Merged GeoDataFrame containing data from all the parquet files.

    Raises
    ------
    ValueError
        If more than 25% of the rows have missing attributes.
    """

    seen_ids: set[str] = set()
    gdfs = []

    # Iterate over each parquet file manually to remove any
    # duplicate sample_ids across files
    for file in parquet_files:
        gdf = gpd.read_parquet(file)
        # Keep only rows with unseen sample_ids
        gdf_filtered = gdf[~gdf["sample_id"].isin(seen_ids)].copy()

        # Update the seen set
        seen_ids.update(gdf_filtered["sample_id"])

        gdfs.append(gdf_filtered)

    # Concatenate after all filtering is done
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    # Check for missing attributes based on one columns
    missing_attrs = gdf["start_date"].isnull()
    if missing_attrs.sum() > 0.25 * len(gdf):
        error_msg = (
            r"More than 25% of the rows have missing attributes. "
            "Please check extractions! No merged parquet will be generated."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif missing_attrs.any():
        logger.warning(
            f"Missing attributes in {missing_attrs.sum()} rows. "
            "This may indicate an issue during extraction. Rows are removed"
        )
        gdf = gdf[~missing_attrs]

    # Make sure we remove the timezone information from the timestamp
    gdf["timestamp"] = gdf["timestamp"].dt.tz_localize(None)

    return gdf


def main(
    connection,
    ref_id,
    ground_truth_file,
    root_folder,
    period="month",
    restart_failed=False,
):
    """
    Main function to orchestrate patch-to-point extractions.

    Parameters
    ----------
    connection : openeo.Connection
        OpenEO connection object.
    ref_id : str
        Reference ID for the extraction.
    ground_truth_file : str
        Path to the ground truth file.
    root_folder : Path
        Root folder for storing extraction outputs.
    period : str, optional
        Period for extractions, either 'month' or 'dekad'. Default is 'month'.
    restart_failed : bool, optional
        Whether to restart failed jobs. Default is False.

    Returns
    -------
    None
    """

    assert period in ["month", "dekad"], "Period must be either 'month' or 'dekad'."

    # Ref_id output folder
    output_folder = (
        root_folder / ref_id if period == "month" else root_folder / f"{ref_id}_10D"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    job_tracking_csv = output_folder / "job_tracking.csv"

    if job_tracking_csv.is_file():
        logger.info("Job tracking file already exists, skipping job creation.")
        job_df = pd.read_csv(job_tracking_csv)
    else:
        logger.info("Job tracking file does not exist, creating new jobs.")
        job_df = create_job_dataframe(ref_id, ground_truth_file)
        # job_df.to_csv(job_tracking_csv, index=False)

    if restart_failed and job_tracking_csv.is_file():
        logger.info("Resetting failed jobs.")
        job_df.loc[
            job_df["status"].isin(["error", "postprocessing-error"]), "status"
        ] = "not_started"
        job_df.to_csv(job_tracking_csv, index=False)

    logger.debug(job_df)

    manager = PatchToPointJobManager(root_dir=output_folder)
    manager.add_backend("terrascope", connection=connection, parallel_jobs=1)
    manager.run_jobs(
        df=job_df,
        start_job=partial(create_job_patch_to_point_worldcereal, period=period),
        job_db=job_tracking_csv,
    )

    # Merge all subparquets
    logger.info("Merging individual files ...")
    parquet_files = list(output_folder.rglob("*.geoparquet"))
    if len(parquet_files) != len(job_df):
        raise ValueError(
            f"Number of parquet files ({len(parquet_files)}) does not match number of jobs ({len(job_df)})"
        )
    logger.info(f"Found {len(parquet_files)} parquet files.")
    merged_gdf = merge_individual_parquet_files(parquet_files)
    merged_dir = (
        root_folder / "MERGED_PARQUETS"
        if period == "month"
        else root_folder / "MERGED_PARQUETS_10D"
    )
    merged_file = merged_dir / f"{ref_id}.geoparquet"
    merged_gdf.to_parquet(merged_file, index=False)
    logger.info(f"Merged parquet file saved to: {merged_file}")

    logger.success("ref_id fully done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run patch-to-point extractions for WorldCereal."
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        required=True,
        help="Root folder for storing extraction outputs.",
    )
    parser.add_argument(
        "--period",
        type=str,
        choices=["month", "dekad"],
        default="month",
        help="Period for extractions, either 'month' or 'dekad'. Default is 'month'.",
    )
    parser.add_argument(
        "--ref-ids",
        type=str,
        nargs="+",
        required=True,
        help="List of ref_ids to process.",
    )
    parser.add_argument(
        "--restart-failed",
        action="store_true",
        help="Restart failed jobs if the job tracking file exists.",
    )

    args = parser.parse_args()

    root_folder = Path(args.root_folder)
    period = args.period
    ref_ids = args.ref_ids
    restart_failed = args.restart_failed

    logger.info("Starting patch to point extractions ...")
    logger.info(f"Root folder: {root_folder}")
    logger.info(f"Period: {period}")

    connection = openeo.connect("openeo.vito.be").authenticate_oidc()

    for ref_id in ref_ids:
        logger.info(f"Processing ref_id: {ref_id}")
        ground_truth_file = (
            f"/vitodata/worldcereal/data/RDM/{ref_id}/harmonized/{ref_id}.geoparquet"
        )

        main(connection, ref_id, ground_truth_file, root_folder, period, restart_failed)

    logger.success("All done!")
