"""Main script to perform extractions. Each collection has it's specifities and
own functions, but the setup and main thread execution is done here."""

import argparse
from pathlib import Path
from typing import Dict, Optional, Union

from openeo_gfmap import Backend

from worldcereal.extract.common import run_extractions
from worldcereal.stac.constants import ExtractionCollection


def main(
    collection: ExtractionCollection,
    output_folder: Path,
    samples_df_path: Path,
    ref_id: str,
    max_locations_per_job: int = 500,
    python_memory: Optional[str] = None,
    driver_memory: Optional[str] = None,
    executor_memory: Optional[str] = None,
    driver_memory_overhead: Optional[str] = None,
    executor_memory_overhead: Optional[str] = None,
    max_executors: Optional[int] = None,
    parallel_jobs: int = 2,
    restart_failed: bool = False,
    extract_value: int = 0,
    backend=Backend.CDSE,
    write_stac_api: bool = False,
    check_existing_extractions: bool = False,
    image_name: Optional[str] = None,
    organization_id: Optional[int] = None,
) -> None:
    """Main function responsible for launching point and patch extractions.

    Parameters
    ----------
    collection : ExtractionCollection
        The collection to extract. Most popular: PATCH_WORLDCEREAL, POINT_WORLDCEREAL
    output_folder : Path
        The folder where to store the extracted data
    samples_df_path : Path
        Path to the input dataframe containing the geometries
        for which extractions need to be done
    ref_id : str
        Official ref_id of the source dataset
    max_locations_per_job : int, optional
        The maximum number of locations to extract per job, by default 500
    python_memory : str, optional
        memory assigned to Python UDFs, sar_backscatter or Sentinel-3 data loading tasks.
        If not specified, the default is to derive this from executor-memoryOverhead.
    driver_memory : str, optional
        memory allocated to the Spark ‘driver’ JVM, which manages the execution of the batch job.
        If not specified, the default value is used, depending on type of collection.
    executor_memory : str, optional
        memory allocated to the workers for the JVM that executes most predefined processes.
        If not specified, the default value is used, depending on type of collection.
    driver_memory_overhead : str, optional
        memory assigned to the spark ‘driver’ on top of JVM memory for Python processes.
        If not specified, the default value is used, depending on type of collection.
    executor_memory_overhead : str, optional
        allocated in addition to the JVM, for example, to run UDFs.
        If not specified, the default value is used, depending on type of collection.
    max_executors : int, optional
        Number of executors to run.
        If not specified, the default value is used, depending on type of collection.
    parallel_jobs : int, optional
        The maximum number of parallel jobs to run at the same time, by default 10
    restart_failed : bool, optional
        Restart the jobs that previously failed, by default False
    extract_value : int, optional
        All samples with an "extract" value equal or larger than this one, will be extracted, by default 0
        so all samples will be extracted
    backend : openeo_gfmap.Backend, optional
        cloud backend where to run the extractions, by default Backend.CDSE
    write_stac_api : bool, optional
        Save metadata of extractions to STAC API (requires authentication), by default False
    check_existing_extractions : bool, optional
        Check existing extractions in the STAC API before creating a job_dataframe,
        by default False
    image_name : str, optional
        Specific openEO image name to use for the jobs, by default None
    organization_id : int, optional
        ID of the organization to use for the job, by default None in which case
        the active organization for the user is used

    Returns
    -------
    None
    """

    # Compile custom job options
    job_options: Optional[Dict[str, Union[str, int]]] = {
        key: value
        for key, value in {
            "python-memory": python_memory,
            "executor-memory": executor_memory,
            "driver-memory": driver_memory,
            "executor-memoryOverhead": executor_memory_overhead,
            "driver-memoryOverhead": driver_memory_overhead,
            "max-executors": max_executors,
            "image-name": image_name,
            "etl_organization_id": organization_id,
        }.items()
        if value is not None
    } or None

    # We need to be sure that the
    # output_folder points to the correct ref_id
    if output_folder.name != ref_id:
        raise ValueError(
            f"`root_folder` should point to ref_id `{ref_id}`, instead got: {output_folder}"
        )

    # Fire up extractions
    _ = run_extractions(
        collection,
        output_folder,
        samples_df_path,
        ref_id,
        max_locations_per_job=max_locations_per_job,
        job_options=job_options,
        parallel_jobs=parallel_jobs,
        restart_failed=restart_failed,
        extract_value=extract_value,
        backend=backend,
        write_stac_api=write_stac_api,
        check_existing_extractions=check_existing_extractions,
    )

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from a collection")
    parser.add_argument(
        "collection",
        type=ExtractionCollection,
        choices=list(ExtractionCollection),
        help="The collection to extract",
    )
    parser.add_argument(
        "output_folder", type=Path, help="The folder where to store the extracted data"
    )
    parser.add_argument(
        "samples_df_path",
        type=Path,
        help="Path to the samples dataframe with the data to extract",
    )
    parser.add_argument(
        "--ref_id",
        type=str,
        required=True,
        help="Official `ref_id` of the source dataset",
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=500,
        help="The maximum number of locations to extract per job",
    )
    parser.add_argument(
        "--python_memory",
        type=str,
        default="1800m",
        help="memory assigned to Python UDFs, sar_backscatter or Sentinel-3 data loading tasks.",
    )
    parser.add_argument(
        "--driver_memory",
        type=str,
        default="2G",
        help="memory allocated to the Spark ‘driver’ JVM, which manages the execution of the batch job.",
    )
    parser.add_argument(
        "--driver_memory_overhead",
        type=str,
        default="1G",
        help="memory assigned to the spark ‘driver’ on top of JVM memory for Python processes.",
    )
    parser.add_argument(
        "--executor_memory",
        type=str,
        default="2G",
        help="memory allocated to the workers for the JVM that executes most predefined processes.",
    )
    parser.add_argument(
        "--executor_memory_overhead",
        type=str,
        default="1800m",
        help="allocated in addition to the JVM, for example, to run UDFs.",
    )
    parser.add_argument(
        "--max_executors", type=int, default=22, help="Number of executors to run."
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=2,
        help="The maximum number of parallel jobs to run at the same time.",
    )
    parser.add_argument(
        "--restart_failed",
        action="store_true",
        help="Restart the jobs that previously failed.",
    )
    parser.add_argument(
        "--extract_value",
        type=int,
        default=0,
        help="The value of the `extract` flag to use in the dataframe.",
    )
    parser.add_argument(
        "--write_stac_api",
        type=bool,
        default=False,
        help="Flag to write S1 and S2 patch extraction results to STAC API or not.",
    )
    parser.add_argument(
        "--check_existing_extractions",
        type=bool,
        default=False,
        help="Check existing extractions in the STAC API before creating a job_dataframe.",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="Specific openEO image name to use for the jobs.",
    )
    parser.add_argument(
        "--organization_id",
        type=int,
        default=None,
        help="ID of the organization to use for the job.",
    )

    args = parser.parse_args()

    main(
        collection=args.collection,
        output_folder=args.output_folder,
        samples_df_path=args.samples_df_path,
        ref_id=args.ref_id,
        max_locations_per_job=args.max_locations,
        python_memory=args.python_memory,
        driver_memory=args.driver_memory,
        executor_memory=args.executor_memory,
        driver_memory_overhead=args.driver_memory_overhead,
        executor_memory_overhead=args.executor_memory_overhead,
        max_executors=args.max_executors,
        parallel_jobs=args.parallel_jobs,
        restart_failed=args.restart_failed,
        extract_value=args.extract_value,
        backend=Backend.CDSE,
        write_stac_api=args.write_stac_api,
        check_existing_extractions=args.check_existing_extractions,
        image_name=args.image_name,
        organization_id=args.organization_id,
    )
