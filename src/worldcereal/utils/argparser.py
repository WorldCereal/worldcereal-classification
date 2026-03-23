from typing import Any, Dict

try:
    from loguru import logger
except ImportError:
    # loguru not available, use standard logging
    import logging

    logger = logging.getLogger(__name__)  # type: ignore


DEFAULT_JOB_OPTIONS: Dict[str, Any] = {
    "driver-memory": "12G",
    "driver-memoryOverhead": "2G",
    "executor-cores": 2,
    "executor-memory": "4G",
    "executor-memoryOverhead": "2G",
    "max-executors": 300,
    "image-name": None,
    "etl_organization_id": None,
}


def parse_job_options_from_args(
    args,
    default_options: Dict[str, Any] = DEFAULT_JOB_OPTIONS,
) -> Dict[str, Any]:
    """
    Parse openEO job options from command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.If no custom job options are provided, defaults are used (see default_options).
        Recognized keys:
            driver-memory, driver-memoryOverhead, executor-cores, executor-memory,
            executor-memoryOverhead, max-executors, image-name, etl_organization_id.
    default_options : dict
        A dictionary of default job options to use if no custom options are provided.
    Returns
    -------
    dict [str, Union[str, int, None]]
    """
    parsed_job_options = {
        key: value
        for key, value in {
            "driver-memory": args.driver_memory,
            "driver-memoryOverhead": args.driver_memoryOverhead,
            "executor-cores": args.executor_cores,
            "executor-memory": args.executor_memory,
            "executor-memoryOverhead": args.executor_memoryOverhead,
            "max-executors": args.max_executors,
            "image-name": args.image_name,
            "etl_organization_id": args.organization_id,
        }.items()
        if value is not None
    } or None
    if parsed_job_options is not None:
        logger.info(
            f"Using custom job options for the following parameters: {list(parsed_job_options.keys())}"
        )
        job_options = {
            k: (v if v is not None else default_options[k])
            for k, v in parsed_job_options.items()
        }
    else:
        logger.info("No custom job options provided, using defaults.")
        job_options = default_options.copy()

    return job_options
