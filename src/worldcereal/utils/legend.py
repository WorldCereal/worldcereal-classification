import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
from loguru import logger

ARTIFACTORY_BASE_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/"
)


def _get_artifactory_credentials():
    """Get credentials for upload and delete operations on Artifactory.
    Returns
    -------
    tuple (str, str)
        Tuple containing the Artifactory username and password.
    Raises
    ------
    ValueError
        if ARTIFACTORY_USERNAME or ARTIFACTORY_PASSWORD are not set as environment variables.
    """

    artifactory_username = os.getenv("ARTIFACTORY_USERNAME")
    artifactory_password = os.getenv("ARTIFACTORY_PASSWORD")

    if not artifactory_username or not artifactory_password:
        raise ValueError(
            "Artifactory credentials not found. "
            "Please set ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD environment variables."
        )

    return artifactory_username, artifactory_password


def _run_curl_cmd(cmd: str, logging_msg: str, retries=3, wait=2) -> dict:
    """Run a curl command with retries and return the output.

    Parameters
    ----------
    cmd : str
        The curl command to be executed
    logging_msg : str
        Message to be logged
    retries : int, optional
        Number of retries, by default 3
    wait : int, optional
        Seconds to wait in between retries, by default 2
    Raises
    ------
    RuntimeError
        if the command fails after all retries
    Returns
    -------
    dict
        The parsed output of the curl command
    """

    for attempt in range(retries):
        try:
            logger.debug(f"{logging_msg} (Attempt {attempt + 1})")
            output, _ = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, shell=True
            ).communicate()
            decoded_output = output.decode("utf-8")

            # Parse as JSON if applicable
            if decoded_output != "":
                parsed_output = json.loads(decoded_output)
            else:
                parsed_output = {}
            logger.debug("Execution successful")
            return parsed_output

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                logger.error(f"Failed to execute command: {cmd}")
                raise

    raise RuntimeError(f"Failed to execute command: {cmd}")


def _upload_file(
    srcpath,
    dstpath,
    username,
    password,
    retries=3,
    wait=2,
) -> str:
    """Function taking care of file upload to Artifactory with retries.

    Parameters
    ----------
    srcpath : Path
        Path to csv file that needs to be uploaded to Artifactory.
    dstpath : str
        Full link to the target location in Artifactory.
    username : str
        Artifactory username.
    password : str
        Artifactory password.
    retries : int, optional
        Number of retries, by default 3
    wait : int, optional
        Seconds to wait in between retries, by default 2

    Returns
    -------
    str
        Full link to the target location in Artifactory.
    """
    # construct the curl command
    cmd = f"curl -u{username}:{password} -T {srcpath} " f'"{dstpath}"'

    # construct logging message
    logging_msg = f"Uploading `{srcpath}` to `{dstpath}`"

    # execute the command with retries
    output = _run_curl_cmd(cmd, logging_msg, retries=retries, wait=wait)

    # return the download link
    return output["downloadUri"]


def upload_legend(
    srcpath: Path,
    date: str,
) -> str:
    """Uploads a CSV file containing the worldcereal land cover/crop type legend to Artifactory.
    Parameters
    ----------
    srcpath : Path
        Path to csv file that needs to be uploaded to Artifactory.
    date : str
        Date tag to be added to the filename. Should be in format YYYYMMDD.
    Returns
    -------
    str
        artifactory download link
    Raises
    ------
    FileNotFoundError
        if srcpath does not exist
    """
    if not srcpath.is_file():
        raise FileNotFoundError(f"Required file `{srcpath}` not found.")

    # Get Artifactory credentials
    artifactory_username, artifactory_password = _get_artifactory_credentials()

    # We  upload the file with a specific date tag and also with a "latest" tag
    dst_names = [
        f"WorldCereal_LC_CT_legend_{date}.csv",
        "WorldCereal_LC_CT_legend_latest.csv",
    ]
    dstpaths = [f"{ARTIFACTORY_BASE_URL}legend/{n}" for n in dst_names]

    for dstpath in dstpaths:
        artifactory_link = _upload_file(
            srcpath, dstpath, artifactory_username, artifactory_password
        )

    # Return the download link of latest uploaded file
    return artifactory_link


def get_legend() -> pd.DataFrame:
    """Get the latest version of the WorldCereal land cover/crop type legend from Artifactory
    as a Pandas Dataframe.

    Returns
    -------
    pd.DataFrame
        The WorldCereal land cover/crop type legend.
    """
    # create temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        dstpath = Path(tmpdirname)
        # download the latest legend file
        legend_path = _download_legend(dstpath)
        # read the legend file
        legend = pd.read_csv(legend_path, header=0, sep=";")

    # clean up the legend file
    legend = legend[legend["ewoc_code"].notna()]
    drop_columns = [c for c in legend.columns if "Unnamed:" in c]
    legend.drop(columns=drop_columns, inplace=True)

    return legend


def _download_legend(
    dstpath: Path,
    retries=3,
    wait=2,
) -> Path:
    """Downloads the latest version of the WorldCereal land cover/crop type legend from Artifactory
    to a specified file path.
    Parameters
    ----------
    dstpath : Path
        Folder where the legend needs to be downloaded to.
    retries : int, optional
        Number of retries, by default 3
    wait : int, optional
        Seconds to wait in between retries, by default 2
    Returns
    -------
    Path
        Path to the downloaded legend file.
    Raises
    ------
    FileNotFoundError
        Raises if no legend files are found in Artifactory.
    """
    # Construct the download link and curl command
    latest_file = "WorldCereal_LC_CT_legend_latest.csv"
    link = f"{ARTIFACTORY_BASE_URL}legend/{latest_file}"
    dstpath.mkdir(parents=True, exist_ok=True)
    download_file = dstpath / latest_file
    cmd = f'curl -o {download_file} "{link}"'

    # construct logging message
    logging_msg = f"Downloading latest legend file: {latest_file}"

    # execute the command with retries
    _run_curl_cmd(cmd, logging_msg, retries=retries, wait=wait)

    # return the path to the downloaded file
    return download_file


def delete_legend_file(
    srcpath: str,
    retries=3,
    wait=2,
) -> None:
    """Deletes a legend file from Artifactory.
    Parameters
    ----------
    srcpath : str
        Path to the legend file in Artifactory.
    retries : int, optional
        Number of retries, by default 3
    wait : int, optional
        Seconds to wait in between retries, by default 2
    """
    # Get Artifactory credentials
    artifactory_username, artifactory_password = _get_artifactory_credentials()

    # construct the curl command
    cmd = f"curl -u{artifactory_username}:{artifactory_password} -X DELETE {srcpath}"

    # construct logging message
    logging_msg = f"Deleting legend file: {srcpath}"

    # execute the command with retries
    _run_curl_cmd(cmd, logging_msg, retries=retries, wait=wait)
