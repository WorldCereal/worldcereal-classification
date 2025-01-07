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

    # execute the command with retries
    for attempt in range(retries):
        try:
            logger.info(f"Uploading `{srcpath}` to `{dstpath}` (Attempt {attempt + 1})")

            output, _ = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, shell=True
            ).communicate()
            decoded_output = output.decode("utf-8")

            # Parse as JSON if applicable
            parsed_output = json.loads(decoded_output)
            logger.info("Upload successful")
            return parsed_output.get("downloadUri")

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Failed to upload file")


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

    for attempt in range(retries):
        try:
            logger.info(
                f"Downloading latest legend file: {latest_file} (Attempt {attempt + 1})"
            )
            subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()
            logger.info("Download successful!")

            return download_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Failed to download file")


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

    # execute the command with retries
    for attempt in range(retries):
        try:
            logger.info(f"Deleting legend file: {srcpath} (Attempt {attempt + 1})")
            subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()

            logger.info("Deletion successful")
            return

        except subprocess.CalledProcessError as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Failed to delete file")
