import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from loguru import logger

ARTIFACTORY_BASE_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/"
)

CROP_LEGEND_URL = ARTIFACTORY_BASE_URL + "legend/WorldCereal_LC_CT_legend_latest.csv"
IRR_LEGEND_URL = ARTIFACTORY_BASE_URL + "legend/WorldCereal_IRR_legend_latest.csv"


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


def _run_request(method: str, url: str, **kwargs) -> requests.Response:
    """Run an HTTP request with retries and return the response.
    Parameters
    ----------
    method : str
        HTTP method to be used
    url : str
        URL to send the request to
    kwargs : dict
        Additional keyword arguments, may include `retries`, `wait` and `logging_msg`
    Raises
    ------
    RuntimeError
        if the command fails after all retries
    Returns
    -------
    requests.Response
        The response of the http request
    """
    retries = kwargs.pop("retries", 3)
    wait = kwargs.pop("wait", 2)
    logging_msg = kwargs.pop("logging_msg", "Request")

    for attempt in range(retries):
        try:
            logger.debug(f"{logging_msg} (Attempt {attempt + 1})")
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            logger.debug("Execution successful")
            return response
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                logger.error(f"Failed to execute request: {url}")
                raise
    raise RuntimeError(f"Failed to execute request: {url}")


def _upload_file(srcpath, dstpath, username, password, retries=3, wait=2):
    """Upload a file to Artifactory.
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
    url = dstpath
    with open(srcpath, "rb") as f:
        file_content = f.read()  # Read the file content as binary
        headers = {
            "Content-Type": "application/octet-stream",  # Set the appropriate content type
        }
        response = _run_request(
            "PUT",
            url,
            data=file_content,  # Send raw file content in the request body
            headers=headers,
            auth=(username, password),
            logging_msg=f"Uploading `{srcpath}` to `{dstpath}`",
            retries=retries,
            wait=wait,
        )
    return response.json()["downloadUri"]


def upload_legend(srcpath: Path, date: str) -> str:
    """Upload a CSV file containing the WorldCereal land cover/crop type legend to Artifactory.
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
    dstpaths = [f"{ARTIFACTORY_BASE_URL}legend/WorldCereal_LC_CT_legend_{date}.csv"]
    dstpaths.append(CROP_LEGEND_URL)

    for dstpath in dstpaths:
        artifactory_link = _upload_file(
            srcpath, dstpath, artifactory_username, artifactory_password
        )

    # Return the download link of latest uploaded file
    return artifactory_link


def get_legend(topic: Literal["landcover", "irrigation"] = "landcover") -> pd.DataFrame:
    """Get the latest version of the WorldCereal land cover/crop type or irrigation legend
    from artifactory.

    Parameters
    ----------
    topic : Literal['landcover', 'irrigation'], optional
        Specifier for the legend file to be downloaded.
        Options are 'landcover' for land cover/crop type legend and 'irrigation' for irrigation legend.

    Returns
    -------
    pd.DataFrame
        requested legend as a Pandas DataFrame

    Raises
    ------
    ValueError
        if topic got an invalid value
    """

    if topic == "landcover":
        url = CROP_LEGEND_URL
    elif topic == "irrigation":
        url = IRR_LEGEND_URL
    else:
        raise ValueError("Invalid topic. Please use 'landcover' or 'irrigation'.")

    legend = pd.read_csv(url, header=0, sep=";")

    return legend


def download_legend(
    dstpath: Path,
    topic: Literal["landcover", "irrigation"] = "landcover",
    retries=3,
    wait=2,
) -> Path:
    """Download the latest version of the WorldCereal legend from Artifactory.
    Parameters
    ----------
    dstpath : Path
        Folder where the legend needs to be downloaded to.
    topic : Literal['landcover', 'irrigation'], optional
        Specifier for the legend file to be downloaded.
        Options are 'landcover' for land cover/crop type legend and 'irrigation' for irrigation legend.
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
    ValueError
        if topic got an invalid value
    """
    # Construct the download link
    if topic == "landcover":
        url = CROP_LEGEND_URL
    elif topic == "irrigation":
        url = IRR_LEGEND_URL
    else:
        raise ValueError("Invalid topic. Please use 'landcover' or 'irrigation'.")

    # Construct target path
    dstpath.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    download_file = dstpath / filename

    response = _run_request(
        "GET",
        url,
        logging_msg=f"Downloading latest legend file: {filename}",
        retries=retries,
        wait=wait,
    )

    with open(download_file, "wb") as f:
        f.write(response.content)

    return download_file


def delete_legend_file(srcpath: str, retries=3, wait=2):
    """Delete a legend file from Artifactory.
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

    _run_request(
        "DELETE",
        srcpath,
        auth=(artifactory_username, artifactory_password),
        logging_msg=f"Deleting legend file: {srcpath}",
        retries=retries,
        wait=wait,
    )
