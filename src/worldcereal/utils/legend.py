import json
import os
import subprocess
import tempfile
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


def upload_legend_csv_artifactory(
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
    target_names = [
        f"WorldCereal_LC_CT_legend_{date}.csv",
        "WorldCereal_LC_CT_legend_latest.csv",
    ]
    targetpaths = [f"{ARTIFACTORY_BASE_URL}legend/{n}" for n in target_names]

    for targetpath in targetpaths:
        logger.info(f"Uploading `{srcpath}` to `{targetpath}`")

        cmd = (
            f"curl -u{artifactory_username}:{artifactory_password} -T {srcpath} "
            f'"{targetpath}"'
        )

        output, _ = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, shell=True
        ).communicate()
        decoded_output = output.decode("utf-8")

        # Parse as JSON if applicable
        try:
            parsed_output = json.loads(decoded_output)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse output as JSON: {decoded_output}") from e

    # Access the desired value
    return parsed_output.get("downloadUri")


def get_latest_legend_from_artifactory() -> pd.DataFrame:
    """Get the latest version of the WorldCereal land cover/crop type legend from Artifactory
    as a Pandas Dataframe.

    Returns
    -------
    pd.DataFrame
        The WorldCereal land cover/crop type legend.
    """
    # create temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        # download the latest legend file
        legend_path = _download_latest_legend_from_artifactory(tmpdir)
        # read the legend file
        legend = pd.read_csv(legend_path, header=0, sep=";")

    # clean up the legend file
    legend = legend[legend["ewoc_code"].notna()]
    drop_columns = [c for c in legend.columns if "Unnamed:" in c]
    legend.drop(columns=drop_columns, inplace=True)

    return legend


def _download_latest_legend_from_artifactory(download_path: Path) -> Path:
    """Downloads the latest version of the WorldCereal land cover/crop type legend from Artifactory
    to a specified file path.
    Parameters
    ----------
    download_path : Path
        Folder where the legend needs to be downloaded to.
    Returns
    -------
    Path
        Path to the downloaded legend file.
    Raises
    ------
    FileNotFoundError
        Raises if no legend files are found in Artifactory.
    """
    latest_file = "WorldCereal_LC_CT_legend_latest.csv"
    link = f"{ARTIFACTORY_BASE_URL}legend/{latest_file}"

    logger.info(f"Downloading latest legend file: {latest_file}")

    download_path.mkdir(parents=True, exist_ok=True)
    download_file = download_path / latest_file

    cmd = f'curl -o {download_file} "{link}"'

    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()

    return download_file


def delete_legend_file(path: str) -> None:
    """Deletes a legend file from Artifactory.
    Parameters
    ----------
    path : str
        Path to the legend file in Artifactory.
    """
    # Get Artifactory credentials
    artifactory_username, artifactory_password = _get_artifactory_credentials()

    cmd = f"curl -u{artifactory_username}:{artifactory_password} -X DELETE {path}"
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()
