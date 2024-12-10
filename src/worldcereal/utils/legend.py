from pathlib import Path
from loguru import logger
import configparser
import subprocess
from bs4 import BeautifulSoup


config_filename = (
    "/data/users/Private/jeroendegerickx/worldcereal/worldcerealconfig.ini"
)
config = configparser.ConfigParser()
config.read(config_filename)

ARTIFACTORY_USERNAME = config["artifactory"]["username"]
ARTIFACTORY_PASSWORD = config["artifactory"]["password"]

ARTIFACTORY_BASE_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/"
)


def upload_legend_csv_artifactory(srcpath: Path, date:str,) -> str:
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
    
    target_name = f"WorldCereal_LC_CT_legend_{date}.csv"
    targetpath = f"{ARTIFACTORY_BASE_URL}legend/{target_name}"

    logger.info(f"Uploading `{srcpath}` to `{targetpath}`")

    cmd = (
        f"curl -u{ARTIFACTORY_USERNAME}:{ARTIFACTORY_PASSWORD} -T {srcpath} "
        f'"{targetpath}"'
    )

    output, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()
    output = eval(output)

    return output["downloadUri"]


def download_latest_legend_from_artifactory(download_path: Path) -> Path:
    """Looks for and downloades the latest version of the WorldCereal land cover/crop type legend from Artifactory.

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
    
    available_files = _get_artifactory_legend_list()

    if not available_files:
        raise FileNotFoundError("No legend files found in Artifactory.")

    dates = [int(file.split("_")[-1].split(".")[0]) for file in available_files]
    latest_file = available_files[dates.index(max(dates))]
    link = f"{ARTIFACTORY_BASE_URL}legend/{latest_file}"
    logger.info(f"Downloading latest legend file: {latest_file}")
    
    download_path.mkdir(parents=True, exist_ok=True)
    download_file = download_path / latest_file

    cmd = f'curl -u{ARTIFACTORY_USERNAME}:{ARTIFACTORY_PASSWORD} -o {download_file} "{link}"'

    output, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()

    return download_file

def _get_artifactory_legend_list() -> list:
    """Get list of all WorldCereal legend files available in Artifactory as csv files.

    Returns
    -------
    list
        list of all WorldCereal legend files available in Artifactory
    """

    cmd = f'curl -u{ARTIFACTORY_USERNAME}:{ARTIFACTORY_PASSWORD} -X GET -k "{ARTIFACTORY_BASE_URL}legend/"'
    output, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()

    html_content = output.decode("utf-8")

    soup = BeautifulSoup(html_content, "html.parser")

    # Find all <a> tags
    links = soup.find_all("a", href=True)

    # Filter links that end with .csv
    csv_links = [link["href"] for link in links if link["href"].endswith(".csv")]

    return csv_links
