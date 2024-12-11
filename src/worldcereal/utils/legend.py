from pathlib import Path
from loguru import logger
import configparser
import subprocess


#TODO: we need a better way to authenticate with Artifactory

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
    
    # We  upload the file with a specific date tag and also with a "latest" tag
    target_names = [f"WorldCereal_LC_CT_legend_{date}.csv",
                    "WorldCereal_LC_CT_legend_latest.csv"]
    targetpaths = [f"{ARTIFACTORY_BASE_URL}legend/{n}" for n in target_names]

    for targetpath in targetpaths:
        logger.info(f"Uploading `{srcpath}` to `{targetpath}`")

        cmd = (
            f"curl -u{ARTIFACTORY_USERNAME}:{ARTIFACTORY_PASSWORD} -T {srcpath} "
            f'"{targetpath}"'
        )

        output, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()
        output = eval(output)
    
    return output["downloadUri"]


def download_latest_legend_from_artifactory(download_path: Path) -> Path:
    """Downloads the latest version of the WorldCereal land cover/crop type legend from Artifactory.

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
    latest_file = 'WorldCereal_LC_CT_legend_latest.csv'
    link = f"{ARTIFACTORY_BASE_URL}legend/{latest_file}"
    
    logger.info(f"Downloading latest legend file: {latest_file}")
    
    download_path.mkdir(parents=True, exist_ok=True)
    download_file = download_path / latest_file

    cmd = f'curl -u{ARTIFACTORY_USERNAME}:{ARTIFACTORY_PASSWORD} -o {download_file} "{link}"'

    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()

    return download_file


def delete_legend_file(path: str) -> None:
    """Deletes a legend file from Artifactory.

    Parameters
    ----------
    path : str
        Path to the legend file in Artifactory.
    """
    cmd = f"curl -u{ARTIFACTORY_USERNAME}:{ARTIFACTORY_PASSWORD} -X DELETE {path}"
    subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()


# if __name__ == '__main__':
    
#     # Example usage
#     srcpath = Path("/data/users/Private/jeroendegerickx/worldcereal/legend/WorldCereal_LC_CT_legend_20241210.csv")
#     date = "20241210"
#     download_path = Path("/data/users/Private/jeroendegerickx/worldcereal/legend")

#     # Upload the legend to Artifactory
#     link = upload_legend_csv_artifactory(srcpath, date)

#     # Download the latest legend from Artifactory
#     download_latest_legend_from_artifactory(download_path)
    
#     # Delete the uploaded legend from Artifactory
#     delete_legend_file(link)
