import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
from loguru import logger

CROP_LEGEND_URL = (
    "https://s3.waw3-1.cloudferro.com/project_dependencies/worldcereal/"
    "WorldCereal_LC_CT_legend_latest.csv"
)
IRR_LEGEND_URL = (
    "https://s3.waw3-1.cloudferro.com/project_dependencies/worldcereal/"
    "WorldCereal_IRR_legend_latest.csv"
)


_LEGEND_UPLOAD_REMOVED_MESSAGE = (
    "Authenticated legend management has been removed. Publish legend files "
    "manually to the public object storage location instead."
)


def _get_artifactory_credentials():
    """Deprecated compatibility stub for removed credential handling."""
    raise RuntimeError(_LEGEND_UPLOAD_REMOVED_MESSAGE)


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
    """Deprecated compatibility stub for removed authenticated uploads."""
    raise RuntimeError(_LEGEND_UPLOAD_REMOVED_MESSAGE)


def upload_legend(srcpath: Path, date: str) -> str:
    """Deprecated compatibility stub for removed authenticated uploads."""
    raise RuntimeError(_LEGEND_UPLOAD_REMOVED_MESSAGE)


@lru_cache(maxsize=2)
def get_legend(topic: Literal["landcover", "irrigation"] = "landcover") -> pd.DataFrame:
    """Get the latest version of the WorldCereal land cover/crop type or irrigation legend
    from public object storage.

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

    Notes
    -----
    This function is cached using lru_cache to avoid repeated downloads.
    """

    if topic == "landcover":
        url = CROP_LEGEND_URL
    elif topic == "irrigation":
        url = IRR_LEGEND_URL
    else:
        raise ValueError("Invalid topic. Please use 'landcover' or 'irrigation'.")

    legend = pd.read_csv(url, header=0, sep=";")

    # Preprocess the legend by removing dashes from ewoc_code and converting to int, and setting it as index
    legend["ewoc_code"] = legend["ewoc_code"].str.replace("-", "").astype(np.int64)
    legend = legend.set_index("ewoc_code")

    return legend.copy()


def download_legend(
    dstpath: Path,
    topic: Literal["landcover", "irrigation"] = "landcover",
    retries=3,
    wait=2,
) -> Path:
    """Download the latest version of the WorldCereal legend.
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
        Raises if the requested legend cannot be downloaded.
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
    """Deprecated compatibility stub for removed authenticated deletion."""
    raise RuntimeError(_LEGEND_UPLOAD_REMOVED_MESSAGE)


def translate_ewoc_codes(
    ewoc_codes: list[int], legend: pd.DataFrame = None
) -> pd.DataFrame:
    """Translate EWOC codes to their corresponding labels in the WorldCereal legend,
        keeping all levels of the hierarchy.

    Parameters
    ----------
    ewoc_codes : list[int]
        List of EWOC codes to be translated.
    legend : pd.DataFrame, optional
        Pre-loaded legend DataFrame. If None, will call get_legend().
        If provided, should already have ewoc_code as index (preprocessed format).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the EWOC codes and their corresponding labels across the hierarchy.
    """

    if legend is None:
        legend = get_legend()
    else:
        legend = legend.copy()
    columns_to_keep = [
        "label_full",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
        "sampling_label",
        "definition",
    ]
    legend = legend[columns_to_keep]
    # Filter the legend to only include the requested EWOC codes
    # but also deal with the case where some EWOC codes are not present in the legend
    codes_not_in_legend = set(ewoc_codes) - set(legend.index)
    ewoc_codes = list(set(ewoc_codes) - codes_not_in_legend)
    legend = legend.loc[ewoc_codes]
    # sort by index
    legend = legend.sort_index()

    # replace NaN's with empty strings
    legend = legend.fillna("")

    if codes_not_in_legend:
        logger.warning(
            f"The following crop type codes are not present in the legend: {codes_not_in_legend}"
        )

    return legend


def ewoc_code_to_label(
    ewoc_codes: list[int], label_type: Literal["full", "sampling"] = "full"
) -> list[str]:
    """Translate EWOC codes to their corresponding full or sampling label in the WorldCereal legend.

    Parameters
    ----------
    ewoc_codes : list[int]
        List of EWOC codes to be translated.
    label_type : Literal["full", "sampling"], optional
        Type of label to return, by default "full"

    Returns
    -------
    list[str]
        List of full labels corresponding to the EWOC codes.
    """

    df = translate_ewoc_codes(ewoc_codes)

    result = []
    for code in ewoc_codes:
        if code not in df.index:
            result.append("Unknown")
        else:
            if label_type == "sampling":
                result.append(df.loc[code, "sampling_label"])
            else:
                result.append(df.loc[code, "label_full"])

    return result
