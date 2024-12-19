"""
Example script showing how to upload, download and delete the WorldCereal crop type legend from Artifactory.
"""

from pathlib import Path

from worldcereal.utils.legend import (
    delete_legend_file,
    download_latest_legend_from_artifactory,
    upload_legend_csv_artifactory,
)

if __name__ == "__main__":

    # Example usage
    srcpath = Path(
        "/vitodata/worldcereal/data/legend/WorldCereal_LC_CT_legend_20241216.csv"
    )
    date = "20241216"
    download_path = Path("/vitodata/worldcereal/data/legend_v2")

    # Upload the legend to Artifactory
    link = upload_legend_csv_artifactory(srcpath, date)

    # Download the latest legend from Artifactory
    download_latest_legend_from_artifactory(download_path)

    # Delete the uploaded legend from Artifactory
    delete_legend_file(link)
