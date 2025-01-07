"""
Example script showing how to upload, download and delete the WorldCereal crop type legend from Artifactory.
"""

from pathlib import Path

from worldcereal.utils.legend import delete_legend_file, get_legend, upload_legend

if __name__ == "__main__":

    # Example usage
    srcpath = Path(
        "/vitodata/worldcereal/data/legend/WorldCereal_LC_CT_legend_20241231.csv"
    )
    date = "20241231"

    # Upload the legend to Artifactory
    link = upload_legend(srcpath, date)

    # Download the latest legend from Artifactory
    legend = get_legend()

    # Delete the uploaded legend from Artifactory
    delete_legend_file(link)
