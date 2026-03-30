"""
Example script showing how to upload, download and delete the WorldCereal crop type legend from Artifactory.
"""

import shutil
from pathlib import Path

from worldcereal.utils.legend import (
    ARTIFACTORY_BASE_URL,
    _get_artifactory_credentials,
    _upload_file,
    # delete_legend_file,
    download_legend,
    # get_legend,
    upload_legend,
)

if __name__ == "__main__":

    # Publish new legend?
    # 1. Copy new CSV and pdf files to folder /vitodata/worldcereal/data/legend/
    # 2. Adjust date:

    date = "20260313"
    srcpath = Path(
        f"/vitodata/worldcereal/data/legend/WorldCereal_LC_CT_legend_{date}.csv"
    )

    # 3. Upload the legend csv to Artifactory
    link = upload_legend(srcpath, date)

    # 4. Download the latest legend from Artifactory
    legend_path = download_legend(Path("/vitodata/worldcereal/data/legend/"))

    # 5. Duplicate the pdf version of the legend and rename to "WorldCereal_LC_CT_legend_latest.pdf"
    src_pdf_path = Path(
        f"/vitodata/worldcereal/data/legend/WorldCereal_LC_CT_legend_{date}.pdf"
    )
    pdf_path = Path(
        "/vitodata/worldcereal/data/legend/WorldCereal_LC_CT_legend_latest.pdf"
    )
    shutil.copy(src_pdf_path, pdf_path)

    # 6. upload new pdf:
    # Get Artifactory credentials
    artifactory_username, artifactory_password = _get_artifactory_credentials()
    dstpath = f"{ARTIFACTORY_BASE_URL}legend/WorldCereal_LC_CT_legend_latest.pdf"
    artifactory_link = _upload_file(
        pdf_path, dstpath, artifactory_username, artifactory_password
    )

    # 7. MAKE SURE YOU ALSO UPDATE THE MAPPINGS FILE (on sharepoint) TO ASSIGN A MAPPING CLASS TO EACH NEWLY ADDED CROP TYPE.
    # How to add landcover/croptype mapping to new classes:
    # - wait for the update of the mirror tab
    # - in the mirror tab, you need to look for rows where LANDCOVER10/CROPTYPE24 columns are NA. you can do this however you like, using filters or just scrolling
    # - copy missing ewoc_codes and label_full to the MAPPINGS tab, just appending to the end
    # - in the MAPPINGS tab, add LANDCOVER/CROPTYPE mappings manually
    # - done! check in the mirror tab again if the formulas picked up the mapping correctly

    # #####
    # Functions to retrieve legend and irr legend:

    # # Get the latest legend from Artifactory (as pandas DataFrame)
    # legend = get_legend()

    # legend_irr = get_legend(topic="irrigation")

    # irr_legend_path = download_legend(Path(""), topic="irrigation")

    # # Delete the uploaded legend from Artifactory
    # # delete_legend_file(link)
