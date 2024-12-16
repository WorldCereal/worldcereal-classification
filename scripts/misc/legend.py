from pathlib import Path
from worldcereal.utils.legend import (upload_legend_csv_artifactory,
                                       download_latest_legend_from_artifactory,
                                       delete_legend_file)


if __name__ == '__main__':
    
    # Example usage
    srcpath = Path("/data/users/Private/jeroendegerickx/worldcereal/legend/WorldCereal_LC_CT_legend_20241216.csv")
    date = "20241216"
    download_path = Path("/data/users/Private/jeroendegerickx/worldcereal/legend_v2")

    # Upload the legend to Artifactory
    link = upload_legend_csv_artifactory(srcpath, date)

    # Download the latest legend from Artifactory
    download_latest_legend_from_artifactory(download_path)
    
    # Delete the uploaded legend from Artifactory
    delete_legend_file(link)