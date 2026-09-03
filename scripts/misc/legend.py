"""
WorldCereal legend publication and download helper.

Publishing is intentionally manual: this repository no longer manages storage
credentials or uploads legends itself. To publish a land-cover/crop-type legend:

1. Place the dated CSV and PDF in ``/vitodata/worldcereal/data/legend/`` using
   ``WorldCereal_LC_CT_legend_YYYYMMDD.{csv,pdf}``.
2. Upload both dated files and their ``latest`` aliases with an approved S3
   client. For the configured ``cloudferro`` rclone remote, for example:

   rclone copyto WorldCereal_LC_CT_legend_YYYYMMDD.csv \
       cloudferro:project_dependencies/worldcereal/WorldCereal_LC_CT_legend_YYYYMMDD.csv
   rclone copyto WorldCereal_LC_CT_legend_YYYYMMDD.csv \
       cloudferro:project_dependencies/worldcereal/WorldCereal_LC_CT_legend_latest.csv
   rclone copyto WorldCereal_LC_CT_legend_YYYYMMDD.pdf \
       cloudferro:project_dependencies/worldcereal/WorldCereal_LC_CT_legend_YYYYMMDD.pdf
   rclone copyto WorldCereal_LC_CT_legend_YYYYMMDD.pdf \
       cloudferro:project_dependencies/worldcereal/WorldCereal_LC_CT_legend_latest.pdf

3. Verify the public ``latest`` files under
   ``https://s3.waw3-1.cloudferro.com/project_dependencies/worldcereal/``.
4. Update the SharePoint mappings file for any new crop types. After its mirror
   refreshes, add missing ``ewoc_code`` and ``label_full`` values to the MAPPINGS
   tab, assign their LANDCOVER/CROPTYPE classes, and confirm the mirror resolves
   all previously empty mappings.
5. Run this script to download the public CSV files as a final smoke test.

Apply the same dated-plus-latest convention when publishing an irrigation
legend. The commands above are documentation only; rclone owns authentication.
"""

from pathlib import Path

from worldcereal.utils.legend import download_legend

if __name__ == "__main__":
    destination = Path("/vitodata/worldcereal/data/legend/")
    download_legend(destination, topic="landcover")
    download_legend(destination, topic="irrigation")
