"""
Use this script to download the latest WorldCereal legend to your active folder.

Publishing a new version of the legend is a manual process:

1. Place the dated CSV and PDF in ``/vitodata/worldcereal/data/legend/`` using
   ``WorldCereal_LC_CT_legend_YYYYMMDD.{csv,pdf}``.
2. Copy those files to ``/vitodata/worldcereal_auxdata/legend/`` and generate for both
a "latest" alias by replacing YYYYMMDD with "latest".
3. Update the SharePoint mappings file for any new crop types. After its mirror
   refreshes, add missing ``ewoc_code`` and ``label_full`` values to the MAPPINGS
   tab, assign their LANDCOVER/CROPTYPE classes, and confirm the mirror resolves
   all previously empty mappings.
4. Run this script to download the public CSV files as a final smoke test.

Apply the same dated-plus-latest convention when publishing an irrigation
legend.
"""

from pathlib import Path

from worldcereal.utils.legend import download_legend

if __name__ == "__main__":
    # Download legend to current folder for testing purposes
    destination = Path(".")
    download_legend(destination, topic="landcover")
    download_legend(destination, topic="irrigation")
