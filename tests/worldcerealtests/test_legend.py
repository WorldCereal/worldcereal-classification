from worldcereal.utils.legend import (
    CROP_LEGEND_URL,
    IRR_LEGEND_URL,
)


def test_legend_urls_use_auxdata_terrascope():
    expected_base = "https://auxdata.terrascope.be/worldcereal/legend/"
    assert CROP_LEGEND_URL == expected_base + "WorldCereal_LC_CT_legend_latest.csv"
    assert IRR_LEGEND_URL == expected_base + "WorldCereal_IRR_legend_latest.csv"
