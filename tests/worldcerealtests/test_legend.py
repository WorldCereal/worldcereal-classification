from pathlib import Path

import pytest

from worldcereal.utils.legend import (
    CROP_LEGEND_URL,
    IRR_LEGEND_URL,
    delete_legend_file,
    upload_legend,
)


def test_legend_urls_use_public_object_storage():
    expected_base = "https://s3.waw3-1.cloudferro.com/project_dependencies/worldcereal/"
    assert CROP_LEGEND_URL == expected_base + "WorldCereal_LC_CT_legend_latest.csv"
    assert IRR_LEGEND_URL == expected_base + "WorldCereal_IRR_legend_latest.csv"


@pytest.mark.parametrize(
    "operation",
    [
        lambda: upload_legend(Path("legend.csv"), "20260903"),
        lambda: delete_legend_file("legend.csv"),
    ],
)
def test_authenticated_legend_management_is_retired(operation):
    with pytest.raises(RuntimeError, match="manually"):
        operation()
