from shapely.geometry import Polygon

from worldcereal.utils.refdata import query_public_extractions_v1


def test_query_public_extractions():
    """Unittest for querying public extractions."""

    # Define small polygon
    poly = Polygon.from_bounds(*(4.535, 51.050719, 4.600936, 51.098176))

    # Query extractions
    df = query_public_extractions_v1(poly, buffer=100)

    # Check if dataframe has samples
    assert not df.empty
