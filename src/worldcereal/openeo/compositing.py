def max_ndvi_selection(ndvi):
    max_ndvi = ndvi.max()
    return ndvi.array_apply(lambda x: x != max_ndvi)


def max_ndvi_composite(s2_cube, composite_window="month"):
    ndvi = s2_cube.ndvi(nir="B08", red="B04")

    rank_mask = ndvi.apply_neighborhood(
        max_ndvi_selection,
        size=[
            {"dimension": "x", "unit": "px", "value": 1},
            {"dimension": "y", "unit": "px", "value": 1},
            {"dimension": "t", "value": "month"},
        ],
        overlap=[],
    )

    s2_cube = s2_cube.mask(rank_mask).aggregate_temporal_period(
        composite_window, "first"
    )

    return s2_cube
