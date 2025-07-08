from typing import List

import numpy as np
import openeo
from loguru import logger
from openeo.api.process import Parameter
from openeo_gfmap.spatial import SpatialContext
from scipy.signal.windows import gaussian


def calculate_cloud_coverage_score(
    cloud_mask: openeo.DataCube, area: SpatialContext, scl: openeo.DataCube
) -> openeo.DataCube:
    """
    Calculate cloud coverage score.
    Args:
        cloud_mask (openeo.datacube.DataCube): Cloud mask data cube.
        area (dict): Geometry area.
        scl (openeo.datacube.DataCube): SCL data cube.
    Returns:
        openeo.datacube.DataCube: Cloud coverage score data cube.
    """
    logger.info("calculating cloud coverage score")
    # Calculate cloud coverage score
    cloud_coverage = (1 - cloud_mask).aggregate_spatial(geometries=area, reducer="mean")
    coverage_score = cloud_coverage.vector_to_raster(scl)
    coverage_score = coverage_score.rename_labels("bands", ["score"])
    return coverage_score


def calculate_date_score(scl: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate date score from SCL data.
    Args:
        scl (DataCube): SCL data cube.
    Returns:
        DataCube: Date score data cube.
    """
    logger.info("calculating date score")
    # Calculate date score
    day_of_month = scl.apply_neighborhood(
        day_of_month_calc,
        size=[
            {"dimension": "x", "unit": "px", "value": 1},
            {"dimension": "y", "unit": "px", "value": 1},
            {"dimension": "t", "value": "month"},
        ],
        overlap=[],
    )
    date_score = (1.0 * day_of_month).apply(date_score_calc)
    date_score = date_score.rename_labels("bands", ["score"])
    return date_score


def calculate_distance_to_cloud_score(
    binary: openeo.DataCube, spatial_resolution: int, max_distance: int = 150
) -> openeo.DataCube:
    """
    Calculate distance to cloud score.
    Args:
        binary (DataCube): Binary cloud mask data cube.
        spatial_resolution (int): Spatial resolution.
        max_distance (int): The maximum distance to cloud (in pixels) above which the DTC score will always be 1. Defined on a spatial resolution of 20m.
    Returns:
        DataCube: Distance to cloud score data cube.
    """
    logger.info("calculating distance to cloud score")

    def round_up_to_odd(f):
        return np.ceil(f) // 2 * 2 + 1

    # Calculate dtc score
    kernel_size = round_up_to_odd(max_distance * 20 / spatial_resolution)
    gaussian_1d = gaussian(M=kernel_size, std=1)
    gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)
    gaussian_kernel /= gaussian_kernel.sum()
    dtc_score = 1 - binary.apply_kernel(gaussian_kernel)
    return dtc_score


def create_rank_mask(score: openeo.DataCube) -> openeo.DataCube:
    """
    Create a rank mask based on the input score.
    Args:
        score (DataCube): Input score data cube.
    Returns:
        DataCube: Rank mask data cube.
    """
    logger.info("calculating rank mask")
    # Create a rank mask
    rank_mask = score.apply_neighborhood(
        max_score_selection,
        size=[
            {"dimension": "x", "unit": "px", "value": 1},
            {"dimension": "y", "unit": "px", "value": 1},
            {"dimension": "t", "value": "month"},
        ],
        overlap=[],
    )
    rank_mask = rank_mask.band("score")
    return rank_mask


def aggregate_BAP_scores(
    dtc_score: openeo.DataCube,
    date_score: openeo.DataCube,
    coverage_score: openeo.DataCube,
    weights: List = [1, 0.8, 0.5],
) -> openeo.DataCube:
    """
    Aggregate BAP scores using weighted sum.
    Args:
        dtc_score (DataCube): Distance to cloud score data cube.
        date_score (DataCube): Date score data cube.
        coverage_score (DataCube): Cloud coverage score data cube.
        weights (List): Weights for each score.
    Returns:
        DataCube: Aggregated BAP score data cube.
    """
    logger.info("aggregating rank score")
    # Aggregate scores
    score = (
        weights[0] * dtc_score + weights[1] * date_score + weights[2] * coverage_score
    ) / sum(weights)
    return score


def day_of_month_calc(input: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate day of month.
    Args:
        input (DataCube): Input data cube.
    Returns:
        DataCube: Day of month data cube.
    """
    label = Parameter("label")
    day = lambda x: 15 + x.process(  # noqa: E731
        "date_difference",
        date1=x.process(
            "date_replace_component", date=label, value=15, component="day"
        ),
        date2=label,
        unit="day",
    )
    return input.array_apply(day)


def date_score_calc(day: openeo.DataCube) -> openeo.DataCube:
    """
    Calculate date score from day of month.
    Args:
        day (DataCube): Day of month data cube.
    Returns:
        DataCube: Date score data cube.
    """
    return (
        day.subtract(15)
        .multiply(0.2)
        .multiply(day.subtract(15).multiply(0.2))
        .multiply(-0.5)
        .exp()
        .multiply(0.07978845)
    )  # Until 'power' and 'divide' are fixed, use this workaround


def max_score_selection(score: openeo.DataCube) -> openeo.DataCube:
    """
    Select maximum score from input score.
    Args:
        score (DataCube): Input score data cube.
    Returns:
        DataCube: Data cube with True where the score is not the maximum, False otherwise.
    """
    max_score = score.max()
    return score.array_apply(lambda x: x != max_score)
