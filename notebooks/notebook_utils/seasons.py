import logging
from calendar import monthrange
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from openeo_gfmap import BoundingBoxExtent
from pyproj import Transformer

from worldcereal.seasons import get_season_dates_for_extent

logging.getLogger("rasterio").setLevel(logging.ERROR)


def get_month_decimal(date):

    return date.timetuple().tm_mon + (
        date.timetuple().tm_mday / monthrange(2021, date.timetuple().tm_mon)[1]
    )


def plot_worldcereal_seasons(
    seasons: dict,
    extent: BoundingBoxExtent,
    tag: str = "",
):
    """Method to plot WorldCereal seasons in a matplotlib plot.
    Parameters
    ----------
    seasons : dict
        dictionary with season names as keys and start and end dates as values
        dates need to be in datetime format
    extent : BoundingBoxExtent
        extent for which to plot the seasons
    tag : str, optional
        tag to add to the title of the plot, by default empty string

    Returns
    -------
    None
    """

    # get lat, lon centroid of extent
    transformer = Transformer.from_crs(
        f"EPSG:{extent.epsg}", "EPSG:4326", always_xy=True
    )
    minx, miny = transformer.transform(extent.west, extent.south)
    maxx, maxy = transformer.transform(extent.east, extent.north)
    lat = (maxy + miny) / 2
    lon = (maxx + minx) / 2
    location = f"lat={lat:.2f}, lon={lon:.2f}"

    # prepare figure
    fig, ax = plt.subplots()
    plt.title(f"WorldCereal seasons {tag} ({location})")
    ax.set_ylim((0.4, len(seasons) + 0.5))
    ax.set_xlim((0, 13))
    ax.set_yticks(range(1, len(seasons) + 1))
    ax.set_yticklabels(list(seasons.keys()))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    facecolor = "darkgoldenrod"

    # Get the start and end date for each season
    idx = 0
    for name, dates in seasons.items():
        sos, eos = dates

        # get start and end month (decimals) for plotting
        start = get_month_decimal(sos)
        end = get_month_decimal(eos)

        # add rectangle to plot
        if start < end:
            ax.add_patch(
                Rectangle((start, idx + 0.75), end - start, 0.5, color=facecolor)
            )
        else:
            ax.add_patch(
                Rectangle((start, idx + 0.75), 12 - start, 0.5, color=facecolor)
            )
            ax.add_patch(Rectangle((1, idx + 0.75), end - 1, 0.5, color=facecolor))

        # add labels to plot
        label_start = sos.strftime("%B %d")
        label_end = eos.strftime("%B %d")
        plt.text(
            start - 0.2,
            idx + 0.65,
            label_start,
            fontsize=8,
            color="darkgreen",
            ha="left",
            va="center",
        )
        plt.text(
            end + 0.2,
            idx + 0.65,
            label_end,
            fontsize=8,
            color="darkred",
            ha="right",
            va="center",
        )

        idx += 1

    # display plot
    plt.show()


def retrieve_worldcereal_seasons(
    extent: BoundingBoxExtent,
    seasons: List[str] = ["s1", "s2"],
    plot: bool = True,
):
    """Method to retrieve default WorldCereal seasons from global crop calendars.
    These will be logged to the screen for informative purposes.

    Parameters
    ----------
    extent : BoundingBoxExtent
        extent for which to load seasonality
    seasons : List[str], optional
        seasons to load, by default s1 and s2
    plot : bool, optional
        whether to plot the seasons, by default True

    Returns
    -------
    dict
        dictionary with season names as keys and start and end dates as values
    """
    results = {}

    # Get the start and end date for each season
    for idx, season in enumerate(seasons):
        seasonal_extent = get_season_dates_for_extent(extent, 2021, f"tc-{season}")
        sos = pd.to_datetime(seasonal_extent.start_date)
        eos = pd.to_datetime(seasonal_extent.end_date)
        results[season] = (sos, eos)

    # Plot the seasons if requested
    if plot:
        plot_worldcereal_seasons(results, extent)

    return results
