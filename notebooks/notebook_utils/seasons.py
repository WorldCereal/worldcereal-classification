import importlib.resources as pkg_resources
import logging
from calendar import monthrange
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.patches import Rectangle
from openeo_gfmap import BoundingBoxExtent
from pyproj import Transformer

from worldcereal.data import cropcalendars
from worldcereal.seasons import (
    get_season_dates_for_extent,
    season_doys_to_dates_refyear,
)

logging.getLogger("rasterio").setLevel(logging.ERROR)


def get_month_decimal(date):

    return date.timetuple().tm_mon + (
        date.timetuple().tm_mday / monthrange(2021, date.timetuple().tm_mon)[1]
    )


def plot_worldcereal_seasons(seasons: dict, extent: BoundingBoxExtent, tag: str = ""):
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
    extent: BoundingBoxExtent, seasons: List[str] = ["s1", "s2"], plot: bool = True
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


def retrieve_phaseI_seasons(extent: BoundingBoxExtent, plot: bool = True):
    """Method to retrieve WorldCereal seasons from Phase I AEZ's.
    Parameters
    ----------
    extent : BoundingBoxExtent
        extent for which to load seasonality
    plot : bool, optional
        whether to plot the seasons, by default True
    Returns
    -------
    dict
        dictionary with AEZ id's as keys and season info as values:
        {AEZ_id: {season_name: (start_date, end_date)}}

    """

    # Read Phase I AEZ's
    with pkg_resources.open_binary(cropcalendars, "AEZ_phase_I.geojson") as aez_file:  # type: ignore
        aez = gpd.read_file(aez_file)

    # Convert extent to shapely geometry
    if extent.epsg != 4326:
        transformer = Transformer.from_crs(
            f"EPSG:{extent.epsg}", "EPSG:4326", always_xy=True
        )
        minx, miny = transformer.transform(extent.west, extent.south)
        maxx, maxy = transformer.transform(extent.east, extent.north)
        extent = BoundingBoxExtent(
            west=minx, south=miny, east=maxx, north=maxy, epsg=4326
        )
    poly = extent.to_geometry()

    # Get the AEZ that intersects with the extent
    aez = aez[aez.intersects(poly)]

    # Log warning if multiple AEZ's occur
    if len(aez) > 1:
        logger.warning("Multiple AEZ's found for extent!")

    # Season definitions in Phase I
    seasons_phaseI = {
        "tc-wintercereals": "ww",
        "tc-maize-main": "m1",
        "tc-maize-second": "m2",
    }
    # For each AEZ, get the start and end dates for each season
    seasons: dict[int, dict] = {}
    for idx, row in aez.iterrows():
        id = int(row["zoneID"])
        seasons[id] = {}
        for season_name, season_id in seasons_phaseI.items():
            sos = row[f"{season_id}sos_min"]
            eos = row[f"{season_id}eos_max"]
            if ~np.isnan(sos) and ~np.isnan(eos):
                start_date, end_date = season_doys_to_dates_refyear(sos, eos, 2021)
                seasons[id][season_name] = (start_date, end_date)

    # Plot the seasons if requested
    if plot:
        for aez_id, season_dates in seasons.items():
            plot_worldcereal_seasons(season_dates, extent, tag=f"AEZ {aez_id}")

    return seasons
