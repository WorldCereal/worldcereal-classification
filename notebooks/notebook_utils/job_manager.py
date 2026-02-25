from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import geopandas as gpd
import pandas as pd
from IPython.display import display
from ipywidgets import Output
from loguru import logger
from tabulate import tabulate

from worldcereal.jobmanager import WorldCerealJobManager

JOB_STATUS_COLORS = {
    "not_started": "grey",
    "created": "gold",
    "queued": "lightsteelblue",
    "queued_for_start": "lightsteelblue",
    "running": "navy",
    "finished": "lime",
    "error": "darkred",
    "skipped": "darkorange",
    "start_failed": "red",
    None: "black",
}


def plot_job_status(
    status_df: pd.DataFrame,
    color_dict: Dict[Optional[str], str],
    center: Dict[str, float],
    zoom: int = 10,
):
    """Plot job status on a map using Plotly Express."""
    import copy

    import plotly.express as px

    status_plot = copy.deepcopy(status_df)
    if isinstance(status_plot, pd.DataFrame) and not isinstance(
        status_plot, gpd.GeoDataFrame
    ):
        status_plot = gpd.GeoDataFrame(status_plot, geometry="geometry")
    status_plot.crs = "EPSG:4326"
    status_plot["color"] = (
        status_plot["status"].map(color_dict).fillna(color_dict[None])
    )

    geojson = status_plot.set_index("tile_name").__geo_interface__

    fig = px.choropleth_mapbox(
        status_plot,
        geojson=geojson,
        locations="tile_name",
        color="status",
        color_discrete_map=color_dict,
        mapbox_style="carto-positron",
        center=center,
        zoom=zoom,
        title="Job Status Overview",
    )
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    return fig


def _ensure_outputs(
    plot_out: Optional[Output],
    log_out: Optional[Output],
    display_outputs: bool,
) -> Tuple[Output, Output]:
    plot_out_created = plot_out is None
    log_out_created = log_out is None

    if plot_out is None:
        plot_out = Output()
    if log_out is None:
        log_out = Output()

    if display_outputs and (plot_out_created or log_out_created):
        if log_out_created:
            display(log_out)
        if plot_out_created:
            display(plot_out)

    return plot_out, log_out


def _log_message(log_out: Output, display_outputs: bool, message: str) -> None:
    if display_outputs:
        with log_out:
            print(message)
    else:
        logger.info(message)


def _loguru_to_output(log_out: Output):
    def _sink(message):
        with log_out:
            print(message, end="")

    return _sink


def notebook_logger(
    plot_out: Optional[Output] = None,
    log_out: Optional[Output] = None,
    display_outputs: bool = True,
) -> Tuple[Output, Output, Callable[[str], None]]:
    plot_out, log_out = _ensure_outputs(plot_out, log_out, display_outputs)

    if display_outputs:
        logger.remove()
        logger.add(_loguru_to_output(log_out), format="{message}")

    def _log(message: str) -> None:
        _log_message(log_out, display_outputs, message)

    return plot_out, log_out, _log


def build_notebook_status_callback(
    manager: WorldCerealJobManager,
    plot_out: Optional[Output] = None,
    log_out: Optional[Output] = None,
    display_outputs: bool = True,
    status_title: str = "Job status",
    zoom: int = 10,
) -> Tuple[Callable[[pd.DataFrame], None], Output, Output]:
    plot_out, log_out = _ensure_outputs(plot_out, log_out, display_outputs)
    status_callback = manager.cli_status_callback(title=status_title)

    if display_outputs and manager.prepared_grid is not None:
        grid_for_map = manager.prepared_grid.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = grid_for_map.total_bounds
        center = {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2}

        def _status_callback(status_df: pd.DataFrame) -> None:
            timestamp = time.strftime("%H:%M:%S")
            if status_df.empty:
                with plot_out:
                    plot_out.clear_output(wait=True)
                    print(f"[{timestamp}] {status_title}: waiting for jobs to start.")
                return
            fig = plot_job_status(
                status_df=status_df,
                color_dict=JOB_STATUS_COLORS,
                center=center,
                zoom=zoom,
            )
            with plot_out:
                plot_out.clear_output(wait=True)
                fig.show(renderer="notebook")
                counts = status_df["status"].value_counts().to_string()
                print(f"[{timestamp}] {status_title}:\n{counts}")

        status_callback = _status_callback

    return status_callback, plot_out, log_out


def run_notebook_job_manager(
    manager: WorldCerealJobManager,
    *,
    run_kwargs: Dict[str, object],
    plot_out: Optional[Output] = None,
    log_out: Optional[Output] = None,
    display_outputs: bool = True,
    status_title: str = "Job status",
    zoom: int = 10,
) -> Tuple[Output, Output]:
    status_callback, plot_out, log_out = build_notebook_status_callback(
        manager,
        plot_out=plot_out,
        log_out=log_out,
        display_outputs=display_outputs,
        status_title=status_title,
        zoom=zoom,
    )

    resolved_kwargs = dict(run_kwargs)
    resolved_kwargs.setdefault("status_callback", status_callback)
    manager.run_jobs(**resolved_kwargs)

    return plot_out, log_out


def _read_job_tracking_csv(outdir: Path) -> pd.DataFrame:
    """Read

    Parameters
    ----------
    outdir : Path
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    tracking_file = outdir / "job_tracking.csv"
    if not tracking_file.exists():
        logger.warning(f"Job tracking file not found at {tracking_file}")
        return pd.DataFrame()

    try:
        job_df = pd.read_csv(tracking_file)
        return job_df
    except Exception as e:
        logger.error(f"Error reading job tracking file: {e}")
        return pd.DataFrame()


def check_job_status(
    outdir: Path,
    print_output: bool = True,
) -> dict:
    """Check the status of the jobs stored in the specified output directory and summarize it in a histogram.

    Parameters
    ----------
    outdir : Path
        The output directory where the job tracking file is stored.
    print_output : bool, default True
        Whether to print the status output.

    Returns
    -------
    dict
        status_histogram
    """

    if not isinstance(outdir, Path):
        raise TypeError("outdir must be a Path instance")

    # Read job tracking file
    job_df = _read_job_tracking_csv(outdir)
    if job_df.empty:
        logger.warning("No jobs found in the job tracking file.")
        return {}

    # Summarize the status in histogram
    status_histogram = job_df["status"].value_counts().to_dict()

    # convert to pandas dataframe
    status_count = pd.DataFrame(status_histogram.items(), columns=["status", "count"])
    status_count = status_count.sort_values(by="count", ascending=False)

    if print_output:
        print("-------------------------------------")
        print("Overall jobs status:")
        print(tabulate(status_count, headers="keys", tablefmt="psql", showindex=False))

    return status_histogram


def fetch_results_from_outdir(
    outdir: Path,
    file_pattern: str,
) -> list[Path]:
    """Fetch local file paths of successfully completed jobs from a specified output directory.

    Parameters
    ----------
    outdir : Path
        The output directory to fetch results from.
    file_pattern : str
        The pattern to match files against in the output directories.

    Returns
    -------
    list[Path]
        A list of paths to the files matching the pattern for successfully completed jobs.
    """

    if not isinstance(outdir, Path):
        raise TypeError("outdir must be a Path instance")

    # Read job status file

    job_df = _read_job_tracking_csv(outdir)
    if job_df.empty:
        logger.warning("No jobs found in the job tracking file.")
        return []

    # Get all jobs with status 'finished'
    job_df_finished = job_df[job_df["status"] == "finished"]
    if job_df_finished.empty:
        logger.warning("No finished jobs found in the job tracking file.")
        return []

    # Extract output paths using tile_name and expected output pattern
    results = []
    for _, row in job_df_finished.iterrows():
        tile_name = row["tile_name"]
        indir = outdir / tile_name
        expected_files = list(indir.glob(file_pattern))
        if expected_files:
            results.extend(expected_files)
        else:
            logger.warning(
                f"No files matching pattern '{file_pattern}' found for tile {tile_name} in {indir}"
            )

    return results
