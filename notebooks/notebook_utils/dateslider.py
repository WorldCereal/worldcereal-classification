from datetime import datetime, timedelta

import ipywidgets as widgets
import pandas as pd
from IPython.core.display import HTML as core_HTML
from IPython.display import display
from loguru import logger
from openeo_gfmap import TemporalContext


class date_slider:
    """Class that provides a slider for selecting a processing period.
    The processing period is fixed in length, amounting to one year.
    The processing period will always start the first day of a month and end the last day of a month.
    """

    def __init__(self, start_date=datetime(2018, 1, 1), end_date=datetime(2024, 12, 1)):

        # Define the slider
        dates = pd.date_range(start_date, end_date, freq="MS")
        options = [(date.strftime("%b %Y"), date) for date in dates]
        self.interval_slider = widgets.SelectionRangeSlider(
            options=options,
            index=(0, 11),  # Default to a 11-month interval
            orientation="horizontal",
            description="",
            continuous_update=False,
            behaviour="drag",
            style={
                "handle_color": "dodgerblue",
            },
            layout=widgets.Layout(
                width="600px",
                margin="0 0 0 10px",
            ),
            readout=False,
        )

        # Define the HTML text widget for the selected range and focus time
        initial_range = [
            (pd.to_datetime(start_date)).strftime("%d %b %Y"),
            (
                pd.to_datetime(start_date)
                + pd.DateOffset(months=12)
                - timedelta(days=1)
            ).strftime("%d %b %Y"),
        ]
        initial_focus_time = (
            pd.to_datetime(start_date) + pd.DateOffset(months=6)
        ).strftime("%b %Y")
        self.html_text = widgets.HTML(
            value=f"<b>Selected range:</b> {initial_range[0]} - {initial_range[1]}<br><b>Season center:</b> {initial_focus_time}",
            placeholder="HTML placeholder",
            description="",
            layout=widgets.Layout(justify_content="center", display="flex"),
        )

        # Attach slider observer
        self.interval_slider.observe(self.on_slider_change, names="value")

        # Add custom CSS for the ticks
        custom_css = """
        <style>
        .widget-container {
            padding-left: 10px; /* Add 20px margin on left and right */
            box-sizing: border-box; /* Include padding in width */
            width: 600px; /* Fixed width for consistent alignment */
            margin: 0 auto; /* Center align the container */
            position: relative;
        }
        .slider-container {
            position: relative;
            width: 100%;
            margin: 0 auto; /* Center align */
        }
        .slider-container .tick-wrapper {
            position: relative;
            width: 100%; /* Match slider width */
            height: 40px; /* Reserve space for ticks and labels */
        }
        .slider-container .tick-mark {
            position: absolute;
            bottom: 25px; /* Adjust to position tick marks relative to labels */
            transform: translateX(-50%);
            font-size: 14px;
            font-weight: bold;
        }
        .slider-container .tick-label {
            position: absolute;
            bottom: 0; /* Place directly under tick marks */
            transform: translateX(-50%);
            font-size: 10px;
            text-align: center;
            line-height: 1.2em; /* For two-line labels */
        }
        </style>
        """

        # # Generate ticks
        tick_dates = pd.date_range(
            start_date, pd.to_datetime(end_date) + pd.DateOffset(months=1), freq="4MS"
        )
        tick_labels = [date.strftime("%b %Y") for date in tick_dates]
        n_labels = len(tick_labels)
        ticks_html = ""
        for i, label in enumerate(tick_labels):
            position = (i / (n_labels - 1)) * 100  # Position as a percentage
            ticks_html += f"""
            <div class="tick-mark" style="left: {position}%; ">|</div>
            <div class="tick-label" style="left: {position}%; ">{label.split()[0]}<br>{label.split()[1]}</div>
            """

        # HTML container for tick marks and labels
        tick_marks_and_labels = widgets.HTML(
            value=f"""
        <div class="widget-container">
            <div class="slider-container">
                <div class="tick-wrapper">
                    {ticks_html}
                </div>
            </div>
        </div>
        """
        )

        # Combine slider and ticks using VBox
        slider_with_ticks = widgets.VBox(
            [self.interval_slider, tick_marks_and_labels],
            layout=widgets.Layout(
                width="640px", align_items="center", justify_content="center"
            ),
        )

        # Add description widget
        descr_widget = widgets.HTML(
            value="""
            <div style='text-align: center;'>
                <div style='font-size: 20px; font-weight: bold;'>
                    Position the slider to select your processing period:
                </div>
            </div>
            """
        )

        # Arrange the description widget, interval slider, ticks and text widget in a VBox
        vbox = widgets.VBox(
            [
                descr_widget,
                slider_with_ticks,
                self.html_text,
            ],
            layout=widgets.Layout(
                align_items="center", justify_content="center", width="650px"
            ),
        )

        display(core_HTML(custom_css))
        display(vbox)

    def on_slider_change(self, change):

        start, end = change["new"]

        # keep the interval fixed
        expected_end = start + pd.DateOffset(months=11)
        if end != expected_end:
            end = start + pd.DateOffset(months=11)
            self.interval_slider.value = (start, end)

        # update the HTML text underneath the slider
        range = [
            (pd.to_datetime(start)).strftime("%d %b %Y"),
            (
                pd.to_datetime(start) + pd.DateOffset(months=12) - timedelta(days=1)
            ).strftime("%d %b %Y"),
        ]
        focus_time = (start + pd.DateOffset(months=6)).strftime("%b %Y")
        self.html_text.value = f"<b>Selected range:</b> {range[0]} - {range[1]}<br><b>Season center:</b> {focus_time}"

    def get_processing_period(self):

        start = pd.to_datetime(self.interval_slider.value[0])
        end = start + pd.DateOffset(months=12) - timedelta(days=1)

        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")
        logger.info(f"Selected processing period: {start} to {end}")

        return TemporalContext(start, end)
