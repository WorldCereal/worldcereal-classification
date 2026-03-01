from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

import ipywidgets as widgets
import pandas as pd
from IPython.core.display import HTML as core_HTML
from IPython.display import display
from loguru import logger
from openeo_gfmap import TemporalContext


@dataclass(frozen=True)
class SeasonSelection:
    season_window: TemporalContext
    processing_period: TemporalContext


class date_slider:
    """Interactive season picker that derives the processing window automatically.

    Users pick the exact growing-season window (max 12 months, aligned to calendar months)
    and we derive a processing period that always spans one year ending on the season end.
    """

    def __init__(
        self,
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2025, 12, 31),
        show_year=True,
        display_interval=2,
        title="Select growing season window:",
        max_window_months: int = 12,
        default_window_months: int = 6,
        *,
        year_selector: bool = True,
        year_selector_years: Optional[Sequence[int]] = None,
        year_selector_initial: Optional[int] = None,
        year_selector_months_before: int = 6,
        year_selector_months_after: int = 6,
    ):
        self.show_year = show_year
        self.max_window_months = max(1, max_window_months)
        self.processing_months = 12
        self.default_window_months = max(
            1, min(default_window_months, self.max_window_months)
        )
        self.display_interval = max(1, display_interval)
        self._title = title
        self._base_start_date = start_date
        self._base_end_date = end_date
        self.year_selector = year_selector
        self.year_selector_months_before = max(0, year_selector_months_before)
        self.year_selector_months_after = max(0, year_selector_months_after)
        self.interval_slider: widgets.SelectionRangeSlider
        self.html_text: widgets.HTML
        self._year_dropdown: Optional[widgets.Dropdown] = None

        custom_css = self._build_custom_css()
        descr_widget = widgets.HTML(
            value=f"""
            <div style='text-align: center;'>
                <div style='font-size: 20px; font-weight: bold;'>
                    {title}
                </div>
            </div>
            """
        )
        self._slider_output = widgets.Output()
        container_children = [descr_widget]

        if self.year_selector:
            years = (
                list(range(start_date.year, end_date.year + 1))
                if not year_selector_years
                else sorted(set(year_selector_years))
            )
            if not years:
                years = [datetime.utcnow().year]
            initial_year = year_selector_initial or max(years)
            if initial_year not in years:
                initial_year = max(years)
            self._year_dropdown = widgets.Dropdown(
                options=years,
                value=initial_year,
                description="Season year",
                layout=widgets.Layout(width="220px"),
            )
            self._year_dropdown.observe(self._on_year_selector_change, names="value")
            container_children.append(self._year_dropdown)
            self._current_year = initial_year
        else:
            self._current_year = None

        container_children.append(self._slider_output)
        container = widgets.VBox(
            container_children,
            layout=widgets.Layout(
                align_items="center", justify_content="center", width="780px"
            ),
        )

        display(core_HTML(custom_css))
        display(container)
        self._render_slider(self._current_year)

    def _build_custom_css(self) -> str:
        return """
        <style>
        .widget-container {
            padding-left: 10px;
            box-sizing: border-box;
            width: 700px;
            margin: 0 auto;
            position: relative;
        }
        .slider-container {
            position: relative;
            width: 100%;
            margin: 0 auto;
        }
        .slider-container .tick-wrapper {
            position: relative;
            width: 100%;
            height: 40px;
        }
        .slider-container .tick-mark {
            position: absolute;
            bottom: 25px;
            transform: translateX(-50%);
            font-size: 14px;
            font-weight: bold;
        }
        .slider-container .tick-label {
            position: absolute;
            bottom: 0;
            transform: translateX(-50%);
            font-size: 10px;
            text-align: center;
            line-height: 1.2em;
        }
        </style>
        """

    def _render_slider(self, focus_year: Optional[int]):
        if focus_year is not None:
            center = pd.Timestamp(focus_year, 1, 1)
            start_date = (
                center - pd.DateOffset(months=self.year_selector_months_before)
            ).to_pydatetime()
            end_date = (
                center + pd.DateOffset(months=11 + self.year_selector_months_after)
            ).to_pydatetime()
        else:
            start_date = self._base_start_date
            end_date = self._base_end_date

        with self._slider_output:
            self._slider_output.clear_output()
            slider_widget = self._build_slider(start_date, end_date, focus_year)
            display(slider_widget)

    def _build_slider(
        self,
        start_date: datetime,
        end_date: datetime,
        focus_year: Optional[int] = None,
    ) -> widgets.VBox:
        dates = pd.date_range(start_date, end_date, freq="MS")
        if self.show_year:
            options = [(date.strftime("%b %Y"), date) for date in dates]
        else:
            options = [(date.strftime("%b"), date) for date in dates]

        default_start_index = 0
        if focus_year is not None:
            for idx, date in enumerate(dates):
                if date.year == focus_year:
                    default_start_index = idx
                    break
        default_end_index = min(
            len(dates) - 1, default_start_index + self.default_window_months - 1
        )

        self.interval_slider = widgets.SelectionRangeSlider(
            options=options,
            value=(dates[default_start_index], dates[default_end_index]),
            orientation="horizontal",
            description="",
            continuous_update=False,
            behaviour="drag",
            style={"handle_color": "dodgerblue"},
            layout=widgets.Layout(width="700px", margin="0 0 0 10px"),
            readout=False,
        )
        self.interval_slider.observe(self.on_slider_change, names="value")

        self.html_text = widgets.HTML(
            value="",
            placeholder="HTML placeholder",
            description="",
            layout=widgets.Layout(justify_content="center", display="flex"),
        )

        frequency = f"{self.display_interval}MS"
        tick_dates = pd.date_range(
            start_date,
            pd.to_datetime(end_date) + pd.DateOffset(months=1),
            freq=frequency,
        )
        if self.show_year:
            tick_labels = [date.strftime("%b %Y") for date in tick_dates]
        else:
            tick_labels = [date.strftime("%b") for date in tick_dates]
        n_labels = max(1, len(tick_labels))
        ticks_html = ""
        for i, label in enumerate(tick_labels):
            position = 0 if n_labels == 1 else (i / (n_labels - 1)) * 100
            parts = label.split(" ") if " " in label else [label, ""]
            top_label = parts[0]
            bottom_label = parts[1] if len(parts) > 1 else ""
            ticks_html += f"""
            <div class="tick-mark" style="left: {position}%; ">|</div>
            <div class="tick-label" style="left: {position}%; ">{top_label}<br>{bottom_label}</div>
            """

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

        slider_with_ticks = widgets.VBox(
            [self.interval_slider, tick_marks_and_labels],
            layout=widgets.Layout(
                width="740px", align_items="center", justify_content="center"
            ),
        )

        vbox = widgets.VBox(
            [slider_with_ticks, self.html_text],
            layout=widgets.Layout(
                align_items="center", justify_content="center", width="750px"
            ),
        )

        self._update_summary(*self.interval_slider.value)
        return vbox

    def _on_year_selector_change(self, change):
        if (
            change.get("name") == "value"
            and change.get("new") is not None
            and change.get("new") != change.get("old")
        ):
            self._current_year = change["new"]
            self._render_slider(self._current_year)

    def on_slider_change(self, change):
        start, end = change["new"]
        months_selected = self._get_month_span(start, end)
        if months_selected > self.max_window_months:
            clamped_end = start + pd.DateOffset(months=self.max_window_months - 1)
            self.interval_slider.value = (start, clamped_end)
            return

        self._update_summary(start, end)

    def get_selected_dates(self):
        logger.info(
            "Processing period derived from season window: {} to {}",
            self._processing_period.start_date,
            self._processing_period.end_date,
        )
        return self._processing_period

    def get_season_window(self):
        """Return the exact season window selected by the user."""

        return self._season_window

    def get_selection(self) -> SeasonSelection:
        """Return both the season window and derived processing period."""

        return SeasonSelection(
            season_window=self._season_window,
            processing_period=self._processing_period,
        )

    def _update_summary(self, start: pd.Timestamp, end: pd.Timestamp):
        season_start = start.replace(day=1)
        season_end = self._get_last_day_of_month(end)

        processing_start_month = season_end.replace(day=1) - pd.DateOffset(
            months=self.processing_months - 1
        )
        processing_start = processing_start_month
        processing_end = season_end

        self._season_window = TemporalContext(
            season_start.strftime("%Y-%m-%d"), season_end.strftime("%Y-%m-%d")
        )
        self._processing_period = TemporalContext(
            processing_start.strftime("%Y-%m-%d"), processing_end.strftime("%Y-%m-%d")
        )

        season_range = [
            (
                season_start.strftime("%d %b %Y")
                if self.show_year
                else season_start.strftime("%d %b")
            ),
            (
                season_end.strftime("%d %b %Y")
                if self.show_year
                else season_end.strftime("%d %b")
            ),
        ]
        processing_range = [
            (
                processing_start.strftime("%d %b %Y")
                if self.show_year
                else processing_start.strftime("%d %b")
            ),
            (
                processing_end.strftime("%d %b %Y")
                if self.show_year
                else processing_end.strftime("%d %b")
            ),
        ]

        self.html_text.value = (
            f"<b>Season window:</b> {season_range[0]} - {season_range[1]}"
            f"<br><b>Processing period (auto):</b> {processing_range[0]} - {processing_range[1]}"
        )

    @staticmethod
    def _get_last_day_of_month(date: pd.Timestamp) -> pd.Timestamp:
        return (date + pd.DateOffset(months=1)) - timedelta(days=1)

    @staticmethod
    def _get_month_span(start: pd.Timestamp, end: pd.Timestamp) -> int:
        months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        return max(1, months)


class season_slider(date_slider):
    """Class that provides a slider for selecting a season.
    Differences with date_slider:
    -  we only show two years, starting from june and ending in june
    -  we don't show a year label in the slider, only the month
    """

    def __init__(self):
        # Set the start and end dates for the slider
        # The slider will cover a period from June 2017 to June 2019
        start_date = pd.to_datetime(("2017-07-01"))
        end_date = pd.to_datetime(("2019-05-01"))

        # Call the parent class constructor
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            show_year=False,
            display_interval=1,
            title="Select season:",
            default_window_months=6,
        )
