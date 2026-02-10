"""
Interactive application for the WorldCereal training and inference workflow.

Usage:
    app = WorldCerealTrainingApp.run()

This module provides an interactive widget-based interface for:
1. Retrieving reference data
2. Inspecting and cleaning datasets
3. Aligning samples to a season window
4. Selecting crops and preparing training data
5. Computing embeddings
6. Training a model
7. Deploying a model
8. Generating a map
9. Visualizing results
"""

import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from notebook_utils.classifier import (
    align_extractions_to_season,
    compute_seasonal_presto_embeddings,
    train_seasonal_torch_head,
)
from notebook_utils.croptypepicker import CropTypePicker, apply_croptypepicker_to_df
from notebook_utils.dateslider import date_slider as season_slider
from notebook_utils.extractions import (
    get_band_statistics,
    query_extractions,
    retrieve_extractions_extent,
    visualize_timeseries,
)
from notebook_utils.production import bbox_extent_to_gdf, merge_maps, run_map_production
from notebook_utils.seasons import retrieve_worldcereal_seasons, valid_time_distribution
from notebook_utils.visualization import visualize_products
from openeo_gfmap import TemporalContext
from openeo_gfmap.backend import cdse_connection
from tabulate import tabulate

from worldcereal.openeo.inference import load_model_artifact
from worldcereal.openeo.preprocessing import WORLDCEREAL_BANDS
from worldcereal.openeo.workflow_config import WorldCerealWorkflowConfig
from worldcereal.parameters import WorldCerealProductType
from worldcereal.utils.legend import (
    ewoc_code_to_label,
    get_legend,
    translate_ewoc_codes,
)
from worldcereal.utils.map import ui_map
from worldcereal.utils.upload import OpenEOArtifactHelper


class WorldCerealTrainingApp:
    @classmethod
    def run(cls) -> "WorldCerealTrainingApp":
        """Instantiate and display the training application."""
        app = cls()
        display(app.tabs_container)
        return app

    def __init__(self):
        self.workflow_mode = "full"
        self._nav_buttons: List[Dict[str, widgets.Button]] = []

        # Tab 1 variables
        self.private_extractions_path = Path(
            "./extractions/worldcereal_merged_extractions.parquet"
        )
        self.tab1_df: Optional[pd.DataFrame] = None  # Tab 1 working results
        self.tab1_saved = False
        self.append_next_query = False

        # Tab 3 variables
        self.processing_period: Optional[TemporalContext] = None
        self.season_window: Optional[TemporalContext] = None
        self.season_id: Optional[str] = None

        # Path to resulting files
        self.training_df_path: Optional[Path] = None
        self.embeddings_df_path: Optional[Path] = None
        self.head_output_path: Optional[Path] = None
        self.head_package_path: Optional[Path] = None

        # Final outputs per tab that need to be stored for use in later tabs
        self.tab2_df: Optional[pd.DataFrame] = None
        self.tab3_df: Optional[pd.DataFrame] = None
        self.tab4_df: Optional[pd.DataFrame] = None
        self.tab4_confirmed = False
        self.tab5_df: Optional[pd.DataFrame] = None
        self.tab7_model_url: Optional[str] = None
        self.tab8_processing_period: Optional[TemporalContext] = None
        self.tab8_season_window: Optional[TemporalContext] = None
        self.tab8_results: Optional[Path] = None
        self.tab9_merged_paths: Dict[str, Path] = {}
        self.tab9_model_url: Optional[str] = None

        self.cdse_auth_cleared = False

        # Widgets per tab
        self.tab0_widgets: Dict[str, Any] = {}
        self.tab1_widgets: Dict[str, Any] = {}
        self.tab2_widgets: Dict[str, Any] = {}
        self.tab3_widgets: Dict[str, Any] = {}
        self.tab4_widgets: Dict[str, Any] = {}
        self.tab5_widgets: Dict[str, Any] = {}
        self.tab6_widgets: Dict[str, Any] = {}
        self.tab7_widgets: Dict[str, Any] = {}
        self.tab8_widgets: Dict[str, Any] = {}
        self.tab9_widgets: Dict[str, Any] = {}

        self._run_proj_fix()

        display(
            HTML(
                """
                <style>
                    .jp-OutputArea-output pre {
                        white-space: pre-wrap !important;
                        word-wrap: break-word !important;
                        overflow-wrap: break-word !important;
                    }
                </style>
                """
            )
        )

        tab0 = self._build_tab0_workflow()
        tab1 = self._build_tab1_retrieve_reference_data()
        tab2 = self._build_tab2_select_training_data()
        tab3 = self._build_tab3_season_alignment()
        tab4 = self._build_tab4_crop_selection()
        tab5 = self._build_tab5_compute_embeddings()
        tab6 = self._build_tab6_train_model()
        tab7 = self._build_tab7_deploy_model()
        tab8 = self._build_tab8_generate_map()
        tab9 = self._build_tab9_visualize_results()

        self.tab_pages = {
            "workflow": tab0,
            "retrieve": tab1,
            "inspect": tab2,
            "season": tab3,
            "crop": tab4,
            "embeddings": tab5,
            "train": tab6,
            "deploy": tab7,
            "generate": tab8,
            "visualize": tab9,
        }

        self.tabs = widgets.Tab(children=[])
        self.tabs.layout = widgets.Layout(width="100%", max_width="100%")

        self._tab_style = widgets.HTML(
            value="""
<style>
.p-TabBar-tabLabel {
    font-size: 14px;
    white-space: normal;
}
.p-TabBar-tab {
    max-width: none;
}
</style>
"""
        )

        self.welcome_screen = tab0
        self.tabs_container = widgets.VBox([self.welcome_screen])
        self.tabs_container.layout = widgets.Layout(
            width="100%", max_width="100%", overflow="hidden", align_items="stretch"
        )
        self.tabs.observe(self._on_tab_change, names="selected_index")
        self._update_nav_buttons()

    def _run_proj_fix(self):
        import os
        import sys

        # Set PROJ environment variables to avoid PROJ database version conflicts
        # This ensures PROJ uses the database from the current conda environment
        proj_path = os.path.join(sys.prefix, "share", "proj")
        os.environ["PROJ_LIB"] = proj_path
        os.environ["PROJ_DATA"] = proj_path

    def _apply_workflow_tabs(self):
        """Update visible tabs based on the selected workflow."""
        if self.workflow_mode == "inference-only":
            children = [
                self.tab_pages["deploy"],
                self.tab_pages["generate"],
                self.tab_pages["visualize"],
            ]
            titles = [
                "1. Deploy Model",
                "2. Generate Map",
                "3. Visualize Map",
            ]
        else:
            children = [
                self.tab_pages["retrieve"],
                self.tab_pages["inspect"],
                self.tab_pages["season"],
                self.tab_pages["crop"],
                self.tab_pages["embeddings"],
                self.tab_pages["train"],
                self.tab_pages["deploy"],
                self.tab_pages["generate"],
                self.tab_pages["visualize"],
            ]
            titles = [
                "1. Retrieve Data",
                "2. Inspect & Clean",
                "3. Season Alignment",
                "4. Crop Selection",
                "5. Compute Embeddings",
                "6. Train Model",
                "7. Deploy Model",
                "8. Generate Map",
                "9. Visualize Map",
            ]

        for child in children:
            if hasattr(child, "layout"):
                child.layout = widgets.Layout(
                    width="100%", max_width="100%", overflow="hidden"
                )

        self.tabs.children = children
        for index, title in enumerate(titles):
            self.tabs.set_title(index, title)
        self._update_nav_buttons()

    def _build_tab_navigation(self) -> widgets.VBox:
        """Create navigation controls for switching tabs."""
        prev_button = widgets.Button(
            description="Go back to previous step",
            button_style="",
            icon="arrow-left",
            layout=widgets.Layout(width="220px"),
        )
        next_button = widgets.Button(
            description="Proceed to next step",
            button_style="primary",
            icon="arrow-right",
            layout=widgets.Layout(width="200px"),
        )

        prev_button.on_click(self._on_prev_tab)
        next_button.on_click(self._on_next_tab)

        self._nav_buttons.append({"prev": prev_button, "next": next_button})

        return widgets.VBox(
            [widgets.HBox([prev_button, next_button])],
            layout=widgets.Layout(margin="16px 0 0 0"),
        )

    def _on_prev_tab(self, _=None):
        if self.tabs is None:
            return
        if self.tabs.selected_index is None:
            return
        if self.tabs.selected_index > 0:
            self.tabs.selected_index -= 1
        self._update_nav_buttons()

    def _on_next_tab(self, _=None):
        if self.tabs is None:
            return
        if self.tabs.selected_index is None:
            return
        if self.tabs.selected_index < len(self.tabs.children) - 1:
            self.tabs.selected_index += 1
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        if self.tabs is None or self.tabs.selected_index is None:
            return
        current = self.tabs.selected_index
        total = len(self.tabs.children)
        for nav in self._nav_buttons:
            nav["prev"].disabled = current <= 0
            nav["next"].disabled = current >= total - 1

    # =========================================================================
    # Tab 0: Choose Workflow
    # =========================================================================

    def _build_tab0_workflow(self) -> widgets.VBox:
        """Build welcome screen: Choose workflow mode and launch."""
        header = widgets.HTML(
            value="<h2>Welcome to the WorldCereal Model Training and Inference Application</h2>"
            "<p>Select whether you want to train a custom model or run inference only, then launch the application.</p>"
        )

        workflow_mode_radio = widgets.RadioButtons(
            options=[
                ("Full workflow (train custom model)", "full"),
                ("Inference only (use existing model)", "inference-only"),
            ],
            value="full",
            description="Workflow:",
        )

        select_button = widgets.Button(
            description="Launch application",
            button_style="primary",
            icon="check",
            layout=widgets.Layout(width="200px"),
        )

        self.tab0_widgets = {
            "workflow_mode_radio": workflow_mode_radio,
            "select_button": select_button,
        }

        workflow_mode_radio.observe(self._on_workflow_mode_change, names="value")
        select_button.on_click(self._on_workflow_mode_select)

        return widgets.VBox(
            [
                header,
                workflow_mode_radio,
                widgets.HBox([select_button]),
            ]
        )

    def _on_workflow_mode_change(self, change):
        """Handle workflow mode selection changes."""
        self.workflow_mode = change["new"]

    def _on_workflow_mode_select(self, button):
        """Apply workflow choice and launch the application."""
        self._apply_workflow_tabs()
        if self.tabs_container is not None and self._tab_style is not None:
            self.tabs_container.children = [self._tab_style, self.tabs]
        mode_label = (
            "Full workflow" if self.workflow_mode == "full" else "Inference only"
        )
        print(f"Mode '{mode_label}' selected. Launching the application...")
        self._update_tab2_state()
        self._update_tab3_state()
        self._update_tab4_state()
        self._update_tab5_state()
        self._update_tab6_state()
        self._update_tab7_state()
        self._update_tab8_state()
        self._update_tab9_state()

    # =========================================================================
    # Tab 1: Retrieve Reference Data
    # =========================================================================

    def _build_tab1_retrieve_reference_data(self) -> widgets.VBox:
        """Build Tab 1: Retrieve reference data."""
        header = widgets.HTML(value="<h2>Retrieve Reference Data</h2>")

        status_message = widgets.HTML(
            value="<i>In this first step we gather reference data for training your model.</i>"
        )

        background_info = self._info_callout(
            "Reference data is defined here as observations of land cover/crop types at specific locations and times, for which satellite data has been extracted and is available in the WorldCereal system.<br><br>"
            "For training a crop identification model, you can use a combination of:<br>"
            "   - <b>publicly available reference data</b> harmonized by the WorldCereal consortium;<br>"
            "   - your own <b>private reference data</b>.<br><br>"
            "The image below provides you with a quick overview of the locations where publicly exposed reference data is already available for you."
        )

        extent_message = widgets.HTML(
            value="<i>For your information, the map below shows the spatial extent of all publicly available reference datasets.</i>"
        )

        extent_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        with extent_output:
            extent_output.clear_output()
            try:
                extent_gdf, extent_map = retrieve_extractions_extent()
                print(f"Total number of datasets available: {len(extent_gdf)}.")
                display(extent_map)

            except Exception as exc:
                print(f"Failed to load public extractions extent: {exc}")

        query_message = widgets.HTML(
            value="<i>Work through the steps 1 to 4 below to specify which data to retrieve.</i>"
        )

        query_explanation = self._info_callout(
            "Once you have specified your query parameters, click the <b>Run Query</b> button below to retrieve reference data matching your criteria.<br><br>"
            "You can run multiple queries to iteratively build up a training dataset, using the <b> Save and Add More button</b>.<br><br>"
            "Once you are satisfied with the results, click the <b>Save and Continue</b> button to store your data and proceed to the next step."
        )

        aoi_message = widgets.HTML(
            value="<i>Select an Area of Interest (AOI) on the map below to spatially constrain your query.</i>"
        )

        aoi_explanation = self._info_callout(
            "If no AOI is selected, the query will consider all available data globally.<br>"
            "       ⚠️ WARNING: This will result in a very large query and may take a long time to complete.<br><br>"
            "You can draw a rectangle using the drawing tools on the left side of the map.<br>"
            "The app will automatically store the coordinates of the last rectangle you drew on the map.<br><br>"
            "Alternatively, you can also upload a vector file (either zipped shapefile or GeoPackage) delineating your area of interest.<br>"
            "In case your vector file contains multiple polygons or points, the total bounds will be automatically computed and serve as your AOI.<br>"
            "Files containing only a single point are not allowed.<br><br>"
        )

        aoi_map = ui_map(display_ui=False)

        buffer_explanation = self._info_callout(
            "By default we apply a 250 km buffer around your selected AOI to ensure sufficient reference data is retrieved.<br>"
        )
        buffer_input = widgets.IntText(
            value=250,
            description="Buffer (km):",
        )

        include_public_checkbox = widgets.Checkbox(
            value=True,
            description="Include public collections",
        )

        private_explanation = self._info_callout(
            "<b>Note on the use of private data</b><br>"
            "In case you would like to include your private data, you will need to:<br>"
            "   1. Add your reference data to the <a href='https://rdm.esa-worldcereal.org/' target='_blank' rel='noopener'>WorldCereal RDM</a>.<br> Detailed instructions can be found <a href='https://worldcereal.github.io/worldcereal-documentation/rdm/upload.html' target='_blank' rel='noopener'>HERE</a>.<br>"
            "   2. Extract satellite data for your reference data by following the steps in <a href='https://github.com/WorldCereal/worldcereal-classification/blob/main/notebooks/worldcereal_private_extractions.ipynb' target='_blank' rel='noopener'>THIS NOTEBOOK</a>.<br>"
        )

        include_private_checkbox = widgets.Checkbox(
            value=False,
            description="Include private collections",
        )

        private_path_display = widgets.HTML(
            value=f"<b>Path to private extractions .parquet file:</b> {self.private_extractions_path}"
        )
        private_path_display.layout.display = "none"

        private_path_input = widgets.Text(
            value=str(self.private_extractions_path),
            placeholder="/path/to/private/extractions/worldcereal_merged_extractions.parquet",
            description="Custom path:",
            layout=widgets.Layout(width="800px", display="none"),
        )

        private_path_button = widgets.Button(
            description="Edit path",
            icon="exclamation-triangle",
            button_style="danger",
            tooltip="Advanced: change only if you know what you're doing",
            layout=widgets.Layout(width="140px", display="none"),
        )

        ref_ids_explanation = self._info_callout(
            "Optionally, specify a comma-separated list of dataset IDs to limit your query.<br>"
            "If left empty, all available reference data will be considered."
        )

        ref_ids_input = widgets.Textarea(
            value="",
            placeholder="ref_id_1, ref_id_2",
            description="Ref IDs:",
            layout=widgets.Layout(width="100%", height="80px"),
        )

        crop_only_explanation = self._info_callout(
            "By default, only temporary crop samples are retrieved.<br>"
            "This effectively means that you will only be able to train a model distinguishing different types of temporary crops.<br>"
            "Uncheck the 'Only temporary crop samples' option below to include all land cover classes in your query.<br>"
        )

        crop_only_checkbox = widgets.Checkbox(
            value=True,
            description="Only temporary crop samples",
            layout=widgets.Layout(width="350px"),
        )

        crop_types_explanation = self._info_callout(
            "Optionally, select specific land cover/crop types to limit your query.<br>"
            "Clicking the button below will open a crop type selection dialog, which takes a while to load.<br>"
            "Check all crops you want to include in your query.<br>"
            "Make use of the Expand buttons to dive into the crop type hierarchy for more detailed selection.<br>"
            "When you are done selecting, click the green Apply button.<br><br>"
            "When you skip this step, all crop types will be considered.<br>"
        )

        select_crops_button = widgets.Button(
            description="Select crops",
            button_style="primary",
            icon="list",
            layout=widgets.Layout(width="160px"),
        )

        croptype_picker = None
        croptype_picker_container = widgets.VBox()
        croptype_picker_status = widgets.HTML(value="")

        run_query_button = widgets.Button(
            description="Run Query",
            button_style="primary",
            icon="search",
            layout=widgets.Layout(width="400px", height="100px"),
        )

        save_continue_button = widgets.Button(
            description="Save & Continue",
            button_style="success",
            icon="arrow-right",
            layout=widgets.Layout(width="220px", height="60px"),
            disabled=True,
        )

        save_add_button = widgets.Button(
            description="Save & Add More",
            button_style="info",
            icon="plus",
            layout=widgets.Layout(width="220px", height="60px"),
            disabled=True,
        )

        discard_button = widgets.Button(
            description="Discard & Start Over",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="220px", height="60px"),
            disabled=True,
        )

        query_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #ccc",
                padding="10px",
            )
        )
        config_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="100px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab1_widgets = {
            "extent_output": extent_output,
            "aoi_map": aoi_map,
            "buffer_input": buffer_input,
            "include_public_checkbox": include_public_checkbox,
            "include_private_checkbox": include_private_checkbox,
            "private_path_display": private_path_display,
            "private_path_input": private_path_input,
            "private_path_button": private_path_button,
            "ref_ids_input": ref_ids_input,
            "select_crops_button": select_crops_button,
            "croptype_picker": croptype_picker,
            "croptype_picker_container": croptype_picker_container,
            "croptype_picker_status": croptype_picker_status,
            "crop_only_checkbox": crop_only_checkbox,
            "run_query_button": run_query_button,
            "query_output": query_output,
            "save_continue_button": save_continue_button,
            "save_add_button": save_add_button,
            "discard_button": discard_button,
            "config_output": config_output,
        }

        run_query_button.on_click(self._on_run_query_click)
        save_continue_button.on_click(self._on_save_continue_click)
        save_add_button.on_click(self._on_save_add_click)
        discard_button.on_click(self._on_discard_click)
        include_private_checkbox.observe(self._on_private_toggle, names="value")
        private_path_button.on_click(self._on_private_edit_click)
        crop_only_checkbox.observe(self._on_crop_only_toggle, names="value")
        select_crops_button.on_click(self._on_select_crops_click)

        return widgets.VBox(
            [
                header,
                status_message,
                background_info,
                widgets.HTML(
                    "<h3 style='margin: 10px 0;'>Public extractions extent</h3>"
                ),
                extent_message,
                extent_output,
                widgets.HTML("<h3 style='margin: 10px 0;'>Query reference data</h3>"),
                query_message,
                query_explanation,
                widgets.HTML("<b>1) AOI selection:</b>"),
                aoi_message,
                aoi_explanation,
                aoi_map.map,
                aoi_map.output,
                buffer_explanation,
                buffer_input,
                widgets.HTML("<b>2) Data sources selection:</b>"),
                private_explanation,
                include_public_checkbox,
                include_private_checkbox,
                private_path_display,
                widgets.HBox([private_path_button]),
                private_path_input,
                widgets.HTML("<b>3) Optional dataset filtering by ID:</b>"),
                ref_ids_explanation,
                ref_ids_input,
                widgets.HTML("<b>4) Optional crop type filtering:</b>"),
                crop_only_explanation,
                crop_only_checkbox,
                crop_types_explanation,
                widgets.HBox([select_crops_button]),
                croptype_picker_status,
                croptype_picker_container,
                widgets.HBox([run_query_button]),
                widgets.HTML("<h3 style='margin: 10px 0;'>Query results</h3>"),
                query_output,
                widgets.HTML("<h3 style='margin: 10px 0;'>Next actions</h3>"),
                widgets.HBox([save_continue_button, save_add_button, discard_button]),
                config_output,
                self._build_tab_navigation(),
            ]
        )

    def _parse_list_input(self, value: str) -> Optional[List[str]]:
        """Parse a comma-separated list input."""
        if not value.strip():
            return None
        return [item.strip() for item in value.split(",") if item.strip()]

    def _parse_crop_types(self, value: str) -> Optional[List[int]]:
        """Parse crop types as list of integers."""
        items = self._parse_list_input(value)
        if not items:
            return None
        parsed = []
        for item in items:
            try:
                parsed.append(int(item))
            except ValueError:
                continue
        return parsed if parsed else None

    def _get_crop_only_codes(self) -> List[int]:
        """Return ewoc codes that start with '11'."""
        legend = get_legend()
        codes = list(legend.index.astype(str))
        return [np.int64(code) for code in codes if code.startswith("11")]

    def _build_croptype_picker(self, crop_only: bool) -> CropTypePicker:
        """Create a CropTypePicker with optional crop-only filtering."""
        ewoc_codes = self._get_crop_only_codes() if crop_only else None
        return CropTypePicker(
            ewoc_codes=ewoc_codes,
            expand=False,
            display_ui=False,
            selection_modes=["Include"],
        )

    def _on_crop_only_toggle(self, change):
        """Rebuild the crop type picker when crop-only selection changes."""
        crop_only = change["new"]
        crop_picker = self.tab1_widgets.get("croptype_picker")
        if crop_picker is None:
            return
        status = self.tab1_widgets.get("croptype_picker_status")
        if status is not None:
            status.value = "<i>Loading crop type picker, this can take a moment...</i>"
        picker_container = self.tab1_widgets.get("croptype_picker_container")
        if picker_container is not None:
            picker_container.children = []
        crop_picker = self._build_croptype_picker(crop_only=crop_only)
        self.tab1_widgets["croptype_picker"] = crop_picker
        if picker_container is not None:
            picker_container.children = [crop_picker.widget]
        if status is not None:
            status.value = ""

    def _on_select_crops_click(self, button):
        """Initialize and show the crop type picker on demand."""
        crop_only = self.tab1_widgets["crop_only_checkbox"].value
        status = self.tab1_widgets.get("croptype_picker_status")
        if status is not None:
            status.value = "<i>Loading crop type picker...</i>"
        picker_container = self.tab1_widgets.get("croptype_picker_container")
        if picker_container is not None:
            picker_container.children = []
        crop_picker = self._build_croptype_picker(crop_only=crop_only)
        self.tab1_widgets["croptype_picker"] = crop_picker
        if picker_container is not None:
            picker_container.children = [crop_picker.widget]
        if status is not None:
            status.value = ""

    def _on_private_toggle(self, change):
        """Show or hide private path controls based on checkbox."""
        display_widget = self.tab1_widgets["private_path_display"]
        input_widget = self.tab1_widgets["private_path_input"]
        button_widget = self.tab1_widgets["private_path_button"]

        if change["new"]:
            display_widget.layout.display = "block"
            button_widget.layout.display = "block"
            input_widget.layout.display = "none"
            button_widget.description = "Edit path"
        else:
            display_widget.layout.display = "none"
            button_widget.layout.display = "none"
            input_widget.layout.display = "none"

    def _on_private_edit_click(self, button):
        """Toggle editing for private path and apply changes."""
        display_widget = self.tab1_widgets["private_path_display"]
        input_widget = self.tab1_widgets["private_path_input"]

        if input_widget.layout.display == "none":
            input_widget.value = str(self.private_extractions_path)
            input_widget.layout.display = "block"
            button.description = "Apply path"
        else:
            new_path = input_widget.value.strip() or "./extractions"
            self.private_extractions_path = Path(new_path)
            display_widget.value = f"<b>Path to private extractions .parquet file:</b> {self.private_extractions_path}"
            input_widget.layout.display = "none"
            button.description = "Edit path"

    def _on_run_query_click(self, button):
        """Run the extractions query and display summary output."""
        query_output = self.tab1_widgets["query_output"]
        with query_output:
            query_output.clear_output()

        bbox_poly = None
        try:
            bbox_poly = self.tab1_widgets["aoi_map"].get_polygon_latlon()
        except Exception as exc:
            with query_output:
                print(f"No AOI selected yet. Proceeding without AOI. ({exc})")

        buffer_value = (
            self.tab1_widgets["buffer_input"].value * 1000
        )  # convert km to meters
        include_public = self.tab1_widgets["include_public_checkbox"].value
        include_private = self.tab1_widgets["include_private_checkbox"].value
        private_parquet_path = (
            self.private_extractions_path if include_private else None
        )

        if not include_public and not include_private:
            with query_output:
                print(
                    "Please select at least one data source (public or private) before running the query."
                )
            return

        ref_ids = self._parse_list_input(self.tab1_widgets["ref_ids_input"].value)
        crop_types = None
        croptype_picker = self.tab1_widgets.get("croptype_picker")
        if croptype_picker is not None:
            try:
                croptype_picker.apply_selection()
                if not croptype_picker.croptypes.empty:
                    crop_types = croptype_picker.croptypes.index.tolist()
            except Exception:
                crop_types = None
        filter_cropland = self.tab1_widgets["crop_only_checkbox"].value
        try:
            with query_output:
                print("Query in progress. This may take a few moments...")
                result_df = query_extractions(
                    bbox_poly=bbox_poly,
                    buffer=buffer_value,
                    filter_cropland=filter_cropland,
                    include_public=include_public,
                    private_parquet_path=private_parquet_path,
                    ref_ids=ref_ids,
                    crop_types=crop_types,
                )
                print("Query complete!")

                if result_df.empty:
                    print(
                        "⚠️ No data returned by the query, cannot continue. \n"
                        "Please adjust your query and try again.\n"
                        "Possible options: \n"
                        "   - change your area and/or increase the buffer\n"
                        "   - contribute private data to the system\n"
                        "   - include more crop types\n"
                    )
                    self.tab1_widgets["save_continue_button"].disabled = True
                    self.tab1_widgets["save_add_button"].disabled = True
                    return

                if self.append_next_query and self.tab1_df is not None:
                    print("Appending results to existing extractions.")
                    self.tab1_df = pd.concat([self.tab1_df, result_df])
                    self.tab1_df = self.tab1_df.drop_duplicates(
                        subset=["sample_id", "timestamp"], keep="first"
                    )
                else:
                    self.tab1_df = result_df
                self.tab1_saved = False

                crop_type_count = None
                if "ewoc_code" in self.tab1_df.columns:
                    crop_type_count = self.tab1_df["ewoc_code"].nunique()

                if crop_type_count is not None and crop_type_count <= 1:
                    print(
                        "⚠️ Only one crop type detected. Please append more data or try again."
                    )
                    self.tab1_widgets["save_continue_button"].disabled = True
                else:
                    self.tab1_widgets["save_continue_button"].disabled = False
        except Exception as exc:
            with query_output:
                print(f"Query failed: {exc}")
            return

        self.tab1_widgets["save_add_button"].disabled = False
        self.tab1_widgets["discard_button"].disabled = False

    def _on_save_continue_click(self, button):
        """Save extractions and continue to next step."""
        output = self.tab1_widgets["config_output"]
        with output:
            print("Extractions saved. You can proceed to the next step.")
        self.tab1_saved = True
        self.tab2_df = None
        self.append_next_query = False
        self._update_tab2_state()

    def _on_save_add_click(self, button):
        """Save current extractions and allow another query to add more."""
        output = self.tab1_widgets["config_output"]
        with output:
            print("Extractions saved. You can run another query to add more.")
        self.append_next_query = True
        self.tab1_saved = False

    def _on_discard_click(self, button):
        """Discard extractions and reset state."""
        self.tab1_df = None
        self.tab1_saved = False
        self.tab2_df = None
        self.append_next_query = False
        output = self.tab1_widgets["config_output"]
        with output:
            output.clear_output()
            print("Extractions discarded. Start over with a new query.")
        self.tab1_widgets["save_continue_button"].disabled = True
        self.tab1_widgets["save_add_button"].disabled = True
        self.tab1_widgets["discard_button"].disabled = True
        self._update_tab2_state()

    # =========================================================================
    # Tab 2: Inspect & Clean Data
    # =========================================================================

    def _build_tab2_select_training_data(self) -> widgets.VBox:
        """Build Tab 2: Select and validate training data."""
        header = widgets.HTML(value="<h2>Optional Data Inspection & Cleaning</h2>")

        status_message = widgets.HTML(
            value="<i>Please retrieve data in Tab 1 first.</i>"
        )
        skip_button = widgets.Button(
            description="Skip this step",
            button_style="warning",
            icon="forward",
            layout=widgets.Layout(width="200px"),
        )
        inspection_info = self._info_callout(
            "In this step you can inspect individual datasets you retrieved in the previous step to get a better understanding of data quality.<br><br>"
            "Use the dropdown below to select a dataset for inspection.<br><br>"
            "Clicking the 'Inspect' button will show you:<br>"
            "    - A table with overall band statistics across all samples.<br>"
            "    - A plot with a few sample time series for the selected dataset.<br><br>"
            "You can adjust which band to visualize, how many samples to show and which crop types to include in the time series graph."
        )
        dataset_dropdown = widgets.Dropdown(
            options=[],
            description="Dataset:",
            layout=widgets.Layout(width="60%"),
        )
        refresh_datasets_button = widgets.Button(
            description="Refresh",
            button_style="info",
            icon="refresh",
            layout=widgets.Layout(width="120px"),
        )
        visualize_button = widgets.Button(
            description="Inspect",
            button_style="primary",
            icon="line-chart",
            layout=widgets.Layout(width="140px"),
        )
        band_options = ["NDVI"] + [
            band for bands in WORLDCEREAL_BANDS.values() for band in bands
        ]
        band_dropdown = widgets.Dropdown(
            options=band_options,
            value="NDVI",
            description="Band:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="260px"),
        )
        n_samples_text = widgets.IntText(
            value=5,
            description="# samples:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )
        crop_type_select = widgets.SelectMultiple(
            options=[],
            description="Crop types:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="60%", height="120px"),
        )
        dataset_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        drop_message = widgets.HTML(
            value="<i>Optionally drop low-quality or irrelevant datasets from the training data.</i>"
        )
        drop_info = self._info_callout(
            "If you found datasets that are of insufficient quality or not relevant for your use case, you can drop them from the training data here.<br><br>"
            "Select one or more datasets from the list below and click the 'Drop selected' button to remove them from the training data."
        )
        drop_select = widgets.SelectMultiple(
            options=[],
            layout=widgets.Layout(width="80%", height="120px"),
        )
        drop_button = widgets.Button(
            description="Drop selected",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="160px"),
        )
        drop_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab2_widgets = {
            "status_message": status_message,
            "skip_button": skip_button,
            "dataset_dropdown": dataset_dropdown,
            "refresh_datasets_button": refresh_datasets_button,
            "visualize_button": visualize_button,
            "band_dropdown": band_dropdown,
            "n_samples_text": n_samples_text,
            "crop_type_select": crop_type_select,
            "dataset_output": dataset_output,
            "drop_select": drop_select,
            "drop_button": drop_button,
            "drop_output": drop_output,
        }

        refresh_datasets_button.on_click(self._on_tab2_refresh_datasets)
        visualize_button.on_click(self._on_tab2_visualize_dataset)
        drop_button.on_click(self._on_tab2_drop_datasets)
        dataset_dropdown.observe(self._on_tab2_dataset_change, names="value")
        skip_button.on_click(self._on_tab2_skip_click)

        return widgets.VBox(
            [
                header,
                status_message,
                skip_button,
                widgets.HTML("<h3>1) Inspect included datasets</h3>"),
                inspection_info,
                widgets.HBox([dataset_dropdown, refresh_datasets_button]),
                widgets.HBox([band_dropdown, n_samples_text]),
                crop_type_select,
                widgets.HBox([visualize_button]),
                dataset_output,
                widgets.HTML("<h3>2) Drop datasets</h3>"),
                drop_message,
                drop_info,
                drop_select,
                widgets.HBox([drop_button]),
                drop_output,
                self._build_tab_navigation(),
            ]
        )

    def _get_tab2_extractions_df(self) -> Optional[pd.DataFrame]:
        """Return the active extractions dataframe for Tab 2."""
        if self.tab2_df is None:
            self._init_tab2_extractions_copy()
        return self.tab2_df

    def _on_tab2_skip_click(self, _=None) -> None:
        """Optionally skip Tab 2 and carry forward Tab 1 extractions."""
        if self.tab1_df is None or not self.tab1_saved:
            output = self.tab2_widgets.get("status_message")
            if output is not None:
                output.value = "<i>Please retrieve and save data in Tab 1 first.</i>"
            return
        self.tab2_df = self.tab1_df.copy()
        self._update_tab2_summary()
        self._on_tab2_refresh_datasets()
        if self.tabs is not None and self.tab_pages.get("season") in self.tabs.children:
            self.tabs.selected_index = self.tabs.children.index(
                self.tab_pages["season"]
            )

    def _build_tab3_season_alignment(self) -> widgets.VBox:
        """Build Tab 3: Season selection and alignment."""
        header = widgets.HTML(value="<h2>Season Selection & Alignment</h2>")

        status_message = widgets.HTML(
            value="<i>Please retrieve data in Tab 1 first.</i>"
        )

        season_generic_info = self._info_callout(
            "Keep in mind that in WorldCereal, we train <b>season-specific</b> crop classifiers.<br>"
            "In this step, you are asked to specify your growing season of interest.<br>"
            "Based on this information, we get rid of irrelevant reference data not matching your season.<br>"
        )
        worldcereal_season_message = widgets.HTML(
            value="<i> Learn about crop seasonality in your area of interest.</i>"
        )
        worldcereal_season_info = self._info_callout(
            "WorldCereal has produced global crop calendars identifying the two most dominant growing seasons for any place on Earth.<br><br>"
            "Clicking the 'Retrieve WorldCereal seasons' button will visualize these seasons for your area of interest.<br><br>"
            'Alternatively, you can consult the <a href="https://ipad.fas.usda.gov/ogamaps/cropcalendar.aspx">USDA crop calendars</a>.'
        )
        seasons_button = widgets.Button(
            description="Retrieve WorldCereal seasons",
            button_style="info",
            icon="calendar",
            layout=widgets.Layout(width="300px"),
        )
        seasons_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        valid_time_message = widgets.HTML(
            value="<i>For which seasons do you actually have reference data?</i>"
        )
        valid_time_info = self._info_callout(
            "Every training sample has a 'valid time' attribute indicating the date when the observed crop or land cover class was actually present.<br><br>"
            "Here, you can visualize the valid time distribution of your data by clicking the 'Show valid time distribution' button.<br><br>"
            "This is important to consider when selecting your season of interest: <br>"
            "it does not make too much sense to train a classifier for a season in which you have barely any valid reference data to work with!"
        )
        valid_time_button = widgets.Button(
            description="Show valid time distribution",
            button_style="primary",
            icon="bar-chart",
            layout=widgets.Layout(width="260px"),
        )
        valid_time_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        season_picking_message = widgets.HTML(
            value="<i>Based on previous insights, define your growing season (max 12 months!). </i>"
        )
        season_picking_info = self._info_callout(
            "Now use the controls below to pin down the exact <b>growing season window and year</b> you plan to target (maximum 12 consecutive months).<br><br>"
            "    1. Pick the target year from the dropdown (this centers the slider around that year).<br>"
            "    2. Drag the slider handles to the desired start/end months.<br>"
            "    3. The summary automatically reports both the growing-season window and the derived full-year processing period (ending on your selected end month).<br><br>"
            "After picking a season, provide a short name for your season (e.g. 'ShortRains') in the text input below.<br>"
            "No spaces or special characters allowed, only letters and numbers.<br>"
            "This name will be used in the next step to refer to your season and to name your trained model, so choose wisely ;).<br><br>"
        )
        season_slider_output = widgets.Output(
            layout=widgets.Layout(width="100%", overflow="auto")
        )
        season_slider_obj = None
        with season_slider_output:
            season_slider_obj = season_slider()

        season_id_input = widgets.Text(
            value="",
            description="Season ID:",
            placeholder="e.g., ShortRains",
            layout=widgets.Layout(width="60%"),
            tooltip="Provide a short name for your season. No spaces or special characters allowed, only letters and numbers.",
        )

        align_message = widgets.HTML(
            value="<i>Drops irrelevant samples and aligns your data to the selected season.</i>"
        )
        align_info = self._info_callout(
            "Once you are satisfied with your season selection, click the 'Align extractions to season' button to let the app automatically drop all samples that do not match your selected season window.<br><br>"
        )
        align_button = widgets.Button(
            description="Align data to season",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="240px"),
        )
        align_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="100px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab3_widgets = {
            "status_message": status_message,
            "seasons_button": seasons_button,
            "seasons_output": seasons_output,
            "valid_time_button": valid_time_button,
            "valid_time_output": valid_time_output,
            "season_slider": season_slider_obj,
            "season_slider_output": season_slider_output,
            "season_id_input": season_id_input,
            "align_button": align_button,
            "align_output": align_output,
        }

        seasons_button.on_click(self._on_tab3_retrieve_seasons)
        valid_time_button.on_click(self._on_tab3_show_valid_time)
        align_button.on_click(self._on_tab3_align_season)

        return widgets.VBox(
            [
                header,
                status_message,
                season_generic_info,
                widgets.HTML("<h3>1) Retrieve WorldCereal seasons</h3>"),
                worldcereal_season_message,
                worldcereal_season_info,
                widgets.HBox([seasons_button]),
                seasons_output,
                widgets.HTML(
                    "<h3>2) Explore distribution of valid time in your dataset</h3>"
                ),
                valid_time_message,
                valid_time_info,
                widgets.HBox([valid_time_button]),
                valid_time_output,
                widgets.HTML("<h3>3) Pick your season of interest</h3>"),
                season_picking_message,
                season_picking_info,
                season_slider_output,
                widgets.HBox([season_id_input]),
                widgets.HTML("<h3>4) Run seasonal alignment</h3>"),
                align_message,
                align_info,
                widgets.HBox([align_button]),
                align_output,
                self._build_tab_navigation(),
            ]
        )

    def _set_tab2_extractions_df(self, df: pd.DataFrame) -> None:
        """Persist updated extractions data for Tab 2."""
        self.tab2_df = df

    def _init_tab2_extractions_copy(self) -> None:
        """Seed Tab 2 from the saved Tab 1 output only."""
        if self.tab1_df is None or not self.tab1_saved:
            return
        if self.tab2_df is None:
            self.tab2_df = self.tab1_df.copy()

    def _update_tab2_summary(self) -> None:
        """Update the Tab 2 summary widget with extraction statistics."""
        status_widget = self.tab2_widgets.get("status_message")
        df = self._get_tab2_extractions_df()
        if status_widget is None:
            return
        summary = self._format_extractions_summary(df)
        if summary is None:
            status_widget.value = "<i>No extractions loaded yet.</i>"
            return

        status_widget.value = (
            "<i>In this step you can inspect the data and have the possibility to drop entire datasets based on your visual inspection.</i><br><br>"
            "<b>Summary of available data:</b><br>"
            f"{summary}"
        )

    def _format_extractions_summary(self, df: Optional[pd.DataFrame]) -> Optional[str]:
        """Return a formatted summary for an extractions dataframe."""
        if df is None or df.empty:
            return None

        sample_count = (
            df["sample_id"].nunique() if "sample_id" in df.columns else len(df)
        )
        dataset_count = df["ref_id"].nunique() if "ref_id" in df.columns else 0

        summary = (
            f"<b>Total samples:</b> {sample_count:,} &nbsp;|&nbsp; "
            f"<b>Datasets:</b> {dataset_count:,}"
        )
        if "ewoc_code" in df.columns:
            class_count = df["ewoc_code"].nunique()
            summary += f" &nbsp;|&nbsp; <b>Unique classes:</b> {class_count:,}"

        if "downstream_class" in df.columns:
            training_class_count = df["downstream_class"].nunique()
            summary += f" &nbsp;|&nbsp; <b>Unique training classes:</b> {training_class_count:,}"

        return summary

    def _update_tab2_crop_type_options(
        self, df: pd.DataFrame, dataset: Optional[str] = None
    ) -> None:
        """Populate crop type filter options based on the data."""
        crop_select = self.tab2_widgets.get("crop_type_select")
        if crop_select is None:
            return
        if df is None or df.empty or "ewoc_code" not in df.columns:
            crop_select.options = []
            return

        if dataset is not None and "ref_id" in df.columns:
            df = df.loc[df["ref_id"] == dataset]
            if df.empty:
                crop_select.options = []
                return

        labels = ewoc_code_to_label(df["ewoc_code"].unique().tolist())
        crop_select.options = sorted(set(labels))

    def _on_tab2_refresh_datasets(self, _=None):
        """Refresh dataset lists from the current extractions."""
        df = self._get_tab2_extractions_df()
        dataset_dropdown = self.tab2_widgets["dataset_dropdown"]
        drop_select = self.tab2_widgets["drop_select"]

        if df is None or df.empty or "ref_id" not in df.columns:
            dataset_dropdown.options = []
            drop_select.options = []
            self._update_tab2_summary()
            self._update_tab2_crop_type_options(df)
            return

        ref_ids = sorted(df["ref_id"].dropna().unique().tolist())
        dataset_dropdown.options = ref_ids
        if dataset_dropdown.value not in ref_ids:
            dataset_dropdown.value = ref_ids[0] if ref_ids else None
        drop_select.options = ref_ids
        self._update_tab2_summary()
        self._update_tab2_crop_type_options(df, dataset_dropdown.value)

    def _on_tab2_dataset_change(self, change):
        """Update crop type filter options when dataset changes."""
        df = self._get_tab2_extractions_df()
        self._update_tab2_crop_type_options(df, change["new"])

    def _on_tab2_visualize_dataset(self, _=None):
        """Visualize extractions for a selected dataset."""
        df = self._get_tab2_extractions_df()
        output = self.tab2_widgets["dataset_output"]
        dataset = self.tab2_widgets["dataset_dropdown"].value
        band = self.tab2_widgets["band_dropdown"].value
        n_samples = self.tab2_widgets["n_samples_text"].value
        crop_filter = list(self.tab2_widgets["crop_type_select"].value)

        with output:
            output.clear_output()
            if df is None or df.empty or dataset is None:
                print("No dataset available to visualize.")
                return
            subset = df.loc[df["ref_id"] == dataset]
            if subset.empty:
                print("No samples found for the selected dataset.")
                return
            if "label_full" not in subset.columns and "ewoc_code" in subset.columns:
                subset = subset.copy()
                subset["label_full"] = ewoc_code_to_label(subset["ewoc_code"].tolist())
            if crop_filter and "label_full" in subset.columns:
                subset = subset[subset["label_full"].isin(crop_filter)]
                if subset.empty:
                    print("No samples found after crop type filtering.")
                    return
            available_samples = (
                subset["sample_id"].nunique()
                if "sample_id" in subset.columns
                else len(subset)
            )
            if available_samples < n_samples:
                print(
                    f"Only {available_samples} sample(s) available with current filters."
                )
                n_samples = available_samples
            print(f"Dataset: {dataset}")
            try:
                get_band_statistics(subset)
            except Exception as exc:
                print(f"Failed to compute band statistics: {exc}")
            try:
                visualize_timeseries(
                    subset,
                    nsamples=n_samples,
                    band=band,
                    crop_label_attr="label_full",
                )
            except Exception as exc:
                print(f"Failed to visualize timeseries: {exc}")

    def _on_tab2_drop_datasets(self, _=None):
        """Drop selected datasets from the extractions."""
        df = self._get_tab2_extractions_df()
        output = self.tab2_widgets["drop_output"]
        to_drop = list(self.tab2_widgets["drop_select"].value)

        with output:
            output.clear_output()
            if df is None or df.empty:
                print("No extractions available to drop datasets from.")
                return
            if not to_drop:
                print("No datasets selected for removal.")
                return
            remaining = df.loc[~df["ref_id"].isin(to_drop)]
            self._set_tab2_extractions_df(remaining)
            nsamples = remaining["sample_id"].nunique()
            print(f"Dropped {len(to_drop)} dataset(s). Remaining samples: {nsamples}")
        if to_drop:
            self._update_tab2_summary()
        self._on_tab2_refresh_datasets()

    def _on_tab3_retrieve_seasons(self, _=None):
        """Retrieve and display WorldCereal seasons for the AOI."""
        output = self.tab3_widgets["seasons_output"]
        aoi_map = self.tab1_widgets.get("aoi_map")

        with output:
            output.clear_output()
            try:
                if aoi_map is None:
                    print("No region of interest specified in Tab 1, cannot continue.")
                    return
                spatial_extent = aoi_map.get_extent()
                retrieve_worldcereal_seasons(spatial_extent)
            except Exception as exc:
                print(f"Failed to retrieve seasons: {exc}")

    def _on_tab3_show_valid_time(self, _=None):
        """Show valid time distribution for current extractions."""
        df = self._get_tab2_extractions_df()
        output = self.tab3_widgets["valid_time_output"]

        with output:
            output.clear_output()
            if df is None or df.empty:
                print("No extractions available to visualize valid time distribution.")
                return
            try:
                valid_time_distribution(df)
            except Exception as exc:
                print(f"Failed to show valid time distribution: {exc}")

    def _on_tab3_align_season(self, _=None):
        """Align extractions to the selected season."""
        df = self._get_tab2_extractions_df()
        output = self.tab3_widgets["align_output"]
        slider = self.tab3_widgets["season_slider"]
        season_id = self.tab3_widgets["season_id_input"].value.strip() or None

        with output:
            output.clear_output()
            if df is None:
                print("Tab 2 output not available. Complete or skip Tab 2 first.")
                return
            if df.empty:
                print("No extractions available to align.")
                return
            if season_id is None:
                print("No season ID provided, cannot continue!")
                return
            if season_id is not None and not season_id.isalnum():
                print(
                    "Invalid season ID. Please use only letters and numbers, no spaces or special characters."
                )
                return
            self.season_id = season_id
            try:
                if slider is None:
                    print("Season slider not initialized.")
                    return
                selection = slider.get_selection()
                self.season_window = selection.season_window
                self.processing_period = selection.processing_period
                self.tab3_df = align_extractions_to_season(
                    df, self.processing_period, season_window=self.season_window
                )
                print("Seasonal alignment completed")
                if self.tab3_df is None or self.tab3_df.empty:
                    self.tab3_df = None
                    print(
                        "No samples remain after alignment, cannot continue!. "
                        "Choose a different season or consider adding more training data by revisiting previous steps."
                    )
                    return
                unique_classes = self.tab3_df["ewoc_code"].nunique()
                if unique_classes <= 1:
                    self.tab3_df = None
                    print(
                        "Only one unique crop type remains after alignment, cannot continue! "
                        "Choose a different season or consider adding more training data by revisiting previous steps."
                    )
                    return
                nsamples = self.tab3_df["sample_id"].nunique()
                print(f"Remaining samples after alignment: {nsamples}")

            except Exception as exc:
                print(f"Failed to align extractions: {exc}")

    def _on_tab4_select_crops(self, _=None):
        """Initialize and display crop type picker for filtered samples."""
        df = self.tab3_df
        output = self.tab4_widgets["croptype_output"]
        picker_container = self.tab4_widgets["croptype_picker_container"]

        with output:
            output.clear_output()
            if df is None:
                print("Tab 3 output not available. Complete Tab 3 first.")
                return
            if df.empty:
                print("No samples available for crop selection.")
                return
        crop_picker = CropTypePicker(
            sample_df=df,
            expand=False,
            display_ui=False,
            selection_modes=["Include", "Drop"],
        )
        self.tab4_widgets["croptype_picker"] = crop_picker
        picker_container.children = [crop_picker.widget]

    def _on_tab4_apply_crops(self, _=None):
        """Apply crop selection to the aligned samples."""
        picker = self.tab4_widgets.get("croptype_picker")
        output = self.tab4_widgets["croptype_output"]
        df = self.tab3_df

        with output:
            output.clear_output()
            if df is None:
                print("Tab 3 output not available. Complete Tab 3 first.")
                return
            if picker is None or df.empty:
                print("No crop selection to apply.")
                return
            try:
                picker.apply_selection()
                filtered = apply_croptypepicker_to_df(df, picker)
                self.tab4_df = filtered
                self.tab4_confirmed = False
                print(f"Crop selection applied. Remaining samples: {len(filtered)}.")
            except Exception as exc:
                print(f"Failed to apply crop selection: {exc}")
                return
        self._render_tab4_summary(output=output, clear_output=False)
        self._render_tab4_other_summary()
        self._update_tab5_state()

    # =========================================================================
    # Tab 4: Prepare Training Data
    # =========================================================================

    def _build_tab4_crop_selection(self) -> widgets.VBox:
        """Build Tab 4: Land cover/crop type selection and training class composition."""
        header = widgets.HTML(
            value="<h2>Land Cover/Crop Type Selection & Training Class Composition</h2>"
        )

        status_message = widgets.HTML(
            value="<i>Please complete the steps in Tab 3 first.</i>"
        )

        croptype_message = widgets.HTML(
            value="<i>Select the land cover or crop types to include or exclude from the training data.</i>"
        )
        croptype_info = self._info_callout(
            "Use the crop type picker to define which classes participate in training.<br><br>"
            "<b>Include vs Drop modes</b><br>"
            "   - <b>Include</b>: selects classes to keep for training. You must select at least one class in this mode.<br>"
            "   - <b>Drop</b>: removes classes from training. Dropped classes are excluded unless they are already covered by an included higher-level group (included parents take precedence and disable their descendants).<br><br>"
            "<b>Hierarchy selection</b><br>"
            'Selecting a higher-level category automatically groups all its descendant classes under one training label. This is a quick way to create broader classes (e.g., selecting "cereals" groups all cereal subclasses).<br><br>'
            "<b>Apply is required</b><br>"
            "After making your selections, click <b>Apply</b> in the picker to lock them in.<br><br>"
            "<b>Apply crop selection</b><br>"
            "Once you are satisfied with the selected and excluded classes, apply your selection to the dataset using the <b>Apply crop selection</b> button.<br><br>"
            "Samples that are <i>not included</i> and also <i>not explicitly dropped</i> are assigned to the <b>other</b> class during training."
        )
        select_crops_button = widgets.Button(
            button_style="primary",
            icon="list",
            description="Open crop type picker",
            layout=widgets.Layout(width="240px"),
        )
        apply_crops_button = widgets.Button(
            description="Apply crop selection",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="240px", height="48px"),
        )
        croptype_picker_container = widgets.VBox()
        croptype_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="100px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        other_class_message = widgets.HTML(
            value="<i>Inspect the composition of the 'other' class and optionally remove specific crop types from it.</i>"
        )
        other_class_info = self._info_callout(
            "Samples that do not belong to any of the selected training classes in the previous step have been assigned to the 'other' class.<br><br>"
            "However, having a very heterogeneous 'other' class can negatively impact classifier performance.<br>"
            "Particularly when some classes in 'other' are very similar to your selected training classes, it can be beneficial to remove them from the training data entirely.<br><br>"
        )
        other_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="100px",
                border="1px solid #ccc",
                padding="10px",
                overflow_x="auto",
            )
        )
        drop_classes_select = widgets.SelectMultiple(
            options=[],
            description="Drop classes:",
            layout=widgets.Layout(width="70%", height="140px"),
        )
        drop_classes_button = widgets.Button(
            description="Drop selected classes",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="220px"),
        )
        drop_classes_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        combine_class_message = widgets.HTML(
            value="<i>Optional possibility to group classes into a new class.</i>"
        )
        combine_class_info = self._info_callout(
            "Having too many and highly similar training classes can complicate the learning task for your classifier.<br><br>"
            "In this step you can combine multiple classes into a new class to simplify your training labels.<br><br>"
            "To do so, provide a name for the new class and list the classes you want to combine (comma-separated).<br><br>"
        )
        combine_label_input = widgets.Text(
            value="",
            description="New class:",
            placeholder="e.g., cereals",
            layout=widgets.Layout(width="60%"),
        )
        combine_classes_input = widgets.Text(
            value="",
            description="Classes to combine:",
            placeholder="comma-separated class labels",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="80%"),
        )
        combine_button = widgets.Button(
            description="Combine classes",
            button_style="primary",
            icon="compress",
            layout=widgets.Layout(width="180px"),
        )
        combine_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="100px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        subsample_message = widgets.HTML(
            value="<i>Optionally subsample one or multiple classes to a maximum number of samples.</i>"
        )
        subsample_info = self._info_callout(
            "If some classes have a very large number of samples compared to others, it can be beneficial to subsample them down to a maximum number of samples.<br><br>"
            "This helps to create a more balanced training dataset and can reduce training time.<br><br>"
            "<b>Note</b>:<br>"
            "During model training, by default we apply class balancing via sample weighting.<br>"
            "Therefore, subsampling is not strictly necessary but can still be useful in some scenarios.<br><br>"
            "If you want to downsample a specific class, simply provide the maximum number of samples and the name of the class you want to target.<br>"
            "Clicking the 'Subsample class' button will generate a stratified random sample, accounting for dataset ID.<br><br>"
        )
        subsample_text = widgets.IntText(
            value=0,
            description="Max samples per class:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="260px"),
        )
        subsample_class_input = widgets.Text(
            value="",
            description="Class to subsample:",
            placeholder="e.g., maize",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="50%"),
        )
        subsample_button = widgets.Button(
            description="Subsample class",
            button_style="warning",
            icon="cut",
            layout=widgets.Layout(width="180px"),
        )
        subsample_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="100px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        confirm_message = widgets.HTML(
            value="<i>Click green to confirm or red to restart training class preparation.</i>"
        )
        confirm_classes_button = widgets.Button(
            description="Confirm training classes",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="240px", height="60px"),
        )
        reset_classes_button = widgets.Button(
            description="Reset training classes",
            button_style="danger",
            icon="undo",
            layout=widgets.Layout(width="220px", height="60px"),
        )
        final_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab4_widgets = {
            "status_message": status_message,
            "select_crops_button": select_crops_button,
            "apply_crops_button": apply_crops_button,
            "croptype_picker_container": croptype_picker_container,
            "croptype_output": croptype_output,
            "croptype_picker": None,
            "other_output": other_output,
            "drop_classes_select": drop_classes_select,
            "drop_classes_button": drop_classes_button,
            "drop_classes_output": drop_classes_output,
            "combine_label_input": combine_label_input,
            "combine_classes_input": combine_classes_input,
            "combine_button": combine_button,
            "combine_output": combine_output,
            "subsample_text": subsample_text,
            "subsample_class_input": subsample_class_input,
            "subsample_button": subsample_button,
            "subsample_output": subsample_output,
            "confirm_classes_button": confirm_classes_button,
            "reset_classes_button": reset_classes_button,
            "final_output": final_output,
        }

        select_crops_button.on_click(self._on_tab4_select_crops)
        apply_crops_button.on_click(self._on_tab4_apply_crops)
        drop_classes_button.on_click(self._on_tab4_drop_classes)
        combine_button.on_click(self._on_tab4_combine_classes)
        subsample_button.on_click(self._on_tab4_subsample_classes)
        confirm_classes_button.on_click(self._on_tab4_confirm_classes)
        reset_classes_button.on_click(self._on_tab4_reset_classes)

        return widgets.VBox(
            [
                header,
                status_message,
                widgets.HTML("<h3>1) Land cover/Crop type selection</h3>"),
                croptype_message,
                croptype_info,
                widgets.HBox(
                    [select_crops_button],
                    layout=widgets.Layout(justify_content="flex-start"),
                ),
                croptype_picker_container,
                widgets.HBox(
                    [apply_crops_button],
                    layout=widgets.Layout(justify_content="flex-start"),
                ),
                croptype_output,
                widgets.HTML("<h3>2) Inspection of the 'other' class</h3>"),
                other_class_message,
                other_class_info,
                other_output,
                drop_classes_select,
                widgets.HBox([drop_classes_button]),
                drop_classes_output,
                widgets.HTML("<h3>3) Combine classes</h3>"),
                combine_class_message,
                combine_class_info,
                combine_label_input,
                combine_classes_input,
                widgets.HBox([combine_button]),
                combine_output,
                widgets.HTML("<h3>4) Subsample classes</h3>"),
                subsample_message,
                subsample_info,
                widgets.HBox([subsample_class_input, subsample_text]),
                widgets.HBox([subsample_button]),
                subsample_output,
                widgets.HTML("<h3>5) Confirm training classes</h3>"),
                confirm_message,
                widgets.HBox([confirm_classes_button, reset_classes_button]),
                final_output,
                self._build_tab_navigation(),
            ]
        )

    def _build_tab5_compute_embeddings(self) -> widgets.VBox:
        """Build Tab 5: Compute Presto embeddings."""
        header = widgets.HTML(value="<h2>Compute Presto Embeddings</h2>")

        status_message = widgets.HTML(
            value="<i>Please prepare data in Tab 4 first or load a prepared training dataframe using the button below to skip steps 1-4.</i>"
        )

        load_title = widgets.HTML(
            value="<h3>Option to load a prepared training dataframe</h3>"
        )
        load_input = widgets.Text(
            value="",
            description="Full path to your training dataframe:",
            placeholder="/path/to/training_df_season-YYYYMMDD-YYYYMMDD_cl-x_YYYYMMDD-HHMMSS.csv",
            layout=widgets.Layout(width="100%"),
            description_width="250px",
            tooltip="Provide the full path to a previously saved training dataframe resulting from steps 1-4.",
        )
        load_button = widgets.Button(
            description="Load training dataframe",
            button_style="info",
            icon="folder-open",
            layout=widgets.Layout(width="260px", height="40px"),
        )
        load_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        embeddings_message = widgets.HTML(
            value="<i>Tune parameters and click the button below to compress the EO time series of your samples into comprehensive geospatial embeddings (your training features).</i>"
        )
        embeddings_info = self._info_callout(
            "Using a geospatial foundation model (Presto), we derive training features for each of your training samples.<br>"
            "Presto was pre-trained on millions of unlabeled samples around the world and finetuned on a global dataset of labelled land cover and crop type observations from the WorldCereal reference database.<br>"
            "The resulting <b>128 embeddings</b> (`presto_ft_0` -> `presto_ft_127`) nicely condense the Sentinel-1, Sentinel-2, meteo timeseries and ancillary data for your season of interest into a limited number of meaningful features which we will use for downstream model training.<br><br>"
            "We provide some options aimed at increasing temporal robustness of your final crop model.<br>"
            "This is controlled by the following arguments:<br>"
            "    - <b>augment</b> parameter: when set to `True`, it introduces slight temporal jittering of the processing window, making the model more robust to slight variations in seasonality across different years. <br>"
            "           By default, this option is set to `True`, but especially when training a model for a specific region and year with good local data, disabling this option could be considered.<br>"
            "    - <b>repeats</b> parameter: number of times each training sample is (re)drawn with its augmentations. Higher values (>1) create more variants (with jitter/masking) and enlarge the effective training set, potentially improving generalization at the cost of longer embedding inference time.<br>"
            "    - <b>mask_on_training</b> parameter: when `True`, applies sensor masking augmentations (e.g. simulating S1/S2 dropouts, additional clouds, ancillary feature removals) only to the training split to improve robustness to real-world data gaps.<br>"
            "           The validation/test split is kept untouched for fair evaluation.<br>"
        )
        augment_checkbox = widgets.Checkbox(
            value=True,
            description="Augment",
        )
        mask_on_training_checkbox = widgets.Checkbox(
            value=True,
            description="Mask on training",
        )
        repeats_input = widgets.IntText(
            value=3,
            description="Repeats:",
            layout=widgets.Layout(width="200px"),
        )
        embeddings_button = widgets.Button(
            description="Compute Presto embeddings",
            button_style="success",
            icon="cogs",
            layout=widgets.Layout(width="260px", height="48px"),
        )
        embeddings_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab5_widgets = {
            "status_message": status_message,
            "load_title": load_title,
            "load_input": load_input,
            "load_button": load_button,
            "load_output": load_output,
            "augment_checkbox": augment_checkbox,
            "mask_on_training_checkbox": mask_on_training_checkbox,
            "repeats_input": repeats_input,
            "embeddings_button": embeddings_button,
            "embeddings_output": embeddings_output,
        }

        def _on_augment_toggle(change):
            if repeats_input is not None:
                repeats_input.disabled = not change["new"]

        augment_checkbox.observe(_on_augment_toggle, names="value")
        _on_augment_toggle({"new": augment_checkbox.value})

        load_button.on_click(self._on_tab5_load_training_df)
        embeddings_button.on_click(self._on_tab5_compute_embeddings)

        return widgets.VBox(
            [
                header,
                status_message,
                load_title,
                load_input,
                widgets.HBox([load_button]),
                load_output,
                widgets.HTML("<h3>Set embedding parameters</h3>"),
                embeddings_message,
                embeddings_info,
                widgets.HBox([augment_checkbox, repeats_input]),
                widgets.HBox([mask_on_training_checkbox]),
                widgets.HBox([embeddings_button]),
                embeddings_output,
                self._build_tab_navigation(),
            ]
        )

    def _on_tab5_load_training_df(self, _=None) -> None:
        """Load a prepared training dataframe from disk with basic checks."""
        load_input = self.tab5_widgets.get("load_input")
        output = self.tab5_widgets.get("load_output")
        status_message = self.tab5_widgets.get("status_message")
        if load_input is None or output is None:
            return

        path_value = load_input.value.strip()
        with output:
            output.clear_output()
            if not path_value:
                print("Provide a training dataframe path to load.")
                return
            path = Path(path_value)
            if not path.exists():
                print(f"File not found: {path}")
                return
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"Failed to read training dataframe: {exc}")
                return
            required_cols = {"downstream_class", "sample_id", "ref_id", "ewoc_code"}
            missing = sorted(required_cols - set(df.columns))
            if missing:
                print(
                    "Training dataframe is missing required columns: "
                    + ", ".join(missing)
                )
                return
            filename = path.stem
            self.season_id, self.season_window = self._parse_season_info_from_filename(
                filename
            )
            if self.season_id is None or self.season_window is None:
                print(
                    "Failed to parse season information from filename. Make sure it follows the format: training_df_season-YYYYMMDD-YYYYMMDD_cl-x_YYYYMMDD-HHMMSS.csv"
                )
                return
            print(
                f"Parsed season ID: {self.season_id}, season window: {self.season_window.start_date} to {self.season_window.end_date}"
            )
            self.tab4_df = df
            self.tab4_confirmed = True
            self.training_df_path = path
            print(f"Training dataframe loaded: {path}")

        if status_message is not None:
            status_message.value = "<i>Training dataframe loaded from file.</i>"
        self._update_tab5_state()

    def _parse_season_info_from_filename(
        self,
        filename: str,
    ) -> Tuple[Optional[str], Optional[TemporalContext]]:
        """Parse season ID and window from a training dataframe filename."""
        try:
            season_info = filename.split("_")[1]
            season_id = season_info.split("-")[0]
            season_start = pd.to_datetime(
                season_info.split("-")[1], format="%Y%m%d"
            ).strftime("%Y-%m-%d")
            season_end = pd.to_datetime(
                season_info.split("-")[2], format="%Y%m%d"
            ).strftime("%Y-%m-%d")
            return season_id, TemporalContext(season_start, season_end)
        except (IndexError, ValueError):
            return None, None

    def _get_tab4_working_df(self) -> Optional[pd.DataFrame]:
        return self.tab4_df

    def _set_tab4_working_df(self, df: pd.DataFrame) -> None:
        self.tab4_df = df

    def _render_tab4_summary(
        self,
        output: Optional[widgets.Output] = None,
        clear_output: bool = True,
    ) -> None:
        """Render the downstream class summary into the Tab 4 output."""
        df = self._get_tab4_working_df()
        drop_select = self.tab4_widgets.get("drop_classes_select")
        output = output or self.tab4_widgets.get("croptype_output")
        if output is None:
            return

        with output:
            if clear_output:
                output.clear_output()
            if df is None or df.empty:
                print("No samples available. Apply crop selection first.")
                return
            if "downstream_class" not in df.columns:
                print("No downstream_class column found. Apply crop selection first.")
                return
            print("Samples per downstream class:")
            crop_counts = df["downstream_class"].value_counts()
            crop_table = crop_counts.reset_index()
            crop_table.columns = ["Downstream Class", "Count"]
            print(
                tabulate(
                    crop_table.values,
                    headers=crop_table.columns,
                    tablefmt="grid",
                    maxcolwidths=[None, 40],
                )
            )

        if drop_select is not None:
            other_class = self._get_other_class_composition()
            options = []
            if other_class is not None and not other_class.empty:
                options = [
                    f"{row['label_full']} ({row['ewoc_code']})"
                    for _, row in other_class.iterrows()
                ]
            drop_select.options = options

    def _get_other_class_composition(self) -> Optional[pd.DataFrame]:
        """Get the composition of the 'other' class if available."""
        df = self._get_tab4_working_df()
        if df is None or df.empty:
            return None
        if "downstream_class" not in df.columns or "ewoc_code" not in df.columns:
            return None
        other_df = df.loc[df["downstream_class"] == "other"]
        if other_df.empty:
            return None
        other_count = other_df["ewoc_code"].value_counts()
        count_name = other_count.name or "count"
        other_count = other_count.rename(count_name)
        other_labels = translate_ewoc_codes(other_count.index.tolist())
        display_df = other_count.to_frame().merge(
            other_labels, left_index=True, right_index=True
        )
        display_df["ewoc_code"] = display_df.index
        return display_df

    def _render_tab4_other_summary(self) -> None:
        """Render detailed composition of the 'other' class."""
        output = self.tab4_widgets.get("other_output")
        if output is None:
            return

        with output:
            output.clear_output()
            other_class = self._get_other_class_composition()
            if other_class is None or other_class.empty:
                print("No samples in 'other' class found.")
                return
            display(other_class)

    def _on_tab4_confirm_classes(self, _=None) -> None:
        """Confirm training classes before moving to Tab 5."""
        df = self._get_tab4_working_df()
        output = self.tab4_widgets.get("final_output")

        if output is not None:
            with output:
                output.clear_output()
                if df is None or df.empty:
                    print(
                        "No training classes selected. At least complete step 1 above."
                    )
                    return
                if "downstream_class" not in df.columns:
                    print(
                        "No downstream_class column found. Something went wrong in step 1 above."
                    )
                    return
                if df["downstream_class"].nunique() <= 1:
                    print(
                        "Only one training class found in your dataset. We cannot continue! Press the reset button below to start over and make sure to include at least two classes in your crop type selection in step 1 above."
                    )
                    return
                training_dir = Path("./training_data")
                training_dir.mkdir(exist_ok=True)
                timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
                season = self.season_window
                season_start = season.start_date.replace("-", "")
                season_end = season.end_date.replace("-", "")
                season_str = f"{season_start}-{season_end}"
                nclasses = df["downstream_class"].nunique()
                training_df_path = (
                    training_dir
                    / f"trainingdf_{self.season_id}-{season_str}_cl-{nclasses}.csv"
                )
                if training_df_path.exists():
                    # append timestamp to avoid overwriting
                    training_df_path = (
                        training_dir
                        / f"trainingdf_{self.season_id}-{season_str}_cl-{nclasses}_{timestamp}.csv"
                    )
                df.to_csv(training_df_path, index=False)
                print(
                    f"{nclasses} training classes confirmed and saved to {training_df_path}.\n"
                )
                self.training_df_path = training_df_path
                print("You can proceed to Tab 5.")

        self.tab4_confirmed = True
        self._update_tab5_state()

    def _on_tab4_reset_classes(self, _=None) -> None:
        """Reset Tab 4 selections and outputs."""
        self.tab4_df = None
        self.tab4_confirmed = False

        self.tab4_widgets["croptype_picker"] = None
        picker_container = self.tab4_widgets.get("croptype_picker_container")
        if picker_container is not None:
            picker_container.children = []

        drop_select = self.tab4_widgets.get("drop_classes_select")
        if drop_select is not None:
            drop_select.options = []

        output = self.tab4_widgets.get("croptype_output")
        if output is not None:
            output.clear_output()

        other_output = self.tab4_widgets.get("other_output")
        if other_output is not None:
            other_output.clear_output()

        drop_classes_output = self.tab4_widgets.get("drop_classes_output")
        if drop_classes_output is not None:
            drop_classes_output.clear_output()

        combine_output = self.tab4_widgets.get("combine_output")
        if combine_output is not None:
            combine_output.clear_output()

        subsample_output = self.tab4_widgets.get("subsample_output")
        if subsample_output is not None:
            subsample_output.clear_output()

        output = self.tab4_widgets.get("final_output")
        if output is not None:
            output.clear_output()
            print("Training classes reset. Start again by applying crop selection.")

        self._update_tab5_state()

    def _on_tab4_drop_classes(self, _=None):
        df = self._get_tab4_working_df()
        output = self.tab4_widgets.get("drop_classes_output")
        to_drop = list(self.tab4_widgets["drop_classes_select"].value)

        with output:
            output.clear_output()
            if df is None or df.empty:
                print("No training samples available. Apply crop selection first.")
                return
            if "downstream_class" not in df.columns:
                print("No downstream_class column found. Apply crop selection first.")
                return
            if not to_drop:
                print("No classes selected for removal.")
                return
            # Retrieve ewoc_codes to be dropped
            codes_to_drop = []
            for value in to_drop:
                code_str = value.split("(")[-1].strip(")")
                try:
                    codes_to_drop.append(int(code_str))
                except ValueError:
                    continue
            if not codes_to_drop:
                print("No valid EWOC codes parsed from selection.")
                return
            df = df.loc[~df["ewoc_code"].isin(codes_to_drop)]
            self._set_tab4_working_df(df)
            self.tab4_confirmed = False
            print(f"Dropped {len(to_drop)} class(es). Remaining samples: {len(df)}")
        self._render_tab4_summary(output=output, clear_output=False)
        self._render_tab4_other_summary()
        self._update_tab5_state()

    def _on_tab4_combine_classes(self, _=None):
        df = self._get_tab4_working_df()
        output = self.tab4_widgets["combine_output"]
        new_label = self.tab4_widgets["combine_label_input"].value.strip()
        classes_str = self.tab4_widgets["combine_classes_input"].value

        with output:
            output.clear_output()
            if df is None or df.empty:
                print("No samples available. Apply crop selection first.")
                return
            if "downstream_class" not in df.columns:
                print("No downstream_class column found. Apply crop selection first.")
                return
            if not new_label:
                print("Provide a target class name.")
                return
            classes = [c.strip() for c in classes_str.split(",") if c.strip()]
            if not classes:
                print("Provide classes to combine.")
                return
            df = df.copy()
            df.loc[df["downstream_class"].isin(classes), "downstream_class"] = new_label
            self._set_tab4_working_df(df)
            self.tab4_confirmed = False
            print(f"Combined {len(classes)} class(es) into '{new_label}'.")
        self._render_tab4_summary(output=output, clear_output=False)
        self._update_tab5_state()

    def _on_tab4_subsample_classes(self, _=None):
        df = self._get_tab4_working_df()
        output = self.tab4_widgets["subsample_output"]
        max_samples = self.tab4_widgets["subsample_text"].value
        target_class = self.tab4_widgets["subsample_class_input"].value.strip()

        with output:
            output.clear_output()
            if df is None or df.empty:
                print("No samples available. Apply crop selection first.")
                return
            if "downstream_class" not in df.columns:
                print("No downstream_class column found. Apply crop selection first.")
                return
            if "ref_id" not in df.columns:
                print("No ref_id column found. Cannot stratify by dataset ID.")
                return
            if max_samples <= 0:
                print("Set a max samples per class > 0.")
                return
            if not target_class:
                print("Provide the class name to subsample.")
                return
            target_df = df.loc[df["downstream_class"] == target_class]
            if target_df.empty:
                print(f"No samples found for class '{target_class}'.")
                return
            if len(target_df) <= max_samples:
                print(
                    f"Class '{target_class}' already has {len(target_df)} samples (<= {max_samples})."
                )
            else:
                sizes = target_df.groupby("ref_id").size()
                total = sizes.sum()
                base = (sizes * max_samples) // total
                remainder = max_samples - base.sum()
                if remainder > 0:
                    frac = (sizes * max_samples) % total
                    top_refs = frac.sort_values(ascending=False).index[:remainder]
                    base.loc[top_refs] += 1

                sampled_chunks = []
                for ref_id, group in target_df.groupby("ref_id"):
                    n = int(base.get(ref_id, 0))
                    if n <= 0:
                        continue
                    sampled_chunks.append(group.sample(n=min(n, len(group))))
                sampled_target = (
                    pd.concat(sampled_chunks) if sampled_chunks else target_df.iloc[0:0]
                )
                df = pd.concat(
                    [df.loc[df["downstream_class"] != target_class], sampled_target]
                ).reset_index(drop=True)
                print(
                    f"Subsampled class '{target_class}' to {len(sampled_target)} samples."
                )
            self._set_tab4_working_df(df)
            self.tab4_confirmed = False
        self._render_tab4_summary(output=output, clear_output=False)
        self._update_tab5_state()

    def _on_tab5_compute_embeddings(self, _=None):
        df = self._get_tab4_working_df()
        output = self.tab5_widgets["embeddings_output"]
        augment_checkbox = self.tab5_widgets.get("augment_checkbox")
        mask_on_training_checkbox = self.tab5_widgets.get("mask_on_training_checkbox")
        repeats_input = self.tab5_widgets.get("repeats_input")

        with output:
            output.clear_output()
            if df is None or df.empty:
                print("No training samples available.")
                return
            if self.season_id is None:
                print(
                    "Season ID missing. Complete Tab 3 season alignment or load a valid training dataframe containing a season ID in its name."
                )
                return
            if self.season_window is None:
                print(
                    "Season window missing. Complete Tab 3 season alignment or load a valid training dataframe containing a season window in its name."
                )
                return
            augment = True if augment_checkbox is None else augment_checkbox.value
            mask_on_training = (
                True
                if mask_on_training_checkbox is None
                else mask_on_training_checkbox.value
            )
            repeats = repeats_input.value if repeats_input is not None else 3
            if repeats <= 0:
                print("Repeats must be >= 1.")
                return
            try:
                print("Computing Presto embeddings...")
                self.tab5_df = compute_seasonal_presto_embeddings(
                    df,
                    season_id=self.season_id,
                    augment=augment,
                    mask_on_training=mask_on_training,
                    repeats=repeats,
                    season_window=self.season_window,
                    season_calendar_mode="custom",
                )
                embeddings_dir = Path("./embeddings")
                embeddings_dir.mkdir(exist_ok=True)
                training_df_name = self.training_df_path.stem
                embedding_df_name = training_df_name.replace("trainingdf", "embeddings")
                embeddings_path = embeddings_dir / f"{embedding_df_name}.csv"
                self.tab5_df.to_csv(embeddings_path, index=False)
                print(f"Embeddings computed: {len(self.tab5_df)} rows.")
                print(f"Embeddings saved to: {embeddings_path}")
                self.embeddings_df_path = embeddings_path
                self._update_tab6_state()
            except Exception as exc:
                print(f"Failed to compute embeddings: {exc}")

    # =========================================================================
    # Tab 6: Train Model
    # =========================================================================

    def _build_tab6_train_model(self) -> widgets.VBox:
        """Build Tab 6: Run model training."""
        header = widgets.HTML(value="<h2>Train Classifier</h2>")

        status_message = widgets.HTML(
            value="<i>Please compute Presto embeddings in Tab 5 first or load embeddings from a file using the button below to skip steps 1-5.</i>"
        )

        load_title = widgets.HTML(
            value="<h3>Option to load a prepared embeddings dataframe</h3>"
        )
        load_input = widgets.Text(
            value="",
            description="Full path to embeddings:",
            placeholder="/path/to/embeddings_YYYYMMDD-HHMMSS.csv",
            layout=widgets.Layout(width="100%", margin="0 0 0 20px"),
        )
        load_button = widgets.Button(
            description="Load embeddings",
            button_style="info",
            icon="folder-open",
            layout=widgets.Layout(width="200px", height="40px"),
        )
        load_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        training_params_message = widgets.HTML(
            value="<i>Configure seasonal torch head training parameters.</i>"
        )
        training_params_info = self._info_callout(
            "The seasonal torch head is a lightweight classifier trained on top of the Presto embeddings.<br><br>"
            "The training process includes a small grid search over learning rates and weight decay values to find the best combination for your dataset.<br><br>"
            "The following parameters need to be set:<br>"
            "<b>Head task</b>: <code>croptype</code> for multi-class crop type prediction or <code>landcover</code> for land-cover training.<br>"
            "<b>Head type</b>: <code>linear</code> uses a single linear layer, <code>mlp</code> adds a small MLP head for extra capacity.<br>"
            "<b>Epochs</b>: number of training epochs for the head.<br>"
            "<b>LR</b>: learning rate (float); only adjust if you know what you're doing.<br>"
            "<b>Weight decay</b>: (float); only adjust if you know what you're doing.<br>"
            "<b>Use class balancing</b>: By default enabled to ensure minority classes are not discarded. However, depending on your training class distribution this may lead to undesired results.<br>"
            "       There is no golden rule here. If your main goal is to make sure the most dominant classes in your training data are very precisely identified in your map, you can opt to NOT apply class balancing.<br><br>"
            "Your ready-to-use classification model, along with cal/val metrics, confusion matrices, and logs will be written to disk.<br>"
            "The resulting artifacts (model weights + config + packaged `.zip`) are stored in the `./downstream_heads/` directory.<br>"
            "Keep this directory around: the zip bundle will be uploaded to CDSE in the next step and the config is reused whenever you redeploy or troubleshoot the head.<br><br>"
            "The name of your model is set automatically based on the season ID and number of classes in your training data.<br>"
            "You have the option to provide a <b>custom suffix</b> to be added to the default model name.<br>"
            "Keep the suffix short and avoid spaces and special characters to ensure compatibility with CDSE upload and future model management."
        )
        head_task_dropdown = widgets.Dropdown(
            options=["croptype", "landcover"],
            value="croptype",
            description="Head task:",
            layout=widgets.Layout(width="240px"),
        )
        head_type_dropdown = widgets.Dropdown(
            options=["linear", "mlp"],
            value="linear",
            description="Head type:",
            layout=widgets.Layout(width="220px"),
        )
        epochs_input = widgets.IntText(
            value=40,
            description="Epochs:",
            layout=widgets.Layout(width="160px"),
        )
        lr_input = widgets.Text(
            value="1e-2",
            description="Learning rate:",
            placeholder="e.g., 1e-2",
            layout=widgets.Layout(width="150px"),
            description_width="150px",
        )
        weight_decay_input = widgets.Text(
            value="0.0",
            description="Weight decay:",
            placeholder="e.g., 0.0",
            layout=widgets.Layout(width="150px"),
            description_width="150px",
        )
        use_balancing_checkbox = widgets.Checkbox(
            value=True,
            description="Use class balancing",
        )
        custom_suffix_input = widgets.Text(
            value="",
            description="Custom suffix:",
            description_width="200px",
            placeholder="optional custom suffix for model name",
            layout=widgets.Layout(width="100%"),
            tooltip="Keep it short and avoid spaces and special characters.",
        )

        train_button = widgets.Button(
            description="Start Training",
            button_style="success",
            icon="play",
            layout=widgets.Layout(width="150px", height="50px"),
            disabled=True,
        )

        progress_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="150px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab6_widgets = {
            "status_message": status_message,
            "load_title": load_title,
            "load_input": load_input,
            "load_button": load_button,
            "load_output": load_output,
            "training_params_message": training_params_message,
            "training_params_info": training_params_info,
            "head_task_dropdown": head_task_dropdown,
            "head_type_dropdown": head_type_dropdown,
            "epochs_input": epochs_input,
            "lr_input": lr_input,
            "weight_decay_input": weight_decay_input,
            "use_balancing_checkbox": use_balancing_checkbox,
            "custom_suffix_input": custom_suffix_input,
            "train_button": train_button,
            "progress_output": progress_output,
        }

        load_button.on_click(self._on_tab6_load_embeddings)
        train_button.on_click(self._on_train_click)

        return widgets.VBox(
            [
                header,
                status_message,
                load_title,
                load_input,
                widgets.HBox([load_button]),
                load_output,
                widgets.HTML("<h3>Set training parameters</h3>"),
                training_params_message,
                training_params_info,
                widgets.HBox([head_task_dropdown, head_type_dropdown]),
                widgets.HBox([epochs_input]),
                widgets.HBox([lr_input, weight_decay_input]),
                widgets.HBox([use_balancing_checkbox]),
                custom_suffix_input,
                widgets.HBox(
                    [train_button],
                    layout=widgets.Layout(justify_content="center", margin="20px 0"),
                ),
                widgets.HTML("<b>Progress:</b>"),
                progress_output,
                self._build_tab_navigation(),
            ]
        )

    def _on_tab6_load_embeddings(self, _=None) -> None:
        """Load embeddings from disk with basic checks."""
        load_input = self.tab6_widgets.get("load_input")
        output = self.tab6_widgets.get("load_output")
        status_message = self.tab6_widgets.get("status_message")
        if load_input is None or output is None:
            return

        path_value = load_input.value.strip()
        with output:
            output.clear_output()
            if not path_value:
                print("Provide an embeddings file path to load.")
                return
            path = Path(path_value)
            if not path.exists():
                print(f"File not found: {path}")
                return
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"Failed to read embeddings file: {exc}")
                return
            required_cols = {"downstream_class", "sample_id", "ref_id"}
            missing = sorted(required_cols - set(df.columns))
            if missing:
                print(
                    "Embeddings file is missing required columns: " + ", ".join(missing)
                )
                return
            if not any(col.startswith("presto_ft_") for col in df.columns):
                print("Embeddings file has no presto_ft_* feature columns.")
                return
            filename = path.stem
            self.season_id, self.season_window = self._parse_season_info_from_filename(
                filename
            )
            if self.season_id is None or self.season_window is None:
                print(
                    "Failed to parse season information from filename. Make sure it follows the format: embeddings_season-YYYYMMDD-YYYYMMDD_cl-x_YYYYMMDD-HHMMSS.csv"
                )
                return
            print(
                f"Parsed season ID: {self.season_id}, season window: {self.season_window.start_date} to {self.season_window.end_date}"
            )
            print(f"Embeddings loaded: {path}")
        self.embeddings_df_path = path
        self.tab5_df = df
        if status_message is not None:
            status_message.value = "<i>Embeddings loaded from file.</i>"
        self._update_tab6_state()

    def _on_train_click(self, button):
        """Handle training click."""
        progress_output = self.tab6_widgets["progress_output"]
        head_task_dropdown = self.tab6_widgets.get("head_task_dropdown")
        head_type_dropdown = self.tab6_widgets.get("head_type_dropdown")
        epochs_input = self.tab6_widgets.get("epochs_input")
        lr_input = self.tab6_widgets.get("lr_input")
        weight_decay_input = self.tab6_widgets.get("weight_decay_input")
        use_balancing_checkbox = self.tab6_widgets.get("use_balancing_checkbox")
        custom_suffix_input = self.tab6_widgets.get("custom_suffix_input")
        df = self.tab5_df

        with progress_output:
            progress_output.clear_output()
            if df is None or df.empty:
                print("No embeddings available. Compute or load embeddings first.")
                return
            if self.season_id is None:
                print(
                    "Season ID missing. Complete Tab 3 season alignment or load a valid training dataframe containing a season ID in its name."
                )
                return
            head_task = head_task_dropdown.value if head_task_dropdown else "croptype"
            head_type = head_type_dropdown.value if head_type_dropdown else "linear"
            epochs = epochs_input.value if epochs_input else 40
            lr = float(lr_input.value) if lr_input is not None else 1e-2
            weight_decay = (
                float(weight_decay_input.value)
                if weight_decay_input is not None
                else 0.0
            )
            use_balancing = (
                use_balancing_checkbox.value
                if use_balancing_checkbox is not None
                else True
            )

            base_output_dir = "./downstream_heads"
            timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
            base_name = self.embeddings_df_path.stem
            model_name = base_name.replace("embeddings", f"{head_task}head-{head_type}")
            if custom_suffix_input and custom_suffix_input.value.strip():
                suffix = custom_suffix_input.value.strip().replace(" ", "-")
                model_name += f"_{suffix}"
            model_name += f"_{timestamp}"
            self.head_output_path = Path(base_output_dir) / model_name
            self.head_output_path.mkdir(parents=True, exist_ok=True)

            try:
                print("Training started...")
                train_seasonal_torch_head(
                    df,
                    season_id=self.season_id,
                    head_task=head_task,
                    output_dir=self.head_output_path,
                    head_type=head_type,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    use_balancing=use_balancing,
                    num_workers=0,
                )
                print(
                    f"Training completed. Artifacts saved to: {self.head_output_path}"
                )
                config_path = self.head_output_path / "config.json"
                if config_path.exists():
                    with config_path.open() as fp:
                        head_config = json.load(fp)
                    package_name = (
                        head_config.get("artifacts", {}).get("packages", {}).get("head")
                    )
                    package_path = self.head_output_path / package_name
                    if package_path.exists():
                        self.head_package_path = package_path
                    else:
                        print(
                            f"Warning: head archive {package_name} not found in output directory, something went wrong!"
                        )
                    print(f"Torch head archive ready at: {self.head_package_path}")
                    self._update_tab7_state()
                else:
                    print(
                        "Warning: training config.json not found; unable to locate the head archive automatically."
                    )
            except Exception as exc:
                print(f"Training failed: {exc}")

    # =========================================================================
    # Tab 7: Deploy Model
    # =========================================================================

    def _build_tab7_deploy_model(self) -> widgets.VBox:
        """Build Tab 7: Deploy trained model."""
        header = widgets.HTML(value="<h2>Deploy Model</h2>")

        status_message = widgets.HTML(
            value="<i>Finish training a model in Tab 6 first or load a trained torch head archive (.zip) using the button below.</i>"
        )

        deploy_info = self._info_callout(
            "The training step produced a zipped bundle containing the PyTorch weights plus the accompanying configuration.<br>"
            "Here, that zip file is uploaded to your private CDSE artifact bucket so the openEO classification workflow can download and access it later.<br><br>"
            "Below, you see the name of the model that is ready for upload.<br>"
            "Clicking the 'Deploy Model' button will generate a temporary (signed) model URL that we need to generate a classification map.<br><br>"
            "Make sure to <b>always keep a local copy of your trained model and its config file</b>!<br>"
            "The cloud copy is only a temporary staging location for deployment and will be deleted after some time.<br><br>"
        )

        load_title = widgets.HTML(
            value="<h3>Option to load a torch head archive from disk</h3>"
        )
        load_input = widgets.Text(
            value="",
            description="Full path to torch head (.zip):",
            placeholder="/path/to/your_head.zip",
            layout=widgets.Layout(width="100%"),
            description_width="220px",
        )
        load_button = widgets.Button(
            description="Load head archive",
            button_style="info",
            icon="folder-open",
            layout=widgets.Layout(width="220px", height="40px"),
        )
        load_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        model_path_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        auth_info = self._info_callout(
            "Your CDSE credentials will be cached locally after your first login on your computer.<br>"
            "If you would like to switch to a different CDSE account, you can click the 'Reset authentication' button below to clear the cached credentials and trigger a new login on your next deploy attempt.<br>"
        )
        reset_auth_button = widgets.Button(
            description="Reset authentication",
            button_style="warning",
            icon="refresh",
            layout=widgets.Layout(width="220px", height="40px"),
        )
        auth_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        deploy_button = widgets.Button(
            description="Deploy Model",
            button_style="info",
            icon="cloud-upload",
            layout=widgets.Layout(width="200px", height="40px"),
            disabled=True,
        )

        report_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab7_widgets = {
            "status_message": status_message,
            "load_title": load_title,
            "load_input": load_input,
            "load_button": load_button,
            "load_output": load_output,
            "model_path_output": model_path_output,
            "auth_info": auth_info,
            "deploy_button": deploy_button,
            "report_output": report_output,
            "auth_output": auth_output,
            "reset_auth_button": reset_auth_button,
        }
        load_button.on_click(self._on_tab7_load_model)
        deploy_button.on_click(self._on_deploy_click)
        reset_auth_button.on_click(self._on_reset_cdse_auth_click)

        return widgets.VBox(
            [
                header,
                status_message,
                deploy_info,
                load_title,
                load_input,
                widgets.HBox([load_button]),
                load_output,
                widgets.HTML("<h3>Model to be deployed:</h3>"),
                model_path_output,
                widgets.HTML("<h3>CDSE authentication</h3>"),
                auth_info,
                widgets.HBox([reset_auth_button]),
                auth_output,
                widgets.HTML("<h3>Deployment</h3>"),
                widgets.HBox(
                    [deploy_button],
                    layout=widgets.Layout(justify_content="center", margin="20px 0"),
                ),
                widgets.HTML("<b>Deployment Status:</b>"),
                report_output,
                self._build_tab_navigation(),
            ]
        )

    def _on_tab7_load_model(self, _=None) -> None:
        """Load a torch head archive path for deployment."""
        load_input = self.tab7_widgets.get("load_input")
        output = self.tab7_widgets.get("load_output")
        status_message = self.tab7_widgets.get("status_message")
        if load_input is None or output is None:
            return

        path_value = load_input.value.strip()
        with output:
            output.clear_output()
            if not path_value:
                print("Provide a path to a torch head .zip file.")
                return
            path = Path(path_value)
            if not path.exists():
                print(f"File not found: {path}")
                return
            if path.suffix.lower() != ".zip":
                print("The provided file is not a .zip archive.")
                return
            self.head_package_path = path
            self.head_output_path = path.parent
            filename = path.parent.stem
            self.season_id, self.season_window = self._parse_season_info_from_filename(
                filename
            )
            if self.season_id is None or self.season_window is None:
                print(
                    "Failed to parse season information from filename. Make sure it follows the format: yourhead_season-YYYYMMDD-YYYYMMDD_*.zip"
                )
                return
            print(
                f"Parsed season ID: {self.season_id}, season window: {self.season_window.start_date} to {self.season_window.end_date}"
            )

            print(f"Torch head archive loaded from: {path}")

        if status_message is not None:
            status_message.value = "<i>Torch head archive loaded. Ready to deploy.</i>"
        self._update_tab7_state()

    def _on_deploy_click(self, button):
        """Handle deploy click."""
        report_output = self.tab7_widgets["report_output"]
        with report_output:
            report_output.clear_output()
            if self.head_package_path is None:
                print(
                    "No torch head archive available. Train a model or load a .zip file first."
                )
                return
            if not self.head_package_path.exists():
                print(f"Torch head archive not found: {self.head_package_path}")
                return
            if self.head_package_path.suffix.lower() != ".zip":
                print("Torch head archive must be a .zip file.")
                return
            target_object_name = self.head_package_path.name
            print(f"Uploading torch head archive as {target_object_name} ...")

        try:
            artifact_helper = OpenEOArtifactHelper.from_openeo_connection(
                cdse_connection()
            )
            model_s3_uri = artifact_helper.upload_file(
                target_object_name, str(self.head_package_path)
            )
            model_url = artifact_helper.get_presigned_url(model_s3_uri)
        except Exception as exc:
            print(f"Deployment failed: {exc}")

        with report_output:
            self.tab7_model_url = model_url
            print(f"S3 URI: {model_s3_uri}")
            print(f"Your torch head can be downloaded from: {model_url}")
            print(
                "You can proceed to the next step to generate a map using your deployed model."
            )
            self._update_tab8_state()
            self._update_tab9_state()

    # =========================================================================
    # Tab 8: Generate Map
    # =========================================================================

    def _build_tab8_generate_map(self) -> widgets.VBox:
        """Build Tab 8: Generate map."""
        header = widgets.HTML(value="<h2>Generate Map</h2>")

        status_message = widgets.HTML(
            value="<i>Configure and deploy a model in previous steps first.</i>"
        )

        upscaling_short = widgets.HTML(
            "<i>First time here? Read this important note on large-scale map production!</i>"
        )
        upscaling_info = self._info_callout(
            "We ALWAYS recommend you to first run your model on a <b>representative set of small test areas</b> (up to 100 km²) to visually check for model performance BEFORE upscaling to large areas!!<br><br>"
            "By default, CDSE users are limited to running 2 processing jobs in parallel, which will result in long processing times for large areas.<br>"
            'When engaging in country-scale mapping, we therefore recommend to <b>contact the WorldCereal team</b> for dedicated support to speed up processing through <a href="https://esa-worldcereal.org/en/contact">our contact form</a>.'
        )

        aoi_message = widgets.HTML(
            value="<i>Select your area of interest (AOI) using the map below.</i>"
        )
        aoi_info = self._info_callout(
            "The WorldCereal system is optimized to process 50 x 50 km tiles.<br>"
            "Large AOIs are automatically split into tiles for processing.<br>"
            "You can manually alter the <b>tile size</b> if you want to experiment with smaller or larger tiles, but keep in mind that this will also impact processing time and the required computational resources.<br><br>"
            "You can draw a rectangle using the drawing tools on the left side of the map.<br>"
            "The app will automatically store the coordinates of the last rectangle you drew on the map.<br><br>"
            "Alternatively, you can also upload a vector file (either zipped shapefile or GeoPackage) delineating your area of interest.<br>"
            "In case your vector file contains multiple polygons or points, the total bounds will be automatically computed and serve as your AOI.<br>"
            "Files containing only a single point are not allowed.<br><br>"
        )
        tile_resolution_input = widgets.IntText(
            value=50,
            description="Tile size (km):",
            layout=widgets.Layout(width="220px"),
        )
        aoi_map = ui_map(display_ui=False)

        bbox_info = self._info_callout(
            "After drawing your AOI on the map, you can save the bounding box coordinates to the local `./bbox` folder for later reuse.<br>"
            "The name you provide in the input field will be used as the filename (without extension).<br>"
            "For example, if you enter 'my_aoi' and click the save button, the AOI coordinates will be saved to `./bbox/my_aoi.gpkg`.<br><br>"
            "This is especially useful when you want to run multiple experiments with the same AOI or want to keep a record of the AOI coordinates that were used for processing.<br>"
            "You can also load the saved bounding box files later to visualize the AOI on the map again or to use the coordinates for other purposes."
        )
        bbox_name_input = widgets.Text(
            value="",
            description="Save AOI as:",
            placeholder="name without extension",
            layout=widgets.Layout(width="60%", margin="0 0 0 12px"),
        )
        bbox_save_button = widgets.Button(
            description="Save AOI to ./bbox",
            button_style="info",
            icon="save",
            layout=widgets.Layout(width="220px"),
        )
        bbox_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        season_message = widgets.HTML(
            value="<i>Select your growing season and year.</i>"
        )
        season_info = self._info_callout(
            "Use the slider to define your growing season of interest(max 12 months).<br>"
            "The tool automatically also derives a 12-month processing period that ends on your selected end month."
        )
        season_hint = widgets.HTML(value="<i>No model loaded yet.</i>")
        season_slider_output = widgets.Output(
            layout=widgets.Layout(width="100%", overflow="auto")
        )
        season_slider_obj = None
        with season_slider_output:
            season_slider_obj = season_slider()
        season_id_input = widgets.Text(
            value=self.season_id,
            description="Season ID:",
            placeholder="e.g., ShortRains",
            layout=widgets.Layout(width="60%", margin="0 0 0 12px"),
            tooltip="Provide a short name for your season. No spaces or special characters allowed, only letters and numbers.",
        )

        processing_params_message = widgets.HTML(
            value="<i>Set other processing parameters for map generation.</i>"
        )
        processing_params_info = self._info_callout(
            "The following parameters are available to configure the map generation process:<br><br>"
            "<b>Output suffix:</b> An optional label that will be appended to the output folder name to help you identify different runs or configurations.<br>"
            "      By default, your output folder receives the name of your model and selected growing season.<br>"
            "<b>Mask cropland:</b> By default enabled to mask out non-cropland areas in the output map. Depending on your use case, you may want to disable this to get predictions for all land cover types.<br>"
            "<b>Export cropland results:</b> By default enabled to export the cropland head predictions as a separate layer in the output map. Disable if you are not interested in the cropland results to save processing time and output storage space.<br>"
            "<b>Export class probabilities:</b> By default enabled to export the predicted class probabilities as separate layers in the output map. Disable if you are not interested in the probabilities to save processing time and output storage space.<br>"
            "<b>Postprocessing:</b> By default enabled to run a majority vote postprocessing step on the predicted classes to smooth the results.<br>"
            "       Depending on your use case, you may want to disable postprocessing to get the raw model predictions without any smoothing.<br>"
            "       Postprocessing can be enabled/disabled separately for the crop type and cropland predictions.<br>"
            "       For the majority vote method, you can adjust the <b>kernel size</b> of the majority vote filter to make it more or less aggressive.<br>"
            "       You can also opt for the simpler and less aggressive <b>'smooth_probabilities'</b> method."
        )
        output_name_input = widgets.Text(
            value="",
            description="Output suffix:",
            placeholder="optional suffix for output folder name",
            layout=widgets.Layout(width="60%", margin="0 0 0 12px"),
            tooltip="Keep it short and avoid spaces and special characters.",
        )

        mask_cropland_checkbox = widgets.Checkbox(
            value=True,
            description="Mask cropland",
        )
        enable_cropland_head_checkbox = widgets.Checkbox(
            value=True,
            description="Export cropland results",
        )
        export_probs_checkbox = widgets.Checkbox(
            value=True,
            description="Export class probabilities",
        )
        croptype_postprocess_enabled = widgets.Checkbox(
            value=True,
            description="Run postprocessing on crop type results",
            description_width="300px",
        )
        croptype_postprocess_method = widgets.Dropdown(
            options=["majority_vote", "smooth_probabilities"],
            value="majority_vote",
            description="Method:",
            layout=widgets.Layout(width="260px"),
        )
        croptype_postprocess_kernel = widgets.IntText(
            value=5,
            description="Kernel:",
            layout=widgets.Layout(width="220px"),
        )
        cropland_postprocess_enabled = widgets.Checkbox(
            value=True,
            description="Run postprocessing on cropland results",
            description_width="300px",
        )
        cropland_postprocess_method = widgets.Dropdown(
            options=["majority_vote", "smooth_probabilities"],
            value="majority_vote",
            description="Method:",
            layout=widgets.Layout(width="260px"),
        )
        cropland_postprocess_kernel = widgets.IntText(
            value=3,
            description="Kernel:",
            layout=widgets.Layout(width="220px"),
        )

        generate_button = widgets.Button(
            description="Generate Map",
            button_style="success",
            icon="map",
            layout=widgets.Layout(width="200px", height="40px"),
            disabled=True,
        )

        output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab8_widgets = {
            "status_message": status_message,
            "aoi_map": aoi_map,
            "bbox_name_input": bbox_name_input,
            "bbox_save_button": bbox_save_button,
            "bbox_output": bbox_output,
            "season_slider": season_slider_obj,
            "season_slider_output": season_slider_output,
            "season_hint": season_hint,
            "season_id_input": season_id_input,
            "output_name_input": output_name_input,
            "mask_cropland_checkbox": mask_cropland_checkbox,
            "enable_cropland_head_checkbox": enable_cropland_head_checkbox,
            "export_probs_checkbox": export_probs_checkbox,
            "croptype_postprocess_enabled": croptype_postprocess_enabled,
            "croptype_postprocess_method": croptype_postprocess_method,
            "croptype_postprocess_kernel": croptype_postprocess_kernel,
            "cropland_postprocess_enabled": cropland_postprocess_enabled,
            "cropland_postprocess_method": cropland_postprocess_method,
            "cropland_postprocess_kernel": cropland_postprocess_kernel,
            "tile_resolution_input": tile_resolution_input,
            "generate_button": generate_button,
            "output": output,
        }

        bbox_save_button.on_click(self._on_tab8_save_bbox)
        generate_button.on_click(self._on_generate_map_click)

        return widgets.VBox(
            [
                header,
                status_message,
                widgets.HTML("<h3>Note on upscaling</h3>"),
                upscaling_short,
                upscaling_info,
                widgets.HTML("<h3>1) Select your area of interest (AOI)</h3>"),
                aoi_message,
                aoi_info,
                widgets.HBox([tile_resolution_input]),
                aoi_map.map,
                aoi_map.output,
                bbox_info,
                widgets.HBox([bbox_name_input, bbox_save_button]),
                bbox_output,
                widgets.HTML("<h3>2) Select season</h3>"),
                season_message,
                season_info,
                season_hint,
                season_slider_output,
                widgets.HBox([season_id_input]),
                widgets.HTML("<h3>3) Processing parameters</h3>"),
                processing_params_message,
                processing_params_info,
                widgets.HBox([output_name_input]),
                widgets.HBox([mask_cropland_checkbox, enable_cropland_head_checkbox]),
                widgets.HBox([export_probs_checkbox]),
                widgets.HBox([croptype_postprocess_enabled]),
                widgets.HBox(
                    [croptype_postprocess_method, croptype_postprocess_kernel]
                ),
                widgets.HBox([cropland_postprocess_enabled]),
                widgets.HBox(
                    [cropland_postprocess_method, cropland_postprocess_kernel]
                ),
                widgets.HTML("<h3>4) Generate your map!</h3>"),
                widgets.HBox(
                    [generate_button],
                    layout=widgets.Layout(justify_content="center", margin="20px 0"),
                ),
                widgets.HTML("<b>Map Generation Status:</b>"),
                output,
                self._build_tab_navigation(),
            ]
        )

    def _on_tab8_save_bbox(self, _=None) -> None:
        """Save the current AOI to ./bbox as a GeoPackage."""
        output = self.tab8_widgets.get("bbox_output")
        name_input = self.tab8_widgets.get("bbox_name_input")
        aoi_map = self.tab8_widgets.get("aoi_map")
        if output is None or name_input is None or aoi_map is None:
            return

        with output:
            output.clear_output()
            name = name_input.value.strip()
            if not name:
                print("Provide a name for your bounding box.")
                return
            bbox_dir = Path("./bbox")
            bbox_dir.mkdir(exist_ok=True)
            outfile = bbox_dir / f"{name}.gpkg"
            try:
                processing_extent = aoi_map.get_extent(projection="latlon")
                if processing_extent is None:
                    print("Draw an AOI on the map before saving.")
                    return
                bbox_extent_to_gdf(processing_extent, outfile)
                print(f"AOI saved to {outfile}")
            except Exception as exc:
                print(f"Failed to save AOI: {exc}")

    def _on_generate_map_click(self, button):
        """Handle map generation click."""
        status_message = self.tab8_widgets.get("status_message")
        aoi_map = self.tab8_widgets.get("aoi_map")
        season_slider_obj = self.tab8_widgets.get("season_slider")
        mask_cropland_checkbox = self.tab8_widgets.get("mask_cropland_checkbox")
        enable_cropland_head_checkbox = self.tab8_widgets.get(
            "enable_cropland_head_checkbox"
        )
        export_probs_checkbox = self.tab8_widgets.get("export_probs_checkbox")
        croptype_postprocess_enabled = self.tab8_widgets.get(
            "croptype_postprocess_enabled"
        )
        croptype_postprocess_method = self.tab8_widgets.get(
            "croptype_postprocess_method"
        )
        croptype_postprocess_kernel = self.tab8_widgets.get(
            "croptype_postprocess_kernel"
        )
        cropland_postprocess_enabled = self.tab8_widgets.get(
            "cropland_postprocess_enabled"
        )
        cropland_postprocess_method = self.tab8_widgets.get(
            "cropland_postprocess_method"
        )
        cropland_postprocess_kernel = self.tab8_widgets.get(
            "cropland_postprocess_kernel"
        )
        tile_resolution_input = self.tab8_widgets.get("tile_resolution_input")
        output_name_input = self.tab8_widgets.get("output_name_input")
        season_id_input = self.tab8_widgets.get("season_id_input")
        output = self.tab8_widgets.get("output")

        # Clear previous output and create separate widgets inside
        if output is not None:
            output.clear_output()

        # Create log_out and plot_out widgets inside the output context
        log_out = widgets.Output()
        plot_out = widgets.Output()

        # Display them inside the main output widget
        with output:
            display(log_out)
            display(plot_out)

        # Validate inputs and show messages in log_out
        if self.tab7_model_url is None:
            with log_out:
                print("Model URL not available. Deploy a model in Tab 7 first.")
            return
        if aoi_map is None:
            with log_out:
                print("AOI map not initialized.")
            return
        if season_slider_obj is None:
            with log_out:
                print("Season slider not initialized.")
            return
        try:
            selection = season_slider_obj.get_selection()
            self.tab8_processing_period = selection.processing_period
            self.tab8_season_window = selection.season_window
        except Exception as exc:
            with log_out:
                print(f"Failed to read season selection: {exc}")
            return

        # Extract parameters
        mask_cropland = (
            mask_cropland_checkbox.value if mask_cropland_checkbox is not None else True
        )
        enable_cropland_head = (
            enable_cropland_head_checkbox.value
            if enable_cropland_head_checkbox is not None
            else True
        )
        export_class_probs = (
            export_probs_checkbox.value if export_probs_checkbox is not None else True
        )
        croptype_pp_enabled = (
            croptype_postprocess_enabled.value
            if croptype_postprocess_enabled is not None
            else True
        )
        croptype_pp_method = (
            croptype_postprocess_method.value
            if croptype_postprocess_method is not None
            else "majority_vote"
        )
        croptype_pp_kernel = (
            croptype_postprocess_kernel.value
            if croptype_postprocess_kernel is not None
            else 5
        )
        cropland_pp_enabled = (
            cropland_postprocess_enabled.value
            if cropland_postprocess_enabled is not None
            else True
        )
        cropland_pp_method = (
            cropland_postprocess_method.value
            if cropland_postprocess_method is not None
            else "majority_vote"
        )
        cropland_pp_kernel = (
            cropland_postprocess_kernel.value
            if cropland_postprocess_kernel is not None
            else 3
        )
        tile_resolution = (
            tile_resolution_input.value if tile_resolution_input is not None else 50
        )

        # Start processing in background thread to avoid blocking the UI, and show logs in log_out
        try:
            run_suffix = output_name_input.value.strip() if output_name_input else ""
            model_name = self.head_package_path.stem
            season_start = pd.Timestamp(self.tab8_season_window.start_date)
            season_end = pd.Timestamp(self.tab8_season_window.end_date)
            season_extent = (
                f"{season_start.strftime('%Y%m%d')}-{season_end.strftime('%Y%m%d')}"
            )
            output_dir = Path("./runs") / f"{model_name}_{run_suffix}_{season_extent}"

            if output_dir.exists():
                with log_out:
                    print(
                        f"Output directory {output_dir} already exists. Choose a different output suffix to avoid overwriting."
                    )
                return
            output_dir.mkdir(parents=True, exist_ok=True)

            processing_extent = aoi_map.get_extent(projection="latlon")
            if processing_extent is None:
                with log_out:
                    print("Draw an AOI on the map before generating a map.")
                return

            season_id = (
                season_id_input.value.strip() if season_id_input is not None else ""
            )
            if not season_id or season_id == "":
                with log_out:
                    print("Provide a season ID before generating a map.")
                return
            if not season_id.isalnum():
                with log_out:
                    print(
                        "Season ID must be alphanumeric (no spaces or special characters)."
                    )
                return

            season_windows = {
                season_id: (
                    season_start.strftime("%Y-%m-%d"),
                    season_end.strftime("%Y-%m-%d"),
                )
            }

            workflow_builder = (
                WorldCerealWorkflowConfig.builder()
                .season_ids([season_id])
                .season_windows(season_windows)
                .croptype_head_zip(self.tab7_model_url)
                .enable_croptype_head(True)
                .enable_cropland_head(enable_cropland_head)
                .enforce_cropland_gate(mask_cropland)
                .export_class_probabilities(export_class_probs)
            )
            workflow_builder = workflow_builder.cropland_postprocess(
                enabled=cropland_pp_enabled,
                method=cropland_pp_method,
                kernel_size=cropland_pp_kernel,
            )
            workflow_builder = workflow_builder.croptype_postprocess(
                enabled=croptype_pp_enabled,
                method=croptype_pp_method,
                kernel_size=croptype_pp_kernel,
            )
            workflow_config = workflow_builder.build()

            if status_message is not None:
                status_message.value = (
                    "<i>Processing started... this may take a while.</i>"
                )

            try:
                _ = run_map_production(
                    spatial_extent=processing_extent,
                    temporal_extent=self.tab8_processing_period,
                    output_dir=output_dir,
                    tile_resolution=tile_resolution,
                    product_type=WorldCerealProductType.CROPTYPE,
                    workflow_config=workflow_config,
                    stop_event=None,
                    plot_out=plot_out,
                    log_out=log_out,
                    display_outputs=True,
                )
                self.tab8_results = output_dir
                if status_message is not None:
                    status_message.value = "<i>Processing finished.</i>"
                with log_out:
                    print(
                        "\n\nProcessing finished. Outputs saved to: " f"{output_dir}\n"
                    )
                self._update_tab9_state()
            except Exception as exc_inner:
                if status_message is not None:
                    status_message.value = "<i>Processing failed. Check logs below.</i>"
                with log_out:
                    print(f"\n\nMap generation failed: {exc_inner}\n")
        except Exception as exc:
            if status_message is not None:
                status_message.value = "<i>Processing failed. Check logs below.</i>"
            with log_out:
                print(f"Map generation failed: {exc}")

    # =========================================================================
    # Tab 9: Visualize Map
    # =========================================================================

    def _build_tab9_visualize_results(self) -> widgets.VBox:
        """Build Tab 9: Visualize map."""
        header = widgets.HTML(value="<h2>Visualize Map</h2>")

        status_message = widgets.HTML(
            value="<i>Please generate a map in Tab 8 or load a results folder below.</i>"
        )

        results_title = widgets.HTML(
            value="<h3>Option to load a results folder from disk</h3>"
        )
        results_input = widgets.Text(
            value="",
            description="Results folder:",
            placeholder="/path/to/runs/Presto...",
            layout=widgets.Layout(width="100%", margin="0 0 0 20px"),
        )
        results_button = widgets.Button(
            description="Load results folder",
            button_style="info",
            icon="folder-open",
            layout=widgets.Layout(width="220px", height="40px"),
        )
        results_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="80px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        products_info = self._info_callout(
            "With each map generation run, up to four products are generated:<br>"
            "- <code>croptype-raw</code>: custom crop type product<br>"
            "- <code>croptype</code>: post-processed crop type product<br>"
            "- <code>cropland-raw</code>: cropland mask (raw)<br>"
            "- <code>cropland</code>: cropland mask (post-processed)<br><br>"
            "Each raster includes at least two bands:<br>"
            "1) winning class label<br>"
            "2) winning class probability (50-100)<br>"
            "3+) class probabilities (optional)<br><br>"
            "By default, all available products will be visualized below.<br><br>"
            "For a more detailed inspection of the products, we advise to use QGIS for visualization."
        )

        model_url_info = self._info_callout(
            "The visualization tool needs to know which model was used to generate the map in order to correctly interpret the class labels and probabilities.<br><br>"
            "If you generated the map yourself in Tab 8, the model URL is pre-filled for you.<br>"
            "In case your model is no longer hosted on your private CDSE bucket (URL expired), make sure to deploy it again in Tab 7.<br><br>"
            "If you want to visualize products which have been generated before, you will always need to provide the URL of the model that was used for product generation.<br>"
        )
        model_url_input = widgets.Text(
            value=self.tab9_model_url or self.tab7_model_url or "",
            description="Model URL:",
            placeholder="https://...",
            layout=widgets.Layout(width="100%", margin="0 0 0 20px"),
        )
        model_url_button = widgets.Button(
            description="Use model URL",
            button_style="warning",
            icon="link",
            layout=widgets.Layout(width="200px", height="40px"),
        )
        model_url_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="60px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        merge_info = self._info_callout(
            "If you generated your map using the default settings in Tab 8, the output is split into multiple tiles that are stored in a single folder. In order to visualize the map, these tiles need to be merged back into a single raster file.<br>"
            "Click the button below to run the merging step.<br><br>"
            "In case an existing merged product is found, it will be overwritten.<br>"
        )
        merge_button = widgets.Button(
            description="Merge tiles",
            button_style="warning",
            icon="compress",
            layout=widgets.Layout(width="200px", height="40px"),
            disabled=True,
        )
        merge_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        visualization_info = self._info_callout(
            "By default your products are shown with matplotlib for quick inspection.<br>"
            "Only the 'classification' layer is shown in this case.<br><br>"
            "Set <b>Interactive mode</b> to use an ipyleaflet viewer where you can toggle layers and inspect all products, along with their probability layers, at once.<br>"
            "In the interactive mode, you can toggle on/off individual layers by clicking the upper-right icon.<br><br>"
            "<b>NOTE:</b> in order for the interactive mode to work in a VSCode environment, you need to switch on port forwarding for port 8889."
        )
        interactive_checkbox = widgets.Checkbox(
            value=False,
            description="Interactive mode",
        )
        visualize_button = widgets.Button(
            description="Visualize",
            button_style="info",
            icon="eye",
            layout=widgets.Layout(width="200px", height="40px"),
            disabled=True,
        )
        visualize_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        self.tab9_widgets = {
            "status_message": status_message,
            "results_title": results_title,
            "results_input": results_input,
            "results_button": results_button,
            "results_output": results_output,
            "model_url_input": model_url_input,
            "model_url_button": model_url_button,
            "model_url_output": model_url_output,
            "interactive_checkbox": interactive_checkbox,
            "merge_button": merge_button,
            "visualize_button": visualize_button,
            "merge_output": merge_output,
            "visualize_output": visualize_output,
        }

        results_button.on_click(self._on_tab9_load_results)
        model_url_button.on_click(self._on_tab9_use_model_url)
        merge_button.on_click(self._on_tab9_merge_maps)
        visualize_button.on_click(self._on_visualize_click)

        return widgets.VBox(
            [
                header,
                status_message,
                products_info,
                results_title,
                results_input,
                widgets.HBox([results_button]),
                results_output,
                widgets.HTML(
                    "<h3>Provide the URL of the model used for product generation</h3>"
                ),
                model_url_info,
                model_url_input,
                widgets.HBox([model_url_button]),
                model_url_output,
                widgets.HTML("<h3>1) Create merged product</h3>"),
                merge_info,
                widgets.HBox([merge_button]),
                merge_output,
                widgets.HTML("<h3>2) Visualize map</h3>"),
                visualization_info,
                widgets.HBox([interactive_checkbox]),
                widgets.HBox(
                    [visualize_button],
                    layout=widgets.Layout(justify_content="center", margin="20px 0"),
                ),
                widgets.HTML("<b>Visualization:</b>"),
                visualize_output,
                self._build_tab_navigation(),
            ]
        )

    def _on_tab9_load_results(self, _=None) -> None:
        """Load a results folder for visualization."""
        results_input = self.tab9_widgets.get("results_input")
        output = self.tab9_widgets.get("results_output")
        status_message = self.tab9_widgets.get("status_message")
        if results_input is None or output is None:
            return

        path_value = results_input.value.strip()
        with output:
            output.clear_output()
            if not path_value:
                print("Provide a results folder path.")
                return
            path = Path(path_value)
            if not path.exists():
                print(f"Folder not found: {path}")
                return
            if not path.is_dir():
                print("Results path must be a folder.")
                return
            self.tab8_results = path
            self.tab9_merged_paths = {}
            print(f"Results folder loaded: {path}")

        if status_message is not None:
            status_message.value = "<i>Results folder loaded. Ready to merge tiles.</i>"
        self._update_tab9_state()

    def _on_tab9_use_model_url(self, _=None) -> None:
        """Use a model URL for visualization."""
        model_url_input = self.tab9_widgets.get("model_url_input")
        output = self.tab9_widgets.get("model_url_output")
        status_message = self.tab9_widgets.get("status_message")
        if model_url_input is None or output is None:
            return

        with output:
            output.clear_output()
            model_url = model_url_input.value.strip()
            if not model_url:
                print("Provide a model URL.")
                return
            if not self._is_valid_url(model_url):
                print("Model URL must start with http:// or https://")
                return
            self.tab9_model_url = model_url
            print("Model URL set!")

        if status_message is not None:
            status_message.value = "<i>Model URL set for visualization.</i>"

    def _on_tab9_merge_maps(self, _=None) -> None:
        """Merge output tiles into a single product."""
        output = self.tab9_widgets.get("merge_output")
        status_message = self.tab9_widgets.get("status_message")
        if output is None:
            return

        with output:
            output.clear_output()
            results_dir = self.tab8_results
            if results_dir is None:
                print(
                    "No results folder available. Generate a map in Tab 8 or load a folder above."
                )
                return
            if not results_dir.exists():
                print(f"Results folder not found: {results_dir}")
                return
            try:
                merged_paths = merge_maps(results_dir)
                self.tab9_merged_paths = {k: Path(v) for k, v in merged_paths.items()}
                if self.tab9_merged_paths:
                    merged_list = "\n".join(
                        f"{name} -> {path}"
                        for name, path in self.tab9_merged_paths.items()
                    )
                    print("Results merged:\n" + merged_list)
                else:
                    print("No products were merged.")
                if status_message is not None:
                    status_message.value = "<i>Merge completed. Ready to visualize.</i>"
                self._update_tab9_state()
            except Exception as exc:
                print(f"Merge failed: {exc}")

    def _on_visualize_click(self, button):
        """Handle visualize click."""
        output = self.tab9_widgets.get("visualize_output")
        interactive_checkbox = self.tab9_widgets.get("interactive_checkbox")
        if output is None:
            return

        with output:
            output.clear_output()
            if not self.tab9_merged_paths:
                print("Merged products not available. Run the merge step first.")
                return
            model_url = self.tab9_model_url or self.tab7_model_url
            if model_url is None:
                print(
                    "Model URL not available. Provide one above or deploy in Tab 7 first."
                )
                return
            try:
                artifact = load_model_artifact(model_url)
                heads = artifact.manifest.get("heads", [])
                interactive_mode = (
                    interactive_checkbox.value
                    if interactive_checkbox is not None
                    else False
                )
                luts = {}
                for task in ("cropland", "croptype"):
                    head = next(
                        (head for head in heads if head.get("task") == task), None
                    )
                    if head and head.get("class_names"):
                        luts[task] = {
                            name: idx for idx, name in enumerate(head["class_names"])
                        }
                if "croptype" not in luts:
                    print("Torch head manifest is missing croptype class metadata.")
                    return

                result = visualize_products(
                    self.tab9_merged_paths,
                    luts=luts,
                    interactive_mode=interactive_mode,
                )
                if interactive_mode and result is not None:
                    display(result)
            except Exception as exc:
                print(f"Visualization failed: {exc}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _on_tab_change(self, change):
        """Update UI state when tabs change."""
        self._update_nav_buttons()
        self._update_tab2_state()
        self._update_tab3_state()
        self._update_tab4_state()
        self._update_tab5_state()
        self._update_tab6_state()
        self._update_tab7_state()
        self._update_tab8_state()
        self._update_tab9_state()

    def _update_tab2_state(self):
        """Enable/disable Tab 2 depending on Tab 1 state."""
        status_message = self.tab2_widgets.get("status_message")
        if status_message:
            if self.workflow_mode == "inference-only":
                status_message.value = "<i>Skipped (inference-only mode).</i>"
                return
            is_ready = self.tab1_saved and self.tab1_df is not None
            if is_ready:
                self._init_tab2_extractions_copy()
                self._update_tab2_summary()
                self._on_tab2_refresh_datasets()
            else:
                status_message.value = "<i>Please retrieve data in Tab 1 first.</i>"

    def _update_tab3_state(self):
        """Enable/disable Tab 3 (season alignment) depending on Tab 1 state."""
        status_message = self.tab3_widgets.get("status_message")
        if status_message:
            if self.workflow_mode == "inference-only":
                status_message.value = "<i>Skipped (inference-only mode).</i>"
                return
            df = self._get_tab2_extractions_df()
            if df is None:
                status_message.value = "<i>Please complete or skip Tab 2 first.</i>"
                return
            summary = self._format_extractions_summary(df)
            if summary is None:
                status_message.value = "<i>No extractions loaded yet.</i>"
                return
            status_message.value = (
                "<i>In this step we ensure we only work with reference data relevant to your season of interest.</i><br><br>"
                "<b>Summary of available data:</b><br>"
                f"{summary}"
            )

    def _update_tab4_state(self):
        """Refresh Tab 4 (crop selection & data prep) state."""
        if self.workflow_mode == "inference-only":
            return
        status_message = self.tab4_widgets.get("status_message")
        df = self.tab3_df
        if status_message is not None:
            if df is None:
                status_message.value = "<i>Please align data in Tab 3 first.</i>"
            else:
                summary = self._format_extractions_summary(df)
                if summary is None:
                    status_message.value = "<i>No extractions loaded yet.</i>"
                else:
                    status_message.value = (
                        "<i>In this step we select the final list of classes to be used for model training.</i><br><br>"
                        "<b>Summary of available data:</b><br>"
                        f"{summary}"
                    )
        if self.tab4_df is not None:
            self._render_tab4_summary()

    def _update_tab5_state(self):
        """Enable/disable Tab 5 (embeddings) depending on prepared samples."""
        embeddings_button = self.tab5_widgets.get("embeddings_button")
        status_message = self.tab5_widgets.get("status_message")
        load_title = self.tab5_widgets.get("load_title")
        load_input = self.tab5_widgets.get("load_input")
        load_button = self.tab5_widgets.get("load_button")
        load_output = self.tab5_widgets.get("load_output")

        if embeddings_button:
            if self.workflow_mode == "inference-only":
                embeddings_button.disabled = True
                return
            embeddings_button.disabled = self.tab4_df is None or not self.tab4_confirmed

        if self.tab4_confirmed:
            load_title.layout.display = "none"
            load_input.layout.display = "none"
            load_button.layout.display = "none"
            load_output.layout.display = "block"
        else:
            load_title.layout.display = "block"
            load_input.layout.display = "block"
            load_button.layout.display = "block"
            load_output.layout.display = "block"

        if status_message is not None:
            df = self.tab4_df
            if df is None:
                status_message.value = "<i>Please prepare data in Tab 4 first or provide the full path to a prepared dataset below to skip steps 1-4.</i>"
            elif not self.tab4_confirmed:
                status_message.value = (
                    "<i>Please confirm training classes in Tab 4 first.</i>"
                )
            else:
                summary = self._format_extractions_summary(df)
                if summary is None:
                    status_message.value = "<i>No training data loaded yet.</i>"
                else:
                    status_message.value = (
                        "<i>In this step we compute embeddings for the prepared training data.</i><br><br>"
                        "<b>Summary of available data:</b><br>"
                        f"{summary}"
                    )

    def _update_tab6_state(self):
        """Enable/disable Tab 6 (train) depending on prepared samples."""
        train_button = self.tab6_widgets.get("train_button")
        status_message = self.tab6_widgets.get("status_message")
        load_title = self.tab6_widgets.get("load_title")
        load_input = self.tab6_widgets.get("load_input")
        load_button = self.tab6_widgets.get("load_button")
        load_output = self.tab6_widgets.get("load_output")

        if train_button and status_message:
            if self.workflow_mode == "inference-only":
                train_button.disabled = True
                status_message.value = "<i>Skipped (inference-only mode).</i>"
                return
            is_ready = self.tab5_df is not None
            train_button.disabled = not is_ready
            if is_ready:
                summary = self._format_extractions_summary(self.tab5_df)
                if summary is None:
                    status_message.value = (
                        "<i>Something went wrong with the embeddings data.</i>"
                    )
                else:
                    status_message.value = (
                        "<i>Ready to train.</i><br><br>"
                        "<b>Summary of available data:</b><br>"
                        f"{summary}"
                    )
            else:
                status_message.value = "<i>Please compute embeddings in Tab 5 or load embeddings below.</i>"

        if self.tab5_df is not None:
            load_title.layout.display = "none"
            load_input.layout.display = "none"
            load_button.layout.display = "none"
            load_output.layout.display = "block"
        else:
            load_title.layout.display = "block"
            load_input.layout.display = "block"
            load_button.layout.display = "block"
            load_output.layout.display = "block"

    def _update_tab7_state(self):
        """Enable/disable Tab 7 (deploy) depending on training artifacts."""
        deploy_button = self.tab7_widgets.get("deploy_button")
        status_message = self.tab7_widgets.get("status_message")
        load_title = self.tab7_widgets.get("load_title")
        load_input = self.tab7_widgets.get("load_input")
        load_button = self.tab7_widgets.get("load_button")
        load_output = self.tab7_widgets.get("load_output")
        auth_output = self.tab7_widgets.get("auth_output")
        reset_auth_button = self.tab7_widgets.get("reset_auth_button")
        model_output = self.tab7_widgets.get("model_path_output")

        if self.workflow_mode == "inference-only":
            not_ready_msg = "<i>No model available. Load a torch head archive (.zip) using the button below to continue.</i>"
        else:
            not_ready_msg = "<i>No trained model available. Finish training a model in Tab 6 first or load a torch head archive (.zip) using the button below to continue.</i>"

        is_ready = (
            self.head_package_path is not None and self.head_package_path.exists()
        )
        deploy_button.disabled = not is_ready
        if is_ready:
            status_message.value = "<i>Ready to deploy model.</i>"
            with model_output:
                model_output.clear_output()
                print(f"Model archive: {self.head_package_path}")
            load_title.layout.display = "none"
            load_input.layout.display = "none"
            load_button.layout.display = "none"
            load_output.layout.display = "block"
        else:
            status_message.value = not_ready_msg
            with model_output:
                model_output.clear_output()
                print("No model archive loaded yet.")
            load_title.layout.display = "block"
            load_input.layout.display = "block"
            load_button.layout.display = "block"
            load_output.layout.display = "block"

        needs_auth = self._needs_cdse_authentication()
        with auth_output:
            auth_output.clear_output()
            if needs_auth:
                print(
                    "No CDSE refresh token found. You will be asked to authenticate upon pressing the 'Deploy Model' button. Make sure to click the link appearing below the application."
                )
            else:
                print(
                    "CDSE refresh token found on this machine. Click the reset button if you want to login with another account."
                )
        if reset_auth_button is not None:
            reset_auth_button.layout.display = "none" if needs_auth else "block"
        return

    def _update_tab8_state(self):
        """Enable/disable Tab 8 (generate map) depending on model availability."""
        generate_button = self.tab8_widgets.get("generate_button")
        status_message = self.tab8_widgets.get("status_message")
        season_hint = self.tab8_widgets.get("season_hint")
        if season_hint is not None:
            hint = None
            if self.season_window is not None:
                hint = (
                    "<i>Growing season for which your model was trained:</i> "
                    f"<b>{self.season_window.start_date}</b> → "
                    f"<b>{self.season_window.end_date}</b>"
                )
            else:
                hint = (
                    "<i>Growing season for which your model was trained:</i>"
                    "<b>No season found yet.</b>"
                )
            season_hint.value = hint
        if generate_button and status_message:
            if self.workflow_mode == "inference-only":
                is_ready = self.tab7_model_url is not None
                generate_button.disabled = not is_ready
                if is_ready:
                    status_message.value = (
                        "<i>Ready to generate map (inference-only mode).</i>"
                    )
                else:
                    status_message.value = (
                        "<i>Get a model URL by deploying your model in Tab 7 first.</i>"
                    )
                return
            is_ready = self.tab7_model_url is not None
            generate_button.disabled = not is_ready
            if is_ready:
                status_message.value = "<i>Ready to generate map.</i>"
            else:
                status_message.value = (
                    "<i>Deploy a model in Tab 7 before generating a map.</i>"
                )

    def _update_tab9_state(self):
        """Enable/disable Tab 9 (visualize) depending on map output availability."""
        visualize_button = self.tab9_widgets.get("visualize_button")
        status_message = self.tab9_widgets.get("status_message")
        merge_button = self.tab9_widgets.get("merge_button")
        results_title = self.tab9_widgets.get("results_title")
        results_input = self.tab9_widgets.get("results_input")
        results_button = self.tab9_widgets.get("results_button")
        results_output = self.tab9_widgets.get("results_output")
        model_url_input = self.tab9_widgets.get("model_url_input")

        if model_url_input is not None and not self.tab9_model_url:
            if self.tab7_model_url:
                model_url_input.value = self.tab7_model_url
                self.tab9_model_url = self.tab7_model_url

        if visualize_button and status_message:
            has_results = self.tab8_results is not None
            has_merged = len(self.tab9_merged_paths) > 0
            if merge_button is not None:
                merge_button.disabled = not has_results
            visualize_button.disabled = not has_merged
            if has_merged:
                status_message.value = "<i>Ready to visualize results.</i>"
            elif has_results:
                status_message.value = "<i>Ready to merge tiles.</i>"
            else:
                status_message.value = "<i>Please generate a map in Tab 8 or load a results folder below.</i>"

        if self.tab8_results is not None:
            results_title.layout.display = "none"
            results_input.layout.display = "none"
            results_button.layout.display = "none"
            results_output.layout.display = "block"
        else:
            results_title.layout.display = "block"
            results_input.layout.display = "block"
            results_button.layout.display = "block"
            results_output.layout.display = "block"

    def _needs_cdse_authentication(self) -> bool:
        if self.cdse_auth_cleared:
            return True

        system = platform.system().lower()
        if "windows" in system:
            token_path = (
                Path.home() / "AppData/Roaming/openeo-python-client/refresh-tokens.json"
            )
        else:
            token_path = (
                Path.home() / ".local/share/openeo-python-client/refresh-tokens.json"
            )

        return not token_path.exists()

    def _on_reset_cdse_auth_click(self, _=None) -> None:
        output = self.tab7_widgets.get("auth_output")
        with output:
            output.clear_output()
            try:
                from notebook_utils.openeo import clear_openeo_token_cache

                clear_openeo_token_cache()
                self.cdse_auth_cleared = True
                print(
                    "<i>CDSE authentication cache cleared. You will be asked to authenticate upon model deployment.</i>"
                )
            except Exception as exc:
                print(f"<i>Failed to clear CDSE authentication cache: {exc}</i>")

        self._update_tab7_state()

    def _is_valid_url(self, value: str) -> bool:
        try:
            parsed = urlparse(value)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _format_season_window_label(
        self, season_window: Optional[TemporalContext]
    ) -> str:
        if season_window is None:
            return "season_unknown"
        start = str(season_window.start_date).replace("-", "")
        end = str(season_window.end_date).replace("-", "")
        return f"{start}_{end}"

    def _info_callout(self, message: str) -> widgets.Widget:
        """Create a collapsible info callout box for inline documentation."""
        info_html = widgets.HTML(
            value=(
                "<div style='"
                "box-sizing:border-box;"
                "width:100%;"
                "max-width:100%;"
                "background:#f6f7f9;"
                "border:1px solid #e3e6ea;"
                "border-left:4px solid #c7cdd4;"
                "padding:6px 8px;"
                "border-radius:4px;"
                "color:#4b5563;"
                "font-size:13px;"
                "line-height:1.4;"
                "word-wrap:break-word;"
                "overflow-wrap:anywhere;"
                "margin:6px 0 8px 0;"
                "'>"
                f"{message}"
                "</div>"
            )
        )
        info_html.layout.display = "none"
        info_html.layout.width = "100%"

        toggle = widgets.ToggleButton(
            value=False,
            description="Learn more",
            icon="info-circle",
            layout=widgets.Layout(width="140px"),
        )

        def _on_toggle(change):
            info_html.layout.display = "block" if change["new"] else "none"

        toggle.observe(_on_toggle, names="value")

        return widgets.VBox([toggle, info_html], layout=widgets.Layout(width="100%"))
