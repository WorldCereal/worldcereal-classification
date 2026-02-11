"""
Interactive application for WorldCereal private extractions workflow.

This module provides an interactive widget-based interface for:
1. Retrieving reference data from RDM or local files
2. Selecting and preparing samples
3. Running EO data extractions
4. Visualizing and inspecting results
"""

import platform
import shutil
import tempfile
import threading
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import ipywidgets as widgets
from IPython.display import HTML, clear_output, display
from loguru import logger
from notebook_utils.auth_utils import trigger_cdse_authentication
from notebook_utils.extractions import (
    get_band_statistics,
    load_point_extractions,
    validate_required_attributes,
    visualize_timeseries,
)
from openeo.extra.job_management import CsvJobDatabase
from openeo_gfmap.manager.job_splitters import load_s2_grid
from tabulate import tabulate

from worldcereal.extract.common import (
    check_job_status,
    get_succeeded_job_details,
    run_extractions,
)
from worldcereal.openeo.preprocessing import WORLDCEREAL_BANDS
from worldcereal.rdm_api import RdmInteraction
from worldcereal.stac.constants import ExtractionCollection
from worldcereal.utils.legend import ewoc_code_to_label, get_legend
from worldcereal.utils.map import visualize_rdm_geoparquet
from worldcereal.utils.refdata import gdf_to_points
from worldcereal.utils.sampling import run_sampling


class WorldCerealExtractionsApp:
    """
    Interactive application for managing WorldCereal private data extractions.

    This class manages the entire workflow from data retrieval through extraction
    to visualization, maintaining state across different tabs.

    Attributes
    ----------
    parquet_file : Optional[Path]
        Path to the reference data parquet file
    collection_id : Optional[str]
        ID of the reference data collection
    raw_gdf : Optional[gpd.GeoDataFrame]
        Raw reference data loaded from parquet file in Tab 1
    samples_df : Optional[gpd.GeoDataFrame]
        Prepared samples dataframe ready for extraction (processed in Tab 2)
    extractions_folder : Path
        Root folder for storing extractions
    extractions_gdf : Optional[gpd.GeoDataFrame]
        Loaded extraction results
    rdm_instance : Optional[RdmInteraction]
        Authenticated RDM interaction instance
    """

    def __init__(self, extractions_folder: Path = Path("./extractions")):
        """
        Initialize the WorldCereal extraction application.

        Parameters
        ----------
        extractions_folder : Path, optional
            Root folder for storing extractions, by default Path("./extractions")
        """
        # State variables
        self.parquet_file: Optional[Path] = None
        self.collection_id: Optional[str] = None
        self.raw_gdf: Optional[gpd.GeoDataFrame] = None  # Raw data loaded in Tab 1
        self.samples_df: Optional[gpd.GeoDataFrame] = (
            None  # Prepared samples from Tab 2
        )
        self.extractions_folder = Path(extractions_folder)
        self.extractions_gdf: Optional[gpd.GeoDataFrame] = None
        self.rdm_instance = None
        self.append_existing = False
        self.append_samples_df: Optional[gpd.GeoDataFrame] = None
        self.cdse_auth_cleared = False
        self.cdse_auth_in_progress = False

        # UI components (initialized in build methods)
        self.tabs: Optional[widgets.Tab] = None
        self._nav_buttons: List[Dict[str, widgets.Button]] = []

        # Tab-specific widgets (will be populated by build methods)
        self.tab1_widgets: Dict[str, Any] = {}
        self.tab2_widgets: Dict[str, Any] = {}
        self.tab3_widgets: Dict[str, Any] = {}
        self.tab4_widgets: Dict[str, Any] = {}

    def display(self):
        """Display the interactive application."""
        self._build_ui()
        display(self.tabs)

    @classmethod
    def run(cls, extractions_folder: Path = Path("./extractions")):
        """Create and display the app with a single call."""
        app = cls(extractions_folder=extractions_folder)
        app.display()

    # =========================================================================
    # Main UI Building
    # =========================================================================

    def _build_ui(self):
        """Build the complete tabbed interface."""
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
        # Create tabs
        tab1 = self._build_tab1_retrieve_data()
        tab2 = self._build_tab2_select_samples()
        tab3 = self._build_tab3_run_extractions()
        tab4 = self._build_tab4_visualize_results()

        # Create tab container
        self.tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4])
        self.tabs.set_title(0, "1. Retrieve Data")
        self.tabs.set_title(1, "2. Select Samples")
        self.tabs.set_title(2, "3. Run Extractions")
        self.tabs.set_title(3, "4. Visualize Results")

        # Observe tab changes to update UI state
        self.tabs.observe(self._on_tab_change, names="selected_index")
        self._update_nav_buttons()

    def _build_tab_navigation(self) -> widgets.VBox:
        """Create navigation controls for switching tabs."""
        note = widgets.HTML(
            value="<i>Use the buttons below to go to the previous or next step.</i>"
        )
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
            [note, widgets.HBox([prev_button, next_button])],
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
    # Tab 1: Retrieve Data
    # =========================================================================

    def _build_tab1_retrieve_data(self) -> widgets.VBox:
        """
        Build Tab 1: Retrieve reference data from RDM or local file.

        Returns
        -------
        widgets.VBox
            Container widget for Tab 1
        """
        # Header
        header = widgets.HTML(
            value="<h3>Retrieve Reference Data</h3>"
            "<p>Choose to query data from the WorldCereal RDM or load a local file.</p>"
        )

        # Main action buttons
        rdm_button = widgets.Button(
            description="Query RDM",
            button_style="primary",
            icon="cloud-download",
            layout=widgets.Layout(width="200px", height="50px"),
        )

        local_button = widgets.Button(
            description="Choose Local File",
            button_style="primary",
            icon="folder-open",
            layout=widgets.Layout(width="200px", height="50px"),
        )

        button_box = widgets.HBox(
            [rdm_button, local_button],
            layout=widgets.Layout(justify_content="center", margin="20px 0"),
        )

        # Dynamic content area (for RDM or local file workflows)
        dynamic_content = widgets.VBox([])

        # Status display
        status_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="100px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        # Store widgets for later reference
        self.tab1_widgets = {
            "rdm_button": rdm_button,
            "local_button": local_button,
            "dynamic_content": dynamic_content,
            "status_output": status_output,
        }

        # Connect callbacks
        rdm_button.on_click(self._on_rdm_button_click)
        local_button.on_click(self._on_local_button_click)

        return widgets.VBox(
            [
                header,
                button_box,
                dynamic_content,
                widgets.HTML("<b>Status:</b>"),
                status_output,
                self._build_tab_navigation(),
            ]
        )

    def _on_rdm_button_click(self, button):
        """Handle RDM button click - show RDM workflow."""
        status_output = self.tab1_widgets["status_output"]
        dynamic_content = self.tab1_widgets["dynamic_content"]

        # Clear any previous content
        dynamic_content.children = []

        with status_output:
            clear_output(wait=True)
            print("Please select collection type to query...")

        background_info = self._info_callout(
            "ℹ️  <strong>About RDM interaction:</strong><br>"
            "To learn more about how to interact with the WorldCereal RDM,<br>"
            ' consult our <a href="https://github.com/WorldCereal/worldcereal-classification/blob/main/notebooks/worldcereal_RDM_demo.ipynb" target="_blank">dedicated notebook on RDM interaction</a>.<br><br>'
            "🔐 <strong>Authentication:</strong><br>"
            "You will be asked to login with your CDSE credentials when accessing private collections."
        )

        # Build collection type selection UI
        collection_type_radio = widgets.RadioButtons(
            options=[
                "Private",
                "Public",
            ],
            value="Private",
            description="Collection Type:",
        )

        collection_type_button = widgets.Button(
            description="Continue",
            button_style="info",
            icon="arrow-right",
            layout=widgets.Layout(width="150px"),
        )

        # Store for later access
        self.tab1_widgets["collection_type_radio"] = collection_type_radio

        # Connect callback
        collection_type_button.on_click(self._on_collection_type_continue)

        # Show in dynamic content area
        dynamic_content.children = [
            widgets.VBox(
                [background_info, collection_type_radio, collection_type_button]
            )
        ]

    def _on_collection_type_continue(self, button):
        """Handle collection type selection and continue with authentication."""
        status_output = self.tab1_widgets["status_output"]
        dynamic_content = self.tab1_widgets["dynamic_content"]
        collection_type_radio = self.tab1_widgets["collection_type_radio"]

        # Hide collection type selection
        dynamic_content.children = []

        with status_output:
            clear_output(wait=True)

        try:
            # Initiate RDM session
            rdm = RdmInteraction()
            self.rdm_instance = rdm

            # Determine query parameters based on selection
            is_public = collection_type_radio.value == "Public"
            is_private = collection_type_radio.value == "Private"

            # Take care of authentication if needed
            if is_private:
                with status_output:
                    print("🔐 Authenticating with RDM...")
                    print(
                        "Click the link you see below to authenticate with your CDSE credentials ⬇️"
                    )

                def _show_verification(info):
                    with status_output:
                        if info.verification_uri_complete:
                            display(
                                HTML(
                                    '<p><a href="{url}" target="_blank" rel="noopener">'
                                    "Open the CDSE login page</a></p>".format(
                                        url=info.verification_uri_complete
                                    )
                                )
                            )
                        else:
                            display(
                                HTML(
                                    '<p>Open <a href="{url}" target="_blank" rel="noopener">'
                                    "this link</a> and enter code <b>{code}</b>.</p>".format(
                                        url=info.verification_uri,
                                        code=info.user_code,
                                    )
                                )
                            )

                def _show_progress(message):
                    with status_output:
                        print(message)

                rdm.authenticate(
                    display_callback=_show_verification,
                    progress_callback=_show_progress,
                )

                with status_output:
                    print("\n✓ Authentication successful!")

            with status_output:
                print(
                    f"📡 Querying {collection_type_radio.value.lower()} collections..."
                )

            # Query collections
            collections = rdm.get_collections(
                include_public=is_public, include_private=is_private
            )

            # Check results
            if len(collections) == 0:
                with status_output:
                    print(
                        f"❌ No {collection_type_radio.value.lower()} collections found."
                    )
                    return

            with status_output:
                print(
                    f"✓ Found {len(collections)} {collection_type_radio.value.lower()} collection(s)"
                )
                print("Please select a collection to download in the dropdown above.")

            # Build collection selection UI
            self._build_collection_selection_ui(collections)

        except Exception as e:
            with status_output:
                print(f"❌ RDM query failed: {str(e)}")
                print("\nFull traceback:")
                traceback.print_exc()

    def _build_collection_selection_ui(self, collections):
        """Build UI for selecting and downloading a collection."""
        dynamic_content = self.tab1_widgets["dynamic_content"]

        # Collection dropdown
        options = [
            (f"{col.id} - {col.title if col.title else 'No title'}", col.id)
            for col in sorted(collections, key=lambda col: col.id)
        ]
        collection_dropdown = widgets.Dropdown(
            description="Collection:",
            options=options,
            layout=widgets.Layout(width="600px"),
        )

        # Download button
        rdm_download_button = widgets.Button(
            description="Download",
            button_style="success",
            icon="download",
            layout=widgets.Layout(width="150px"),
        )

        # Store widgets
        self.tab1_widgets["collection_dropdown"] = collection_dropdown

        # Connect callback
        rdm_download_button.on_click(self._on_rdm_download)

        # Display in dynamic content area
        dynamic_content.children = [
            widgets.VBox([collection_dropdown, rdm_download_button])
        ]

    def _on_rdm_download(self, button):
        """Handle RDM download button click."""
        status_output = self.tab1_widgets["status_output"]
        collection_dropdown = self.tab1_widgets["collection_dropdown"]

        with status_output:
            clear_output(wait=True)

            if not collection_dropdown.value:
                print("❌ Please select a collection")
                return

            print(f"📥 Downloading full collection: {collection_dropdown.value}")

            try:
                if self.rdm_instance is None:
                    print("❌ Not authenticated. Please authenticate first.")
                    return

                dwnld_folder = Path("./download")
                dwnld_folder.mkdir(parents=True, exist_ok=True)

                parquet_file = self.rdm_instance.download_collection_geoparquet(
                    collection_dropdown.value,
                    str(dwnld_folder),
                    subset=False,
                )

                # Store results in app state
                self.parquet_file = Path(parquet_file)
                self.collection_id = collection_dropdown.value

                print(f"✓ Successfully downloaded to: {parquet_file}")
                print(f"✓ Collection ID: {collection_dropdown.value}")

                # Load data and validate attributes
                self.raw_gdf = gpd.read_parquet(self.parquet_file)
                self.raw_gdf["ref_id"] = self.collection_id

                # Save original extract column if it exists (for preserving RDM preselection)
                if "extract" in self.raw_gdf.columns:
                    self.raw_gdf["extract_original"] = self.raw_gdf["extract"].copy()

                # Validate required attributes
                try:
                    validate_required_attributes(self.raw_gdf)
                    print("✓ All required attributes present")
                except ValueError as e:
                    print(f"⚠️  Warning: {str(e)}")
                    print("\n You need to add missing attributes before extraction.")
                    print("\n Cannot continue!")
                    return

                n_samples = len(self.raw_gdf)
                print(f"✓ Total samples in dataset: {n_samples:,}")

                # Update Tab 2 state immediately so counts are ready on first open
                self._update_tab2_state()

                print(
                    "\n✅ Data ready! You can now proceed to Tab 2 to prepare samples."
                )

            except Exception as e:
                print(f"❌ Download failed: {str(e)}")
                traceback.print_exc()

                followup = self._info_callout(
                    "<strong>Download failed?:</strong><br>"
                    "In case the RDM API returns an http error upon downloading your dataset,<br>"
                    "you can proceed by browsing to your dataset in the RDM user interface and manually download the harmonized version of the dataset.<br>"
                    "Make sure to then upload the resulting .parquet file to the machine where you are running this app.<br>"
                    "You can re-run step 1 of the app, clicking the Choose Local File button and supplying the full path to your .parquet file."
                )
                display(followup)

    def _on_local_button_click(self, button):
        """Handle local file button click - show file selection UI."""
        status_output = self.tab1_widgets["status_output"]
        dynamic_content = self.tab1_widgets["dynamic_content"]

        # Clear any previous content
        dynamic_content.children = []

        with status_output:
            clear_output(wait=True)

        # Build local file selection UI
        explanation = self._info_callout(
            "<strong>Attention:</strong> We only accept <code>.parquet</code> files which have been"
            " harmonized through and downloaded <a href='https://rdm.esa-worldcereal.org/' target='_blank' rel='noopener'>WorldCereal RDM</a>.<br><br>"
            "<strong>Note:</strong> We use the name of your file as the ID of your dataset."
        )

        file_path_text = widgets.Text(
            description="File Path:",
            placeholder="Enter the FULL path to your .parquet file",
            layout=widgets.Layout(width="600px"),
        )

        file_load_button = widgets.Button(
            description="Load",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="150px"),
        )

        # Store widgets
        self.tab1_widgets["file_path_text"] = file_path_text

        # Connect callback
        file_load_button.on_click(self._on_file_load)

        # Display in dynamic content area
        dynamic_content.children = [
            widgets.VBox([explanation, file_path_text, file_load_button])
        ]

    def _on_file_load(self, button):
        """Handle local file load button click."""
        status_output = self.tab1_widgets["status_output"]
        file_path_text = self.tab1_widgets["file_path_text"]

        with status_output:
            clear_output(wait=True)

            file = Path(file_path_text.value)

            if not file.exists():
                print(f"❌ File not found: {file}")
                return

            if not file.suffix == ".parquet":
                print("❌ File must be a .parquet file")
                return

            try:
                # Load and validate parquet file
                self.raw_gdf = gpd.read_parquet(file)
                self.raw_gdf["ref_id"] = file.stem

                # Save original extract column if it exists (for preserving RDM preselection)
                if "extract" in self.raw_gdf.columns:
                    self.raw_gdf["extract_original"] = self.raw_gdf["extract"].copy()

                # Store results in app state
                self.parquet_file = file
                self.collection_id = file.stem

                print(f"✓ Successfully loaded: {file}")
                print(f"✓ Using filename as ID: {file.stem}")

                # Validate required attributes
                try:
                    validate_required_attributes(self.raw_gdf)
                    print("✓ All required attributes present")
                except ValueError as e:
                    print(f"⚠️  Warning: {str(e)}")
                    print("\n You need to add missing attributes before extraction.")
                    print("\n Cannot continue!")
                    return

                # Display sample count
                n_samples = len(self.raw_gdf)
                print(f"✓ Total samples in dataset: {n_samples:,}")

                # Update Tab 2 state immediately so counts are ready on first open
                self._update_tab2_state()

                print(
                    "\n✅ Data ready! You can now proceed to Tab 2 to prepare samples."
                )

            except Exception as e:
                print(f"❌ Failed to load file: {str(e)}")
                traceback.print_exc()

    # =========================================================================
    # Tab 2: Select Samples
    # =========================================================================

    def _build_tab2_select_samples(self) -> widgets.VBox:
        """
        Build Tab 2: Select and prepare samples for extraction.

        Returns
        -------
        widgets.VBox
            Container widget for Tab 2
        """
        # Header
        header = widgets.HTML(
            value="<h3>Select and Prepare Samples</h3>"
            "<p>Review your reference data and prepare samples for extraction.</p>"
        )

        # Status message
        status_message = widgets.HTML(
            value="<i>Please load reference data in Tab 1 first.</i>"
        )

        # Dataset info display
        info_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        # Sample selection options
        selection_header = widgets.HTML("<h4>Sample Selection</h4>")

        explanation_selection = self._info_callout(
            "A representative subset of samples is automatically available for each reference dataset uploaded to the RDM.<br>"
            "Learn more about this subsampling routine, <a href='https://worldcereal.github.io/worldcereal-documentation/rdm/refdata.html#dataset-subsampling' target='_blank' rel='noopener'>HERE</a>.<br><br>"
            "<b>You now have the choice to:</b><br>"
            "   - make use of this subset (default),<br>"
            "   - use all samples, or<br>"
            "   - select only your classes of interest and/or run your own sampling routine."
        )

        sample_selection_radio = widgets.RadioButtons(
            options=[],
            description="Make your choice:",
            disabled=True,
            layout=widgets.Layout(width="auto"),
        )

        # Prepare button
        prepare_button = widgets.Button(
            description="Prepare Samples",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="200px", height="40px"),
            disabled=True,
        )

        # Results output
        results_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        # Store widgets
        self.tab2_widgets = {
            "status_message": status_message,
            "info_output": info_output,
            "sample_selection_radio": sample_selection_radio,
            "prepare_button": prepare_button,
            "results_output": results_output,
        }

        # Connect callbacks
        prepare_button.on_click(self._on_prepare_samples_click)

        return widgets.VBox(
            [
                header,
                status_message,
                widgets.HTML("<b>Dataset Information:</b>"),
                info_output,
                selection_header,
                explanation_selection,
                sample_selection_radio,
                widgets.HBox(
                    [prepare_button],
                    layout=widgets.Layout(justify_content="center", margin="20px 0"),
                ),
                widgets.HTML("<b>Preparation Results:</b>"),
                results_output,
                self._build_tab_navigation(),
            ]
        )

    def _on_prepare_samples_click(self, button):
        """Handle prepare samples button click."""
        sample_selection_radio = self.tab2_widgets["sample_selection_radio"]

        # Get selection mode
        selection_mode = sample_selection_radio.value

        # If custom sampling is selected, show parameters UI first
        if selection_mode == "custom":
            self._show_custom_sampling_ui()
            return

        # Otherwise, proceed with preparation
        self._run_sample_preparation(selection_mode)

    def _show_custom_sampling_ui(self):
        """Show custom sampling parameters UI."""
        results_output = self.tab2_widgets["results_output"]

        with results_output:
            clear_output(wait=True)
            print("=" * 60)
            print("CUSTOM SAMPLING PARAMETERS")
            print("=" * 60)
            print("\nℹ️  Custom sampling uses a stratified approach that:")
            print("   • Balances samples across crop classes")
            print("   • Ensures minimum distance between samples")
            print("   • Distributes samples across H3 spatial cells")
            print("   • Prioritizes rarest classes first")
            print("\n" + "-" * 60)

        # Get unique crop types from the data, ordered by frequency
        crop_type_counts = self.raw_gdf["ewoc_code"].value_counts()
        unique_crop_types = crop_type_counts.index.tolist()
        crop_type_labels = ewoc_code_to_label(unique_crop_types, label_type="full")

        # Create options as (label, code) tuples - label only for display
        crop_type_options = [
            (label, code) for code, label in zip(unique_crop_types, crop_type_labels)
        ]

        # Create crop type selection widget
        crop_type_selector = widgets.SelectMultiple(
            options=crop_type_options,
            value=unique_crop_types,  # Select all by default
            description="Crop types:",
            tooltip="Select crop types to include in sampling (Ctrl/Cmd+Click for multi-select)",
            style={"description_width": "150px"},
            layout=widgets.Layout(width="600px", height="150px"),
        )

        # Create parameter input widgets
        max_samples_per_cell = widgets.IntText(
            value=50,
            description="Max samples per class:",
            tooltip=("Maximum allowed samples per H3 L3 cell and crop class"),
            style={"description_width": "200px"},
        )

        sampling_distance = widgets.IntText(
            value=500,
            description="Sampling distance (m):",
            tooltip="Minimum distance between samples in meters",
            style={"description_width": "150px"},
        )

        # Store widgets for later access
        self.tab2_widgets["crop_type_selector"] = crop_type_selector
        self.tab2_widgets["max_samples_per_cell"] = max_samples_per_cell
        self.tab2_widgets["sampling_distance"] = sampling_distance

        # Create run button
        run_sampling_button = widgets.Button(
            description="Run Sampling",
            button_style="success",
            icon="play",
            layout=widgets.Layout(width="200px", height="40px"),
        )

        run_sampling_button.on_click(self._on_run_custom_sampling)

        # Display the UI
        with results_output:
            print("\nCrop Type Selection:")
            print(
                "(All crop types selected by default. Use Ctrl/Cmd+Click to select/deselect)"
            )
            display(crop_type_selector)
            print("\nSampling Parameters:")
            display(
                widgets.VBox(
                    [
                        max_samples_per_cell,
                        sampling_distance,
                        widgets.HBox(
                            [run_sampling_button],
                            layout=widgets.Layout(margin="20px 0"),
                        ),
                    ]
                )
            )

    def _on_run_custom_sampling(self, button):
        """Handle custom sampling execution."""
        results_output = self.tab2_widgets["results_output"]

        # Get parameters
        selected_crop_types = list(self.tab2_widgets["crop_type_selector"].value)
        max_samples_per_cell = self.tab2_widgets["max_samples_per_cell"].value
        max_samples_lc = max_samples_per_cell
        max_samples_ct = max_samples_per_cell
        sampling_distance = self.tab2_widgets["sampling_distance"].value

        with results_output:
            clear_output(wait=True)
            print("🔄 Running custom sampling...")
            print(f"   Selected crop types: {len(selected_crop_types)}")
            print(f"   Max samples per class: {max_samples_per_cell}")
            print(f"   Sampling distance: {sampling_distance}m")

            try:
                # Get legend
                legend = get_legend()

                # Prepare data
                print("\n🔄 Preparing data for sampling...")
                gdf = self.raw_gdf.copy()

                # Filter by selected crop types
                if len(selected_crop_types) == 0:
                    print("\n❌ No crop types selected!")
                    print("   Please select at least one crop type.")
                    return

                initial_count = len(gdf)
                gdf = gdf[gdf["ewoc_code"].isin(selected_crop_types)]
                filtered_count = len(gdf)
                print(
                    f"✓ Filtered to {filtered_count:,} / {initial_count:,} samples based on crop type selection"
                )

                if filtered_count == 0:
                    print("\n❌ No samples remaining after crop type filtering!")
                    return

                # Check for required attributes
                required_attrs = [
                    "sample_id",
                    "ewoc_code",
                    "h3_l3_cell",
                    "quality_score_ct",
                    "quality_score_lc",
                ]
                missing_attrs = [
                    attr for attr in required_attrs if attr not in gdf.columns
                ]
                if missing_attrs:
                    print(f"\n❌ Missing required attributes: {missing_attrs}")
                    print("   Cannot perform custom sampling.")
                    return

                # Run sampling
                print("\n🔄 Running stratified sampling algorithm...")
                gdf = run_sampling(
                    gdf=gdf,
                    legend=legend,
                    max_samples_lc=max_samples_lc,
                    max_samples_ct=max_samples_ct,
                    sampling_distance=sampling_distance,
                )

                # Preserve original pre-selection the first time custom sampling runs
                if "extract_original" not in self.raw_gdf.columns:
                    if "extract" in self.raw_gdf.columns:
                        self.raw_gdf["extract_original"] = self.raw_gdf[
                            "extract"
                        ].copy()
                    else:
                        self.raw_gdf["extract_original"] = 0

                # Update raw_gdf: set extract=0 for all, then merge sampled results
                self.raw_gdf["extract"] = 0
                # Update extract column for sampled data
                self.raw_gdf.loc[
                    self.raw_gdf["sample_id"].isin(
                        gdf[gdf["extract"] > 0]["sample_id"]
                    ),
                    "extract",
                ] = 1

                # Get statistics per crop type
                sampled_gdf = self.raw_gdf[self.raw_gdf["extract"] > 0]
                selected_count = len(sampled_gdf)

                print("\n✓ Sampling complete!")
                print(
                    f"   Selected {selected_count:,} / {initial_count:,} samples ({selected_count/initial_count*100:.1f}%)"
                )

                # Display samples per crop type
                print("\n" + "=" * 60)
                print("SAMPLES PER CROP TYPE")
                print("=" * 60)

                crop_type_counts = sampled_gdf.groupby("ewoc_code").size()
                crop_type_labels = ewoc_code_to_label(
                    crop_type_counts.index.tolist(), label_type="full"
                )

                for code, label, count in zip(
                    crop_type_counts.index, crop_type_labels, crop_type_counts.values
                ):
                    print(f"   {label}: {count:,} samples")

                print("=" * 60)

                # Show confirmation buttons
                print("\n📊 Review the sampling results above.")
                print(
                    "   Would you like to continue with this selection or adjust the parameters and try again?\n"
                )

                # Create confirmation buttons
                continue_button = widgets.Button(
                    description="Continue with Selection",
                    button_style="success",
                    icon="check",
                    layout=widgets.Layout(width="250px", height="40px"),
                )

                adjust_button = widgets.Button(
                    description="Adjust Parameters",
                    button_style="warning",
                    icon="edit",
                    layout=widgets.Layout(width="250px", height="40px"),
                )

                # Create a container for the buttons
                button_container = widgets.HBox(
                    [continue_button, adjust_button],
                    layout=widgets.Layout(margin="10px 0"),
                )

                # Define button handlers
                def on_continue_click(btn):
                    # Clear the buttons and proceed with preparation
                    button_container.close()
                    self._run_sample_preparation("custom")

                def on_adjust_click(btn):
                    # Clear the results and show the sampling UI again
                    button_container.close()
                    # Restore the original extract column to preserve RDM preselection
                    if "extract_original" in self.raw_gdf.columns:
                        self.raw_gdf["extract"] = self.raw_gdf[
                            "extract_original"
                        ].copy()
                    elif "extract" in self.raw_gdf.columns:
                        # If no original was saved, reset to 0
                        self.raw_gdf["extract"] = 0
                    with results_output:
                        print(
                            "\n🔄 Resetting selection and adjusting sampling parameters..."
                        )
                    self._show_custom_sampling_ui()

                continue_button.on_click(on_continue_click)
                adjust_button.on_click(on_adjust_click)

                # Display the buttons
                display(button_container)

            except Exception as e:
                print(f"\n❌ Error during custom sampling: {str(e)}")
                traceback.print_exc()
                return

    def _run_sample_preparation(self, selection_mode):
        """Run sample preparation based on selection mode."""
        results_output = self.tab2_widgets["results_output"]

        with results_output:
            clear_output(wait=True)
            print("🔄 Preparing samples for extraction...")

            try:
                # Start with raw data
                gdf = self.raw_gdf.copy()

                # Apply selection filter
                if selection_mode == "preselected":
                    initial_count = len(gdf)
                    if "extract_original" in gdf.columns:
                        gdf = gdf[gdf["extract_original"] > 0]
                    else:
                        gdf = gdf[gdf["extract"] > 0]
                    print(
                        f"✓ Filtered to pre-selected samples: {len(gdf):,} / {initial_count:,}"
                    )
                elif selection_mode == "custom":
                    initial_count = len(gdf)
                    gdf = gdf[gdf["extract"] > 0]
                    print(
                        f"✓ Filtered to custom-sampled samples: {len(gdf):,} / {initial_count:,}"
                    )
                else:  # 'all'
                    print(f"✓ Using all samples: {len(gdf):,}")
                    # Mark all for extraction in raw_gdf as well
                    self.raw_gdf["extract"] = 1
                    gdf["extract"] = 1

                if len(gdf) == 0:
                    print("\n❌ No samples remaining after filtering!")
                    return

                # Convert geometries to points
                print("\n🔄 Converting geometries to points...")
                gdf = gdf_to_points(gdf)

                if len(gdf) == 0:
                    print("❌ No valid samples left after point conversion.")
                    return

                print(f"✓ Valid point geometries: {len(gdf):,}")

                # Warn if many samples
                if len(gdf) > 1000:
                    print("\n⚠️  Warning: More than 1,000 samples in your dataset.")
                    print("   Extractions will likely consume considerable credits.")

                # Check spatial coverage (S2 tiles)
                print("\n🔄 Checking spatial coverage...")
                gdf = gdf.to_crs(epsg=4326)
                s2_grid = load_s2_grid()
                gdf = gpd.sjoin(
                    gdf,
                    s2_grid[["tile", "geometry"]],
                    predicate="intersects",
                ).drop(columns=["index_right"])

                n_tiles = len(gdf["tile"].unique())
                print(f"✓ Samples cover {n_tiles} Sentinel-2 tiles")

                if n_tiles > 50:
                    print("\n⚠️  Warning: High number of S2 tiles.")
                    print("   Extractions will take considerable time.")

                # Drop tile column (not needed anymore)
                gdf = gdf.drop(columns=["tile"])

                # Store prepared samples
                self.samples_df = gdf

            except Exception as e:
                print(f"\n❌ Error during sample preparation: {str(e)}")
                traceback.print_exc()
                return

        # Display updated dataset information after successful preparation
        if self.samples_df is not None:
            self._display_dataset_info(
                gdf=self.samples_df,
                title="PREPARED SAMPLES INFORMATION",
                output_widget=results_output,
            )

            with results_output:
                print("\n🗺️  Generating map visualization...")

            # Visualize ALL samples on a map with distinction between selected/non-selected
            try:
                # Save all samples to temporary file for visualization
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".parquet", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                    self.raw_gdf.to_parquet(tmp_path)

                # Get list of selected sample_ids
                selected_ids = self.samples_df["sample_id"].tolist()

                # Create map visualization with selected/non-selected distinction
                map_widget = visualize_rdm_geoparquet(
                    tmp_path, selected_sample_ids=selected_ids
                )

                with results_output:
                    print("✓ Map generated successfully")
                    print("\n📍 Samples Map:")
                    print(
                        f"   🟢 Green (large) = Selected samples ({len(self.samples_df):,})"
                    )
                    print(
                        f"   🔴 Red (small) = Non-selected samples ({len(self.raw_gdf) - len(self.samples_df):,})"
                    )
                    display(map_widget)

                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

            except Exception as e:
                with results_output:
                    print(f"⚠️  Could not generate map: {str(e)}")
                    logger.warning(f"Map visualization failed: {e}")

            with results_output:
                print("\n✅ You can now proceed to Tab 3 to run extractions.")

    def _display_dataset_info(
        self,
        gdf: Optional[gpd.GeoDataFrame] = None,
        title: Optional[str] = None,
        output_widget: Optional[widgets.Output] = None,
    ):
        """Display information about a dataset.

        Parameters
        ----------
        gdf : Optional[gpd.GeoDataFrame], optional
            GeoDataFrame to display info for. If None, uses self.raw_gdf.
        title : Optional[str], optional
            Title for the information display, by default None
        output_widget : Optional[widgets.Output], optional
            Output widget to display to. If None, uses self.tab2_widgets["info_output"].
        """
        if output_widget is None:
            output_widget = self.tab2_widgets["info_output"]

        with output_widget:
            # Only clear if using the default info_output (not results_output)
            if output_widget == self.tab2_widgets.get("info_output"):
                clear_output(wait=True)

            if gdf is None:
                gdf = self.raw_gdf

            if gdf is None:
                print("No data loaded.")
                return

            if title is not None:
                print("\n" + "=" * 60)
                print(title)
                print("=" * 60)

            # Add labels if not present
            if "label_full" not in gdf.columns:
                gdf["label_full"] = ewoc_code_to_label(
                    gdf["ewoc_code"], label_type="full"
                )
            if "sampling_label" not in gdf.columns:
                gdf["sampling_label"] = ewoc_code_to_label(
                    gdf["ewoc_code"], label_type="sampling"
                )

            # Get unique samples
            unique_samples = gdf.drop_duplicates(subset=["sample_id"])

            # Basic statistics
            total_samples = len(unique_samples)
            total_crop_types = unique_samples["ewoc_code"].nunique()
            unique_crop_groups = sorted(unique_samples["sampling_label"].unique())

            stats_table = [
                ["Total Samples", f"{total_samples:,}"],
                ["Unique Crop Types", f"{total_crop_types}"],
            ]

            print("\nOverall Statistics:")
            print(tabulate(stats_table, headers=["Metric", "Value"], tablefmt="grid"))

            # Crop groups
            print("\nCrop Groups Present:")
            crop_groups_str = ", ".join(unique_crop_groups)
            # Wrap text for display
            import textwrap

            wrapped = "\n".join(textwrap.wrap(crop_groups_str, width=55))
            print(wrapped)

            # Detailed crop type breakdown
            print("\nDetailed Crop Type Distribution:")
            crop_counts = (
                unique_samples.groupby("label_full").size().reset_index(name="count")
            )
            crop_counts = crop_counts.sort_values("count", ascending=False)
            crop_counts["percentage"] = (
                crop_counts["count"] / total_samples * 100
            ).round(1)

            # Filter to show crop types with > 3% and format for display
            crop_counts_filtered = crop_counts[crop_counts["percentage"] > 3.0]

            display_table = []
            for _, row in crop_counts_filtered.iterrows():
                display_table.append(
                    [
                        row["label_full"][:50],  # Truncate long names
                        f"{row['count']:,}",
                        f"{row['percentage']:.1f}%",
                    ]
                )

            print(
                tabulate(
                    display_table,
                    headers=["Crop Type", "Samples", "%"],
                    tablefmt="grid",
                )
            )

            # Show count of crop types not displayed (< 3%)
            n_not_displayed = len(crop_counts) - len(crop_counts_filtered)
            if n_not_displayed > 0:
                print(f"\n... and {n_not_displayed} more crop types (each < 3%).")

            print("=" * 60 + "\n")

    # =========================================================================
    # Tab 3: Run Extractions
    # =========================================================================

    def _build_tab3_run_extractions(self) -> widgets.VBox:
        """
        Build Tab 3: Run EO data extractions.

        Returns
        -------
        widgets.VBox
            Container widget for Tab 3
        """
        # Header
        header = widgets.HTML(
            value="<h3>Run EO Data Extractions</h3>"
            "<p>Extract satellite time series for your reference samples.</p>"
        )

        general_info = self._info_callout(
            "The specific start and end date of the time series is automatically set to resp. 9 months prior and 9 months after `valid_time` for each sample.<br><br>"
            "For each sample, we extract the following monthly time series:<br><br>"
            "            - Sentinel-2 L2A data (all bands)<br>"
            "            - Sentinel-1 SIGMA0, VH and VV<br>"
            "            - Average air temperature and precipitation sum derived from AgERA5<br>"
            "And additionally, slope and elevation from Copernicus DEM<br><br>"
            "Note that pre-processing of the time series (e.g. cloud masking, temporal compositing) happens on the fly during the extractions.<br><br>"
            "<strong>Note:</strong> As you may have noticed during sample selection, we automatically convert all geometries to points for extraction purposes.<br>"
            "This means that for polygon samples, we only extract the EO time series at the centroid point of the polygon.<br>"
            "More extensive within-polygon sampling is not currently supported in this application.<br>"
        )

        # Status message
        status_message = widgets.HTML(
            value="<i>Please prepare samples in Tab 2 first.</i>"
        )

        # Warning/info display for existing extractions
        warning_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        # CDSE authentication section
        auth_header = widgets.HTML("<h4>CDSE Authentication</h4>")
        auth_status = widgets.HTML(
            value="<i>Checking CDSE authentication status...</i>"
        )
        authentication_explanation = self._info_callout(
            "To run extractions, you need to authenticate with a valid <a href='https://dataspace.copernicus.eu/' target='_blank' rel='noopener'>CDSE</a> account.<br>"
            "Your account credentials are automatically stored on your computer in a secure manner and used for extraction requests.<br>"
            "For your first time authentication, click the 'Authenticate' button below.<br>"
            "If you wish to switch accounts, use the 'Reset CDSE Authentication' button."
        )

        authenticate_button = widgets.Button(
            description="Authenticate",
            button_style="success",
            icon="key",
            layout=widgets.Layout(width="150px", height="40px"),
        )
        reset_auth_button = widgets.Button(
            description="Reset CDSE Authentication",
            button_style="warning",
            icon="refresh",
            layout=widgets.Layout(width="220px", height="40px"),
        )
        auth_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        # Extraction settings
        settings_header = widgets.HTML("<h4>Extraction Settings</h4>")

        # Restart failed jobs - with explanation
        restart_failed_explanation = widgets.HTML(
            value="<p style='margin: 5px 0 5px 0; color: #555;'>"
            "📌 <b>Restart Failed Jobs:</b> If previous extractions exist with failed jobs, "
            "enabling this will retry those failed jobs.</p>"
        )
        restart_failed_checkbox = widgets.Checkbox(
            value=False,
            description="Restart failed jobs",
            style={"description_width": "auto"},
            layout=widgets.Layout(width="auto"),
        )

        # Run button
        run_button = widgets.Button(
            description="Start Extractions",
            button_style="success",
            icon="play",
            layout=widgets.Layout(width="150px", height="50px"),
            disabled=True,
        )

        background_info = self._info_callout(
            "ℹ️ <strong>Background information:</strong><br>"
            "Samples to be extracted are now automatically split into one or several OpenEO extraction jobs.<br>"
            "The OpenEO job manager automatically handles the execution of these jobs on the CDSE platform.<br><br>"
            "<b>Want to stop extractions?</b><br>"
            "Kill this app by restarting the kernel.<br>"
            "Important: visit the <a href='https://openeo.dataspace.copernicus.eu/' target='_blank' rel='noopener'>OpenEO web editor</a> to check for any running jobs and stop them if needed to avoid unnecessary credit consumption.<br>"
            "Make sure to enable the 'Restart Failed Jobs' option if you want to continue with existing extractions later on.<br><br>"
            "Execution of an OpenEO job will consume credits from your CDSE account.<br>"
            "Average credit consumption of one job amounts to 30 credits, but can vary up to 300 credits depending on local data density.<br><br"
            "<b>Extractions done?</b><br>"
            "Upon finalization of all extraction jobs, the extracted data will be saved automatically to your output directory.<br>"
            "The output directory is called <code>extractions_output</code> and is created in the same folder where this notebook is located.<br>"
            "This directory contains one subfolder per collection ID, holding the extracted data, and the status of the jobs in the job tracking csv.<br>"
            "All your private extractions will be automatically grouped into one partitioned <b>worldcereal_merged_extractions.geoparquet</b> file, directly located in your extractions folder.<br>"
        )

        # Progress display
        progress_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="200px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        status_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="140px",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        # Store widgets
        self.tab3_widgets = {
            "status_message": status_message,
            "warning_output": warning_output,
            "auth_status": auth_status,
            "authenticate_button": authenticate_button,
            "reset_auth_button": reset_auth_button,
            "auth_output": auth_output,
            "restart_failed_checkbox": restart_failed_checkbox,
            "run_button": run_button,
            "progress_output": progress_output,
            "status_output": status_output,
        }

        # Connect callbacks
        run_button.on_click(self._on_run_extractions_click)
        reset_auth_button.on_click(self._on_reset_auth_click)
        authenticate_button.on_click(self._on_authenticate_cdse_click)

        # CSS for larger checkbox text
        checkbox_style = widgets.HTML(
            """
            <style>
                .widget-checkbox label {
                    font-size: 14px !important;
                    font-weight: 500 !important;
                }
            </style>
            """
        )

        return widgets.VBox(
            [
                checkbox_style,
                header,
                general_info,
                status_message,
                warning_output,
                # auth_header,
                # authentication_explanation,
                # auth_status,
                # widgets.HBox(
                #     [authenticate_button, reset_auth_button],
                #     layout=widgets.Layout(
                #         justify_content="flex-start", margin="10px 0"
                #     ),
                # ),
                # auth_output,
                settings_header,
                restart_failed_explanation,
                restart_failed_checkbox,
                widgets.HBox(
                    [run_button],
                    layout=widgets.Layout(justify_content="center", margin="20px 0"),
                ),
                background_info,
                widgets.HTML("<b>Progress:</b>"),
                progress_output,
                widgets.HTML("<b>Extraction Status Summary:</b>"),
                status_output,
                self._build_tab_navigation(),
            ]
        )

    def _delete_existing_extractions(self):
        """Delete existing extractions for the current collection."""
        warning_output = self.tab3_widgets["warning_output"]

        with warning_output:
            clear_output()
            self.append_existing = False
            self.append_samples_df = None
            print("🗑️  Deleting existing extractions...\n")

            outfolder_col = self.extractions_folder / self.collection_id

            if outfolder_col.exists():
                # Delete the entire collection folder
                shutil.rmtree(outfolder_col)
                print(f"   ✓ Deleted collection folder: {outfolder_col}")

                # Also clean up partitions in merged parquet if it exists
                merged_parquet = (
                    self.extractions_folder / "worldcereal_merged_extractions.parquet"
                )
                if merged_parquet.exists():
                    partitions = [
                        d
                        for d in merged_parquet.iterdir()
                        if d.is_dir()
                        and d.name.startswith(f"ref_id={self.collection_id}")
                    ]

                    for partition in partitions:
                        shutil.rmtree(partition)
                        print(f"   ✓ Deleted partition: {partition.name}")

                    # Check if any partitions remain
                    remaining_partitions = [
                        d
                        for d in merged_parquet.iterdir()
                        if d.is_dir() and d.name.startswith("ref_id=")
                    ]

                    if len(remaining_partitions) == 0:
                        # No partitions left, delete entire merged file
                        shutil.rmtree(merged_parquet)
                        print("   ✓ No partitions remaining, deleted merged file")
                    else:
                        print(
                            f"   ℹ️  {len(remaining_partitions)} partition(s) remaining"
                        )
                else:
                    print("   ℹ️  No merged parquet file found")

                # Recreate the collection folder for new extraction
                outfolder_col.mkdir(parents=True, exist_ok=True)
                print("\n✅ Deletion complete. Ready for fresh extraction.")
            else:
                print("   ℹ️  Collection folder doesn't exist.")

    def _get_existing_extraction_sample_ids(self, outfolder_col: Path) -> set:
        """Collect sample_id values already present in existing extractions."""
        existing_files = list(outfolder_col.glob("**/*.geoparquet"))
        existing_files = [f for f in existing_files if not f.is_dir()]
        existing_ids: set = set()

        for infile in existing_files:
            try:
                gdf = gpd.read_parquet(infile, columns=["sample_id"])
                existing_ids.update(gdf["sample_id"].dropna().unique().tolist())
            except Exception:
                try:
                    gdf = gpd.read_parquet(infile, columns=["sampleID"])
                    existing_ids.update(gdf["sampleID"].dropna().unique().tolist())
                except Exception:
                    try:
                        gdf = gpd.read_parquet(infile)
                        if "sample_id" in gdf.columns:
                            existing_ids.update(
                                gdf["sample_id"].dropna().unique().tolist()
                            )
                        elif "sampleID" in gdf.columns:
                            existing_ids.update(
                                gdf["sampleID"].dropna().unique().tolist()
                            )
                        else:
                            logger.warning(
                                "Could not find sample_id column in %s. Skipping.",
                                infile,
                            )
                    except Exception:
                        logger.warning(
                            "Could not read sample_id column from %s. Skipping.",
                            infile,
                        )

        return existing_ids

    def _prepare_append_samples(
        self, outfolder_col: Path
    ) -> Optional[gpd.GeoDataFrame]:
        """Filter samples_df to only those not yet extracted."""
        if self.samples_df is None:
            return None

        existing_ids = self._get_existing_extraction_sample_ids(outfolder_col)
        if len(existing_ids) == 0:
            return self.samples_df.copy()

        filtered = self.samples_df[~self.samples_df["sample_id"].isin(existing_ids)]
        return filtered

    def _on_run_extractions_click(self, button):
        """Handle run extractions button click."""
        progress_output = self.tab3_widgets["progress_output"]
        status_output = self.tab3_widgets["status_output"]
        restart_failed_checkbox = self.tab3_widgets["restart_failed_checkbox"]

        with progress_output:
            clear_output(wait=True)
            print("🔄 Starting EO data extractions...")
            print("=" * 60)

            try:
                # Get parameters (output folder is fixed to self.extractions_folder)
                output_folder = self.extractions_folder
                restart_failed = restart_failed_checkbox.value

                # Validate samples_df exists
                if self.samples_df is None:
                    print("❌ No prepared samples found!")
                    print("   Please prepare samples in Tab 2 first.")
                    return

                # Setup output folder
                print(f"\n📁 Output folder: {output_folder}")
                outfolder_col = output_folder / self.collection_id
                outfolder_col.mkdir(parents=True, exist_ok=True)
                print(f"✓ Collection folder: {outfolder_col}")

                # Save samples dataframe to file
                print("\n💾 Saving samples to file...")
                samples_df = (
                    self.append_samples_df if self.append_existing else self.samples_df
                )
                if samples_df is None or len(samples_df) == 0:
                    print("❌ No samples left to extract after filtering.")
                    return

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                samples_df_path = outfolder_col / "samples_gdf.gpkg"

                if self.append_existing:
                    # Backup existing samples and tracking with the same timestamp
                    samples_gdf_existing = outfolder_col / "samples_gdf.gpkg"
                    if samples_gdf_existing.exists():
                        samples_backup_path = (
                            outfolder_col
                            / f"samples_gdf_append_backup_{timestamp}.gpkg"
                        )
                        samples_gdf_existing.rename(samples_backup_path)
                        print(
                            f"ℹ️  Backed up existing samples GDF to: {samples_backup_path}"
                        )

                    tracking_path = outfolder_col / "job_tracking.csv"
                    if tracking_path.exists():
                        backup_path = (
                            outfolder_col
                            / f"job_tracking_append_backup_{timestamp}.csv"
                        )
                        tracking_path.rename(backup_path)
                        print(
                            f"ℹ️  Backed up existing job tracking file to: {backup_path}"
                        )

                samples_df.to_file(samples_df_path, driver="GPKG")
                print(f"✓ Saved to: {samples_df_path}")

                # Display extraction info
                n_samples = len(samples_df)
                print(f"\n📊 Total samples to extract: {n_samples:,}")
                print(f"   Collection ID: {self.collection_id}")
                print(f"   Restart failed jobs: {restart_failed}")
                if self.append_existing:
                    print("   Append mode: enabled (existing samples skipped)")

                # Run extractions
                print("\n" + "=" * 60)
                print("🚀 STARTING EXTRACTIONS")
                print("=" * 60)
                print("\nNote: This may take a while depending on dataset size.")
                print("You can monitor progress below...\n")

                # Start status summary thread to show job tracking overview
                status_active = {"running": True}
                status_interval = 30  # seconds

                def render_status_summary():
                    """Render a summary of job statuses to the dedicated output widget."""
                    tracking_path = outfolder_col / "job_tracking.csv"
                    job_db = CsvJobDatabase(tracking_path)

                    with status_output:
                        clear_output(wait=True)
                        print("🧭 Extraction status summary")
                        print(
                            f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )

                        if not tracking_path.exists():
                            print("Waiting for job tracking file to be created...")
                            return

                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    category=FutureWarning,
                                    message=".*WKTReadingError is deprecated.*",
                                )
                                check_job_status(job_db)
                        except Exception as exc:
                            print(f"⚠️  Could not read job status: {exc}")

                def status_updater():
                    """Periodically update extraction status summary."""
                    while status_active["running"]:
                        render_status_summary()
                        time.sleep(status_interval)

                status_thread = threading.Thread(target=status_updater, daemon=True)
                status_thread.start()

                try:
                    # Run the extraction workflow
                    job_db = run_extractions(
                        ExtractionCollection.POINT_WORLDCEREAL,
                        outfolder_col,
                        samples_df_path,
                        self.collection_id,
                        extract_value=1,  # Fixed value - only extract samples with extract=1
                        restart_failed=restart_failed,
                    )
                except KeyboardInterrupt:
                    print("\n🛑 Extractions interrupted by user")
                finally:
                    # Stop status update thread
                    status_active["running"] = False
                    status_thread.join(timeout=1)
                    render_status_summary()

                print("\n" + "=" * 60)
                print("✅ EXTRACTIONS COMPLETED")
                print("=" * 60)

                # Reset append mode state so existing extractions are re-evaluated
                self.append_existing = False
                self.append_samples_df = None

                # Check job status
                print("\n📈 Checking job status...")
                check_job_status(job_db)

                # Get details of succeeded jobs
                get_succeeded_job_details(job_db)

                print("\n✅ You can now proceed to Tab 4 to visualize results.")

            except Exception as e:
                print(f"\n❌ Error during extractions: {str(e)}")
                traceback.print_exc()
                return

    def _on_reset_auth_click(self, button):
        """Clear openEO token cache and inform user."""
        auth_output = self.tab3_widgets["auth_output"]
        auth_output.clear_output()
        with auth_output:
            try:
                from notebook_utils.openeo import clear_openeo_token_cache

                clear_openeo_token_cache()
                self.cdse_auth_cleared = True
                self.cdse_auth_in_progress = False
                print("✅ CDSE authentication token has been cleared.")
                print(
                    "Click the link below to authenticate with your CDSE credentials ⬇️"
                )
            except Exception as e:
                print(f"❌ Error clearing authentication token: {e}")
                return
            trigger_cdse_authentication(auth_output)

        self.cdse_auth_cleared = False
        self._update_cdse_auth_ui()

    def _on_authenticate_cdse_click(self, button):
        """Trigger CDSE device authentication flow."""
        auth_output = self.tab3_widgets["auth_output"]
        if self.cdse_auth_in_progress:
            return
        self.cdse_auth_in_progress = True
        auth_output.clear_output()
        with auth_output:
            print("🔐 Authenticating with CDSE...")
            print("Click the link below to authenticate with your CDSE credentials ⬇️")

        def _run_auth() -> None:
            trigger_cdse_authentication(auth_output)
            self.cdse_auth_in_progress = False
            self.cdse_auth_cleared = False
            self._update_cdse_auth_ui()

        threading.Thread(target=_run_auth, daemon=True).start()

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

    # =========================================================================
    # Tab 4: Visualize Results
    # =========================================================================

    def _build_tab4_visualize_results(self) -> widgets.VBox:
        """
        Build Tab 4: Visualize and inspect extraction results.

        Returns
        -------
        widgets.VBox
            Container widget for Tab 4
        """
        # Header
        header = widgets.HTML(
            value="<h3>Visualize Results</h3>"
            "<p>Inspect and visualize your extraction results.</p>"
        )

        # Status message
        status_message = widgets.HTML(
            value="<i>Please complete extractions in Tab 3 first.</i>"
        )

        # Data info display
        data_info = widgets.HTML(value="")

        # Load results button
        load_button = widgets.Button(
            description="Load Results",
            button_style="info",
            icon="refresh",
            layout=widgets.Layout(width="200px", height="40px"),
        )

        # Load status output (kept near the load button)
        load_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                border="1px solid #ccc",
                padding="10px",
            )
        )

        # Band statistics info
        info_statistics = self._info_callout(
            "For each band, we display basic statistics including mean, median, standard deviation, min, max, and count of valid observations across all samples and time steps.<br>"
        )

        # Statistics display
        stats_output = widgets.Output()

        # Visualization options
        viz_header = widgets.HTML("<h4>Time Series Visualization</h4>")

        viz_info = self._info_callout(
            "Here you have the option to visualize time series for a selected band.<br>"
            "You can choose the number of samples to display and filter by specific crop types if desired.<br>"
            "After selecting your options, click the 'Visualize Time Series' button to generate the plots.<br>"
        )

        # Get all available bands (including NDVI as special case)
        all_bands = ["NDVI"] + [
            band for bands in WORLDCEREAL_BANDS.values() for band in bands
        ]

        band_dropdown = widgets.Dropdown(
            options=all_bands,
            value="NDVI",
            description="Band:",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="300px"),
        )

        n_samples_text = widgets.IntText(
            value=5,
            description="# samples:",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="200px"),
        )

        # Crop type selection (initially empty, populated after loading data)
        crop_type_select = widgets.SelectMultiple(
            options=[],
            description="Crop types:",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="400px", height="150px"),
            disabled=True,
        )

        crop_type_help = widgets.HTML(
            value="<i>Select one or more crop types to filter samples. Leave empty to show all.</i>"
        )

        visualize_button = widgets.Button(
            description="Visualize Time Series",
            button_style="primary",
            icon="line-chart",
            disabled=True,
            layout=widgets.Layout(width="250px", height="40px"),
        )

        # Results display area
        results_output = widgets.Output()

        # Store widgets
        self.tab4_widgets = {
            "status_message": status_message,
            "data_info": data_info,
            "load_button": load_button,
            "load_output": load_output,
            "band_dropdown": band_dropdown,
            "n_samples_text": n_samples_text,
            "crop_type_select": crop_type_select,
            "visualize_button": visualize_button,
            "results_output": results_output,
            "stats_output": stats_output,
        }

        # Connect callbacks
        load_button.on_click(self._on_load_results_click)
        visualize_button.on_click(self._on_visualize_click)

        return widgets.VBox(
            [
                header,
                status_message,
                widgets.HBox([load_button], layout=widgets.Layout(margin="10px 0")),
                load_output,
                data_info,
                widgets.HTML("<h4>Band Statistics</h4>"),
                info_statistics,
                stats_output,
                viz_header,
                viz_info,
                widgets.HBox([band_dropdown, n_samples_text]),
                crop_type_help,
                crop_type_select,
                widgets.HBox(
                    [visualize_button], layout=widgets.Layout(margin="10px 0")
                ),
                results_output,
                self._build_tab_navigation(),
            ]
        )

    def _on_load_results_click(self, button):
        """Handle load results button click."""
        output = self.tab4_widgets["load_output"]

        with output:
            clear_output()
            try:
                if self.collection_id is None:
                    print("⚠ No collection ID set. Please complete previous steps.")
                    return

                extraction_path = self.extractions_folder / self.collection_id

                if not extraction_path.exists():
                    print(f"⚠ Extraction folder not found: {extraction_path}")
                    print("Please complete extractions in Tab 3 first.")
                    return

                # Load extractions using the existing function
                self.extractions_gdf = load_point_extractions(
                    extraction_path, subset=False
                )

                # Add crop type labels to the dataframe
                self.extractions_gdf["label_full"] = ewoc_code_to_label(
                    self.extractions_gdf["ewoc_code"], label_type="full"
                )
                self.extractions_gdf["sampling_label"] = ewoc_code_to_label(
                    self.extractions_gdf["ewoc_code"], label_type="sampling"
                )

                # Display data information
                n_samples = self.extractions_gdf["sample_id"].nunique()
                n_records = len(self.extractions_gdf)

                self.tab4_widgets["data_info"].value = (
                    f"<div style='background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                    f"<b>✓ Data loaded successfully</b><br>"
                    f"<b>Total samples:</b> {n_samples}<br>"
                    f"<b>Total records:</b> {n_records} (timesteps × samples)"
                    f"</div>"
                )

                # Populate crop type selector
                crop_types = sorted(self.extractions_gdf["label_full"].unique())
                self.tab4_widgets["crop_type_select"].options = crop_types
                self.tab4_widgets["crop_type_select"].disabled = False

                # Enable visualization button
                self.tab4_widgets["visualize_button"].disabled = False

                # Display band statistics
                self._display_band_statistics()

            except Exception as e:
                print(f"❌ Error loading results: {str(e)}")
                logger.error(f"Error loading results: {traceback.format_exc()}")

    def _on_visualize_click(self, button):
        """Handle visualize button click."""
        output = self.tab4_widgets["results_output"]

        with output:
            clear_output()
            try:
                if self.extractions_gdf is None:
                    print(
                        "⚠ No extraction data loaded. Please click 'Load Results' first."
                    )
                    return

                # Get visualization parameters
                band = self.tab4_widgets["band_dropdown"].value
                n_samples = self.tab4_widgets["n_samples_text"].value
                selected_crop_types = list(self.tab4_widgets["crop_type_select"].value)

                # Filter by crop type if specified
                if selected_crop_types:
                    filtered_gdf = self.extractions_gdf[
                        self.extractions_gdf["label_full"].isin(selected_crop_types)
                    ]
                    print(
                        f"Filtering to {len(selected_crop_types)} crop type(s): {', '.join(selected_crop_types)}"
                    )
                else:
                    filtered_gdf = self.extractions_gdf
                    print("Showing all crop types")

                # Check if we have enough samples
                available_samples = filtered_gdf["sample_id"].nunique()
                if available_samples == 0:
                    print("⚠ No samples match the selected crop type filter.")
                    return

                if available_samples < n_samples:
                    print(
                        f"⚠ Only {available_samples} samples available with current filter."
                    )
                    n_samples = available_samples

                print(f"Visualizing {band} for {n_samples} sample(s)...\n")

                # Call the visualization function
                visualize_timeseries(
                    filtered_gdf,
                    nsamples=n_samples,
                    band=band,
                    crop_label_attr="label_full",
                )

            except Exception as e:
                print(f"❌ Error creating visualization: {str(e)}")
                logger.error(f"Error in visualization: {traceback.format_exc()}")

    def _display_band_statistics(self):
        """Display statistics for each extracted band."""
        stats_output = self.tab4_widgets["stats_output"]

        with stats_output:
            clear_output()
            try:
                if self.extractions_gdf is None:
                    print("No data loaded.")
                    return

                # Call the statistics function which prints the table
                get_band_statistics(self.extractions_gdf)

            except Exception as e:
                print(f"❌ Error displaying statistics: {str(e)}")
                logger.error(f"Error in statistics display: {traceback.format_exc()}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _on_tab_change(self, change):
        """Handle tab change events to update UI state."""
        tab_index = change["new"]
        logger.info(f"Switched to tab {tab_index}")

        self._update_nav_buttons()

        # Update UI based on current state when switching tabs
        if tab_index == 1:  # Select Samples tab
            self._update_tab2_state()
        elif tab_index == 2:  # Run Extractions tab
            self._update_tab3_state()
        elif tab_index == 3:  # Visualize Results tab
            self._update_tab4_state()

    def _update_tab2_state(self):
        """Update Tab 2 UI based on whether data is loaded."""
        if self.raw_gdf is not None:
            self.tab2_widgets["status_message"].value = (
                f"<b>✓ Data loaded:</b> {self.collection_id}"
            )
            self.tab2_widgets["prepare_button"].disabled = False
            self.tab2_widgets["sample_selection_radio"].disabled = False

            # Update selection options with counts
            unique_samples = self.raw_gdf.drop_duplicates(subset=["sample_id"])
            total_samples = len(unique_samples)
            extract_col = (
                "extract_original"
                if "extract_original" in unique_samples.columns
                else "extract"
            )
            if extract_col in unique_samples.columns:
                preselected = len(unique_samples[unique_samples[extract_col] > 0])
            else:
                preselected = 0

            self.tab2_widgets["sample_selection_radio"].options = [
                (
                    f"Use pre-selected samples only ({preselected:,})",
                    "preselected",
                ),
                (f"Use all samples ({total_samples:,})", "all"),
                ("Select classes and/or run custom sampling script", "custom"),
            ]
            # Ensure a valid default value after options are populated
            self.tab2_widgets["sample_selection_radio"].value = "preselected"

            # Display dataset information
            self._display_dataset_info()
        else:
            self.tab2_widgets["status_message"].value = (
                "<i>⚠ Please load reference data in Tab 1 first.</i>"
            )
            self.tab2_widgets["prepare_button"].disabled = True
            self.tab2_widgets["sample_selection_radio"].disabled = True

    def _update_tab3_state(self):
        """Update Tab 3 UI based on whether samples are prepared and check for existing extractions."""
        warning_output = self.tab3_widgets["warning_output"]

        self._update_cdse_auth_ui()

        if self.samples_df is not None and self.collection_id is not None:
            self.tab3_widgets["status_message"].value = (
                f"<b>✓ Samples prepared:</b> {len(self.samples_df)} samples ready for extraction<br>"
                f"<b>Output folder:</b> {self.extractions_folder / self.collection_id}"
            )

            if self.append_existing and self.append_samples_df is not None:
                with warning_output:
                    clear_output()
                    remaining_samples = len(self.append_samples_df)
                    skipped_samples = len(self.samples_df) - remaining_samples
                    print("✅ Append mode enabled.")
                    print(f"   Existing sample_ids skipped: {skipped_samples:,}")
                    print(f"   Samples remaining for extraction: {remaining_samples:,}")
                    if remaining_samples == 0:
                        print("\n⚠️  All samples already exist in extractions.")
                        self.tab3_widgets["run_button"].disabled = True
                    else:
                        self.tab3_widgets["run_button"].disabled = False
                        print("\n✓ Ready to append new samples.")
                return

            # Check if extractions already exist
            outfolder_col = self.extractions_folder / self.collection_id
            samples_df_path = outfolder_col / "samples_gdf.gpkg"
            existing_extractions = []
            if outfolder_col.exists():
                existing_extractions = list(outfolder_col.glob("**/*.geoparquet"))
                existing_extractions = [
                    f for f in existing_extractions if not f.is_dir()
                ]

            if samples_df_path.exists() or len(existing_extractions) > 0:
                # Existing extractions found - show warning and options
                self.tab3_widgets["run_button"].disabled = True

                with warning_output:
                    clear_output()
                    print("⚠️  WARNING: Existing extractions found for this dataset!")
                    print("\n📋 Please choose an action:")

                    # Create option buttons
                    delete_button = widgets.Button(
                        description="Delete Existing & Start Fresh",
                        button_style="danger",
                        icon="trash",
                        layout=widgets.Layout(width="250px", height="40px"),
                    )

                    append_button = widgets.Button(
                        description="Append New Samples To Existing",
                        button_style="warning",
                        icon="plus",
                        layout=widgets.Layout(width="250px", height="40px"),
                    )

                    button_container = widgets.HBox(
                        [delete_button, append_button],
                        layout=widgets.Layout(margin="10px 0"),
                    )

                    print(
                        "\nℹ️  Selecting the append method will skip any selected samples already extracted."
                    )

                    def on_delete_click(btn):
                        button_container.close()
                        try:
                            self._delete_existing_extractions()
                        except Exception as e:
                            with warning_output:
                                clear_output()
                                print("❌ Error while deleting existing extractions.")
                                print(f"   {e}")
                            return
                        # Re-check state after successful deletion
                        self._update_tab3_state()

                    def on_append_click(btn):
                        button_container.close()
                        self.append_existing = True
                        self.append_samples_df = self._prepare_append_samples(
                            outfolder_col
                        )

                        with warning_output:
                            clear_output()
                            if self.append_samples_df is None:
                                print("❌ No samples prepared. Please prepare samples.")
                                self.tab3_widgets["run_button"].disabled = True
                                return

                            total_samples = len(self.samples_df)
                            remaining_samples = len(self.append_samples_df)
                            skipped_samples = total_samples - remaining_samples
                            print("✅ Append mode enabled.")
                            print(
                                f"   Existing sample_ids skipped: {skipped_samples:,}"
                            )
                            print(
                                f"   Samples remaining for extraction: {remaining_samples:,}"
                            )

                            if remaining_samples == 0:
                                print("\n⚠️  All samples already exist in extractions.")
                                self.tab3_widgets["run_button"].disabled = True
                            else:
                                self.tab3_widgets["run_button"].disabled = False
                                print("\n✓ Ready to append new samples.")

                    delete_button.on_click(on_delete_click)
                    append_button.on_click(on_append_click)

                    display(button_container)
            else:
                # No existing extractions - enable run button
                self.tab3_widgets["run_button"].disabled = False
                with warning_output:
                    clear_output()
                    print("✓ No existing extractions found. Ready to start.")
        else:
            self.tab3_widgets["status_message"].value = (
                "<i>⚠ Please prepare samples in Tab 2 first.</i>"
            )
            self.tab3_widgets["run_button"].disabled = True
            with warning_output:
                clear_output()

        if self._needs_cdse_authentication():
            self.tab3_widgets["run_button"].disabled = True

    def _update_cdse_auth_ui(self):
        """Update CDSE authentication UI elements based on token availability."""
        auth_status = self.tab3_widgets.get("auth_status")
        authenticate_button = self.tab3_widgets.get("authenticate_button")
        reset_auth_button = self.tab3_widgets.get("reset_auth_button")

        if (
            auth_status is None
            or authenticate_button is None
            or reset_auth_button is None
        ):
            return

        if self.cdse_auth_in_progress:
            if auth_status is not None:
                auth_status.value = "<b>⏳ CDSE authentication in progress...</b>"
            authenticate_button.disabled = True
            reset_auth_button.disabled = True
            return

        has_token = not self._needs_cdse_authentication()
        if has_token:
            auth_status.value = "<b>✓ CDSE refresh token found.</b>"
            authenticate_button.disabled = True
            reset_auth_button.disabled = False
        else:
            auth_status.value = "<b>✗ No CDSE refresh token found.</b>"
            authenticate_button.disabled = False
            reset_auth_button.disabled = True

    def _update_tab4_state(self):
        """Update Tab 4 UI based on whether extractions exist."""
        # Check if extractions folder exists and has data
        if self.collection_id is not None:
            extraction_path = self.extractions_folder / self.collection_id
            if extraction_path.exists():
                self.tab4_widgets["status_message"].value = (
                    f"<b>Extraction folder found:</b> {extraction_path}"
                )
            else:
                self.tab4_widgets["status_message"].value = (
                    "<i>⚠ No extraction results found. Please complete extractions in Tab 3.</i>"
                )

    def _log_to_output(
        self, output_widget: widgets.Output, message: str, level: str = "info"
    ):
        """
        Log a message to an output widget.

        Parameters
        ----------
        output_widget : widgets.Output
            The output widget to log to
        message : str
            The message to log
        level : str
            Log level (info, warning, error, success)
        """
        colors = {
            "info": "black",
            "warning": "orange",
            "error": "red",
            "success": "green",
        }

        with output_widget:
            color = colors.get(level, "black")
            print(f"<span style='color: {color}'>[{level.upper()}] {message}</span>")

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
