import threading
from typing import List, Optional

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

from worldcereal.utils.legend import (
    ewoc_code_to_label,
    get_legend,
    translate_ewoc_codes,
)

DEMO_CROPS = [
    1100000000,  # temporary_crops
    1101000000,  # cereals
    1101000001,  # unknown winter cereal
    1101000002,  # unknown spring cereal
    1101010000,  # wheat
    1101010001,  # winterwheat
    1101010002,  # springwheat
    1101020000,  # barley
    1101030000,  # rye
    1101060000,  # maize
    1101070000,  # sorghum
    1101080000,  # rice
    1101120000,  # millet
    1103000000,  # vegetables
    1103060040,  # cauliflower
    1103080080,  # spinach
    1103090040,  # carrots
    1103110040,  # oinions
    1105000000,  # dry pulses
    1105000030,  # lentils
    1105010000,  # beans_peas
    1105010010,  # beans
    1105010040,  # chickpeas
    1106000000,  # oilseeds
    1106000010,  # sunflower
    1106000020,  # soy
    1106000030,  # rapeseed
    1107000000,  # root_tuber
    1107000010,  # potatoes
    1107000040,  # cassava
    1108000000,  # fibre_crops
    1109000000,  # herbs_spice
    1110000000,  # flowers
    1111000000,  # grass_fodder
    1111020000,  # fodder_legumes
    1111020010,  # alfalfa
    1114000000,  # mixed_arable_crops
    1115000000,  # fallow
    1200000000,  # permanent crops
    1201010020,  # apples
    1201000010,  # grapes
    1203000000,  # olives
    1203000030,  # oilpalm
    2000000000,  # non_cropland_herbaceous
    2001000000,  # grasslands
    2002000000,  # wetlands
    2500000000,  # non_cropland_mixed
    3000000000,  # shrubland
    4000000000,  # trees_unspecified
    4100000000,  # trees_broadleaved
    4200000000,  # trees_coniferous
    5000000000,  # bare_sparseley_vegetated
    6000000000,  # built-up
    7000000000,  # open_water
]


class CropTypePicker:
    def __init__(
        self,
        ewoc_codes: Optional[List[int]] = None,
        sample_df: pd.DataFrame = None,
        count_threshold: int = 0,
        expand: bool = False,
        display_ui: bool = True,
        selection_modes: Optional[List[str]] = None,
    ):
        """
        Crop type picker widget for selecting crop types of interest.

        Parameters
        ----------
        ewoc_codes : list[int], optional
            List of EWOC codes to be included in the crop type picker.
            By default None, meaning all crop types from the legend are included
            (this takes a while to load).
        sample_df : pd.DataFrame, optional
            DataFrame containing samples. There should be a column "ewoc_code" indicating the crop type for each sample.
            By default None.
        count_threshold : int, optional
            Minimum count threshold for a crop type to be included in the picker.
            If a crop has a lower count, it will not be displayed.
            By default 0.
        expand : bool, optional
            Whether to expand the widget by default.
            By default False.
        selection_modes : list[str], optional
            Available selection modes. Choose from ["Include", "Drop"].
            By default both modes are available.
        """

        self.ewoc_codes = ewoc_codes
        if sample_df is not None:
            # Count number of samples for each crop type
            sample_count_df = sample_df["ewoc_code"].value_counts().sort_index()
            self.df = sample_count_df.rename("count")
        else:
            self.df = None
        self.count_threshold = count_threshold
        self.expand = expand

        if selection_modes is None:
            selection_modes = ["Include", "Drop"]
        selection_modes = [
            mode for mode in selection_modes if mode in ["Include", "Drop"]
        ]
        if not selection_modes:
            raise ValueError(
                "selection_modes must include at least one of 'Include' or 'Drop'."
            )

        self.legend = None
        self.hierarchy = None
        self.widget = None
        self.widgets_dict: dict[tuple, widgets.Checkbox] = {}
        self.croptypes = pd.DataFrame()
        self.included_croptypes = pd.DataFrame()
        self.dropped_croptypes = pd.DataFrame()
        self.output = widgets.Output()
        self._selection_cache = {"Include": set(), "Drop": set()}
        self._available_modes = selection_modes
        self._current_mode = selection_modes[0]
        self._mode_locked_paths = set()
        self._node_meta = {}
        self._root_paths = []
        self._search_active = False
        self._toggle_state_before_search = {}

        # Initialize the hierarchy and widget
        self._build_hierarchy()
        self._create_widget()

        if display_ui:
            display(self.widget)

    def _simplify_legend(self):
        """Simplify the legend filling missing values with lower levels of the hierarchy"""

        for i in range(4, 1, -1):
            self.legend[f"level_{i}"] = self.legend[f"level_{i}"].fillna(
                self.legend[f"level_{i + 1}"]
            )

        for i in range(2, 4):
            upper = f"level_{i}"
            lower = f"level_{i + 1}"
            self.legend.loc[self.legend[upper] == self.legend[lower], lower] = (
                self.legend[f"level_{i + 2}"]
            )

        # set duplicates to NaN
        for i in range(5, 1, -1):
            self.legend.loc[
                self.legend[f"level_{i}"] == self.legend[f"level_{i - 1}"], f"level_{i}"
            ] = np.nan

    def _legend_to_hierarchy(self):

        levels = [f"level_{i}" for i in range(1, 6)]

        hierarchy = {}
        for i, row in self.legend.iterrows():
            hierarchy[i] = (tuple(row[levels].dropna().values), row["count"])

        # Pre-build lookup dictionaries for O(1) access instead of O(n) searches
        label_to_code = dict(
            zip(self.full_legend["label_full"], self.full_legend.index)
        )
        level_lookups = {}
        for i in range(1, 6):
            level_series = self.full_legend[f"level_{i}"].dropna()
            level_lookups[i] = dict(zip(level_series.values, level_series.index))

        nested_hierarchy = {}

        for ewoc_code, info in hierarchy.items():
            node = nested_hierarchy
            for key in info[0]:  # Traverse through all keys in the hierarchy
                if key not in node:
                    if key == "cereals":
                        code_to_assign = 0
                    if key == info[0][-1]:
                        code_to_assign = ewoc_code
                    elif key in label_to_code:
                        code_to_assign = label_to_code[key]
                    else:
                        # rare case where the key is not in the legend
                        code_to_assign = None
                        for i in range(1, 6):
                            if key in level_lookups[i]:
                                code_to_assign = level_lookups[i][key]
                                break
                        if code_to_assign is None:
                            code_to_assign = 0

                    node[key] = {
                        "__count__": info[1],
                        "ewoc_code": code_to_assign,
                        "children": [],
                    }
                else:
                    # add count to previous level of the hierarchy
                    node[key]["__count__"] += info[1]
                # set ewoc_code only if it is the last key in the hierarchy
                if key != info[0][-1]:
                    # node[key]["ewoc_code"] = ewoc_code

                    # add ewoc_code to the list of children
                    node[key]["children"].append(ewoc_code)

                node = node[key]

        return nested_hierarchy

    def _auto_expand_single_root(self) -> None:
        """Expand the only root node when a single top-level item is available."""
        if len(self._root_paths) != 1:
            return
        root_meta = self._node_meta.get(self._root_paths[0])
        if root_meta is None or root_meta.get("toggle") is None:
            return
        root_meta["toggle"].value = True

    def _filter_low_counts(self, hierarchy) -> dict:
        """Filter out low counts from the hierarchy"""

        def recursive_filter_low_counts(node):
            if isinstance(node, dict):
                # Create a new dictionary to hold updated entries
                updated_node = {}
                for key in list(node.keys()):
                    value = node[key]
                    if isinstance(value, dict):
                        if value["__count__"] >= self.count_threshold:
                            updated_node[key] = recursive_filter_low_counts(value)
                    else:
                        updated_node[key] = value
                return updated_node
            return node

        return recursive_filter_low_counts(hierarchy)

    def _build_hierarchy(self):

        # First get the legend (make a copy to avoid modifying cached version)
        self.full_legend = get_legend().copy()

        # Get rid of unknown class
        self.legend = self.full_legend.loc[self.full_legend.index != 0]

        # Filter legend based on EWOC codes
        if self.ewoc_codes is not None:
            self.legend = self.legend.loc[self.ewoc_codes]

        # filter legend based on codes in the count dataframe
        if self.df is not None:
            self.legend = self.legend.loc[self.df.index]
            # add counts to legend using join operation
            self.legend = self.legend.join(self.df)
            # Sort on index
            self.legend = self.legend.sort_index()
        else:
            # Add empty count column to legend
            self.legend = self.legend.assign(count=0)

        # Simplify the legend by removing NaN's in intermediate levels
        self._simplify_legend()

        # Now create a nested dictionary from the legend
        hierarchy = self._legend_to_hierarchy()

        # Reduce complexity in the hierarchy by filtering out levels that only have one child
        if (self.df is not None) or (self.ewoc_codes is not None):
            hierarchy = self._simplify_hierarchy(hierarchy)

        # Filter out classes with low sample counts
        if self.df is not None:
            hierarchy = self._filter_low_counts(hierarchy)

        self.hierarchy = hierarchy

    def _disable_descendants(self, widget):
        if isinstance(widget, widgets.Checkbox):
            widget.disabled = True
            widget.value = True
        if hasattr(widget, "children"):
            for child in widget.children:
                self._disable_descendants(child)

    def _enable_descendants(self, widget):
        if isinstance(widget, widgets.Checkbox):
            widget.disabled = False
        if hasattr(widget, "children"):
            for child in widget.children:
                self._enable_descendants(child)

    def _clear_descendants(self, widget):
        if isinstance(widget, widgets.Checkbox):
            widget.disabled = False
            widget.value = False
        if hasattr(widget, "children"):
            for child in widget.children:
                self._clear_descendants(child)

    def _set_toggle_state(self, toggle_button, enabled: bool) -> None:
        toggle_button.disabled = not enabled
        toggle_button.style.button_color = "#16a34a" if enabled else "#A9A9A9"

    def _lock_descendants_async(
        self, children_vbox, load_children=None, toggle_button=None
    ):
        if toggle_button is not None:
            self._set_toggle_state(toggle_button, False)

        def _worker():
            if load_children is not None:
                load_children()
            self._disable_descendants(children_vbox)
            if toggle_button is not None:
                self._set_toggle_state(toggle_button, True)

        threading.Thread(target=_worker, daemon=True).start()

    def _clear_descendants_async(self, children_vbox, toggle_button=None):
        if toggle_button is not None:
            self._set_toggle_state(toggle_button, False)

        def _worker():
            self._clear_descendants(children_vbox)
            if toggle_button is not None:
                self._set_toggle_state(toggle_button, True)

        threading.Thread(target=_worker, daemon=True).start()

    def _is_blocked_by_parent(self, path):
        if len(path) <= 1:
            return False
        for i in range(1, len(path)):
            parent = path[:i]
            if parent in self.widgets_dict and self.widgets_dict[parent].value:
                return True
        return False

    def _update_mode_locks(self, current_mode: str):
        if len(self._available_modes) < 2:
            self._mode_locked_paths = set()
            for path, checkbox in self.widgets_dict.items():
                if not self._is_blocked_by_parent(path):
                    checkbox.disabled = False
            return

        other_mode = "Drop" if current_mode == "Include" else "Include"
        locked_paths = set(self._selection_cache.get(other_mode, set()))

        for path, checkbox in self.widgets_dict.items():
            if path in locked_paths:
                checkbox.value = False
                checkbox.disabled = True
            else:
                if path in self._mode_locked_paths and not self._is_blocked_by_parent(
                    path
                ):
                    checkbox.disabled = False

        self._mode_locked_paths = locked_paths

    def _create_widget(self):
        indent_px = 16

        def build_node(node_key, node_value, path, level):
            current_path = tuple(path + [node_key])
            count = node_value.get("__count__", 0)
            if self.df is not None:
                description = f"{node_key} ({count} samples)"
            else:
                nchildren = len(node_value.get("children", []))
                description = f"{node_key} ({nchildren} subclasses)"

            if level > 0:
                description = f"↳ {description}"

            indent = widgets.HTML(
                value=f"<div style='width:{level * indent_px}px'></div>",
                layout=widgets.Layout(height="0px"),
            )

            checkbox = widgets.Checkbox(
                value=False,
                description=description,
                layout=widgets.Layout(width="auto", max_width="95%"),
            )
            self.widgets_dict[current_path] = checkbox

            child_keys = [
                k
                for k in node_value.keys()
                if k not in ["__count__", "ewoc_code", "children"]
            ]
            if not child_keys:
                leaf_container = widgets.HBox(
                    [indent, checkbox],
                    layout=widgets.Layout(width="100%", align_items="center"),
                )
                self._node_meta[current_path] = {
                    "container": leaf_container,
                    "checkbox": checkbox,
                    "children": [],
                    "children_vbox": None,
                    "toggle": None,
                    "load_children": None,
                    "loaded_flag": None,
                    "label": str(node_key),
                }
                return leaf_container

            children_vbox = widgets.VBox(
                [],
                layout=widgets.Layout(width="100%", align_items="flex-start"),
            )
            children_vbox.layout.display = "none"

            loaded = {"value": False}
            parent_selected = {"value": False}

            def load_children():
                if loaded["value"]:
                    return
                child_widgets = []
                for child_key in child_keys:
                    child_widgets.append(
                        build_node(
                            child_key,
                            node_value[child_key],
                            list(current_path),
                            level + 1,
                        )
                    )
                children_vbox.children = child_widgets
                loaded["value"] = True
                if parent_selected["value"]:
                    self._disable_descendants(children_vbox)

            if self.expand:
                load_children()
                children_vbox.layout.display = "flex"
                toggle_value = True
                toggle_desc = "Collapse"
                toggle_icon = "chevron-up"
            else:
                toggle_value = False
                toggle_desc = "Expand"
                toggle_icon = "chevron-down"

            toggle_button = widgets.ToggleButton(
                value=toggle_value,
                description=toggle_desc,
                icon=toggle_icon,
                layout=widgets.Layout(width="150px", margin="0 5px 0 0"),
                style={"button_color": "#A9A9A9"},
            )
            self._set_toggle_state(toggle_button, True)

            def toggle_visibility(change):
                if change["new"]:
                    load_children()
                    if parent_selected["value"]:
                        self._disable_descendants(children_vbox)
                    children_vbox.layout.display = "flex"
                    toggle_button.description = "Collapse"
                    toggle_button.icon = "chevron-up"
                else:
                    children_vbox.layout.display = "none"
                    toggle_button.description = "Expand"
                    toggle_button.icon = "chevron-down"

            toggle_button.observe(toggle_visibility, names="value")

            def on_parent_change(change):
                parent_selected["value"] = change["new"]
                if loaded["value"]:
                    if change["new"]:
                        children_vbox.layout.display = "none"
                        toggle_button.value = False
                        self._lock_descendants_async(
                            children_vbox, toggle_button=toggle_button
                        )
                    else:
                        children_vbox.layout.display = "none"
                        toggle_button.value = False
                        self._clear_descendants_async(
                            children_vbox, toggle_button=toggle_button
                        )
                else:
                    if change["new"]:
                        children_vbox.layout.display = "none"
                        toggle_button.value = False
                        self._lock_descendants_async(
                            children_vbox,
                            load_children,
                            toggle_button=toggle_button,
                        )

            checkbox.observe(on_parent_change, names="value")

            node_stack = widgets.VBox(
                [checkbox, toggle_button, children_vbox],
                layout=widgets.Layout(width="100%", align_items="flex-start"),
            )

            container = widgets.HBox(
                [indent, node_stack],
                layout=widgets.Layout(width="100%", align_items="flex-start"),
            )

            child_paths = [tuple(list(current_path) + [key]) for key in child_keys]
            self._node_meta[current_path] = {
                "container": container,
                "checkbox": checkbox,
                "children": child_paths,
                "children_vbox": children_vbox,
                "toggle": toggle_button,
                "load_children": load_children,
                "loaded_flag": loaded,
                "label": str(node_key),
            }

            return container

        items = []
        for key, value in self.hierarchy.items():
            if key in ["__count__", "ewoc_code", "children"]:
                continue
            node_widget = build_node(key, value, [], 0)
            items.append(node_widget)
            self._root_paths.append((key,))

        tree = widgets.VBox(
            items, layout=widgets.Layout(width="100%", align_items="flex-start")
        )

        self._auto_expand_single_root()

        submit_button = widgets.Button(description="Apply", button_style="success")
        submit_button.on_click(self.apply_selection)

        clear_button = widgets.Button(
            description="Clear selection", button_style="danger"
        )
        clear_button.on_click(self.clear_selection)

        buttons = widgets.HBox(
            [submit_button, clear_button],
            layout=widgets.Layout(
                aligh_items="flex-start",
                justify_content="flex-start",
                width="100%",
            ),
        )

        selection_title = widgets.HTML("<h2>First pick a selection mode:</h2>")

        selection_mode = widgets.ToggleButtons(
            options=self._available_modes,
            value=self._current_mode,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )
        self.selection_mode = selection_mode

        if len(self._available_modes) < 2:
            selection_title.layout.display = "none"
            selection_mode.layout.display = "none"

        title = widgets.HTML()

        def update_title(mode: str):
            if mode == "Drop":
                title.value = """<h2>Select classes to drop:</h2>"""
            else:
                title.value = """<h2>Select classes to include:</h2>"""

        update_title(selection_mode.value)
        self._current_mode = selection_mode.value
        self._update_mode_locks(self._current_mode)

        def get_selected_paths():
            return {
                path for path, checkbox in self.widgets_dict.items() if checkbox.value
            }

        def set_selected_paths(paths):
            for path, checkbox in self.widgets_dict.items():
                checkbox.value = path in paths

        def reset_search_and_collapse():
            search_box.value = ""
            self._toggle_state_before_search = {}
            self._search_active = False
            for meta in self._node_meta.values():
                meta["container"].layout.display = "flex"
                if meta.get("toggle") is not None:
                    meta["toggle"].value = False
                if meta.get("children_vbox") is not None:
                    meta["children_vbox"].layout.display = "none"
            self._auto_expand_single_root()

        def on_mode_change(change):
            previous_mode = self._current_mode
            self._selection_cache[previous_mode] = get_selected_paths()
            self._current_mode = change["new"]
            update_title(change["new"])
            reset_search_and_collapse()
            set_selected_paths(self._selection_cache[self._current_mode])
            self._update_mode_locks(self._current_mode)

        selection_mode.observe(on_mode_change, names="value")

        search_box = widgets.Text(
            description="Search:",
            placeholder="Type to filter classes...",
            continuous_update=False,
            layout=widgets.Layout(width="60%"),
        )

        search_button = widgets.Button(
            description="Search",
            button_style="primary",
            layout=widgets.Layout(width="120px"),
        )

        def apply_search(term: str):
            term = term.strip().lower()
            if term:
                if not self._search_active:
                    self._toggle_state_before_search = {
                        path: meta["toggle"].value
                        for path, meta in self._node_meta.items()
                        if meta.get("toggle") is not None
                    }
                    self._search_active = True

                def get_child_keys(node_value):
                    return [
                        key
                        for key in node_value.keys()
                        if key not in ["__count__", "ewoc_code", "children"]
                    ]

                show_set = set()
                expand_set = set()

                def recurse(node_value, path):
                    label = str(path[-1]).lower()
                    label_match = term in label
                    child_match = False
                    for child_key in get_child_keys(node_value):
                        if recurse(node_value[child_key], path + (child_key,)):
                            child_match = True
                    match = label_match or child_match
                    if match:
                        show_set.add(path)
                    if child_match:
                        expand_set.add(path)
                    return match

                for root_key in self.hierarchy.keys():
                    if root_key in ["__count__", "ewoc_code", "children"]:
                        continue
                    recurse(self.hierarchy[root_key], (root_key,))

                def ensure_path_loaded(path):
                    for i in range(1, len(path) + 1):
                        prefix = path[:i]
                        meta = self._node_meta.get(prefix)
                        if not meta:
                            return
                        if meta.get("toggle") is not None:
                            if not meta["loaded_flag"]["value"]:
                                meta["load_children"]()
                            meta["children_vbox"].layout.display = "flex"
                            meta["toggle"].value = True

                for path in sorted(expand_set, key=len):
                    ensure_path_loaded(path)

                for path, meta in self._node_meta.items():
                    meta["container"].layout.display = (
                        "flex" if path in show_set else "none"
                    )
                    if meta.get("children_vbox") is not None:
                        if path in expand_set:
                            meta["children_vbox"].layout.display = "flex"
                        else:
                            meta["children_vbox"].layout.display = "none"
            else:
                if self._search_active:
                    for path, value in self._toggle_state_before_search.items():
                        meta = self._node_meta.get(path)
                        if meta and meta.get("toggle") is not None:
                            meta["toggle"].value = value
                    self._toggle_state_before_search = {}
                    self._search_active = False

                for meta in self._node_meta.values():
                    meta["container"].layout.display = "flex"
                    if meta.get("children_vbox") is not None and meta.get("toggle"):
                        meta["children_vbox"].layout.display = (
                            "flex" if meta["toggle"].value else "none"
                        )

        def on_search_click(_):
            apply_search(search_box.value)

        search_button.on_click(on_search_click)

        search_row = widgets.HBox(
            [search_box, search_button],
            layout=widgets.Layout(
                width="100%",
                justify_content="flex-start",
                align_items="center",
            ),
        )

        self.widget = widgets.VBox(
            [
                selection_title,
                selection_mode,
                title,
                search_row,
                tree,
                buttons,
                self.output,
            ],
            layout=widgets.Layout(
                overflow="hidden",
                width="100%",
                align_items="flex-start",
            ),
        )

    def apply_selection(self, change=None):
        """Extract the selected ewoc_codes by the user"""

        selected_paths = [
            path for path, checkbox in self.widgets_dict.items() if checkbox.value
        ]

        self._selection_cache[self.selection_mode.value] = set(selected_paths)

        if "Include" in self._available_modes:
            include_paths = list(self._selection_cache.get("Include", set()))
            if not include_paths:
                raise ValueError("No crop types selected.")
            self.included_croptypes = self._build_croptype_df(include_paths)
            self.croptypes = self.included_croptypes
        else:
            self.included_croptypes = pd.DataFrame()
            self.croptypes = pd.DataFrame()

        if "Drop" in self._available_modes:
            drop_paths = list(self._selection_cache.get("Drop", set()))
            if drop_paths:
                self.dropped_croptypes = self._build_croptype_df(drop_paths)
            else:
                self.dropped_croptypes = pd.DataFrame()
        else:
            self.dropped_croptypes = pd.DataFrame()

        self._update_mode_locks(self.selection_mode.value)

        final_include = (
            np.unique(self.included_croptypes["new_label"].values)
            if not self.included_croptypes.empty
            else []
        )
        dropped_count = (
            len(self.dropped_croptypes) if not self.dropped_croptypes.empty else 0
        )

        include_items = sorted(final_include) if len(final_include) > 0 else []
        excluded_items = (
            sorted(np.unique(self.dropped_croptypes["new_label"].values))
            if dropped_count > 0
            else []
        )

        include_list = (
            "".join(f"<li>{ct}</li>" for ct in include_items)
            if include_items
            else "<li><em>None selected</em></li>"
        )
        excluded_list = (
            "".join(f"<li>{ct}</li>" for ct in excluded_items)
            if excluded_items
            else "<li><em>None selected</em></li>"
        )

        included_leaf_count = len(self.included_croptypes)
        excluded_leaf_count = len(self.dropped_croptypes)

        include_block = ""
        if "Include" in self._available_modes:
            include_block = f"""
            <div style=\"margin-bottom: 8px;\">
                <div style=\"color: #16a34a; font-weight: 600;\">Included classes</div>
                <div style=\"margin: 2px 0 0 0; font-size: 0.95em;\">
                    Total lowest-level classes selected: <b>{included_leaf_count}</b>
                </div>
                <div style=\"margin: 4px 0 0 0; font-weight: 600;\">Final groups</div>
                <ul style=\"margin: 4px 0 0 16px;\">{include_list}</ul>
            </div>
            """

        exclude_block = ""
        if "Drop" in self._available_modes:
            exclude_block = f"""
            <div>
                <div style=\"color: #dc2626; font-weight: 600;\">Excluded classes</div>
                <div style=\"margin: 2px 0 0 0; font-size: 0.95em;\">
                    Total lowest-level classes excluded: <b>{excluded_leaf_count}</b>
                </div>
                <div style=\"margin: 4px 0 0 0; font-weight: 600;\">Final groups</div>
                <ul style=\"margin: 4px 0 0 16px;\">{excluded_list}</ul>
            </div>
            """

        summary_html = f"""
        <div style="margin-top: 8px;">
            <h3 style="margin: 0 0 8px 0;">Selection summary</h3>
            {include_block}
            {exclude_block}
        </div>
        """

        with self.output:
            self.output.clear_output()
            display(widgets.HTML(summary_html))

    def clear_selection(self, change=None):
        """Clear the selection of crop types"""

        for path, checkbox in self.widgets_dict.items():
            checkbox.value = False
        self.croptypes = pd.DataFrame()
        self.included_croptypes = pd.DataFrame()
        self.dropped_croptypes = pd.DataFrame()
        self._selection_cache = {"Include": set(), "Drop": set()}
        self._mode_locked_paths = set()
        self._update_mode_locks(self.selection_mode.value)
        with self.output:
            self.output.clear_output()
            print("Selection cleared.")

    def _build_croptype_df(self, paths_to_search):
        """Apply the selected crop types on the hierarchy and return the extensive list of crop types
        as a Pandas DataFrame"""

        rows = []

        def recursive_search_path(node, path, lvl):

            for key, value in node.items():
                if key == path[lvl - 1]:
                    if lvl == len(path):
                        rows.append(
                            pd.DataFrame.from_dict(
                                {value["ewoc_code"]: key},
                                orient="index",
                                columns=["new_label"],
                            )
                        )
                        for child in value.get("children", []):
                            rows.append(
                                pd.DataFrame.from_dict(
                                    {child: key},
                                    orient="index",
                                    columns=["new_label"],
                                )
                            )
                    else:
                        recursive_search_path(value, path, lvl + 1)

        for i in range(1, 6):
            paths_lvl = [path for path in paths_to_search if len(path) == i]
            for path in paths_lvl:
                recursive_search_path(self.hierarchy, path, 1)

        # convert rows to DataFrame
        croptype_df = pd.concat(rows)

        # ensure no crop types got selected which were not in the df or list of ewoc_codes
        if self.df is not None:
            croptype_df = croptype_df[croptype_df.index.isin(self.df.index)]
        if self.ewoc_codes is not None:
            croptype_df = croptype_df[croptype_df.index.isin(self.ewoc_codes)]

        croptype_df = croptype_df[~croptype_df.index.duplicated(keep="first")]

        # add original labels for readability
        labels = ewoc_code_to_label(croptype_df.index.values)
        croptype_df["original_label"] = labels

        return croptype_df

    def _simplify_hierarchy(self, hierarchy):
        """
        Simplifies the dictionary hierarchy by moving up children of nodes with ewoc_code 0
        and only one child. Updates the parent key to the new key whenever replacing a node.
        """

        def recursive_simplify(node):
            if isinstance(node, dict):
                # Check if the node should be replaced by its single child
                code = node.get("ewoc_code", -1)
                code_check = code != -1
                if code_check:
                    if self.ewoc_codes is not None:
                        code_check = code not in self.ewoc_codes
                    else:
                        code_check = code not in self.df.index
                if code_check and "children" in node and len(node["children"]) == 1:
                    new_key = [
                        k
                        for k in node.keys()
                        if k not in ["__count__", "children", "ewoc_code"]
                    ][0]
                    if len(node[new_key].get("children", [])) == 0:
                        return recursive_simplify(node[new_key])

                # Process other nested dictionaries
                for key, value in node.items():
                    if isinstance(value, dict) and key not in [
                        "__count__",
                        "children",
                        "ewoc_code",
                    ]:
                        node[key] = recursive_simplify(value)

            return node

        for i in range(3):
            hierarchy = recursive_simplify(hierarchy)

        result = clean_hierarchy_keys(hierarchy, self.full_legend)

        return result


def clean_hierarchy_keys(hierarchy, legend=None):
    """Clean hierarchy keys using a pre-loaded legend to avoid repeated downloads.

    Parameters
    ----------
    hierarchy : dict
        The hierarchy dictionary to clean
    legend : pd.DataFrame, optional
        Pre-loaded legend DataFrame to avoid repeated get_legend() calls
    """

    # Recursively check and update keys
    def recursive_update_key(node):
        if isinstance(node, dict):
            # Create a new dictionary to hold updated entries
            updated_node = {}
            for key in list(node.keys()):
                value = node[key]
                if isinstance(value, dict):
                    if value.get("ewoc_code", 0) != 0:
                        code = value["ewoc_code"]
                        full_name = translate_ewoc_codes([code], legend=legend)[
                            "label_full"
                        ].values[0]
                        if (key != full_name) and (len(value.get("children", [])) == 0):
                            # Replace key and update value recursively
                            updated_node[full_name] = recursive_update_key(value)
                        else:
                            # Keep key and update value recursively
                            updated_node[key] = recursive_update_key(value)
                    else:
                        # No ewoc_code; process value recursively
                        updated_node[key] = recursive_update_key(value)
                else:
                    # Leaf value, keep it as is
                    updated_node[key] = value
            return updated_node
        return node  # Return leaf nodes unchanged

    return recursive_update_key(hierarchy)


def apply_croptypepicker_to_df(df, croptypepicker, other_label: str = "other"):
    """Apply the selected crop types from the CropTypePicker to a DataFrame of samples
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing samples. There should be a column "ewoc_code" indicating the crop type for each sample.
    croptypepicker : CropTypePicker
        CropTypePicker object containing the selected crop types.
    other_label : str, optional
        Label to assign to samples that do not belong to the selected crop types.
        By default "other".

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the samples that belong to the selected crop types.
    """

    if croptypepicker.included_croptypes.empty:
        raise ValueError("No crop types selected for inclusion, cannot proceed.")

    if not croptypepicker.dropped_croptypes.empty:
        dropped = croptypepicker.dropped_croptypes.index.values
        df = df.loc[~df["ewoc_code"].isin(dropped)].copy()

    # Isolate all samples that have NOT been selected
    included = croptypepicker.included_croptypes.index.values
    excluded_sample_ids = df[~df["ewoc_code"].isin(included)]["sample_id"].to_list()

    # Prepare a mapping dictionary from original labels (index) to new labels
    label_mapping = croptypepicker.included_croptypes["new_label"].to_dict()

    # Apply the mapping to the ewoc_code column
    df["downstream_class"] = df["ewoc_code"].map(label_mapping)

    # Excluded samples are assigned to "other" class
    df.loc[df["sample_id"].isin(excluded_sample_ids), "downstream_class"] = other_label

    return df
