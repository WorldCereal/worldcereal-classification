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

        self.legend = None
        self.hierarchy = None
        self.widget = None
        self.widgets_dict: dict[int, widgets.Checkbox] = {}
        self.croptypes = pd.DataFrame()
        self.output = widgets.Output()

        # Initialize the hierarchy and widget
        self._build_hierarchy()
        self._create_widget()

        display(self.widget)

    def _simplify_legend(self):
        """Simplify the legend filling missing values with lower levels of the hierarchy"""

        for i in range(4, 1, -1):
            self.legend[f"level_{i}"] = self.legend[f"level_{i}"].fillna(
                self.legend[f"level_{i+1}"]
            )

        for i in range(2, 4):
            upper = f"level_{i}"
            lower = f"level_{i+1}"
            self.legend.loc[self.legend[upper] == self.legend[lower], lower] = (
                self.legend[f"level_{i+2}"]
            )

        # set duplicates to NaN
        for i in range(5, 1, -1):
            self.legend.loc[
                self.legend[f"level_{i}"] == self.legend[f"level_{i-1}"], f"level_{i}"
            ] = np.nan

    def _legend_to_hierarchy(self):

        levels = [f"level_{i}" for i in range(1, 6)]

        hierarchy = {}
        for i, row in self.legend.iterrows():
            hierarchy[i] = (tuple(row[levels].dropna().values), row["count"])

        nested_hierarchy = {}

        for ewoc_code, info in hierarchy.items():
            node = nested_hierarchy
            for key in info[0]:  # Traverse through all keys in the hierarchy
                if key not in node:
                    if key == "cereals":
                        code_to_assign = 0
                    if key == info[0][-1]:
                        code_to_assign = ewoc_code
                    elif key in self.full_legend["label_full"].values:
                        code_to_assign = self.full_legend.loc[
                            self.full_legend["label_full"] == key
                        ].index.values[0]
                    else:
                        # rare case where the key is not in the legend
                        for i in range(1, 6):
                            if key in self.full_legend[f"level_{i}"].values:
                                code_to_assign = self.full_legend.loc[
                                    self.full_legend[f"level_{i}"] == key
                                ].index.values[0]
                                break

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

        # First get the legend
        self.full_legend = get_legend()
        self.full_legend["ewoc_code"] = (
            self.full_legend["ewoc_code"].str.replace("-", "").astype(int)
        )
        self.full_legend = self.full_legend.set_index("ewoc_code")

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
        widget.disabled = True
        widget.value = False
        if isinstance(widget, widgets.VBox):
            for child in widget.children:
                self._disable_descendants(child)

    def _enable_descendants(self, widget):
        widget.disabled = False
        if isinstance(widget, widgets.VBox):
            for child in widget.children:
                self._enable_descendants(child)

    def _create_widget(self):
        def recursive_create_widgets(hierarchy, path=None, level=0):
            if path is None:
                path = []

            items = []
            for key, value in hierarchy.items():
                if key in ["__count__", "ewoc_code", "children"]:
                    continue
                current_path = tuple(path + [key])
                count = value.get("__count__", 0)
                if self.df is not None:
                    description = f"{key} ({count} samples)"
                else:
                    nchildren = len(value.get("children", []))
                    description = f"{key} ({nchildren} subclasses)"
                checkbox = widgets.Checkbox(
                    value=False,
                    description=description,
                    layout=widgets.Layout(
                        margin=f"0 0 0 {level * 20}px", width="auto", max_width="95%"
                    ),
                )
                self.widgets_dict[current_path] = checkbox

                # Process child items recursively
                child_items = []
                for child_key, child_value in value.items():
                    if child_key not in ["__count__", "ewoc_code", "children"]:
                        child_items.append(
                            recursive_create_widgets(
                                {child_key: child_value},
                                list(current_path),
                                level + 1,
                            )
                        )

                # Append children if they exist
                if child_items:
                    # Create a collapsible section with a toggle button
                    children_vbox = widgets.VBox(
                        child_items,
                        layout=widgets.Layout(width="100%", align_items="flex-start"),
                    )
                    if self.expand:
                        children_vbox.layout.display = "flex"
                    else:
                        children_vbox.layout.display = "none"  # Hide by default

                    if self.expand:
                        value = True
                        description = "Collapse"
                        icon = "chevron-up"
                    else:
                        value = False
                        description = "Expand"
                        icon = "chevron-down"

                    toggle_button = widgets.ToggleButton(
                        value=value,
                        description=description,
                        icon=icon,
                        layout=widgets.Layout(
                            width="150px", margin=f"0 5px 0 {level * 20}px"
                        ),
                        style={"button_color": "#A9A9A9"},
                    )

                    def toggle_visibility(
                        change, target=children_vbox, button=toggle_button
                    ):
                        if change["new"]:
                            target.layout.display = "flex"
                            button.description = "Collapse"
                            button.icon = "chevron-up"
                        else:
                            target.layout.display = "none"
                            button.description = "Expand"
                            button.icon = "chevron-down"

                    toggle_button.observe(toggle_visibility, names="value")

                    vbox = widgets.VBox(
                        [checkbox, toggle_button, children_vbox],
                        layout=widgets.Layout(width="100%", align_items="flex-start"),
                    )

                    # Define behavior for disabling all descendants when a parent is selected
                    def on_parent_change(change, target_child_items=child_items):
                        if change["new"]:
                            for child in target_child_items:
                                self._disable_descendants(child)
                        else:
                            for child in target_child_items:
                                self._enable_descendants(child)

                    checkbox.observe(on_parent_change, names="value")
                    items.append(vbox)
                else:
                    items.append(checkbox)
            return widgets.VBox(
                items, layout=widgets.Layout(width="100%", align_items="flex-start")
            )

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

        title = widgets.HTML("""<h2>Select your crop types of interest:</h2>""")

        self.widget = widgets.VBox(
            [title, recursive_create_widgets(self.hierarchy), buttons, self.output],
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

        if not selected_paths:
            raise ValueError("No crop types selected.")

        self._apply_hierarchy_on_selection(selected_paths)

    def clear_selection(self, change=None):
        """Clear the selection of crop types"""

        for path, checkbox in self.widgets_dict.items():
            checkbox.value = False
        self.croptypes = pd.DataFrame()
        with self.output:
            self.output.clear_output()
            print("Selection cleared.")

    def _apply_hierarchy_on_selection(self, paths_to_search):
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

        self.croptypes = croptype_df

        final_types = np.unique(self.croptypes["new_label"].values)

        with self.output:
            self.output.clear_output()
            print(
                f"Selected {len(self.croptypes)} crop types, aggregated to {len(final_types)} classes:"
            )
            for ct in final_types:
                print(f"- {ct}")

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

        result = clean_hierarchy_keys(hierarchy)

        return result


def clean_hierarchy_keys(hierarchy):

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
                        full_name = translate_ewoc_codes([code])["label_full"].values[0]
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

    if croptypepicker.croptypes.empty:
        raise ValueError("No crop types selected, cannot proceed.")

    # Isolate all crop types that have NOT been selected
    included = croptypepicker.croptypes.index.values
    excluded = df[~df["ewoc_code"].isin(included)]

    # Prepare a mapping dictionary from original labels (index) to new labels
    label_mapping = croptypepicker.croptypes["new_label"].to_dict()

    # Apply the mapping to the ewoc_code column
    df["downstream_class"] = df["ewoc_code"].map(label_mapping)

    # Excluded crop types are assigned to "other" class
    df.loc[excluded.index, "downstream_class"] = "other"

    return df
