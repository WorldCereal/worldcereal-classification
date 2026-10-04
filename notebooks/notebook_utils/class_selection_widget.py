"""Notebook-friendly crop class selection tool for WorldCereal workflows.

This widget replaces the old Include/Drop -> Other inspection -> Combine flow
with one explicit mapping from source EWOC classes to a final class list.

The source classes are displayed using the hierarchy already encoded in the
WorldCereal legend (``level_1`` ... ``level_5``). Intermediate legend nodes are
shown as collapsible structural parents; only EWOC classes actually available in
``sample_df`` (or explicitly supplied through ``ewoc_codes``) are selectable.

Typical use
-----------
    from class_selection_widget import ClassSelectionWidget

    selector = ClassSelectionWidget(sample_df=aligned_df)
    # interact with the widget, then:
    selector.apply_selection()
    selected_df = selector.apply_to_df(aligned_df)

Design principles
-----------------
* Every source class starts **not selected**; nothing is assumed about the
    final class list until the user decides.
* The WorldCereal legend hierarchy guides users when composing broader groups.
* Users explicitly select, combine, or leave out classes. Bulk actions let the
    remaining not-selected classes be kept individually or grouped into a
    single class in one step.
* No implicit ``other`` class is created automatically; any "everything else"
  class is the result of an explicit bulk action.
* The resulting mapping directly produces the ``downstream_class`` column used
  by the WorldCereal classification workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

try:
    from worldcereal.utils.legend import get_legend
except ImportError:  # keeps the module importable for lightweight UI prototyping
    get_legend = None


DEMO_CROPS = [
    1100000000, 1101000000, 1101000001, 1101000002, 1101010000,
    1101010001, 1101010002, 1101020000, 1101030000, 1101060000,
    1101070000, 1101080000, 1101120000, 1103000000, 1103060040,
    1103080080, 1103090040, 1103110040, 1105000000, 1105000030,
    1105010000, 1105010010, 1105010040, 1106000000, 1106000010,
    1106000020, 1106000030, 1107000000, 1107000010, 1107000040,
    1108000000, 1109000000, 1110000000, 1111000000, 1111020000,
    1111020010, 1114000000, 1115000000, 1200000000,
    1201010020, 1201000010, 1203000000, 1203000030, 2000000000,
    2001000000, 2002000000, 2500000000, 3000000000, 4000000000,
    4100000000, 4200000000, 5000000000, 6000000000, 7000000000,
]

EXCLUDED = "__EXCLUDED__"
LEGEND_LEVELS = [f"level_{i}" for i in range(1, 6)]


@dataclass(frozen=True)
class SourceClass:
    """One EWOC source class available to the selector."""

    ewoc_code: int
    label: str
    count: int
    hierarchy_path: Tuple[str, ...]


class ClassSelectionWidget:
    """Build a final class list from source EWOC classes in one step.

    Parameters
    ----------
    sample_df : pandas.DataFrame, optional
        DataFrame with an ``ewoc_code`` column. Sample counts are derived from it.
    ewoc_codes : iterable of int, optional
        Explicit classes to show when no sample dataframe is supplied.
        When both ``sample_df`` and ``ewoc_codes`` are omitted, the full
        WorldCereal legend (via ``get_legend()``) is used instead.
    count_threshold : int, default 0
        Hide source classes with fewer samples than this threshold.
    labels : dict[int, str], optional
        Explicit EWOC-code-to-label mapping. Primarily useful in demos/tests.
    hierarchy_paths : dict[int, sequence[str]], optional
        Explicit hierarchy path per EWOC code. Primarily useful in demos/tests.
        During normal WorldCereal use the paths come from ``level_1`` ...
        ``level_5`` of ``get_legend()``.
    legend : pandas.DataFrame, optional
        Preloaded WorldCereal legend. When omitted, ``get_legend()`` is used.
        Supplying it is useful for tests or to avoid another legend lookup.
    display_ui : bool, default True
        Display the widget immediately.

    Notes
    -----
    Every source class starts outside the final list. The user explicitly
    decides which classes to keep, combine, move, or leave out.
    """

    def __init__(
        self,
        sample_df: Optional[pd.DataFrame] = None,
        ewoc_codes: Optional[Iterable[int]] = None,
        count_threshold: int = 0,
        labels: Optional[Mapping[int, str]] = None,
        hierarchy_paths: Optional[Mapping[int, Sequence[str]]] = None,
        legend: Optional[pd.DataFrame] = None,
        display_ui: bool = True,
    ):
        if sample_df is not None and "ewoc_code" not in sample_df.columns:
            raise ValueError("sample_df must contain an 'ewoc_code' column.")

        self.sample_df = sample_df
        self.count_threshold = int(count_threshold)
        self._explicit_labels = {int(k): str(v) for k, v in (labels or {}).items()}
        self._explicit_paths = {
            int(k): tuple(str(part) for part in path if str(part).strip())
            for k, path in (hierarchy_paths or {}).items()
        }
        self._legend = legend.copy() if legend is not None else self._load_legend()

        if sample_df is None and ewoc_codes is None:
            # No explicit filter: fall back to the full WorldCereal legend so
            # e.g. "no crop-only filtering" still yields a usable widget.
            if self._legend is None:
                raise ValueError(
                    "Provide sample_df or ewoc_codes, or ensure get_legend() is available."
                )
            ewoc_codes = list(self._legend.index)

        self.sources = self._build_source_classes(sample_df, ewoc_codes)
        if not self.sources:
            raise ValueError("No classes remain after applying the filters.")

        # One explicit assignment per EWOC code. Every class starts as EXCLUDED
        # (not selected); the user opts individual classes in explicitly.
        self.assignments: Dict[int, str] = {
            source.ewoc_code: EXCLUDED for source in self.sources
        }

        # Public outputs, deliberately shaped like the old picker where practical.
        self.croptypes = pd.DataFrame()
        self.included_croptypes = pd.DataFrame()
        self.dropped_croptypes = pd.DataFrame()

        # Source-tree state is kept separately from the widgets so searches and
        # redraws do not lose selections or expansion state.
        self._selected_source_codes: set[int] = set()
        self._expanded_paths: set[Tuple[str, ...]] = set()
        self._source_checkboxes: Dict[int, widgets.Checkbox] = {}
        self._branch_toggles: Dict[Tuple[str, ...], widgets.ToggleButton] = {}
        self._active_branch_path: Optional[Tuple[str, ...]] = None
        self._active_branch_label: Optional[str] = None
        self._active_branch_codes: Tuple[int, ...] = tuple()
        self._expanded_group_names: set[str] = set()
        self._last_auto_group_name: str = ""
        self._tree = self._build_hierarchy_tree(self.sources)

        self._build_widget()
        self._initialize_expansion_state()
        self._refresh_all()

        if display_ui:
            display(self.widget)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _load_legend(self) -> Optional[pd.DataFrame]:
        if get_legend is None:
            return None
        try:
            return get_legend().copy()
        except Exception:
            return None

    def _build_source_classes(
        self,
        sample_df: Optional[pd.DataFrame],
        ewoc_codes: Optional[Iterable[int]],
    ) -> List[SourceClass]:
        if sample_df is not None:
            counts = sample_df["ewoc_code"].dropna().astype("int64").value_counts()
            if ewoc_codes is not None:
                allowed = {int(code) for code in ewoc_codes}
                counts = counts[counts.index.map(lambda code: int(code) in allowed)]
            code_counts = {int(code): int(count) for code, count in counts.items()}
        else:
            code_counts = {int(code): 0 for code in ewoc_codes or []}

        code_counts = {
            code: count
            for code, count in code_counts.items()
            if count >= self.count_threshold
        }

        result: List[SourceClass] = []
        for code, count in code_counts.items():
            label = self._resolve_label(code)
            path = self._resolve_hierarchy_path(code, label)
            result.append(SourceClass(code, label, count, path))

        # Sort by hierarchy path so the source tree follows the legend structure.
        return sorted(
            result,
            key=lambda item: tuple(part.casefold() for part in item.hierarchy_path),
        )

    def _legend_row(self, code: int) -> Optional[pd.Series]:
        if self._legend is None or code not in self._legend.index:
            return None
        row = self._legend.loc[code]
        # Defensive handling in case a malformed legend contains duplicate codes.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    @staticmethod
    def _clean_legend_value(value) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _deduplicate_path(parts: Iterable[str]) -> Tuple[str, ...]:
        clean: List[str] = []
        for part in parts:
            text = str(part).strip()
            if not text:
                continue
            # The legend can repeat a label on adjacent levels. Showing it once
            # is enough to preserve the hierarchy while avoiding visual noise.
            if not clean or clean[-1] != text:
                clean.append(text)
        return tuple(clean)

    def _resolve_label(self, code: int) -> str:
        if code in self._explicit_labels:
            return self._explicit_labels[code]

        row = self._legend_row(code)
        if row is not None:
            label = self._clean_legend_value(row.get("label_full"))
            if label:
                return label

        return str(code)

    def _resolve_hierarchy_path(self, code: int, label: str) -> Tuple[str, ...]:
        if code in self._explicit_paths:
            path = self._deduplicate_path(self._explicit_paths[code])
            return path or (label,)

        row = self._legend_row(code)
        if row is None:
            return (label,)

        parts: List[str] = []
        for column in LEGEND_LEVELS:
            if column not in row.index:
                continue
            value = self._clean_legend_value(row.get(column))
            if value:
                parts.append(value)

        path = self._deduplicate_path(parts)
        if not path:
            return (label,)

        # Usually label_full is already the deepest legend level. If it is not,
        # append it so the source class still has a visible leaf node.
        if label not in path:
            path = (*path, label)
        return path

    @staticmethod
    def _empty_tree_node(label: Optional[str] = None) -> dict:
        return {
            "label": label,
            "children": {},
            "source_codes": [],
            "descendant_codes": set(),
            "sample_count": 0,
        }

    def _build_hierarchy_tree(self, sources: Sequence[SourceClass]) -> dict:
        root = self._empty_tree_node()
        for source in sources:
            node = root
            for part in source.hierarchy_path:
                node = node["children"].setdefault(part, self._empty_tree_node(part))
            node["source_codes"].append(source.ewoc_code)

        lookup = {source.ewoc_code: source for source in sources}

        def aggregate(node: dict) -> set[int]:
            codes = set(node["source_codes"])
            for child in node["children"].values():
                codes.update(aggregate(child))
            node["descendant_codes"] = codes
            node["sample_count"] = sum(lookup[code].count for code in codes)
            return codes

        aggregate(root)
        return root

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_widget(self) -> None:
        """Build the notebook UI around a stable source -> result composition."""
        layout_css = widgets.HTML(
            """<style>
            .wc-class-selection-widget,
            .wc-class-selection-widget * { box-sizing: border-box; }
            .wc-class-selection-widget .widget-hbox,
            .wc-class-selection-widget .widget-vbox,
            .wc-class-selection-widget .widget-box,
            .wc-class-selection-widget .jupyter-widgets { min-width: 0 !important; max-width: 100%; }
            .wc-class-selection-widget .widget-label {
                white-space: normal !important;
                overflow-wrap: anywhere;
            }
            .wc-class-selection-widget select,
            .wc-class-selection-widget input { max-width: 100%; }
            .wc-class-selection-widget .wc-source-scroll {
                overflow-x: auto !important;
                overflow-y: scroll !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: stretch !important;
            }
            .wc-class-selection-widget .wc-source-scroll > * {
                flex: 0 0 auto !important;
                flex-shrink: 0 !important;
            }
            /* Let long source rows grow past the box width instead of wrapping,
               so a horizontal scrollbar appears only when it's actually needed. */
            .wc-class-selection-widget .wc-source-scroll .widget-vbox,
            .wc-class-selection-widget .wc-source-scroll .widget-hbox,
            .wc-class-selection-widget .wc-source-scroll .jupyter-widgets {
                min-width: 100% !important;
                width: max-content !important;
                max-width: none !important;
            }
            .wc-class-selection-widget .wc-source-scroll .widget-label {
                white-space: nowrap !important;
                overflow-wrap: normal !important;
            }
            .wc-class-selection-widget .wc-groups-scroll {
                overflow-x: hidden !important;
                overflow-y: auto !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: stretch !important;
            }
            .wc-class-selection-widget .wc-groups-scroll > * {
                flex: 0 0 auto !important;
                flex-shrink: 0 !important;
            }
            .wc-class-selection-widget .wc-groups-scroll > .widget-vbox { max-width: 100% !important; }

            /* Make the legend hierarchy visually legible without making leaf classes noisy. */
            .wc-class-selection-widget .wc-tree-children {
                border-left: 2px solid #d9dee5;
                padding-left: 8px !important;
            }
            .wc-class-selection-widget .wc-tree-level-1 {
                font-weight: 700 !important;
                border-left: 4px solid #315c3a !important;
            }
            .wc-class-selection-widget .wc-tree-level-2 {
                font-weight: 650 !important;
                border-left: 4px solid #698369 !important;
            }
            .wc-class-selection-widget .wc-tree-level-3 {
                font-weight: 600 !important;
                border-left: 4px solid #b39a45 !important;
            }
            .wc-class-selection-widget .wc-tree-level-4,
            .wc-class-selection-widget .wc-tree-level-5 {
                font-weight: 550 !important;
                border-left: 4px solid #c9c1a0 !important;
            }
            .wc-class-selection-widget .wc-source-leaf {
                padding: 2px 0 !important;
            }

            /* Distinct functional zones: source -> selection/actions -> result. */
            .wc-class-selection-widget .wc-section-card {
                border: 1px solid #d9dee5;
                border-radius: 6px;
                padding: 10px !important;
                background: #ffffff;
            }
            .wc-class-selection-widget .wc-section-source {
                border-top: 4px solid #64748b;
            }
            .wc-class-selection-widget .wc-section-selection {
                border-left: 4px solid #4f83b8;
                background: #f8fbff;
            }
            .wc-class-selection-widget .wc-section-actions {
                border-left: 4px solid #b58a35;
                background: #fffdf7;
            }
            .wc-class-selection-widget .wc-section-final {
                border-top: 4px solid #4f6f52;
                background: #fbfdfb;
            }
            .wc-class-selection-widget .wc-section-title {
                margin: 0 0 4px 0;
                font-size: 15px;
                font-weight: 700;
            }
            </style>"""
        )

        intro_title = widgets.HTML(
            "<div style='box-sizing:border-box;width:100%;margin:0 0 4px 0;padding:10px 12px;border-left:4px solid #4f6f52;background:#f7faf7'>"
            "<b>Class selection tool</b>"
            "</div>"
        )
        intro_info = widgets.HTML(
            "<div style='box-sizing:border-box;width:100%;max-width:100%;"
            "background:#f6f7f9;border:1px solid #e3e6ea;border-left:4px solid #c7cdd4;"
            "padding:6px 8px;border-radius:4px;color:#4b5563;font-size:13px;"
            "line-height:1.4;overflow-wrap:anywhere;margin:6px 0 8px 0'>"
            "Every source class starts <b>not selected</b>.<br>"
            "Select individual classes (checkboxes) or class groups (buttons) on the left and choose an action on the right to move them to the final selection.<br>"
            "You can see the final classes on the bottom and have the option to edit them as needed there.<br>"
            "Use the <i>Not selected</i> section in the Final classes overview for bulk actions, including merging everything "
            "you haven't touched into one final class."
            "</div>"
        )
        intro_info.layout.display = "none"
        intro_toggle = widgets.ToggleButton(
            value=False,
            description="More information",
            icon="info-circle",
            layout=widgets.Layout(width="170px", margin="0 0 8px 0"),
        )

        def _toggle_intro(change):
            intro_info.layout.display = "block" if change["new"] else "none"

        intro_toggle.observe(_toggle_intro, names="value")
        intro = widgets.VBox(
            [intro_title, intro_toggle, intro_info],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )

        self.search = widgets.Text(
            placeholder="Search source classes...",
            description="Search:",
            continuous_update=True,
            style={"description_width": "55px"},
            layout=widgets.Layout(width="100%", min_width="0"),
        )
        self.search.observe(lambda _: self._refresh_source_tree(), names="value")
        self.source_status_filter = widgets.Dropdown(
            options=[
                ("All", "all"),
                ("Not selected", "not_selected"),
                ("Selected", "selected"),
            ],
            value="all",
            description="Show:",
            style={"description_width": "45px"},
            layout=widgets.Layout(width="100%", min_width="0"),
        )
        self.source_status_filter.observe(
            lambda _: self._refresh_source_tree(), names="value"
        )

        self.expand_all_button = widgets.Button(
            description="Expand all",
            icon="plus-square",
            layout=widgets.Layout(width="118px", margin="0 6px 6px 0"),
        )
        self.collapse_all_button = widgets.Button(
            description="Collapse all",
            icon="minus-square",
            layout=widgets.Layout(width="125px", margin="0 6px 6px 0"),
        )
        self.clear_source_selection_button = widgets.Button(
            description="Clear",
            icon="eraser",
            layout=widgets.Layout(width="90px", margin="0 0 6px 0"),
        )
        self.expand_all_button.on_click(self._on_expand_all)
        self.collapse_all_button.on_click(self._on_collapse_all)
        self.clear_source_selection_button.on_click(self._on_clear_source_selection)

        self.source_tree_box = widgets.VBox(
            [],
            layout=widgets.Layout(
                width="100%",
                height="280px",
                min_height="280px",
                max_height="280px",
                overflow="auto auto",
                border="1px solid #d1d5db",
                padding="6px 8px",
                min_width="0",
            ),
        )
        self.source_tree_box.add_class("wc-source-scroll")
        self.source_selection_status = widgets.HTML()

        # Controls used by the context-sensitive action area below the tree.
        self.group_dropdown = widgets.Dropdown(
            options=[],
            description="Target class:",
            style={"description_width": "85px"},
            layout=widgets.Layout(width="200px", min_width="0"),
        )
        self.new_group_name = widgets.Text(
            placeholder="e.g. wheat",
            description="Class name:",
            style={"description_width": "85px"},
            layout=widgets.Layout(width="200px", min_width="0"),
        )
        self.assign_button = widgets.Button(
            description="Merge",
            icon="arrow-right",
            button_style="primary",
            layout=widgets.Layout(width="100px", margin="0 6px 6px 0"),
        )
        self.new_group_button = widgets.Button(
            description="Combine",
            icon="object-group",
            button_style="success",
            layout=widgets.Layout(width="100px", margin="0 6px 6px 0"),
        )
        self.exclude_button = widgets.Button(
            description="Remove",
            icon="trash",
            button_style="danger",
            layout=widgets.Layout(width="100px", margin="0 6px 6px 0"),
        )
        self.keep_individual_button = widgets.Button(
            description="Keep as individual class",
            icon="user",
            button_style="info",
            layout=widgets.Layout(width="185px", margin="0 6px 6px 0"),
        )
        self.assign_button.on_click(self._on_assign)
        self.new_group_button.on_click(self._on_new_group)
        self.exclude_button.on_click(self._on_exclude)
        self.keep_individual_button.on_click(self._on_keep_individual)

        self.branch_context = widgets.HTML()
        self.branch_select_button = widgets.Button(
            description="Select classes in branch",
            icon="check-square-o",
            layout=widgets.Layout(width="185px", margin="0 6px 6px 0"),
        )
        self.branch_group_button = widgets.Button(
            description="Create hierarchy class",
            icon="object-group",
            layout=widgets.Layout(width="205px", margin="0 0 6px 0"),
        )
        self.branch_select_button.on_click(self._on_select_active_branch)
        self.branch_group_button.on_click(self._on_group_active_branch)

        self.context_actions_box = widgets.VBox(
            [],
            layout=widgets.Layout(
                width="100%",
                min_width="0",
                padding="2px 0 0 0",
                overflow="hidden",
            ),
        )

        # Bulk actions for classes that have not been explicitly handled yet.
        # Rendered inside the "Final classes" panel, below a divider.
        self.bulk_label_input = widgets.Text(
            placeholder="e.g. other",
            description="Class name:",
            style={"description_width": "85px"},
            layout=widgets.Layout(width="200px", min_width="0"),
        )
        self.bulk_assign_button = widgets.Button(
            description="Merge all",
            icon="object-group",
            button_style="success",
            layout=widgets.Layout(width="110px", margin="0 6px 6px 0"),
        )
        self.bulk_keep_individual_button = widgets.Button(
            description="Select all",
            icon="list",
            layout=widgets.Layout(width="140px", margin="0 6px 6px 0"),
        )
        self.bulk_keep_individual_button.on_click(self._on_bulk_keep_individual)
        self.bulk_assign_button.on_click(self._on_bulk_assign)

        # Final groups are rendered as an editable live result instead of through
        # a separate group-selection dropdown.
        # Keep the final-class list scrollable, but keep its compact totals
        # visible as a footer inside the same result panel.
        self.final_groups_list_box = widgets.VBox(
            [],
            layout=widgets.Layout(
                width="100%",
                height="492px",
                overflow="hidden auto",
                padding="6px 8px",
                min_width="0",
            ),
        )
        self.final_groups_list_box.add_class("wc-groups-scroll")

        self.final_groups_box = widgets.VBox(
            [self.final_groups_list_box],
            layout=widgets.Layout(
                width="100%",
                height="500px",
                border="1px solid #d1d5db",
                min_width="0",
                overflow="hidden",
                align_items="stretch",
            ),
        )
        self.message = widgets.HTML()
        self.apply_button = widgets.Button(
            description="Use these final classes",
            icon="check",
            button_style="success",
            layout=widgets.Layout(width="240px", height="44px", margin="0 8px 6px 0"),
        )
        self.reset_button = widgets.Button(
            description="Reset",
            icon="refresh",
            layout=widgets.Layout(width="120px", height="44px", margin="0 0 6px 0"),
        )
        self.apply_button.on_click(self._on_apply)
        self.reset_button.on_click(self._on_reset)

        source_toolbar = widgets.HBox(
            [
                self.expand_all_button,
                self.collapse_all_button,
            ],
            layout=widgets.Layout(
                width="100%",
                flex_flow="row wrap",
                overflow="hidden",
                align_items="flex-start",
            ),
        )

        source_panel = widgets.VBox(
            [
                widgets.HTML(
                    "<div class='wc-section-title'>Source classes</div>"
                    "<div style='font-size:12px;color:#666;margin-bottom:6px'>"
                    "Browse the available classes below. Select the class(es) you want to group/remove.<br>"
                    "Alternatively, click a group (button) such as <i>cereals</i> to use it as a shortcut."
                    "</div>"
                ),
                self.search,
                self.source_status_filter,
                source_toolbar,
                self.source_tree_box,
            ],
            layout=widgets.Layout(
                width="100%", min_width="0", overflow="hidden", align_items="stretch", flex="0 0 auto"
            ),
        )
        source_panel.add_class("wc-section-card")
        source_panel.add_class("wc-section-source")

        selection_panel = widgets.VBox(
            [   
                widgets.HTML(
                                "<div class='wc-section-title'>Selection</div>"
                            ),
                self.source_selection_status,
            ],
            layout=widgets.Layout(
                width="100%", min_width="0", overflow="hidden", align_items="stretch", flex="0 0 auto"
            ),
        )
        selection_panel.add_class("wc-section-card")
        selection_panel.add_class("wc-section-selection")

        actions_panel = widgets.VBox(
            [
                widgets.HTML(
                    "<div class='wc-section-title'>Actions</div>"
                ),
                self.context_actions_box,
            ],
            layout=widgets.Layout(
                width="100%", min_width="0", overflow="auto", align_items="stretch", flex="1 1 auto"
            ),
        )
        actions_panel.add_class("wc-section-card")
        actions_panel.add_class("wc-section-actions")

        side_column = widgets.VBox(
            [selection_panel, actions_panel],
            layout=widgets.Layout(
                width="100%",
                min_width="0",
                overflow="auto",
                align_items="stretch",
                gap="10px",
                display="flex",
                flex_direction="column",
            ),
        )

        top_row = widgets.GridBox(
            [source_panel, side_column],
            layout=widgets.Layout(
                width="100%",
                grid_template_columns="minmax(0, 62%) minmax(0, 38%)",
                grid_gap="14px",
                align_items="flex-start",
                overflow="visible",
            ),
        )

        final_groups_panel = widgets.VBox(
            [
                widgets.HTML(
                    "<div class='wc-section-title'>Final classes</div>"
                    "<div style='font-size:12px;color:#666;margin-bottom:6px'>"
                    "Click a class to inspect, edit, or leave it out."
                    "</div>"
                ),
                self.final_groups_box,
            ],
            layout=widgets.Layout(
                width="100%", min_width="0", overflow="hidden", align_items="stretch"
            ),
        )
        final_groups_panel.add_class("wc-section-card")
        final_groups_panel.add_class("wc-section-final")

        comparison_section = widgets.VBox(
            [top_row, final_groups_panel],
            layout=widgets.Layout(
                width="100%",
                min_width="0",
                overflow="visible",
                align_items="stretch",
                gap="14px",
            ),
        )

        final_actions = widgets.VBox(
            [
                self.message,
                widgets.HBox(
                    [self.apply_button, self.reset_button],
                    layout=widgets.Layout(
                        width="100%",
                        flex_flow="row wrap",
                        overflow="hidden",
                        align_items="flex-start",
                    ),
                ),
            ],
            layout=widgets.Layout(
                width="100%", min_width="0", overflow="hidden", margin="10px 0 0 0"
            ),
        )

        self.widget = widgets.VBox(
            [layout_css, intro, comparison_section, final_actions],
            layout=widgets.Layout(width="100%", min_width="0", overflow="hidden"),
        )
        self.widget.add_class("wc-class-selection-widget")

    # ------------------------------------------------------------------
    # Source hierarchy rendering
    # ------------------------------------------------------------------
    @property
    def _source_lookup(self) -> Dict[int, SourceClass]:
        return {source.ewoc_code: source for source in self.sources}

    def _initialize_expansion_state(self) -> None:
        root_children = list(self._tree["children"].items())
        if len(root_children) == 1:
            label, _ = root_children[0]
            self._expanded_paths.add((label,))

    def _selected_codes(self) -> List[int]:
        return sorted(self._selected_source_codes)

    def _groups(self) -> List[str]:
        return sorted(
            {group for group in self.assignments.values() if group != EXCLUDED},
            key=str.casefold,
        )

    def _not_selected_codes(self) -> List[int]:
        return sorted(
            code for code, group in self.assignments.items() if group == EXCLUDED
        )

    def _source_matches_search(self, source: SourceClass, term: str) -> bool:
        if not term:
            return True
        haystack = " ".join(
            [source.label, str(source.ewoc_code), *source.hierarchy_path]
        ).casefold()
        return term in haystack

    def _visible_codes_for_search(self) -> set[int]:
        term = self.search.value.strip().casefold()
        status = self.source_status_filter.value
        return {
            source.ewoc_code
            for source in self.sources
            if self._source_matches_search(source, term)
            and (
                status == "all"
                or (status == "not_selected" and self.assignments[source.ewoc_code] == EXCLUDED)
                or (status == "selected" and self.assignments[source.ewoc_code] != EXCLUDED)
            )
        }

    def _format_source_assignment(self, code: int) -> str:
        source = self._source_lookup[code]
        group = self.assignments[code]
        count_part = f"  ({source.count:,})" if self.sample_df is not None else ""
        if group == EXCLUDED:
            return f"{source.label}{count_part}  · not selected"
        return f"{source.label}{count_part}  → {group}"

    def _make_source_checkbox(self, code: int) -> widgets.Checkbox:
        checkbox = widgets.Checkbox(
            value=code in self._selected_source_codes,
            description=self._format_source_assignment(code),
            indent=False,
            layout=widgets.Layout(
                width="100%", max_width="100%", overflow="hidden", flex="0 0 auto"
            ),
        )

        def on_change(change, ewoc_code=code):
            if change["new"]:
                self._selected_source_codes.add(ewoc_code)
            else:
                self._selected_source_codes.discard(ewoc_code)
            self._refresh_source_selection_status()
            self._refresh_context_actions()

        checkbox.observe(on_change, names="value")
        checkbox.add_class("wc-source-leaf")
        self._source_checkboxes[code] = checkbox
        return checkbox

    def _make_branch_toggle(
        self,
        path: Tuple[str, ...],
        label: str,
        node: dict,
        children_box: widgets.VBox,
        force_expanded: bool,
        on_first_expand: Optional[Callable[[], None]] = None,
    ) -> widgets.ToggleButton:
        source_count = len(node["descendant_codes"])
        sample_count = node["sample_count"]
        if self.sample_df is not None:
            suffix = f"{source_count} classes, {sample_count:,} samples"
        else:
            suffix = f"{source_count} classes"

        expanded = force_expanded or path in self._expanded_paths
        toggle = widgets.ToggleButton(
            value=expanded,
            description=f"{label}  ({suffix})",
            icon="chevron-down" if expanded else "chevron-right",
            tooltip="Click to expand/collapse and make this hierarchy branch active",
            layout=widgets.Layout(width="100%", max_width="100%", overflow="hidden"),
        )
        depth = min(len(path), 5)
        level_colors = {
            1: "#e2ebe3",
            2: "#edf3e9",
            3: "#f6f1d9",
            4: "#faf7e9",
            5: "#fbfbf8",
        }
        toggle.style.button_color = (
            "#dbeafe"
            if path == self._active_branch_path
            else level_colors[depth]
        )
        toggle.add_class(f"wc-tree-level-{depth}")
        children_box.layout.display = "flex" if expanded else "none"

        def on_toggle(change, current_path=path, branch_label=label, branch_node=node):
            is_open = bool(change["new"])
            if is_open:
                self._expanded_paths.add(current_path)
                if on_first_expand is not None:
                    on_first_expand()
            else:
                self._expanded_paths.discard(current_path)
            toggle.icon = "chevron-down" if is_open else "chevron-right"
            children_box.layout.display = "flex" if is_open else "none"
            self._set_active_branch(current_path, branch_label, branch_node)

        toggle.observe(on_toggle, names="value")
        self._branch_toggles[path] = toggle
        return toggle

    def _set_active_branch(
        self,
        path: Tuple[str, ...],
        label: str,
        node: dict,
    ) -> None:
        self._active_branch_path = path
        self._active_branch_label = label
        self._active_branch_codes = tuple(sorted(node["descendant_codes"]))
        level_colors = {
            1: "#e2ebe3",
            2: "#edf3e9",
            3: "#f6f1d9",
            4: "#faf7e9",
            5: "#fbfbf8",
        }
        for toggle_path, toggle in self._branch_toggles.items():
            depth = min(len(toggle_path), 5)
            toggle.style.button_color = (
                "#dbeafe" if toggle_path == path else level_colors[depth]
            )
        self._refresh_source_selection_status()
        self._refresh_context_actions()

    def _render_tree_node(
        self,
        label: str,
        node: dict,
        path: Tuple[str, ...],
        visible_codes: set[int],
        search_active: bool,
    ) -> Optional[widgets.Widget]:
        # descendant_codes/sample_count are pre-aggregated on the tree, so we can
        # decide visibility and the checkbox-collapse shortcut without touching
        # (let alone building widgets for) any collapsed descendant branch.
        subtree_visible = node["descendant_codes"] & visible_codes
        if not subtree_visible:
            return None

        direct_visible = [c for c in node["source_codes"] if c in visible_codes]
        has_visible_children = any(
            child["descendant_codes"] & visible_codes
            for child in node["children"].values()
        )
        if direct_visible and not has_visible_children and len(direct_visible) == 1:
            return self._make_source_checkbox(direct_visible[0])

        indent_px = 26
        children_box = widgets.VBox(
            [],
            layout=widgets.Layout(
                width=f"calc(100% - {indent_px}px)",
                margin=f"3px 0 4px {indent_px}px",
                min_width="0",
                overflow="hidden",
                align_items="flex-start",
            ),
        )
        children_box.add_class("wc-tree-children")

        # Building all descendant checkboxes/toggles up front does not scale to
        # the full legend. Only build a branch's children the first time it is
        # actually expanded (either already-expanded, forced by search, or on
        # first click of its toggle).
        built = False

        def build_children() -> None:
            nonlocal built
            if built:
                return
            built = True
            child_widgets: List[widgets.Widget] = []
            for code in node["source_codes"]:
                if code in visible_codes:
                    child_widgets.append(self._make_source_checkbox(code))
            for child_label, child_node in node["children"].items():
                rendered = self._render_tree_node(
                    child_label,
                    child_node,
                    (*path, child_label),
                    visible_codes,
                    search_active,
                )
                if rendered is not None:
                    child_widgets.append(rendered)
            children_box.children = child_widgets

        expanded = search_active or path in self._expanded_paths
        if expanded:
            build_children()

        toggle = self._make_branch_toggle(
            path=path,
            label=label,
            node=node,
            children_box=children_box,
            force_expanded=search_active,
            on_first_expand=build_children,
        )
        return widgets.VBox(
            [toggle, children_box],
            layout=widgets.Layout(
                width="100%",
                min_width="0",
                overflow="hidden",
                align_items="flex-start",
                flex="0 0 auto",
            ),
        )

    def _refresh_source_tree(self) -> None:
        visible_codes = self._visible_codes_for_search()
        search_active = bool(self.search.value.strip())
        self._source_checkboxes = {}
        self._branch_toggles = {}

        root_widgets: List[widgets.Widget] = []
        for label, node in self._tree["children"].items():
            rendered = self._render_tree_node(
                label,
                node,
                (label,),
                visible_codes,
                search_active,
            )
            if rendered is not None:
                root_widgets.append(rendered)

        if not root_widgets:
            root_widgets = [widgets.HTML("<i>No source classes match the search.</i>")]
        self.source_tree_box.children = root_widgets
        self._refresh_source_selection_status()
        self._refresh_context_actions()

    def _refresh_source_selection_status(self) -> None:
        """Describe the current source-class or hierarchy selection."""
        selected = self._selected_codes()
        lookup = self._source_lookup

        if selected:
            labels = [lookup[code].label for code in selected]
            preview = ", ".join(escape(label) for label in labels[:4])
            if len(labels) > 4:
                preview += f", +{len(labels) - 4} more"
            header = (
                f"<b>{len(selected)} source class"
                f"{'es' if len(selected) != 1 else ''}</b> selected"
            )
            if self.sample_df is not None:
                sample_count = sum(lookup[code].count for code in selected)
                header += f" <span style='color:#666'>· {sample_count:,} samples</span>"
            self.source_selection_status.value = (
                f"<div style='font-size:13px'>{header}<br>"
                f"<span style='font-size:12px;color:#666'>{preview}</span></div>"
            )
            return

        if self._active_branch_label and self._active_branch_codes:
            codes = [code for code in self._active_branch_codes if code in self.assignments]
            labels = [lookup[code].label for code in codes]
            preview = ", ".join(escape(label) for label in labels[:4])
            if len(labels) > 4:
                preview += f", +{len(labels) - 4} more"
            count_line = f"{len(codes)} source class{'es' if len(codes) != 1 else ''}"
            if self.sample_df is not None:
                sample_count = sum(lookup[code].count for code in codes)
                count_line += f" · {sample_count:,} samples"
            self.source_selection_status.value = (
                f"<div style='font-size:13px'><b>Group: "
                f"{escape(self._active_branch_label)}</b><br>"
                f"<span style='font-size:12px;color:#666'>{count_line}</span><br>"
                f"<span style='font-size:12px;color:#666'>{preview}</span></div>"
            )
            return

        self.source_selection_status.value = (
            "<div style='font-size:12px;color:#666'>"
            "Nothing selected yet. Select source classes with the checkboxes, or click a group."
            "</div>"
        )

    def _suggest_group_name(self, codes: Sequence[int]) -> str:
        """Suggest the deepest common legend parent for a multi-selection."""
        if len(codes) < 2:
            return ""
        paths = [self._source_lookup[int(code)].hierarchy_path for code in codes]
        common: List[str] = []
        for parts in zip(*paths):
            if len(set(parts)) == 1:
                common.append(parts[0])
            else:
                break
        if not common:
            return ""
        candidate = common[-1]
        source_labels = {self._source_lookup[int(code)].label for code in codes}
        return "" if candidate in source_labels else candidate

    def _secondary_selection_actions(
        self, selected: Sequence[int]
    ) -> List[widgets.Widget]:
        """Return the "move to an existing final class" action, when applicable."""
        children: List[widgets.Widget] = []

        if self._groups():
            children.extend(
                [
                    widgets.HTML(
                        "<div style='font-size:12px;color:#555;margin:8px 0 4px 0'>"
                        "<b>Move to an existing final class</b><br>"
                        "</div>"
                    ),
                    self.group_dropdown,
                    widgets.HBox(
                        [self.assign_button],
                        layout=widgets.Layout(width="100%", overflow="hidden"),
                    ),
                ]
            )

        return children

    def _refresh_context_actions(self) -> None:
        """Show all relevant actions for the current selection, with concise guidance."""
        children: List[widgets.Widget] = []
        selected = self._selected_codes()
        lookup = self._source_lookup

        # Individual class selection takes precedence over an older active branch,
        # avoiding two competing sets of actions on screen at the same time.
        if selected:
            labels = [lookup[code].label for code in selected]
            preview = ", ".join(escape(label) for label in labels[:3])
            if len(labels) > 3:
                preview += f", +{len(labels) - 3} more"

            if len(selected) == 1:
                code = selected[0]
                current = self.assignments[code]
                is_excluded = current == EXCLUDED

                if is_excluded:
                    children.extend(
                        [
                            widgets.HTML(
                                "<div style='font-size:12px;color:#555;margin:0 0 4px 0'>"
                                "<b>Keep as individual final class</b><br>"
                                "</div>"
                            ),
                            widgets.HBox(
                                [self.keep_individual_button],
                                layout=widgets.Layout(width="100%", overflow="hidden"),
                            ),
                        ]
                    )
                else:
                    children.extend(
                        [
                            widgets.HTML(
                                "<div style='font-size:12px;color:#555;margin:0 0 4px 0'>"
                                "<b>Leave class out of final list</b><br>"
                                "</div>"
                            ),
                            widgets.HBox(
                                [self.exclude_button],
                                layout=widgets.Layout(width="100%", overflow="hidden"),
                            ),
                        ]
                    )

                # For one class, grouping is intentionally not offered: a class cannot
                # meaningfully be grouped with itself. Moving to an existing class is useful.
                children.extend(self._secondary_selection_actions(selected))
            else:
                suggestion = self._suggest_group_name(selected)
                if not self.new_group_name.value or self.new_group_name.value == self._last_auto_group_name:
                    self.new_group_name.value = suggestion
                    self._last_auto_group_name = suggestion

                any_not_selected = any(self.assignments[c] == EXCLUDED for c in selected)
                all_not_selected = all(self.assignments[c] == EXCLUDED for c in selected)

                children.extend(
                    [
                        widgets.HTML(
                            "<div style='font-size:12px;color:#555;margin:0 0 4px 0'>"
                            "<b>Combine into one final class</b><br>"
                            "</div>"
                        ),
                        self.new_group_name,
                        widgets.HBox(
                            [self.new_group_button],
                            layout=widgets.Layout(width="100%", overflow="hidden"),
                        ),
                    ]
                )

                if any_not_selected:
                    children.extend(
                        [
                            widgets.HTML(
                                "<div style='font-size:12px;color:#555;margin:8px 0 4px 0'>"
                                "<b>Keep each selected class individually</b><br>"
                                "</div>"
                            ),
                            widgets.HBox(
                                [self.keep_individual_button],
                                layout=widgets.Layout(width="100%", overflow="hidden"),
                            ),
                        ]
                    )

                if not all_not_selected:
                    children.extend(
                        [
                            widgets.HTML(
                                "<div style='font-size:12px;color:#555;margin:8px 0 4px 0'>"
                                "<b>Leave out of final list</b><br>"
                                "</div>"
                            ),
                            widgets.HBox(
                                [self.exclude_button],
                                layout=widgets.Layout(width="100%", overflow="hidden"),
                            ),
                        ]
                    )

                children.extend(self._secondary_selection_actions(selected))

        elif self._active_branch_label and self._active_branch_codes:
            codes = [code for code in self._active_branch_codes if code in self.assignments]
            already_grouped = bool(codes) and all(
                self.assignments[code] == self._active_branch_label for code in codes
            )
            member_labels = [lookup[code].label for code in codes]
            preview = ", ".join(escape(label) for label in member_labels[:4])
            if len(member_labels) > 4:
                preview += f", +{len(member_labels) - 4} more"

            if len(codes) >= 2:
                if already_grouped:
                    children.append(
                        widgets.HTML(
                            f"<div style='font-size:12px;color:#166534;margin-bottom:8px'>✓ These classes already form the final class <b>{escape(self._active_branch_label)}</b>.</div>"
                        )
                    )
                else:
                    children.extend(
                        [
                            widgets.HTML(
                                "<div style='font-size:12px;color:#555;margin-bottom:4px'>"
                                "<b>Combine all classes in this group.</b><br>"
                                "</div>"
                            ),
                        ]
                    )
                    self.branch_group_button.description = f'Create "{self._active_branch_label}" class'
                    self.branch_group_button.disabled = False
                    self.branch_group_button.button_style = "success"
                    children.append(
                        widgets.HBox(
                            [self.branch_group_button],
                            layout=widgets.Layout(width="100%", overflow="hidden"),
                        )
                    )

                self.branch_select_button.description = "Select classes"
                self.branch_select_button.disabled = not bool(codes)
                children.extend(
                    [
                        widgets.HTML(
                            "<div style='font-size:12px;color:#555;margin:8px 0 4px 0'>"
                            "<b>Select all classes in this group</b><br>"
                            "</div>"
                        ),
                        widgets.HBox(
                            [self.branch_select_button],
                            layout=widgets.Layout(width="100%", overflow="hidden"),
                        ),
                    ]
                )
            else:
                children.append(
                    widgets.HTML(
                        "<div style='font-size:12px;color:#666'>This group contains only one class, so there is nothing to combine.</div>"
                    )
                )

        else:
            children.append(
                widgets.HTML(
                    "<div style='font-size:12px;color:#666'>"
                    "Actions appear here for the current selection."
                    "</div>"
                )
            )

        self.context_actions_box.children = children

    def _on_select_active_branch(self, _=None) -> None:
        codes = [code for code in self._active_branch_codes if code in self.assignments]
        if not codes:
            self._set_message("No source classes in the active branch.", error=True)
            return
        self._selected_source_codes.update(codes)
        for code, checkbox in self._source_checkboxes.items():
            checkbox.value = code in self._selected_source_codes
        self._refresh_source_selection_status()
        self._refresh_context_actions()

    def _on_group_active_branch(self, _=None) -> None:
        if not self._active_branch_label:
            self._set_message("Select a hierarchy branch first.", error=True)
            return
        self._group_hierarchy_node(
            self._active_branch_label,
            self._active_branch_codes,
        )

    def _on_expand_all(self, _=None) -> None:
        def collect(node: dict, path: Tuple[str, ...]):
            if node["children"]:
                self._expanded_paths.add(path)
                for child_label, child in node["children"].items():
                    collect(child, (*path, child_label))

        for label, node in self._tree["children"].items():
            collect(node, (label,))
        self._refresh_source_tree()

    def _on_collapse_all(self, _=None) -> None:
        self._expanded_paths.clear()
        self._refresh_source_tree()

    def _on_clear_source_selection(self, _=None) -> None:
        self._selected_source_codes.clear()
        for checkbox in self._source_checkboxes.values():
            checkbox.value = False
        self._refresh_source_selection_status()
        self._refresh_context_actions()

    # ------------------------------------------------------------------
    # Final class rendering and summary
    # ------------------------------------------------------------------
    def _refresh_group_controls(self) -> None:
        groups = self._groups()
        previous_target = self.group_dropdown.value
        self.group_dropdown.options = groups
        if previous_target in groups:
            self.group_dropdown.value = previous_target
        self._refresh_final_groups()
        self._refresh_context_actions()

    def _make_final_group_card(self, group: str) -> widgets.Widget:
        lookup = self._source_lookup
        members = [
            code for code, assigned in self.assignments.items() if assigned == group
        ]
        members = sorted(
            members,
            key=lambda c: tuple(part.casefold() for part in lookup[c].hierarchy_path),
        )
        expanded = group in self._expanded_group_names

        count_desc = f"{len(members)} source class{'es' if len(members) != 1 else ''}"
        if self.sample_df is not None:
            sample_count = sum(lookup[code].count for code in members)
            count_desc += f"   ·   {sample_count:,} samples"

        header = widgets.ToggleButton(
            value=expanded,
            description=f"{group}   ·   {count_desc}",
            icon="chevron-down" if expanded else "chevron-right",
            layout=widgets.Layout(width="100%", max_width="100%", overflow="hidden"),
        )
        header.style.button_color = "#f3f4f6"

        member_select = widgets.SelectMultiple(
            options=[
                (
                    f"{lookup[code].label} ({lookup[code].count:,})"
                    if self.sample_df is not None
                    else lookup[code].label,
                    code,
                )
                for code in members
            ],
            rows=min(max(len(members), 2), 6),
            description="",
            layout=widgets.Layout(width="100%", max_width="100%"),
        )
        rename_input = widgets.Text(
            placeholder="New class name",
            description="Rename:",
            style={"description_width": "60px"},
            layout=widgets.Layout(width="100%", min_width="0"),
        )
        rename_button = widgets.Button(
            description="Rename",
            icon="pencil",
            layout=widgets.Layout(width="105px", margin="0 6px 6px 0"),
        )
        split_selected_button = widgets.Button(
            description="Split selected",
            icon="unlink",
            disabled=len(members) <= 1,
            layout=widgets.Layout(width="135px", margin="0 6px 6px 0"),
        )
        split_all_button = widgets.Button(
            description="Split group",
            icon="random",
            disabled=len(members) <= 1,
            layout=widgets.Layout(width="120px", margin="0 6px 6px 0"),
        )
        exclude_group_button = widgets.Button(
            description="Remove class",
            icon="trash",
            button_style="danger",
            tooltip="Remove this entire final class and all of its source classes from the selected list",
            layout=widgets.Layout(width="155px", margin="0 0 6px 0"),
        )

        controls = widgets.VBox(
            [
                member_select,
                rename_input,
                widgets.HBox(
                    [rename_button, split_selected_button, split_all_button, exclude_group_button],
                    layout=widgets.Layout(
                        width="100%",
                        flex_flow="row wrap",
                        overflow="hidden",
                        align_items="flex-start",
                    ),
                ),
            ],
            layout=widgets.Layout(
                width="calc(100% - 12px)",
                margin="4px 0 8px 12px",
                padding="5px 0 0 0",
                display="flex" if expanded else "none",
                overflow="hidden",
                min_width="0",
            ),
        )

        def on_toggle(change, group_name=group):
            is_open = bool(change["new"])
            if is_open:
                self._expanded_group_names.add(group_name)
            else:
                self._expanded_group_names.discard(group_name)
            header.icon = "chevron-down" if is_open else "chevron-right"
            controls.layout.display = "flex" if is_open else "none"

        def on_rename(_button, old_name=group, input_widget=rename_input):
            new_name = input_widget.value.strip()
            if not new_name:
                self._set_message("Provide a new final-class name.", error=True)
                return
            if new_name == EXCLUDED:
                self._set_message("That group name is reserved.", error=True)
                return
            for code, assigned in list(self.assignments.items()):
                if assigned == old_name:
                    self.assignments[code] = new_name
            self._expanded_group_names.discard(old_name)
            self._expanded_group_names.add(new_name)
            self._set_message(f"Renamed '{old_name}' to '{new_name}'.")
            self._refresh_all()

        def split_codes(codes: Sequence[int], group_name=group):
            valid = [int(code) for code in codes if int(code) in self.assignments]
            if not valid:
                self._set_message("Select one or more source classes to split.", error=True)
                return
            for code in valid:
                self.assignments[code] = lookup[code].label
            self._set_message(
                f"Split {len(valid)} source class(es) out of '{group_name}'."
            )
            self._refresh_all()

        def on_split_selected(_button, selector=member_select):
            split_codes(selector.value)

        def on_split_all(_button, group_members=tuple(members)):
            split_codes(group_members)

        def on_exclude_group(_button, group_name=group, group_members=tuple(members)):
            for code in group_members:
                self.assignments[int(code)] = EXCLUDED
            self._expanded_group_names.discard(group_name)
            self._set_message(
                f"Removed final class '{group_name}' and all {len(group_members)} "
                f"of its source class(es) from the final list."
            )
            self._refresh_all()

        header.observe(on_toggle, names="value")
        rename_button.on_click(on_rename)
        split_selected_button.on_click(on_split_selected)
        split_all_button.on_click(on_split_all)
        exclude_group_button.on_click(on_exclude_group)

        return widgets.VBox(
            [header, controls],
            layout=widgets.Layout(
                width="100%",
                min_width="0",
                overflow="hidden",
                margin="0 0 4px 0",
            ),
        )

    def _build_not_selected_section(self, codes: Sequence[int]) -> widgets.Widget:
        """Build the "Not selected" section with bulk actions for it."""
        lookup = self._source_lookup
        count_text = f"{len(codes)} source class{'es' if len(codes) != 1 else ''}"
        if self.sample_df is not None:
            sample_count = sum(lookup[code].count for code in codes)
            count_text += f" &middot; {sample_count:,} samples"
        header = widgets.HTML(
            "<div style='font-size:13px;font-weight:700;color:#92400e;margin:2px 0 4px 0'>"
            f"Not selected &middot; {count_text}"
            "</div>"
        )
        return widgets.VBox(
            [
                header,
                widgets.HBox(
                    [self.bulk_label_input, self.bulk_assign_button],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", overflow="hidden", align_items="flex-start"
                    ),
                ),
                widgets.HBox(
                    [self.bulk_keep_individual_button],
                    layout=widgets.Layout(
                        width="100%", flex_flow="row wrap", overflow="hidden", align_items="flex-start"
                    ),
                ),
            ],
            layout=widgets.Layout(width="100%", min_width="0", overflow="hidden", margin="0 0 6px 0"),
        )

    def _selected_classes_header(self, groups: Sequence[str]) -> widgets.Widget:
        """Build the "Selected classes" header summarizing the current result."""
        lookup = self._source_lookup
        group_set = set(groups)
        codes = [code for code, group in self.assignments.items() if group in group_set]
        count_text = f"{len(groups)} final class{'es' if len(groups) != 1 else ''}"
        if self.sample_df is not None:
            sample_count = sum(lookup[code].count for code in codes)
            count_text += f" &middot; {sample_count:,} samples"
        return widgets.HTML(
            "<div style='font-size:13px;font-weight:700;color:#166534;margin:2px 0 4px 0'>"
            f"Selected classes &middot; {count_text}"
            "</div>"
        )

    def _refresh_final_groups(self) -> None:
        groups = self._groups()
        not_selected = self._not_selected_codes()

        cards: List[widgets.Widget] = []
        cards.append(self._selected_classes_header(groups))
        if groups:
            cards.extend(self._make_final_group_card(group) for group in groups)
        else:
            cards.append(widgets.HTML("<i>No final classes yet.</i>"))

        if not_selected:
            cards.append(
                widgets.HTML(
                    "<hr style='margin:10px 2px;border:none;border-top:2px solid #d1d5db'>"
                )
            )
            cards.append(self._build_not_selected_section(not_selected))

        self.final_groups_list_box.children = cards

    def _refresh_all(self) -> None:
        self._refresh_source_tree()
        self._refresh_group_controls()

    def _set_message(self, text: str, error: bool = False) -> None:
        color = "#991b1b" if error else "#166534"
        border = "#fecaca" if error else "#bbdfc2"
        background = "#fff7f7" if error else "#f4faf5"
        prefix = "⚠ " if error else "✓ "
        self.message.value = (
            f"<div style='margin:8px 0;padding:7px 10px;border:1px solid {border};"
            f"background:{background};color:{color}'>{prefix}{escape(text)}</div>"
        )

    def _clear_selected_after_action(self) -> None:
        self._selected_source_codes.clear()
        self._active_branch_path = None
        self._active_branch_label = None
        self._active_branch_codes = tuple()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def _on_assign(self, _=None) -> None:
        codes = self._selected_codes()
        group = self.group_dropdown.value
        if not codes:
            self._set_message("Select one or more source classes first.", error=True)
            return
        if not group:
            self._set_message("Select a target group first.", error=True)
            return
        for code in codes:
            self.assignments[code] = group
        self._clear_selected_after_action()
        self._set_message(f"Moved {len(codes)} source class(es) to '{group}'.")
        self._refresh_all()

    def _on_new_group(self, _=None) -> None:
        codes = self._selected_codes()
        name = self.new_group_name.value.strip()
        if not codes:
            self._set_message("Select one or more source classes first.", error=True)
            return
        if not name:
            self._set_message("Provide a name for the new final class.", error=True)
            return
        if name == EXCLUDED:
            self._set_message("That group name is reserved.", error=True)
            return
        for code in codes:
            self.assignments[code] = name
        self.new_group_name.value = ""
        self._clear_selected_after_action()
        self._set_message(
            f"Assigned {len(codes)} source class(es) to new group '{name}'."
        )
        self._refresh_all()

    def _on_exclude(self, _=None) -> None:
        codes = self._selected_codes()
        if not codes:
            self._set_message("Select one or more source classes first.", error=True)
            return
        for code in codes:
            self.assignments[code] = EXCLUDED
        self._clear_selected_after_action()
        self._set_message(f"Left {len(codes)} source class(es) out of the final list.")
        self._refresh_all()

    def _on_keep_individual(self, _=None) -> None:
        codes = self._selected_codes()
        if not codes:
            self._set_message("Select one or more source classes first.", error=True)
            return
        lookup = self._source_lookup
        for code in codes:
            self.assignments[code] = lookup[code].label
        self._clear_selected_after_action()
        self._set_message(
            f"Kept {len(codes)} source class(es) as individual final classes."
        )
        self._refresh_all()

    def _on_bulk_keep_individual(self, _=None) -> None:
        codes = self._not_selected_codes()
        if not codes:
            self._set_message("No not-selected source classes remain.", error=True)
            return
        lookup = self._source_lookup
        for code in codes:
            self.assignments[code] = lookup[code].label
        self._clear_selected_after_action()
        self._set_message(
            f"Kept {len(codes)} remaining source class(es) as individual final classes."
        )
        self._refresh_all()

    def _on_bulk_assign(self, _=None) -> None:
        codes = self._not_selected_codes()
        name = self.bulk_label_input.value.strip()
        if not codes:
            self._set_message("No not-selected source classes remain.", error=True)
            return
        if not name:
            self._set_message("Provide a name for the target final class.", error=True)
            return
        if name == EXCLUDED:
            self._set_message("That group name is reserved.", error=True)
            return
        for code in codes:
            self.assignments[code] = name
        self.bulk_label_input.value = ""
        self._clear_selected_after_action()
        self._set_message(f"Merged {len(codes)} remaining source class(es) into '{name}'.")
        self._refresh_all()

    def _group_hierarchy_node(
        self,
        label: str,
        codes: Sequence[int],
    ) -> None:
        valid_codes = [int(code) for code in codes if int(code) in self.assignments]
        if len(valid_codes) < 2:
            self._set_message(
                f"'{label}' does not contain multiple source classes to group.",
                error=True,
            )
            return
        for code in valid_codes:
            self.assignments[code] = label
        self._clear_selected_after_action()
        message = f"Created final class '{label}' from {len(valid_codes)} source classes"
        if self.sample_df is not None:
            sample_count = sum(self._source_lookup[code].count for code in valid_codes)
            message += f" ({sample_count:,} samples)"
        self._set_message(f"{message}.")
        self._refresh_all()

    def _on_apply(self, _=None) -> None:
        try:
            self.apply_selection()
        except Exception as exc:
            self._set_message(str(exc), error=True)
            return
        self._set_message(
            f"Final class list applied: "
            f"{self.croptypes['new_label'].nunique()} classes."
        )

    def _on_reset(self, _=None) -> None:
        self.assignments = {
            source.ewoc_code: EXCLUDED for source in self.sources
        }
        self.croptypes = pd.DataFrame()
        self.included_croptypes = pd.DataFrame()
        self.dropped_croptypes = pd.DataFrame()
        self._selected_source_codes.clear()
        self._last_auto_group_name = ""
        self._active_branch_path = None
        self._active_branch_label = None
        self._active_branch_codes = tuple()
        self._expanded_group_names.clear()
        self.search.value = ""
        self.source_status_filter.value = "all"
        self._set_message("Reset: all source classes are unselected again.")
        self._refresh_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def hierarchy_dataframe(self) -> pd.DataFrame:
        """Return the resolved legend hierarchy used by the source tree."""
        rows = []
        for source in self.sources:
            row = {
                "ewoc_code": source.ewoc_code,
                "original_label": source.label,
                "count": source.count,
            }
            for i, part in enumerate(source.hierarchy_path, start=1):
                row[f"hierarchy_{i}"] = part
            rows.append(row)
        return pd.DataFrame(rows)

    def mapping_dataframe(self, include_excluded: bool = True) -> pd.DataFrame:
        """Return the current source-to-final-class mapping."""
        lookup = self._source_lookup
        rows = []
        for code, group in self.assignments.items():
            excluded = group == EXCLUDED
            if excluded and not include_excluded:
                continue
            rows.append(
                {
                    "ewoc_code": code,
                    "original_label": lookup[code].label,
                    "new_label": None if excluded else group,
                    "excluded": excluded,
                    "count": lookup[code].count,
                    "hierarchy_path": " > ".join(lookup[code].hierarchy_path),
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["excluded", "new_label", "hierarchy_path", "original_label"],
            na_position="last",
        )

    def apply_selection(self, change=None) -> pd.DataFrame:
        """Freeze the current composition into compatibility dataframes."""
        mapping = self.mapping_dataframe(include_excluded=True)
        included = mapping.loc[~mapping["excluded"]].copy()
        excluded = mapping.loc[mapping["excluded"]].copy()

        if included.empty:
            raise ValueError("All classes are not selected. Keep at least one final class.")

        included = included.set_index("ewoc_code")[[
            "new_label",
            "original_label",
            "count",
        ]]
        excluded = excluded.set_index("ewoc_code")[["original_label", "count"]]
        if not excluded.empty:
            excluded["new_label"] = EXCLUDED
            excluded = excluded[["new_label", "original_label", "count"]]

        self.included_croptypes = included
        self.croptypes = included
        self.dropped_croptypes = excluded
        return included

    def apply_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the final class list and return a dataframe.

        Excluded EWOC classes are removed. Every remaining sample receives an
        explicit ``downstream_class``. No implicit ``other`` class is created.
        """
        if "ewoc_code" not in df.columns:
            raise ValueError("df must contain an 'ewoc_code' column.")
        self.apply_selection()
        mapping = self.included_croptypes["new_label"].to_dict()
        result = df.loc[df["ewoc_code"].isin(mapping)].copy()
        result["downstream_class"] = result["ewoc_code"].map(mapping)
        return result


def apply_class_selection_to_df(
    df: pd.DataFrame, selector: ClassSelectionWidget
) -> pd.DataFrame:
    """Apply a ClassSelectionWidget selection to a sample dataframe."""
    return selector.apply_to_df(df)


# Small notebook demo that works without WorldCereal by supplying both labels
# and hierarchy paths explicitly.
def demo(display_ui: bool = True) -> ClassSelectionWidget:
    """Display a self-contained hierarchical demo and return the selector."""
    labels = {
        1101010001: "winter wheat",
        1101010002: "spring wheat",
        1101020000: "barley",
        1101060000: "maize",
        1106000020: "soy",
        1106000030: "rapeseed",
        1115000000: "fallow",
    }
    hierarchy_paths = {
        1101010001: ("temporary crops", "cereals", "wheat", "winter wheat"),
        1101010002: ("temporary crops", "cereals", "wheat", "spring wheat"),
        1101020000: ("temporary crops", "cereals", "barley"),
        1101060000: ("temporary crops", "cereals", "maize"),
        1106000020: ("temporary crops", "oilseeds", "soy"),
        1106000030: ("temporary crops", "oilseeds", "rapeseed"),
        1115000000: ("temporary crops", "fallow"),
    }
    counts = {
        1101010001: 120,
        1101010002: 80,
        1101020000: 95,
        1101060000: 220,
        1106000020: 150,
        1106000030: 70,
        1115000000: 40,
    }
    rows = []
    sample_id = 0
    for code, count in counts.items():
        for _ in range(count):
            rows.append({"sample_id": sample_id, "ewoc_code": code})
            sample_id += 1
    df = pd.DataFrame(rows)
    return ClassSelectionWidget(
        sample_df=df,
        labels=labels,
        hierarchy_paths=hierarchy_paths,
        display_ui=display_ui,
    )
