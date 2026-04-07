"""
Interactive application for querying WorldCereal extractions.

Designed as a lightweight companion to the full classification app, this module
provides a focused widget interface for:
  - Drawing a bounding box on an interactive map.
  - Optionally querying the public WorldCereal S3 extraction bucket.
  - Adding one or more local parquet file/directory paths to include in the query.
  - Automatically deduplicating overlapping samples across sources.
  - Saving the merged result to a user-specified parquet file for use with the
    WorldCereal finetuning script (--parquet_files).

Usage:
    from notebook_utils.query_extractions_app import WorldCerealQueryExtractionsApp
    app = WorldCerealQueryExtractionsApp.run()
"""

import textwrap
from pathlib import Path
from typing import List, Optional

import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from loguru import logger
from notebook_utils.extractions import retrieve_extractions_extent
from shapely.geometry import Polygon
from tabulate import tabulate

from worldcereal.utils.legend import ewoc_code_to_label
from worldcereal.utils.map import ui_map
from worldcereal.utils.refdata import (
    query_private_extractions,
    query_public_extractions,
)


class WorldCerealQueryExtractionsApp:
    """
    Interactive notebook widget for querying and exporting WorldCereal extraction data.

    Supports combining the public WorldCereal S3 extraction bucket with one or more
    local parquet files or directories (e.g. restricted / institutional datasets).
    All sources are spatially filtered by the bounding box drawn on the map and
    deduplicated before being saved to a single output parquet file.
    """

    @classmethod
    def run(cls) -> "WorldCerealQueryExtractionsApp":
        """Instantiate and display the query application."""
        app = cls()
        display(app.ui)
        return app

    def __init__(self):
        self.result_df: Optional[pd.DataFrame] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _info_callout(text: str) -> widgets.HTML:
        """Render a blue-accented callout box."""
        return widgets.HTML(
            value=(
                "<div style='background:#f0f4ff;border-left:4px solid #3b82f6;"
                "padding:8px 12px;margin:6px 0;border-radius:4px;font-size:0.92em'>"
                f"{text}</div>"
            )
        )

    def _make_path_row(self, value: str = "") -> widgets.HBox:
        """Create one dynamic path-input row with a remove button."""
        text = widgets.Text(
            value=value,
            placeholder=(
                "/path/to/extractions.parquet  or  "
                "/path/to/directory/containing/parquet/files"
            ),
            layout=widgets.Layout(width="740px"),
        )
        remove_btn = widgets.Button(
            description="✕",
            button_style="danger",
            tooltip="Remove this path",
            layout=widgets.Layout(width="40px"),
        )
        row = widgets.HBox(
            [text, remove_btn],
            layout=widgets.Layout(margin="2px 0"),
        )

        def _on_remove(_b):
            self._paths_vbox.children = tuple(
                c for c in self._paths_vbox.children if c is not row
            )

        remove_btn.on_click(_on_remove)
        return row

    def _get_local_paths(self) -> List[str]:
        """Return non-empty path strings from the dynamic path list."""
        paths = []
        for row in self._paths_vbox.children:
            if isinstance(row, widgets.HBox):
                text_widget = row.children[0]
                val = text_widget.value.strip()
                if val:
                    paths.append(val)
        return paths

    def _get_bbox_polygon(self) -> Optional[Polygon]:
        """Return the current AOI polygon from the map without requiring Submit.

        Checks map internal state in priority order:
        1. Already-committed gdf (draw or upload + Submit was clicked).
        2. Drawn geometry pending submit (``_last_drawn_geometry``).
        3. Uploaded file pending submit (``_pending_gdf`` total_bounds).
        """
        from shapely.geometry import box as shapely_box

        try:
            gdf = self._aoi_map.gdf
            if gdf is not None and not gdf.empty:
                return shapely_box(*gdf.total_bounds)
        except Exception:
            pass

        poly = getattr(self._aoi_map, "_last_drawn_geometry", None)
        if poly is not None:
            return poly

        pending = getattr(self._aoi_map, "_pending_gdf", None)
        if pending is not None and not pending.empty:
            return shapely_box(*pending.total_bounds)

        return None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- header -------------------------------------------------------
        header = widgets.HTML(
            value=(
                "<h2>Query WorldCereal Extractions</h2>"
                "<p>Use this tool to spatially query public and/or local WorldCereal "
                "extraction datasets and export the merged result to a Parquet file "
                "for use with the finetuning script (<code>--parquet_files</code>).</p>"
            )
        )

        # ---- public extent reference map ----------------------------------
        extent_header = widgets.HTML(
            value=(
                "<h3 style='margin: 10px 0;'>Public extraction datasets</h3>"
                "<i>The map below shows where publicly available reference datasets "
                "exist. Use it as a reference when drawing your bounding box.</i>"
            )
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
            try:
                extent_gdf, extent_map = retrieve_extractions_extent()
                print(
                    f"Total number of publicly available datasets: {len(extent_gdf)}."
                )
                display(extent_map)
            except Exception as exc:
                print(f"Failed to load public extractions extent: {exc}")

        # ---- 1) AOI -------------------------------------------------------
        aoi_header = widgets.HTML(
            value="<h3 style='margin: 10px 0;'>1) Area of Interest</h3>"
        )
        aoi_explanation = self._info_callout(
            "Draw a rectangle on the map to spatially constrain your query — "
            "that's all, no further steps required.<br>"
            "You can also upload a vector file (zipped shapefile, GeoPackage or "
            "Parquet) via the upload button on the top-right of the map — the "
            "bounding box of its geometries will be used as the AOI.<br><br>"
            "⚠️ If no AOI is selected the query will consider all available data "
            "globally, which may produce a very large result."
        )
        self._aoi_map = ui_map(display_ui=False)

        # ---- 2) Data sources ----------------------------------------------
        sources_header = widgets.HTML(
            value="<h3 style='margin: 10px 0;'>2) Data Sources</h3>"
        )

        # Public S3 checkbox
        self._public_checkbox = widgets.Checkbox(
            value=True,
            description="Query public S3 bucket",
            layout=widgets.Layout(width="280px"),
        )
        public_explanation = self._info_callout(
            "When checked, the WorldCereal public S3 extraction bucket will be "
            "queried for samples within your AOI. Requires internet access."
        )

        # Local paths
        local_header = widgets.HTML(
            value=(
                "<b>Local extraction path(s):</b> "
                "<i>(optional — add one or more local parquet files or directories "
                "to include alongside the public bucket query)</i>"
            )
        )
        local_explanation = self._info_callout(
            "Each entry can be a path to a single <code>.parquet</code> file or a "
            "directory that will be searched recursively for "
            "<code>*.parquet</code> files.<br>"
            "This is particularly useful for accessing restricted or "
            "institutional datasets that are not available on the public S3 bucket "
            "(but may still contain the same samples — duplicates are removed "
            "automatically after merging)."
        )

        self._paths_vbox = widgets.VBox(
            [],
            layout=widgets.Layout(margin="4px 0"),
        )

        add_path_btn = widgets.Button(
            description="+ Add path",
            button_style="success",
            icon="plus",
            layout=widgets.Layout(width="140px"),
        )

        def _on_add_path(_b):
            self._paths_vbox.children = tuple(self._paths_vbox.children) + (
                self._make_path_row(),
            )

        add_path_btn.on_click(_on_add_path)

        # ---- 3) Filter options --------------------------------------------
        filter_header = widgets.HTML(
            value="<h3 style='margin: 10px 0;'>3) Filter Options</h3>"
        )
        crop_only_explanation = self._info_callout(
            "When checked, only <b>temporary crop samples</b> "
            "(ewoc_code range 1100000000–1114999999) are retrieved — "
            "useful for crop-type-only classification.<br>"
            "Leave unchecked to include all land-cover categories (recommended for "
            "end-to-end finetuning where the model must also distinguish cropland "
            "from non-cropland)."
        )
        self._crop_only_checkbox = widgets.Checkbox(
            value=False,
            description="Only temporary crop samples",
            layout=widgets.Layout(width="300px"),
        )

        # ---- 4) Run query -------------------------------------------------
        query_header = widgets.HTML(
            value="<h3 style='margin: 10px 0;'>4) Run Query</h3>"
        )
        self._run_btn = widgets.Button(
            description="Run Query",
            button_style="primary",
            icon="search",
            layout=widgets.Layout(width="400px", height="80px"),
        )
        self._query_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #ccc",
                padding="10px",
            )
        )
        self._run_btn.on_click(self._on_run_query)

        # ---- 5) Save results ----------------------------------------------
        save_header = widgets.HTML(
            value="<h3 style='margin: 10px 0;'>5) Save Results</h3>"
        )
        save_explanation = self._info_callout(
            "Specify an output path for the merged Parquet file and click "
            "<b>Save to Parquet</b>.<br>"
            "The resulting file can be passed directly to the finetuning script via "
            "<code>--parquet_files \"/path/to/output.parquet\"</code>."
        )
        self._save_path_input = widgets.Text(
            value="./worldcereal_query_result.parquet",
            placeholder="/path/to/output.parquet",
            description="Output path:",
            layout=widgets.Layout(width="720px"),
        )
        self._save_btn = widgets.Button(
            description="Save to Parquet",
            button_style="success",
            icon="save",
            layout=widgets.Layout(width="220px"),
            disabled=True,
        )
        self._save_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                min_height="60px",
                border="1px solid #ccc",
                padding="10px",
            )
        )
        self._save_btn.on_click(self._on_save)

        # ---- assemble UI --------------------------------------------------
        self.ui = widgets.VBox(
            [
                header,
                extent_header,
                extent_output,
                aoi_header,
                aoi_explanation,
                self._aoi_map.map,
                self._aoi_map.output,
                sources_header,
                public_explanation,
                self._public_checkbox,
                local_header,
                local_explanation,
                self._paths_vbox,
                widgets.HBox([add_path_btn]),
                filter_header,
                crop_only_explanation,
                self._crop_only_checkbox,
                query_header,
                widgets.HBox([self._run_btn]),
                self._query_output,
                save_header,
                save_explanation,
                self._save_path_input,
                widgets.HBox([self._save_btn]),
                self._save_output,
            ]
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_run_query(self, _b):
        """Execute the query against all configured sources and display a summary."""
        with self._query_output:
            self._query_output.clear_output()
            self._save_btn.disabled = True
            self.result_df = None

            # --- resolve AOI ---
            bbox_poly = self._get_bbox_polygon()
            if bbox_poly is None:
                print(
                    "No AOI selected — querying without spatial constraint.\n"
                    "⚠️  This may return a very large result set.\n"
                )

            include_public = self._public_checkbox.value
            local_paths = self._get_local_paths()
            filter_cropland = self._crop_only_checkbox.value

            if not include_public and not local_paths:
                print(
                    "⚠️  Please enable 'Query public S3 bucket' or add at least one "
                    "local path before running the query."
                )
                return

            results: list[pd.DataFrame] = []
            extraction_summary: list[dict] = []

            # --- public S3 ---
            if include_public:
                print("Querying public S3 bucket …")
                try:
                    public_df = query_public_extractions(
                        bbox_poly=bbox_poly,
                        buffer=0,
                        filter_cropland=filter_cropland,
                    )
                    if len(public_df) > 0:
                        results.append(public_df)
                        for ref_id in sorted(public_df["ref_id"].unique()):
                            d = public_df[public_df["ref_id"] == ref_id]
                            extraction_summary.append(
                                {
                                    "Source": "Public S3",
                                    "Dataset": ref_id,
                                    "Samples": d["sample_id"].nunique(),
                                    "Crop Types": d["ewoc_code"].nunique(),
                                }
                            )
                        print(
                            f"  → {public_df['sample_id'].nunique()} samples from "
                            f"{public_df['ref_id'].nunique()} public dataset(s)."
                        )
                    else:
                        print("  → No public samples found for the specified criteria.")
                except Exception as exc:
                    print(f"  ⚠️  Public S3 query failed: {exc}")

            # --- local paths ---
            for path_str in local_paths:
                print(f"Querying local path: {path_str} …")
                try:
                    local_df = query_private_extractions(
                        path_str,
                        bbox_poly=bbox_poly,
                        filter_cropland=filter_cropland,
                        buffer=0,
                    )
                    if len(local_df) > 0:
                        results.append(local_df)
                        source_label = Path(path_str).name
                        for ref_id in sorted(local_df["ref_id"].unique()):
                            d = local_df[local_df["ref_id"] == ref_id]
                            extraction_summary.append(
                                {
                                    "Source": f"Local ({source_label})",
                                    "Dataset": ref_id,
                                    "Samples": d["sample_id"].nunique(),
                                    "Crop Types": d["ewoc_code"].nunique(),
                                }
                            )
                        print(
                            f"  → {local_df['sample_id'].nunique()} samples from "
                            f"{local_df['ref_id'].nunique()} dataset(s)."
                        )
                    else:
                        print(f"  → No samples found at: {path_str}")
                except Exception as exc:
                    print(f"  ⚠️  Query failed for {path_str}: {exc}")

            if not results:
                print(
                    "\n⚠️  No extractions found for the specified criteria. "
                    "Try expanding your area of interest or adding more data sources."
                )
                return

            # --- merge & deduplicate ---
            if len(results) > 1:
                merged_df = pd.concat(results, ignore_index=True)
                n_before = merged_df["sample_id"].nunique()
                merged_df = merged_df.drop_duplicates(
                    subset=["sample_id", "timestamp"], keep="first"
                )
                n_after = merged_df["sample_id"].nunique()
                n_dropped_samples = n_before - n_after
                if n_dropped_samples:
                    print(
                        f"\nDeduplication: removed {n_dropped_samples} duplicate "
                        f"sample(s) that appeared in more than one source."
                    )
            else:
                merged_df = results[0]

            # --- attach human-readable labels ---
            merged_df["label_full"] = ewoc_code_to_label(
                merged_df["ewoc_code"], label_type="full"
            )
            merged_df["sampling_label"] = ewoc_code_to_label(
                merged_df["ewoc_code"], label_type="sampling"
            )
            if "feature_index" in merged_df.columns:
                merged_df = merged_df.drop(columns=["feature_index"])

            # --- summary table ---
            DEFAULT_COL_WIDTH = 40
            print("\n" + "=" * 80)
            print("QUERY EXTRACTIONS SUMMARY")
            print("=" * 80)

            summary_df = pd.DataFrame(extraction_summary)
            for col in summary_df.columns:
                if summary_df[col].dtype == object:
                    summary_df[col] = summary_df[col].astype(str).apply(
                        lambda s: (s[: DEFAULT_COL_WIDTH - 1] + "…")
                        if len(s) > DEFAULT_COL_WIDTH
                        else s
                    )
            print("\nDatasets Retrieved:")
            print(
                tabulate(
                    summary_df,
                    headers="keys",
                    tablefmt="grid",
                    showindex=False,
                    maxcolwidths=[DEFAULT_COL_WIDTH] * summary_df.shape[1],
                )
            )

            total_samples = merged_df["sample_id"].nunique()
            total_datasets = len(extraction_summary)
            total_crop_types = merged_df["ewoc_code"].nunique()
            unique_crop_groups = sorted(merged_df["sampling_label"].unique())
            wrapped_crop_groups = "\n".join(
                textwrap.wrap(", ".join(unique_crop_groups), width=40)
            )
            stats_table = [
                ["Total Samples", total_samples],
                ["Total Datasets", total_datasets],
                ["Unique Crop Types", total_crop_types],
                ["Crop Groups", wrapped_crop_groups],
            ]
            print("\nOverall Statistics:")
            print(
                tabulate(
                    stats_table,
                    headers=["Metric", "Value"],
                    tablefmt="grid",
                    maxcolwidths=[None, 40],
                )
            )
            print("=" * 80 + "\n")

            if total_crop_types <= 1:
                logger.warning(
                    "Only one crop type found. Consider expanding your area of "
                    "interest or including additional data sources."
                )

            self.result_df = merged_df
            self._save_btn.disabled = False
            print(
                "✅ Query complete. Adjust the output path below and click "
                "'Save to Parquet'."
            )

    def _on_save(self, _b):
        """Save the query result DataFrame to the specified parquet file."""
        with self._save_output:
            self._save_output.clear_output()

            if self.result_df is None:
                print("⚠️  No query result available. Run the query first.")
                return

            out_path = Path(self._save_path_input.value.strip())
            if not out_path.suffix:
                out_path = out_path.with_suffix(".parquet")

            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                self.result_df.to_parquet(out_path, index=False)
                print(
                    f"✅ Saved {self.result_df['sample_id'].nunique()} unique samples "
                    f"({len(self.result_df)} rows) → {out_path}"
                )
                print(
                    f"\nPass this file to the finetuning script with:\n"
                    f'  --parquet_files "{out_path}"'
                )
            except Exception as exc:
                print(f"⚠️  Save failed: {exc}")
