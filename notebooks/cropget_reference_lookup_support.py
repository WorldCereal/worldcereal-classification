"""Support code for the CropGET precomputed reference lookup notebook."""
from __future__ import annotations

from pathlib import Path
import html
import os

import folium
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ipywidgets as widgets
from IPython.display import FileLink, clear_output, display

CROPS = ("barley", "maize", "millet", "rapeseed", "rice", "rye", "sorghum", "soybean", "sunflower", "wheat")
CROP_COLORS = {
    "barley": "#56B4E9", "maize": "#E69F00", "millet": "#CC79A7",
    "rapeseed": "#009E73", "rice": "#0072B2", "rye": "#7A7A7A",
    "sorghum": "#D55E00", "soybean": "#6A3D9A", "sunflower": "#F0E442",
    "wheat": "#8C564B",
}
DEFAULT_LOOKUP_ROOT = Path(os.environ.get(
    "CROPGET_TRAIN_POINTS_LOOKUP_ROOT",
    Path.cwd() / "data" / "cropget_lookup",
))


def _normalize_id(value):
    """Normalize id.

    Args:
        value: Value tested, transformed, or compared by this function.

    Returns:
        str: The constant value ``.
        object: The result of `text[:-2]`.
        object: The computed `text` value.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("+-").isdigit():
        return text[:-2]
    return text


def _find_column(frame, candidates):
    """Find column.

    Args:
        frame: Input value for `frame` used by `_find_column`.
        candidates: Input value for `candidates` used by `_find_column`.

    Returns:
        object: The result of `next`.
    """
    lower = {str(column).lower(): column for column in frame.columns}
    return next((lower[name] for name in candidates if name in lower), None)


def _prepare_queries(parquet_path):
    """Prepare queries.

    Args:
        parquet_path: Path to the parquet used by this function.

    Returns:
        tuple: Tuple containing `query`, `source_rows` in that order.

    Raises:
        FileNotFoundError: If the corresponding error condition is detected.
        ValueError: If the corresponding error condition is detected.
    """
    parquet_path = Path(parquet_path).expanduser()
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    raw = pd.read_parquet(parquet_path)
    source_rows = len(raw)
    ref_col = _find_column(raw, ("ref_id",))
    sample_col = _find_column(raw, ("sample_id", "sampleid"))
    if ref_col is None or sample_col is None:
        raise ValueError("The Parquet must contain ref_id and sample_id columns.")

    query = pd.DataFrame({
        "_input_row": np.arange(len(raw), dtype=np.int64),
        "query_ref_id": raw[ref_col].map(_normalize_id),
        "query_sample_id": raw[sample_col].map(_normalize_id),
    })
    lat_col = _find_column(raw, ("latitude", "lat", "lat_x", "query_lat"))
    lon_col = _find_column(raw, ("longitude", "lon", "lon_x", "query_lon"))
    query["_uploaded_query_lat"] = pd.to_numeric(raw[lat_col], errors="coerce") if lat_col else np.nan
    query["_uploaded_query_lon"] = pd.to_numeric(raw[lon_col], errors="coerce") if lon_col else np.nan
    query["_ref_key"] = query["query_ref_id"]
    query["_sample_key"] = query["query_sample_id"]
    query = query.loc[query["_ref_key"].ne("") & query["_sample_key"].ne("")].copy()
    keys = ["_ref_key", "_sample_key"]
    query = query.sort_values("_input_row", kind="stable")
    query["_uploaded_query_lat"] = query.groupby(keys, sort=False)["_uploaded_query_lat"].transform("first")
    query["_uploaded_query_lon"] = query.groupby(keys, sort=False)["_uploaded_query_lon"].transform("first")
    query = query.drop_duplicates(keys, keep="first").reset_index(drop=True)
    return query, source_rows


def _lookup_path(root, crop, season):
    """Handle lookup path.

    Args:
        root: Root directory containing the relevant project or data files.
        crop: Crop identifier.
        season: Season identifier.

    Returns:
        object: The computed `path` value.
    """
    root = Path(root).expanduser()
    base = root.parent if root.name.lower() in {"top10", "top25"} else root
    path = base / "top25" / crop / f"season={season}.parquet"
    if not path.is_file() and root.name.lower() == "top25":
        path = root / crop / f"season={season}.parquet"
    return path


def _load_lookup(path):
    """Load lookup.

    Args:
        path: Filesystem path used to read input or write output.

    Returns:
        object: The computed `frame` value.

    Raises:
        ValueError: If the corresponding error condition is detected.
    """
    columns = pq.read_schema(path).names
    required = {
        "query_ref_id", "query_sample_id", "neighbor_rank", "matched_sample_id",
        "matched_lat", "matched_lon", "similarity", "spectral_similarity",
    }
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Lookup is missing columns: {sorted(missing)}")
    optional = {"query_lat", "query_lon"}
    frame = pd.read_parquet(path, columns=sorted(required | (optional & set(columns))))
    frame["_ref_key"] = frame["query_ref_id"].map(_normalize_id)
    frame["_sample_key"] = frame["query_sample_id"].map(_normalize_id)
    for name in ("query_lat", "query_lon", "matched_lat", "matched_lon", "similarity", "spectral_similarity"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["neighbor_rank"] = pd.to_numeric(frame["neighbor_rank"], errors="coerce")
    return frame


def _match_crop(queries, path, crop, season, top_k, sort_by):
    """Handle match crop.

    Args:
        queries: Query samples processed by this function.
        path: Filesystem path used to read input or write output.
        crop: Crop identifier.
        season: Season identifier.
        top_k: Maximum number of top-ranked results to return.
        sort_by: Input value for `sort_by` used by `_match_crop`.

    Returns:
        tuple: Tuple containing `pd.DataFrame()`, `missing` in that order.
        tuple: Tuple containing `joined.reset_index(drop=True)`, `missing` in that order.
    """
    lookup = _load_lookup(path)
    lookup = lookup.loc[lookup["neighbor_rank"].le(int(top_k))].copy()
    keys = ["_ref_key", "_sample_key"]
    available = lookup[keys].drop_duplicates()
    requested = queries[keys].drop_duplicates()
    missing = int(requested.merge(available, on=keys, how="left", indicator=True)["_merge"].eq("left_only").sum())
    lookup_columns = keys + ["neighbor_rank", "matched_sample_id", "matched_lat", "matched_lon", "similarity", "spectral_similarity"]
    for name in ("query_lat", "query_lon"):
        if name in lookup:
            lookup_columns.append(name)
    joined = queries.merge(lookup[lookup_columns], on=keys, how="inner", validate="one_to_many")
    if joined.empty:
        return pd.DataFrame(), missing

    lookup_lat = joined["query_lat"] if "query_lat" in joined else np.nan
    lookup_lon = joined["query_lon"] if "query_lon" in joined else np.nan
    joined["query_lat"] = joined["_uploaded_query_lat"].where(joined["_uploaded_query_lat"].notna(), lookup_lat)
    joined["query_lon"] = joined["_uploaded_query_lon"].where(joined["_uploaded_query_lon"].notna(), lookup_lon)
    joined = joined.rename(columns={
        "matched_sample_id": "retrieved_sample_id", "matched_lat": "retrieved_lat",
        "matched_lon": "retrieved_lon", "neighbor_rank": "retrieval_rank",
    })
    joined["crop"] = crop
    joined["season"] = season
    output_columns = ["_input_row", "query_ref_id", "query_sample_id", "query_lat", "query_lon", "crop", "season",
                      "retrieval_rank", "retrieved_sample_id", "retrieved_lat", "retrieved_lon", "similarity", "spectral_similarity"]
    joined = joined[output_columns]
    if sort_by == "Spectral similarity (highest first)":
        joined = joined.sort_values(["_input_row", "spectral_similarity", "retrieval_rank"], ascending=[True, False, True], kind="stable")
    else:
        joined = joined.sort_values(["_input_row", "retrieval_rank"], ascending=[True, True], kind="stable")
    return joined.reset_index(drop=True), missing


def _make_map(matches):
    """Create map.

    Args:
        matches: Input value for `matches` used by `_make_map`.

    Returns:
        object: The computed `fmap` value.
    """
    matches = matches.copy()
    coordinate_columns = ("query_lat", "query_lon", "retrieved_lat", "retrieved_lon")
    for column in coordinate_columns:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")
    query_points = matches.drop_duplicates(["crop", "query_ref_id", "query_sample_id"])
    coordinate_parts = [query_points[["query_lat", "query_lon"]].dropna().to_numpy(),
                        matches[["retrieved_lat", "retrieved_lon"]].dropna().to_numpy()]
    valid_parts = [part for part in coordinate_parts if len(part)]
    coords = np.vstack(valid_parts) if valid_parts else np.empty((0, 2))
    center = coords.mean(axis=0).tolist() if len(coords) else [20, 0]
    fmap = folium.Map(location=center, zoom_start=2 if not len(coords) else 5,
                      tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)

    # Draw reference circles first; query diamonds are added afterward to stay visible.
    for crop, part in matches.groupby("crop", sort=False):
        color = CROP_COLORS.get(crop, "#315C9B")
        layer = folium.FeatureGroup(name=f"{crop.title()} · retrieved", show=True)
        seen = set()
        for row in part.itertuples(index=False):
            if pd.isna(row.retrieved_lat) or pd.isna(row.retrieved_lon):
                continue
            key = (str(row.retrieved_sample_id), float(row.retrieved_lat), float(row.retrieved_lon))
            if key in seen:
                continue
            seen.add(key)
            folium.CircleMarker([key[1], key[2]], radius=5, color=color, weight=1.2,
                fill=True, fill_color=color, fill_opacity=.78,
                tooltip=f"Retrieved · {crop} · {html.escape(str(row.retrieved_sample_id))}").add_to(layer)
        layer.add_to(fmap)

    for crop, part in query_points.groupby("crop", sort=False):
        color = CROP_COLORS.get(crop, "#315C9B")
        layer = folium.FeatureGroup(name=f"{crop.title()} · query", show=True)
        for row in part.itertuples(index=False):
            if pd.isna(row.query_lat) or pd.isna(row.query_lon):
                continue
            folium.Marker([row.query_lat, row.query_lon], icon=folium.DivIcon(
                icon_size=(14, 14), icon_anchor=(7, 7),
                html=f'<div style="width:10px;height:10px;background:{color};border:2px solid #17343B;transform:rotate(45deg);box-shadow:0 1px 3px #333"></div>',
            ), tooltip=f"Query · {crop} · {html.escape(str(row.query_sample_id))}").add_to(layer)
        layer.add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    if len(coords):
        fmap.fit_bounds([[float(coords[:, 0].min()), float(coords[:, 1].min())],
                         [float(coords[:, 0].max()), float(coords[:, 1].max())]], padding=(20, 20))
    return fmap


def create_app(lookup_root=DEFAULT_LOOKUP_ROOT, output_root=None):
    """Create and display the notebook controls; return the widget container.

    Args:
        lookup_root: Input value for `lookup_root` used by `create_app`.
        output_root: Input value for `output_root` used by `create_app`.

    Returns:
        object: The computed `app` value.
    """
    lookup_root = widgets.Text(value=str(lookup_root), description="Folder:",
        layout=widgets.Layout(flex="1 1 auto", min_width="250px"),
        style={"description_width": "65px"})
    folder_browse_control = widgets.Button(description="Browse…", icon="folder-open",
        layout=widgets.Layout(width="115px", height="34px"))
    folder_row = widgets.HBox([lookup_root, folder_browse_control],
        layout=widgets.Layout(width="100%", align_items="center", gap="8px"))
    folder_panel = widgets.Accordion(children=[folder_row])
    folder_panel.set_title(0, "Lookup folder")
    folder_panel.selected_index = None

    crop_controls = {
        crop: widgets.Checkbox(value=(crop == "maize"), description=crop.title(), indent=False,
            layout=widgets.Layout(width="145px"))
        for crop in CROPS
    }
    crop_grid = widgets.GridBox(
        children=list(crop_controls.values()),
        layout=widgets.Layout(grid_template_columns="repeat(5, 145px)", grid_gap="2px 10px",
                              width="780px", max_width="100%")
    )
    season_control = widgets.Dropdown(options=["s1", "s2"], value="s1", description="Season:",
        style={"description_width": "70px"}, layout=widgets.Layout(width="180px"))
    k_control = widgets.IntSlider(value=10, min=1, max=25, step=1, description="Top K:", continuous_update=False,
        style={"description_width": "70px"}, layout=widgets.Layout(width="300px"))
    sort_control = widgets.Dropdown(options=["Retrieval rank", "Spectral similarity (highest first)"],
        value="Retrieval rank", description="Sort:", style={"description_width": "70px"},
        layout=widgets.Layout(width="330px"))
    parquet_path = widgets.Text(
        value="", placeholder=r"Choose or paste the WorldCereal .parquet path",
        description="Parquet:", layout=widgets.Layout(width="100%"),
        style={"description_width": "70px"}
    )
    browse_control = widgets.Button(description="Browse…", icon="folder-open",
        layout=widgets.Layout(width="115px", height="34px"))
    run_control = widgets.Button(description="Find neighbors", button_style="warning", icon="search",
        layout=widgets.Layout(width="190px", height="38px"))
    status = widgets.Output()
    results = widgets.Output()

    def run_lookup(_):
        """Run lookup.

        Args:
            _: Input value for `_` used by `run_lookup`.

        Returns:
            None: No value is returned.

        Raises:
            FileNotFoundError: If the corresponding error condition is detected.
            ValueError: If the corresponding error condition is detected.
        """
        with status:
            clear_output(wait=True)
            try:
                selected_crops = [crop for crop, control in crop_controls.items() if control.value]
                if not selected_crops:
                    raise ValueError("Select at least one crop.")
                season, top_k, sort_by = season_control.value, int(k_control.value), sort_control.value
                if not parquet_path.value.strip():
                    raise ValueError("Choose or paste the WorldCereal Parquet path.")
                queries, source_rows = _prepare_queries(parquet_path.value.strip())
                if queries.empty:
                    raise ValueError("No valid ref_id/sample_id pairs were found.")
                root = Path(lookup_root.value).expanduser()
                paths = {crop: _lookup_path(root, crop, season) for crop in selected_crops}
                absent = [f"{crop}: {path}" for crop, path in paths.items() if not path.is_file()]
                if absent:
                    raise FileNotFoundError("Lookup file(s) not found:\n" + "\n".join(absent))

                parts, missing_by_crop = [], {}
                for crop, path in paths.items():
                    found, missing = _match_crop(queries, path, crop, season, top_k, sort_by)
                    missing_by_crop[crop] = missing
                    if not found.empty:
                        parts.append(found)
                columns = ["_input_row", "query_ref_id", "query_sample_id", "query_lat", "query_lon", "crop", "season",
                           "retrieval_rank", "retrieved_sample_id", "retrieved_lat", "retrieved_lon", "similarity", "spectral_similarity"]
                matches = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
                query_count = len(queries)
                matched_query_count = matches[["crop", "query_ref_id", "query_sample_id"]].drop_duplicates().shape[0]
                collapsed = source_rows - query_count
                unique_retrieved_ids = matches["retrieved_sample_id"].nunique(dropna=True)
                print(
                    f"Queries: {matched_query_count:,}/{query_count:,} matched  ·  "
                    f"Unique retrieved IDs: {unique_retrieved_ids:,}  ·  "
                    f"Collapsed rows: {collapsed:,}"
                )
                unmatched_text = "  |  ".join(f"{crop}: {count:,} unmatched" for crop, count in missing_by_crop.items())
                print(unmatched_text)

                with results:
                    clear_output(wait=True)
                    if matches.empty:
                        print("No IDs matched the selected lookup.")
                        return
                    display(_make_map(matches))
                    display(matches.drop(columns=["_input_row"]).head(10000))
                    if len(matches) > 10000:
                        print("Table preview limited to 10,000 rows; CSV has all matches.")
                    destination = Path(output_root) if output_root else Path.cwd() / "cropget_notebook_outputs"
                    destination.mkdir(parents=True, exist_ok=True)
                    crop_tag = "-".join(selected_crops)
                    csv_path = destination / f"cropget_{crop_tag}_{season}_top{top_k}_matches.csv"
                    matches.drop(columns=["_input_row"]).to_csv(csv_path, index=False)
                    display(FileLink(str(csv_path), result_html_prefix="CSV: "))
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")

    def browse_for_parquet(_):
        """Handle browse for parquet.

        Args:
            _: Input value for `_` used by `browse_for_parquet`.
        """
        with status:
            clear_output(wait=True)
            try:
                from tkinter import Tk, filedialog
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                selected = filedialog.askopenfilename(
                    title="Select WorldCereal Parquet",
                    filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")],
                )
                root.destroy()
                if selected:
                    parquet_path.value = selected
            except Exception as exc:
                print(f"File browser unavailable: {exc}. Paste the Parquet path into the field instead.")

    def browse_for_lookup_folder(_):
        """Handle browse for lookup folder.

        Args:
            _: Input value for `_` used by `browse_for_lookup_folder`.
        """
        with status:
            clear_output(wait=True)
            try:
                from tkinter import Tk, filedialog
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                current = Path(lookup_root.value).expanduser()
                selected = filedialog.askdirectory(
                    title="Select crop lookup folder",
                    initialdir=str(current) if current.is_dir() else str(Path.home()),
                    mustexist=True,
                )
                root.destroy()
                if selected:
                    lookup_root.value = selected
            except Exception as exc:
                print(f"Folder browser unavailable: {exc}. Paste the folder path into the field instead.")

    folder_browse_control.on_click(browse_for_lookup_folder)
    browse_control.on_click(browse_for_parquet)
    run_control.on_click(run_lookup)
    banner = widgets.HTML("""
    <div style="background:linear-gradient(110deg,#193F50,#286576);color:white;padding:14px 18px;border-radius:12px;margin:0 0 4px">
      <span style="font-size:22px;font-weight:700">CropGET</span>
      <span style="margin-left:12px;font-size:16px">Reference data lookup</span>
    </div>
    """)
    controls = widgets.HBox([widgets.VBox([season_control, k_control]), sort_control],
        layout=widgets.Layout(align_items="center", gap="18px", flex_flow="row wrap"))
    parquet_row = widgets.HBox([parquet_path, browse_control],
        layout=widgets.Layout(width="100%", align_items="center", gap="8px"))
    app = widgets.VBox([banner, folder_panel, widgets.HTML("<b>Crops</b>"), crop_grid,
                        controls, parquet_row, run_control, status, results],
        layout=widgets.Layout(width="100%", gap="10px"))
    display(app)
    return app
