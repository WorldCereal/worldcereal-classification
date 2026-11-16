"""CropGet local Find app for WorldCereal queries and locations."""
from __future__ import annotations

import os
import base64
import io
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from folium.plugins import Draw
from sklearn.neighbors import BallTree
from streamlit_folium import st_folium

CROPS = ("maize", "sunflower", "soybean")
TRAIN_POINT_CROPS = ("barley", "maize", "millet", "rapeseed", "rice", "rye", "sorghum", "soybean", "sunflower", "wheat")
CROP_COLORS = {
    "barley": "#56B4E9", "maize": "#E69F00", "millet": "#CC79A7",
    "rapeseed": "#009E73", "rice": "#0072B2", "rye": "#7A7A7A",
    "sorghum": "#D55E00", "soybean": "#6A3D9A", "sunflower": "#F0E442",
    "wheat": "#8C564B",
}
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = Path(os.environ.get(
    "CROPGET_MODE1_ASSET_ROOT", PACKAGE_ROOT / "data" / "cropget_assets"
))
DEFAULT_TRAIN_LOOKUP_ROOT = Path(os.environ.get(
    "CROPGET_TRAIN_POINTS_LOOKUP_ROOT",
    PACKAGE_ROOT / "data" / "cropget_lookup",
))


def apply_style() -> None:
    """Apply style."""
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;500;600;700&display=swap');
      :root { --blue:#20435A; --teal:#286576; --cyan:#66BDC7; --mustard:#D9AE20;
              --lime:#A4B52C; --canvas:#F3F6F5; --ink:#17343B; --muted:#536A70; --line:#DCE5E3; }
      .stApp,[data-testid="stAppViewContainer"] { background:var(--canvas); color:var(--ink);
        font-family:'Comfortaa',Arial,sans-serif; }
      section[data-testid="stSidebar"] { background:#EEF2F6; color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(244,246,245,.95); }
      .block-container { max-width:1580px; padding-top:1.4rem; padding-bottom:2rem; }
      h1,h2,h3,p,div,label,button,span { font-family:'Comfortaa',Arial,sans-serif; }
      h1,h2,h3,p,label,[data-testid="stMarkdownContainer"],
      [data-testid="stWidgetLabel"] p { color:var(--ink)!important; }
      .hero-panel { background:linear-gradient(110deg,#193F50,#286576); color:white;
              border-radius:16px; padding:32px 30px; min-height:154px; box-sizing:border-box;
              display:flex; align-items:center; overflow:visible; line-height:1.35;
              box-shadow:0 8px 24px rgba(23,52,59,.12); }
      .hero-panel h1 { color:white!important; font-size:clamp(1.45rem,2.3vw,2rem); line-height:1.3;
              margin:0; letter-spacing:.01em; overflow:visible; }
      .logo-panel { background:white; border:1px solid #E7ECEA; border-radius:16px;
              padding:24px 22px; min-height:154px; box-sizing:border-box; display:flex; align-items:center; justify-content:center;
              box-shadow:0 8px 24px rgba(23,52,59,.08); }
      .logo-panel img { max-height:104px; }
      [data-testid="stTabs"] [role="tab"] { background:white; border:1px solid var(--line);
        border-radius:8px 8px 0 0; color:var(--ink); padding:8px 15px; font-weight:700; }
      [data-testid="stTabs"] [aria-selected="true"] { background:var(--mustard); color:#111!important; }
      [data-testid="stTabs"] [aria-selected="true"] * { color:#111!important; }
      [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line)!important; border-radius:10px!important; }
      div.stButton > button[kind="primary"], div.stDownloadButton > button {
        background:var(--mustard); color:#111; border:1px solid #C49A12; border-radius:8px;
        font-weight:700; min-height:2.8rem; }
      div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button:hover {
        background:#E7C54D; color:#111; border-color:#927510; }
      [data-testid="stMetric"] { background:white; border-left:4px solid var(--blue);
        border-radius:8px; padding:9px 12px; }
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:var(--ink)!important; }
      [data-testid="stMetricDelta"] { color:#6D5A0A!important; }
      [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div { color:#17343B; background:#F7F8FA; }
      [data-testid="stExpander"] { border-color:var(--line); }
      [data-testid="stCaptionContainer"] p { color:var(--muted)!important; }
      [data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] p {
        color:#263F46!important; font-weight:700!important; }
      [data-testid="stAlert"] { border-radius:10px; }
      .section-label { color:var(--blue); text-transform:uppercase; letter-spacing:.06em;
        font-size:.72rem; font-weight:700; margin:3px 0 4px; }
      .home-card { background:white; border:1px solid var(--line); border-radius:16px;
        padding:22px 24px; height:300px; box-sizing:border-box; overflow:hidden;
        box-shadow:0 5px 18px rgba(23,52,59,.06); }
      .home-card h3 { margin-top:0; }
      div[role="radiogroup"] { gap:8px; }
      div[role="radiogroup"] label[data-baseweb="radio"] { background:#fff; border:1px solid var(--line);
        border-radius:8px; padding:7px 14px; color:#263F46!important; }
      div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display:none; }
      div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) { background:#286576; border-color:#286576; }
      div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) * { color:white!important; }
      @media(max-width:800px){ .hero-panel{min-height:120px;padding:22px} .logo-panel{min-height:120px;padding:14px} .home-card{height:auto;min-height:250px} .block-container{padding-left:1rem;padding-right:1rem} }
    </style>
    """, unsafe_allow_html=True)


def header() -> None:
    """Implement header."""
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown('<div class="hero-panel"><h1>CropGET</h1></div>', unsafe_allow_html=True)
    with right:
        logo_path = Path(__file__).resolve().parent / "partner_logos.png"
        if logo_path.is_file():
            encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            st.markdown(
                '<div class="logo-panel"><img alt="CARMA project partners" '
                f'src="data:image/png;base64,{encoded}" style="width:100%;height:auto;object-fit:contain"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="logo-panel"><b>CARMA · CropGet</b></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading compact crop lookup…")
def load_lookup(path_text: str) -> pd.DataFrame:
    """Load lookup.

    Args:
        path_text: Input value for `path_text` used by `load_lookup`. Expected type: `str`.

    Returns:
        object: The result of `frame.dropna`.

    Raises:
        ValueError: If the corresponding error condition is detected.
    """
    schema = pq.read_schema(path_text).names
    required = {"season", "grid_lat", "grid_lon", "matched_sample_id", "matched_lat", "matched_lon"}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"Lookup is missing columns: {sorted(missing)}")
    optional = {"cohort_year", "grid_id", "crop"}
    frame = pd.read_parquet(path_text, columns=sorted(required | (optional & set(schema))))
    for name in ("grid_lat", "grid_lon", "matched_lat", "matched_lon"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["season"] = frame["season"].astype(str).str.lower()
    if "cohort_year" in frame:
        frame["cohort_year"] = frame["cohort_year"].astype("string")
    return frame.dropna(subset=["grid_lat", "grid_lon", "matched_lat", "matched_lon"])


def drawing_bounds(drawing: dict) -> tuple[float, float, float, float] | None:
    """Handle drawing bounds.

    Args:
        drawing: Input value for `drawing` used by `drawing_bounds`. Expected type: `dict`.

    Returns:
        None: No value is returned.
        tuple: Tuple containing `min(lats)`, `min(lons)`, `max(lats)`, `max(lons)` in that order.
    """
    try:
        geom = drawing.get("geometry", {})
        if geom.get("type") != "Polygon":
            return None
        coords = geom["coordinates"][0]
        lons, lats = [float(p[0]) for p in coords], [float(p[1]) for p in coords]
        return min(lats), min(lons), max(lats), max(lons)
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def sample_box(bounds: tuple[float, float, float, float], count: int, rng: np.random.Generator) -> np.ndarray:
    """Handle sample box.

    Args:
        bounds: Input value for `bounds` used by `sample_box`. Expected type: `tuple[float, float, float, float]`.
        count: Number of items to use. Expected type: `int`.
        rng: Input value for `rng` used by `sample_box`. Expected type: `np.random.Generator`.

    Returns:
        object: The result of `np.column_stack`.
    """
    south, west, north, east = bounds
    sin_lat = rng.uniform(np.sin(np.radians(south)), np.sin(np.radians(north)), size=count)
    return np.column_stack([np.degrees(np.arcsin(sin_lat)), rng.uniform(west, east, size=count)])


def nearest_matches(lookup: pd.DataFrame, points: pd.DataFrame, crop: str, season: str, year: str) -> pd.DataFrame:
    """Handle nearest matches.

    Args:
        lookup: Input value for `lookup` used by `nearest_matches`. Expected type: `pd.DataFrame`.
        points: Input value for `points` used by `nearest_matches`. Expected type: `pd.DataFrame`.
        crop: Crop identifier. Expected type: `str`.
        season: Season identifier. Expected type: `str`.
        year: Input value for `year` used by `nearest_matches`. Expected type: `str`.

    Returns:
        object: The result of `pd.DataFrame`.
        object: The result of `pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()`.
    """
    subset = lookup.loc[lookup.season.eq(season.lower())].copy()
    if year != "All years" and "cohort_year" in subset:
        subset = subset.loc[subset.cohort_year.astype(str).eq(year)]
    if subset.empty:
        return pd.DataFrame()
    groups = list(subset.groupby("cohort_year", dropna=False, sort=True)) if year == "All years" and "cohort_year" in subset else [(year, subset)]
    qrad = np.radians(points[["query_lat", "query_lon"]].to_numpy(dtype=float))
    parts = []
    for cohort_year, year_frame in groups:
        tree = BallTree(np.radians(year_frame[["grid_lat", "grid_lon"]].to_numpy(dtype=float)), metric="haversine")
        distances, indices = tree.query(qrad, k=1)
        nearest = year_frame.iloc[indices[:, 0]].reset_index(drop=True)
        out = points.reset_index(drop=True).copy()
        out["crop"], out["season"] = crop, season
        if "cohort_year" in nearest:
            out["cohort_year"] = nearest.cohort_year.astype(str).to_numpy()
        elif cohort_year != "All years":
            out["cohort_year"] = str(cohort_year)
        out["distance_to_grid_km"] = distances[:, 0] * 6371.0088
        for col in ("grid_id", "grid_lat", "grid_lon", "matched_sample_id", "matched_lat", "matched_lon"):
            if col in nearest:
                out[col] = nearest[col].to_numpy()
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _duplicate_query_ids(matches: pd.DataFrame) -> list[str]:
    """Handle duplicate query ids.

    Args:
        matches: Input value for `matches` used by `_duplicate_query_ids`. Expected type: `pd.DataFrame`.

    Returns:
        list: List of the values constructed in this function.
        object: The result of `matches.loc[duplicate_rows, 'query_id'].astype(str).drop_duplicates().tolist`.
    """
    if matches.empty or "matched_sample_id" not in matches:
        return []
    # A sample ID counts as repeated anywhere in this crop's final result set,
    # including when the same ID is present under more than one cohort year.
    keys = ["crop", "matched_sample_id"]
    duplicate_rows = matches.duplicated(keys, keep="first")
    return matches.loc[duplicate_rows, "query_id"].astype(str).drop_duplicates().tolist()


def find_location_matches(
    asset_root: Path,
    points: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    season: str,
    year: str,
    *,
    retry_duplicates: bool,
    max_rounds: int = 25,
    seed: int = 42,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find nearest references and retry duplicate queries with new AOI points.

    Args:
        asset_root: Input value for `asset_root` used by `find_location_matches`. Expected type: `Path`.
        points: Input value for `points` used by `find_location_matches`. Expected type: `pd.DataFrame`.
        bounds: Input value for `bounds` used by `find_location_matches`. Expected type: `tuple[float, float, float, float]`.
        season: Season identifier. Expected type: `str`.
        year: Input value for `year` used by `find_location_matches`. Expected type: `str`.
        retry_duplicates: Input value for `retry_duplicates` used by `find_location_matches`. Expected type: `bool`.
        max_rounds: Input value for `max_rounds` used by `find_location_matches`. Expected type: `int`.
        seed: Random seed used for reproducible sampling or model training. Expected type: `int`.
        progress: Input value for `progress` used by `find_location_matches`.

    Returns:
        tuple: Tuple containing `final_points`, `matches` in that order.
    """
    rng = np.random.default_rng(seed)
    final_points = points.copy().reset_index(drop=True)
    match_parts = []
    retries: dict[str, int] = {str(qid): 0 for qid in final_points["query_id"]}
    crops = final_points["crop"].drop_duplicates().tolist()
    for crop_index, crop in enumerate(crops, start=1):
        lookup_path = asset_root / "lightweight_top1_lookup" / crop / "top1_lookup.parquet"
        lookup = load_lookup(str(lookup_path))
        crop_points = final_points.loc[final_points.crop.eq(crop)].copy()
        found = nearest_matches(lookup, crop_points, crop, season, year)
        if not found.empty:
            found["retry_attempts"] = 0
        rounds = range(1, max_rounds + 1) if retry_duplicates else ()
        for round_number in rounds:
            duplicate_ids = _duplicate_query_ids(found)
            retry_ids = [qid for qid in duplicate_ids if retries.get(qid, 0) < max_rounds]
            if not retry_ids:
                break
            retry_mask = final_points.query_id.astype(str).isin(retry_ids) & final_points.crop.eq(crop)
            new_coords = sample_box(bounds, int(retry_mask.sum()), rng)
            final_points.loc[retry_mask, "query_lat"] = new_coords[:, 0]
            final_points.loc[retry_mask, "query_lon"] = new_coords[:, 1]
            for qid in retry_ids:
                retries[qid] = retries.get(qid, 0) + 1
            replacement = nearest_matches(lookup, final_points.loc[retry_mask], crop, season, year)
            if not replacement.empty:
                replacement["retry_attempts"] = replacement.query_id.astype(str).map(retries).fillna(0).astype(int)
                found = pd.concat([found.loc[~found.query_id.astype(str).isin(retry_ids)], replacement], ignore_index=True)
            if progress is not None:
                progress((crop_index - 1 + round_number / max_rounds) / max(1, len(crops)),
                         f"{crop.title()} · retry {round_number}/{max_rounds}")
        if not found.empty:
            match_parts.append(found)
        if progress is not None:
            progress(crop_index / max(1, len(crops)), f"{crop.title()} complete")
    matches = pd.concat(match_parts, ignore_index=True) if match_parts else pd.DataFrame()
    if not matches.empty:
        keys = ["crop", "matched_sample_id"]
        matches["duplicate_match"] = matches.duplicated(keys, keep="first")
        matches["retry_attempts"] = matches.query_id.astype(str).map(retries).fillna(0).astype(int)
    return final_points, matches


def _add_marker(group: folium.FeatureGroup, lat: float, lon: float, color: str, label: str, radius: int) -> None:
    """Handle add marker.

    Args:
        group: Input value for `group` used by `_add_marker`. Expected type: `folium.FeatureGroup`.
        lat: Latitude in decimal degrees. Expected type: `float`.
        lon: Longitude in decimal degrees. Expected type: `float`.
        color: Input value for `color` used by `_add_marker`. Expected type: `str`.
        label: Input value for `label` used by `_add_marker`. Expected type: `str`.
        radius: Input value for `radius` used by `_add_marker`. Expected type: `int`.
    """
    folium.CircleMarker([lat, lon], radius=radius, color=color, weight=1.5,
        fill=True, fill_color=color, fill_opacity=.82, tooltip=label).add_to(group)


def crop_map_legend(crops: list[str]) -> None:
    """Show crop colors and distinguish retrieved circles from query diamonds.

    Args:
        crops: Input value for `crops` used by `crop_map_legend`. Expected type: `list[str]`.
    """
    crop_items = "".join(
        '<span style="display:inline-flex;align-items:center;gap:7px;margin:3px 14px 3px 0;white-space:nowrap">'
        f'<i style="display:inline-block;width:13px;height:13px;border-radius:50%;background:{CROP_COLORS.get(crop, "#315C9B")};border:1px solid #263F46"></i>'
        f'<b>{crop.title()}</b></span>'
        for crop in crops
    )
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;'
        'background:#fff;border:1px solid #DCE5E3;border-radius:10px;padding:9px 14px;'
        'margin:8px 0 10px;color:#263F46">'
        f'<div style="display:flex;flex-wrap:wrap;align-items:center">{crop_items}</div>'
        '<span style="display:inline-flex;align-items:center;gap:7px;margin-right:12px;white-space:nowrap">'
        '<i style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#72858A;border:1px solid #263F46"></i>'
        'Retrieved reference</span>'
        '<span style="display:inline-flex;align-items:center;gap:8px;white-space:nowrap">'
        '<i style="display:inline-block;width:11px;height:11px;background:#72858A;border:1px solid #263F46;transform:rotate(45deg)"></i>'
        'Uploaded query</span></div>',
        unsafe_allow_html=True,
    )


def make_map(bounds=None, query_points=None, matches=None, *, mode1_crop: str | None = None, draw_enabled: bool = False):
    """Create map.

    Args:
        bounds: Input value for `bounds` used by `make_map`.
        query_points: Input value for `query_points` used by `make_map`.
        matches: Input value for `matches` used by `make_map`.
        mode1_crop: Input value for `mode1_crop` used by `make_map`. Expected type: `str | None`.
        draw_enabled: Input value for `draw_enabled` used by `make_map`. Expected type: `bool`.

    Returns:
        object: The computed `fmap` value.
    """
    center = [20, 0] if bounds is None else [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
    fmap = folium.Map(location=center, zoom_start=2 if bounds is None else 6,
        tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)
    if draw_enabled:
        Draw(
            position="topleft",
            draw_options={"polyline": False, "polygon": False, "circle": False,
                          "circlemarker": False, "marker": False, "rectangle": True},
            edit_options={"edit": True, "remove": True},
        ).add_to(fmap)
    if bounds is not None:
        south, west, north, east = bounds
        folium.Rectangle([[south, west], [north, east]], color="#D9A823", weight=3, fill=False).add_to(fmap)
    if matches is not None and not matches.empty:
        crop_col = "crop" if "crop" in matches else None
        for crop, part in (matches.groupby(crop_col, sort=False) if crop_col else [(mode1_crop or "maize", matches)]):
            color = CROP_COLORS.get(str(crop), "#315C9B")
            group = folium.FeatureGroup(name=f"{str(crop).title()} · retrieved", show=True)
            seen = set()
            for row in part.itertuples(index=False):
                key = (getattr(row, "cohort_year", None), getattr(row, "retrieved_sample_id", getattr(row, "matched_sample_id", None)))
                if key in seen:
                    continue
                seen.add(key)
                lat = getattr(row, "retrieved_lat", getattr(row, "matched_lat", None))
                lon = getattr(row, "retrieved_lon", getattr(row, "matched_lon", None))
                if pd.notna(lat) and pd.notna(lon):
                    sid = getattr(row, "retrieved_sample_id", getattr(row, "matched_sample_id", ""))
                    _add_marker(group, float(lat), float(lon), color, f"Retrieved · {crop} · {sid}", 5)
            group.add_to(fmap)
    # Draw query points last so they stay visually on top of the more numerous
    # retrieved points. Diamonds identify queries; circles identify references.
    if query_points is not None and not query_points.empty:
        for crop, part in query_points.groupby("crop", sort=False):
            color = CROP_COLORS.get(str(crop), "#315C9B")
            group = folium.FeatureGroup(name=f"{str(crop).title()} · query", show=True)
            for row in part.itertuples(index=False):
                label = f"Query · {crop} · {getattr(row, 'query_id', getattr(row, 'query_sample_id', ''))}"
                folium.Marker(
                    [row.query_lat, row.query_lon],
                    icon=folium.DivIcon(
                        icon_size=(14, 14), icon_anchor=(7, 7),
                        html=f'<div style="width:11px;height:11px;background:{color};border:2px solid #17343B;transform:rotate(45deg);box-shadow:0 1px 3px #333"></div>',
                    ),
                    tooltip=label,
                ).add_to(group)
            group.add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def duplicate_summary(frame: pd.DataFrame, id_col: str, *, crop_col: str = "crop", group_by_year: bool = False) -> pd.DataFrame:
    """Handle duplicate summary.

    Args:
        frame: Input value for `frame` used by `duplicate_summary`. Expected type: `pd.DataFrame`.
        id_col: Input value for `id_col` used by `duplicate_summary`. Expected type: `str`.
        crop_col: Input value for `crop_col` used by `duplicate_summary`. Expected type: `str`.
        group_by_year: Input value for `group_by_year` used by `duplicate_summary`. Expected type: `bool`.

    Returns:
        object: The result of `pd.DataFrame`.
    """
    if frame.empty or id_col not in frame:
        return pd.DataFrame(columns=["crop", "matches", "duplicate_rows", "duplicate_pct"])
    group_cols = [crop_col] if crop_col in frame else []
    if group_by_year and "cohort_year" in frame:
        group_cols.append("cohort_year")
    if not group_cols:
        frame = frame.copy()
        frame["crop"] = "selected crop"
        group_cols = ["crop"]
    rows = []
    for keys, part in frame.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        crop = str(keys[group_cols.index(crop_col)]) if crop_col in group_cols else "selected crop"
        ids = part[id_col].dropna().astype(str)
        total = len(ids)
        duplicates = total - ids.nunique()
        rows.append({"crop": crop, "cohort_year": str(keys[group_cols.index("cohort_year")]) if "cohort_year" in group_cols else "",
            "matches": total, "duplicate_rows": duplicates, "duplicate_pct": (100 * duplicates / total) if total else 0.0})
    return pd.DataFrame(rows)


def _coord_columns(frame: pd.DataFrame, kind: str) -> tuple[str | None, str | None]:
    """Handle coord columns.

    Args:
        frame: Input value for `frame` used by `_coord_columns`. Expected type: `pd.DataFrame`.
        kind: Input value for `kind` used by `_coord_columns`. Expected type: `str`.

    Returns:
        tuple: Tuple containing `next((lower[n] for n in lat_names if n in lower), None)`, `next((lower[n] for n in lon_names if n in lower), None)` in that order.
    """
    lower = {str(col).lower(): col for col in frame.columns}
    if kind == "query":
        lat_names = ("query_lat", "lat_x", "lat", "latitude", "query_latitude")
        lon_names = ("query_lon", "lon_x", "lon", "longitude", "query_longitude")
    else:
        lat_names = ("retrieved_lat", "matched_lat", "lat_x", "lat", "latitude")
        lon_names = ("retrieved_lon", "matched_lon", "lon_x", "lon", "longitude")
    return next((lower[n] for n in lat_names if n in lower), None), next((lower[n] for n in lon_names if n in lower), None)


def _normalize_id(value) -> str:
    """Normalize id.

    Args:
        value: Value tested, transformed, or compared by this function.

    Returns:
        str: The constant value ``.
        object: The computed `text` value.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("+-").isdigit():
        text = text[:-2]
    return text


def _column_case_insensitive(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    """Handle column case insensitive.

    Args:
        frame: Input value for `frame` used by `_column_case_insensitive`. Expected type: `pd.DataFrame`.
        names: Input value for `names` used by `_column_case_insensitive`. Expected type: `tuple[str, ...]`.

    Returns:
        object: The result of `next`.
    """
    lower = {str(name).lower(): name for name in frame.columns}
    return next((lower[name] for name in names if name in lower), None)


def _selected_train_lookup(root: Path, crop: str, season: str) -> Path:
    """Resolve the top-25 source; the requested K is applied after lookup.

    Args:
        root: Root directory containing the relevant project or data files. Expected type: `Path`.
        crop: Crop identifier. Expected type: `str`.
        season: Season identifier. Expected type: `str`.

    Returns:
        object: The computed `path` value.
    """
    base = root.parent if root.name.lower() in {"top10", "top25"} else root
    path = base / "top25" / crop / f"season={season}.parquet"
    if not path.is_file() and root.name.lower() == "top25":
        path = root / crop / f"season={season}.parquet"
    return path


def _browse_train_lookup_folder() -> None:
    """Open a local folder picker and store the selected lookup root."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        current = Path(st.session_state.get("m1_lookup_root", str(DEFAULT_TRAIN_LOOKUP_ROOT))).expanduser()
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Select CropGET top-25 lookup folder",
            initialdir=str(current if current.is_dir() else Path.home()),
            mustexist=True,
        )
        root.destroy()
        if selected:
            st.session_state["m1_lookup_root"] = selected
        st.session_state.pop("m1_lookup_browse_error", None)
    except Exception as exc:
        st.session_state["m1_lookup_browse_error"] = str(exc)


@st.cache_data(show_spinner="Loading precomputed crop lookup…")
def load_train_point_lookup(path_text: str) -> pd.DataFrame:
    """Load train point lookup.

    Args:
        path_text: Input value for `path_text` used by `load_train_point_lookup`. Expected type: `str`.

    Returns:
        object: The computed `frame` value.

    Raises:
        ValueError: If the corresponding error condition is detected.
    """
    columns = pq.read_schema(path_text).names
    required = {
        "query_ref_id", "query_sample_id", "neighbor_rank", "matched_sample_id",
        "matched_lat", "matched_lon", "similarity", "spectral_similarity",
    }
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Training-point lookup is missing columns: {sorted(missing)}")
    optional = {"query_lat", "query_lon", "crop", "season", "query_year"}
    frame = pd.read_parquet(path_text, columns=sorted(required | (optional & set(columns))))
    frame["_ref_key"] = frame["query_ref_id"].map(_normalize_id)
    frame["_sample_key"] = frame["query_sample_id"].map(_normalize_id)
    for name in ("query_lat", "query_lon", "matched_lat", "matched_lon", "similarity", "spectral_similarity"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["neighbor_rank"] = pd.to_numeric(frame["neighbor_rank"], errors="coerce").astype("Int64")
    return frame


def _uploaded_query_frame(uploaded_file) -> tuple[pd.DataFrame, str, str, int]:
    """Handle uploaded query frame.

    Args:
        uploaded_file: Path to the uploaded used by this function.

    Returns:
        tuple: Tuple containing `frame`, `ref_col`, `sample_col`, `source_rows` in that order.

    Raises:
        ValueError: If the corresponding error condition is detected.
    """
    frame = pd.read_parquet(io.BytesIO(uploaded_file.getvalue()))
    source_rows = len(frame)
    ref_col = _column_case_insensitive(frame, ("ref_id",))
    sample_col = _column_case_insensitive(frame, ("sample_id", "sampleid"))
    if ref_col is None or sample_col is None:
        raise ValueError("Uploaded Parquet must contain ref_id and sample_id columns.")
    frame = frame.copy()
    frame["_input_row"] = np.arange(len(frame), dtype=np.int64)
    frame["query_ref_id"] = frame[ref_col].map(_normalize_id)
    frame["query_sample_id"] = frame[sample_col].map(_normalize_id)
    frame["_ref_key"] = frame["query_ref_id"]
    frame["_sample_key"] = frame["query_sample_id"]
    qlat = _column_case_insensitive(frame, ("latitude", "lat", "lat_x", "query_lat"))
    qlon = _column_case_insensitive(frame, ("longitude", "lon", "lon_x", "query_lon"))
    frame["_uploaded_query_lat"] = pd.to_numeric(frame[qlat], errors="coerce") if qlat else np.nan
    frame["_uploaded_query_lon"] = pd.to_numeric(frame[qlon], errors="coerce") if qlon else np.nan
    frame = frame.loc[frame["_ref_key"].ne("") & frame["_sample_key"].ne("")].copy()
    # WorldCereal Parquets can have one row per acquisition for the same
    # sample. Collapse those temporal rows before joining the point lookup.
    # Keep the first row's IDs and first non-null coordinates.
    keys = ["_ref_key", "_sample_key"]
    frame = frame.sort_values("_input_row", kind="stable")
    frame["_first_query_lat"] = frame.groupby(keys, sort=False)["_uploaded_query_lat"].transform("first")
    frame["_first_query_lon"] = frame.groupby(keys, sort=False)["_uploaded_query_lon"].transform("first")
    frame = frame.drop_duplicates(keys, keep="first").copy()
    frame["_uploaded_query_lat"] = frame.pop("_first_query_lat")
    frame["_uploaded_query_lon"] = frame.pop("_first_query_lon")
    return frame, ref_col, sample_col, source_rows


def _search_queries(queries: pd.DataFrame, lookup_path: Path, crop: str, top_k: int, sort_by: str) -> tuple[pd.DataFrame, int, int]:
    """Search queries.

    Args:
        queries: Query samples processed by this function. Expected type: `pd.DataFrame`.
        lookup_path: Path to the lookup used by this function. Expected type: `Path`.
        crop: Crop identifier. Expected type: `str`.
        top_k: Maximum number of top-ranked results to return. Expected type: `int`.
        sort_by: Input value for `sort_by` used by `_search_queries`. Expected type: `str`.

    Returns:
        tuple: Tuple containing `pd.DataFrame()`, `missing`, `len(requested)` in that order.
        tuple: Tuple containing `joined`, `missing`, `len(requested)` in that order.

    Raises:
        ValueError: If the corresponding error condition is detected.
    """
    if queries.empty:
        raise ValueError("The uploaded Parquet contains no valid ref_id/sample_id pairs.")
    lookup = load_train_point_lookup(str(lookup_path))
    lookup = lookup.loc[lookup.neighbor_rank.le(int(top_k))].copy()
    keys = ["_ref_key", "_sample_key"]
    available = lookup[keys].drop_duplicates()
    requested = queries[keys].drop_duplicates()
    missing = len(requested.merge(available, on=keys, how="left", indicator=True).query("_merge == 'left_only'"))

    joined = queries.merge(lookup, on=keys, how="inner", suffixes=("", "_lookup"), validate="many_to_many")
    if joined.empty:
        return pd.DataFrame(), missing, len(requested)
    lookup_lat = joined["query_lat_lookup"] if "query_lat_lookup" in joined else joined.get("query_lat", pd.Series(np.nan, index=joined.index))
    lookup_lon = joined["query_lon_lookup"] if "query_lon_lookup" in joined else joined.get("query_lon", pd.Series(np.nan, index=joined.index))
    joined["query_lat"] = joined["_uploaded_query_lat"].where(joined["_uploaded_query_lat"].notna(), lookup_lat)
    joined["query_lon"] = joined["_uploaded_query_lon"].where(joined["_uploaded_query_lon"].notna(), lookup_lon)
    # Keep a compact, app-ready schema; the parquet itself remains unchanged.
    joined = joined.rename(columns={
        "matched_sample_id": "retrieved_sample_id",
        "matched_lat": "retrieved_lat",
        "matched_lon": "retrieved_lon",
        "neighbor_rank": "retrieval_rank",
    })
    joined["crop"] = str(lookup_path.parent.name).lower()
    season_name = lookup_path.name.split("=", 1)[-1].split(".", 1)[0]
    joined["season"] = season_name
    keep = [
        "_input_row", "query_ref_id", "query_sample_id", "query_lat", "query_lon",
        "crop", "season", "retrieval_rank", "retrieved_sample_id", "retrieved_lat",
        "retrieved_lon", "similarity", "spectral_similarity",
    ]
    if "query_year" in joined:
        keep.append("query_year")
    joined = joined[[name for name in keep if name in joined.columns]]
    order_cols = ["_input_row", "spectral_similarity"] if sort_by == "Spectral similarity (highest first)" else ["_input_row", "retrieval_rank"]
    ascending = [True, False] if sort_by == "Spectral similarity (highest first)" else [True, True]
    joined = joined.sort_values(order_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    return joined, missing, len(requested)


def mode1_tab() -> None:
    """Handle mode1 tab.

    Returns:
        None: No value is returned.

    Raises:
        FileNotFoundError: If the corresponding error condition is detected.
        ValueError: If the corresponding error condition is detected.
    """
    st.subheader("1. Retrieve other relevant data for your reference data")
    st.caption("Upload a Parquet with ref_id and sample_id. The app joins those IDs to the precomputed crop-season neighbors; it does not encode the upload or search the model memory.")
    with st.expander("Lookup folder", expanded=False):
        path_col, browse_col = st.columns([8, 1], vertical_alignment="bottom")
        with path_col:
            lookup_text = st.text_input("Top-25 lookup root", str(DEFAULT_TRAIN_LOOKUP_ROOT), key="m1_lookup_root")
        with browse_col:
            st.button("Browse…", key="m1_browse_lookup_folder", on_click=_browse_train_lookup_folder, use_container_width=True)
        browse_error = st.session_state.pop("m1_lookup_browse_error", None)
        if browse_error:
            st.warning(f"Folder picker unavailable: {browse_error}. You can still paste or edit the path above.")
        st.caption("Use the lookup parent folder or point directly to its top25 folder. The app filters the stored top-25 list to your requested K.")
    c1, c2, c3 = st.columns(3)
    with c1:
        crops = st.multiselect("Crops", TRAIN_POINT_CROPS, default=["maize"], key="m1_crops")
    with c2:
        season = st.selectbox("Season", ["s1", "s2"], key="m1_season")
    with c3:
        top_k = st.number_input("Neighbors per query (1–25)", min_value=1, max_value=25, value=10, step=1, key="m1_topk")
    sort_by = st.selectbox("Order results by", ["Retrieval rank", "Spectral similarity (highest first)"], key="m1_sort")
    uploaded_file = st.file_uploader("Upload WorldCereal Parquet", type=["parquet"], key="m1_upload")
    lookup_paths = {crop: _selected_train_lookup(Path(lookup_text), crop, season) for crop in crops}
    if crops:
        st.caption("Selected top-25 lookup files: " + " · ".join(f"{crop}: {path}" for crop, path in lookup_paths.items()))
    if uploaded_file is None:
        st.info("Upload a Parquet containing ref_id and sample_id to look up its precomputed neighbors.")
        return
    if not crops:
        st.info("Select at least one crop.")
        return
    if st.button("Find precomputed neighbors", type="primary", key="run_mode1"):
        st.session_state.pop("m1_matches", None)
        try:
            absent = [f"{crop}: {path}" for crop, path in lookup_paths.items() if not path.is_file()]
            if absent:
                raise FileNotFoundError("Lookup file(s) not found:\n" + "\n".join(absent))
            queries, _, _, source_rows = _uploaded_query_frame(uploaded_file)
            if queries.empty:
                raise ValueError("The uploaded Parquet contains no valid ref_id/sample_id pairs.")
            with st.spinner("Matching uploaded IDs to the precomputed lookup…"):
                match_parts, missing_by_crop, query_count = [], {}, len(queries)
                for crop, lookup_path in lookup_paths.items():
                    found, missing, _ = _search_queries(queries, lookup_path, crop, int(top_k), sort_by)
                    missing_by_crop[crop] = missing
                    if not found.empty:
                        match_parts.append(found)
                matches = pd.concat(match_parts, ignore_index=True) if match_parts else pd.DataFrame()
            st.session_state["m1_matches"] = matches
            st.session_state["m1_crops_success"] = crops
            st.session_state["m1_missing_by_crop"] = missing_by_crop
            st.session_state["m1_query_count"] = query_count
            st.session_state["m1_source_rows"] = source_rows
            st.session_state["m1_topk_success"] = int(top_k)
            st.session_state["m1_season_success"] = season
            st.session_state["m1_sort_success"] = sort_by
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

    matches = st.session_state.get("m1_matches", pd.DataFrame())
    result_is_current = (
        st.session_state.get("m1_crops_success") == crops
        and st.session_state.get("m1_season_success") == season
        and st.session_state.get("m1_topk_success") == int(top_k)
        and st.session_state.get("m1_sort_success") == sort_by
    )
    if result_is_current and matches.empty:
        st.warning("No uploaded ref_id/sample_id pairs matched this crop and season lookup.")
    if result_is_current and not matches.empty:
        st.success("Lookup completed.")
        duplicate_rows = max(0, int(st.session_state.get("m1_source_rows", 0)) - int(st.session_state.get("m1_query_count", 0)))
        if duplicate_rows:
            st.caption(f"Collapsed {duplicate_rows:,} repeated/invalid Parquet rows to one lookup per unique ref_id/sample_id pair.")
        missing_by_crop = st.session_state.get("m1_missing_by_crop", {})
        missing_parts = [f"{crop}: {count:,}/{st.session_state['m1_query_count']:,} not found" for crop, count in missing_by_crop.items() if count]
        if missing_parts:
            st.warning("Unmatched uploaded IDs by crop — " + " · ".join(missing_parts))
        metrics = st.columns(3)
        metrics[0].metric("Uploaded query IDs", f"{st.session_state['m1_query_count']:,}")
        metrics[1].metric("Matched query IDs", f"{matches[['crop', 'query_ref_id', 'query_sample_id']].drop_duplicates().shape[0]:,}")
        unique_retrieved_ids = matches["retrieved_sample_id"].nunique(dropna=True)
        metrics[2].metric("Unique retrieved IDs", f"{unique_retrieved_ids:,}")
        st.caption("The results table keeps each query–neighbor match; this count deduplicates retrieved sample IDs across queries and selected crops.")
        show_all_map_points = st.checkbox(
            "Display all matched points on the map",
            value=False,
            key="m1_show_all_map_points",
        )
        map_matches = matches
        if not show_all_map_points and len(matches) > 5000:
            map_matches = matches.groupby(
                ["crop", "query_ref_id", "query_sample_id"],
                sort=False,
                group_keys=False,
            ).head(max(1, 5000 // int(top_k)))
            st.caption("Showing a 5,000-row map preview. The downloaded CSV contains every match.")
        elif show_all_map_points and len(matches) > 50000:
            st.warning(
                f"Rendering all {len(matches):,} query-neighbor rows may be slow or make your browser unresponsive."
            )
        if map_matches[["query_lat", "query_lon", "retrieved_lat", "retrieved_lon"]].notna().all(axis=1).any():
            qmap = map_matches.drop_duplicates(["crop", "query_ref_id", "query_sample_id"])
            crop_map_legend(sorted(map_matches["crop"].dropna().astype(str).unique(), key=str.lower))
            st_folium(make_map(query_points=qmap, matches=map_matches),
                height=500, use_container_width=True, returned_objects=[], key="mode1_results_map")
        show_cols = [c for c in (
            "query_ref_id", "query_sample_id", "query_lat", "query_lon", "crop", "season",
            "retrieval_rank", "retrieved_sample_id", "retrieved_lat", "retrieved_lon",
            "similarity", "spectral_similarity",
        ) if c in matches.columns]
        st.dataframe(matches[show_cols].head(10000), use_container_width=True, hide_index=True)
        if len(matches) > 10000:
            st.caption("Showing the first 10,000 rows. Download includes all matched rows.")
        st.download_button("Download neighbor table", matches[show_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"cropget_{'-'.join(crops)}_{season}_top{st.session_state['m1_topk_success']}_matches.csv",
            mime="text/csv", key="download_mode1")

def mode3_tab(asset_root: Path, available: list[str]) -> None:
    """Handle mode3 tab.

    Args:
        asset_root: Input value for `asset_root` used by `mode3_tab`. Expected type: `Path`.
        available: Input value for `available` used by `mode3_tab`. Expected type: `list[str]`.

    Returns:
        None: No value is returned.
    """
    st.subheader("Location lookup (Experimental)")
    with st.expander("Location lookup folder", expanded=False):
        asset_root = Path(st.text_input("Asset folder", str(asset_root), key="asset_root_location"))
        available = [crop for crop in CROPS if (asset_root / "lightweight_top1_lookup" / crop / "top1_lookup.parquet").is_file()]
        st.caption("Available location crops: " + (", ".join(available) if available else "none found"))
    if not available:
        st.error("No crop lookup Parquets found. Expected lightweight_top1_lookup/<crop>/top1_lookup.parquet under the asset folder.")
        return
    season_col, year_col, seed_col = st.columns([1, 1, 1])
    with season_col:
        season = st.selectbox("Season", ["s1", "s2"], key="m3_season")
    with year_col:
        year = st.selectbox("Cohort year", ["All years", "2022", "2023", "2024"], key="m3_year")
    with seed_col:
        seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, step=1, key="m3_seed")
    st.markdown('<div class="section-label">Draw search area</div>', unsafe_allow_html=True)
    bounds = st.session_state.get("m3_bounds")
    prior_matches = st.session_state.get("m3_matches", pd.DataFrame())
    draw_queries = None if not prior_matches.empty else st.session_state.get("m3_queries")
    map_state = st_folium(make_map(bounds, draw_queries, prior_matches, draw_enabled=True),
        height=510, use_container_width=True, returned_objects=["all_drawings", "last_active_drawing"], key="m3_draw_map")
    drawings = map_state.get("all_drawings") or []
    if map_state.get("last_active_drawing"):
        drawings = [*drawings, map_state["last_active_drawing"]]
    for drawing in reversed(drawings):
        parsed = drawing_bounds(drawing)
        if parsed:
            bounds = parsed
            st.session_state["m3_bounds"] = parsed
            break
    if bounds:
        south, west, north, east = bounds
        st.caption(f"Area: {south:.2f}–{north:.2f}° N · {west:.2f}–{east:.2f}° E")
    else:
        st.info("Use the rectangle tool on the map.")
    st.markdown('<div class="section-label">Points per crop</div>', unsafe_allow_html=True)
    count_cols = st.columns(max(1, len(available)))
    counts = {}
    for col, crop in zip(count_cols, available):
        with col:
            counts[crop] = st.number_input(f"{crop.title()} query points", min_value=0, max_value=10000, value=5, step=1, key=f"m3_count_{crop}")
    max_retries = int(st.number_input(
        "Maximum retries per duplicate point",
        min_value=0,
        max_value=100,
        value=25,
        step=1,
        help="Retries stop early once every final matched sample ID is unique. Set to 0 to disable retries.",
        key="m3_max_retries",
    ))
    if st.button("Generate query points", type="primary", disabled=bounds is None, key="m3_generate"):
        rng, parts = np.random.default_rng(int(seed)), []
        for crop, count in counts.items():
            pts = sample_box(bounds, int(count), rng)
            if len(pts):
                parts.append(pd.DataFrame({"crop":crop, "query_id":[f"{crop}_{i+1:04d}" for i in range(len(pts))],
                    "query_lat":pts[:,0], "query_lon":pts[:,1]}))
        st.session_state["m3_queries"] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        st.session_state.pop("m3_matches", None)
        st.session_state.pop("m3_success", None)
    queries = st.session_state.get("m3_queries", pd.DataFrame())
    if not queries.empty:
        st.dataframe(queries, use_container_width=True, hide_index=True)
        if st.button("Find references by location", type="primary", key="m3_find"):
            progress = st.progress(0, text="Finding references…")
            try:
                final_points, found = find_location_matches(
                    asset_root, queries, bounds, season, year,
                    retry_duplicates=max_retries > 0, max_rounds=max_retries, seed=int(seed),
                    progress=lambda value, message: progress.progress(min(1.0, value), text=message),
                )
                progress.empty()
                st.session_state["m3_queries"] = final_points
                st.session_state["m3_matches"] = found
                st.session_state["m3_success"] = not found.empty
                st.session_state["m3_retry_limit_used"] = max_retries
                if found.empty:
                    st.warning("No references found for this season and year.")
            except Exception as exc:
                progress.empty()
                st.session_state["m3_success"] = False
                st.error(f"Location search failed: {exc}")
        queries = st.session_state.get("m3_queries", pd.DataFrame())
    matches = st.session_state.get("m3_matches", pd.DataFrame())
    if not matches.empty:
        if st.session_state.get("m3_success"):
            st.success("Find completed successfully.")
        st.markdown('<div class="section-label">3 · Retrieved references</div>', unsafe_allow_html=True)
        # Report repeats across the final set by sample ID, independent of year.
        duplicate_key = ["crop", "matched_sample_id"]
        duplicate_mask = matches.duplicated(duplicate_key, keep="first")
        summaries = matches.assign(_duplicate=duplicate_mask).groupby("crop", sort=False).agg(
            matches=("matched_sample_id", "size"), duplicate_rows=("_duplicate", "sum")
        ).reset_index()
        retried = matches.loc[matches.retry_attempts.gt(0)].groupby("crop")["query_id"].nunique()
        summaries["retried_points"] = summaries.crop.map(retried).fillna(0).astype(int)
        summaries["duplicate_pct"] = 100 * summaries["duplicate_rows"] / summaries["matches"].clip(lower=1)
        stat_cols = st.columns(max(1, len(summaries)))
        for col, row in zip(stat_cols, summaries.itertuples(index=False)):
            col.metric(f"{row.crop.title()} · {row.matches:,} final matches", f"{row.duplicate_pct:.1f}% repeated",
                       f"{row.duplicate_rows:,} of {row.matches:,} rows · {row.retried_points:,} points retried")
        retry_limit_used = int(st.session_state.get("m3_retry_limit_used", max_retries))
        st.caption(
            "Final repeat rate = repeated matched sample ID rows ÷ final matches. "
            f"Retry limit: {retry_limit_used} per duplicate point; stops early when no repeats remain."
        )
        result_map = make_map(bounds, None, matches)
        st_folium(result_map, height=490, use_container_width=True, returned_objects=[], key="m3_results_map")
        cols = [c for c in ("crop", "season", "cohort_year", "query_id", "query_lat", "query_lon", "distance_to_grid_km",
            "grid_id", "grid_lat", "grid_lon", "matched_sample_id", "matched_lat", "matched_lon", "retry_attempts", "duplicate_match") if c in matches.columns]
        st.dataframe(matches[cols], use_container_width=True, hide_index=True)
        st.download_button("Download Mode 3 match table", matches.to_csv(index=False).encode("utf-8"),
            file_name="cropget_mode3_location_matches.csv", mime="text/csv", key="download_mode3")


def _navigate_to(page: str) -> None:
    """Update navigation from a button callback before the next app rerun.

    Args:
        page: Input value for `page` used by `_navigate_to`. Expected type: `str`.
    """
    st.session_state["nav_selection"] = page


def landing_page() -> None:
    """Implement landing page."""
    st.markdown(
        '<div style="text-align:center;padding:22px 12px 16px">'
        '<div class="section-label">CROPGET REFERENCE DATA</div>'
        '<h2 style="font-size:1.65rem;margin:6px 0 0">Find relevant reference data for your analysis</h2>'
        '</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            '<div class="home-card"><div class="section-label">1 · PRECOMPUTED LOOKUP</div>'
            '<h3>Retrieve other relevant data for your reference data</h3>'
            '<p>Upload a WorldCereal Parquet and match its ref_id/sample_id pairs to stored CropGet neighbors.</p></div>',
            unsafe_allow_html=True,
        )
        st.button("Open reference-data lookup", type="primary", use_container_width=True,
                  key="home_open_lookup", on_click=_navigate_to, args=("Reference lookup",))
    with right:
        st.markdown(
            '<div class="home-card"><div class="section-label">2 · EXPERIMENTAL</div>'
            '<h3>Retrieve relevant reference data, anywhere, anytime (Experimental)</h3>'
            '<p>Experimental retrieval across locations and reference datasets.</p></div>',
            unsafe_allow_html=True,
        )
        st.button("Open location lookup (Experimental)", use_container_width=True,
                  key="home_open_experimental", on_click=_navigate_to,
                  args=("Location lookup (Experimental)",))


def main() -> None:
    """Run the script entry point."""
    st.set_page_config(page_title="CropGet · Find", page_icon="🌱", layout="wide")
    apply_style()
    header()
    asset_root = DEFAULT_ASSET_ROOT
    pages = ["Home", "Reference lookup", "Location lookup (Experimental)"]
    if st.session_state.get("nav_selection") not in pages:
        st.session_state["nav_selection"] = "Home"
    selected = st.radio("Navigation", pages, horizontal=True,
                        label_visibility="collapsed", key="nav_selection")
    if selected == "Home":
        landing_page()
    elif selected == "Reference lookup":
        mode1_tab()
    elif selected == "Location lookup (Experimental)":
        lookup_crops = [crop for crop in CROPS if (asset_root / "lightweight_top1_lookup" / crop / "top1_lookup.parquet").is_file()]
        mode3_tab(asset_root, lookup_crops)
    else:
        st.subheader("Location lookup (Experimental)")


if __name__ == "__main__":
    main()
