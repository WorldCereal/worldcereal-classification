#!/usr/bin/env python3
"""Build balanced train/val/test splits for landcover and croptype datasets.

This script builds H3-based splits (level 3 primary), assigns splits to
sample_ids, enforces class coverage per split for both LC and CT, and writes
a unified parquet with sample_id -> split.
"""

from __future__ import annotations

import glob
import logging
from collections import Counter
from pathlib import Path
from typing import (Dict, Iterable, List, Mapping, Optional, Sequence, Set,
                    Tuple)

import duckdb
import h3
import numpy as np
import pandas as pd
from tqdm import tqdm
from worldcereal.utils.sharepoint import (build_class_mappings,
                                          get_excel_from_sharepoint)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_WORLD_BOUNDS_PATH = (
    REPO_ROOT / "src/worldcereal/data/world-" \
    "administrative-boundaries/world-administrative-boundaries.geoparquet"
)
DEFAULT_EXTRACTS_GLOB = (
    REPO_ROOT / "data/worldcereal_data/EXTRACTIONS/WORLDCEREAL/WORLDCEREAL_ALL_EXTRACTIONS/"
    "worldcereal_all_extractions.parquet/**/*.parquet"
)
DEFAULT_LEGEND_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/legend/"
    "WorldCereal_LC_CT_legend_latest.csv"
)
DEFAULT_UNIFIED_SPLITS_PATH = Path(
    "./unified_train_splits.parquet"
)

TRAIN_ONLY_REF_IDS: Tuple[str, ...] = (
    "2024_ARG_INTA-SUMMER_POINT_110",
    "2022_ARG_INTA-SUMMER_POINT_110",
    "2023_ARG_INTA-SUMMER_POINT_110",
    "2020_ARG_INTA-SUMMER_POINT_110",
    "2020_ARG_INTA-WINTER_POINT_110",
    "2022_ARG_INTA-WINTER_POINT_110",
    "2021_ARG_INTA-SUMMER_POINT_110",
    "2021_ARG_INTA-WINTER_POINT_110",
    "2023_ARG_INTA-WINTER_POINT_110",
    "2017_BRA_MAPBIOMAS-ZHENG_POINT_110",
    "2018_BRA_MAPBIOMAS-SONG_POINT_110",
    "2018_BRA_MAPBIOMAS-ZHENG_POINT_110",
    "2019_BRA_MAPBIOMAS-SONG_POINT_110",
    "2019_BRA_MAPBIOMAS-ZHENG_POINT_110",
    "2020_BRA_MAPBIOMAS-SONG_POINT_110",
    "2021_BRA_MAPBIOMAS-SONG_POINT_110",
    "2022_BRA_MAPBIOMAS-SONG_POINT_110",
    "2023_BRA_MAPBIOMAS-SONG_POINT_110",
    "2017_CHN_YOU-HAN-SHEN-RICE_POINT_110",
    "2018_CHN_LIU-ZANG_POINT_110",
    "2018_CHN_YOU-HAN-SHEN-RICE_POINT_110",
    "2019_CHN_LIU-ZANG_POINT_110",
    "2019_CHN_YOU-HAN-SHEN-RICE_POINT_110",
    "2019_CHN_YOU-LI-LI_POINT_110",
    "2019_CHN_YOU-MEI-SOYBEAN_POINT_110",
    "2020_CHN_DONG-HU-LIU-YANG_POINT_110",
    "2020_CHN_KANG_POINT_110",
    "2020_CHN_LIU-ZANG_POINT_110",
    "2021_CHN_HU-LIU-YANG_POINT_110",
    "2021_CHN_KANG_POINT_110",
    "2021_CHN_LIU-ZANG_POINT_110",
    "2022_CHN_HU-LIU-YANG_POINT_110",
    "2018_VNM_HAN-JAXA-LI_POINT_110",
    "2019_VNM_HAN-JAXA-LI-SUN_POINT_110",
    "2020_VNM_JAXA-LI_POINT_110",
    "2021_VNM_GINTING-LI_POINT_110",
    "2017_CHL_HAN_POINT_110",
    "2018_CHL_HAN_POINT_110",
    "2019_CHL_HAN_POINT_110",
    "2018_CRI_CENAT-OILPALMS_POINT_110",
    "2018_CRI_CENAT-PINEAPPLES_POINT_110",
    "2019_CRI_CENAT-OILPALMS_POINT_110",
    "2019_CRI_CENAT-PINEAPPLES_POINT_110",
    "2021_IDN_GINTING-LI_POINT_110",
    "2018_JPN_CARRASCO-HAN-JAXA-LI_POINT_110",
    "2019_JPN_CARRASCO-HAN-JAXA-LI_POINT_110",
    "2020_JPN_JAXA-LI_POINT_110",
    "2020_JPN_JAXA-OKINAWA_POINT_110",
    "2022_JPN_JAXA-LI_POINT_110",
    "2023_JPN_LI-SONG_POINT_110",
    "2021_KHM_GINTING-LI_POINT_110",
    "2018_KOR_HAN-JO-LI_POINT_110",
    "2019_KOR_HAN-JO-LI_POINT_110",
    "2020_KOR_JO-LI_POINT_110",
    "2021_KOR_JO-LI_POINT_110",
    "2023_KOR_LI-SONG_POINT_110",
    "2021_LAO_GINTING-LI_POINT_110",
    "2021_MMR_GINTING-LI_POINT_110",
    "2021_MYS_GINTING-LI_POINT_110",
    "2021_PHL_GINTING-LI_POINT_110",
    "2019_THA_BOKU_POINT_110",
    "2021_THA_GINTING-LI_POINT_110",
    "2022_URY_SIT-OAN_POINT_110",
    "2022_URY_SONG-OAN_POINT_110",
    "2024_HND_ICF-FAO_POINT_110",
    # datasets that are too valuable to not be in train
    "2023_ESP_Eurocrops_POLY_110",
)

IGNORE_LABEL = "ignore"
RECENT_YEARS = (2024, 2025)
TARGET_TRAIN_RATIO = 0.70
TARGET_VAL_RATIO = 0.15
TARGET_TEST_RATIO = 0.15
BUDGET_TOLERANCE_RATIO = 0.02
MAX_BUDGET_CORRECTION_MOVES = 200

LOGGER = logging.getLogger(__name__)
AMBIGUOUS_REF_COUNTRY_CODES = {"GLO"}


def init_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with the spatial extension loaded."""
    conn = duckdb.connect()
    conn.sql("INSTALL spatial")
    conn.load_extension("spatial")
    return conn


def load_mappings(
    class_mappings: Optional[dict] = None,
    lc_name: str = "LANDCOVER10",
    ct_name: str = "CROPTYPE27",
    sharepoint_site_url: Optional[str] = None,
    sharepoint_file_url: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load CROPTYPE and LANDCOVER mappings.

    Parameters
    ----------
    class_mappings : dict, optional
        Dictionary containing class mappings. If not provided, it is loaded
        from SharePoint using :func:`get_excel_from_sharepoint`.
    sharepoint_site_url : str, optional
        SharePoint site URL to fetch the legend/mappings Excel.
    sharepoint_file_url : str, optional
        Server-relative path to the Excel file.

    Returns
    -------
    tuple
        CROPTYPE mapping and LANDCOVER mapping.
    """
    if class_mappings is None:
        if not sharepoint_site_url or not sharepoint_file_url:
            raise ValueError(
                "sharepoint_site_url and sharepoint_file_url are required when class_mappings is None."
            )
        legend = get_excel_from_sharepoint(
            site_url=sharepoint_site_url,
            file_server_relative_url=sharepoint_file_url,
            sheet_name=0,
        )
        legend["ewoc_code"] = legend["ewoc_code"].str.replace("-", "").astype(int)
        class_mappings = build_class_mappings(legend)
    return class_mappings[ct_name], class_mappings[lc_name]


def load_legend(url: str = DEFAULT_LEGEND_URL) -> pd.DataFrame:
    """Load and normalize the WorldCereal legend table."""
    legend = pd.read_csv(url, header=0, sep=";")
    legend["ewoc_code"] = legend["ewoc_code"].str.replace("-", "").astype(int)
    return legend.ffill(axis=1)


def load_extraction_paths(extractions_glob: str) -> List[str]:
    """Collect extraction parquet paths from a glob pattern."""
    return glob.glob(extractions_glob)


def _sql_quote(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _ref_country_from_ref_id(ref_id: str) -> str:
    parts = ref_id.split("_")
    if len(parts) < 2:
        return "Unknown"
    return parts[1]


def _is_ambiguous_country_code(code: str) -> bool:
    return code in AMBIGUOUS_REF_COUNTRY_CODES or len(code) != 3


def build_extractions_counts(
    conn: duckdb.DuckDBPyConnection,
    extraction_paths: Sequence[str],
    croptype_mapping: Mapping[str, str],
    landcover_mapping: Mapping[str, str],
    world_bounds_path: Path = DEFAULT_WORLD_BOUNDS_PATH,
    exclude_ewoc_code: int = 1000000000,
) -> pd.DataFrame:
    """Aggregate per-cell sample counts for CROPTYPE and LANDCOVER classes.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        DuckDB connection with spatial extension loaded.
    extraction_paths : Sequence[str]
        Collection of parquet paths containing WorldCereal extractions.
    croptype_mapping : Mapping[str, str]
        EWOC code to CROPTYPE27 mapping.
    landcover_mapping : Mapping[str, str]
        EWOC code to LANDCOVER10 mapping.
    exclude_ewoc_code : int, optional
        EWOC code to exclude, by default 1000000000.

    Returns
    -------
    pandas.DataFrame
        Long-form counts with columns: ref_id, year, h3_l3_cell, ewoc_code,
        ct27_class, lc10_class, country, region, n_samples.
    """
    if not extraction_paths:
        raise ValueError("No extraction files found. Check the input glob.")

    ct27_values = ", ".join(
        f"({_sql_quote(k)}, {_sql_quote(v)})" for k, v in croptype_mapping.items()
    )
    lc10_values = ", ".join(
        f"({_sql_quote(k)}, {_sql_quote(v)})" for k, v in landcover_mapping.items()
    )

    results: List[pd.DataFrame] = []
    for file_path in tqdm(extraction_paths, desc="Aggregating extractions"):
        ref_id = file_path.split("/")[-1].split(".")[0]
        ref_country = _ref_country_from_ref_id(ref_id)
        use_geometry = _is_ambiguous_country_code(ref_country)

        if use_geometry:
            query = f"""
            set enable_progress_bar=false;
            with ct27_map(ewoc_code, ct27_class) as (
                values
                    {ct27_values}
            ),
            lc10_map(ewoc_code, lc10_class) as (
                values
                    {lc10_values}
            ),
            bounds as (
                select iso3, region, geometry as geom
                from read_parquet('{world_bounds_path}')
            ),
            samples as (
                select t.sample_id,
                       t.ewoc_code,
                       lc10_class,
                       ct27_class,
                       t.h3_l3_cell,
                       ST_PointOnSurface(t.geometry) as geom
                from read_parquet('{file_path}') as t
                left join ct27_map m
                    on cast(t.ewoc_code as varchar) = m.ewoc_code
                left join lc10_map l
                    on cast(t.ewoc_code as varchar) = l.ewoc_code
                where t.ewoc_code <> {exclude_ewoc_code}
            ),
            geom_match as (
                select s.sample_id,
                       min(b.iso3) as iso3,
                       min(b.region) as region
                from samples s
                left join bounds b
                    on ST_Intersects(s.geom, b.geom)
                group by s.sample_id
            ),
            enriched as (
                select s.ewoc_code,
                       s.lc10_class,
                       s.ct27_class,
                       s.h3_l3_cell,
                       coalesce(g.iso3, 'Unknown') as country,
                       coalesce(g.region, 'Unknown') as region,
                       s.sample_id
                from samples s
                left join geom_match g
                    on s.sample_id = g.sample_id
            )
            select ewoc_code,
                   lc10_class,
                   ct27_class,
                   h3_l3_cell,
                   country,
                   region,
                   count(distinct sample_id) as n_samples
            from enriched
            group by ewoc_code, ct27_class, lc10_class, h3_l3_cell, country, region
            """
        else:
            query = f"""
            set enable_progress_bar=false;
            with ct27_map(ewoc_code, ct27_class) as (
                values
                    {ct27_values}
            ),
            lc10_map(ewoc_code, lc10_class) as (
                values
                    {lc10_values}
            ),
            bounds as (
                select iso3, region
                from read_parquet('{world_bounds_path}')
            )
            select t.ewoc_code,
                   lc10_class,
                   ct27_class,
                   t.h3_l3_cell,
                   '{ref_country}' as country,
                   coalesce(b.region, 'Unknown') as region,
                   count(distinct sample_id) as n_samples
            from read_parquet('{file_path}') as t
            left join ct27_map m
                on cast(t.ewoc_code as varchar) = m.ewoc_code
            left join lc10_map l
                on cast(t.ewoc_code as varchar) = l.ewoc_code
            left join bounds b
                on b.iso3 = '{ref_country}'
            where t.ewoc_code <> {exclude_ewoc_code}
            group by t.ewoc_code, ct27_class, lc10_class, t.h3_l3_cell, country, region
            """

        tdata = conn.sql(query).df()
        tdata["ref_id"] = ref_id
        tdata["year"] = int(ref_id.split("_")[0])
        results.append(tdata)

    return pd.concat(results, axis=0, ignore_index=True)


def attach_legend_labels(df: pd.DataFrame, legend: pd.DataFrame) -> pd.DataFrame:
    """Attach legend labels to a counts dataframe if ewoc_code is available."""
    if "ewoc_code" not in df.columns:
        return df
    df = df.copy()
    legend_indexed = legend.set_index("ewoc_code")
    df["label_full"] = df["ewoc_code"].map(legend_indexed["label_full"])
    df["sampling_label"] = df["ewoc_code"].map(legend_indexed["sampling_label"])
    return df


def normalize_class_labels(
    df: pd.DataFrame,
    lc_col: str = "lc10_class",
    ct_col: str = "ct27_class",
    ignore_label: str = IGNORE_LABEL,
) -> pd.DataFrame:
    """Ensure class label columns have a consistent ignore label."""
    df = df.copy()
    df[lc_col] = df[lc_col].fillna(ignore_label)
    df[ct_col] = df[ct_col].fillna(ignore_label)
    return df


def _normalize_h3_cell(cell: object) -> Optional[str]:
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    if isinstance(cell, (int, np.integer)):
        try:
            return h3.int_to_str(int(cell))
        except Exception:
            return None
    cell_str = str(cell)
    try:
        if hasattr(h3, "is_valid_cell") and not h3.is_valid_cell(cell_str):
            try:
                return h3.int_to_str(int(cell_str))
            except Exception:
                return None
    except Exception:
        pass
    return cell_str


def build_h3_parent_map(h3_cells: Iterable[object]) -> pd.DataFrame:
    """Build a mapping of H3 L3 -> L2/L1 cells (keeps original L3 values)."""
    rows: List[Tuple[object, str, str]] = []
    for cell in h3_cells:
        cell_norm = _normalize_h3_cell(cell)
        if not cell_norm:
            continue
        try:
            rows.append(
                (
                    cell,
                    h3.cell_to_parent(cell_norm, 2),
                    h3.cell_to_parent(cell_norm, 1),
                )
            )
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["h3_l3_cell", "h3_l2_cell", "h3_l1_cell"])


def _class_set_by_cell(
    df: pd.DataFrame,
    class_col: str,
    mask: pd.Series,
) -> Dict[object, Set[str]]:
    if df.empty:
        return {}
    subset = df.loc[mask, ["h3_l3_cell", class_col]]
    if subset.empty:
        return {}
    grouped = subset.groupby("h3_l3_cell")[class_col].agg(lambda s: set(s))
    return grouped.to_dict()


def _class_set_by_region_cell(
    df: pd.DataFrame,
    class_col: str,
    mask: pd.Series,
) -> Dict[str, Dict[object, Set[str]]]:
    if df.empty:
        return {}
    subset = df.loc[mask, ["region", "h3_l3_cell", class_col]]
    if subset.empty:
        return {}
    grouped = subset.groupby(["region", "h3_l3_cell"])[class_col].agg(lambda s: set(s))
    region_cell: Dict[str, Dict[object, Set[str]]] = {}
    for (region, cell), classes in grouped.items():
        region_cell.setdefault(region, {})[cell] = classes
    return region_cell


def build_cell_stats(
    counts_df: pd.DataFrame,
    train_only_ref_ids: Sequence[str],
    recent_years: Sequence[int] = RECENT_YEARS,
    lc_col: str = "lc10_class",
    ct_col: str = "ct27_class",
    ignore_label: str = IGNORE_LABEL,
) -> Tuple[pd.DataFrame, Dict[str, Dict[object, Set[str]]], Dict[str, Set[str]]]:
    """Build per-cell summaries and class sets for split selection."""
    df = counts_df.copy()
    if "region" not in df.columns:
        df["region"] = "Unknown"
    df["is_train_only"] = df["ref_id"].isin(train_only_ref_ids)
    df["lc_valid"] = df[lc_col] != ignore_label
    df["ct_valid"] = df[ct_col] != ignore_label
    df["both_ignore"] = ~df["lc_valid"] & ~df["ct_valid"]
    df["is_recent_year"] = df["year"].isin(recent_years)

    h3_map = build_h3_parent_map(df["h3_l3_cell"].unique())
    df = df.merge(h3_map, on="h3_l3_cell", how="left")

    non_train = df[~df["is_train_only"]]
    total_labeled = (
        non_train[~non_train["both_ignore"]]
        .groupby("h3_l3_cell")["n_samples"]
        .sum()
    )
    recent_labeled = (
        non_train[~non_train["both_ignore"] & non_train["is_recent_year"]]
        .groupby("h3_l3_cell")["n_samples"]
        .sum()
    )
    n_lc_classes = (
        non_train[non_train["lc_valid"]]
        .groupby("h3_l3_cell")[lc_col]
        .nunique()
    )
    n_ct_classes = (
        non_train[non_train["ct_valid"]]
        .groupby("h3_l3_cell")[ct_col]
        .nunique()
    )

    summary = h3_map.copy()
    summary = summary.merge(
        total_labeled.rename("total_labeled_samples"),
        on="h3_l3_cell",
        how="left",
    )
    summary = summary.merge(
        recent_labeled.rename("recent_labeled_samples"),
        on="h3_l3_cell",
        how="left",
    )
    summary = summary.merge(
        n_lc_classes.rename("n_lc_classes"),
        on="h3_l3_cell",
        how="left",
    )
    summary = summary.merge(
        n_ct_classes.rename("n_ct_classes"),
        on="h3_l3_cell",
        how="left",
    )
    summary[["total_labeled_samples", "recent_labeled_samples"]] = summary[
        ["total_labeled_samples", "recent_labeled_samples"]
    ].fillna(0)
    summary[["n_lc_classes", "n_ct_classes"]] = summary[
        ["n_lc_classes", "n_ct_classes"]
    ].fillna(0)
    summary["total_labeled_samples"] = summary["total_labeled_samples"].astype(int)
    summary["recent_labeled_samples"] = summary["recent_labeled_samples"].astype(int)
    summary["n_lc_classes"] = summary["n_lc_classes"].astype(int)
    summary["n_ct_classes"] = summary["n_ct_classes"].astype(int)

    lc_non_trainonly = _class_set_by_cell(
        df,
        lc_col,
        (~df["is_train_only"]) & df["lc_valid"],
    )
    ct_non_trainonly = _class_set_by_cell(
        df,
        ct_col,
        (~df["is_train_only"]) & df["ct_valid"],
    )
    lc_trainonly = _class_set_by_cell(
        df,
        lc_col,
        df["is_train_only"] & df["lc_valid"],
    )
    ct_trainonly = _class_set_by_cell(
        df,
        ct_col,
        df["is_train_only"] & df["ct_valid"],
    )
    lc_non_trainonly_region = _class_set_by_region_cell(
        df,
        lc_col,
        (~df["is_train_only"]) & df["lc_valid"],
    )
    ct_non_trainonly_region = _class_set_by_region_cell(
        df,
        ct_col,
        (~df["is_train_only"]) & df["ct_valid"],
    )

    cell_lc_train: Dict[object, Set[str]] = {}
    cell_ct_train: Dict[object, Set[str]] = {}
    all_cells = summary["h3_l3_cell"].tolist()
    for cell in all_cells:
        cell_lc_train[cell] = lc_non_trainonly.get(cell, set()) | lc_trainonly.get(
            cell, set()
        )
        cell_ct_train[cell] = ct_non_trainonly.get(cell, set()) | ct_trainonly.get(
            cell, set()
        )

    class_sets = {
        "lc_non_trainonly": lc_non_trainonly,
        "ct_non_trainonly": ct_non_trainonly,
        "lc_trainonly": lc_trainonly,
        "ct_trainonly": ct_trainonly,
        "lc_train": cell_lc_train,
        "ct_train": cell_ct_train,
        "lc_non_trainonly_region_cell": lc_non_trainonly_region,
        "ct_non_trainonly_region_cell": ct_non_trainonly_region,
    }
    class_targets = {
        "lc_all": set(df.loc[df["lc_valid"], lc_col].unique()),
        "ct_all": set(df.loc[df["ct_valid"], ct_col].unique()),
        "lc_non_trainonly": set(non_train.loc[non_train["lc_valid"], lc_col].unique()),
        "ct_non_trainonly": set(non_train.loc[non_train["ct_valid"], ct_col].unique()),
        "lc_non_trainonly_by_region": {
            region: set(classes)
            for region, classes in non_train[non_train["lc_valid"]]
            .groupby("region")[lc_col]
            .unique()
            .items()
            if region != "Unknown"
        },
        "ct_non_trainonly_by_region": {
            region: set(classes)
            for region, classes in non_train[non_train["ct_valid"]]
            .groupby("region")[ct_col]
            .unique()
            .items()
            if region != "Unknown"
        },
    }
    return summary, class_sets, class_targets


def select_initial_cells(cell_summary: pd.DataFrame) -> Tuple[Set[object], Set[object], int]:
    """Select initial val/test cells by ranking within each H3 L1 cell."""
    val_cells: Set[object] = set()
    test_cells: Set[object] = set()
    missing_level1 = 0
    for _, group in cell_summary.groupby("h3_l1_cell"):
        eligible = group[group["total_labeled_samples"] > 0]
        if len(eligible) < 2:
            missing_level1 += 1
            continue
        ranked = eligible.sort_values(
            [
                "n_lc_classes",
                "n_ct_classes",
                "recent_labeled_samples",
                "total_labeled_samples",
                "h3_l3_cell",
            ],
            ascending=[False, False, False, False, True],
        )
        test_cell = ranked.iloc[0]["h3_l3_cell"]
        val_cell = ranked.iloc[1]["h3_l3_cell"]
        test_cells.add(test_cell)
        val_cells.add(val_cell)
    return val_cells, test_cells, missing_level1


def _coverage(split_cells: Set[object], cell_classes: Mapping[object, Set[str]]) -> Set[str]:
    classes: Set[str] = set()
    for cell in split_cells:
        classes |= cell_classes.get(cell, set())
    return classes


def _class_counts(
    split_cells: Set[object],
    cell_classes: Mapping[object, Set[str]],
) -> Counter:
    counts: Counter = Counter()
    for cell in split_cells:
        for cls in cell_classes.get(cell, set()):
            counts[cls] += 1
    return counts


def _can_remove_cell(
    cell: object,
    counts_lc: Counter,
    counts_ct: Counter,
    cell_lc: Set[str],
    cell_ct: Set[str],
    target_lc: Set[str],
    target_ct: Set[str],
) -> bool:
    for cls in cell_lc:
        if cls in target_lc and counts_lc.get(cls, 0) <= 1:
            return False
    for cls in cell_ct:
        if cls in target_ct and counts_ct.get(cls, 0) <= 1:
            return False
    return True


def add_cells_for_split(
    split_cells: Set[object],
    candidate_cells: Set[object],
    cell_lc: Mapping[object, Set[str]],
    cell_ct: Mapping[object, Set[str]],
    target_lc: Set[str],
    target_ct: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """Add cells from candidates to cover missing LC/CT classes."""
    coverage_lc = _coverage(split_cells, cell_lc)
    coverage_ct = _coverage(split_cells, cell_ct)
    missing_lc = set(target_lc) - coverage_lc
    missing_ct = set(target_ct) - coverage_ct

    while missing_lc or missing_ct:
        best_cell = None
        best_gain = 0
        for cell in list(candidate_cells):
            gain = len(cell_lc.get(cell, set()) & missing_lc) + len(
                cell_ct.get(cell, set()) & missing_ct
            )
            if gain > best_gain:
                best_gain = gain
                best_cell = cell
        if best_cell is None or best_gain == 0:
            break
        split_cells.add(best_cell)
        candidate_cells.remove(best_cell)
        coverage_lc |= cell_lc.get(best_cell, set())
        coverage_ct |= cell_ct.get(best_cell, set())
        missing_lc = set(target_lc) - coverage_lc
        missing_ct = set(target_ct) - coverage_ct

    return missing_lc, missing_ct


def add_cells_for_region_coverage(
    split_cells: Set[object],
    candidate_cells: Set[object],
    region_cell_lc: Mapping[str, Mapping[object, Set[str]]],
    region_cell_ct: Mapping[str, Mapping[object, Set[str]]],
    region_target_lc: Mapping[str, Set[str]],
    region_target_ct: Mapping[str, Set[str]],
) -> Dict[str, Tuple[Set[str], Set[str]]]:
    """Add cells to cover missing LC/CT classes within each region."""
    missing_by_region: Dict[str, Tuple[Set[str], Set[str]]] = {}
    all_regions = set(region_target_lc) | set(region_target_ct)

    for region in sorted(all_regions):
        region_cells_lc = region_cell_lc.get(region, {})
        region_cells_ct = region_cell_ct.get(region, {})
        if not region_cells_lc and not region_cells_ct:
            continue

        region_candidates = {
            cell
            for cell in candidate_cells
            if cell in region_cells_lc or cell in region_cells_ct
        }
        if not region_candidates:
            continue

        split_region_cells = {
            cell for cell in split_cells if cell in region_cells_lc or cell in region_cells_ct
        }
        coverage_lc = _coverage(split_region_cells, region_cells_lc)
        coverage_ct = _coverage(split_region_cells, region_cells_ct)
        missing_lc = set(region_target_lc.get(region, set())) - coverage_lc
        missing_ct = set(region_target_ct.get(region, set())) - coverage_ct

        while missing_lc or missing_ct:
            best_cell = None
            best_gain = 0
            for cell in list(region_candidates):
                gain = len(region_cells_lc.get(cell, set()) & missing_lc) + len(
                    region_cells_ct.get(cell, set()) & missing_ct
                )
                if gain > best_gain:
                    best_gain = gain
                    best_cell = cell
            if best_cell is None or best_gain == 0:
                break
            split_cells.add(best_cell)
            candidate_cells.remove(best_cell)
            region_candidates.remove(best_cell)
            coverage_lc |= region_cells_lc.get(best_cell, set())
            coverage_ct |= region_cells_ct.get(best_cell, set())
            missing_lc = set(region_target_lc.get(region, set())) - coverage_lc
            missing_ct = set(region_target_ct.get(region, set())) - coverage_ct

        if missing_lc or missing_ct:
            missing_by_region[region] = (missing_lc, missing_ct)

    return missing_by_region


def ensure_train_coverage(
    train_cells: Set[object],
    val_cells: Set[object],
    test_cells: Set[object],
    cell_lc_train: Mapping[object, Set[str]],
    cell_ct_train: Mapping[object, Set[str]],
    cell_lc_non_trainonly: Mapping[object, Set[str]],
    cell_ct_non_trainonly: Mapping[object, Set[str]],
    target_lc: Set[str],
    target_ct: Set[str],
    val_target_lc: Set[str],
    val_target_ct: Set[str],
    test_target_lc: Set[str],
    test_target_ct: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """Move cells from val/test to train when possible to restore train coverage."""
    missing_lc = set(target_lc) - _coverage(train_cells, cell_lc_train)
    missing_ct = set(target_ct) - _coverage(train_cells, cell_ct_train)
    if not missing_lc and not missing_ct:
        return missing_lc, missing_ct

    for split_name, split_cells, target_lc_split, target_ct_split in (
        ("val", val_cells, val_target_lc, val_target_ct),
        ("test", test_cells, test_target_lc, test_target_ct),
    ):
        counts_lc = _class_counts(split_cells, cell_lc_non_trainonly)
        counts_ct = _class_counts(split_cells, cell_ct_non_trainonly)
        moved = True
        while moved and (missing_lc or missing_ct):
            moved = False
            for cell in list(split_cells):
                provides_lc = cell_lc_train.get(cell, set()) & missing_lc
                provides_ct = cell_ct_train.get(cell, set()) & missing_ct
                if not provides_lc and not provides_ct:
                    continue
                if not _can_remove_cell(
                    cell,
                    counts_lc,
                    counts_ct,
                    cell_lc_non_trainonly.get(cell, set()),
                    cell_ct_non_trainonly.get(cell, set()),
                    target_lc_split,
                    target_ct_split,
                ):
                    continue
                split_cells.remove(cell)
                train_cells.add(cell)
                for cls in cell_lc_non_trainonly.get(cell, set()):
                    counts_lc[cls] -= 1
                for cls in cell_ct_non_trainonly.get(cell, set()):
                    counts_ct[cls] -= 1
                missing_lc -= provides_lc
                missing_ct -= provides_ct
                moved = True
                if not missing_lc and not missing_ct:
                    break
        if not missing_lc and not missing_ct:
            break

    return missing_lc, missing_ct


def ensure_recent_years(
    test_cells: Set[object],
    train_cells: Set[object],
    cell_summary: pd.DataFrame,
    min_recent_samples: int = 1,
) -> int:
    """Ensure test split has at least min_recent_samples recent samples."""
    recent_by_cell = dict(
        zip(cell_summary["h3_l3_cell"], cell_summary["recent_labeled_samples"])
    )
    current_recent = sum(recent_by_cell.get(cell, 0) for cell in test_cells)
    if current_recent >= min_recent_samples:
        return current_recent

    candidates = [
        cell
        for cell in train_cells
        if recent_by_cell.get(cell, 0) > 0
    ]
    candidates = sorted(candidates, key=lambda c: recent_by_cell.get(c, 0), reverse=True)
    for cell in candidates:
        if current_recent >= min_recent_samples:
            break
        test_cells.add(cell)
        train_cells.remove(cell)
        current_recent += recent_by_cell.get(cell, 0)
    return current_recent


def _split_sample_totals(
    cell_summary: pd.DataFrame,
    val_cells: Set[object],
    test_cells: Set[object],
) -> Tuple[Dict[object, int], Dict[str, int]]:
    samples_by_cell = dict(
        zip(cell_summary["h3_l3_cell"], cell_summary["total_labeled_samples"])
    )
    total = int(sum(samples_by_cell.values()))
    val_total = int(sum(samples_by_cell.get(cell, 0) for cell in val_cells))
    test_total = int(sum(samples_by_cell.get(cell, 0) for cell in test_cells))
    train_total = total - val_total - test_total
    return samples_by_cell, {
        "total": total,
        "train": train_total,
        "val": val_total,
        "test": test_total,
    }


def _split_cell_totals(
    cell_summary: pd.DataFrame,
    val_cells: Set[object],
    test_cells: Set[object],
) -> Dict[str, int]:
    all_cells = set(cell_summary["h3_l3_cell"])
    train_cells = all_cells - val_cells - test_cells
    return {
        "total": len(all_cells),
        "train": len(train_cells),
        "val": len(val_cells),
        "test": len(test_cells),
    }


def report_split_distribution(
    cell_summary: pd.DataFrame,
    val_cells: Set[object],
    test_cells: Set[object],
    logger: logging.Logger = LOGGER,
    note: str = "",
) -> None:
    samples_by_cell, sample_totals = _split_sample_totals(
        cell_summary, val_cells, test_cells
    )
    cell_totals = _split_cell_totals(cell_summary, val_cells, test_cells)
    total_samples = sample_totals["total"]
    total_cells = cell_totals["total"]
    prefix = f"{note}: " if note else ""
    if total_samples > 0:
        logger.info(
            "%sSplit labeled-sample distribution (non-train-only): train=%s (%.1f%%) val=%s (%.1f%%) test=%s (%.1f%%) total=%s",
            prefix,
            sample_totals["train"],
            100.0 * sample_totals["train"] / total_samples,
            sample_totals["val"],
            100.0 * sample_totals["val"] / total_samples,
            sample_totals["test"],
            100.0 * sample_totals["test"] / total_samples,
            total_samples,
        )
    if total_cells > 0:
        logger.info(
            "%sSplit cell distribution: train=%s (%.1f%%) val=%s (%.1f%%) test=%s (%.1f%%) total=%s",
            prefix,
            cell_totals["train"],
            100.0 * cell_totals["train"] / total_cells,
            cell_totals["val"],
            100.0 * cell_totals["val"] / total_cells,
            cell_totals["test"],
            100.0 * cell_totals["test"] / total_cells,
            total_cells,
        )


def _region_class_counts(
    split_cells: Set[object],
    region_cell_classes: Mapping[str, Mapping[object, Set[str]]],
) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = {region: Counter() for region in region_cell_classes}
    for region, cell_map in region_cell_classes.items():
        region_counts = counts.setdefault(region, Counter())
        for cell in split_cells:
            for cls in cell_map.get(cell, set()):
                region_counts[cls] += 1
    return counts


def _can_remove_cell_region(
    cell: object,
    region_counts_lc: Mapping[str, Counter],
    region_counts_ct: Mapping[str, Counter],
    region_cell_lc: Mapping[str, Mapping[object, Set[str]]],
    region_cell_ct: Mapping[str, Mapping[object, Set[str]]],
    region_target_lc: Mapping[str, Set[str]],
    region_target_ct: Mapping[str, Set[str]],
) -> bool:
    for region, target_lc in region_target_lc.items():
        if not target_lc:
            continue
        cell_classes = region_cell_lc.get(region, {}).get(cell, set())
        if not cell_classes:
            continue
        counts = region_counts_lc.get(region, Counter())
        for cls in cell_classes:
            if cls in target_lc and counts.get(cls, 0) <= 1:
                return False
    for region, target_ct in region_target_ct.items():
        if not target_ct:
            continue
        cell_classes = region_cell_ct.get(region, {}).get(cell, set())
        if not cell_classes:
            continue
        counts = region_counts_ct.get(region, Counter())
        for cls in cell_classes:
            if cls in target_ct and counts.get(cls, 0) <= 1:
                return False
    return True


def _adjust_class_counts(counts: Counter, classes: Set[str], delta: int) -> None:
    for cls in classes:
        counts[cls] += delta
        if counts[cls] <= 0:
            counts.pop(cls, None)


def _adjust_region_counts(
    counts: Dict[str, Counter],
    region_cell_classes: Mapping[str, Mapping[object, Set[str]]],
    cell: object,
    delta: int,
) -> None:
    for region, cell_map in region_cell_classes.items():
        classes = cell_map.get(cell, set())
        if not classes:
            continue
        region_counts = counts.setdefault(region, Counter())
        for cls in classes:
            region_counts[cls] += delta
            if region_counts[cls] <= 0:
                region_counts.pop(cls, None)


def correct_split_budget(
    cell_summary: pd.DataFrame,
    val_cells: Set[object],
    test_cells: Set[object],
    class_sets: Mapping[str, Mapping],
    class_targets: Mapping[str, Mapping],
    min_recent_samples: int = 1,
    target_ratios: Tuple[float, float, float] = (
        TARGET_TRAIN_RATIO,
        TARGET_VAL_RATIO,
        TARGET_TEST_RATIO,
    ),
    tolerance_ratio: float = BUDGET_TOLERANCE_RATIO,
    max_moves: int = MAX_BUDGET_CORRECTION_MOVES,
    logger: logging.Logger = LOGGER,
) -> int:
    samples_by_cell, sample_totals = _split_sample_totals(
        cell_summary, val_cells, test_cells
    )
    total_samples = sample_totals["total"]
    if total_samples <= 0:
        logger.info("No labeled samples available; skipping budget correction.")
        return 0

    ratio_train, ratio_val, ratio_test = target_ratios
    desired = {
        "train": total_samples * ratio_train,
        "val": total_samples * ratio_val,
        "test": total_samples * ratio_test,
    }
    tolerance = total_samples * tolerance_ratio

    all_cells = set(cell_summary["h3_l3_cell"])
    train_cells = all_cells - val_cells - test_cells

    train_lc_counts = _class_counts(train_cells, class_sets["lc_train"])
    train_ct_counts = _class_counts(train_cells, class_sets["ct_train"])
    val_lc_counts = _class_counts(val_cells, class_sets["lc_non_trainonly"])
    val_ct_counts = _class_counts(val_cells, class_sets["ct_non_trainonly"])
    test_lc_counts = _class_counts(test_cells, class_sets["lc_non_trainonly"])
    test_ct_counts = _class_counts(test_cells, class_sets["ct_non_trainonly"])

    region_cell_lc = class_sets["lc_non_trainonly_region_cell"]
    region_cell_ct = class_sets["ct_non_trainonly_region_cell"]
    region_target_lc = class_targets["lc_non_trainonly_by_region"]
    region_target_ct = class_targets["ct_non_trainonly_by_region"]
    val_region_lc_counts = _region_class_counts(val_cells, region_cell_lc)
    val_region_ct_counts = _region_class_counts(val_cells, region_cell_ct)
    test_region_lc_counts = _region_class_counts(test_cells, region_cell_lc)
    test_region_ct_counts = _region_class_counts(test_cells, region_cell_ct)

    recent_by_cell = dict(
        zip(cell_summary["h3_l3_cell"], cell_summary["recent_labeled_samples"])
    )
    current_recent = sum(recent_by_cell.get(cell, 0) for cell in test_cells)

    split_cells = {"train": train_cells, "val": val_cells, "test": test_cells}
    split_totals = {
        "train": sample_totals["train"],
        "val": sample_totals["val"],
        "test": sample_totals["test"],
    }
    split_counts = {
        "train": (train_lc_counts, train_ct_counts),
        "val": (val_lc_counts, val_ct_counts),
        "test": (test_lc_counts, test_ct_counts),
    }
    split_region_counts = {
        "val": (val_region_lc_counts, val_region_ct_counts),
        "test": (test_region_lc_counts, test_region_ct_counts),
    }

    def within_tolerance(diff: float) -> bool:
        return abs(diff) <= tolerance

    def can_remove_from_split(cell: object, split_name: str) -> bool:
        if split_name == "train":
            return _can_remove_cell(
                cell,
                train_lc_counts,
                train_ct_counts,
                class_sets["lc_train"].get(cell, set()),
                class_sets["ct_train"].get(cell, set()),
                class_targets["lc_all"],
                class_targets["ct_all"],
            )
        if split_name in ("val", "test"):
            counts_lc, counts_ct = split_counts[split_name]
            if not _can_remove_cell(
                cell,
                counts_lc,
                counts_ct,
                class_sets["lc_non_trainonly"].get(cell, set()),
                class_sets["ct_non_trainonly"].get(cell, set()),
                class_targets["lc_non_trainonly"],
                class_targets["ct_non_trainonly"],
            ):
                return False
            region_counts_lc, region_counts_ct = split_region_counts[split_name]
            if not _can_remove_cell_region(
                cell,
                region_counts_lc,
                region_counts_ct,
                region_cell_lc,
                region_cell_ct,
                region_target_lc,
                region_target_ct,
            ):
                return False
            if split_name == "test" and min_recent_samples > 0:
                if current_recent - recent_by_cell.get(cell, 0) < min_recent_samples:
                    return False
            return True
        return False

    def apply_move(cell: object, source: str, target: str) -> None:
        nonlocal current_recent
        delta = samples_by_cell.get(cell, 0)
        split_cells[source].remove(cell)
        split_cells[target].add(cell)
        split_totals[source] -= delta
        split_totals[target] += delta

        if source == "train":
            _adjust_class_counts(
                train_lc_counts, class_sets["lc_train"].get(cell, set()), -1
            )
            _adjust_class_counts(
                train_ct_counts, class_sets["ct_train"].get(cell, set()), -1
            )
        else:
            src_lc_counts, src_ct_counts = split_counts[source]
            _adjust_class_counts(
                src_lc_counts, class_sets["lc_non_trainonly"].get(cell, set()), -1
            )
            _adjust_class_counts(
                src_ct_counts, class_sets["ct_non_trainonly"].get(cell, set()), -1
            )
            src_region_lc, src_region_ct = split_region_counts[source]
            _adjust_region_counts(src_region_lc, region_cell_lc, cell, -1)
            _adjust_region_counts(src_region_ct, region_cell_ct, cell, -1)
            if source == "test":
                current_recent -= recent_by_cell.get(cell, 0)

        if target == "train":
            _adjust_class_counts(
                train_lc_counts, class_sets["lc_train"].get(cell, set()), 1
            )
            _adjust_class_counts(
                train_ct_counts, class_sets["ct_train"].get(cell, set()), 1
            )
        else:
            tgt_lc_counts, tgt_ct_counts = split_counts[target]
            _adjust_class_counts(
                tgt_lc_counts, class_sets["lc_non_trainonly"].get(cell, set()), 1
            )
            _adjust_class_counts(
                tgt_ct_counts, class_sets["ct_non_trainonly"].get(cell, set()), 1
            )
            tgt_region_lc, tgt_region_ct = split_region_counts[target]
            _adjust_region_counts(tgt_region_lc, region_cell_lc, cell, 1)
            _adjust_region_counts(tgt_region_ct, region_cell_ct, cell, 1)
            if target == "test":
                current_recent += recent_by_cell.get(cell, 0)

    moves = 0
    while moves < max_moves:
        diffs = {
            split: split_totals[split] - desired[split] for split in ("train", "val", "test")
        }
        overfull = {split: diff for split, diff in diffs.items() if diff > tolerance}
        underfull = {split: diff for split, diff in diffs.items() if diff < -tolerance}
        if not overfull or not underfull:
            break

        candidates_found = False
        for source in sorted(overfull, key=overfull.get, reverse=True):
            for target in sorted(underfull, key=underfull.get):
                source_total = split_totals[source]
                target_total = split_totals[target]
                desired_source = desired[source]
                desired_target = desired[target]
                current_score = abs(source_total - desired_source) + abs(
                    target_total - desired_target
                )
                best_cell = None
                best_score = current_score
                for cell in split_cells[source]:
                    delta = samples_by_cell.get(cell, 0)
                    if delta <= 0:
                        continue
                    if not can_remove_from_split(cell, source):
                        continue
                    new_source_total = source_total - delta
                    new_target_total = target_total + delta
                    new_score = abs(new_source_total - desired_source) + abs(
                        new_target_total - desired_target
                    )
                    if new_score < best_score:
                        best_score = new_score
                        best_cell = cell
                if best_cell is not None:
                    apply_move(best_cell, source, target)
                    moves += 1
                    candidates_found = True
                    break
            if candidates_found:
                break
        if not candidates_found:
            break

    final_diffs = {
        split: split_totals[split] - desired[split] for split in ("train", "val", "test")
    }
    if moves:
        logger.info("Budget correction moved %s cells.", moves)
    if all(within_tolerance(diff) for diff in final_diffs.values()):
        logger.info(
            "Budget correction within tolerance (%.1f%% of labeled samples).",
            tolerance_ratio * 100.0,
        )
    else:
        logger.warning(
            "Budget correction incomplete. Final diffs: train=%s val=%s test=%s (tolerance %.1f%%).",
            int(round(final_diffs["train"])),
            int(round(final_diffs["val"])),
            int(round(final_diffs["test"])),
            tolerance_ratio * 100.0,
        )
    return moves


def write_sample_splits_parquet(
    conn: duckdb.DuckDBPyConnection,
    extraction_paths: Sequence[str],
    croptype_mapping: Mapping[str, str],
    landcover_mapping: Mapping[str, str],
    val_cells: Set[object],
    test_cells: Set[object],
    output_path: Path,
    train_only_ref_ids: Sequence[str],
    exclude_ewoc_code: int = 1000000000,
    ignore_label: str = IGNORE_LABEL,
) -> None:
    """Write sample_id -> split parquet with train-only leakage protection."""
    if not extraction_paths:
        raise ValueError("No extraction files found. Check the input glob.")

    cell_splits_df = pd.DataFrame(
        [
            *[(cell, "val") for cell in sorted(val_cells)],
            *[(cell, "test") for cell in sorted(test_cells)],
        ],
        columns=["h3_l3_cell", "split"],
    )
    conn.register("cell_splits", cell_splits_df)

    train_only_df = pd.DataFrame({"ref_id": list(train_only_ref_ids)})
    conn.register("train_only", train_only_df)

    ct27_values = ", ".join(
        f"({_sql_quote(k)}, {_sql_quote(v)})" for k, v in croptype_mapping.items()
    )
    lc10_values = ", ".join(
        f"({_sql_quote(k)}, {_sql_quote(v)})" for k, v in landcover_mapping.items()
    )
    paths_sql = ", ".join(_sql_quote(p) for p in extraction_paths)
    conn.sql("set enable_progress_bar=false")
    query = f"""
    with ct27_map(ewoc_code, ct27_class) as (
        values
            {ct27_values}
    ),
    lc10_map(ewoc_code, lc10_class) as (
        values
            {lc10_values}
    ),
    samples as (
        select distinct t.sample_id,
               t.h3_l3_cell,
               lc10_class,
               ct27_class,
               regexp_extract(filename, '([^/]+)\\\\.parquet$', 1) as ref_id
        from read_parquet([{paths_sql}], filename=true) as t
        left join ct27_map m
            on cast(t.ewoc_code as varchar) = m.ewoc_code
        left join lc10_map l
            on cast(t.ewoc_code as varchar) = l.ewoc_code
        where t.ewoc_code <> {exclude_ewoc_code}
    )
    select sample_id,
           case
               when lc10_class = '{ignore_label}' and ct27_class = '{ignore_label}'
                   then '{ignore_label}'
               when cs.split is null then 'train'
               when tr.ref_id is not null then '{ignore_label}'
               else cs.split
           end as split
    from samples
    left join cell_splits cs on samples.h3_l3_cell = cs.h3_l3_cell
    left join train_only tr on samples.ref_id = tr.ref_id
    """
    conn.sql(f"copy ({query}) to '{output_path}' (format 'parquet')")

def main(
    extractions_glob: str = DEFAULT_EXTRACTS_GLOB,
    class_mappings: Optional[dict] = None,
    lc_name: str = "LANDCOVER10",
    ct_name: str = "CROPTYPE27",
    legend_url: str = DEFAULT_LEGEND_URL,
    include_legend: bool = True,
    world_bounds_path: Path = DEFAULT_WORLD_BOUNDS_PATH,
    output_path: Path = DEFAULT_UNIFIED_SPLITS_PATH,
    train_only_ref_ids: Sequence[str] = TRAIN_ONLY_REF_IDS,
    sharepoint_site_url: Optional[str] = None,
    sharepoint_file_url: Optional[str] = None,
) -> None:
    """Build unified H3-based splits for CROPTYPE27 and LANDCOVER10 datasets."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    croptype_mapping, landcover_mapping = load_mappings(
        class_mappings,
        lc_name,
        ct_name,
        sharepoint_site_url=sharepoint_site_url,
        sharepoint_file_url=sharepoint_file_url,
    )
    extraction_paths = load_extraction_paths(extractions_glob)

    conn = init_duckdb_connection()
    counts_df = build_extractions_counts(
        conn,
        extraction_paths,
        croptype_mapping,
        landcover_mapping,
        world_bounds_path=world_bounds_path,
    )
    if include_legend:
        legend = load_legend(legend_url)
        counts_df = attach_legend_labels(counts_df, legend)

    counts_df = normalize_class_labels(counts_df)

    cell_summary, class_sets, class_targets = build_cell_stats(
        counts_df,
        train_only_ref_ids=train_only_ref_ids,
        recent_years=RECENT_YEARS,
    )

    val_cells, test_cells, missing_level1 = select_initial_cells(cell_summary)
    all_cells = set(cell_summary["h3_l3_cell"])
    train_cells = all_cells - val_cells - test_cells

    lc_only_train = class_targets["lc_all"] - class_targets["lc_non_trainonly"]
    ct_only_train = class_targets["ct_all"] - class_targets["ct_non_trainonly"]
    if lc_only_train:
        LOGGER.warning(
            "LC classes only in train-only datasets (out of %s); cannot place in val/test: %s",
            len(class_targets["lc_all"]),
            sorted(lc_only_train),
        )
    if ct_only_train:
        LOGGER.warning(
            "CT classes only in train-only datasets (out of %s); cannot place in val/test: %s",
            len(class_targets["ct_all"]),
            sorted(ct_only_train),
        )

    region_missing_test = add_cells_for_region_coverage(
        test_cells,
        train_cells,
        class_sets["lc_non_trainonly_region_cell"],
        class_sets["ct_non_trainonly_region_cell"],
        class_targets["lc_non_trainonly_by_region"],
        class_targets["ct_non_trainonly_by_region"],
    )
    train_cells = all_cells - val_cells - test_cells
    region_missing_val = add_cells_for_region_coverage(
        val_cells,
        train_cells,
        class_sets["lc_non_trainonly_region_cell"],
        class_sets["ct_non_trainonly_region_cell"],
        class_targets["lc_non_trainonly_by_region"],
        class_targets["ct_non_trainonly_by_region"],
    )
    train_cells = all_cells - val_cells - test_cells

    recent_total = ensure_recent_years(
        test_cells,
        train_cells,
        cell_summary,
        min_recent_samples=1,
    )
    train_cells = all_cells - val_cells - test_cells

    test_missing_lc, test_missing_ct = add_cells_for_split(
        test_cells,
        train_cells,
        class_sets["lc_non_trainonly"],
        class_sets["ct_non_trainonly"],
        class_targets["lc_non_trainonly"],
        class_targets["ct_non_trainonly"],
    )
    train_cells = all_cells - val_cells - test_cells
    val_missing_lc, val_missing_ct = add_cells_for_split(
        val_cells,
        train_cells,
        class_sets["lc_non_trainonly"],
        class_sets["ct_non_trainonly"],
        class_targets["lc_non_trainonly"],
        class_targets["ct_non_trainonly"],
    )

    train_cells = all_cells - val_cells - test_cells
    recent_total = ensure_recent_years(
        test_cells,
        train_cells,
        cell_summary,
        min_recent_samples=1,
    )
    train_cells = all_cells - val_cells - test_cells

    train_missing_lc, train_missing_ct = ensure_train_coverage(
        train_cells,
        val_cells,
        test_cells,
        class_sets["lc_train"],
        class_sets["ct_train"],
        class_sets["lc_non_trainonly"],
        class_sets["ct_non_trainonly"],
        class_targets["lc_all"],
        class_targets["ct_all"],
        class_targets["lc_non_trainonly"],
        class_targets["ct_non_trainonly"],
        class_targets["lc_non_trainonly"],
        class_targets["ct_non_trainonly"],
    )

    report_split_distribution(
        cell_summary,
        val_cells,
        test_cells,
        note="Before budget correction",
    )
    correct_split_budget(
        cell_summary,
        val_cells,
        test_cells,
        class_sets,
        class_targets,
        min_recent_samples=1,
    )
    report_split_distribution(
        cell_summary,
        val_cells,
        test_cells,
        note="After budget correction",
    )

    train_cells = all_cells - val_cells - test_cells
    test_missing_lc = class_targets["lc_non_trainonly"] - _coverage(
        test_cells, class_sets["lc_non_trainonly"]
    )
    test_missing_ct = class_targets["ct_non_trainonly"] - _coverage(
        test_cells, class_sets["ct_non_trainonly"]
    )
    val_missing_lc = class_targets["lc_non_trainonly"] - _coverage(
        val_cells, class_sets["lc_non_trainonly"]
    )
    val_missing_ct = class_targets["ct_non_trainonly"] - _coverage(
        val_cells, class_sets["ct_non_trainonly"]
    )
    train_missing_lc = class_targets["lc_all"] - _coverage(
        train_cells, class_sets["lc_train"]
    )
    train_missing_ct = class_targets["ct_all"] - _coverage(
        train_cells, class_sets["ct_train"]
    )

    total_l1 = cell_summary["h3_l1_cell"].nunique()
    total_lc_nt = len(class_targets["lc_non_trainonly"])
    total_ct_nt = len(class_targets["ct_non_trainonly"])
    total_lc_all = len(class_targets["lc_all"])
    total_ct_all = len(class_targets["ct_all"])
    test_total = int(
        cell_summary.loc[
            cell_summary["h3_l3_cell"].isin(test_cells), "total_labeled_samples"
        ].sum()
    )
    recent_total = int(
        cell_summary.loc[
            cell_summary["h3_l3_cell"].isin(test_cells), "recent_labeled_samples"
        ].sum()
    )

    if missing_level1:
        LOGGER.info(
            "%s H3 L1 cells lack enough data for val/test (out of %s).",
            missing_level1,
            total_l1,
        )
    if region_missing_test:
        LOGGER.warning(
            "Could not cover all region classes in test for %s regions. Examples: %s",
            len(region_missing_test),
            list(region_missing_test.items())[:5],
        )
    if region_missing_val:
        LOGGER.warning(
            "Could not cover all region classes in val for %s regions. Examples: %s",
            len(region_missing_val),
            list(region_missing_val.items())[:5],
        )
    if test_missing_lc or test_missing_ct:
        LOGGER.warning(
            "Could not cover all classes in test. Missing LC=%s (out of %s) CT=%s (out of %s).",
            sorted(test_missing_lc),
            total_lc_nt,
            sorted(test_missing_ct),
            total_ct_nt,
        )
    if val_missing_lc or val_missing_ct:
        LOGGER.warning(
            "Could not cover all classes in val. Missing LC=%s (out of %s) CT=%s (out of %s).",
            sorted(val_missing_lc),
            total_lc_nt,
            sorted(val_missing_ct),
            total_ct_nt,
        )
    if train_missing_lc or train_missing_ct:
        LOGGER.warning(
            "Could not cover all classes in train. Missing LC=%s (out of %s) CT=%s (out of %s).",
            sorted(train_missing_lc),
            total_lc_all,
            sorted(train_missing_ct),
            total_ct_all,
        )
    LOGGER.info(
        "Recent-year samples in test: %s (out of %s labeled samples).",
        recent_total,
        test_total,
    )

    write_sample_splits_parquet(
        conn,
        extraction_paths,
        croptype_mapping,
        landcover_mapping,
        val_cells,
        test_cells,
        output_path,
        train_only_ref_ids,
    )


if __name__ == "__main__":
    main()
