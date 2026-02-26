"""Anomaly detection pipeline operating purely on cached Presto embeddings.

This version assumes a DuckDB cache already exists with columns:
``sample_id, model_hash, ref_id, ewoc_code, h3_l3_cell, embedding_0..embedding_127``.
Embeddings are never recomputed; the pipeline always loads them from the cache.
Optional label domain switching between ``ewoc_code`` and mapped ``finetune_class``.

Grouping:
- Slices are defined by: group_cols (optional) + [h3 cell] + [label col]
- group_cols defaults to [] (i.e., global per (h3, label) slices)

Adaptive H3 resolution:
- When ``h3_level`` is a list (e.g. ``[1, 2, 3]``), levels are tried
  coarsest → finest.  A slice is resolved at the coarsest level where its
  size is within [min_slice_size, max_slice_size].  Dense regions (Europe)
  with oversized coarse-level slices are pushed to finer cells; sparse
  regions (Africa) are resolved at a coarser level.

Module layout
~~~~~~~~~~~~~
- **anomaly_utils.py** — pure computation helpers (scoring, metrics, mapping,
  flagging, adaptive H3 assignment).  Stateless building blocks.
- **anomaly.py** *(this file)* — pipeline orchestration: data loading,
  incremental mode, class mapping, scoring dispatch, anomaly categorization,
  and output writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from worldcereal.utils.refdata import get_class_mappings, map_classes

# All computation helpers live in anomaly_utils — import them here so that
# any downstream code doing ``from worldcereal.train.anomaly import <func>``
# continues to work.
from worldcereal.train.anomaly_utils import (
    MIN_SCORING_SLICE_SIZE,
    _SCORE_COLS,
    _as_label_levels,
    _require_label_columns,
    _load_mapping_df,
    _add_hierarchical_ref_outlier_class,
    assign_adaptive_h3_level,
    merge_small_slices,
    compute_slice_centroids,
    compute_scores_for_slice,
    _score_group_simple,
    score_slices_hierarchical,
    add_alt_class_centroid_metrics,
    add_knn_label_purity_for_flagged,
    add_confidence_from_score,
    add_flagged_robust_confidence,
    apply_confidence_fusion,
    flag_anomalies,
)


# ===================================================================
# Pipeline
# ===================================================================


def _load_embeddings(
    con: duckdb.DuckDBPyConnection,
    group_cols: list[str],
    restrict_model_hash: Optional[str],
) -> Tuple[pd.DataFrame, list[str]]:
    """Load cached embeddings from DuckDB.

    Returns ``(df, embed_cols)`` where *embed_cols* are the raw
    ``embedding_0 … embedding_N`` column names.
    """
    cols_df = con.execute("PRAGMA table_info('embeddings_cache')").fetchdf()
    embed_cols = [c for c in cols_df.name.tolist() if c.startswith("embedding_")]

    base_cols = [
        "sample_id",
        "ewoc_code",
        "model_hash",
        "ref_id",
        "h3_l3_cell",
        "lat",
        "lon",
        # "country",
    ]
    select_cols = list(dict.fromkeys([*base_cols, *group_cols]))  # preserve order, unique

    query = f"SELECT {', '.join(select_cols + embed_cols)} FROM embeddings_cache"
    if restrict_model_hash:
        query += f" WHERE model_hash='{restrict_model_hash}'"

    df = con.execute(query).fetchdf()
    return df, embed_cols


def _handle_incremental_mode(
    df: pd.DataFrame,
    output_samples_path: Optional[str],
    con: duckdb.DuckDBPyConnection,
) -> Tuple[pd.DataFrame, Optional[gpd.GeoDataFrame], set]:
    """Filter out already-processed sample_ids when resuming.

    Returns ``(df_filtered, existing_df_full_or_None, existing_ids_set)``.
    """
    if not output_samples_path:
        print(
            "[anomaly] WARNING: skip_existing_samples=True but "
            "output_samples_path not set. Processing all samples."
        )
        return df, None, set()

    out_path = Path(output_samples_path)
    if not out_path.exists():
        print(
            f"[anomaly] skip_existing_samples=True but output file doesn't exist yet: "
            f"{output_samples_path}"
        )
        print(f"[anomaly] Processing all {len(df):,} samples from scratch...")
        return df, None, set()

    print(f"[anomaly] Loading existing results from {output_samples_path}...")
    existing_df_full = gpd.read_parquet(output_samples_path)
    if "sample_id" not in existing_df_full.columns:
        con.close()
        raise ValueError(
            f"Existing output_samples_path has no 'sample_id' column: {output_samples_path}"
        )
    existing_ids = set(existing_df_full["sample_id"].astype(str).unique())

    before_count = len(df)
    df_sample_ids = df["sample_id"].astype(str)
    df = df[~df_sample_ids.isin(existing_ids)].copy()
    after_count = len(df)

    print(f"[anomaly] Found {len(existing_ids):,} existing samples")
    print(f"[anomaly] Filtering: {before_count:,} -> {after_count:,} rows to process")

    return df, existing_df_full, existing_ids


def _apply_class_mapping(
    df: pd.DataFrame,
    *,
    map_to_finetune: bool,
    mapping_file: Optional[Union[str, dict]],
    label_cols: list[str],
    class_mappings_name: str,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Map ewoc_code → label column(s) using the chosen strategy."""

    if map_to_finetune:
        print(f"[anomaly] Mapping classes using '{class_mappings_name}'...")
        return map_classes(df, class_mappings_name)

    if mapping_file is None:
        return df

    print(f"[anomaly] Mapping classes using mapping_file: {mapping_file}")

    map_df = _load_mapping_df(
        mapping_file,
        label_cols=label_cols,
        class_mappings_name=class_mappings_name,
    )

    if "ewoc_code" not in map_df.columns:
        con.close()
        raise ValueError("mapping_file must contain an 'ewoc_code' column")

    missing_map_cols = [c for c in label_cols if c not in map_df.columns]
    if missing_map_cols:
        con.close()
        raise ValueError(f"mapping_file missing required label column(s): {missing_map_cols}")

    # Normalize ewoc_code for joining (remove dashes)
    keep_cols = ["ewoc_code", *label_cols]
    map_df = map_df[keep_cols].copy()
    map_df["ewoc_code_clean"] = (
        map_df["ewoc_code"].astype(str).str.replace("-", "", regex=False)
    )

    df["ewoc_code_clean"] = df["ewoc_code"].astype(str).str.replace("-", "", regex=False)

    # Drop pre-existing label columns to avoid collisions
    df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")

    df = df.merge(
        map_df[["ewoc_code_clean", *label_cols]],
        on="ewoc_code_clean",
        how="left",
    )
    df = df.drop(columns=["ewoc_code_clean"], errors="ignore")
    return df


def _assign_anomaly_categories(
    flagged_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign ``S_anomaly`` and ``combined_anomaly`` escalation categories.

    Operates in-place on *flagged_df* and returns it.
    """
    # flagged_df["S_rank_min"] = np.minimum(flagged_df["cos_rank"], flagged_df["knn_rank"])
    is_flagged = flagged_df["flagged"].fillna(False).to_numpy(dtype=bool)

    # --- S_anomaly: based on S and rank_percentile ---
    S_anomaly = "S_anomaly"
    flagged_df[S_anomaly] = "normal"
    flagged_df.loc[is_flagged, S_anomaly] = "flagged"
    flagged_df.loc[
        (flagged_df["rank_percentile"] >= 0.98)
        & (flagged_df["S"] >= 0.95)
        & is_flagged,
        S_anomaly,
    ] = "suspect"
    flagged_df.loc[
        (flagged_df["rank_percentile"] >= 0.99)
        & (flagged_df["S"] >= 0.99)
        & is_flagged,
        S_anomaly,
    ] = "candidate"

    # --- combined_anomaly: consensus of S_rank, S_rank_min, S_z ---
    combined_anomaly = "combined_anomaly"
    flagged_df[combined_anomaly] = "normal"
    flagged_df.loc[is_flagged, combined_anomaly] = "flagged"

    # Consensus-based escalation using multiple score variants (more robust than S alone)
    # All three are in [0,1] where higher => more anomalous
    suspect_thr = 0.98
    candidate_thr = 0.99

    suspect_k_of_m = 2   # require 2-of-3 signals for "suspect"
    candidate_k_of_m = 3  # require 3-of-3 signals for "candidate" (strict)

    score_votes_suspect = (
        (flagged_df["S_rank"] >= suspect_thr).astype(int)
        + (flagged_df["S_rank_min"] >= suspect_thr).astype(int)
        + (flagged_df["S_z"] >= suspect_thr).astype(int)
    )
    score_votes_candidate = (
        (flagged_df["S_rank"] >= candidate_thr).astype(int)
        + (flagged_df["S_rank_min"] >= candidate_thr).astype(int)
        + (flagged_df["S_z"] >= candidate_thr).astype(int)
    )

    # Escalate categories only for already-flagged samples
    flagged_df.loc[
        is_flagged & (score_votes_suspect >= suspect_k_of_m),
        combined_anomaly,
    ] = "suspect"

    flagged_df.loc[
        is_flagged & (score_votes_candidate >= candidate_k_of_m),
        combined_anomaly,
    ] = "candidate"

    # Optional: downgrade confidence for undersized slices (if column exists)
    if "undersized_slice" in flagged_df.columns:
        low_conf = flagged_df["undersized_slice"] == True  # noqa: E712
        # candidate -> suspect; suspect -> flagged (only when undersized)
        flagged_df.loc[
            low_conf & (flagged_df[S_anomaly] == "candidate"), S_anomaly
        ] = "suspect"
        flagged_df.loc[
            low_conf & (flagged_df[S_anomaly] == "suspect"), S_anomaly
        ] = "flagged"
        flagged_df.loc[
            low_conf & (flagged_df[combined_anomaly] == "candidate"), combined_anomaly
        ] = "suspect"
        flagged_df.loc[
            low_conf & (flagged_df[combined_anomaly] == "suspect"), combined_anomaly
        ] = "flagged"

    return flagged_df


def _merge_with_existing(
    flagged_gdf: gpd.GeoDataFrame,
    existing_df_full: Optional[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Append newly-scored rows to previously-saved results (incremental mode)."""
    if existing_df_full is None:
        return flagged_gdf

    print(
        f"[anomaly] Merging {len(flagged_gdf):,} new results with "
        f"{len(existing_df_full):,} existing results..."
    )

    # Align schemas (union of columns) to avoid missing-column issues
    all_cols = list(
        dict.fromkeys(
            [*existing_df_full.columns.tolist(), *flagged_gdf.columns.tolist()]
        )
    )
    existing_aligned = existing_df_full.reindex(columns=all_cols)
    new_aligned = flagged_gdf.reindex(columns=all_cols)

    combined = pd.concat([existing_aligned, new_aligned], axis=0, ignore_index=True)

    # Safety: if any overlaps happen, keep the last occurrence (new wins)
    if "sample_id" in combined.columns:
        combined["sample_id"] = combined["sample_id"].astype(str)
        combined = combined.drop_duplicates(
            subset=["sample_id"], keep="last"
        ).reset_index(drop=True)

    print(f"[anomaly] Total combined: {len(combined):,} samples")
    return combined


def _write_outputs(
    flagged_gdf: gpd.GeoDataFrame,
    summary_df: Optional[pd.DataFrame],
    slice_keys: list[str],
    output_samples_path: Optional[str],
    output_summary_path: Optional[str],
) -> None:
    """Persist results to disk (parquet + Excel)."""
    S_anomaly = "S_anomaly"
    combined_anomaly = "combined_anomaly"

    if output_samples_path:
        print(f"[anomaly] Writing flagged samples -> {output_samples_path}")
        # flagged_gdf = flagged_gdf.drop(
        #     columns=["embedding", "base_embedding"], errors="ignore"
        # )
        flagged_gdf = flagged_gdf.drop(columns=["base_embedding"], errors="ignore")
        flagged_gdf.to_parquet(output_samples_path, index=False)

    if output_summary_path:
        print(f"[anomaly] Writing summary -> {output_summary_path}")
        summary_df.to_parquet(output_summary_path, index=False)
        summary_df.to_excel(
            Path(output_summary_path).with_suffix(".xlsx"),
            index=False,
        )

        # Cross-tabulation: long form
        cross_long = (
            flagged_gdf.groupby([*slice_keys, S_anomaly, combined_anomaly]), dropna=False
            .size()
            .reset_index(name="n")
        )
        cross_long.to_parquet(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_long.parquet"
            ),
            index=False,
        )
        cross_long.to_excel(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_long.xlsx"
            ),
            index=False,
        )

        # Cross-tabulation: wide matrix with flattened column names
        cross_wide = cross_long.pivot_table(
            index=slice_keys,
            columns=[S_anomaly, combined_anomaly],
            values="n",
            fill_value=0,
            aggfunc="sum",
        )
        # Flatten MultiIndex columns -> e.g. "S=candidate__C=suspect"
        cross_wide.columns = [
            f"S={s}__C={c}" for (s, c) in cross_wide.columns.to_list()
        ]
        cross_wide = cross_wide.reset_index()

        cross_wide.to_parquet(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_wide.parquet"
            ),
            index=False,
        )
        cross_wide.to_excel(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_wide.xlsx"
            ),
            index=True,
        )


# ===================================================================
# Main entry point
# ===================================================================


def run_pipeline(
    embeddings_db_path: str,
    restrict_model_hash: Optional[str] = None,
    label_domain: Union[str, Sequence[str]] = "ewoc_code",
    map_to_finetune: bool = False,
    class_mappings_name: str = "LANDCOVER10",
    mapping_file: Optional[Union[str, dict]] = None,
    h3_level: Union[int, Sequence[int]] = 3,
    group_cols: Optional[Sequence[str]] = None,
    min_slice_size: int = 100,
    max_slice_size: Optional[int] = None,
    merge_small_slice: bool = True,
    max_merge_iterations=10,
    threshold_mode: str = "percentile",
    percentile_q: float = 0.96,
    mad_k: float = 3.0,
    abs_threshold: Optional[float] = None,
    fdr_alpha: float = 0.05,
    min_flagged_per_slice: Optional[int] = None,
    max_flagged_fraction: Optional[float] = None,
    max_full_pairwise_n: Optional[int] = 20000,
    norm_percentiles: Tuple[float, float] = (5.0, 95.0),
    output_samples_path: Optional[str] = None,
    output_summary_path: Optional[str] = None,
    skip_existing_samples: bool = False,
    skip_classes: Optional[Sequence[str]] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run anomaly detection using only cached embeddings.

    Grouping
    --------
    slice = group_cols + [h3 cell at chosen level] + [label_col]

    Parameters
    ----------
    h3_level
        H3 resolution(s) for spatial grouping.

        - **Single int** (e.g. ``3``): use a fixed H3 level for all points
          (original behaviour).
        - **List of ints** (e.g. ``[1, 2, 3]``): *adaptive* mode.  Levels
          are tried **coarsest → finest** (ascending by H3 number).
          A slice is resolved at the coarsest level where its size is both
          ≥ *min_slice_size* and ≤ *max_slice_size*.  Slices that are too
          large at a coarse level are pushed to finer levels where the
          geographic cell is smaller.  Slices that are too small are also
          pushed finer; any still-unresolved points after the finest level
          are assigned there unconditionally and handled later by
          ``merge_small_slices``.
    max_slice_size
        (Adaptive mode only.)  Upper cap on slice size per level.  If a
        slice at the current (coarse) H3 level exceeds this, those points
        are pushed to the next finer level.  At the finest level the cap is
        not enforced — all remaining points are resolved unconditionally.
        Ignored when *h3_level* is a single int.
    norm_percentiles
        Percentiles used for per-slice min-max normalization of
        cosine_distance and knn_distance.  Default ``(5, 95)``
        preserves existing behavior.
    skip_existing_samples
        If *True* and *output_samples_path* exists, loads existing results,
        skips already-processed sample_id rows, computes only missing ones,
        then appends old + new and writes back.  This does **not** recompute
        outlier scores for existing sample_ids.
    skip_classes
        Optional list of label values (in the *label_domain* column) that
        are excluded from all scoring, flagging, and confidence steps.
        Their rows are held aside and re-joined to the output at the end
        with all score / outlier columns set to NaN.  Pass e.g.
        ``skip_classes=["built-up", "ignore"]``.
    """
    group_cols = list(group_cols or [])
    label_cols = _as_label_levels(label_domain)
    label_col = label_cols[0]  # keep existing logic anchored to level-0

    # Normalize h3_level into a list; determine if adaptive mode is active
    if isinstance(h3_level, (list, tuple)):
        h3_levels = [int(x) for x in h3_level]
        adaptive_h3 = len(h3_levels) > 1
    else:
        h3_levels = [int(h3_level)]
        adaptive_h3 = False

    # Only enforce when mapping_file is not provided
    if mapping_file is None:
        if isinstance(label_domain, (list, tuple)):
            raise ValueError(
                "Hierarchical label_domain requires mapping_file "
                "(labels provided by Excel or JSON, or an in-memory CLASS_MAPPINGS dict)."
            )
        if label_domain not in {"ewoc_code", "finetune_class", "balancing_class"}:
            raise ValueError("label_domain must be 'ewoc_code' or 'finetune_class'")

    # ------------------------------------------------------------------
    # 1. Load embeddings from DuckDB
    # ------------------------------------------------------------------
    print("[anomaly] Connecting DuckDB and loading cached embeddings...")
    con = duckdb.connect(embeddings_db_path)
    df, embed_cols = _load_embeddings(con, group_cols, restrict_model_hash)

    print(f"[anomaly] Loaded {len(df):,} rows from embeddings_cache")
    if df.empty:
        con.close()
        raise ValueError(
            "No rows loaded from embeddings_cache. Check model_hash or DB path."
        )

    # ------------------------------------------------------------------
    # 2. Incremental mode — skip already-processed sample_ids
    # ------------------------------------------------------------------
    existing_df_full: Optional[gpd.GeoDataFrame] = None

    if skip_existing_samples:
        df, existing_df_full, existing_ids = _handle_incremental_mode(
            df, output_samples_path, con
        )
        if df.empty:
            print("[anomaly] All samples already processed. Returning existing results...")
            con.close()
            return existing_df_full, None

    # ------------------------------------------------------------------
    # 3. Validation & column setup
    # ------------------------------------------------------------------
    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        con.close()
        raise ValueError(
            f"Requested group_cols not found in loaded data: {missing_group_cols}"
        )

    # For adaptive mode, we use the finest level as the reference for
    # debug filtering; the actual adaptive assignment happens after
    # class mapping + embedding preparation (section 5b).
    # For fixed mode, we use the single level as before.
    _finest_h3_level = max(h3_levels)  # finest = highest resolution number
    h3_level_name = "effective_h3_cell" if adaptive_h3 else f"h3_l{h3_levels[0]}_cell"

    if not adaptive_h3:
        _fixed_level = h3_levels[0]
        if _fixed_level != 3:
            df[h3_level_name] = df["h3_l3_cell"].apply(
                lambda h: h3.cell_to_parent(h, _fixed_level)
            )
        else:
            df[h3_level_name] = df["h3_l3_cell"]

    if df["ewoc_code"].dtype != np.int64:
        df["ewoc_code"] = pd.to_numeric(df["ewoc_code"], errors="coerce").astype("Int64")

    if debug:
        print(
            "[DEBUG] Running in debug mode: restricting to small sample of data, "
            "only loading 10 H3 cells..."
        )
        # Use finest level for debug cell sampling
        if adaptive_h3:
            _debug_col = f"_h3_l{_finest_h3_level}_dbg"
            df[_debug_col] = df["h3_l3_cell"].apply(
                lambda h: h3.cell_to_parent(h, _finest_h3_level)
            ) if _finest_h3_level != 3 else df["h3_l3_cell"]
            sample_cells = df[_debug_col].unique()[:10].tolist()
            df = df[df[_debug_col].isin(sample_cells)]
            df = df.drop(columns=[_debug_col], errors="ignore")
        else:
            sample_cells = df[h3_level_name].unique()[:10].tolist()
            df = df[df[h3_level_name].isin(sample_cells)]

    # ------------------------------------------------------------------
    # 4. Class mapping
    # ------------------------------------------------------------------
    df = _apply_class_mapping(
        df,
        map_to_finetune=map_to_finetune,
        mapping_file=mapping_file,
        label_cols=label_cols,
        class_mappings_name=class_mappings_name,
        con=con,
    )

    # ------------------------------------------------------------------
    # 4b. Split out skip_classes rows — they bypass all scoring
    # ------------------------------------------------------------------
    skip_classes = list(skip_classes or [])
    df_skipped: pd.DataFrame = pd.DataFrame()
    if skip_classes:
        skip_mask = df[label_col].astype(str).isin([str(c) for c in skip_classes])
        df_skipped = df[skip_mask].copy()
        df = df[~skip_mask].copy()
        print(
            f"[anomaly] skip_classes {skip_classes}: held aside "
            f"{len(df_skipped):,} rows, processing {len(df):,} rows."
        )

    # ------------------------------------------------------------------
    # 5. Prepare embedding vectors & drop NaN labels
    # ------------------------------------------------------------------
    print("[anomaly] Preparing embeddings array...")
    embed_array = df[embed_cols].to_numpy(dtype=np.float32)
    df["embedding"] = [row for row in embed_array]
    # Drop raw embedding_0..embedding_127 columns early (we keep only df["embedding"])
    df = df.drop(columns=embed_cols, errors="ignore")

    _require_label_columns(df, label_cols)

    # For hierarchical mode, enforce all levels present (required for fallback)
    count_before_drop = len(df)
    print(f"[anomaly] count_before_drop: {count_before_drop:,}")
    df = df.dropna(subset=label_cols).copy()
    count_after_drop = len(df)
    print(f"[anomaly] count_after_drop: {count_after_drop:,}")
    print(
        f"[anomaly] Dropped {count_before_drop - count_after_drop:,} rows with "
        f"missing label columns {label_cols} and dropped!"
    )

    label_col = label_cols[0]
    slice_keys = [*group_cols, h3_level_name, label_col]

    # ------------------------------------------------------------------
    # 5b. Adaptive H3 level assignment (if h3_level is a list)
    # ------------------------------------------------------------------
    if adaptive_h3:
        print(
            f"[anomaly] Adaptive H3 mode: levels {h3_levels} "
            f"(finest→coarsest), min_slice_size={min_slice_size}"
        )
        if max_slice_size is not None:
            print(f"[anomaly] Max slice size cap: {max_slice_size:,}")
        df = assign_adaptive_h3_level(
            df,
            h3_levels=h3_levels,
            label_col=label_col,
            group_cols=group_cols,
            min_slice_size=min_slice_size,
            max_slice_size=max_slice_size,
        )
        # h3_level_name is already "effective_h3_cell" for adaptive mode
        # Update slice_keys to use the effective cell
        slice_keys = [*group_cols, h3_level_name, label_col]

    # ------------------------------------------------------------------
    # 6. Merge small slices
    # ------------------------------------------------------------------
    if merge_small_slice:
        _n_slices_before_merge = df.groupby(slice_keys).ngroups
        print(
            f"[anomaly] Merging small slices (min_size={min_slice_size})... "
            f"[{_n_slices_before_merge:,} slices before merge]"
        )
        df = merge_small_slices(
            df,
            min_size=min_slice_size,
            label_col=label_col,
            h3_level_name=h3_level_name,
            group_cols=group_cols,
            max_iterations=max_merge_iterations,
        )
        _n_slices_after_merge = df.groupby(slice_keys).ngroups
        print(f"[anomaly] After merge: {_n_slices_after_merge:,} slices")
    else:
        print("[anomaly] Skipping merge_small_slices for coarse H3 level")

    # ------------------------------------------------------------------
    # 7. Hierarchical ref-class assignment + context centroid metrics
    # ------------------------------------------------------------------
    df = _add_hierarchical_ref_outlier_class(
        df,
        label_cols=label_cols,
        group_cols=group_cols,
        h3_level_name=h3_level_name,
        min_slice_size=min_slice_size,
        out_ref_class_col="ref_outlier_class",
        out_ref_level_col="ref_outlier_level",
        out_ref_group_n_col="ref_group_n",
    )

    # adding context centroid metrics
    print("[anomaly] Computing context centroid metrics...")
    context_cols = [*group_cols, h3_level_name]
    df = add_alt_class_centroid_metrics(
        df,
        label_col=label_col,
        context_cols=context_cols,
        embedding_col="embedding",
    )

    # ------------------------------------------------------------------
    # 8. Scoring
    # ------------------------------------------------------------------
    print("[anomaly] Scoring slices...")

    if len(label_cols) > 1:
        # Hierarchical scoring path
        print(f"[anomaly] Hierarchical label_domain enabled: {label_cols}")
        scored_df = score_slices_hierarchical(
            df,
            label_cols=label_cols,
            group_cols=group_cols,
            h3_level_name=h3_level_name,
            min_slice_size=min_slice_size,
            norm_percentiles=norm_percentiles,
            max_full_pairwise_n=max_full_pairwise_n,
            ref_level_col="ref_outlier_level",
            ref_class_col="ref_outlier_class",
        )
    else:
        # Single-level scoring path
        print("[anomaly] Computing per-slice centroids...")
        centroids = compute_slice_centroids(
            df,
            label_col=label_col,
            h3_level_name=h3_level_name,
            group_cols=group_cols,
        )

        df_with_centroid = df.merge(
            centroids,
            on=[*group_cols, h3_level_name, label_col],
            how="left",
        )

        def _score_group(g: pd.DataFrame) -> pd.DataFrame:
            g["slice_n"] = len(g)
            if len(g) < MIN_SCORING_SLICE_SIZE:
                g = g.copy()
                g = g[[c for c in g.columns if "embedding" not in c]]
                g["cosine_distance"] = 0.0
                g["knn_distance"] = 0.0
                g["cos_norm"] = 0.0
                g["knn_norm"] = 0.0
                g["S"] = 0.0
                g["rank_percentile"] = 0.0
                g["S_rank_min"] = 0.0
                g["cos_rank"] = 0.0
                g["knn_rank"] = 0.0
                g["S_rank"] = 0.0
                g["cos_z"] = 0.0
                g["knn_z"] = 0.0
                g["S_z"] = 0.0
                g["mean_score"] = 0.0
                g["confidence"] = 0.99
                return g

            return compute_scores_for_slice(
                g,
                centroid=g["centroid"].iloc[0],
                norm_percentiles=norm_percentiles,
                max_full_pairwise_n=max_full_pairwise_n,
                force_knn=False,
                knn_k=10,
            )

        from tqdm import tqdm as tqdm_cls

        groups = list(df_with_centroid.groupby(slice_keys, group_keys=False))
        results = []
        with tqdm_cls(groups, desc="Scoring slices", unit="slice") as pbar:
            for key, group in pbar:
                # key is a tuple matching slice_keys; last element is the label
                label_val = key[-1] if isinstance(key, tuple) else key
                n_pts = len(group)
                pbar.set_postfix_str(f"{n_pts:,} pts | {label_val}", refresh=False)
                results.append(_score_group(group))

        scored_df = pd.concat(results, ignore_index=True)

    # Drop embedding columns to save memory
    scored_df = scored_df.drop(columns=embed_cols, errors="ignore")
    # scored_df = scored_df.drop(columns=["embedding", "base_embedding"], errors="ignore")
    scored_df = scored_df.drop(columns=["base_embedding"], errors="ignore")

    # ------------------------------------------------------------------
    # 9. Flagging
    # ------------------------------------------------------------------
    print(f"[anomaly] Flagging anomalies (mode={threshold_mode})...")
    flagged_df, summary_df = flag_anomalies(
        scored_df,
        label_col=label_col,
        h3_level_name=h3_level_name,
        group_cols=group_cols,
        threshold_mode=threshold_mode,
        percentile_q=percentile_q,
        mad_k=mad_k,
        abs_threshold=abs_threshold,
        fdr_alpha=fdr_alpha,
        min_flagged_per_slice=min_flagged_per_slice,
        max_flagged_fraction=max_flagged_fraction,
    )

    # ------------------------------------------------------------------
    # 10. Confidence scoring
    # ------------------------------------------------------------------
    print("[anomaly] Computing robust confidence for flagged points...")

    flagged_df = add_confidence_from_score(
        flagged_df, score_col="mean_score", out_col="confidence"
    )

    # Boost confidence for undersized slices
    small = flagged_df["slice_n"] < MIN_SCORING_SLICE_SIZE
    flagged_df.loc[small, "confidence"] = np.maximum(
        flagged_df.loc[small, "confidence"].to_numpy(), 0.95
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 11. kNN label purity + confidence fusion
    # ------------------------------------------------------------------
    print("[anomaly] Computing kNN label purity for flagged points...")
    context_cols = [*group_cols, h3_level_name]
    flagged_df = add_knn_label_purity_for_flagged(
        df_all=df,              # this df still has embeddings
        flagged_df=flagged_df,  # from flag_anomalies
        label_col=label_col,
        context_cols=context_cols,
        embedding_col="embedding",
        purity_knn_k=10,
        cap_sqrt_k=50,
    )

    print("[anomaly] Applying confidence fusion...")
    flagged_df = apply_confidence_fusion(flagged_df)  # produces confidence_alt

    # ------------------------------------------------------------------
    # 12. Anomaly categorization
    # ------------------------------------------------------------------
    flagged_df = _assign_anomaly_categories(flagged_df)

    # ------------------------------------------------------------------
    # 13. Final cleanup & output
    # ------------------------------------------------------------------
    # Re-attach skipped-class rows with NaN for all score/outlier columns
    if not df_skipped.empty:
        # Drop raw embedding columns from skipped rows (not needed in output)
        df_skipped = df_skipped.drop(columns=embed_cols, errors="ignore")
        df_skipped = df_skipped.drop(columns=["embedding", "base_embedding"], errors="ignore")
        score_outlier_cols = [
            *_SCORE_COLS,
            "flagged", "flag_threshold", "slice_n", "undersized_slice",
            "ref_outlier_class", "ref_outlier_level", "ref_group_n",
            "self_centroid_dist_ctx", "alt_centroid_dist_ctx",
            "knn_same_label_frac_ctx", "knn_majority_frac_ctx",
            "p_margin", "p_purity",
            "confidence", "confidence_alt",
            "S_anomaly", "combined_anomaly",
        ]
        for col in score_outlier_cols:
            if col not in df_skipped.columns:
                df_skipped[col] = np.nan
        flagged_df = pd.concat([flagged_df, df_skipped], axis=0, ignore_index=True)
        print(f"[anomaly] Re-attached {len(df_skipped):,} skipped-class rows with NaN scores.")

    # Convert float64 → float32 to reduce output size
    for c in flagged_df.select_dtypes(include=["float64"]).columns:
        flagged_df[c] = flagged_df[c].astype(np.float32)

    flagged_df["geometry"] = gpd.points_from_xy(flagged_df["lon"], flagged_df["lat"])
    flagged_gdf = gpd.GeoDataFrame(flagged_df, geometry="geometry", crs="EPSG:4326")

    # Drop extra columns to reduce size
    # ["cosine_distance", "knn_distance", "cos_norm", "knn_norm",
    #              "cos_rank", "knn_rank", "S_rank", "S_rank_min",
    #              "cos_z", "knn_z", "S_z", "mean_score"]
    drop_cols = [
        "centroid",
        "cosine_distance",
        "knn_distance",
        "cos_norm",
        "knn_norm",
        "cos_rank",
        "knn_rank",
        "cos_z",
        "knn_z",
        "p_margin",
        "p_purity",
        "self_centroid_dist_ctx",
        "alt_centroid_dist_ctx",
        "knn_same_label_frac_ctx",
        "knn_majority_frac_ctx",
        "rank_percentile",
    ]
    # drop_cols = [c for c in drop_cols if c in flagged_gdf.columns]
    embed_raw = [c for c in flagged_gdf.columns if c.startswith("embedding_")]
    # drop_cols = [] + embed_raw
    drop_cols += embed_raw
    flagged_gdf = flagged_gdf.drop(columns=drop_cols, errors="ignore")

    # Incremental merge (if resuming from existing output)
    if skip_existing_samples and existing_df_full is not None:
        flagged_gdf = _merge_with_existing(flagged_gdf, existing_df_full)
    flagged_gdf.rename(columns={'confidence': 'confidence_nonoutlier', 'combined_anomaly' : 'anomaly_flag'}, inplace=True)
    # Write to disk
    _write_outputs(
        flagged_gdf, summary_df, slice_keys,
        output_samples_path, output_summary_path,
    )

    con.close()
    return flagged_gdf, summary_df


# ===================================================================
# CLI
# ===================================================================
# class_mappings_name answers "how to map", while label_domain answers "what to slice on after mapping"

if __name__ == "__main__":
    out_folder = Path(
        "/home/vito/shahs/TestFolder/Outliers/"
        "h3l2_mad_3_maxrank_groupRefId_sqrtk_norm2_98_new"
    )
    out_folder.mkdir(parents=True, exist_ok=True)
    run_pipeline(
        embeddings_db_path=(
            "/projects/worldcereal/data/cached_embeddings/"
            "embeddings_cache_LANDCOVER10_geo.duckdb"
        ),
        restrict_model_hash=None,
        label_domain="finetune_class",
        map_to_finetune=True,
        class_mappings_name="LANDCOVER10",
        # Adaptive H3: try level 3 first (finest), fall back to 2 then 1
        # for sparse regions.  Use a single int (e.g. h3_level=2) for
        # fixed-level mode (original behaviour).
        h3_level=[3, 2, 1],
        group_cols=["ref_id"],
        min_slice_size=100,
        max_slice_size=1000,  # cap to prevent runaway merging in dense areas
        merge_small_slice=True,
        threshold_mode="mad",
        percentile_q=0.96,
        mad_k=3.0,
        abs_threshold=None,
        fdr_alpha=0.05,
        min_flagged_per_slice=None,
        max_flagged_fraction=0.1,
        max_full_pairwise_n=0,  # disable full pairwise matrix calculation
        norm_percentiles=(2.0, 98.0),
        output_samples_path=str(
            out_folder
            / "outliers_h3l2_mad_3_maxrank_groupRefId_ranked_sqrtk_norm2_98_new.parquet"
        ),
        output_summary_path=str(
            out_folder
            / "outliers_h3l2_mad_3_maxrank_groupRefId_summary_sqrtk_norm2_98_new.parquet"
        ),
        debug=False,
    )
