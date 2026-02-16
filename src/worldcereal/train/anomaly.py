"""Anomaly detection utilities operating purely on cached Presto embeddings.

This version assumes a DuckDB cache already exists with columns:
``sample_id, model_hash, ref_id, ewoc_code, h3_l3_cell, embedding_0..embedding_127``.
Embeddings are never recomputed; the pipeline always loads them from the cache.
Optional label domain switching between ``ewoc_code`` and mapped ``finetune_class``.

Grouping:
- Slices are defined by: group_cols (optional) + [h3 cell] + [label col]
- group_cols defaults to [] (i.e., global per (h3, label) slices)
"""

from pathlib import Path
from statistics import median
from typing import Iterable, List, Optional, Tuple, Sequence, Union
import json

import duckdb
import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
    
from worldcereal.utils.refdata import get_class_mappings, map_classes
from sklearn.neighbors import NearestNeighbors
MIN_SCORING_SLICE_SIZE = 50

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sim = normed @ normed.T
    return 1.0 - sim

def _normalize_percentile_minmax(
    metric: np.ndarray, norm_percentiles: Tuple[float, float] = (5.0, 95.0)
) -> np.ndarray:
    """Robust-ish min-max normalization using slice percentiles; output clipped to [0,1]."""
    lo, hi = norm_percentiles
    p_lo, p_hi = np.percentile(metric, [lo, hi])
    denom = p_hi - p_lo if p_hi > p_lo else 1.0
    return np.clip((metric - p_lo) / denom, 0.0, 1.0)

def _rank_pct(metric: np.ndarray) -> np.ndarray:
    """Rank-percentile in [0,1]. Higher metric => higher rank."""
    if metric.size == 0:
        return metric
    return pd.Series(metric).rank(pct=True, method="max").to_numpy(dtype=np.float32)

def _robust_z(metric: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Robust z-score using median/MAD (not scaled)."""
    if metric.size == 0:
        return metric
    med = np.median(metric)
    mad = np.median(np.abs(metric - med))
    denom = mad if mad > 0 else 1.0
    return (metric - med) / (denom + eps)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable-ish sigmoid for moderate x ranges."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def add_alt_class_centroid_metrics(
    df: pd.DataFrame,
    *,
    label_col: str,
    context_cols: Sequence[str],
    embedding_col: str = "embedding",
) -> pd.DataFrame:
    """
    For each context group (context_cols), compute per-label centroids and for each point:
      - self_centroid_dist: cosine dist to centroid of its own label within context
      - alt_label: closest other label centroid
      - alt_centroid_dist: cosine dist to closest other label centroid
      - alt_margin: alt_centroid_dist - self_centroid_dist  (<=0 suggests confusion)
      - context_n_labels: number of labels present in context
    """
    df = df.copy()

    out_alt_label = np.full(len(df), None, dtype=object)
    out_self = np.full(len(df), np.nan, dtype=np.float32)
    out_alt = np.full(len(df), np.nan, dtype=np.float32)
    out_margin = np.full(len(df), np.nan, dtype=np.float32)
    out_nlab = np.zeros(len(df), dtype=np.uint16)

    # before grouping
    pos = np.arange(len(df), dtype=np.int64)
    df = df.copy()
    df["_pos"] = pos
    # Pre-extract embeddings into ndarray-of-rows for fast vstack in groups
    emb_series = df[embedding_col].to_numpy()

    # Keep original row indices to write back
    for _, g in df.groupby(list(context_cols), dropna=False, sort=False):
        idx = g["_pos"].to_numpy() #g.index.to_numpy()
        labels = g[label_col].to_numpy()

        # If only one label in this context, nothing to compare against
        uniq = pd.unique(labels)
        out_nlab[idx] = len(uniq)
        if len(uniq) < 2:
            continue

        X = np.vstack(emb_series[idx]).astype(np.float32, copy=False)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        # Centroids per label
        centroids = []
        cent_labels = []
        for lab in uniq:
            m = labels == lab
            C = Xn[m].mean(axis=0)
            C = C / (np.linalg.norm(C) + 1e-12)
            centroids.append(C)
            cent_labels.append(lab)

        C = np.vstack(centroids).astype(np.float32, copy=False)          # (L, D)
        sims = Xn @ C.T                                                  # (N, L)
        dists = 1.0 - sims                                               # cosine distances

        # Map each point to its own centroid column
        lab_to_j = {lab: j for j, lab in enumerate(cent_labels)}
        own_j = np.array([lab_to_j[lab] for lab in labels], dtype=np.int32)

        self_dist = dists[np.arange(len(idx)), own_j]

        # Mask own label to find nearest OTHER centroid
        dists_other = dists.copy()
        dists_other[np.arange(len(idx)), own_j] = np.inf
        alt_j = np.argmin(dists_other, axis=1)
        alt_dist = dists[np.arange(len(idx)), alt_j]
        alt_lab = np.array([cent_labels[j] for j in alt_j], dtype=object)

        out_self[idx] = self_dist.astype(np.float32, copy=False)
        out_alt[idx] = alt_dist.astype(np.float32, copy=False)
        out_alt_label[idx] = alt_lab
        out_margin[idx] = (alt_dist - self_dist).astype(np.float32, copy=False)

    df["context_n_labels"] = out_nlab
    df["self_centroid_dist_ctx"] = out_self
    df["alt_label_ctx"] = out_alt_label
    df["alt_centroid_dist_ctx"] = out_alt
    df["alt_margin_ctx"] = out_margin
    df = df.drop(columns=["_pos"], errors="ignore")
    return df

def add_knn_label_purity_for_flagged(
    df_all: pd.DataFrame,
    flagged_df: pd.DataFrame,
    *,
    label_col: str,
    context_cols: Sequence[str],
    embedding_col: str = "embedding",
    purity_knn_k: int = 10,
    cap_sqrt_k: int = 50,
) -> pd.DataFrame:
    """
    Compute kNN label-purity within each context group, but only for rows flagged==True.
    Uses df_all for embeddings and full neighbourhood; writes results back into flagged_df.
    """

    # flagged subset keys (to limit work to only contexts that contain flagged points)
    flagged_only = flagged_df.loc[flagged_df["flagged"] == True, ["sample_id", *context_cols, label_col]]
    # Use all rows in flagged_df, regardless of flagged status
    # flagged_only = flagged_df[["sample_id", *context_cols, label_col]]
    if flagged_only.empty:
        flagged_df["knn_same_label_frac_ctx"] = np.nan
        flagged_df["knn_majority_label_ctx"] = None
        flagged_df["knn_majority_frac_ctx"] = np.nan
        return flagged_df

    # Restrict df_all to only relevant contexts
    ctx_keys = flagged_only[context_cols].drop_duplicates()
    df_sub = df_all.merge(ctx_keys, on=list(context_cols), how="inner")

    # Prepare outputs keyed by sample_id (avoid attaching embeddings to flagged_df)
    out_same = {}
    out_maj_lab = {}
    out_maj_frac = {}

    # Fast membership check per context group
    flagged_ids = set(flagged_only["sample_id"].tolist())

    for _, g in df_sub.groupby(list(context_cols), dropna=False, sort=False):
        n = len(g)
        if n < 2:
            continue

        # Determine k similar to your scoring logic (sqrt(n) capped, but at least purity_knn_k)
        k = min(max(int(purity_knn_k), min(int(np.sqrt(n)), int(cap_sqrt_k))), n - 1)
        if k <= 0:
            continue

        sids = g["sample_id"].to_numpy()
        labels = g[label_col].to_numpy()

        # Indices of flagged samples inside this context
        flagged_mask = np.array([sid in flagged_ids for sid in sids], dtype=bool)
        if not flagged_mask.any():
            continue

        X = np.vstack(g[embedding_col].to_numpy()).astype(np.float32, copy=False)

        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(X)

        # Robustly drop self-neighbour if present
        q_idx = np.where(flagged_mask)[0]
        distances, neigh = nn.kneighbors(X[q_idx], return_distance=True)

        rows = []
        for r, qi in enumerate(q_idx):
            nn_ids = neigh[r]
            nn_ids = nn_ids[nn_ids != qi]
            rows.append(nn_ids[:k])
        neigh = np.vstack(rows)

        neigh_labels = labels[neigh]  # (n_flagged, k)

        for row_i, qi in enumerate(q_idx):
            sid = sids[qi]
            own = labels[qi]
            nl = neigh_labels[row_i]

            same_frac = float(np.mean(nl == own))

            # majority label among neighbours
            vals, counts = np.unique(nl, return_counts=True)
            j = int(np.argmax(counts))
            maj_lab = vals[j]
            maj_frac = float(counts[j] / len(nl))

            out_same[sid] = same_frac
            out_maj_lab[sid] = maj_lab
            out_maj_frac[sid] = maj_frac

    # Write back to flagged_df
    flagged_df = flagged_df.copy()
    flagged_df["knn_same_label_frac_ctx"] = flagged_df["sample_id"].map(out_same).astype("float32")
    flagged_df["knn_majority_label_ctx"] = flagged_df["sample_id"].map(out_maj_lab)
    flagged_df["knn_majority_frac_ctx"] = flagged_df["sample_id"].map(out_maj_frac).astype("float32")
    return flagged_df

def apply_confidence_fusion(
    df: pd.DataFrame,
    base_conf_col: str = "confidence",
    out_conf_col: str = "confidence_alt",
    # margin inputs (either provide margin_col directly or provide self/alt cols)
    margin_col: str = "alt_margin_ctx",
    self_dist_col: str = "self_centroid_dist_ctx",
    alt_dist_col: str = "alt_centroid_dist_ctx",
    # purity input
    purity_col: str = "knn_same_label_frac_ctx",
    # margin penalty params
    margin_m0: float = 0.001,
    margin_a: float = 10.0,
    # purity penalty params
    purity_beta: float = 0.5,
    # behavior
    default_factor_if_nan: float = 1.0,
    clip_exp: float = 50.0,
) -> pd.DataFrame:
    """
    Fuse auxiliary ambiguity signals into base confidence:

      confidence_alt = confidence * p_margin * p_purity

    p_margin: logistic of (alt_margin - m0)
      alt_margin = alt_centroid_dist_ctx - self_centroid_dist_ctx
      (larger margin => clearer separation => less penalty)

    p_purity: (knn_same_label_frac_ctx) ** beta
      (lower purity => stronger penalty)

    - If margin/purity missing (NaN), factor defaults to 1.0 (no penalty).
    - Output is float32 in [0,1].
    """
    if base_conf_col not in df.columns:
        raise KeyError(f"Missing base confidence column: '{base_conf_col}'")

    # Work in float64 for numerical stability, cast at end
    conf0 = pd.to_numeric(df[base_conf_col], errors="coerce").astype("float64").to_numpy()
    conf0 = np.clip(conf0, 0.0, 1.0)

    # Margin
    if margin_col in df.columns:
        margin = pd.to_numeric(df[margin_col], errors="coerce").astype("float64").to_numpy()
    elif (self_dist_col in df.columns) and (alt_dist_col in df.columns):
        self_d = pd.to_numeric(df[self_dist_col], errors="coerce").astype("float64").to_numpy()
        alt_d = pd.to_numeric(df[alt_dist_col], errors="coerce").astype("float64").to_numpy()
        margin = alt_d - self_d
    else:
        margin = np.full(len(df), np.nan, dtype="float64")

    # p_margin = sigmoid(a*(margin - m0))
    z = margin_a * (margin - margin_m0)
    z = np.clip(z, -clip_exp, clip_exp)
    p_margin = 1.0 / (1.0 + np.exp(-z))
    p_margin = np.where(np.isfinite(p_margin), p_margin, default_factor_if_nan)

    # Purity
    if purity_col in df.columns:
        pur = pd.to_numeric(df[purity_col], errors="coerce").astype("float64").to_numpy()
        pur = np.clip(pur, 0.0, 1.0)
        p_purity = np.power(pur, purity_beta)
        p_purity = np.where(np.isfinite(p_purity), p_purity, default_factor_if_nan)
    else:
        p_purity = np.full(len(df), default_factor_if_nan, dtype="float64")

    # for unflagged, p_margin and p_purity is 1.0
    if "flagged" in df.columns:
        flagged_mask = df["flagged"].fillna(False).to_numpy(dtype=bool)
        unflagged = ~flagged_mask
        p_margin = np.where(unflagged, 1.0, p_margin)
        p_purity = np.where(unflagged, 1.0, p_purity)
        
    # capping the minimum values to avoid too much confidence reduction
    p_margin = np.maximum(p_margin, 0.85)
    p_purity = np.maximum(p_purity, 0.85)

    # Final fusion
    conf = conf0 * p_margin * p_purity
    conf = np.clip(conf, 0.0, 1.0)

    out = df.copy()
    out[out_conf_col] = conf.astype(np.float32)
    out["p_margin"] = p_margin.astype(np.float32)
    out["p_purity"] = p_purity.astype(np.float32)
    return out

def _as_label_levels(label_domain: Union[str, Sequence[str]]) -> List[str]:
    """Normalize label_domain into an ordered list of label columns (fine -> coarse)."""
    if isinstance(label_domain, (list, tuple)):
        return [str(x) for x in label_domain]
    return [str(label_domain)]


def _require_label_columns(df: pd.DataFrame, label_cols: Sequence[str]) -> None:
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Requested label column(s) not found after mapping: {missing}")


_EXCEL_SUFFIXES = {
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".ods",
}


def _load_mapping_df(
    mapping_file: str,
    *,
    label_cols: Sequence[str],
    class_mappings_name: str,
) -> pd.DataFrame:
    """Load a mapping file (Excel or JSON) into a DataFrame with columns:
    - ewoc_code
    - label_cols...

    JSON formats supported:
    1) {"LANDCOVER10": {"110...": "temporary_crops", ...}, "CROPTYPE25": {...}}
       (uses `class_mappings_name` to select the inner mapping)
    2) {"110...": "label", ...}  (single label column)
    3) {"110...": {"lvl0": "...", "lvl1": "..."}, ...}  (hierarchical)
    4) {"110...": ["lvl0", "lvl1", ...], ...}  (hierarchical by position)
    5) [{"ewoc_code": "110...", "lvl0": "...", ...}, ...]  (table)
    """

    p = Path(mapping_file)
    suf = p.suffix.lower()

    if suf in _EXCEL_SUFFIXES:
        return pd.read_excel(mapping_file)

    if suf != ".json":
        raise ValueError(
            f"Unsupported mapping_file type '{suf}'. Use an Excel file ({sorted(_EXCEL_SUFFIXES)}) or a .json file."
        )

    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Table-like JSON
    if isinstance(data, list):
        return pd.DataFrame(data)

    if not isinstance(data, dict):
        raise ValueError("mapping_file JSON must be a dict or a list of records")

    # If this is a multi-mapping JSON (e.g. class_mappings.json), select the named mapping
    if class_mappings_name in data and isinstance(data[class_mappings_name], dict):
        data = data[class_mappings_name]

    rows = []
    for ewoc_code, v in data.items():
        row = {"ewoc_code": ewoc_code}

        if isinstance(v, (str, int, float)) or v is None:
            if len(label_cols) != 1:
                raise ValueError(
                    "mapping_file JSON maps ewoc_code to a single value, but label_domain requests multiple label columns. "
                    f"Expected columns: {list(label_cols)}"
                )
            row[label_cols[0]] = v

        elif isinstance(v, dict):
            for lc in label_cols:
                if lc in v:
                    row[lc] = v[lc]

        elif isinstance(v, (list, tuple)):
            if len(v) < len(label_cols):
                raise ValueError(
                    f"mapping_file JSON list for ewoc_code={ewoc_code} has {len(v)} values but {len(label_cols)} label columns were requested: {list(label_cols)}"
                )
            for lc, vv in zip(label_cols, v):
                row[lc] = vv

        else:
            raise ValueError(
                f"Unsupported JSON mapping value type for ewoc_code={ewoc_code}: {type(v)}"
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _add_hierarchical_ref_outlier_class(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    group_cols: Sequence[str],
    h3_level_name: str,
    min_slice_size: int,
    out_ref_class_col: str = "ref_outlier_class",
    out_ref_level_col: str = "ref_outlier_level",
    out_ref_group_n_col: str = "ref_group_n",
) -> pd.DataFrame:
    """
    Decide, per point (constant within each level-0 slice), which label level is used for scoring.
      - Level 0 if level-0 slice size >= min_slice_size
      - else first higher level with group size >= min_slice_size
      - else coarsest level
    Also computes:
      - slice_n: size of level-0 slice (post merge_small_slices)
      - ref_group_n: size of the chosen reference group
    """
    df = df.copy()

    if not label_cols:
        raise ValueError("label_cols must be non-empty")

    # Level-0 slice size (after any merge_small_slices modifications)
    slice_keys_v0 = [*group_cols, h3_level_name, label_cols[0]]
    df["slice_n"] = (
        df.groupby(slice_keys_v0)["sample_id"]
        .transform("size")
        .astype(np.int32)
    )

    # Single-level mode: always score at level 0
    if len(label_cols) < 2:
        df[out_ref_level_col] = np.int8(0)
        df[out_ref_class_col] = df[label_cols[0]].astype(object)
        df[out_ref_group_n_col] = df["slice_n"].astype(np.int32)
        return df

    # Group sizes for higher levels
    n_cols = {}
    for lc in label_cols[0:]:
        keys = [*group_cols, h3_level_name, lc]
        ncol = f"_n_{lc}"
        df[ncol] = (
            df.groupby(keys)["sample_id"]
            .transform("size")
            .astype(np.int32)
        )
        n_cols[lc] = ncol

    n = len(df)
    ref_level = np.full(n, -1, dtype=np.int8)

    # Level 0 if big enough
    big0 = df["slice_n"].to_numpy() >= int(min_slice_size)
    ref_level[big0] = 0

    # First higher level that meets threshold
    for lvl, lc in enumerate(label_cols[1:], start=1):
        ncol = n_cols[lc]
        ok = (ref_level == -1) & (df[ncol].to_numpy() >= int(min_slice_size))
        ref_level[ok] = np.int8(lvl)

    # Remaining: coarsest
    ref_level[ref_level == -1] = np.int8(len(label_cols) - 1)

    df[out_ref_level_col] = ref_level

    # ref_outlier_class and ref_group_n
    ref_class = df[label_cols[0]].astype(object).to_numpy()
    ref_n = df["slice_n"].to_numpy()

    for lvl, lc in enumerate(label_cols[1:], start=1):
        m = ref_level == lvl
        if not m.any():
            continue
        ref_class[m] = df[lc].astype(object).to_numpy()[m]
        ref_n[m] = df[n_cols[lc]].to_numpy()[m]

    df[out_ref_class_col] = ref_class
    df[out_ref_group_n_col] = ref_n.astype(np.int32)

    return df


_SCORE_COLS = [
    "cosine_distance",
    "knn_distance",
    "cos_norm",
    "knn_norm",
    "S",
    "rank_percentile",
    "cos_rank",
    "knn_rank",
    "S_rank",
    "S_rank_min",
    "cos_z",
    "knn_z",
    "S_z",
    "mean_score",
]


def _score_group_simple(
    g: pd.DataFrame,
    norm_percentiles: Tuple[float, float],
    max_full_pairwise_n: Optional[int],
) -> pd.DataFrame:
    # g must contain at least ["sample_id", "embedding"]
    if len(g) < MIN_SCORING_SLICE_SIZE:
        g = g.copy()
        for c in _SCORE_COLS:
            g[c] = 0.0
        return g

    return compute_scores_for_slice(
        g,
        centroid=None,  # computed inside
        norm_percentiles=norm_percentiles,
        max_full_pairwise_n=max_full_pairwise_n,
        force_knn=False,
        knn_k=10,
    )


def score_slices_hierarchical(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    group_cols: Sequence[str],
    h3_level_name: str,
    min_slice_size: int,
    norm_percentiles: Tuple[float, float],
    max_full_pairwise_n: Optional[int],
    ref_level_col: str = "ref_outlier_level",
    ref_class_col: str = "ref_outlier_class",
) -> pd.DataFrame:
    """
    Scores points by level-0 slices, but for undersized level-0 slices it uses
    a larger reference set defined by a coarser label level.

    Scores are written back ONLY for the original undersized slice points.
    """
    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique for hierarchical scoring updates")

    df = df.copy()

    for c in _SCORE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Ensure slice_n exists (size of level-0 slice)
    slice_keys_v0 = [*group_cols, h3_level_name, label_cols[0]]
    if "slice_n" not in df.columns:
        df["slice_n"] = (
            df.groupby(slice_keys_v0)["sample_id"]
            .transform("size")
            .astype(np.int32)
        )

    df_idx = df.set_index("sample_id", drop=False)

    from tqdm import tqdm
    tqdm.pandas()

    # 1) Score rows that use level 0 directly (normal path)
    direct = df_idx[df_idx[ref_level_col] == 0]
    if not direct.empty:
        g0 = direct[[*group_cols, h3_level_name, label_cols[0], "sample_id", "embedding"]].reset_index(drop=True)

        scored0 = (
            g0.groupby([*group_cols, h3_level_name, label_cols[0]], group_keys=False)
            .progress_apply(lambda g: _score_group_simple(g, norm_percentiles, max_full_pairwise_n))
            .reset_index(drop=True)
        )
        scored0 = scored0.set_index("sample_id", drop=False)
        df_idx.loc[scored0.index, _SCORE_COLS] = scored0[_SCORE_COLS].to_numpy()

    # 2) Score fallback groups once, then write back only to target rows
    fallback = df_idx[df_idx[ref_level_col] > 0]
    if not fallback.empty:
        fb_keys = [ref_level_col, *group_cols, h3_level_name, ref_class_col]
        target_map = fallback.groupby(fb_keys)["sample_id"].apply(list)

        for key, target_ids in tqdm(target_map.items(), total=len(target_map), desc="Scoring fallback ref groups"):
            ref_level = int(key[0])
            ref_class = key[-1]
            ref_label_col = label_cols[ref_level]

            # Build reference set mask on the FULL dataframe
            m = (df_idx[ref_label_col].astype(object).to_numpy() == ref_class)

            offset = 1
            for i, gc in enumerate(group_cols):
                m &= (df_idx[gc].astype(object).to_numpy() == key[offset + i])

            h3_val = key[offset + len(group_cols)]
            m &= (df_idx[h3_level_name].astype(object).to_numpy() == h3_val)

            ref_df = df_idx.loc[m, ["sample_id", "embedding"]].reset_index(drop=True)
            if ref_df.empty:
                continue

            scored_ref = _score_group_simple(ref_df, norm_percentiles, max_full_pairwise_n)
            scored_ref = scored_ref[scored_ref["sample_id"].isin(target_ids)]
            if scored_ref.empty:
                continue

            scored_ref = scored_ref.set_index("sample_id", drop=False)
            df_idx.loc[scored_ref.index, _SCORE_COLS] = scored_ref[_SCORE_COLS].to_numpy()

    # Hard check: no NaNs in required scoring columns
    if df_idx[_SCORE_COLS].isna().any().any():
        bad = df_idx[df_idx[_SCORE_COLS].isna().any(axis=1)][["sample_id", ref_level_col, ref_class_col]].head(20)
        raise ValueError(f"Hierarchical scoring left NaNs in score columns. Example rows:\n{bad}")

    return df_idx.reset_index(drop=True)

def merge_small_slices(
    df: pd.DataFrame,
    min_size: int = 100,
    label_col: str = "ewoc_code",
    h3_level_name: str = "h3_l3_cell",
    group_cols: Optional[Sequence[str]] = None,
    max_iterations: int = 25,
    min_improvement: float = 0.05,
    mark_undersized: bool = True,
) -> pd.DataFrame:
    """Merge small slices with neighbouring H3 cells until they exceed ``min_size``.

    A "slice" is defined by: group_cols + [label_col] + [h3_level_name]
    """

    df = df.copy()
    group_cols = list(group_cols or [])
    key_cols = [*group_cols, label_col, h3_level_name]

    # Precompute H3 neighbors for all cells present to avoid repeated calls
    unique_cells = df[h3_level_name].unique().tolist()
    neighbour_map = {
        cell: list(set(h3.grid_disk(cell, 1)) - {cell}) for cell in unique_cells
    }

    # Iterative bulk merge
    counts = df.groupby(key_cols).size()
    for _ in range(max_iterations):
        small = counts[counts < min_size]
        if small.empty:
            break

        before_total_small = int(small.sum())

        # Build a DataFrame of candidate merges: each small key with its best neighbour
        merge_rows: List[Tuple] = []
        for key, _ in small.items():
            # key is a tuple: (*group_vals, label_value, cell)
            if not isinstance(key, tuple):
                key = (key,)
            group_vals = key[:-2]
            label_value = key[-2]
            cell = key[-1]

            neighbours = neighbour_map.get(cell, [])
            if not neighbours:
                continue

            # Find neighbour with maximum existing count for the same (*group_vals, label)
            best_target = None
            best_count = 0
            for n in neighbours:
                c = int(counts.get((*group_vals, label_value, n), 0))
                if c > best_count:
                    best_count = c
                    best_target = n

            if best_target is not None and best_count > 0:
                merge_rows.append((*group_vals, label_value, cell, best_target))

        if not merge_rows:
            break

        merge_df = pd.DataFrame(
            merge_rows, columns=[*group_cols, label_col, h3_level_name, "target_cell"]
        )

        # Apply merges in bulk via join on keys
        df = df.merge(merge_df, on=key_cols, how="left")
        mask = df["target_cell"].notna()
        if mask.any():
            df.loc[mask, h3_level_name] = df.loc[mask, "target_cell"].astype(str)
        df = df.drop(columns=["target_cell"], errors="ignore")

        # Recompute counts and check improvement
        counts = df.groupby(key_cols).size()
        small_after = counts[counts < min_size]
        after_total_small = int(small_after.sum())
        improvement = (
            (before_total_small - after_total_small) / before_total_small
            if before_total_small > 0
            else 0.0
        )
        if improvement < min_improvement:
            break

    if mark_undersized:
        final_counts = df.groupby(key_cols).size()
        undersized_keys = set(final_counts[final_counts < min_size].index)
        df["undersized_slice"] = df.set_index(key_cols).index.isin(undersized_keys)
    
    df["slice_id"] = df.groupby(key_cols, sort=True).ngroup().astype(np.uint32)

    return df


def compute_slice_centroids(
    df: pd.DataFrame,
    label_col: str = "ewoc_code",
    h3_level_name: str = "h3_l3_cell",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute centroids of embeddings per slice (group_cols + h3 + label)."""

    group_cols = list(group_cols or [])
    group_keys = [*group_cols, h3_level_name, label_col]

    def _centroid(emb_list: Iterable[np.ndarray]) -> np.ndarray:
        arr = np.vstack(list(emb_list))
        return arr.mean(axis=0)

    centroids = (
        df.groupby(group_keys)["embedding"]
        .apply(_centroid)
        .reset_index()
        .rename(columns={"embedding": "centroid"})
    )
    return centroids



def compute_scores_for_slice(
    df_slice: pd.DataFrame,
    centroid: Optional[np.ndarray] = None,
    norm_percentiles: Tuple[float, float] = (5.0, 95.0),
    max_full_pairwise_n: Optional[int] = None,
    force_knn: bool = False,
    knn_k: int = 10,
) -> pd.DataFrame:
    """Compute anomaly scores for a single slice×class dataframe.

    This function auto-selects the kNN computation strategy:
      - Full pairwise NxN distance matrix if feasible
      - kNN-only computation (NearestNeighbors) if N is large or force_knn=True

    Returns:
      - cosine_distance, knn_distance, cos_norm, knn_norm, S, rank_percentile
      - cos_rank, knn_rank, S_rank, cos_z, knn_z, S_z
    """

    embeddings = np.vstack(df_slice["embedding"].to_numpy()).astype("float32", copy=False)
    n = embeddings.shape[0]

    if centroid is None:
        centroid = embeddings.mean(axis=0)

    # Cosine distance to centroid
    cos_dist = np.array([1.0 - _cosine_similarity(e, centroid) for e in embeddings], dtype=np.float32)

    # Choose kNN strategy, sqrt(N) or fixed k of 10
    # k = min(int(knn_k), n - 1) if n > 1 else 0
    
    SQRT_N = int(np.sqrt(n))
    k = min(max(int(knn_k), min(int(SQRT_N), 50)), n-1) if n > 1 else 0
    
    use_knn_only = force_knn or (max_full_pairwise_n is not None and n > max_full_pairwise_n)

    if k <= 0:
        knn_dist = np.zeros(n, dtype=np.float32)
    elif use_knn_only:
        # kNN-only (memory friendly)
        nn = NearestNeighbors(
            n_neighbors=k + 1,          # include self, then drop it
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        knn_dist = distances[:, 1:].mean(axis=1).astype(np.float32, copy=False)
    else:
        # Full pairwise NxN (more memory intensive)
        dist_matrix = _cosine_distance_matrix(embeddings)
        np.fill_diagonal(dist_matrix, np.inf)
        knn_dist = np.partition(dist_matrix, k, axis=1)[:, :k].mean(axis=1).astype(np.float32, copy=False)

    knn_dist = np.nan_to_num(knn_dist, nan=0.0, posinf=0.0, neginf=0.0)

    # Existing percentile-based normalization (parameterized)
    cos_norm = _normalize_percentile_minmax(cos_dist, norm_percentiles=norm_percentiles)
    knn_norm = _normalize_percentile_minmax(knn_dist, norm_percentiles=norm_percentiles)
    scores = 0.5 * (cos_norm + knn_norm)

    # Rank-based scores
    cos_rank = _rank_pct(cos_dist)
    knn_rank = _rank_pct(knn_dist)
    s_rank = 0.5 * (cos_rank + knn_rank)
    # rank_percentile_rank = pd.Series(s_rank).rank(pct=True, method="max").to_numpy(dtype=np.float32)

    # Robust z-score scores (median/MAD) + sigmoid squashing
    cos_z = _robust_z(cos_dist)
    knn_z = _robust_z(knn_dist)
    s_z = 0.5 * (_sigmoid(cos_z) + _sigmoid(knn_z))

    ranks = pd.Series(s_rank).rank(pct=True, method="max").to_numpy()

    df_scored = df_slice.copy()[[c for c in df_slice.columns if "embedding" not in c]]
    df_scored["cosine_distance"] = cos_dist
    df_scored["knn_distance"] = knn_dist
    df_scored["cos_norm"] = cos_norm
    df_scored["knn_norm"] = knn_norm
    df_scored["S"] = scores
    df_scored["rank_percentile"] = ranks.astype(np.float32)

    
    df_scored["cos_rank"] = cos_rank
    df_scored["knn_rank"] = knn_rank
    df_scored["S_rank"] = s_rank
    df_scored["S_rank_min"] = np.minimum(cos_rank, knn_rank).astype(np.float32)
    # df_scored["rank_percentile_rank"] = rank_percentile_rank
    df_scored["cos_z"] = cos_z
    df_scored["knn_z"] = knn_z
    df_scored["S_z"] = s_z
    # create an confidence score column for easier interpretation based on average of S_rank, S_rank_min, S_z
    df_scored["mean_score"] = (
        (df_scored["S_rank"] + df_scored["S_rank_min"] + df_scored["S_z"]) / 3.0
    ).astype(np.float32)
    
    return df_scored



def add_flagged_robust_confidence(
    df: pd.DataFrame,
    score_col: str = "mean_score",
    flagged_col: str = "flagged",
    out_z_col: str = "z_mad",
    out_conf_col: str = "confidence",
    # mapping params
    z_knee: float = 3.0,        # where confidence becomes 0.5
    eps_conf: float = 1e-3,     # desired confidence at z=10 (very extreme)
    z_extreme: float = 10.0,
    clip_exp: float = 50.0,
    default_unflagged_conf: float = 1.0,
) -> pd.DataFrame:
    """
    For each slice (i.e., current df passed in is already one slice),
    compute robust MAD-z from score_col and assign confidence:
      - if not flagged: confidence = default_unflagged_conf
      - if flagged: confidence = 1 / (1 + exp(k*(z - z_knee)))

    Also writes z_mad for debugging/auditing.
    """
    out = df.copy()

    x = pd.to_numeric(out[score_col], errors="coerce").astype("float64")
    med = float(np.nanmedian(x.to_numpy()))
    abs_dev = np.abs(x - med)
    mad = float(np.nanmedian(abs_dev.to_numpy()))
    denom = mad if (np.isfinite(mad) and mad > 0.0) else 1.0

    z = (x - med) / denom
    z = z.clip(lower=0.0)  # only penalize high-side outliers; keep non-outliers at z=0

    out[out_z_col] = z.astype(np.float32)

    # choose k so that confidence(z_extreme) ~= eps_conf
    # conf(z) = 1/(1+exp(k*(z - z_knee)))  -> exp(k*(z_extreme-z_knee)) = 1/eps - 1
    k = float(np.log(1.0 / eps_conf - 1.0) / max(1e-6, (z_extreme - z_knee)))

    z_arg = np.clip(k * (z.to_numpy() - z_knee), -clip_exp, clip_exp)
    conf_flagged = 1.0 / (1.0 + np.exp(z_arg))

    flagged = out[flagged_col].fillna(False).to_numpy(dtype=bool)
    conf = np.full(len(out), float(default_unflagged_conf), dtype="float64")
    conf[flagged] = conf_flagged[flagged]

    out[out_conf_col] = np.clip(conf, 0.0, 1.0).astype(np.float32)
    return out

# def add_confidence_from_score(
#     df,
#     score_col="mean_score",
#     out_col="confidence",
#     t=0.985,
#     eps=0.05,
#     clip=50.0,
# ):
#     x = pd.to_numeric(df[score_col], errors="coerce").astype("float64").clip(0.0, 1.0)
#     k = float(np.log(1.0 / eps - 1.0) / (1.0 - t))
#     # z = np.clip(k * (x.to_numpy() - t), -clip, clip)
#     # no clipping
#     z = k * (x.to_numpy() - t)
#     df[out_col] = (1.0 / (1.0 + np.exp(z))).astype(np.float32)
#     # df[out_col] = np.where(df["flagged"], df[out_col], 1.0).astype(np.float32)
#     return df


def add_confidence_from_score(
    df: pd.DataFrame,
    score_col: str = "mean_score",
    out_col: str = "confidence",
    t: float = 0.975,          # knee: confidence starts dropping after this
    alpha: float = 0.3,        # tail sharpness (bigger => harsher near 1)
    conf_min: float = 0.01,    # never go below this
    eps: float = 1e-9,         # numerical stability near 1
) -> pd.DataFrame:
    """
    Accelerating confidence drop as score -> 1, with hard floor conf_min.

    y = clip((x - t) / (1 - t), 0, 1)
    conf_raw = exp(-alpha * y / (1 - y + eps))
    confidence = conf_min + (1 - conf_min) * conf_raw

    - x <= t  => y=0 => conf_raw=1 => confidence=1
    - x -> 1  => y->1 => conf_raw->0 => confidence->conf_min (not 0)
    """
    x = pd.to_numeric(df[score_col], errors="coerce").astype("float64")
    x = x.clip(lower=0.0, upper=1.0).to_numpy()

    if not (0.0 < t < 1.0):
        raise ValueError("t must be in (0, 1)")
    if not (alpha > 0.0):
        raise ValueError("alpha must be > 0")
    if not (0.0 < conf_min < 1.0):
        raise ValueError("conf_min must be in (0, 1)")
    if not (eps > 0.0):
        raise ValueError("eps must be > 0")

    y = (x - t) / max(1e-12, (1.0 - t))
    y = np.clip(y, 0.0, 1.0)

    conf_raw = np.exp(-alpha * (y / (1.0 - y + eps)))
    conf = conf_min + (1.0 - conf_min) * conf_raw
    conf = np.clip(conf, conf_min, 1.0).astype(np.float32)

    df[out_col] = conf
    return df


def flag_anomalies(
    df_scores: pd.DataFrame,
    label_col: str = "ewoc_code",
    threshold_mode: str = "percentile",
    h3_level_name: str = "h3_l3_cell",
    group_cols: Optional[Sequence[str]] = None,
    percentile_q: float = 0.96,
    mad_k: float = 3.0,
    abs_threshold: Optional[float] = None,
    fdr_alpha: float = 0.05,
    min_flagged_per_slice: Optional[int] = None,
    max_flagged_fraction: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flag anomalies within each slice group.

    Slice keys: group_cols + [h3_level_name] + [label_col]
    """

    group_cols = list(group_cols or [])
    group_keys = [*group_cols, h3_level_name, label_col]
    flag_col = "S"
    def _flag_group(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        if g.empty:
            g["flagged"] = False
            return g

        g = g.sort_values(flag_col, ascending=False)
        if threshold_mode == "percentile":
            thr = g[flag_col].quantile(percentile_q)
            g["flagged"] = g[flag_col] >= thr
        elif threshold_mode == "mad":
            med = g[flag_col].median()
            mad = (g[flag_col] - med).abs().median()
            thr = med + mad_k * (mad if mad > 0 else 1.0)
            g["flagged"] = g[flag_col] >= thr
        elif threshold_mode == "absolute":
            if abs_threshold is None:
                raise ValueError(
                    "abs_threshold must be set when threshold_mode='absolute'"
                )
            g["flagged"] = g[flag_col] >= float(abs_threshold)
        elif threshold_mode == "fdr":
            n = len(g)
            ranks = g[flag_col].rank(ascending=False, method="max")
            pvals = ranks / (n + 1.0)
            order = np.argsort(pvals.to_numpy())
            p_sorted = pvals.to_numpy()[order]
            thresh = (np.arange(1, n + 1) / n) * fdr_alpha
            passed = p_sorted <= thresh
            if passed.any():
                k = int(np.max(np.where(passed)))
                p_cut = p_sorted[k]
                g["flagged"] = pvals <= p_cut
            else:
                g["flagged"] = False
        else:
            raise ValueError(
                "threshold_mode must be one of {'percentile','mad','absolute','fdr'}"
            )

        n = len(g)
        flags = g["flagged"].to_numpy().astype(bool)
        n_flag = int(flags.sum())

        if max_flagged_fraction is not None:
            max_allowed = int(np.floor(max_flagged_fraction * n))
            if max_allowed < 0:
                max_allowed = 0
            if n_flag > max_allowed:
                if max_allowed == 0:
                    flags[:] = False
                else:
                    flags[:] = False
                    flags[:max_allowed] = True
                n_flag = int(flags.sum())

        if min_flagged_per_slice is not None and min_flagged_per_slice > 0:
            if n_flag < min_flagged_per_slice:
                k = min(min_flagged_per_slice, n)
                flags[:] = False
                flags[:k] = True
                n_flag = int(flags.sum())

        g["flagged"] = flags
        return g

    flagged_df = (
        df_scores.groupby(group_keys, group_keys=False)
        .apply(_flag_group)
        .reset_index(drop=True)
    )

    summary = (
        flagged_df.groupby(group_keys)
        .agg(total_samples=("S", "size"), flagged_samples=("flagged", "sum"))
        .reset_index()
    )
    summary["flagged_fraction"] = summary["flagged_samples"] / summary["total_samples"]
    return flagged_df, summary

def run_pipeline(
    embeddings_db_path: str,
    restrict_model_hash: Optional[str] = None,
    label_domain: Union[str, Sequence[str]] = "ewoc_code",
    map_to_finetune: bool = False,
    class_mappings_name: str = "LANDCOVER10",
    mapping_file: Optional[str] = None,
    h3_level: int = 3,
    group_cols: Optional[Sequence[str]] = None,
    min_slice_size: int = 100,
    merge_small_slice: bool = True,
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
    skip_existing_samples: bool = False,  # NEW: incremental/resumable processing
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run anomaly detection using only cached embeddings.

    Grouping:
      slice = group_cols + [h3 cell at chosen level] + [label_col]

    norm_percentiles:
      Percentiles used for per-slice min-max normalization of cosine_distance and knn_distance.
      Default (5,95) preserves existing behavior.

    skip_existing_samples:
      If True and output_samples_path exists, loads existing results, skips already-processed
      sample_id rows, computes only missing ones, then appends old + new and writes back.
      This does NOT recompute outlier scores for existing sample_ids.
    """

    group_cols = list(group_cols or [])
    label_cols = _as_label_levels(label_domain)
    label_col = label_cols[0]  # keep existing logic anchored to level-0

    # only enforce when mapping_file is not provided
    if mapping_file is None:
        if isinstance(label_domain, (list, tuple)):
            raise ValueError(
                "Hierarchical label_domain requires mapping_file (labels provided by Excel or JSON)."
            )
        if label_domain not in {"ewoc_code", "finetune_class", "balancing_class"}:
            raise ValueError("label_domain must be 'ewoc_code' or 'finetune_class'")

    print("[anomaly] Connecting DuckDB and loading cached embeddings...")
    con = duckdb.connect(embeddings_db_path)
    cols_df = con.execute("PRAGMA table_info('embeddings_cache')").fetchdf()
    embed_cols = [c for c in cols_df.name.tolist() if c.startswith("embedding_")]

    # Ensure we select any requested grouping columns from DuckDB
    base_cols = [
        "sample_id",
        "ewoc_code",
        "model_hash",
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
    print(f"[anomaly] Loaded {len(df):,} rows from embeddings_cache")
    if df.empty:
        con.close()
        raise ValueError(
            "No rows loaded from embeddings_cache. Check model_hash or DB path."
        )

    # ----------------------------
    # NEW: Incremental mode
    # ----------------------------
    existing_df_full = None
    existing_ids: set = set()

    if skip_existing_samples:
        if not output_samples_path:
            print("[anomaly] WARNING: skip_existing_samples=True but output_samples_path not set. Processing all samples.")
        else:
            out_path = Path(output_samples_path)
            if out_path.exists():
                print(f"[anomaly] Loading existing results from {output_samples_path}...")
                # Load FULL existing output so we can append without losing columns
                existing_df_full = gpd.read_parquet(output_samples_path)
                if "sample_id" not in existing_df_full.columns:
                    con.close()
                    raise ValueError(
                        f"Existing output_samples_path has no 'sample_id' column: {output_samples_path}"
                    )
                existing_ids = set(existing_df_full["sample_id"].astype(str).unique())

                before_count = len(df)
                # Ensure types consistent for filtering
                df_sample_ids = df["sample_id"].astype(str)
                df = df[~df_sample_ids.isin(existing_ids)].copy()
                after_count = len(df)

                print(f"[anomaly] Found {len(existing_ids):,} existing samples")
                print(f"[anomaly] Filtering: {before_count:,} -> {after_count:,} rows to process")

                if df.empty:
                    print("[anomaly] All samples already processed. Returning existing results...")
                    con.close()
                    # Return the existing full results; summary not recomputed here
                    return existing_df_full, None
            else:
                print(f"[anomaly] skip_existing_samples=True but output file doesn't exist yet: {output_samples_path}")
                print(f"[anomaly] Processing all {len(df):,} samples from scratch...")

    # ----------------------------
    # Existing validations / setup
    # ----------------------------
    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        con.close()
        raise ValueError(
            f"Requested group_cols not found in loaded data: {missing_group_cols}"
        )

    h3_level_name = f"h3_l{h3_level}_cell"
    if h3_level != 3:
        df[h3_level_name] = df["h3_l3_cell"].apply(lambda h: h3.cell_to_parent(h, h3_level))

    if df["ewoc_code"].dtype != np.int64:
        df["ewoc_code"] = pd.to_numeric(df["ewoc_code"], errors="coerce").astype("Int64")

    if debug:
        print("[DUBUG] Running in debug mode: restricting to small sample of data, only loading 10 H3 cells...")
        # df = df.head(5000)
        # load only two h3 cells for faster testing
        sample_cells = df[h3_level_name].unique()[:10].tolist()
        df = df[df[h3_level_name].isin(sample_cells)]

    if map_to_finetune:
        print(f"[anomaly] Mapping classes using '{class_mappings_name}'...")
        df = map_classes(df, class_mappings_name)

    elif mapping_file is not None:
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
        map_df["ewoc_code_clean"] = map_df["ewoc_code"].astype(str).str.replace("-", "", regex=False)

        df["ewoc_code_clean"] = df["ewoc_code"].astype(str).str.replace("-", "", regex=False)

        # If any label columns were already present in df, drop them to avoid collisions
        df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")

        df = df.merge(
            map_df[["ewoc_code_clean", *label_cols]],
            on="ewoc_code_clean",
            how="left",
        )

        df = df.drop(columns=["ewoc_code_clean"], errors="ignore")

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
    print(f"[anomaly] Dropped {count_before_drop - count_after_drop:,} rows with missing label columns {label_cols} and dropped!")

    label_col = label_cols[0]

    slice_keys = [*group_cols, h3_level_name, label_col]

    if merge_small_slice:
        print(f"[anomaly] Merging small slices (min_size={min_slice_size})...")
        df = merge_small_slices(
            df,
            min_size=min_slice_size,
            label_col=label_col,
            h3_level_name=h3_level_name,
            group_cols=group_cols,
        )
    else:
        print("[anomaly] Skipping merge_small_slices for coarse H3 level")

    # Decide scoring label level (fine->coarse) and add ref columns
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

    # adding centext centroid metrics
    print("[anomaly] Computing context centroid metrics...")
    context_cols = [*group_cols, h3_level_name]
    df = add_alt_class_centroid_metrics(
        df,
        label_col=label_col,
        context_cols=context_cols,
        embedding_col="embedding",
    )

    print("[anomaly] Scoring slices...")

    if len(label_cols) > 1:
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

        from tqdm import tqdm
        tqdm.pandas()
        scored_df = (
            df_with_centroid.groupby(slice_keys, group_keys=False)
            .progress_apply(_score_group)
            .reset_index(drop=True)
        )

    # drop embeddings to save memory, any column name starting with "embedding_"
    scored_df = scored_df.drop(columns=embed_cols, errors="ignore")

    # scored_df = scored_df.drop(columns=["embedding", "base_embedding"], errors="ignore")
    scored_df = scored_df.drop(columns=["base_embedding"], errors="ignore")

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

    print("[anomaly] Computing robust confidence for flagged points...")
    # flagged_df = (
    #     flagged_df.groupby(slice_keys, group_keys=False)
    #     .apply(lambda g: add_flagged_robust_confidence(
    #         g,
    #         score_col="mean_score",
    #         flagged_col="flagged",
    #         out_z_col="z_mad",
    #         out_conf_col="confidence",
    #         z_knee=mad_k,
    #         eps_conf=1e-3,
    #         z_extreme=10.0,
    #     ))
    #     .reset_index(drop=True)
    # )

    # t_mean = float(flagged_df.loc[flagged_df["flagged"], "mean_score"].median())
    # print(f"[anomaly] Using t_mean={t_mean:.4f} for confidence mapping...")
    flagged_df = add_confidence_from_score(flagged_df, score_col="mean_score", out_col="confidence")

    # boost confidence for undersized slices
    small = flagged_df["slice_n"] < MIN_SCORING_SLICE_SIZE
    flagged_df.loc[small, "confidence"] = np.maximum(
        flagged_df.loc[small, "confidence"].to_numpy(),
        0.95
    ).astype(np.float32)

    print("[anomaly] Computing kNN label purity for flagged points...")
    context_cols = [*group_cols, h3_level_name]
    flagged_df = add_knn_label_purity_for_flagged(
        df_all=df,                 # this df still has embeddings
        flagged_df=flagged_df,      # from flag_anomalies
        label_col=label_col,
        context_cols=context_cols,
        embedding_col="embedding",
        purity_knn_k=10,
        cap_sqrt_k=50,
    )
    print("[anomaly] Applying confidence fusion...")
    flagged_df = apply_confidence_fusion(flagged_df)            # produces confidence_alt

    # flagged_df["S_rank_min"] = np.minimum(flagged_df["cos_rank"], flagged_df["knn_rank"])
    is_flagged = flagged_df["flagged"].fillna(False).to_numpy(dtype=bool)

    S_anomaly = 'S_anomaly'
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

    # Addiotnal anomaly categories based on combination of scores
    combined_anomaly = 'combined_anomaly'
    flagged_df[combined_anomaly] = "normal"
    flagged_df.loc[is_flagged, combined_anomaly] = "flagged"

    # Consensus-based escalation using multiple score variants (more robust than S alone)
    # All three are in [0,1] where higher => more anomalous
    suspect_thr = 0.98
    candidate_thr = 0.99

    suspect_k_of_m = 2   # require 2-of-3 signals for "suspect"
    candidate_k_of_m = 3 # require 3-of-3 signals for "candidate" (strict)

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
        low_conf = flagged_df["undersized_slice"] == True
        # candidate -> suspect; suspect -> flagged (only when undersized)
        flagged_df.loc[low_conf & (flagged_df[S_anomaly] == "candidate"), S_anomaly] = "suspect"
        flagged_df.loc[low_conf & (flagged_df[S_anomaly] == "suspect"), S_anomaly] = "flagged"
        flagged_df.loc[low_conf & (flagged_df[combined_anomaly] == "candidate"), combined_anomaly] = "suspect"
        flagged_df.loc[low_conf & (flagged_df[combined_anomaly] == "suspect"), combined_anomaly] = "flagged"

    # convert any flot64 to float32 to reduce output size
    for c in flagged_df.select_dtypes(include=['float64']).columns:
        flagged_df[c] = flagged_df[c].astype(np.float32)

    flagged_df["geometry"] = gpd.points_from_xy(flagged_df["lon"], flagged_df["lat"])

    flagged_gdf = gpd.GeoDataFrame(flagged_df, geometry="geometry", crs="EPSG:4326")
    # drop extra columns to reduce size
    # ["cosine_distance", "knn_distance", "cos_norm", "knn_norm",
    #              "cos_rank", "knn_rank", "S_rank", "S_rank_min",
    #              "cos_z", "knn_z", "S_z", "mean_score"]
    drop_cols = ["centroid", "cosine_distance", "knn_distance", "cos_norm", "knn_norm",
                 "cos_rank", "knn_rank", "cos_z", "knn_z", "p_margin","p_purity",
                 "self_centroid_dist_ctx", "alt_centroid_dist_ctx","knn_same_label_frac_ctx",
                 "knn_majority_frac_ctx", "rank_percentile"]
    # drop_cols = [c for c in drop_cols if c in flagged_gdf.columns]
    embed_raw = [c for c in flagged_gdf.columns if c.startswith("embedding_")]
    # drop_cols = [] + embed_raw
    drop_cols += embed_raw
    flagged_gdf = flagged_gdf.drop(columns=drop_cols, errors="ignore")

    # ----------------------------
    # NEW: Append existing + new (no recompute for existing sample_id)
    # ----------------------------
    if skip_existing_samples and existing_df_full is not None:
        print(f"[anomaly] Merging {len(flagged_gdf):,} new results with {len(existing_df_full):,} existing results...")

        # Align schemas (union of columns) to avoid missing-column issues
        all_cols = list(dict.fromkeys([*existing_df_full.columns.tolist(), *flagged_gdf.columns.tolist()]))
        existing_aligned = existing_df_full.reindex(columns=all_cols)
        new_aligned = flagged_gdf.reindex(columns=all_cols)

        combined = pd.concat([existing_aligned, new_aligned], axis=0, ignore_index=True)

        # Safety: if any overlaps happen, keep the last occurrence (new wins)
        if "sample_id" in combined.columns:
            combined["sample_id"] = combined["sample_id"].astype(str)
            combined = combined.drop_duplicates(subset=["sample_id"], keep="last").reset_index(drop=True)

        flagged_gdf = combined
        print(f"[anomaly] Total combined: {len(flagged_gdf):,} samples")

    if output_samples_path:
        print(f"[anomaly] Writing flagged samples -> {output_samples_path}")
        # flagged_gdf = flagged_gdf.drop(
        #     columns=["embedding", "base_embedding"], errors="ignore"
        # )
        flagged_gdf = flagged_gdf.drop(
            columns=["base_embedding"], errors="ignore"
        )
        flagged_gdf.to_parquet(output_samples_path, index=False)

    if output_summary_path:
        print(f"[anomaly] Writing summary -> {output_summary_path}")
        summary_df.to_parquet(output_summary_path, index=False)
        summary_df.to_excel(Path(output_summary_path).with_suffix(".xlsx"),
            index=False,)

        cross_long = (
            flagged_gdf.groupby([*slice_keys, S_anomaly, combined_anomaly])
            .size()
            .reset_index(name="n"))
        cross_long.to_parquet(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_long.parquet"
            ),
            index=False,)
        cross_long.to_excel(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_long.xlsx"
            ),
            index=False,)

        # wide matrix form per slice with flattened column names
        cross_wide = cross_long.pivot_table(
            index=slice_keys,
            columns=[S_anomaly, combined_anomaly],
            values="n",
            fill_value=0,
            aggfunc="sum",
        )

        # Flatten MultiIndex columns -> e.g. "S=candidate__C=suspect"
        cross_wide.columns = [f"S={s}__C={c}" for (s, c) in cross_wide.columns.to_list()]
        cross_wide = cross_wide.reset_index()

        cross_wide.to_parquet(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_wide.parquet"
            ),
            index=False,)
        cross_wide.to_excel(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_wide.xlsx"
            ),
            index=True)

    con.close()
    return flagged_df, summary_df


if __name__ == "__main__":
    out_folder = Path("/home/vito/shahs/TestFolder/Outliers/h3l2_mad_3_maxrank_groupRefId_sqrtk_norm2_98_new")
    out_folder.mkdir(parents=True, exist_ok=True)
    run_pipeline(
        embeddings_db_path="/projects/worldcereal/data/cached_embeddings/embeddings_cache_LANDCOVER10_geo.duckdb",
        restrict_model_hash=None,
        label_domain="finetune_class",
        map_to_finetune=True,
        class_mappings_name="LANDCOVER10",
        h3_level=2,
        group_cols=["ref_id"],
        min_slice_size=100,
        merge_small_slice = True,
        threshold_mode="mad",
        percentile_q=0.96,
        mad_k=3.0,
        abs_threshold=None,
        fdr_alpha=0.05,
        min_flagged_per_slice=None,
        max_flagged_fraction=0.1,
        max_full_pairwise_n=0, # disable full pairwise matrix calculation
        norm_percentiles=(2.0, 98.0),
        output_samples_path=str(out_folder / "outliers_h3l2_mad_3_maxrank_groupRefId_ranked_sqrtk_norm2_98_new.parquet"),
        output_summary_path=str(out_folder / "outliers_h3l2_mad_3_maxrank_groupRefId_summary_sqrtk_norm2_98_new.parquet"),
        debug=False,
    )