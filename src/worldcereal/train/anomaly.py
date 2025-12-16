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
from typing import Iterable, List, Optional, Tuple, Sequence

import duckdb
import h3
import numpy as np
import pandas as pd

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

    ranks = pd.Series(scores).rank(pct=True, method="max").to_numpy()

    df_scored = df_slice.copy()[[c for c in df_slice.columns if "embedding" not in c]]
    df_scored["cosine_distance"] = cos_dist
    df_scored["knn_distance"] = knn_dist
    df_scored["cos_norm"] = cos_norm
    df_scored["knn_norm"] = knn_norm
    df_scored["S"] = scores
    df_scored["rank_percentile"] = ranks
    
    df_scored["cos_rank"] = cos_rank
    df_scored["knn_rank"] = knn_rank
    df_scored["S_rank"] = s_rank
    # df_scored["rank_percentile_rank"] = rank_percentile_rank
    df_scored["cos_z"] = cos_z
    df_scored["knn_z"] = knn_z
    df_scored["S_z"] = s_z

    return df_scored

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

    def _flag_group(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        if g.empty:
            g["flagged"] = False
            return g

        g = g.sort_values("S", ascending=False)
        if threshold_mode == "percentile":
            thr = g["S"].quantile(percentile_q)
            g["flagged"] = g["S"] >= thr
        elif threshold_mode == "mad":
            med = g["S"].median()
            mad = (g["S"] - med).abs().median()
            thr = med + mad_k * (mad if mad > 0 else 1.0)
            g["flagged"] = g["S"] >= thr
        elif threshold_mode == "absolute":
            if abs_threshold is None:
                raise ValueError(
                    "abs_threshold must be set when threshold_mode='absolute'"
                )
            g["flagged"] = g["S"] >= float(abs_threshold)
        elif threshold_mode == "fdr":
            n = len(g)
            ranks = g["S"].rank(ascending=False, method="max")
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
    label_domain: str = "ewoc_code",
    map_to_finetune: bool = False,
    class_mappings_name: str = "LANDCOVER10",
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
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run anomaly detection using only cached embeddings.

    Grouping:
      slice = group_cols + [h3 cell at chosen level] + [label_col]

    norm_percentiles:
      Percentiles used for per-slice min-max normalization of cosine_distance and knn_distance.
      Default (5,95) preserves existing behavior.
    """

    group_cols = list(group_cols or [])

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
    ]
    select_cols = list(dict.fromkeys([*base_cols, *group_cols]))  # preserve order, unique

    query = f"SELECT {', '.join(select_cols + embed_cols)} FROM embeddings_cache"
    if restrict_model_hash:
        query += f" WHERE model_hash='{restrict_model_hash}'"

    df = con.execute(query).fetchdf()
    print(f"[anomaly] Loaded {len(df):,} rows from embeddings_cache")
    if df.empty:
        raise ValueError(
            "No rows loaded from embeddings_cache. Check model_hash or DB path."
        )

    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        raise ValueError(
            f"Requested group_cols not found in loaded data: {missing_group_cols}"
        )

    h3_level_name = f"h3_l{h3_level}_cell"
    if h3_level != 3:
        df[h3_level_name] = df["h3_l3_cell"].apply(lambda h: h3.cell_to_parent(h, h3_level))

    if df["ewoc_code"].dtype != np.int64:
        df["ewoc_code"] = df["ewoc_code"].astype(np.int64)

    if debug:
        print("Loading subset ...")
        # df = df.head(5000)
        # load only two h3 cells for faster testing
        sample_cells = df[h3_level_name].unique()[:25].tolist()
        df = df[df[h3_level_name].isin(sample_cells)]

    if map_to_finetune:
        print(f"[anomaly] Mapping classes using '{class_mappings_name}'...")
        df = map_classes(df, class_mappings_name)

    print("[anomaly] Preparing embeddings array...")
    embed_array = df[embed_cols].to_numpy(dtype=np.float32)
    df["embedding"] = [row for row in embed_array]

    label_col = label_domain
    if label_col not in df.columns:
        raise ValueError(
            f"Requested label column '{label_col}' not found after mapping"
        )

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

    print("[anomaly] Computing per-slice centroids...")
    centroids = compute_slice_centroids(
        df,
        label_col=label_col,
        h3_level_name=h3_level_name,
        group_cols=group_cols,
    )

    print("[anomaly] Scoring slices...")
    df_with_centroid = df.merge(
        centroids,
        on=slice_keys,
        how="left",
    )

    def _score_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < MIN_SCORING_SLICE_SIZE:
            g = g.copy()
            g = g[[c for c in g.columns if "embedding" not in c]]
            g["cosine_distance"] = 0.0
            g["knn_distance"] = 0.0
            g["cos_norm"] = 0.0
            g["knn_norm"] = 0.0
            g["S"] = 0.0
            g["rank_percentile"] = 0.0
            # g["rank_percentile_rank"] = 0.0
            g["cos_rank"] = 0.0
            g["knn_rank"] = 0.0
            g["S_rank"] = 0.0
            g["cos_z"] = 0.0
            g["knn_z"] = 0.0
            g["S_z"] = 0.0
            return g

        return compute_scores_for_slice(
            g,
            centroid=g["centroid"].iloc[0],
            norm_percentiles=norm_percentiles,
            max_full_pairwise_n=max_full_pairwise_n,  # <-- switch happens inside
            force_knn=False,
            knn_k=10,
        )
        
    # Can we have a progress bar here?
    from tqdm import tqdm
    tqdm.pandas()
    scored_df = (
        df_with_centroid.groupby(slice_keys, group_keys=False)
        .progress_apply(_score_group)
        .reset_index(drop=True)
    )

    scored_df = scored_df.drop(columns=["embedding", "base_embedding"], errors="ignore")

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

    import geopandas as gpd
    from shapely.geometry import Point
    
    flagged_df["S_rank_min"] = np.minimum(flagged_df["cos_rank"], flagged_df["knn_rank"])

    S_anomaly = 'S_anomaly'
    flagged_df[S_anomaly] = "normal"
    flagged_df.loc[flagged_df["flagged"] == True, S_anomaly] = "flagged"
    flagged_df.loc[
        (flagged_df["rank_percentile"] >= 0.98)
        & (flagged_df["S"] >= 0.95)
        & (flagged_df["flagged"] == True),
        S_anomaly,
    ] = "suspect"
    flagged_df.loc[
        (flagged_df["rank_percentile"] >= 0.99)
        & (flagged_df["S"] >= 0.99)
        & (flagged_df["flagged"] == True),
        S_anomaly,
    ] = "candidate"

    # Addiotnal anomaly categories based on combination of scores
    combined_anomaly = 'combined_anomaly'
    flagged_df[combined_anomaly] = "normal"
    flagged_df.loc[flagged_df["flagged"] == True, combined_anomaly] = "flagged"

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
    is_flagged = flagged_df["flagged"] == True

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

    if output_samples_path:
        print(f"[anomaly] Writing flagged samples -> {output_samples_path}")
        flagged_gdf = flagged_gdf.drop(
            columns=["embedding", "base_embedding"], errors="ignore"
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
    run_pipeline(
        embeddings_db_path="/projects/worldcereal/data/cached_embeddings/embeddings_cache_LANDCOVER10.duckdb",
        label_domain="finetune_class",
        map_to_finetune=True,
        threshold_mode="percentile",
        percentile_q=0.96,
        group_cols=["ref_id"],  # <- can be [] or e.g. ["ref_id","year","country"]
        output_samples_path="LANDCOVER10_4%_outliers.parquet",
        output_summary_path="LANDCOVER10_4%_summary.parquet",
        debug=False,
    )