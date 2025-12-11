"""Anomaly detection utilities operating purely on cached Presto embeddings.

This version assumes a DuckDB cache already exists with columns:
``sample_id, model_hash, ref_id, ewoc_code, h3_l3_cell, embedding_0..embedding_127``.
Embeddings are never recomputed; the pipeline always loads them from the cache.
Optional label domain switching between ``ewoc_code`` and mapped ``finetune_class``.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import duckdb
import h3
import numpy as np
import pandas as pd

from worldcereal.utils.refdata import get_class_mappings, map_classes


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


def merge_small_slices(
    df: pd.DataFrame,
    min_size: int = 100,
    label_col: str = "ewoc_code",
    h3_level_name: str = "h3_l3_cell",
    max_iterations: int = 25,
    min_improvement: float = 0.05,
    mark_undersized: bool = True,
) -> pd.DataFrame:
    """Merge small slices with neighbouring H3 cells until they exceed ``min_size``.

    Optimized default behavior:
    - Bulk, single-pass target selection per iteration using precomputed neighbour counts
    - Early exit when fractional improvement < ``min_improvement``
    - Optional ``undersized_slice`` flagging post-merge
    """

    df = df.copy()
    key_cols = ["ref_id", label_col, h3_level_name]

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
        # columns: ref_id, label_col, h3_l3_cell, target_cell
        merge_rows: List[Tuple] = []
        for (ref_id, label_value, cell), _ in small.items():
            neighbours = neighbour_map.get(cell, [])
            if not neighbours:
                continue
            # Find neighbour with maximum existing count for the same (ref_id, label)
            best_target = None
            best_count = 0
            for n in neighbours:
                c = int(counts.get((ref_id, label_value, n), 0))
                if c > best_count:
                    best_count = c
                    best_target = n
            if best_target is not None and best_count > 0:
                merge_rows.append((ref_id, label_value, cell, best_target))

        if not merge_rows:
            break

        merge_df = pd.DataFrame(
            merge_rows, columns=["ref_id", label_col, h3_level_name, "target_cell"]
        )

        # Apply merges in bulk via join on keys
        df = df.merge(merge_df, on=key_cols, how="left")
        # Where target_cell is set, update h3_l3_cell
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

    return df


def compute_slice_centroids(
    df: pd.DataFrame, label_col: str = "ewoc_code", h3_level_name: str = "h3_l3_cell"
) -> pd.DataFrame:
    """Compute centroids of embeddings per ``(ref_id, h3_l3_cell, label_col)``."""

    def _centroid(emb_list: Iterable[np.ndarray]) -> np.ndarray:
        arr = np.vstack(list(emb_list))
        return arr.mean(axis=0)

    centroids = (
        df.groupby(["ref_id", h3_level_name, label_col])["embedding"]
        .apply(_centroid)
        .reset_index()
        .rename(columns={"embedding": "centroid"})
    )
    return centroids


def compute_scores_for_slice(
    df_slice: pd.DataFrame, centroid: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Compute anomaly scores for a single slice×class dataframe."""

    embeddings = np.vstack(df_slice["embedding"].to_numpy())
    if centroid is None:
        centroid = embeddings.mean(axis=0)

    cos_dist = np.array([1 - _cosine_similarity(e, centroid) for e in embeddings])

    dist_matrix = _cosine_distance_matrix(embeddings)
    np.fill_diagonal(dist_matrix, np.inf)
    k = min(10, dist_matrix.shape[0] - 1) if dist_matrix.shape[0] > 1 else 0
    if k > 0:
        knn_dist = np.partition(dist_matrix, k, axis=1)[:, :k].mean(axis=1)
    else:
        knn_dist = np.zeros(len(df_slice))

    def _normalize(metric: np.ndarray) -> np.ndarray:
        p5, p95 = np.percentile(metric, [5, 95])
        denom = p95 - p5 if p95 > p5 else 1.0
        return np.clip((metric - p5) / denom, 0.0, 1.0)

    cos_norm = _normalize(cos_dist)
    knn_norm = _normalize(knn_dist)
    scores = 0.5 * (cos_norm + knn_norm)

    ranks = pd.Series(scores).rank(pct=True, method="average").to_numpy()

    df_scored = df_slice.copy()[[c for c in df_slice.columns if "embedding" not in c]]
    df_scored["cosine_distance"] = cos_dist
    df_scored["knn_distance"] = knn_dist
    df_scored["S"] = scores
    df_scored["rank_percentile"] = ranks

    return df_scored


def flag_anomalies(
    df_scores: pd.DataFrame,
    label_col: str = "ewoc_code",
    threshold_mode: str = "percentile",
    h3_level_name: str = "h3_l3_cell",
    percentile_q: float = 0.96,
    mad_k: float = 3.0,
    abs_threshold: Optional[float] = None,
    fdr_alpha: float = 0.05,
    min_flagged_per_slice: Optional[int] = None,
    max_flagged_fraction: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flag anomalies within each (ref_id, h3_l3_cell, label_col) group.

    Modes
    - percentile: flag S above group quantile `percentile_q` (default 0.96)
    - mad: flag S above median + `mad_k` * MAD (robust)
    - absolute: flag S above `abs_threshold` (requires value)
    - fdr: Benjamini–Hochberg on S-as-scores converted to p via rank; flags those with q<=`fdr_alpha`
    - min_flagged_per_slice: If not None and slice has fewer flagged samples than this, force the
        top-S samples to be flagged up to this minimum (or n if n is smaller).
    - max_flagged_fraction: If not None and slice has more flagged samples than this fraction of n,
        keep only the top-S flagged samples up to that cap.
    """

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
            # Convert ranks to p-values assuming higher S = more extreme
            n = len(g)
            ranks = g["S"].rank(ascending=False, method="average")
            pvals = ranks / (n + 1.0)
            # Benjamini–Hochberg
            order = np.argsort(pvals.to_numpy())
            p_sorted = pvals.to_numpy()[order]
            thresh = (np.arange(1, n + 1) / n) * fdr_alpha
            passed = p_sorted <= thresh
            # Largest index passing determines cutoff
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

        # Enforce per-slice min/max constraints on flagged count
        n = len(g)
        flags = g["flagged"].to_numpy().astype(bool)
        n_flag = int(flags.sum())

        # Apply max_flagged_fraction (cap)
        if max_flagged_fraction is not None:
            # convert fraction to max allowed count in [0, n]
            max_allowed = int(np.floor(max_flagged_fraction * n))
            if max_allowed < 0:
                max_allowed = 0
            if n_flag > max_allowed:
                if max_allowed == 0:
                    flags[:] = False
                else:
                    # g is sorted by S descending; keep top-S points
                    flags[:] = False
                    flags[:max_allowed] = True
                n_flag = int(flags.sum())

        # Apply min_flagged_per_slice (floor)
        if min_flagged_per_slice is not None and min_flagged_per_slice > 0:
            if n_flag < min_flagged_per_slice:
                k = min(min_flagged_per_slice, n)
                # Force top-k S to be flagged
                flags[:] = False
                flags[:k] = True
                n_flag = int(flags.sum())

        g["flagged"] = flags
        return g

    flagged_df = (
        df_scores.groupby(["ref_id", h3_level_name, label_col], group_keys=False)
        .apply(_flag_group)
        .reset_index(drop=True)
    )

    summary = (
        flagged_df.groupby(["ref_id", h3_level_name, label_col])
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
    min_slice_size: int = 100,
    threshold_mode: str = "percentile",
    percentile_q: float = 0.96,
    mad_k: float = 3.0,
    abs_threshold: Optional[float] = None,
    fdr_alpha: float = 0.05,
    min_flagged_per_slice: Optional[int] = None,
    max_flagged_fraction: Optional[float] = None,
    output_samples_path: Optional[str] = None,
    output_summary_path: Optional[str] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run anomaly detection using only cached embeddings.

    Parameters
    ----------
    embeddings_db_path : str
        Path to DuckDB file with table ``embeddings_cache``.
    restrict_model_hash : Optional[str]
        If provided, filter rows to this model hash only.
    label_domain : str
        'ewoc_code', 'finetune_class' or 'balancing_class' to choose grouping labels.
    map_to_finetune : bool
        Whether to map ewoc codes to finetune classes (adds column).
    class_mappings_name : str
        Mapping key passed to reference data mapping function.
    min_slice_size : int
        Minimum slice size after merging neighbouring H3 cells.
    output_samples_path / output_summary_path : Optional[str]
        Optional parquet outputs.
    """

    if label_domain not in {"ewoc_code", "finetune_class", "balancing_class"}:
        raise ValueError("label_domain must be 'ewoc_code' or 'finetune_class'")

    print("[anomaly] Connecting DuckDB and loading cached embeddings...")
    con = duckdb.connect(embeddings_db_path)
    cols_df = con.execute("PRAGMA table_info('embeddings_cache')").fetchdf()
    embed_cols = [c for c in cols_df.name.tolist() if c.startswith("embedding_")]
    base_cols = [
        "sample_id",
        "ref_id",
        "ewoc_code",
        "model_hash",
        "h3_l3_cell",
        "lat",
        "lon",
    ]
    query = f"SELECT {', '.join(base_cols + embed_cols)} FROM embeddings_cache"
    if restrict_model_hash:
        query += f" WHERE model_hash='{restrict_model_hash}'"
    df = con.execute(query).fetchdf()
    print(f"[anomaly] Loaded {len(df):,} rows from embeddings_cache")
    if df.empty:
        raise ValueError(
            "No rows loaded from embeddings_cache. Check model_hash or DB path."
        )
    # from h3_l3_cell
    h3_level_name = f"h3_l{h3_level}_cell"
    if h3_level != 3:
        df[h3_level_name] = df["h3_l3_cell"].apply(lambda h: h3.cell_to_parent(h, h3_level))

    # or from lat/lon
    # df[h3_level_name] = [
    #     h3.latlng_to_cell(lat, lon, h3_level)  # or geo_to_h3, depending on h3 version
    #     for lat, lon in zip(df["lat"], df["lon"])
    # ]
    # Cast ewoc_code to int if needed
    if df["ewoc_code"].dtype != np.int64:
        df["ewoc_code"] = df["ewoc_code"].astype(np.int64)

    if debug:
        print("Loading subset ...")
        df = df.head(5000)

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

    print(f"[anomaly] Merging small slices (min_size={min_slice_size})...")
    df = merge_small_slices(df, min_size=min_slice_size, label_col=label_col, h3_level_name=h3_level_name)
    print("[anomaly] Computing per-slice centroids...")
    centroids = compute_slice_centroids(df, label_col=label_col, h3_level_name=h3_level_name)

    print("[anomaly] Scoring slices...")
    # Attach centroid vectors to each row via a merge, then groupby-apply without explicit Python loop
    centroids_df = centroids.rename(columns={label_col: "__label__"})
    df_with_centroid = df.merge(
        centroids_df.rename(columns={"__label__": label_col}),
        on=["ref_id", h3_level_name, label_col],
        how="left",
    )

    def _score_group(g: pd.DataFrame) -> pd.DataFrame:
        # centroid column contains the vector for this group
        return compute_scores_for_slice(g, centroid=g["centroid"].iloc[0])

    scored_df = (
        df_with_centroid.groupby(["ref_id", h3_level_name, label_col], group_keys=False)
        .apply(_score_group)
        .reset_index(drop=True)
    )
    # Drop high-dimensional embedding columns to reduce memory footprint
    scored_df = scored_df.drop(columns=["embedding", "base_embedding"], errors="ignore")
    print(f"[anomaly] Flagging anomalies (mode={threshold_mode})...")
    flagged_df, summary_df = flag_anomalies(
        scored_df,
        label_col=label_col,
        h3_level_name=h3_level_name,
        threshold_mode=threshold_mode,
        percentile_q=percentile_q,
        mad_k=mad_k,
        abs_threshold=abs_threshold,
        fdr_alpha=fdr_alpha,
        min_flagged_per_slice=min_flagged_per_slice,
        max_flagged_fraction=max_flagged_fraction,
    )

    # Add geometry and convert to GeoDataFrame
    import geopandas as gpd
    from shapely.geometry import Point

    flagged_df["geometry"] = flagged_df.apply(
        lambda row: Point(row["lon"], row["lat"]), axis=1
    )
    flagged_gdf = gpd.GeoDataFrame(flagged_df, geometry="geometry", crs="EPSG:4326")

    if output_samples_path:
        print(f"[anomaly] Writing flagged samples -> {output_samples_path}")
        # Ensure embeddings are not written out
        flagged_gdf = flagged_gdf.drop(
            columns=["embedding", "base_embedding"], errors="ignore"
        )
        flagged_gdf.to_parquet(output_samples_path, index=False)
    if output_summary_path:
        print(f"[anomaly] Writing summary -> {output_summary_path}")
        summary_df.to_parquet(output_summary_path, index=False)

    return flagged_df, summary_df


if __name__ == "__main__":
    run_pipeline(
        embeddings_db_path="/projects/worldcereal/data/cached_embeddings/embeddings_cache_LANDCOVER10.duckdb",
        label_domain="finetune_class",
        map_to_finetune=True,
        threshold_mode="mad",
        percentile_q=0.96,
        output_samples_path="LANDCOVER10_4%_outliers.parquet",
        output_summary_path="LANDCOVER10_4%_summary.parquet",
        debug=False,
    )
