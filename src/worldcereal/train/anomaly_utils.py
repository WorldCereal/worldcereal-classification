"""Anomaly detection utilities — pure computation helpers, scoring, and metrics.

Extracted from anomaly.py to improve maintainability.  Every public function
here is a stateless building block consumed by the orchestration layer in
``anomaly.py``.

Sections
--------
1. Constants
2. Math / distance helpers
3. Normalization & rank helpers
4. Label-domain & mapping helpers
5. Slice operations (merge, centroids)
6. Scoring (per-slice, hierarchical)
7. Context-aware metrics (centroid margins, kNN purity)
8. Confidence computation & fusion
9. Flagging / thresholding
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

MIN_SCORING_SLICE_SIZE: int = 50
"""Slices smaller than this get zero scores (not enough data to be meaningful)."""

_SCORE_COLS: List[str] = [
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

_EXCEL_SUFFIXES = {
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".ods",
}

# ---------------------------------------------------------------------------
# 2. Math / distance helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0 when either is zero-norm)."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Full NxN cosine-distance matrix (1 – cosine-similarity)."""
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sim = normed @ normed.T
    return 1.0 - sim


# ---------------------------------------------------------------------------
# 3. Normalization & rank helpers
# ---------------------------------------------------------------------------


def _normalize_percentile_minmax(
    metric: np.ndarray, norm_percentiles: Tuple[float, float] = (5.0, 95.0)
) -> np.ndarray:
    """Robust-ish min-max normalization using slice percentiles; output clipped to [0,1]."""
    lo, hi = norm_percentiles
    p_lo, p_hi = np.percentile(metric, [lo, hi])
    denom = p_hi - p_lo if p_hi > p_lo else 1.0
    return np.clip((metric - p_lo) / denom, 0.0, 1.0)


def _rank_pct(metric: np.ndarray) -> np.ndarray:
    """Rank-percentile in [0,1].  Higher metric => higher rank."""
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


# ---------------------------------------------------------------------------
# 4. Label-domain & mapping helpers
# ---------------------------------------------------------------------------


def _as_label_levels(label_domain: Union[str, Sequence[str]]) -> List[str]:
    """Normalize *label_domain* into an ordered list of label columns (fine -> coarse)."""
    if isinstance(label_domain, (list, tuple)):
        return [str(x) for x in label_domain]
    return [str(label_domain)]


def _require_label_columns(df: pd.DataFrame, label_cols: Sequence[str]) -> None:
    """Raise if any of the requested label columns are missing from *df*."""
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Requested label column(s) not found after mapping: {missing}")


def _load_mapping_df(
    mapping_file: str,
    *,
    label_cols: Sequence[str],
    class_mappings_name: str,
) -> pd.DataFrame:
    """Load a mapping file (Excel or JSON) into a DataFrame with columns:
    ``ewoc_code`` + *label_cols*.

    JSON formats supported
    ~~~~~~~~~~~~~~~~~~~~~~
    1) ``{"LANDCOVER10": {"110...": "temporary_crops", ...}, "CROPTYPE25": {...}}``
       — uses *class_mappings_name* to select the inner mapping.
    2) ``{"110...": "label", ...}``  — single label column.
    3) ``{"110...": {"lvl0": "...", "lvl1": "..."}, ...}``  — hierarchical.
    4) ``{"110...": ["lvl0", "lvl1", ...], ...}``  — hierarchical by position.
    5) ``[{"ewoc_code": "110...", "lvl0": "...", ...}, ...]``  — table.
    """
    p = Path(mapping_file)
    suf = p.suffix.lower()

    if suf in _EXCEL_SUFFIXES:
        return pd.read_excel(mapping_file)

    if suf != ".json":
        raise ValueError(
            f"Unsupported mapping_file type '{suf}'. "
            f"Use an Excel file ({sorted(_EXCEL_SUFFIXES)}) or a .json file."
        )

    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Table-like JSON
    if isinstance(data, list):
        return pd.DataFrame(data)

    if not isinstance(data, dict):
        raise ValueError("mapping_file JSON must be a dict or a list of records")

    # Multi-mapping JSON (e.g. class_mappings.json) — select the named mapping
    if class_mappings_name in data and isinstance(data[class_mappings_name], dict):
        data = data[class_mappings_name]

    rows: list = []
    for ewoc_code, v in data.items():
        row: dict = {"ewoc_code": ewoc_code}

        if isinstance(v, (str, int, float)) or v is None:
            if len(label_cols) != 1:
                raise ValueError(
                    "mapping_file JSON maps ewoc_code to a single value, but label_domain "
                    f"requests multiple label columns.  Expected columns: {list(label_cols)}"
                )
            row[label_cols[0]] = v

        elif isinstance(v, dict):
            for lc in label_cols:
                if lc in v:
                    row[lc] = v[lc]

        elif isinstance(v, (list, tuple)):
            if len(v) < len(label_cols):
                raise ValueError(
                    f"mapping_file JSON list for ewoc_code={ewoc_code} has {len(v)} values "
                    f"but {len(label_cols)} label columns were requested: {list(label_cols)}"
                )
            for lc, vv in zip(label_cols, v):
                row[lc] = vv

        else:
            raise ValueError(
                f"Unsupported JSON mapping value type for ewoc_code={ewoc_code}: {type(v)}"
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Slice operations (merge, centroids, adaptive H3)
# ---------------------------------------------------------------------------


def assign_adaptive_h3_level(
    df: pd.DataFrame,
    h3_levels: Sequence[int],
    label_col: str = "ewoc_code",
    group_cols: Optional[Sequence[str]] = None,
    min_slice_size: int = 100,
    max_slice_size: Optional[int] = None,
) -> pd.DataFrame:
    """Assign each point an effective H3 cell based on point density.

    Iterates from **coarsest** to **finest** H3 resolution.  For each
    ``(group_cols, label, h3_cell)`` slice at the current level:

    - If the slice has **≤ max_slice_size** points (or *max_slice_size* is
      None) → resolve those points at this level, regardless of whether the
      slice is small or large.  Small slices are handled later by
      ``merge_small_slices``.
    - If the slice **exceeds max_slice_size** → leave those points unresolved
      and push them to the next finer level where the geographic cell is
      smaller and the slice will naturally shrink.
    - At the finest level all remaining unresolved points are resolved
      unconditionally (every point must end up somewhere).

    After the loop, any still-unresolved points are assigned the finest
    requested level unconditionally.

    Example with h3_levels=[1, 2, 3], min_slice_size=100, max_slice_size=4000
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Dense Europe cell at L1 with 12 000 points in a slice → too big,
      pushed to L2.
    - At L2 the cell splits; each sub-cell has ~1 500 points → resolved ✓
    - Sparse Africa cell at L1 with 200 points → within bounds, resolved ✓
    - Very sparse cell: only 40 points even at L3 → resolved at L3
      unconditionally (handled later by merge_small_slices as undersized).

    New columns added
    ~~~~~~~~~~~~~~~~~
    - ``effective_h3_cell`` : the H3 cell index used for this point
    - ``h3_effective_level``: the H3 resolution that was selected (int)

    Parameters
    ----------
    h3_levels
        H3 resolutions to try, e.g. ``[1, 2, 3]`` or ``[3, 2, 1]``.
        Internally always sorted **coarsest → finest** (ascending by H3
        resolution number, i.e. smallest number first).
    min_slice_size
        Minimum number of points required to resolve a slice at a given level.
        Slices below this are pushed to a finer level.
    max_slice_size
        Maximum number of points allowed in a slice at a given level.  Slices
        exceeding this are pushed to a finer level.  At the finest level this
        cap is not applied — every remaining point is resolved unconditionally.
        If None, no upper cap is applied and all slices >= min_slice_size are
        resolved at the coarsest level where they first meet the minimum.
    """
    import h3 as _h3

    # Ensure coarsest → finest ordering (lowest H3 number = coarsest)
    h3_levels = sorted(h3_levels, reverse=False)

    df = df.copy()
    group_cols = list(group_cols or [])

    # Make sure h3_l3_cell exists (source column)
    if "h3_l3_cell" not in df.columns:
        raise ValueError("DataFrame must contain an 'h3_l3_cell' column")

    # Pre-compute H3 cells at every requested level from h3_l3_cell.
    # - Levels < 3 (coarser): use cell_to_parent — h3_l3_cell is the child.
    # - Level == 3: direct copy.
    # - Levels > 3 (finer): use cell_to_children — h3_l3_cell is the parent,
    #   so each L3 row maps to ONE of its children that covers the original
    #   point.  We use h3.cell_to_center_child which gives the single child
    #   at the target resolution whose centre is closest to the L3 cell centre
    #   (deterministic, no explosion of rows).
    h3_col_map: dict[int, str] = {}
    for lvl in h3_levels:
        col = f"_h3_l{lvl}_cell"
        if lvl == 3:
            df[col] = df["h3_l3_cell"]
        elif lvl < 3:
            df[col] = df["h3_l3_cell"].apply(
                lambda h, _lvl=lvl: _h3.cell_to_parent(h, _lvl)
            )
        else:  # lvl > 3 — derive finer cell from lat/lon
            if "lat" not in df.columns or "lon" not in df.columns:
                raise ValueError(
                    f"H3 level {lvl} is finer than the cached L3 cells. "
                    "DataFrame must contain 'lat' and 'lon' columns to derive "
                    "finer H3 cells via lat/lon coordinates."
                )
            df[col] = df.apply(
                lambda row, _lvl=lvl: _h3.latlng_to_cell(row["lat"], row["lon"], _lvl),
                axis=1,
            )
        h3_col_map[lvl] = col

    # Track which rows are resolved
    resolved = np.zeros(len(df), dtype=bool)
    effective_cell = np.empty(len(df), dtype=object)
    effective_level = np.full(len(df), -1, dtype=np.int8)

    finest_level = h3_levels[-1]  # last in coarsest→finest order

    for lvl in h3_levels:
        if resolved.all():
            break

        h3_col = h3_col_map[lvl]
        unresolved_idx = np.where(~resolved)[0]
        if len(unresolved_idx) == 0:
            break

        is_finest = (lvl == finest_level)

        # Build slice keys for unresolved rows at this level
        sub = df.iloc[unresolved_idx]
        key_cols = [*group_cols, label_col, h3_col]
        counts = sub.groupby(key_cols).size()

        if is_finest:
            # At the finest level: resolve ALL remaining points unconditionally.
            # No max_slice_size cap — every point must be assigned somewhere.
            resolve_keys = set(counts.index.tolist())
        else:
            # Only push slices to a finer level if they are TOO BIG
            # (> max_slice_size). These will naturally split into smaller
            # sub-cells at a finer resolution.
            #
            # Slices that are too SMALL (< min_slice_size) are resolved HERE
            # at the current (coarser) level — going finer would only make
            # them smaller still. merge_small_slices will absorb them into
            # neighbouring cells afterwards.
            #
            # So: resolve everything EXCEPT slices that exceed max_slice_size.
            if max_slice_size is not None:
                resolve_keys = set(
                    counts[counts <= max_slice_size].index.tolist()
                )
            else:
                resolve_keys = set(counts.index.tolist())

        if not resolve_keys:
            if not is_finest:
                n_oversized = len(counts) - len(resolve_keys)
                print(
                    f"[adaptive_h3]   L{lvl}: 0 slices resolved, "
                    f"{n_oversized} slices too big → pushing to next level"
                )
            continue

        # Mark matching unresolved rows as resolved at this level
        sub_indexed = sub.set_index(key_cols)
        match_mask = sub_indexed.index.isin(resolve_keys)
        matched_positions = unresolved_idx[match_mask]

        resolved[matched_positions] = True
        effective_cell[matched_positions] = df.iloc[matched_positions][h3_col].to_numpy()
        effective_level[matched_positions] = np.int8(lvl)

        if not is_finest:
            n_resolved = len(resolve_keys)
            n_oversized = len(counts) - n_resolved
            n_pts_resolved = int(match_mask.sum())
            n_pts_oversized = int(len(unresolved_idx) - n_pts_resolved)
            print(
                f"[adaptive_h3]   L{lvl}: {n_resolved} slices resolved "
                f"({n_pts_resolved:,} pts), "
                f"{n_oversized} slices too big ({n_pts_oversized:,} pts) → next level"
            )

    # Safety: assign any still-unresolved points to the finest level
    # (should only happen if h3_levels has a single entry)
    still_unresolved = ~resolved
    if still_unresolved.any():
        finest_col = h3_col_map[finest_level]
        effective_cell[still_unresolved] = (
            df.loc[still_unresolved, finest_col].to_numpy()
        )
        effective_level[still_unresolved] = np.int8(finest_level)

    df["effective_h3_cell"] = effective_cell
    df["h3_effective_level"] = effective_level

    # Clean up temporary columns
    for col in h3_col_map.values():
        df = df.drop(columns=[col], errors="ignore")

    # Summary stats (printed coarsest → finest)
    for lvl in h3_levels:
        n_at_lvl = int((effective_level == lvl).sum())
        print(f"[adaptive_h3] Level {lvl}: {n_at_lvl:,} points")

    return df


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
    """Merge small slices with neighbouring H3 cells until they exceed *min_size*.

    A "slice" is defined by: ``group_cols + [label_col] + [h3_level_name]``.
    """
    import h3 as _h3

    df = df.copy()
    group_cols = list(group_cols or [])
    key_cols = [*group_cols, label_col, h3_level_name]

    # Pre-compute H3 neighbours for all cells present
    unique_cells = df[h3_level_name].unique().tolist()
    neighbour_map = {
        cell: list(set(_h3.grid_disk(cell, 1)) - {cell}) for cell in unique_cells
    }

    # Iterative bulk merge
    counts = df.groupby(key_cols).size()
    for _ in range(max_iterations):
        small = counts[counts < min_size]
        if small.empty:
            break

        before_total_small = int(small.sum())

        # Build candidate merges: each small key with its best neighbour
        merge_rows: List[Tuple] = []
        for key, _ in small.items():
            if not isinstance(key, tuple):
                key = (key,)
            group_vals = key[:-2]
            label_value = key[-2]
            cell = key[-1]

            neighbours = neighbour_map.get(cell, [])
            if not neighbours:
                continue

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


# ---------------------------------------------------------------------------
# 6. Scoring (per-slice, hierarchical)
# ---------------------------------------------------------------------------


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
      - kNN-only computation (NearestNeighbors) if N is large or *force_knn=True*

    Returns columns:
      cosine_distance, knn_distance, cos_norm, knn_norm, S, rank_percentile,
      cos_rank, knn_rank, S_rank, S_rank_min, cos_z, knn_z, S_z, mean_score
    """
    embeddings = np.vstack(df_slice["embedding"].to_numpy()).astype("float32", copy=False)
    n = embeddings.shape[0]

    if centroid is None:
        centroid = embeddings.mean(axis=0)

    # Cosine distance to centroid
    cos_dist = np.array(
        [1.0 - _cosine_similarity(e, centroid) for e in embeddings], dtype=np.float32
    )

    # Choose kNN strategy: sqrt(N) capped at 50, but at least knn_k
    # k = min(int(knn_k), n - 1) if n > 1 else 0
    SQRT_N = int(np.sqrt(n))
    k = min(max(int(knn_k), min(int(SQRT_N), 50)), n - 1) if n > 1 else 0

    use_knn_only = force_knn or (max_full_pairwise_n is not None and n > max_full_pairwise_n)

    if k <= 0:
        knn_dist = np.zeros(n, dtype=np.float32)
    elif use_knn_only:
        # kNN-only path (memory friendly for large slices)
        nn = NearestNeighbors(
            n_neighbors=k + 1,  # include self, then drop it
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
        knn_dist = (
            np.partition(dist_matrix, k, axis=1)[:, :k]
            .mean(axis=1)
            .astype(np.float32, copy=False)
        )

    knn_dist = np.nan_to_num(knn_dist, nan=0.0, posinf=0.0, neginf=0.0)

    # Percentile-based normalization
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

    # Build output — drop embedding columns to keep the result lean
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

    # Confidence score: average of the three score variants
    df_scored["mean_score"] = (
        (df_scored["S_rank"] + df_scored["S_rank_min"] + df_scored["S_z"]) / 3.0
    ).astype(np.float32)

    return df_scored


def _score_group_simple(
    g: pd.DataFrame,
    norm_percentiles: Tuple[float, float],
    max_full_pairwise_n: Optional[int],
) -> pd.DataFrame:
    """Score a single group, returning zero scores when the slice is too small."""
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
    """Decide, per point, which label level is used for scoring.

    - Level 0 if level-0 slice size >= *min_slice_size*
    - Else first higher level with group size >= *min_slice_size*
    - Else coarsest level

    Also computes ``slice_n``, ``ref_group_n``.
    """
    df = df.copy()

    if not label_cols:
        raise ValueError("label_cols must be non-empty")

    # Level-0 slice size
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
    n_cols: dict = {}
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
    """Score points by level-0 slices, falling back to coarser label levels
    for undersized slices.

    Scores are written back ONLY for the original undersized-slice points.
    """
    from tqdm import tqdm

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

    tqdm.pandas()

    # 1) Score rows that use level 0 directly (normal path)
    direct = df_idx[df_idx[ref_level_col] == 0]
    if not direct.empty:
        g0 = direct[
            [*group_cols, h3_level_name, label_cols[0], "sample_id", "embedding"]
        ].reset_index(drop=True)

        scored0 = (
            g0.groupby([*group_cols, h3_level_name, label_cols[0]], group_keys=False)
            .progress_apply(
                lambda g: _score_group_simple(g, norm_percentiles, max_full_pairwise_n)
            )
            .reset_index(drop=True)
        )
        scored0 = scored0.set_index("sample_id", drop=False)
        df_idx.loc[scored0.index, _SCORE_COLS] = scored0[_SCORE_COLS].to_numpy()

    # 2) Score fallback groups once, then write back only to target rows
    fallback = df_idx[df_idx[ref_level_col] > 0]
    if not fallback.empty:
        fb_keys = [ref_level_col, *group_cols, h3_level_name, ref_class_col]
        target_map = fallback.groupby(fb_keys)["sample_id"].apply(list)

        for key, target_ids in tqdm(
            target_map.items(), total=len(target_map), desc="Scoring fallback ref groups"
        ):
            ref_level = int(key[0])
            ref_class = key[-1]
            ref_label_col = label_cols[ref_level]

            # Build reference set mask on the FULL dataframe
            m = df_idx[ref_label_col].astype(object).to_numpy() == ref_class

            offset = 1
            for i, gc in enumerate(group_cols):
                m &= df_idx[gc].astype(object).to_numpy() == key[offset + i]

            h3_val = key[offset + len(group_cols)]
            m &= df_idx[h3_level_name].astype(object).to_numpy() == h3_val

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
        bad = df_idx[df_idx[_SCORE_COLS].isna().any(axis=1)][
            ["sample_id", ref_level_col, ref_class_col]
        ].head(20)
        raise ValueError(
            f"Hierarchical scoring left NaNs in score columns. Example rows:\n{bad}"
        )

    return df_idx.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Context-aware metrics (centroid margins, kNN purity)
# ---------------------------------------------------------------------------


def add_alt_class_centroid_metrics(
    df: pd.DataFrame,
    *,
    label_col: str,
    context_cols: Sequence[str],
    embedding_col: str = "embedding",
) -> pd.DataFrame:
    """For each context group (*context_cols*), compute per-label centroids and
    for each point:

    - ``self_centroid_dist_ctx`` : cosine dist to centroid of its own label
    - ``alt_label_ctx``         : closest other label centroid
    - ``alt_centroid_dist_ctx`` : cosine dist to closest other label centroid
    - ``alt_margin_ctx``        : alt – self  (≤0 suggests confusion)
    - ``context_n_labels``      : number of labels present in context
    """
    df = df.copy()

    out_alt_label = np.full(len(df), None, dtype=object)
    out_self = np.full(len(df), np.nan, dtype=np.float32)
    out_alt = np.full(len(df), np.nan, dtype=np.float32)
    out_margin = np.full(len(df), np.nan, dtype=np.float32)
    out_nlab = np.zeros(len(df), dtype=np.uint16)

    # Positional index for writing back into flat arrays
    pos = np.arange(len(df), dtype=np.int64)
    df = df.copy()
    df["_pos"] = pos

    # Pre-extract embeddings for fast vstack inside groups
    emb_series = df[embedding_col].to_numpy()

    for _, g in df.groupby(list(context_cols), dropna=False, sort=False):
        idx = g["_pos"].to_numpy()
        labels = g[label_col].to_numpy()

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

        C = np.vstack(centroids).astype(np.float32, copy=False)  # (L, D)
        sims = Xn @ C.T                                          # (N, L)
        dists = 1.0 - sims                                       # cosine distances

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
    """Compute kNN label-purity within each context group, but only for
    rows where ``flagged == True``.

    Uses *df_all* for embeddings and full neighbourhood; writes results back
    into *flagged_df*.
    """
    # Flagged subset keys — limit work to contexts that contain flagged points
    flagged_only = flagged_df.loc[
        flagged_df["flagged"] == True,  # noqa: E712
        ["sample_id", *context_cols, label_col],
    ]
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

    # Prepare outputs keyed by sample_id
    out_same: dict = {}
    out_maj_lab: dict = {}
    out_maj_frac: dict = {}

    flagged_ids = set(flagged_only["sample_id"].tolist())

    for _, g in df_sub.groupby(list(context_cols), dropna=False, sort=False):
        n = len(g)
        if n < 2:
            continue

        # k similar to scoring logic (sqrt(n) capped, but at least purity_knn_k)
        k = min(max(int(purity_knn_k), min(int(np.sqrt(n)), int(cap_sqrt_k))), n - 1)
        if k <= 0:
            continue

        sids = g["sample_id"].to_numpy()
        labels = g[label_col].to_numpy()

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

            vals, counts = np.unique(nl, return_counts=True)
            j = int(np.argmax(counts))
            maj_lab = vals[j]
            maj_frac = float(counts[j] / len(nl))

            out_same[sid] = same_frac
            out_maj_lab[sid] = maj_lab
            out_maj_frac[sid] = maj_frac

    # Write back
    flagged_df = flagged_df.copy()
    flagged_df["knn_same_label_frac_ctx"] = (
        flagged_df["sample_id"].map(out_same).astype("float32")
    )
    flagged_df["knn_majority_label_ctx"] = flagged_df["sample_id"].map(out_maj_lab)
    flagged_df["knn_majority_frac_ctx"] = (
        flagged_df["sample_id"].map(out_maj_frac).astype("float32")
    )
    return flagged_df


# ---------------------------------------------------------------------------
# 8. Confidence computation & fusion
# ---------------------------------------------------------------------------


def add_confidence_from_score(
    df: pd.DataFrame,
    score_col: str = "mean_score",
    out_col: str = "confidence",
    t: float = 0.975,        # knee: confidence starts dropping after this
    alpha: float = 0.3,      # tail sharpness (bigger => harsher near 1)
    conf_min: float = 0.01,  # never go below this
    eps: float = 1e-9,       # numerical stability near 1
) -> pd.DataFrame:
    """Accelerating confidence drop as score → 1, with hard floor *conf_min*.

    .. math::

        y = \\text{clip}((x - t) / (1 - t), 0, 1)

        \\text{conf\\_raw} = \\exp(-\\alpha \\cdot y / (1 - y + \\varepsilon))

        \\text{confidence} = \\text{conf\\_min} + (1 - \\text{conf\\_min}) \\cdot \\text{conf\\_raw}

    - ``x <= t``  ⇒  confidence = 1
    - ``x → 1``   ⇒  confidence → conf_min (not 0)
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


def add_flagged_robust_confidence(
    df: pd.DataFrame,
    score_col: str = "mean_score",
    flagged_col: str = "flagged",
    out_z_col: str = "z_mad",
    out_conf_col: str = "confidence",
    # mapping params
    z_knee: float = 3.0,
    eps_conf: float = 1e-3,
    z_extreme: float = 10.0,
    clip_exp: float = 50.0,
    default_unflagged_conf: float = 1.0,
) -> pd.DataFrame:
    """For each slice (caller passes one slice at a time), compute robust
    MAD-z from *score_col* and assign confidence:

    - if not flagged → ``default_unflagged_conf``
    - if flagged → ``1 / (1 + exp(k*(z - z_knee)))``

    Also writes ``z_mad`` for debugging/auditing.
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


def apply_confidence_fusion(
    df: pd.DataFrame,
    base_conf_col: str = "confidence",
    out_conf_col: str = "confidence_alt",
    # margin inputs
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
    """Fuse auxiliary ambiguity signals into base confidence::

        confidence_alt = confidence × p_margin × p_purity

    ``p_margin``
        Logistic of ``(alt_margin - m0)``; larger margin ⇒ clearer separation.

    ``p_purity``
        ``(knn_same_label_frac_ctx) ** beta``; lower purity ⇒ stronger penalty.

    - If margin / purity is NaN, factor defaults to 1.0 (no penalty).
    - Output is float32 in [0, 1].
    """
    if base_conf_col not in df.columns:
        raise KeyError(f"Missing base confidence column: '{base_conf_col}'")

    conf0 = pd.to_numeric(df[base_conf_col], errors="coerce").astype("float64").to_numpy()
    conf0 = np.clip(conf0, 0.0, 1.0)

    # --- Margin ----------------------------------------------------------
    if margin_col in df.columns:
        margin = pd.to_numeric(df[margin_col], errors="coerce").astype("float64").to_numpy()
    elif (self_dist_col in df.columns) and (alt_dist_col in df.columns):
        self_d = pd.to_numeric(df[self_dist_col], errors="coerce").astype("float64").to_numpy()
        alt_d = pd.to_numeric(df[alt_dist_col], errors="coerce").astype("float64").to_numpy()
        margin = alt_d - self_d
    else:
        margin = np.full(len(df), np.nan, dtype="float64")

    z = margin_a * (margin - margin_m0)
    z = np.clip(z, -clip_exp, clip_exp)
    p_margin = 1.0 / (1.0 + np.exp(-z))
    p_margin = np.where(np.isfinite(p_margin), p_margin, default_factor_if_nan)

    # --- Purity ----------------------------------------------------------
    if purity_col in df.columns:
        pur = pd.to_numeric(df[purity_col], errors="coerce").astype("float64").to_numpy()
        pur = np.clip(pur, 0.0, 1.0)
        p_purity = np.power(pur, purity_beta)
        p_purity = np.where(np.isfinite(p_purity), p_purity, default_factor_if_nan)
    else:
        p_purity = np.full(len(df), default_factor_if_nan, dtype="float64")

    # For unflagged rows, p_margin and p_purity remain 1.0
    if "flagged" in df.columns:
        flagged_mask = df["flagged"].fillna(False).to_numpy(dtype=bool)
        unflagged = ~flagged_mask
        p_margin = np.where(unflagged, 1.0, p_margin)
        p_purity = np.where(unflagged, 1.0, p_purity)

    # Cap minimum values to avoid too much confidence reduction
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


# ---------------------------------------------------------------------------
# 9. Flagging / thresholding
# ---------------------------------------------------------------------------


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

    Slice keys: ``group_cols + [h3_level_name] + [label_col]``.
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
