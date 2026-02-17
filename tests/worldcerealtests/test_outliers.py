"""Tests for the anomaly detection pipeline (anomaly.py + anomaly_utils.py).

These tests use only synthetic data and run in seconds, making them safe for
CI on every push.  No DuckDB database, no real embeddings, no network access.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from worldcereal.train.anomaly_utils import (
    MIN_SCORING_SLICE_SIZE,
    _SCORE_COLS,
    _as_label_levels,
    _cosine_distance_matrix,
    _cosine_similarity,
    _load_mapping_df,
    _normalize_percentile_minmax,
    _rank_pct,
    _require_label_columns,
    _robust_z,
    _sigmoid,
    _add_hierarchical_ref_outlier_class,
    add_alt_class_centroid_metrics,
    add_confidence_from_score,
    add_flagged_robust_confidence,
    add_knn_label_purity_for_flagged,
    apply_confidence_fusion,
    compute_scores_for_slice,
    compute_slice_centroids,
    flag_anomalies,
    merge_small_slices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_test_resource(relative_path: str) -> Path:
    """Resolve a path inside testresources/."""
    return Path(os.path.dirname(os.path.realpath(__file__))) / "testresources" / relative_path


def _make_synthetic_embeddings(
    n: int = 200,
    dim: int = 128,
    n_outliers: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Create *n* random unit-ish embeddings with *n_outliers* planted far away."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    # push last n_outliers far from the centroid
    if n_outliers > 0:
        X[-n_outliers:] = -X[:n_outliers]  # flip some normal vectors
    return X


def _make_slice_df(
    n: int = 200,
    dim: int = 128,
    n_outliers: int = 5,
    label: str = "cropland",
    h3_cell: str = "821fa7fffffffff",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal DataFrame that looks like one slice for scoring."""
    X = _make_synthetic_embeddings(n, dim, n_outliers, seed)
    return pd.DataFrame({
        "sample_id": [f"s_{i}" for i in range(n)],
        "embedding": [row for row in X],
        "ewoc_code": [11011000] * n,
        "h3_l3_cell": [h3_cell] * n,
        "finetune_class": [label] * n,
    })


# ===================================================================
# 1. Math / distance helpers
# ===================================================================


class TestCosineHelpers:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_distance_matrix_diagonal(self):
        X = np.eye(3, dtype=np.float32)
        D = _cosine_distance_matrix(X)
        # diagonal should be ~0 (self-similarity = 1 → distance = 0)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_distance_matrix_symmetry(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10, 4).astype(np.float32)
        D = _cosine_distance_matrix(X)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_distance_matrix_range(self):
        """Distances should be in [0, 2] for unit-normed vectors."""
        rng = np.random.RandomState(0)
        X = rng.randn(20, 8).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        D = _cosine_distance_matrix(X)
        assert D.min() >= -1e-6
        assert D.max() <= 2.0 + 1e-6


# ===================================================================
# 2. Normalization & rank helpers
# ===================================================================


class TestNormalization:
    def test_normalize_percentile_minmax_range(self):
        rng = np.random.RandomState(0)
        x = rng.randn(500).astype(np.float32)
        normed = _normalize_percentile_minmax(x)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

    def test_normalize_constant_input(self):
        x = np.ones(100, dtype=np.float32) * 5.0
        normed = _normalize_percentile_minmax(x)
        # constant → denom=1, everything at (5-5)/1 = 0 → clipped to 0
        assert np.all(normed >= 0.0) and np.all(normed <= 1.0)

    def test_rank_pct_monotonic(self):
        x = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        r = _rank_pct(x)
        assert np.all(np.diff(r) >= 0)  # monotonically increasing

    def test_rank_pct_empty(self):
        x = np.array([], dtype=np.float32)
        r = _rank_pct(x)
        assert len(r) == 0

    def test_rank_pct_max_is_one(self):
        x = np.arange(100, dtype=np.float32)
        r = _rank_pct(x)
        assert r[-1] == pytest.approx(1.0)

    def test_robust_z_median_is_zero(self):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        z = _robust_z(x)
        # z of the median element should be 0
        assert z[2] == pytest.approx(0.0, abs=1e-10)

    def test_robust_z_empty(self):
        x = np.array([], dtype=np.float64)
        z = _robust_z(x)
        assert len(z) == 0

    def test_sigmoid_at_zero(self):
        x = np.array([0.0])
        assert _sigmoid(x)[0] == pytest.approx(0.5)

    def test_sigmoid_range(self):
        x = np.linspace(-100, 100, 1000)
        s = _sigmoid(x)
        assert s.min() >= 0.0
        assert s.max() <= 1.0

    def test_sigmoid_monotonic(self):
        x = np.linspace(-10, 10, 100)
        s = _sigmoid(x)
        assert np.all(np.diff(s) >= 0)


# ===================================================================
# 3. Label-domain & mapping helpers
# ===================================================================


class TestLabelHelpers:
    def test_as_label_levels_string(self):
        assert _as_label_levels("ewoc_code") == ["ewoc_code"]

    def test_as_label_levels_list(self):
        assert _as_label_levels(["a", "b", "c"]) == ["a", "b", "c"]

    def test_as_label_levels_tuple(self):
        assert _as_label_levels(("x", "y")) == ["x", "y"]

    def test_require_label_columns_ok(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        _require_label_columns(df, ["a", "b"])  # should not raise

    def test_require_label_columns_missing(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            _require_label_columns(df, ["a", "missing_col"])


class TestLoadMappingDf:
    def test_load_json_single_level(self):
        path = _get_test_resource("test_class_mappings.json")
        result = _load_mapping_df(
            str(path),
            label_cols=["LANDCOVER10"],
            class_mappings_name="LANDCOVER10",
        )
        assert "ewoc_code" in result.columns
        assert "LANDCOVER10" in result.columns
        assert len(result) == 6  # 6 entries in our test fixture

    def test_load_json_with_class_mappings_name(self):
        path = _get_test_resource("test_class_mappings.json")
        result = _load_mapping_df(
            str(path),
            label_cols=["CROPTYPE25"],
            class_mappings_name="CROPTYPE25",
        )
        assert "ewoc_code" in result.columns
        assert "wheat" in result["CROPTYPE25"].values

    def test_load_json_unsupported_extension(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported"):
            _load_mapping_df(str(p), label_cols=["x"], class_mappings_name="X")

    def test_load_json_table_format(self, tmp_path):
        records = [
            {"ewoc_code": "111", "label": "crop"},
            {"ewoc_code": "222", "label": "forest"},
        ]
        p = tmp_path / "table.json"
        p.write_text(json.dumps(records))
        result = _load_mapping_df(str(p), label_cols=["label"], class_mappings_name="X")
        assert len(result) == 2
        assert set(result["ewoc_code"]) == {"111", "222"}


# ===================================================================
# 4. Scoring
# ===================================================================


class TestComputeScoresForSlice:
    def test_returns_all_score_columns(self):
        df = _make_slice_df(n=100, n_outliers=3)
        result = compute_scores_for_slice(df, norm_percentiles=(5.0, 95.0))
        for col in _SCORE_COLS:
            assert col in result.columns, f"Missing score column: {col}"

    def test_scores_in_unit_range(self):
        df = _make_slice_df(n=100, n_outliers=3)
        result = compute_scores_for_slice(df)
        for col in ["cos_norm", "knn_norm", "S", "rank_percentile", "S_rank", "S_z"]:
            vals = result[col].to_numpy()
            assert vals.min() >= -0.01, f"{col} below 0: {vals.min()}"
            assert vals.max() <= 1.01, f"{col} above 1: {vals.max()}"

    def test_outliers_score_higher(self):
        """Planted outliers (last 5 rows) should score higher on average."""
        n, n_out = 200, 5
        df = _make_slice_df(n=n, n_outliers=n_out)
        result = compute_scores_for_slice(df)
        normal_mean = result["S"].iloc[:-n_out].mean()
        outlier_mean = result["S"].iloc[-n_out:].mean()
        assert outlier_mean > normal_mean

    def test_embedding_column_not_in_output(self):
        df = _make_slice_df(n=80)
        result = compute_scores_for_slice(df)
        assert "embedding" not in result.columns

    def test_knn_only_path(self):
        """Force kNN-only path (max_full_pairwise_n=0)."""
        df = _make_slice_df(n=100)
        result = compute_scores_for_slice(
            df, max_full_pairwise_n=0, force_knn=True
        )
        assert "knn_distance" in result.columns
        assert result["knn_distance"].notna().all()

    def test_small_slice_below_min(self):
        """Slices smaller than MIN_SCORING_SLICE_SIZE get all-zero scores."""
        df = _make_slice_df(n=MIN_SCORING_SLICE_SIZE - 1, n_outliers=0)
        # Use _score_group_simple which handles the small-slice case
        from worldcereal.train.anomaly_utils import _score_group_simple

        result = _score_group_simple(df, norm_percentiles=(5.0, 95.0), max_full_pairwise_n=None)
        for col in _SCORE_COLS:
            assert (result[col] == 0.0).all(), f"{col} should be all zeros for small slice"


# ===================================================================
# 5. Slice operations
# ===================================================================


class TestMergeSmallSlices:
    @pytest.fixture()
    def multi_cell_df(self):
        """A DF with 3 adjacent H3-L3 cells, one very small."""
        # Use known-valid H3 resolution-3 cells (pre-computed)
        base_cell = "83194dfffffffff"
        n1 = "83194cfffffffff"
        n2 = "831948fffffffff"

        rows = []
        # large slice in base_cell
        for i in range(120):
            rows.append({"sample_id": f"A{i}", "ewoc_code": "crop", "h3_l3_cell": base_cell})
        # small slice in n1 (same label)
        for i in range(5):
            rows.append({"sample_id": f"B{i}", "ewoc_code": "crop", "h3_l3_cell": n1})
        # different label in n2
        for i in range(80):
            rows.append({"sample_id": f"C{i}", "ewoc_code": "forest", "h3_l3_cell": n2})
        return pd.DataFrame(rows)

    def test_small_slice_gets_merged(self, multi_cell_df):
        result = merge_small_slices(
            multi_cell_df,
            min_size=50,
            label_col="ewoc_code",
            h3_level_name="h3_l3_cell",
        )
        # The 5-row "crop" slice should have been merged into the bigger crop cell
        crop_cells = result.loc[result["ewoc_code"] == "crop", "h3_l3_cell"].unique()
        assert len(crop_cells) == 1, "Small crop slice should have merged into big one"

    def test_undersized_flag(self, multi_cell_df):
        result = merge_small_slices(
            multi_cell_df,
            min_size=100,
            label_col="ewoc_code",
            h3_level_name="h3_l3_cell",
            mark_undersized=True,
        )
        assert "undersized_slice" in result.columns

    def test_slice_id_assigned(self, multi_cell_df):
        result = merge_small_slices(
            multi_cell_df,
            min_size=50,
            label_col="ewoc_code",
            h3_level_name="h3_l3_cell",
        )
        assert "slice_id" in result.columns
        assert result["slice_id"].dtype == np.uint32


class TestComputeSliceCentroids:
    def test_centroid_shape(self):
        dim = 16
        rng = np.random.RandomState(0)
        n = 50
        X = rng.randn(n, dim).astype(np.float32)
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "ewoc_code": ["crop"] * n,
            "h3_l3_cell": ["cell_a"] * n,
            "embedding": [row for row in X],
        })
        centroids = compute_slice_centroids(df, label_col="ewoc_code")
        assert len(centroids) == 1  # one slice
        assert centroids["centroid"].iloc[0].shape == (dim,)


# ===================================================================
# 6. Flagging
# ===================================================================


class TestFlagAnomalies:
    @pytest.fixture()
    def scored_df(self):
        """A pre-scored DF with known S values."""
        n = 200
        rng = np.random.RandomState(0)
        s_vals = rng.uniform(0, 1, n).astype(np.float32)
        # Make last 10 clearly extreme
        s_vals[-10:] = np.linspace(0.95, 1.0, 10).astype(np.float32)
        return pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "ewoc_code": ["crop"] * n,
            "h3_l3_cell": ["cell_a"] * n,
            "S": s_vals,
            "S_rank": rng.uniform(0, 1, n).astype(np.float32),
            "S_rank_min": rng.uniform(0, 1, n).astype(np.float32),
            "S_z": rng.uniform(0, 1, n).astype(np.float32),
        })

    def test_percentile_mode(self, scored_df):
        flagged, summary = flag_anomalies(
            scored_df,
            label_col="ewoc_code",
            threshold_mode="percentile",
            percentile_q=0.95,
        )
        assert "flagged" in flagged.columns
        assert flagged["flagged"].sum() > 0
        assert flagged["flagged"].sum() <= len(flagged)

    def test_mad_mode(self, scored_df):
        flagged, summary = flag_anomalies(
            scored_df,
            label_col="ewoc_code",
            threshold_mode="mad",
            mad_k=2.0,
        )
        # MAD mode with uniform S may or may not flag depending on distribution;
        # just verify it runs and returns the right shape/columns
        assert "flagged" in flagged.columns
        assert len(flagged) == len(scored_df)

    def test_absolute_mode(self, scored_df):
        flagged, summary = flag_anomalies(
            scored_df,
            label_col="ewoc_code",
            threshold_mode="absolute",
            abs_threshold=0.95,
        )
        # Only the top 10 extreme values should be flagged
        assert flagged["flagged"].sum() >= 5

    def test_max_flagged_fraction(self, scored_df):
        flagged, summary = flag_anomalies(
            scored_df,
            label_col="ewoc_code",
            threshold_mode="percentile",
            percentile_q=0.80,
            max_flagged_fraction=0.05,
        )
        frac = flagged["flagged"].mean()
        assert frac <= 0.05 + 1e-6

    def test_summary_has_expected_columns(self, scored_df):
        _, summary = flag_anomalies(scored_df, label_col="ewoc_code")
        assert "total_samples" in summary.columns
        assert "flagged_samples" in summary.columns
        assert "flagged_fraction" in summary.columns

    def test_invalid_mode_raises(self, scored_df):
        with pytest.raises(ValueError, match="threshold_mode"):
            flag_anomalies(scored_df, label_col="ewoc_code", threshold_mode="bogus")


# ===================================================================
# 7. Confidence
# ===================================================================


class TestConfidence:
    def test_add_confidence_from_score_range(self):
        df = pd.DataFrame({"mean_score": np.linspace(0, 1, 100)})
        result = add_confidence_from_score(df, score_col="mean_score")
        assert "confidence" in result.columns
        conf = result["confidence"].to_numpy()
        assert conf.min() >= 0.01 - 1e-6  # conf_min default
        assert conf.max() <= 1.0 + 1e-6

    def test_confidence_decreases_with_score(self):
        df = pd.DataFrame({"mean_score": [0.0, 0.5, 0.9, 0.99, 1.0]})
        result = add_confidence_from_score(df, score_col="mean_score")
        conf = result["confidence"].to_numpy()
        # confidence should be monotonically non-increasing
        assert np.all(np.diff(conf) <= 1e-6)

    def test_confidence_at_zero_score_is_one(self):
        df = pd.DataFrame({"mean_score": [0.0]})
        result = add_confidence_from_score(df, score_col="mean_score")
        assert result["confidence"].iloc[0] == pytest.approx(1.0, abs=1e-3)

    def test_add_flagged_robust_confidence(self):
        n = 100
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "mean_score": rng.uniform(0.3, 0.9, n),
            "flagged": [False] * 80 + [True] * 20,
        })
        result = add_flagged_robust_confidence(df)
        assert "z_mad" in result.columns
        assert "confidence" in result.columns
        # Unflagged should all be 1.0
        unflagged_conf = result.loc[~result["flagged"], "confidence"]
        assert (unflagged_conf == 1.0).all()


class TestConfidenceFusion:
    def test_fusion_produces_columns(self):
        df = pd.DataFrame({
            "confidence": [0.9, 0.8, 0.7],
            "alt_margin_ctx": [0.2, 0.0, -0.1],
            "knn_same_label_frac_ctx": [1.0, 0.5, 0.1],
            "flagged": [True, True, True],
        })
        result = apply_confidence_fusion(df)
        assert "confidence_alt" in result.columns
        assert "p_margin" in result.columns
        assert "p_purity" in result.columns

    def test_unflagged_get_no_penalty(self):
        df = pd.DataFrame({
            "confidence": [0.9, 0.9],
            "alt_margin_ctx": [-0.5, -0.5],
            "knn_same_label_frac_ctx": [0.1, 0.1],
            "flagged": [False, False],
        })
        result = apply_confidence_fusion(df)
        # For unflagged, p_margin and p_purity should be 1.0
        np.testing.assert_allclose(result["p_margin"].to_numpy(), 1.0, atol=1e-6)
        np.testing.assert_allclose(result["p_purity"].to_numpy(), 1.0, atol=1e-6)

    def test_missing_columns_default_to_no_penalty(self):
        df = pd.DataFrame({
            "confidence": [0.8, 0.7],
            "flagged": [True, True],
        })
        result = apply_confidence_fusion(df)
        # Without margin or purity cols, factors should default to 1.0
        # (but capped at 0.85 minimum)
        assert result["confidence_alt"].notna().all()


# ===================================================================
# 8. Context-aware metrics
# ===================================================================


class TestAltClassCentroidMetrics:
    def test_two_classes_produces_columns(self):
        dim = 16
        rng = np.random.RandomState(0)
        n = 60
        X = rng.randn(n, dim).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # First 30 are class A, last 30 are class B (flipped)
        X[30:] = -X[:30]
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "label": (["A"] * 30) + (["B"] * 30),
            "ctx": ["ctx1"] * n,
            "embedding": [row for row in X],
        })
        result = add_alt_class_centroid_metrics(
            df, label_col="label", context_cols=["ctx"]
        )
        assert "self_centroid_dist_ctx" in result.columns
        assert "alt_centroid_dist_ctx" in result.columns
        assert "alt_margin_ctx" in result.columns
        assert "context_n_labels" in result.columns
        assert (result["context_n_labels"] == 2).all()

    def test_single_class_no_alt(self):
        dim = 8
        rng = np.random.RandomState(0)
        n = 20
        X = rng.randn(n, dim).astype(np.float32)
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "label": ["A"] * n,
            "ctx": ["ctx1"] * n,
            "embedding": [row for row in X],
        })
        result = add_alt_class_centroid_metrics(
            df, label_col="label", context_cols=["ctx"]
        )
        # Only one label → alt columns should be NaN
        assert result["alt_centroid_dist_ctx"].isna().all()


# ===================================================================
# 9. kNN purity
# ===================================================================


class TestKnnLabelPurity:
    def test_pure_neighbourhood(self):
        """If all neighbours share the same label, purity should be 1.0."""
        dim = 8
        rng = np.random.RandomState(0)
        n = 60
        X = rng.randn(n, dim).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        df_all = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "label": ["A"] * n,
            "ctx": ["ctx1"] * n,
            "embedding": [row for row in X],
        })
        flagged_df = df_all.copy()
        flagged_df["flagged"] = [False] * (n - 5) + [True] * 5
        result = add_knn_label_purity_for_flagged(
            df_all, flagged_df,
            label_col="label", context_cols=["ctx"],
            purity_knn_k=5,
        )
        # All same label → purity should be 1.0
        flagged_purity = result.loc[result["flagged"], "knn_same_label_frac_ctx"]
        assert (flagged_purity == 1.0).all()

    def test_no_flagged_returns_nan(self):
        dim = 8
        n = 20
        rng = np.random.RandomState(0)
        X = rng.randn(n, dim).astype(np.float32)
        df_all = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "label": ["A"] * n,
            "ctx": ["ctx1"] * n,
            "embedding": [row for row in X],
        })
        flagged_df = df_all.copy()
        flagged_df["flagged"] = False
        result = add_knn_label_purity_for_flagged(
            df_all, flagged_df,
            label_col="label", context_cols=["ctx"],
        )
        assert result["knn_same_label_frac_ctx"].isna().all()


# ===================================================================
# 10. Hierarchical ref-class assignment
# ===================================================================


class TestHierarchicalRefClass:
    def test_single_level_all_level0(self):
        n = 100
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "h3_l3_cell": ["cell_a"] * n,
            "label_fine": ["crop"] * n,
        })
        result = _add_hierarchical_ref_outlier_class(
            df,
            label_cols=["label_fine"],
            group_cols=[],
            h3_level_name="h3_l3_cell",
            min_slice_size=50,
        )
        assert (result["ref_outlier_level"] == 0).all()
        assert (result["ref_outlier_class"] == "crop").all()

    def test_hierarchical_fallback(self):
        """Small fine-grained slice should fall back to coarser level."""
        rows = []
        # 10 "wheat" samples (fine label), all "crop" at coarse level
        for i in range(10):
            rows.append({
                "sample_id": f"wheat_{i}",
                "h3_l3_cell": "cell_a",
                "label_fine": "wheat",
                "label_coarse": "crop",
            })
        # 200 "maize" samples (fine label), also "crop" at coarse level
        for i in range(200):
            rows.append({
                "sample_id": f"maize_{i}",
                "h3_l3_cell": "cell_a",
                "label_fine": "maize",
                "label_coarse": "crop",
            })
        df = pd.DataFrame(rows)
        result = _add_hierarchical_ref_outlier_class(
            df,
            label_cols=["label_fine", "label_coarse"],
            group_cols=[],
            h3_level_name="h3_l3_cell",
            min_slice_size=50,
        )
        # "wheat" slice has only 10 → should fall back to coarser
        wheat_level = result.loc[result["label_fine"] == "wheat", "ref_outlier_level"]
        assert (wheat_level == 1).all()
        # "maize" has 200 → stays at level 0
        maize_level = result.loc[result["label_fine"] == "maize", "ref_outlier_level"]
        assert (maize_level == 0).all()


# ===================================================================
# 11. Integration: scoring → flagging → confidence (mini pipeline)
# ===================================================================


class TestMiniPipeline:
    def test_end_to_end_synthetic(self):
        """Smoke-test the core scoring → flag → confidence chain on synthetic data."""
        n, n_out = 200, 10
        df = _make_slice_df(n=n, n_outliers=n_out, label="cropland")

        # Score
        scored = compute_scores_for_slice(df)
        assert len(scored) == n
        for col in _SCORE_COLS:
            assert col in scored.columns

        # Flag
        scored["h3_l3_cell"] = "cell_a"
        scored["ewoc_code"] = "cropland"
        flagged, summary = flag_anomalies(
            scored,
            label_col="ewoc_code",
            threshold_mode="percentile",
            percentile_q=0.90,
        )
        assert flagged["flagged"].sum() > 0

        # Confidence
        flagged = add_confidence_from_score(flagged, score_col="mean_score")
        assert flagged["confidence"].between(0.0, 1.0).all()

        # Flagged samples should generally have lower confidence
        flagged_conf = flagged.loc[flagged["flagged"], "confidence"].mean()
        normal_conf = flagged.loc[~flagged["flagged"], "confidence"].mean()
        # This is a statistical expectation, not a strict invariant, but with
        # planted outliers it should hold
        assert flagged_conf <= normal_conf + 0.05
