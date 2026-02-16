from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, classification_report

from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count


SplitMode = Literal["stratified_random", "group_h3", "group_ref_id"]
WeightMode = Literal["none", "filter", "conf_linear", "conf_power", "conf_clip", "conf_affine"]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train_conf_thresh: Optional[float] = None
    val_conf_thresh: Optional[float] = None
    weight_mode: WeightMode = "none"
    weight_power: float = 1.0              # for conf_power
    weight_min: float = 0.75               # for conf_clip / conf_affine
    weight_max: float = 1.0                # for conf_clip / conf_affine
    weight_affine_a: float = 1.0           # for conf_affine: w = clip(a*conf + b)
    weight_affine_b: float = 0.0           # for conf_affine: w = clip(a*conf + b)


def _find_conf_col(df: pd.DataFrame, prefer: Sequence[str] = ("confidence","mean_score")) -> str:
    for c in prefer:
        if c in df.columns:
            return c
    raise ValueError(f"No confidence column found. Tried: {prefer}. Available cols: {list(df.columns)[:50]}...")


def _extract_embeddings_matrix(
    df: pd.DataFrame,
    embedding_col: str = "embedding",
    embedding_prefix: str = "embedding_",
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      X: float32 matrix [n, d]
      used_cols: columns used or ['embedding'] if vector-col was used
    """
    wide_cols = [c for c in df.columns if c.startswith(embedding_prefix)]
    if wide_cols:
        wide_cols = sorted(wide_cols, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)
        X = df[wide_cols].to_numpy(dtype=np.float32, copy=True)
        return X, wide_cols

    if embedding_col in df.columns:
        # Expect list/np.ndarray per row
        arr = df[embedding_col].to_list()
        X = np.asarray(arr, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Column '{embedding_col}' could not be converted to 2D array. Got shape {X.shape}.")
        return X, [embedding_col]

    raise ValueError(
        f"No embeddings found. Expected wide columns '{embedding_prefix}*' or vector column '{embedding_col}'."
    )


def _apply_conf_filter(
    df: pd.DataFrame,
    conf_col: str,
    thresh: Optional[float],
) -> pd.DataFrame:
    if thresh is None:
        return df
    if not (0.0 <= float(thresh) <= 1.0):
        raise ValueError(f"Confidence threshold must be in [0,1]. Got {thresh}")
    return df[df[conf_col] >= float(thresh)].copy()


def _compute_sample_weights(
    conf: np.ndarray,
    mode: WeightMode,
    power: float,
    wmin: float,
    wmax: float,
    affine_a: float,
    affine_b: float,
) -> Optional[np.ndarray]:
    """
    Higher confidence -> higher weight.

    Returns None if mode == 'none' (CatBoost will treat all weights equal).
    """
    conf = conf.astype(np.float32, copy=False)

    if mode == "none" or mode == "filter":
        return None

    if mode == "conf_linear":
        w = conf

    elif mode == "conf_power":
        if power <= 0:
            raise ValueError(f"weight_power must be > 0. Got {power}")
        w = np.power(conf, power)

    elif mode == "conf_clip":
        w = np.clip(conf, wmin, wmax)

    elif mode == "conf_affine":
        w = np.clip(affine_a * conf + affine_b, wmin, wmax)

    else:
        raise ValueError(f"Unknown weight_mode: {mode}")

    # Avoid all-zero or degenerate weights
    w = w.astype(np.float32)
    if not np.isfinite(w).all():
        raise ValueError("Non-finite weights produced.")
    if float(w.max()) <= 0:
        raise ValueError("All weights are <= 0.")
    return w


def _split_train_val(
    df: pd.DataFrame,
    label_col: str,
    split_mode: SplitMode,
    val_size: float,
    seed: int,
    group_h3_col: Optional[str] = None,
    group_ref_col: str = "ref_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0,1).")

    if split_mode == "stratified_random":
        y = df[label_col].astype(str)
        tr_idx, va_idx = train_test_split(
            df.index.to_numpy(),
            test_size=val_size,
            random_state=seed,
            stratify=y,
        )
        return df.loc[tr_idx].copy(), df.loc[va_idx].copy()

    if split_mode == "group_h3":
        if group_h3_col is None or group_h3_col not in df.columns:
            raise ValueError(f"group_h3_col='{group_h3_col}' not found in df. Provide a valid H3 column.")
        groups = df[group_h3_col].astype(str).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_idx, va_idx = next(gss.split(df, groups=groups))
        return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

    if split_mode == "group_ref_id":
        if group_ref_col not in df.columns:
            raise ValueError(f"group_ref_col='{group_ref_col}' not found.")
        groups = df[group_ref_col].astype(str).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_idx, va_idx = next(gss.split(df, groups=groups))
        return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

    raise ValueError(f"Unknown split_mode: {split_mode}")


def _train_catboost_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: Optional[np.ndarray],
    seed: int,
    params: Optional[Dict[str, Any]] = None,
) -> CatBoostClassifier:
    # Sensible defaults for embeddings
    # check if gpu is available
    gpu_available = get_gpu_device_count() > 0
    task_type = "GPU" if gpu_available else "CPU"
    p = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1",     # CatBoost has TotalF1 for multiclass
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=100,
        verbose=False,
        task_type=task_type,
    )
    if params:
        p.update(params)

    model = CatBoostClassifier(**p)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        sample_weight=sample_weight,
        use_best_model=True,
    )
    return model


def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average_labels: Sequence[str] = ("macro", "weighted"),
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    for avg in average_labels:
        out[f"f1_{avg}"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    return out


def run_training_tests(
    outliers_parquet: str,
    ref_ids: Sequence[str],
    label_col: str,
    *,
    split_mode: SplitMode = "stratified_random",
    group_h3_col: Optional[str] = None,    # e.g., "h3_l1_cell"
    val_size: float = 0.2,
    seeds: Sequence[int] = (0, 1, 2),
    experiments: Optional[Sequence[ExperimentConfig]] = None,
    catboost_params: Optional[Dict[str, Any]] = None,
    conf_col_preference: Sequence[str] = ("confidence","mean_score"),  
    min_class_count: int = 20,             # drop labels that are too rare (after ref_id subset)
    max_rows: Optional[int] = None,        # optional cap for quick tests
    return_predictions: bool = False,      # set True to return per-sample preds for deeper analysis
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Loads outlier parquet (with embeddings), subsets to ref_ids, runs multiple experiments,
    returns a summary table with train/val metrics aggregated over seeds.

    Important: For "filter" experiments, training/validation sets are filtered by confidence thresholds.
               For weighting experiments, the train set keeps all rows but uses sample_weight derived from confidence.
    """
    # if outliers_parquet is a dataframe, use it directly
    if isinstance(outliers_parquet, pd.DataFrame):
        df = outliers_parquet.copy()
    else:
        df = pd.read_parquet(outliers_parquet)
    if "ref_id" not in df.columns:
        raise ValueError("Expected column 'ref_id' in parquet.")
    if ref_ids:
        print(f"Subsetting to {len(ref_ids)} ref_ids.")
        df = df[df["ref_id"].isin(list(ref_ids))].copy()
        
    print(f"Initial shape before splits: {df.shape}")
        
    if max_rows is not None and len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=0).copy()

    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Available: {list(df.columns)[:50]}...")

    conf_col = _find_conf_col(df, prefer=conf_col_preference)
    if conf_col == "mean_score":
        # we need 1-mean_score instead
        df["mean_score_conf"] = 1.0 - df["mean_score"]
        conf_col = "mean_score_conf"
        print("[WARN] Using transformed confidence column 'mean_score_conf' = 1 - mean_score")
    # Clean labels
    df = df.dropna(subset=[label_col, conf_col]).copy()
    df[label_col] = df[label_col].astype(str)
    df[conf_col] = df[conf_col].astype(float)

    print("Using conf_col with percentile values:", conf_col)
    print(df[conf_col].quantile([0, .01, .05, .1, .5, .9, .95, .99, 1]))

    # Drop very rare classes to keep metrics stable
    vc = df[label_col].value_counts()
    keep_labels = vc[vc >= int(min_class_count)].index
    df = df[df[label_col].isin(keep_labels)].copy()

    if df[label_col].nunique() < 2:
        raise ValueError("Need at least 2 classes after filtering by ref_ids/min_class_count.")

    X_all, used_embed_cols = _extract_embeddings_matrix(df)

    # Add X back via aligned index (stable row order)
    df = df.reset_index(drop=True)
    y_all = df[label_col].to_numpy()
    c_all = df[conf_col].to_numpy(dtype=np.float32)

    if experiments is None:
        # A default suite that usually answers the key question quickly.
        experiments = [
            ExperimentConfig(name="baseline_all", weight_mode="none"),
            ExperimentConfig(name="train>=0.75_val>=0.75", train_conf_thresh=0.75, val_conf_thresh=0.75, weight_mode="filter"),
            ExperimentConfig(name="train>=0.9_val>=0.9", train_conf_thresh=0.9, val_conf_thresh=0.9, weight_mode="filter"),
            ExperimentConfig(name="train>=0.99_val>=0.99", train_conf_thresh=0.99, val_conf_thresh=0.99, weight_mode="filter"),
            ExperimentConfig(name="weight_linear", weight_mode="conf_linear"),
            ExperimentConfig(name="weight_power_2", weight_mode="conf_power", weight_power=2.0),
            ExperimentConfig(name="weight_power_4", weight_mode="conf_power", weight_power=4.0),
            ExperimentConfig(name="weight_power_8", weight_mode="conf_power", weight_power=8.0),
            ExperimentConfig(name="val_filter_conf>=0.75", val_conf_thresh=0.75, weight_mode="none"),
            ExperimentConfig(name="train>=0.95_val>=0.95", train_conf_thresh=0.95, val_conf_thresh=0.95, weight_mode="filter"),
            ExperimentConfig(name="weight_power_2_val>=0.95", val_conf_thresh=0.95, weight_mode="conf_power", weight_power=2.0),
        ]

    rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []

    for seed in seeds:
        tr_df, va_df = _split_train_val(
            df, label_col=label_col, split_mode=split_mode, val_size=val_size, seed=int(seed),
            group_h3_col=group_h3_col,
        )

        # pre-extract matrices for speed using original row indices
        tr_idx = tr_df.index.to_numpy()
        va_idx = va_df.index.to_numpy()

        X_tr_full = X_all[tr_idx]
        y_tr_full = y_all[tr_idx]
        c_tr_full = c_all[tr_idx]

        X_va_full = X_all[va_idx]
        y_va_full = y_all[va_idx]
        c_va_full = c_all[va_idx]

        for exp in experiments:
            print(f"\nRunning experiment '{exp.name}' (seed={seed})...")
            # Train/val filtering if requested
            tr_mask = np.ones(len(tr_df), dtype=bool)
            va_mask = np.ones(len(va_df), dtype=bool)

            if exp.train_conf_thresh is not None:
                tr_mask = c_tr_full >= float(exp.train_conf_thresh)
            if exp.val_conf_thresh is not None:
                va_mask = c_va_full >= float(exp.val_conf_thresh)

            if exp.weight_mode == "filter":
                # Filtering-driven experiment: actually drop rows
                X_tr = X_tr_full[tr_mask]
                y_tr = y_tr_full[tr_mask]
                c_tr = c_tr_full[tr_mask]
                X_va = X_va_full[va_mask]
                y_va = y_va_full[va_mask]
                c_va = c_va_full[va_mask]
                # print how many samples are left after filtering
                print(f"Filtered train samples: {len(X_tr)}, Filtered val samples: {len(X_va)}")
            else:
                # Weighting-driven experiment: keep all rows in train; optionally filter val
                X_tr = X_tr_full
                y_tr = y_tr_full
                c_tr = c_tr_full
                X_va = X_va_full[va_mask]
                y_va = y_va_full[va_mask]
                c_va = c_va_full[va_mask]
                # print how many samples are left after filtering
                print(f"Filtered weighted train samples: {len(X_tr)}, Filtered weighted val samples: {len(X_va)}")
                
            tr_labels = set(np.unique(y_tr))
            va_labels = set(np.unique(y_va))
            missing_in_train = sorted(va_labels - tr_labels)
            missing_in_val = sorted(tr_labels - va_labels)

            if missing_in_train:
                print(f"[WARN] Labels present in VAL but missing in TRAIN after filtering: {missing_in_train[:20]} "
                    f"(+{max(0, len(missing_in_train)-20)} more)")

            def enforce_min_per_class(y, min_n):
                vc = pd.Series(y).value_counts()
                return set(vc[vc >= min_n].index)

            keep = enforce_min_per_class(y_tr, 20) & enforce_min_per_class(y_va, 10)
            mask_tr = np.isin(y_tr, list(keep))
            mask_va = np.isin(y_va, list(keep))
            X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
            X_va, y_va = X_va[mask_va], y_va[mask_va]
            c_tr = c_tr[mask_tr]
            c_va = c_va[mask_va]  # optional, not used for training weights but keep consistent

            if len(X_tr) < 100 or len(X_va) < 100:
                print(f"[\nWARN] Skipping experiment '{exp.name}' (seed={seed}) due to insufficient samples: train={len(X_tr)}, val={len(X_va)}")
                # avoid unstable fits in tiny debug subsets
                rows.append({
                    "experiment": exp.name,
                    "seed": seed,
                    "n_train": int(len(X_tr)),
                    "n_val": int(len(X_va)),
                    "skipped": True,
                    **{k: np.nan for k in ["train_accuracy", "train_f1_macro", "train_f1_weighted",
                                           "val_accuracy", "val_f1_macro", "val_f1_weighted"]},
                    "label_col": label_col,
                    "conf_col": conf_col,
                    "split_mode": split_mode,
                    "embed_cols": "wide" if used_embed_cols and used_embed_cols[0].startswith("embedding_") else "vector",
                    **asdict(exp),
                })
                continue
            
            # Sample weights (train only)
            w = _compute_sample_weights(
                conf=c_tr,
                mode=exp.weight_mode,
                power=exp.weight_power,
                wmin=exp.weight_min,
                wmax=exp.weight_max,
                affine_a=exp.weight_affine_a,
                affine_b=exp.weight_affine_b,
            )

            model = _train_catboost_multiclass(
                X_train=X_tr, y_train=y_tr,
                X_val=X_va, y_val=y_va,
                sample_weight=w,
                seed=int(seed),
                params=catboost_params,
            )

            # Evaluate on train (same subset used for training if filter-mode; else full train)
            yhat_tr = model.predict(X_tr).astype(str).reshape(-1)
            train_metrics = _evaluate(y_tr, yhat_tr)

            # Evaluate on val (possibly filtered)
            yhat_va = model.predict(X_va).astype(str).reshape(-1)
            val_metrics = _evaluate(y_va, yhat_va)

            rows.append({
                "experiment": exp.name,
                "seed": seed,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "skipped": False,
                "train_accuracy": train_metrics["accuracy"],
                "train_f1_macro": train_metrics["f1_macro"],
                "train_f1_weighted": train_metrics["f1_weighted"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "label_col": label_col,
                "conf_col": conf_col,
                "split_mode": split_mode,
                "embed_cols": "wide" if used_embed_cols and used_embed_cols[0].startswith("embedding_") else "vector",
                **asdict(exp),
            })
            print(f"  Train F1 macro: {train_metrics['f1_macro']:.4f}, Val F1 macro: {val_metrics['f1_macro']:.4f}")
            if return_predictions:
                # Store per-sample predictions for the val set only
                tmp = va_df.iloc[va_mask].copy()
                tmp = tmp.iloc[mask_va].copy()   # apply the second-stage mask
                tmp["pred_label"] = yhat_va
                tmp["true_label"] = y_va
                tmp["experiment"] = exp.name
                tmp["seed"] = seed
                pred_rows.append(tmp[["sample_id", "ref_id", label_col, conf_col, "true_label", "pred_label", "experiment", "seed"]])

    res = pd.DataFrame(rows)

    # Aggregate across seeds: mean/std per experiment (excluding skipped rows)
    agg = (
        res[res["skipped"] == False]
        .groupby("experiment", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            n_train_mean=("n_train", "mean"),
            n_val_mean=("n_val", "mean"),
            val_f1_macro_mean=("val_f1_macro", "mean"),
            val_f1_macro_std=("val_f1_macro", "std"),
            val_f1_weighted_mean=("val_f1_weighted", "mean"),
            val_f1_weighted_std=("val_f1_weighted", "std"),
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            train_f1_macro_mean=("train_f1_macro", "mean"),
            train_f1_macro_std=("train_f1_macro", "std"),
            train_f1_weighted_mean=("train_f1_weighted", "mean"),
            train_f1_weighted_std=("train_f1_weighted", "std"),
        )
        .sort_values("val_f1_macro_mean", ascending=False)
        .reset_index(drop=True)
    )

    if return_predictions:
        preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        return agg, preds

    return agg
