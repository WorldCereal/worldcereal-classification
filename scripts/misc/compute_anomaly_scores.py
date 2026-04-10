#!/usr/bin/env python3
"""Full end-to-end pipeline: long-format parquets → Presto embeddings → outlier scores → write back.

Two modes controlled by ``--mode``:

- **rerun** (default): Full pipeline on all data.  Converts long→wide parquets,
  merges them, populates / updates the DuckDB Presto embeddings cache, then runs
  outlier scoring for LANDCOVER10 and CROPTYPE24 over the **entire** embeddings
  cache and writes the 6 anomaly columns back to every long-format parquet.

- **update**: Incremental pipeline.  Runs the same long→wide / merge / embeddings
  steps (which are already incremental by nature), then — instead of re-scoring
  everything — scans the output long-format parquets for rows with NaN anomaly
  scores, identifies the geographic impact zone (H3 cells containing new points
  + their neighbours), loads only the embeddings that fall inside that zone, and
  re-scores only those slices.  Only the affected parquet files are rewritten.
  Use this after adding new labelled datasets to an existing scored collection.

Steps (each step skipped if its output already exists, unless --overwrite-* is set):

  1. Discover long-format parquet files
  2. Long → Wide via process_parquet        (skip per-file if wide file exists)
  3. Stream-merge wide → one parquet        (skip if merged file exists)
  4. Populate / update DuckDB embeddings    (incremental by model-hash + sample_id)
  5. Load class mappings (SharePoint with JSON cache fallback, or --class-mappings-json)
  6. Run LANDCOVER10 outlier pipeline       (skip if lc10 samples parquet exists)
  7. Run CROPTYPE24  outlier pipeline       (skip if cty24 samples parquet exists)
  8. Merge LC10 + CTY24 scores             (skip if merged scores parquet exists)
  9. Write scores → long-format parquets
 10. Write scores → merged wide parquet    (streaming Arrow, skip if file exists)

Example usage
-------------
# Full rerun (derive all paths from --input-long-dir and --suffix):
python compute_anomaly_scores.py \\
    --mode rerun \\
    --input-long-dir /projects/worldcereal/data/worldcereal_all_extractions.parquet

# Incremental update (new datasets added, same embeddings DB):
python compute_anomaly_scores.py \\
    --mode update \\
    --input-long-dir /projects/worldcereal/data/worldcereal_all_extractions.parquet \\
    --output-long-dir /projects/worldcereal/data/worldcereal_all_extractions_with_anomalies.parquet \\
    --embeddings-db-path /projects/worldcereal/data/cached_embeddings/embeddings_cache.duckdb \\
    --class-mappings-json /path/to/class_mappings.json

python compute_anomaly_scores.py \\
--mode update \\
--input-format geoparquet \\
--input-long-dir /data/worldcereal_data/EXTRACTIONS/WORLDCEREAL/MERGED_PARQUETS_PHASEII_WITH_ANOMALY \\
--embeddings-db-path /data/worldcereal_data/EXTRACTIONS/WORLDCEREAL/EMBEDDINGS_CACHE/embeddings_cache_LANDCOVER10_updated.duckdb \\
--wide-dir /data/worldcereal_data/EXTRACTIONS/WORLDCEREAL/CACHED_WIDE_MERGED/cached_wide_merged/cached_wide_parquets \\
--merged-wide-path /data/worldcereal_data/EXTRACTIONS/WORLDCEREAL/CACHED_WIDE_MERGED/cached_wide_merged/worldcereal_all_extractions_wide_month.parquet \\
--sp-env-file /home/wcextractions/.sharepointenv 
  
# Skip the heavy embeddings rebuild and re-score only:
python compute_anomaly_scores.py --mode rerun --skip-embeddings \\
    --embeddings-db-path /data/.../embeddings_cache.duckdb \\
    --class-mappings-json /data/.../class_mappings.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Ensure worldcereal is importable when running from this script's directory
# ---------------------------------------------------------------------------

def _ensure_worldcereal_importable() -> None:
    """Ensure the LOCAL worldcereal source tree is importable.

    Always inserts the local ``src/`` directory at the front of sys.path so
    that local changes take priority over any installed worldcereal package.
    """
    here = Path(__file__).resolve()
    # This script lives at worldcereal-classification/scripts/misc/
    candidate_src = here.parents[2] / "src"
    if candidate_src.exists() and str(candidate_src) not in sys.path:
        sys.path.insert(0, str(candidate_src))


_ensure_worldcereal_importable()

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from prometheo.models import Presto  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from worldcereal.train.anomaly import run_pipeline  # noqa: E402
from worldcereal.train.anomaly_utils import (  # noqa: E402
    ANOMALY_COLUMNS,
    LC10_ANOMALY_COLUMNS,
    CTY24_ANOMALY_COLUMNS,
    find_unscored_samples,
    compute_impact_zone,
    load_affected_embeddings_from_cache,
    merge_scores_to_long_parquets,
)
from worldcereal.train.embeddings_cache import (  # noqa: E402
    compute_embeddings,
    get_model_hash,
    init_cache,
)
from worldcereal.utils.timeseries import process_parquet  # noqa: E402


# ---------------------------------------------------------------------------
# Default Presto checkpoint
# ---------------------------------------------------------------------------

_DEFAULT_PRESTO_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/"
    "PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
)


# ===========================================================================
# Configuration dataclasses
# ===========================================================================


@dataclass(frozen=True)
class ParquetProcessConfig:
    freq: Literal["month", "dekad"] = "month"
    required_min_timesteps: int | None = None
    use_valid_time: bool = True
    min_edge_buffer: int = 1
    max_timesteps_trim: int | str | None = 18
    engine: Literal["pyarrow", "fastparquet"] = "fastparquet"
    compression: str = "snappy"


@dataclass(frozen=True)
class MergeConfig:
    batch_rows: int = 100_000
    row_group_size: int = 100_000
    compression: str = "zstd"


@dataclass(frozen=True)
class EmbeddingsConfig:
    presto_url_or_path: str
    embeddings_db_path: Path
    batch_size: int = 16_384
    num_workers: int = 2
    force_recompute: bool = False
    prematch: bool = True
    show_progress: bool = True
    parquet_batch_rows: int = 300_000


@dataclass
class AnomalyRunConfig:
    """Configuration for a single anomaly scoring domain (LC10 or CTY24)."""
    label_domain: str
    class_mappings_name: str
    h3_levels: List[int]
    min_slice_size: int
    max_slice_size: Optional[int]
    merge_small_slice: bool = True
    max_merge_iterations: int = 16
    threshold_mode: str = "mad"
    mad_k: float = 4.0
    percentile_q: float = 0.96
    max_full_pairwise_n: int = 0
    norm_percentiles: Tuple[float, float] = (2.0, 98.0)
    skip_classes: Optional[List[str]] = field(default_factory=lambda: ["ignore"])
    # Final output column names
    confidence_col_name: str = "confidence_nonoutlier"
    anomaly_flag_col_name: str = "anomaly_flag"
    label_col_rename: Optional[str] = None  # e.g. "outlier_LC10_cls"


# Defaults matching the notebook's current runs
LC10_CONFIG = AnomalyRunConfig(
    label_domain="LANDCOVER10",
    class_mappings_name="LANDCOVER10",
    h3_levels=[2, 3],
    min_slice_size=200,
    max_slice_size=10_000,
    max_merge_iterations=16,
    skip_classes=["ignore"],
    confidence_col_name="LC10_confidence_nonoutlier",
    anomaly_flag_col_name="LC10_anomaly_flag",
    label_col_rename="outlier_LC10_cls",
)

CTY24_CONFIG = AnomalyRunConfig(
    label_domain="CROPTYPE24",
    class_mappings_name="CROPTYPE24",
    h3_levels=[2, 3, 4],
    min_slice_size=100,
    max_slice_size=5_000,
    max_merge_iterations=8,
    skip_classes=["ignore"],
    confidence_col_name="CTY24_confidence_nonoutlier",
    anomaly_flag_col_name="CTY24_anomaly_flag",
    label_col_rename="outlier_CTY24_cls",
)


# ===========================================================================
# Model loading
# ===========================================================================


def load_presto_model_compat(url_or_path: str) -> torch.nn.Module:
    """Load a Presto model, trying Presto(pretrained_model_path=...) first."""
    try:
        model = Presto(pretrained_model_path=url_or_path)
        model.eval()
        return model
    except Exception as e:
        logger.warning(
            f"Presto(pretrained_model_path=...) failed ({type(e).__name__}: {e}); "
            "falling back to worldcereal loader"
        )
        from worldcereal.train.embeddings_cache import load_presto_model as _load
        return _load(url_or_path)


# ===========================================================================
# Step 1 – Discover parquets
# ===========================================================================


def discover_parquets(input_dir: Path, pattern: str = "**/*.parquet") -> list[Path]:
    """Return sorted list of parquet files under *input_dir* matching *pattern*."""
    if input_dir.is_file() and input_dir.suffix == ".parquet":
        return [input_dir]
    if not input_dir.exists():
        raise FileNotFoundError(str(input_dir))
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


# ===========================================================================
# Step 2 – Long → Wide conversion
# ===========================================================================


def _wide_out_path(wide_dir: Path, raw_path: Path, suffix: str = "_ppq") -> Path:
    return wide_dir / f"{raw_path.stem}{suffix}.parquet"


def long_to_wide_parquets(
    raw_files: Sequence[Path],
    wide_dir: Path,
    cfg: ParquetProcessConfig,
    overwrite: bool = False,
) -> tuple[list[Path], list[Path], list[Path], bool]:
    """Convert long-format parquets to wide-format.

    Returns (produced, empty, errored, wrote_any).
    """
    wide_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    empty: list[Path] = []
    errored: list[Path] = []
    wrote_any = False

    for i, pf in enumerate(raw_files, start=1):
        out_path = _wide_out_path(wide_dir, pf)
        if out_path.exists() and not overwrite:
            logger.info(f"[{i}/{len(raw_files)}] skip existing: {out_path.name}")
            produced.append(out_path)
            continue

        try:
            df_long = pd.read_parquet(pf)
            df_long.reset_index(drop=False, inplace=True)
            df_wide = process_parquet(
                df_long,
                freq=cfg.freq,
                required_min_timesteps=cfg.required_min_timesteps,
                use_valid_time=cfg.use_valid_time,
                min_edge_buffer=cfg.min_edge_buffer,
                max_timesteps_trim=cfg.max_timesteps_trim,
            )
        except Exception as e:
            logger.exception(f"[{i}/{len(raw_files)}] ERROR processing {pf}: {e}")
            errored.append(pf)
            continue

        if df_wide.empty:
            logger.warning(f"[{i}/{len(raw_files)}] empty wide dataframe for {pf}")
            empty.append(pf)
            continue

        df_wide.to_parquet(out_path, engine=cfg.engine, compression=cfg.compression, index=False)
        wrote_any = True
        produced.append(out_path)
        logger.info(f"[{i}/{len(raw_files)}] wrote {out_path.name} shape={df_wide.shape}")
        del df_long, df_wide
        gc.collect()

    return produced, empty, errored, wrote_any


# ===========================================================================
# Step 3 – Merge wide parquets → one
# ===========================================================================


def _is_numeric(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def _build_target_schema(files: Sequence[Path]) -> pa.Schema:
    """Unified Arrow schema from the union of all file schemas."""
    type_map: dict[str, set[pa.DataType]] = {}
    for f in files:
        sch = pq.ParquetFile(str(f)).schema_arrow
        for fld in sch:
            type_map.setdefault(fld.name, set()).add(fld.type)

    fields: list[pa.Field] = []
    for name, typeset in sorted(type_map.items()):
        if len(typeset) == 1:
            fields.append(pa.field(name, next(iter(typeset))))
            continue
        if any(_is_numeric(t) for t in typeset):
            fields.append(pa.field(name, pa.float32()))
        elif any(pa.types.is_timestamp(t) for t in typeset):
            fields.append(pa.field(name, pa.timestamp("us")))
        elif any(pa.types.is_boolean(t) for t in typeset):
            fields.append(pa.field(name, pa.bool_()))
        else:
            fields.append(pa.field(name, pa.string()))
    return pa.schema(fields)


def _align_table_to_schema(tbl: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast / null-fill a table to match *schema* exactly."""
    arrays = []
    for fld in schema:
        name = fld.name
        if name in tbl.column_names:
            col = tbl[name]
            if not col.type.equals(fld.type):
                col = pc.cast(col, fld.type, safe=False)
            arrays.append(col)
        else:
            arrays.append(pa.nulls(tbl.num_rows, type=fld.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def merge_parquets_stream_to_one(
    files: Sequence[Path],
    out_path: Path,
    cfg: MergeConfig,
    overwrite: bool = False,
) -> Path:
    """Stream-merge many parquet files into one output parquet (memory-safe)."""
    if not files:
        raise ValueError("No input parquet files provided for merge.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        logger.info(f"skip merge — output exists: {out_path}")
        return out_path

    target_schema = _build_target_schema(files)
    logger.info(f"merge target schema: {len(target_schema)} fields")

    writer = pq.ParquetWriter(
        str(out_path), target_schema, compression=cfg.compression,
        use_dictionary=True, write_statistics=True,
    )
    total_rows = 0
    try:
        for fi, f in enumerate(files, start=1):
            pf = pq.ParquetFile(str(f))
            logger.info(f"[{fi}/{len(files)}] merging {f.name} ({pf.metadata.num_rows:,} rows)")
            for batch in pf.iter_batches(batch_size=cfg.batch_rows):
                tbl = pa.Table.from_batches([batch])
                tbl = _align_table_to_schema(tbl, target_schema)
                writer.write_table(tbl, row_group_size=cfg.row_group_size)
                total_rows += tbl.num_rows
                del batch, tbl
                gc.collect()
        logger.info(f"merge complete — {total_rows:,} rows → {out_path}")
    finally:
        writer.close()
    return out_path


# ===========================================================================
# Step 4 – Embeddings cache (incremental, with pre-match)
# ===========================================================================


def _open_cache_readonly(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def _count_missing_for_ids_batch(
    *,
    db_path: Path,
    model_hash: str,
    ids_df: pd.DataFrame,
    use_ref_id: bool,
) -> int:
    if ids_df.empty:
        return 0
    con = _open_cache_readonly(db_path)
    try:
        con.register("ids", ids_df)
        if use_ref_id:
            stats = con.execute(
                """SELECT COUNT(*) AS n_total,
                          SUM(CASE WHEN e.sample_id IS NULL THEN 1 ELSE 0 END) AS n_missing
                   FROM ids i
                   LEFT JOIN embeddings_cache e
                     ON e.model_hash = ? AND e.sample_id = i.sample_id
                        AND e.ref_id IS NOT DISTINCT FROM i.ref_id""",
                [model_hash],
            ).fetchone()
        else:
            stats = con.execute(
                """SELECT COUNT(*) AS n_total,
                          SUM(CASE WHEN e.sample_id IS NULL THEN 1 ELSE 0 END) AS n_missing
                   FROM ids i
                   LEFT JOIN embeddings_cache e
                     ON e.model_hash = ? AND e.sample_id = i.sample_id""",
                [model_hash],
            ).fetchone()
        return int((stats[1] if stats else 0) or 0)
    finally:
        con.close()


def compute_embeddings_from_merged_parquet(
    merged_wide_parquet: Path,
    model: torch.nn.Module,
    model_hash: str,
    cfg: EmbeddingsConfig,
) -> None:
    """Iterate ALL Arrow batches in the merged wide parquet and populate the cache."""
    pf = pq.ParquetFile(str(merged_wide_parquet))
    logger.info(f"embedding: rows={pf.metadata.num_rows} row_groups={pf.metadata.num_row_groups}")

    iterator = pf.iter_batches(batch_size=cfg.parquet_batch_rows)
    if cfg.show_progress:
        iterator = tqdm(iterator, total=None, desc="Parquet batches", unit="batch")  # type: ignore

    for batch in iterator:
        tbl = pa.Table.from_batches([batch])
        if tbl.num_rows == 0:
            del batch, tbl
            gc.collect()
            continue

        if "sample_id" not in tbl.column_names:
            raise ValueError(f"Merged parquet missing 'sample_id': {merged_wide_parquet}")

        # Pre-match: skip batch if all sample_ids are already cached
        if cfg.prematch and not cfg.force_recompute:
            id_cols = ["sample_id"] + (["ref_id"] if "ref_id" in tbl.column_names else [])
            ids_df = tbl.select(id_cols).to_pandas()
            ids_df["sample_id"] = ids_df["sample_id"].astype(str)
            use_ref_id = "ref_id" in ids_df.columns
            if use_ref_id:
                ids_df["ref_id"] = ids_df["ref_id"].astype(str)
            if _count_missing_for_ids_batch(
                db_path=cfg.embeddings_db_path, model_hash=model_hash,
                ids_df=ids_df, use_ref_id=use_ref_id,
            ) == 0:
                del batch, tbl, ids_df
                gc.collect()
                continue
            del ids_df

        df = tbl.to_pandas()
        compute_embeddings(
            df, model=model, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            embeddings_db_path=str(cfg.embeddings_db_path),
            force_recompute=cfg.force_recompute, show_progress=cfg.show_progress,
        )
        del batch, tbl, df
        gc.collect()


# ===========================================================================
# Step 5 – Class mappings
# ===========================================================================


def load_class_mappings(
    *,
    class_mappings_json: Path | None = None,
    sp_env_file: Path | None = None,
) -> dict:
    """Return class mappings dict.

    Priority:
      1. Local JSON file (--class-mappings-json) — no network required.
      2. SharePoint fetch with JSON cache fallback.
    """
    if class_mappings_json is not None:
        logger.info(f"loading class mappings from local JSON: {class_mappings_json}")
        with open(class_mappings_json) as f:
            return json.load(f)

    try:
        from dotenv import load_dotenv
    except ImportError:
        raise ImportError(
            "python-dotenv is required. Install with: pip install python-dotenv"
        )

    env_path = sp_env_file or Path.home() / ".sharepointenv"
    if not env_path.exists():
        raise FileNotFoundError(
            f"SharePoint env file not found: {env_path}. "
            "Provide --class-mappings-json or --sp-env-file."
        )
    load_dotenv(env_path, override=True)

    site_url = os.environ.get("WORLDCEREAL_SP_SITE_URL")
    file_url = os.environ.get("WORLDCEREAL_SP_FILE_URL")
    if not site_url or not file_url:
        raise EnvironmentError(
            "WORLDCEREAL_SP_SITE_URL and WORLDCEREAL_SP_FILE_URL must be set in "
            f"{env_path} (or as environment variables)."
        )

    from worldcereal.utils.sharepoint import build_class_mappings, get_excel_from_sharepoint

    logger.info("fetching class mappings from SharePoint ...")
    legend = get_excel_from_sharepoint(
        site_url=site_url,
        file_server_relative_url=file_url,
        retries=10,
        sheet_name=0,
    )
    legend["ewoc_code"] = (
        legend["ewoc_code"]
        .astype("string")
        .str.replace("-", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    mappings = build_class_mappings(legend)
    logger.info(f"class mappings loaded — keys: {list(mappings.keys())}")
    return mappings


# ===========================================================================
# Steps 6 & 7 – Single-domain scoring (shared by both modes)
# ===========================================================================


def run_single_domain_scoring(
    embeddings_db_path: str,
    config: AnomalyRunConfig,
    class_mappings: dict,
    *,
    embeddings_df: Optional[Tuple[pd.DataFrame, list]] = None,
    restrict_model_hash: Optional[str] = None,
    output_samples_path: Optional[str] = None,
    output_summary_path: Optional[str] = None,
    write_outputs: bool = True,
    overwrite: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """Run outlier scoring for a single label domain (LC10 or CTY24).

    **rerun mode**: call with ``embeddings_df=None``.  ``run_pipeline`` loads
    all embeddings from the DuckDB cache at *embeddings_db_path*.

    **update mode**: call with ``embeddings_df=(df, embed_cols)`` — a pre-loaded
    subset containing only the impact-zone embeddings.  ``run_pipeline`` bypasses
    the DuckDB load entirely and uses this DataFrame directly.  This is the key
    difference between the two modes and is what makes incremental scoring fast.

    Returns a DataFrame with ``ref_id``, ``sample_id``, and the domain-prefixed
    anomaly columns (e.g. ``LC10_confidence_nonoutlier``, ``LC10_anomaly_flag``,
    ``outlier_LC10_cls``).
    """
    if output_samples_path and Path(output_samples_path).exists() and not overwrite:
        logger.info(
            f"skip {config.label_domain} scoring — output exists: {output_samples_path}"
        )
        label_col = config.label_domain
        df = pd.read_parquet(
            output_samples_path,
            columns=["ref_id", "sample_id", label_col, "confidence_nonoutlier", "anomaly_flag"],
        )
    else:
        flagged_gdf, _ = run_pipeline(
            embeddings_db_path=embeddings_db_path,
            restrict_model_hash=restrict_model_hash,
            label_domain=config.label_domain,
            map_to_finetune=False,
            class_mappings_name=config.class_mappings_name,
            skip_classes=config.skip_classes,
            mapping_file=class_mappings,
            h3_level=config.h3_levels,
            group_cols=None,
            min_slice_size=config.min_slice_size,
            max_slice_size=config.max_slice_size,
            merge_small_slice=config.merge_small_slice,
            max_merge_iterations=config.max_merge_iterations,
            threshold_mode=config.threshold_mode,
            percentile_q=config.percentile_q,
            mad_k=config.mad_k,
            abs_threshold=None,
            fdr_alpha=0.05,
            min_flagged_per_slice=None,
            max_flagged_fraction=None,
            max_full_pairwise_n=config.max_full_pairwise_n,
            norm_percentiles=config.norm_percentiles,
            output_samples_path=output_samples_path,
            output_summary_path=output_summary_path,
            write_outputs=write_outputs,
            debug=debug,
            embeddings_df=embeddings_df,
        )
        label_col = config.label_domain
        keep = ["ref_id", "sample_id"]
        if label_col in flagged_gdf.columns:
            keep.append(label_col)
        keep += ["confidence_nonoutlier", "anomaly_flag"]
        keep = [c for c in keep if c in flagged_gdf.columns]
        df = flagged_gdf[keep].copy()

    logger.info(f"{config.label_domain}: {len(df):,} samples scored")

    # Rename to domain-prefixed column names
    rename_map = {}
    if "confidence_nonoutlier" in df.columns:
        rename_map["confidence_nonoutlier"] = config.confidence_col_name
    if "anomaly_flag" in df.columns:
        rename_map["anomaly_flag"] = config.anomaly_flag_col_name
    if config.label_domain in df.columns and config.label_col_rename:
        rename_map[config.label_domain] = config.label_col_rename
    df.rename(columns=rename_map, inplace=True)
    return df


# ===========================================================================
# Step 8 – Merge LC10 + CTY24 scores
# ===========================================================================


def merge_lc10_cty24_scores(
    lc10_df: pd.DataFrame,
    cty24_df: pd.DataFrame,
    merged_scores_path: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Outer-join LC10 and CTY24 scored DataFrames on (ref_id, sample_id)."""
    if merged_scores_path.exists() and not overwrite:
        logger.info(f"skip merge — reading existing: {merged_scores_path}")
        return pd.read_parquet(str(merged_scores_path))

    merged = cty24_df.merge(lc10_df, on=["ref_id", "sample_id"], how="outer")
    merged.sort_values(["ref_id", "sample_id"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged_scores_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(merged_scores_path), index=False)
    logger.info(f"merged scores: {len(merged):,} rows → {merged_scores_path}")
    return merged


# ===========================================================================
# Step 9 – Write scores back to long-format parquets
# ===========================================================================


def write_scores_to_long_parquets(
    merged_scores: pd.DataFrame,
    input_long_dir: Path,
    output_long_dir: Path,
    anomaly_cols: list[str],
    parquet_glob: str = "**/*.parquet",
    only_affected_ref_ids: Optional[set] = None,
) -> int:
    """Left-join anomaly scores onto every long-format parquet file."""
    # Don't mkdir when writing in-place (the directory already exists and
    # mkdir would fail or be a no-op; we avoid it to keep the call clean).
    if input_long_dir.resolve() != output_long_dir.resolve():
        output_long_dir.mkdir(parents=True, exist_ok=True)
    return merge_scores_to_long_parquets(
        scored_df=merged_scores,
        long_parquet_dir=input_long_dir,
        output_parquet_dir=output_long_dir,
        anomaly_cols=anomaly_cols,
        parquet_glob=parquet_glob,
        only_affected_ref_ids=only_affected_ref_ids,
    )


# ===========================================================================
# Step 10 – Write scores back to merged wide parquet (streaming Arrow)
# ===========================================================================


def write_scores_to_wide_parquet(
    merged_scores: pd.DataFrame,
    src_wide_path: Path,
    output_wide_path: Path,
    anomaly_cols: list[str],
    batch_rows: int = 100_000,
    row_group_size: int = 100_000,
    overwrite: bool = False,
) -> None:
    """Stream the wide parquet, left-join anomaly scores, write the result.

    **Incremental update** (update mode): when the output file already exists
    and ``overwrite=False``, instead of skipping, the function reads the
    *existing output* (which already has anomaly columns for previously scored
    rows), updates / fills in anomaly values for any ``(ref_id, sample_id)``
    present in *merged_scores*, and rewrites the file atomically via a temp
    file.  Rows not in *merged_scores* keep their existing anomaly values.

    **Full rewrite** (rerun mode / ``overwrite=True``): reads the *source*
    wide parquet (without anomaly columns), left-joins all scores, and writes
    the output from scratch.
    """
    scores_lookup = (
        merged_scores[["ref_id", "sample_id"] + anomaly_cols]
        .drop_duplicates(subset="sample_id")
        .copy()
    )

    # ------------------------------------------------------------------
    # Incremental update: the output file already exists and we don't want
    # a full overwrite — read the EXISTING output, patch the anomaly cols
    # for any newly scored sample_ids, and rewrite atomically.
    # ------------------------------------------------------------------
    if output_wide_path.exists() and not overwrite:
        logger.info(
            f"[wide-update] Incremental update — patching {len(scores_lookup):,} "
            f"sample scores into {output_wide_path}"
        )
        scored_ids = set(scores_lookup["sample_id"].unique())
        existing_pf = pq.ParquetFile(str(output_wide_path))
        existing_schema = existing_pf.schema_arrow
        logger.info(
            f"[wide-update] Existing output: {existing_pf.metadata.num_rows:,} rows, "
            f"{len(existing_schema)} cols"
        )

        # Ensure the target schema contains all anomaly columns
        extra_fields: list[pa.Field] = []
        for col in anomaly_cols:
            if col in existing_schema.names:
                continue
            dtype = scores_lookup[col].dtype
            arrow_type = pa.float32() if str(dtype).startswith("float") else pa.string()
            extra_fields.append(pa.field(col, arrow_type))
        if extra_fields:
            base_fields = [f for f in existing_schema]
            target_schema = pa.schema(base_fields + extra_fields)
        else:
            target_schema = existing_schema

        tmp_path = output_wide_path.with_suffix(".tmp.parquet")
        writer = pq.ParquetWriter(
            str(tmp_path), target_schema, compression="zstd",
            use_dictionary=True, write_statistics=True,
        )
        rows_written = 0
        rows_updated = 0
        try:
            for batch in existing_pf.iter_batches(batch_size=batch_rows):
                df_batch = batch.to_pandas()
                # Identify rows in this batch that have new scores
                mask = df_batch["sample_id"].isin(scored_ids)
                n_hits = int(mask.sum())
                if n_hits > 0:
                    # Drop existing anomaly cols for matched rows, merge new values
                    matched = df_batch.loc[mask].copy()
                    cols_to_drop = [c for c in anomaly_cols if c in matched.columns]
                    if cols_to_drop:
                        matched.drop(columns=cols_to_drop, inplace=True)
                    matched = matched.merge(
                        scores_lookup, on=["ref_id", "sample_id"], how="left",
                    )
                    # Recombine: unmatched rows keep their original anomaly values
                    unmatched = df_batch.loc[~mask]
                    df_batch = pd.concat([unmatched, matched], ignore_index=True)
                    df_batch.sort_values("sample_id", inplace=True)
                    rows_updated += n_hits

                # Fill any missing anomaly columns for rows not in scores_lookup
                for col in anomaly_cols:
                    if col not in df_batch.columns:
                        df_batch[col] = float("nan")

                tbl = pa.Table.from_pandas(df_batch, schema=target_schema, safe=False)
                writer.write_table(tbl, row_group_size=row_group_size)
                rows_written += len(df_batch)
                del df_batch, tbl, batch
                gc.collect()
            logger.info(
                f"[wide-update] {rows_written:,} rows written, "
                f"{rows_updated:,} updated → {output_wide_path}"
            )
        finally:
            writer.close()

        # Atomic rename
        tmp_path.replace(output_wide_path)
        return

    # ------------------------------------------------------------------
    # Full rewrite: read the source wide parquet (no anomaly cols) and
    # left-join all scores.  Used by rerun mode or when output doesn't
    # exist yet.
    # ------------------------------------------------------------------
    src_pf = pq.ParquetFile(str(src_wide_path))
    src_schema = src_pf.schema_arrow
    logger.info(f"wide source: {src_pf.metadata.num_rows:,} rows, {len(src_schema)} cols")

    extra_fields: list[pa.Field] = []
    for col in anomaly_cols:
        if col in src_schema.names:
            continue
        dtype = scores_lookup[col].dtype
        arrow_type = pa.float32() if str(dtype).startswith("float") else pa.string()
        extra_fields.append(pa.field(col, arrow_type))

    base_fields = [f for f in src_schema if f.name not in anomaly_cols]
    target_schema = pa.schema(base_fields + extra_fields)

    output_wide_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        str(output_wide_path), target_schema, compression="zstd",
        use_dictionary=True, write_statistics=True,
    )

    rows_written = 0
    try:
        for batch in src_pf.iter_batches(batch_size=batch_rows):
            df_batch = batch.to_pandas()
            cols_to_drop = [c for c in anomaly_cols if c in df_batch.columns]
            if cols_to_drop:
                df_batch.drop(columns=cols_to_drop, inplace=True)
            df_batch = df_batch.merge(scores_lookup, on=["ref_id", "sample_id"], how="left")
            tbl = pa.Table.from_pandas(df_batch, schema=target_schema, safe=False)
            writer.write_table(tbl, row_group_size=row_group_size)
            rows_written += len(df_batch)
            del df_batch, tbl, batch
            gc.collect()
        logger.info(f"wide with scores: {rows_written:,} rows → {output_wide_path}")
    finally:
        writer.close()


# ===========================================================================
# Mode: rerun — score ALL embeddings from DuckDB, write ALL parquets
# ===========================================================================


def _run_scoring_rerun(
    *,
    embeddings_db_path: Path,
    class_mappings: dict,
    lc10_samples_path: Path,
    lc10_summary_path: Path,
    cty24_samples_path: Path,
    cty24_summary_path: Path,
    merged_scores_path: Path,
    lc10_config: AnomalyRunConfig,
    cty24_config: AnomalyRunConfig,
    overwrite_scores: bool,
    overwrite_merged_scores: bool,
    write_review_parquets: bool,
    restrict_model_hash: Optional[str],
    debug: bool,
) -> pd.DataFrame:
    """Run both pipelines over the full DuckDB cache and return merged scores."""
    logger.info("[rerun] Running LANDCOVER10 scoring (full cache) ...")
    lc10_df = run_single_domain_scoring(
        embeddings_db_path=str(embeddings_db_path),
        config=lc10_config,
        class_mappings=class_mappings,
        output_samples_path=str(lc10_samples_path) if write_review_parquets else None,
        output_summary_path=str(lc10_summary_path) if write_review_parquets else None,
        write_outputs=write_review_parquets,
        overwrite=overwrite_scores,
        restrict_model_hash=restrict_model_hash,
        debug=debug,
    )

    logger.info("[rerun] Running CROPTYPE24 scoring (full cache) ...")
    cty24_df = run_single_domain_scoring(
        embeddings_db_path=str(embeddings_db_path),
        config=cty24_config,
        class_mappings=class_mappings,
        output_samples_path=str(cty24_samples_path) if write_review_parquets else None,
        output_summary_path=str(cty24_summary_path) if write_review_parquets else None,
        write_outputs=write_review_parquets,
        overwrite=overwrite_scores,
        restrict_model_hash=restrict_model_hash,
        debug=debug,
    )

    return merge_lc10_cty24_scores(
        lc10_df=lc10_df,
        cty24_df=cty24_df,
        merged_scores_path=merged_scores_path,
        overwrite=overwrite_merged_scores,
    )


# ===========================================================================
# Mode: update — impact zone only, write only affected parquets
# ===========================================================================


def _resolve_skip_ewoc_codes(
    class_mappings: dict,
    class_mappings_name: str,
    skip_classes: Optional[List[str]],
) -> set:
    """Return the set of ``ewoc_code`` strings that map to a *skip_class* label.

    Uses the class_mappings dict (keyed by domain name, e.g. ``CROPTYPE24``)
    to look up which ewoc_codes produce a label in *skip_classes*.  Returns
    an empty set when *skip_classes* is ``None`` or empty, or when the
    mapping is not found.
    """
    if not skip_classes:
        return set()
    mapping = class_mappings.get(class_mappings_name)
    if not isinstance(mapping, dict):
        return set()
    skip_set = {str(s).lower() for s in skip_classes}
    return {
        str(ewoc_code)
        for ewoc_code, label in mapping.items()
        if str(label).lower() in skip_set
    }


def _discover_domain_impact(
    *,
    domain_label: str,
    domain_anomaly_cols: list[str],
    h3_levels: list[int],
    long_parquet_dir: Path,
    embeddings_db_path: Path,
    neighbour_rings: int,
    restrict_model_hash: Optional[str],
    skip_ewoc_codes: Optional[set] = None,
    parquet_glob: str = "**/*.parquet",
) -> Tuple[Optional[pd.DataFrame], Optional[list], Optional[set]]:
    """Per-domain unscored detection → impact zone → embeddings load.

    Returns ``(affected_df, embed_cols, rescored_ref_ids)`` or
    ``(None, None, None)`` when there is nothing to do for this domain.

    By checking only the domain-specific anomaly columns (e.g. the 3 LC10
    columns instead of all 6), rows that are legitimately NaN because their
    label is a *skip_class* in the other domain are no longer pulled in.
    This typically reduces the "unscored" count from millions to only the
    genuinely new / missing samples.

    When *skip_ewoc_codes* is provided, unscored sample_ids whose
    ``ewoc_code`` in the embeddings cache matches a skip class for this
    domain are excluded **before** computing the impact zone.  These rows
    will always be NaN in the anomaly columns (by design — ``run_pipeline``
    holds them aside), so there is no point expanding the impact zone for
    them.  This is the key optimisation for CROPTYPE24 where ~half the
    samples map to "ignore".
    """
    logger.info(f"[update/{domain_label}] Scanning for unscored samples "
                f"(checking {domain_anomaly_cols}) ...")
    unscored = find_unscored_samples(
        long_parquet_dir=long_parquet_dir,
        anomaly_cols=domain_anomaly_cols,
        parquet_glob=parquet_glob,
    )
    if unscored.empty:
        logger.info(f"[update/{domain_label}] No unscored samples — skipping domain.")
        return None, None, None

    logger.info(f"[update/{domain_label}] {len(unscored):,} unscored (ref_id, sample_id) pairs")

    # Look up their H3 cells (and ewoc_code for skip-class filtering) in cache
    logger.info(f"[update/{domain_label}] Looking up H3 cells for unscored samples ...")
    con = duckdb.connect(str(embeddings_db_path), read_only=True)
    try:
        id_df = pd.DataFrame({"sample_id": unscored["sample_id"].astype(str).tolist()})
        con.register("unscored_ids", id_df)
        h3_df = con.execute(
            """SELECT DISTINCT e.sample_id, e.h3_l3_cell, e.ewoc_code
               FROM unscored_ids u
               INNER JOIN embeddings_cache e ON e.sample_id = u.sample_id"""
        ).fetchdf()
    finally:
        con.close()

    # Filter out samples whose ewoc_code maps to a skip class for this domain.
    # These will always be NaN in the anomaly columns (run_pipeline holds them
    # aside with NaN scores), so including them would needlessly inflate the
    # impact zone.
    if skip_ewoc_codes and not h3_df.empty:
        before = len(h3_df)
        h3_df = h3_df[~h3_df["ewoc_code"].astype(str).isin(skip_ewoc_codes)].copy()
        n_skipped = before - len(h3_df)
        if n_skipped > 0:
            logger.info(
                f"[update/{domain_label}] Excluded {n_skipped:,} samples "
                f"with skip-class ewoc_codes ({before:,} → {len(h3_df):,})"
            )

    if h3_df.empty:
        logger.warning(
            f"[update/{domain_label}] Unscored samples not found in embeddings cache "
            "(or all were skip-class). "
            "Run the embeddings pipeline (steps 1–4) first if this is unexpected."
        )
        return None, None, None

    logger.info(f"[update/{domain_label}] {len(h3_df):,} unscored samples found in cache")

    # Drop ewoc_code — no longer needed after skip-class filtering
    h3_df = h3_df[["sample_id", "h3_l3_cell"]]

    unscored_h3_cells = h3_df["h3_l3_cell"].dropna().unique().tolist()
    impact_cells = compute_impact_zone(
        unscored_h3_cells=unscored_h3_cells,
        h3_levels=h3_levels,
        neighbour_rings=neighbour_rings,
    )
    logger.info(f"[update/{domain_label}] Impact zone: {len(impact_cells):,} H3 cells")

    logger.info(f"[update/{domain_label}] Loading impact-zone embeddings from cache ...")
    affected_df, embed_cols = load_affected_embeddings_from_cache(
        embeddings_db_path=str(embeddings_db_path),
        impact_cells=impact_cells,
        h3_levels=h3_levels,
        restrict_model_hash=restrict_model_hash,
    )
    if affected_df.empty:
        logger.warning(f"[update/{domain_label}] No embeddings found in impact zone.")
        return None, None, None

    rescored_ref_ids = set(affected_df["ref_id"].astype(str).unique())
    logger.info(
        f"[update/{domain_label}] Re-scoring {len(affected_df):,} samples "
        f"across {len(rescored_ref_ids):,} ref_ids"
    )
    return affected_df, embed_cols, rescored_ref_ids


def _run_scoring_update(
    *,
    embeddings_db_path: Path,
    class_mappings: dict,
    long_parquet_dir: Path,
    lc10_config: AnomalyRunConfig,
    cty24_config: AnomalyRunConfig,
    neighbour_rings: int,
    output_review_dir_path: Optional[Path],
    restrict_model_hash: Optional[str],
    debug: bool,
    parquet_glob: str = "**/*.parquet",
) -> Tuple[pd.DataFrame, set]:
    """Incrementally re-score only geographic slices affected by new data.

    Returns (merged_scores_df, rescored_ref_ids).

    How this differs from rerun
    ---------------------------
    Instead of calling ``run_pipeline`` pointing at the full DuckDB cache
    (which loads and scores millions of embeddings), we:

    1. For **each domain separately** (LC10, CTY24), scan the output long
       parquets checking only that domain's 3 anomaly columns.  This avoids
       incorrectly flagging rows that are NaN only because their label is a
       skip_class in the *other* domain (e.g. "ignore" for CROPTYPE24).
    2. For each domain, look up the H3 cells of genuinely unscored samples,
       expand by *neighbour_rings*, and load only those embeddings.
    3. Score each domain with its own (smaller) impact-zone embeddings.
    4. Outer-merge the two sets of scores.

    Result: only the affected slices are recomputed per domain; other slices
    keep their existing scores.  Memory usage is dramatically lower because
    each domain loads only the embeddings it actually needs.
    """

    # Resolve skip-class ewoc_codes for each domain so we can exclude them
    # from the unscored set before computing the impact zone.
    lc10_skip_codes = _resolve_skip_ewoc_codes(
        class_mappings, lc10_config.class_mappings_name, lc10_config.skip_classes,
    )
    cty24_skip_codes = _resolve_skip_ewoc_codes(
        class_mappings, cty24_config.class_mappings_name, cty24_config.skip_classes,
    )
    logger.info(f"[update] Skip-class ewoc_codes — LC10: {len(lc10_skip_codes):,}, "
                f"CTY24: {len(cty24_skip_codes):,}")

    # ----- LC10 domain -----
    logger.info("[update] === LANDCOVER10 domain ===")
    lc10_result = _discover_domain_impact(
        domain_label="LC10",
        domain_anomaly_cols=list(LC10_ANOMALY_COLUMNS),
        h3_levels=sorted(lc10_config.h3_levels),
        long_parquet_dir=long_parquet_dir,
        embeddings_db_path=embeddings_db_path,
        neighbour_rings=neighbour_rings,
        restrict_model_hash=restrict_model_hash,
        skip_ewoc_codes=lc10_skip_codes,
        parquet_glob=parquet_glob,
    )
    lc10_affected_df, lc10_embed_cols, lc10_ref_ids = lc10_result

    lc10_df = pd.DataFrame()
    if lc10_affected_df is not None:
        lc10_out = str(output_review_dir_path / "LC10_update_flagged.parquet") if output_review_dir_path else None
        lc10_sum = str(output_review_dir_path / "LC10_update_summary.parquet") if output_review_dir_path else None

        logger.info("[update] Running LANDCOVER10 scoring (impact zone) ...")
        lc10_df = run_single_domain_scoring(
            embeddings_db_path=str(embeddings_db_path),
            config=lc10_config,
            class_mappings=class_mappings,
            embeddings_df=(lc10_affected_df, lc10_embed_cols),
            output_samples_path=lc10_out,
            output_summary_path=lc10_sum,
            write_outputs=output_review_dir_path is not None,
            overwrite=True,
            restrict_model_hash=restrict_model_hash,
            debug=debug,
        )
        # Free memory — the embeddings DataFrame can be large
        del lc10_affected_df
        gc.collect()
    else:
        lc10_ref_ids = set()

    # ----- CTY24 domain -----
    logger.info("[update] === CROPTYPE24 domain ===")
    cty24_result = _discover_domain_impact(
        domain_label="CTY24",
        domain_anomaly_cols=list(CTY24_ANOMALY_COLUMNS),
        h3_levels=sorted(cty24_config.h3_levels),
        long_parquet_dir=long_parquet_dir,
        embeddings_db_path=embeddings_db_path,
        neighbour_rings=neighbour_rings,
        restrict_model_hash=restrict_model_hash,
        skip_ewoc_codes=cty24_skip_codes,
        parquet_glob=parquet_glob,
    )
    cty24_affected_df, cty24_embed_cols, cty24_ref_ids = cty24_result

    cty24_df = pd.DataFrame()
    if cty24_affected_df is not None:
        cty24_out = str(output_review_dir_path / "CTY24_update_flagged.parquet") if output_review_dir_path else None
        cty24_sum = str(output_review_dir_path / "CTY24_update_summary.parquet") if output_review_dir_path else None

        logger.info("[update] Running CROPTYPE24 scoring (impact zone) ...")
        cty24_df = run_single_domain_scoring(
            embeddings_db_path=str(embeddings_db_path),
            config=cty24_config,
            class_mappings=class_mappings,
            embeddings_df=(cty24_affected_df, cty24_embed_cols),
            output_samples_path=cty24_out,
            output_summary_path=cty24_sum,
            write_outputs=output_review_dir_path is not None,
            overwrite=True,
            restrict_model_hash=restrict_model_hash,
            debug=debug,
        )
        del cty24_affected_df
    else:
        cty24_ref_ids = set()

    # ----- Merge results -----
    all_rescored_ref_ids = (lc10_ref_ids or set()) | (cty24_ref_ids or set())

    if lc10_df.empty and cty24_df.empty:
        logger.info("[update] No unscored samples found in either domain — nothing to do.")
        return pd.DataFrame(), set()

    if lc10_df.empty:
        merged = cty24_df
    elif cty24_df.empty:
        merged = lc10_df
    else:
        merged = cty24_df.merge(lc10_df, on=["ref_id", "sample_id"], how="outer")

    logger.info(f"[update] Merged scores: {len(merged):,} rows "
                f"(LC10: {len(lc10_df):,}, CTY24: {len(cty24_df):,})")
    return merged, all_rescored_ref_ids


# ===========================================================================
# Orchestrator
# ===========================================================================


def compute_anomaly_scores(
    *,
    mode: Literal["rerun", "update"],
    # --- I/O paths ---
    input_long_dir: Path,
    suffix: str,
    wide_dir: Optional[Path],
    merged_wide_path: Optional[Path],
    embeddings_db_path: Optional[Path],
    review_dir: Optional[Path],
    output_long_dir: Optional[Path],
    output_wide_path: Optional[Path],
    parquet_glob: str,
    # --- Presto ---
    presto_url_or_path: str,
    # --- Configs ---
    process_cfg: ParquetProcessConfig,
    merge_cfg: MergeConfig,
    emb_cfg: EmbeddingsConfig,
    lc10_config: AnomalyRunConfig,
    cty24_config: AnomalyRunConfig,
    # --- Class mappings ---
    class_mappings_json: Optional[Path],
    sp_env_file: Optional[Path],
    # --- Incremental-specific ---
    neighbour_rings: int,
    # --- Overwrite / skip flags ---
    overwrite_wide: bool,
    overwrite_merged: bool,
    overwrite_scores: bool,
    overwrite_merged_scores: bool,
    overwrite_wide_scores: bool,
    skip_embeddings: bool,
    skip_scoring: bool,
    skip_write_back: bool,
) -> None:
    """Run the full anomaly scoring pipeline end to end."""

    # ------------------------------------------------------------------
    # Derive default paths
    # ------------------------------------------------------------------
    input_parent = input_long_dir.parent
    input_stem = input_long_dir.stem if input_long_dir.is_file() else input_long_dir.name

    effective_wide_dir = wide_dir or (
        input_parent / f"cached_wide_merged/cached_wide_parquets{suffix}"
    )
    effective_merged_wide = merged_wide_path or (
        input_parent / f"cached_wide_merged/{input_stem}_wide_{process_cfg.freq}{suffix}.parquet"
    )
    effective_db = embeddings_db_path or (
        input_parent / f"cached_embeddings/embeddings_cache_LANDCOVER10_updated{suffix}.duckdb"
    )
    effective_review = review_dir or (input_parent / f"outlier_scores{suffix}")

    lc10_base = f"h3levels23_10kmax_mad4_jsonLC10{suffix}"
    cty24_base = f"h3levels234_5kmax_mad4_jsonCTY24{suffix}"
    lc10_dir = effective_review / lc10_base
    cty24_dir = effective_review / cty24_base
    lc10_samples = lc10_dir / f"{lc10_base}.parquet"
    lc10_summary = lc10_dir / f"{lc10_base}_summary.parquet"
    cty24_samples = cty24_dir / f"{cty24_base}.parquet"
    cty24_summary = cty24_dir / f"{cty24_base}_summary.parquet"
    merged_scores_path = effective_review / f"merged_LC10_CTY24_flagged_gdf{suffix}.parquet"

    effective_output_long = output_long_dir or (
        input_parent / f"{input_stem}_with_anomalies{suffix}.parquet"
    )
    effective_output_wide = output_wide_path or (
        effective_merged_wide.parent
        / f"{effective_merged_wide.stem}_with_anomalies.parquet"
    )

    # Embed config with resolved DB path
    effective_emb_cfg = EmbeddingsConfig(
        presto_url_or_path=presto_url_or_path,
        embeddings_db_path=effective_db,
        batch_size=emb_cfg.batch_size,
        num_workers=emb_cfg.num_workers,
        force_recompute=emb_cfg.force_recompute,
        prematch=emb_cfg.prematch,
        show_progress=emb_cfg.show_progress,
        parquet_batch_rows=emb_cfg.parquet_batch_rows,
    )

    logger.info("=" * 70)
    logger.info(f"compute_anomaly_scores  mode={mode!r}  suffix={suffix!r}")
    logger.info(f"  input_long_dir     : {input_long_dir}")
    logger.info(f"  wide_dir           : {effective_wide_dir}")
    logger.info(f"  merged_wide_path   : {effective_merged_wide}")
    logger.info(f"  embeddings_db_path : {effective_db}")
    logger.info(f"  review_dir         : {effective_review}")
    logger.info(f"  output_long_dir    : {effective_output_long}")
    logger.info(f"  output_wide_path   : {effective_output_wide}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Steps 1–4: Embeddings pipeline
    # ------------------------------------------------------------------
    if not skip_embeddings:
        logger.info("[1/10] Discovering long-format parquets ...")
        raw_files = discover_parquets(input_long_dir, pattern=parquet_glob)
        if not raw_files:
            raise RuntimeError(
                f"No parquet files found under: {input_long_dir} (pattern={parquet_glob})"
            )
        logger.info(f"  found {len(raw_files)} files")

        logger.info("[2/10] Converting long → wide ...")
        wide_files, empty_files, errored_files, wrote_any = long_to_wide_parquets(
            raw_files, wide_dir=effective_wide_dir, cfg=process_cfg, overwrite=overwrite_wide,
        )
        logger.info(
            f"  wide={len(wide_files)} empty={len(empty_files)} errored={len(errored_files)}"
        )

        logger.info("[3/10] Merging wide parquets ...")
        wide_files = [p for p in wide_files if p.exists()]
        if not wide_files:
            raise RuntimeError("No wide parquet files available for merging.")
        merged_wide = merge_parquets_stream_to_one(
            files=wide_files, out_path=effective_merged_wide, cfg=merge_cfg,
            overwrite=overwrite_merged or wrote_any,
        )

        logger.info("[4/10] Populating embeddings cache ...")
        init_cache(str(effective_db))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  device={device}")
        model = load_presto_model_compat(presto_url_or_path)
        model.eval().to(device)
        model_hash = get_model_hash(model)
        logger.info(f"  model_hash={model_hash}")
        compute_embeddings_from_merged_parquet(
            merged_wide_parquet=merged_wide, model=model,
            model_hash=model_hash, cfg=effective_emb_cfg,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("[2–4/10] Skipping embeddings steps (--skip-embeddings)")

    # ------------------------------------------------------------------
    # Steps 5–8: Scoring
    # ------------------------------------------------------------------
    rescored_ref_ids: Optional[set] = None  # None = rerun mode (all files)

    if skip_scoring:
        logger.info("[5–8/10] Skipping scoring (--skip-scoring)")
        if not skip_write_back:
            if not merged_scores_path.exists():
                raise FileNotFoundError(
                    f"--skip-scoring but merged scores not found: {merged_scores_path}"
                )
            merged_scores = pd.read_parquet(str(merged_scores_path))
        else:
            return
    else:
        logger.info("[5/10] Loading class mappings ...")
        class_mappings = load_class_mappings(
            class_mappings_json=class_mappings_json,
            sp_env_file=sp_env_file,
        )
        lc10_dir.mkdir(parents=True, exist_ok=True)
        cty24_dir.mkdir(parents=True, exist_ok=True)

        if mode == "rerun":
            logger.info("[6–8/10] Scoring mode: rerun (full DuckDB cache) ...")
            merged_scores = _run_scoring_rerun(
                embeddings_db_path=effective_db,
                class_mappings=class_mappings,
                lc10_samples_path=lc10_samples,
                lc10_summary_path=lc10_summary,
                cty24_samples_path=cty24_samples,
                cty24_summary_path=cty24_summary,
                merged_scores_path=merged_scores_path,
                lc10_config=lc10_config,
                cty24_config=cty24_config,
                overwrite_scores=overwrite_scores,
                overwrite_merged_scores=overwrite_merged_scores,
                write_review_parquets=True,
                restrict_model_hash=None,
                debug=False,
            )
            # rescored_ref_ids stays None → write_scores_to_long_parquets rewrites all

        elif mode == "update":
            logger.info("[6–8/10] Scoring mode: update (impact zone only) ...")

            in_place = input_long_dir.resolve() == effective_output_long.resolve()

            if in_place:
                # Geoparquet mode: files are updated in-place so we scan the input
                # dir directly — every file already has anomaly columns (or is missing
                # them entirely for newly added datasets).  No stub files needed.
                logger.info("[update] In-place mode — scanning input dir for unscored files.")
                scan_dir = input_long_dir
            else:
                # Hive-partitioned mode: write NaN-stub files for any input parquets
                # that have no corresponding output file yet (brand-new ref_ids arriving
                # for the first time).  This makes them visible to find_unscored_samples.
                if effective_output_long.exists():
                    out_ref_ids = {
                        p.parent.name.split("=", 1)[1]
                        for p in effective_output_long.glob(parquet_glob)
                        if "ref_id=" in p.parent.name
                    }
                    for src_pf in sorted(input_long_dir.glob(parquet_glob)):
                        src_ref_id = None
                        for part in src_pf.parts:
                            if part.startswith("ref_id="):
                                src_ref_id = part.split("=", 1)[1]
                                break
                        if src_ref_id is None:
                            src_ref_id = src_pf.stem
                        if src_ref_id not in out_ref_ids:
                            stub_path = effective_output_long / src_pf.relative_to(input_long_dir)
                            stub_path.parent.mkdir(parents=True, exist_ok=True)
                            df_stub = pd.read_parquet(src_pf)
                            for col in ANOMALY_COLUMNS:
                                df_stub[col] = float("nan")
                            df_stub.to_parquet(stub_path, index=False)
                            logger.info(
                                f"[update] Wrote NaN stub for new ref_id: {src_ref_id} "
                                f"({len(df_stub):,} rows)"
                            )
                # Scan the scored output dir for NaN anomaly rows
                scan_dir = effective_output_long if effective_output_long.exists() else input_long_dir
            merged_scores, rescored_ref_ids = _run_scoring_update(
                embeddings_db_path=effective_db,
                class_mappings=class_mappings,
                long_parquet_dir=scan_dir,
                lc10_config=lc10_config,
                cty24_config=cty24_config,
                neighbour_rings=neighbour_rings,
                output_review_dir_path=effective_review,
                restrict_model_hash=None,
                debug=False,
                parquet_glob=parquet_glob,
            )
            if merged_scores.empty:
                logger.info("No re-scoring needed — all parquets are already up to date.")
                return
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Must be 'rerun' or 'update'.")

    if skip_write_back:
        logger.info("[9–10/10] Skipping write-back (--skip-write-back)")
        return

    anomaly_cols = [c for c in ANOMALY_COLUMNS if c in merged_scores.columns]
    if not anomaly_cols:
        anomaly_cols = list(merged_scores.columns.difference(["ref_id", "sample_id"]))

    # ------------------------------------------------------------------
    # Step 9 – Write back to long parquets
    # ------------------------------------------------------------------
    logger.info("[9/10] Writing scores to long-format parquets ...")
    n_written = write_scores_to_long_parquets(
        merged_scores=merged_scores,
        input_long_dir=input_long_dir,
        output_long_dir=effective_output_long,
        anomaly_cols=anomaly_cols,
        parquet_glob=parquet_glob,
        only_affected_ref_ids=rescored_ref_ids,  # None = write all; set = write only affected
    )
    logger.info(f"  wrote {n_written} files → {effective_output_long}")

    # ------------------------------------------------------------------
    # Step 10 – Write back to merged wide parquet
    # ------------------------------------------------------------------
    logger.info("[10/10] Writing scores to merged wide parquet ...")
    if not effective_merged_wide.exists():
        logger.warning(
            f"Merged wide parquet not found ({effective_merged_wide}); skipping wide write-back."
        )
    else:
        write_scores_to_wide_parquet(
            merged_scores=merged_scores,
            src_wide_path=effective_merged_wide,
            output_wide_path=effective_output_wide,
            anomaly_cols=anomaly_cols,
            batch_rows=merge_cfg.batch_rows,
            row_group_size=merge_cfg.row_group_size,
            overwrite=overwrite_wide_scores,
        )

    logger.info("Pipeline complete.")


# ===========================================================================
# CLI
# ===========================================================================


def _parse_max_timesteps_trim(v: str) -> int | str | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if s.lower() in {"none", "null"}:
        return None
    if s.lower() == "auto":
        return "auto"
    try:
        return int(s)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            "--max-timesteps-trim must be an int, 'auto', or 'None'"
        ) from e


def _default_input_dir() -> Path | None:
    for c in [
        Path("/home/vito/shahs/projects/worldcereal/data/worldcereal_all_extractions.parquet"),
        Path("/projects/worldcereal/data/worldcereal_all_extractions.parquet"),
    ]:
        if c.exists():
            return c
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Full end-to-end pipeline: long-format parquets → Presto embeddings\n"
            "→ outlier scores → write back.\n\n"
            "  --mode rerun   Full scoring of all points from DuckDB (default)\n"
            "  --mode update  Incremental: only re-score slices affected by new data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    default_in = _default_input_dir()

    # Mode
    p.add_argument(
        "--mode",
        choices=["rerun", "update"],
        default="rerun",
        help=(
            "'rerun' = full scoring of all embeddings in DuckDB (default); "
            "'update' = incremental re-scoring of the geographic impact zone only."
        ),
    )

    # Input format
    p.add_argument(
        "--input-format",
        choices=["parquet", "geoparquet"],
        default="parquet",
        help=(
            "'parquet' (default): nested hive-partitioned .parquet dataset (HPC layout). "
            "'geoparquet': flat folder of per-dataset .geoparquet files (VM layout). "
            "In 'geoparquet' mode scores are written back IN-PLACE, the output long dir "
            "defaults to the same as the input, and geo metadata is preserved."
        ),
    )

    # Core I/O
    p.add_argument(
        "--input-long-dir", type=Path, default=default_in, required=default_in is None,
        help=(
            "Root folder with raw long-format parquet files (or a single parquet file). "
            "In 'update' mode this is the ORIGINAL extractions dir (not the scored output)."
        ),
    )
    p.add_argument(
        "--suffix", type=str, default="",
        help="Appended to all output dir/file names for side-by-side runs (e.g. '_v2').",
    )
    p.add_argument(
        "--parquet-glob", type=str, default=None,
        help=(
            "Glob pattern to find input files. "
            "Defaults to '*.geoparquet' when --input-format=geoparquet, "
            "otherwise '**/*.parquet'."
        ),
    )

    # Optional path overrides (all derived from --input-long-dir + --suffix by default)
    p.add_argument("--wide-dir",           type=Path, default=None)
    p.add_argument("--merged-wide-path",   type=Path, default=None)
    p.add_argument("--embeddings-db-path", type=Path, default=None)
    p.add_argument(
        "--review-dir", type=Path, default=None,
        help="Root dir for per-domain scored parquets and merged scores.",
    )
    p.add_argument("--output-long-dir",  type=Path, default=None)
    p.add_argument("--output-wide-path", type=Path, default=None)

    # Presto
    p.add_argument("--presto-url-or-path", type=str, default=_DEFAULT_PRESTO_URL)

    # process_parquet
    p.add_argument("--freq", choices=["month", "dekad"], default="month")
    p.add_argument("--required-min-timesteps", type=int, default=None)
    p.add_argument("--use-valid-time",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-edge-buffer", type=int, default=1)
    p.add_argument("--max-timesteps-trim", default="18")
    p.add_argument("--wide-engine",      choices=["fastparquet", "pyarrow"], default="fastparquet")
    p.add_argument("--wide-compression", type=str, default="snappy")

    # merge
    p.add_argument("--merge-batch-rows",     type=int, default=100_000)
    p.add_argument("--merge-row-group-size", type=int, default=100_000)
    p.add_argument("--merge-compression",    type=str, default="zstd")

    # embeddings
    p.add_argument("--batch-size",         type=int, default=4096)
    p.add_argument("--num-workers",        type=int, default=2)
    p.add_argument("--parquet-batch-rows", type=int, default=100_000)
    p.add_argument("--force-recompute",    action="store_true")
    p.add_argument("--prematch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-progress", action="store_true")

    # Class mappings
    p.add_argument(
        "--class-mappings-json", type=Path, default=None,
        help="Local JSON class mappings file (bypasses SharePoint).",
    )
    p.add_argument(
        "--sp-env-file", type=Path, default=Path.home() / ".sharepointenv",
        help="Dotenv file with SharePoint credentials.",
    )

    # LC10 scoring
    lc10 = p.add_argument_group("LANDCOVER10 scoring")
    lc10.add_argument("--lc10-h3-levels",           type=int, nargs="+", default=[2, 3])
    lc10.add_argument("--lc10-min-slice-size",       type=int, default=200)
    lc10.add_argument("--lc10-max-slice-size",       type=int, default=10_000)
    lc10.add_argument("--lc10-max-merge-iterations", type=int, default=16)
    lc10.add_argument("--lc10-mad-k",                type=float, default=4.0)

    # CTY24 scoring
    cty = p.add_argument_group("CROPTYPE24 scoring")
    cty.add_argument("--cty24-h3-levels",            type=int, nargs="+", default=[2, 3, 4])
    cty.add_argument("--cty24-min-slice-size",        type=int, default=100)
    cty.add_argument("--cty24-max-slice-size",        type=int, default=5_000)
    cty.add_argument("--cty24-max-merge-iterations",  type=int, default=8)
    cty.add_argument("--cty24-mad-k",                 type=float, default=4.0)

    # Common scoring
    common = p.add_argument_group("common scoring")
    common.add_argument("--threshold-mode",      type=str, default="mad")
    common.add_argument("--percentile-q",        type=float, default=0.96)
    common.add_argument("--norm-percentiles",    type=float, nargs=2, default=[2.0, 98.0],
                        metavar=("LO", "HI"))
    common.add_argument("--fdr-alpha",           type=float, default=0.05)
    common.add_argument("--skip-classes",        type=str, nargs="+", default=["ignore"])
    common.add_argument("--max-full-pairwise-n", type=int, default=0)

    # Update-specific
    upd = p.add_argument_group("update mode")
    upd.add_argument(
        "--neighbour-rings", type=int, default=1,
        help="H3 neighbour rings added around the new-data impact zone (update mode only).",
    )

    # Overwrite flags
    ow = p.add_argument_group("overwrite flags")
    ow.add_argument("--overwrite-wide",          action="store_true",
                    help="Re-run long→wide even if per-file outputs exist.")
    ow.add_argument("--overwrite-merged",        action="store_true",
                    help="Re-build the merged wide parquet.")
    ow.add_argument("--overwrite-scores",        action="store_true",
                    help="Re-run LC10/CTY24 pipelines even if output parquets exist.")
    ow.add_argument("--overwrite-merged-scores", action="store_true",
                    help="Re-merge LC10+CTY24 even if merged scores parquet exists.")
    ow.add_argument("--overwrite-wide-scores",   action="store_true",
                    help="Re-write the wide parquet with anomaly scores.")

    # Skip flags
    sk = p.add_argument_group("skip flags")
    sk.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip steps 2–4 (long→wide, merge, embeddings). Cache must already be up-to-date.",
    )
    sk.add_argument(
        "--skip-scoring", action="store_true",
        help="Skip steps 5–8. Merged scores parquet must already exist on disk.",
    )
    sk.add_argument(
        "--skip-write-back", action="store_true",
        help="Skip steps 9–10 (writing scores back to long/wide parquets).",
    )

    p.add_argument("--log-level", type=str, default="INFO")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    # Resolve parquet glob default from input format
    if args.parquet_glob is not None:
        parquet_glob = args.parquet_glob
    elif args.input_format == "geoparquet":
        parquet_glob = "*.geoparquet"
    else:
        parquet_glob = "**/*.parquet"

    # In geoparquet mode: in-place update — output dir == input dir by default
    output_long_dir = args.output_long_dir
    if output_long_dir is None and args.input_format == "geoparquet":
        output_long_dir = args.input_long_dir

    process_cfg = ParquetProcessConfig(
        freq=args.freq,
        required_min_timesteps=args.required_min_timesteps,
        use_valid_time=bool(args.use_valid_time),
        min_edge_buffer=args.min_edge_buffer,
        max_timesteps_trim=_parse_max_timesteps_trim(args.max_timesteps_trim),
        engine=args.wide_engine,
        compression=args.wide_compression,
    )
    merge_cfg = MergeConfig(
        batch_rows=args.merge_batch_rows,
        row_group_size=args.merge_row_group_size,
        compression=args.merge_compression,
    )
    # EmbeddingsConfig: db_path is a placeholder, resolved inside the orchestrator
    emb_cfg = EmbeddingsConfig(
        presto_url_or_path=args.presto_url_or_path,
        embeddings_db_path=Path("/dev/null"),  # overridden inside compute_anomaly_scores()
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_recompute=bool(args.force_recompute),
        prematch=bool(args.prematch),
        show_progress=not bool(args.no_progress),
        parquet_batch_rows=args.parquet_batch_rows,
    )

    common_kw = dict(
        threshold_mode=args.threshold_mode,
        percentile_q=args.percentile_q,
        norm_percentiles=tuple(args.norm_percentiles),
        skip_classes=args.skip_classes,
        max_full_pairwise_n=args.max_full_pairwise_n,
    )
    lc10_config = AnomalyRunConfig(
        label_domain="LANDCOVER10",
        class_mappings_name="LANDCOVER10",
        h3_levels=args.lc10_h3_levels,
        min_slice_size=args.lc10_min_slice_size,
        max_slice_size=args.lc10_max_slice_size,
        max_merge_iterations=args.lc10_max_merge_iterations,
        mad_k=args.lc10_mad_k,
        confidence_col_name="LC10_confidence_nonoutlier",
        anomaly_flag_col_name="LC10_anomaly_flag",
        label_col_rename="outlier_LC10_cls",
        **common_kw,
    )
    cty24_config = AnomalyRunConfig(
        label_domain="CROPTYPE24",
        class_mappings_name="CROPTYPE24",
        h3_levels=args.cty24_h3_levels,
        min_slice_size=args.cty24_min_slice_size,
        max_slice_size=args.cty24_max_slice_size,
        max_merge_iterations=args.cty24_max_merge_iterations,
        mad_k=args.cty24_mad_k,
        confidence_col_name="CTY24_confidence_nonoutlier",
        anomaly_flag_col_name="CTY24_anomaly_flag",
        label_col_rename="outlier_CTY24_cls",
        **common_kw,
    )

    compute_anomaly_scores(
        mode=args.mode,
        input_long_dir=args.input_long_dir,
        suffix=args.suffix,
        wide_dir=args.wide_dir,
        merged_wide_path=args.merged_wide_path,
        embeddings_db_path=args.embeddings_db_path,
        review_dir=args.review_dir,
        output_long_dir=output_long_dir,
        output_wide_path=args.output_wide_path,
        parquet_glob=parquet_glob,
        presto_url_or_path=args.presto_url_or_path,
        process_cfg=process_cfg,
        merge_cfg=merge_cfg,
        emb_cfg=emb_cfg,
        lc10_config=lc10_config,
        cty24_config=cty24_config,
        class_mappings_json=args.class_mappings_json,
        sp_env_file=args.sp_env_file,
        neighbour_rings=args.neighbour_rings,
        overwrite_wide=bool(args.overwrite_wide),
        overwrite_merged=bool(args.overwrite_merged),
        overwrite_scores=bool(args.overwrite_scores),
        overwrite_merged_scores=bool(args.overwrite_merged_scores),
        overwrite_wide_scores=bool(args.overwrite_wide_scores),
        skip_embeddings=bool(args.skip_embeddings),
        skip_scoring=bool(args.skip_scoring),
        skip_write_back=bool(args.skip_write_back),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
