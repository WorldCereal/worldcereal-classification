"""DuckDB-backed cache for Presto embedding vectors.

Embeddings are stored in a wide table with columns:
    sample_id VARCHAR
    model_hash VARCHAR
    ref_id VARCHAR (optional, if present in batch dataframe)
    ewoc_code VARCHAR (optional)
    h3_l3_cell VARCHAR (optional)
    embedding_0 FLOAT .. embedding_127 FLOAT

Keyed logically by (sample_id, model_hash). We do not enforce a physical
PRIMARY KEY constraint (DuckDB support is limited) but we avoid duplicates
insertion in code. A deterministic model hash (SHA256 over ordered state_dict
tensor bytes) segments embeddings per model variant.
"""

import hashlib
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

import duckdb
import h3
import numpy as np
import pandas as pd
import torch
from loguru import logger
from prometheo.models import Presto
from prometheo.predictors import Predictors
from prometheo.utils import device
from torch.utils.data import DataLoader, default_collate
from tqdm.auto import tqdm

from worldcereal.train.datasets import WorldCerealTrainingDataset

EMBED_DIM = 128
EMBED_COLS = [f"embedding_{i}" for i in range(EMBED_DIM)]


def load_presto_model(url_or_path: str):
    """Load a Presto checkpoint from a URL or local path.

    The classifier head (if present) is removed and the model is moved to GPU if
    available, otherwise CPU.
    """

    from prometheo.models import Presto
    from prometheo.models.presto.wrapper import load_presto_weights

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Presto()
    model = load_presto_weights(model, url_or_path).to(device)
    model.eval().to(target_device)

    return model


def init_cache(db_path: str) -> duckdb.DuckDBPyConnection:
    """Create / open the DuckDB database and ensure the embeddings table exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    # Create table if not exists
    columns_sql = ",\n        ".join([f"embedding_{i} FLOAT" for i in range(EMBED_DIM)])
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS embeddings_cache (
            sample_id VARCHAR,
            model_hash VARCHAR,
            ref_id VARCHAR,
            ewoc_code VARCHAR,
            h3_l3_cell VARCHAR,
            lat FLOAT,
            lon FLOAT,
            {columns_sql}
        );
        """
    )
    return con


def get_model_hash(model: torch.nn.Module) -> str:
    """Deterministic SHA256 hash of a model's state_dict.

    Each tensor is cast to float32, moved to CPU, converted to contiguous bytes.
    We update the digest with the parameter name followed by its bytes to avoid
    accidental collisions should two tensors share raw byte sequences.
    """
    h = hashlib.sha256()
    state = model.state_dict()
    for k in sorted(state.keys()):
        h.update(k.encode("utf-8"))
        t = state[k]
        if isinstance(t, torch.Tensor):
            t_bytes = t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
            h.update(t_bytes)
    digest = h.hexdigest()
    logger.debug(f"Computed model hash: {digest}")
    return digest


def list_cached_ids(db_path: str, model_hash: str) -> Set[str]:
    """Return set of sample_ids cached for a given model hash.

    Ensures the cache table exists prior to querying.
    """
    con = init_cache(db_path)
    df = con.execute(
        "SELECT sample_id FROM embeddings_cache WHERE model_hash = ?", [model_hash]
    ).fetchdf()
    return set(df.sample_id.tolist()) if not df.empty else set()


def delete_cached_ids(db_path: str, model_hash: str, sample_ids: Iterable[str]) -> None:
    """Delete cached embeddings for provided sample_ids (force recompute path)."""
    sample_ids = list(sample_ids)
    if not sample_ids:
        return
    con = init_cache(db_path)
    df_ids = pd.DataFrame({"sample_id": sample_ids})
    con.register("to_delete", df_ids)
    con.execute(
        "DELETE FROM embeddings_cache WHERE model_hash = ? AND sample_id IN (SELECT sample_id FROM to_delete)",
        [model_hash],
    )
    con.execute("CHECKPOINT")
    logger.info(f"Deleted {len(sample_ids)} cached embeddings for model {model_hash}")


def _wide_batch_dataframe(df_batch: pd.DataFrame, model_hash: str) -> pd.DataFrame:
    """Expand a batch dataframe with 'embedding' arrays into wide float columns."""
    if "embedding" not in df_batch.columns:
        raise ValueError("df_batch must contain an 'embedding' column")
    emb_matrix = np.vstack(df_batch["embedding"].to_numpy()).astype(np.float32)
    if emb_matrix.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Expected embedding dim {EMBED_DIM}, got {emb_matrix.shape[1]}"
        )

    wide_df = pd.DataFrame(emb_matrix, columns=EMBED_COLS)
    meta_cols = [
        c
        for c in ["sample_id", "ref_id", "ewoc_code", "h3_l3_cell", "lat", "lon"]
        if c in df_batch.columns
    ]
    for c in meta_cols:
        wide_df[c] = df_batch[c].values
    wide_df["model_hash"] = model_hash
    return wide_df[
        ["sample_id", "model_hash", "ref_id", "ewoc_code", "h3_l3_cell", "lat", "lon"]
        + EMBED_COLS
    ]


def insert_embeddings(
    df_batch: pd.DataFrame,
    model_hash: str,
    db_path: str,
    existing_ids: Set[str] | None = None,
) -> int:
    """Insert a batch of embeddings (skip already cached sample_ids).

    Parameters
    ----------
    df_batch : pd.DataFrame
        Must contain columns: 'sample_id' and 'embedding' plus optional metadata.
    model_hash : str
        Hash identifying the model weights.
    db_path : str
        Path to DuckDB database file.
    existing_ids : Optional[Set[str]]
        If provided, treated as authoritative set of already cached sample_ids
        for this model (will be updated in-place). Avoids repeated DB queries
        for large streaming insertion loops.

    Returns
    -------
    int
        Number of newly inserted rows.
    """
    if df_batch.empty:
        return 0
    con = init_cache(db_path)
    if existing_ids is None:
        existing_ids = list_cached_ids(db_path, model_hash)
    batch_ids = df_batch["sample_id"].astype(str)
    new_mask = ~batch_ids.isin(existing_ids)
    new_df = df_batch.loc[new_mask].copy()
    if new_df.empty:
        return 0
    wide_df = _wide_batch_dataframe(new_df, model_hash)
    con.register("emb_batch", wide_df)
    con.execute("INSERT INTO embeddings_cache SELECT * FROM emb_batch")
    # Periodic checkpoint after insertion (DuckDB may auto-checkpoint but we enforce)
    con.execute("CHECKPOINT")
    inserted = wide_df.shape[0]
    if inserted:
        logger.debug(
            f"Inserted {inserted} embeddings into cache for model {model_hash}"
        )
        if existing_ids is not None:
            existing_ids.update(new_df["sample_id"].astype(str))
    return inserted


def fetch_embeddings(
    db_path: str, model_hash: str, sample_ids: Iterable[str]
) -> pd.DataFrame:
    """Fetch wide embeddings for provided sample_ids; returns DataFrame.

    Ensures table exists; returns empty frame if none cached.
    """
    sample_ids = list(sample_ids)
    if not sample_ids:
        return pd.DataFrame()
    con = init_cache(db_path)
    df_ids = pd.DataFrame({"sample_id": sample_ids})
    con.register("requested", df_ids)
    cols_select = ", ".join(
        [
            f"e.{c}"
            for c in ["sample_id", "ref_id", "ewoc_code", "h3_l3_cell", "lat", "lon"]
            if c
        ]
    )
    emb_select = ", ".join([f"e.embedding_{i}" for i in range(EMBED_DIM)])
    query = f"""
        SELECT {cols_select}, {emb_select}
        FROM embeddings_cache e
        INNER JOIN requested r USING(sample_id)
        WHERE e.model_hash = ?
    """
    df = con.execute(query, [model_hash]).fetchdf()
    return df


def rehydrate_embedding_vectors(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide embedding columns back into vector form in 'embedding'."""
    if wide_df.empty:
        return wide_df
    emb_cols_present = [c for c in EMBED_COLS if c in wide_df.columns]
    emb_matrix = wide_df[emb_cols_present].to_numpy(dtype=np.float32)
    vectors: List[np.ndarray] = [row for row in emb_matrix]
    out = wide_df.copy()
    out["embedding"] = vectors
    return out


def _collate_predictors_with_ids(batch):
    predictors, sample_ids = zip(*batch)
    collated = default_collate([p.as_dict(ignore_nones=True) for p in predictors])
    return Predictors(**collated), list(sample_ids)


def compute_embeddings(
    data_df: pd.DataFrame,
    model: torch.nn.Module,
    batch_size: int = 2048,
    num_workers: int = 0,
    embeddings_db_path: Optional[str] = None,
    force_recompute: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute or fetch cached embeddings.

    If ``embeddings_db_path`` is None falls back to direct computation.
    Requires a 'sample_id' column when caching is enabled.
    """
    if embeddings_db_path is None:
        raise NotImplementedError(
            "Direct computation without caching is not supported in this function."
        )
    if "sample_id" not in data_df.columns:
        raise ValueError("data_df must contain a 'sample_id' column for caching")
    # ensure DB and table exist before querying cache
    init_cache(embeddings_db_path)
    model_hash = get_model_hash(model)
    requested_ids = set(data_df.sample_id.astype(str).tolist())
    cached_ids = list_cached_ids(embeddings_db_path, model_hash)
    if force_recompute:
        delete_cached_ids(embeddings_db_path, model_hash, requested_ids)
        cached_ids = set()
    missing_ids = requested_ids - cached_ids
    if missing_ids:
        logger.info(
            f"Computing {len(missing_ids)} / {len(requested_ids)} embeddings (cache miss)."
        )
        missing_df = data_df[data_df.sample_id.isin(missing_ids)].reset_index(drop=True)
        num_outputs = (
            len(sorted(missing_df.finetune_class.unique()))
            if "finetune_class" in missing_df.columns
            else 1
        )
        ds = WorldCerealTrainingDataset(
            missing_df,
            task_type="multiclass" if num_outputs > 1 else "binary",
            num_outputs=num_outputs,
            augment=False,
        )
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=_collate_predictors_with_ids,
        )
        model.eval().to(device)
        meta_cols = [
            c
            for c in ["ref_id", "ewoc_code", "h3_l3_cell", "lat", "lon"]
            if c in missing_df.columns
        ]
        meta_lookup = (
            missing_df.set_index("sample_id")[meta_cols] if meta_cols else None
        )
        existing_ids_mut = set(cached_ids)
        iterator = dl
        if show_progress:
            iterator = tqdm(
                dl,
                desc="Embedding batches",
                unit="batch",
                leave=False,
            )
        total_inserted = 0
        for predictors, attrs_list in iterator:  # type: ignore
            with torch.no_grad():
                enc = model(predictors).detach().cpu().numpy().reshape((-1, EMBED_DIM))
            batch_sample_ids = [attrs["sample_id"] for attrs in attrs_list]
            batch_df = pd.DataFrame({"sample_id": batch_sample_ids})
            if meta_lookup is not None:
                batch_df = batch_df.join(meta_lookup, on="sample_id")

            if (
                batch_df["h3_l3_cell"].isnull().any()
                or batch_df["h3_l3_cell"].eq("").any()
            ):
                print(
                    "Some sample_ids are missing h3_l3_cell metadata: computing from lat/lon"
                )
                batch_df["h3_l3_cell"] = [
                    h3.latlng_to_cell(row["lat"], row["lon"], 3)
                    for ix, row in batch_df.iterrows()
                ]

            batch_df["embedding"] = list(enc)
            inserted = insert_embeddings(
                batch_df,
                model_hash,
                embeddings_db_path,
                existing_ids=existing_ids_mut,
            )
            total_inserted += inserted
            if show_progress and tqdm is not None:
                iterator.set_postfix({"inserted": total_inserted})  # type: ignore
        logger.info(
            f"Finished embedding computation; inserted {total_inserted} new rows."
        )
    else:
        logger.info("All embeddings present in cache; skipping computation.")
    wide_df = fetch_embeddings(embeddings_db_path, model_hash, requested_ids)
    hydrated = rehydrate_embedding_vectors(wide_df)
    emb_map = {sid: emb for sid, emb in zip(hydrated.sample_id, hydrated.embedding)}
    ordered_embeddings = [emb_map[sid] for sid in data_df.sample_id.astype(str)]
    out_df = data_df.copy().reset_index(drop=True)
    out_df["embedding"] = ordered_embeddings
    return out_df


if __name__ == "__main__":
    cached_parquet = Path(
        "/projects/worldcereal/data/cached_wide_parquets/worldcereal_all_extractions_wide_month_LANDCOVER10.parquet"
    )
    model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {target_device}")
    model = Presto(pretrained_model_path=model_url)
    model.eval().to(target_device)
    embeddings_tag = "LANDCOVER10"

    compute_embeddings(
        pd.read_parquet(cached_parquet),
        model=model,
        batch_size=8192,
        num_workers=7,
        embeddings_db_path=f"/projects/worldcereal/data/cached_embeddings/embeddings_cache_{embeddings_tag}.duckdb",
        force_recompute=False,
    )
