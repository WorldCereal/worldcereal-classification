#!/usr/bin/env python3
"""Build / update a DuckDB Presto embeddings cache from raw parquet extractions.

This script extracts the reusable pipeline from
`notebooks/embeddings/explore_sample_embeddings.ipynb`:

1) Discover raw *long-format* parquet files
2) Convert each to *wide-format* parquet via `worldcereal.utils.timeseries.process_parquet`
3) Stream-merge wide parquets into one parquet file
4) Populate / update the DuckDB embeddings cache via
   `worldcereal.train.embeddings_cache.compute_embeddings`

It is designed to be repeatable and safe to re-run:
- It can skip already-processed wide parquets
- It can skip the entire pipeline if nothing new is detected
- It supports `--force-recompute` to delete/recompute cached embeddings
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence


def _ensure_worldcereal_importable() -> None:
	"""Best-effort sys.path tweak for this workspace layout.

	This script lives in `/home/vito/shahs/TestFolder/scripts/` while the package
	is in `/home/vito/shahs/TestFolder/worldcereal-classification/src/`.
	If `worldcereal` is already installed, this is a no-op.
	"""

	try:
		import worldcereal  # noqa: F401

		return
	except Exception:
		pass

	here = Path(__file__).resolve()
	candidate_src = here.parents[1] / "worldcereal-classification" / "src"
	if candidate_src.exists():
		sys.path.insert(0, str(candidate_src))


_ensure_worldcereal_importable()

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from prometheo.models import Presto  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from worldcereal.train.embeddings_cache import (  # noqa: E402
	compute_embeddings,
	get_model_hash,
	init_cache,
)
from worldcereal.utils.timeseries import process_parquet  # noqa: E402


def load_presto_model_compat(url_or_path: str) -> torch.nn.Module:
	"""Load a Presto model in the most compatible way for this environment.

	The notebook uses `Presto(pretrained_model_path=...)`, which can handle
	some checkpoint variants that `load_presto_weights()` loads strictly.
	We try that first, then fall back to the `worldcereal` helper.
	"""

	try:
		model = Presto(pretrained_model_path=url_or_path)
		model.eval()
		return model
	except Exception as e:
		logger.warning(
			f"Presto(pretrained_model_path=...) failed ({type(e).__name__}: {e}); falling back to worldcereal loader"
		)
		from worldcereal.train.embeddings_cache import load_presto_model as _load

		return _load(url_or_path)


CompareOn = Literal["ref_id", "sample_id"]


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
	num_workers: int = 8
	force_recompute: bool = False
	prematch: bool = True
	show_progress: bool = True
	parquet_batch_rows: int = 200_000


def discover_parquets(input_dir: Path, pattern: str = "**/*.parquet") -> list[Path]:
	if input_dir.is_file() and input_dir.suffix == ".parquet":
		return [input_dir]
	if not input_dir.exists():
		raise FileNotFoundError(str(input_dir))
	files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
	return files


def _wide_out_path(wide_dir: Path, raw_path: Path, suffix: str = "_ppq") -> Path:
	return wide_dir / f"{raw_path.stem}{suffix}.parquet"


def long_to_wide_parquets(
	raw_files: Sequence[Path],
	wide_dir: Path,
	cfg: ParquetProcessConfig,
	overwrite: bool = False,
) -> tuple[list[Path], list[Path], list[Path]]:
	"""Convert many long-format parquet files to wide-format parquet files."""

	wide_dir.mkdir(parents=True, exist_ok=True)

	produced: list[Path] = []
	empty: list[Path] = []
	errored: list[Path] = []

	for i, pf in enumerate(raw_files, start=1):
		out_path = _wide_out_path(wide_dir, pf)
		if out_path.exists() and not overwrite:
			logger.info(f"[{i}/{len(raw_files)}] skip existing wide parquet: {out_path.name}")
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
			logger.warning(f"[{i}/{len(raw_files)}] wide dataframe empty for {pf}")
			empty.append(pf)
			continue

		df_wide.to_parquet(
			out_path,
			engine=cfg.engine,
			compression=cfg.compression,
			index=False,
		)
		produced.append(out_path)
		logger.info(f"[{i}/{len(raw_files)}] wrote wide parquet: {out_path.name} shape={df_wide.shape}")
		del df_long, df_wide
		gc.collect()

	return produced, empty, errored


def _is_numeric(t: pa.DataType) -> bool:
	return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def _build_target_schema(files: Sequence[Path]) -> pa.Schema:
	type_map: dict[str, set[pa.DataType]] = {}
	for f in files:
		sch = pq.ParquetFile(str(f)).schema_arrow
		for field in sch:
			type_map.setdefault(field.name, set()).add(field.type)

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
	arrays = []
	for field in schema:
		name = field.name
		if name in tbl.column_names:
			col = tbl[name]
			if not col.type.equals(field.type):
				col = pc.cast(col, field.type, safe=False)
			arrays.append(col)
		else:
			arrays.append(pa.nulls(tbl.num_rows, type=field.type))
	return pa.Table.from_arrays(arrays, schema=schema)


def merge_parquets_stream_to_one(
	files: Sequence[Path],
	out_path: Path,
	cfg: MergeConfig,
	overwrite: bool = False,
) -> Path:
	"""Stream-merge many parquet files into one parquet (memory safe)."""

	if not files:
		raise ValueError("No input parquet files provided for merge.")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	if out_path.exists() and not overwrite:
		logger.info(f"skip merge; output exists: {out_path}")
		return out_path

	target_schema = _build_target_schema(files)
	logger.info(f"merge target schema fields={len(target_schema)}")

	writer = pq.ParquetWriter(
		str(out_path),
		target_schema,
		compression=cfg.compression,
		use_dictionary=True,
		write_statistics=True,
	)
	total_rows = 0
	try:
		for fi, f in enumerate(files, start=1):
			pf = pq.ParquetFile(str(f))
			logger.info(f"[{fi}/{len(files)}] merge input={f.name} rows={pf.metadata.num_rows}")
			for batch in pf.iter_batches(batch_size=cfg.batch_rows):
				tbl = pa.Table.from_batches([batch])
				tbl = _align_table_to_schema(tbl, target_schema)
				writer.write_table(tbl, row_group_size=cfg.row_group_size)
				total_rows += tbl.num_rows
				del batch, tbl
				gc.collect()
		logger.info(f"merge wrote rows={total_rows} -> {out_path}")
	finally:
		writer.close()

	return out_path


def _read_single_value_from_parquet(
	parquet_path: Path, col: str
) -> str | None:
	"""Try to read a single non-null value from a parquet column."""

	try:
		pf = pq.ParquetFile(str(parquet_path))
		for batch in pf.iter_batches(columns=[col], batch_size=1024):
			arr = batch.column(0)
			if arr.null_count == len(arr):
				continue
			# Find first non-null
			for v in arr.to_pylist():
				if v is not None and str(v) != "":
					return str(v)
		return None
	except Exception:
		return None


def _derive_ref_id_from_sample_id(sample_id: str) -> str:
	# Mirrors `worldcereal.utils.timeseries.get_ref_id` heuristic
	parts = str(sample_id).split("_")
	return "_".join(parts[:-1]) if len(parts) > 1 else str(sample_id)


def _input_ref_ids(raw_files: Sequence[Path]) -> set[str]:
	ref_ids: set[str] = set()
	for f in raw_files:
		rid = _read_single_value_from_parquet(f, "ref_id")
		if rid:
			ref_ids.add(rid)
			continue
		sid = _read_single_value_from_parquet(f, "sample_id")
		if sid:
			ref_ids.add(_derive_ref_id_from_sample_id(sid))
	return ref_ids


def _cached_ref_ids(db_path: Path, model_hash: str) -> set[str]:
	con = init_cache(str(db_path))
	df = con.execute(
		"""
		SELECT DISTINCT ref_id
		FROM embeddings_cache
		WHERE model_hash = ?
		  AND ref_id IS NOT NULL
		  AND ref_id <> ''
		""",
		[model_hash],
	).fetchdf()
	if df.empty:
		return set()
	return set(df["ref_id"].astype(str).tolist())


def _iter_sample_id_batches_from_parquet(
	parquet_path: Path, id_col: str = "sample_id", batch_rows: int = 500_000
) -> Iterator[list[str]]:
	pf = pq.ParquetFile(str(parquet_path))
	for batch in pf.iter_batches(columns=[id_col], batch_size=batch_rows):
		arr = batch.column(0)
		# drop nulls, keep strings
		ids = [str(v) for v in arr.to_pylist() if v is not None and str(v) != ""]
		if not ids:
			continue
		yield list(dict.fromkeys(ids))  # preserve order, de-dupe


def _any_missing_sample_ids(
	raw_files: Sequence[Path],
	db_path: Path,
	model_hash: str,
	id_col: str = "sample_id",
	parquet_batch_rows: int = 500_000,
	check_batch_ids: int = 200_000,
) -> bool:
	"""Return True if any input sample_id is missing in cache for this model."""

	# Avoid importing a large list at once: scan ids from parquet and check in chunks.
	con = init_cache(str(db_path))
	for f in raw_files:
		batch_buf: list[str] = []
		for ids in _iter_sample_id_batches_from_parquet(
			f, id_col=id_col, batch_rows=parquet_batch_rows
		):
			batch_buf.extend(ids)
			if len(batch_buf) < check_batch_ids:
				continue
			df_ids = pd.DataFrame({"sample_id": batch_buf})
			con.register("requested", df_ids)
			df = con.execute(
				"""
				SELECT COUNT(*) AS n_cached
				FROM requested r
				INNER JOIN embeddings_cache e
				  ON e.sample_id = r.sample_id
				WHERE e.model_hash = ?
				""",
				[model_hash],
			).fetchdf()
			n_cached = int(df.loc[0, "n_cached"]) if not df.empty else 0
			if n_cached < len(set(batch_buf)):
				return True
			batch_buf = []
		if batch_buf:
			df_ids = pd.DataFrame({"sample_id": batch_buf})
			con.register("requested", df_ids)
			df = con.execute(
				"""
				SELECT COUNT(*) AS n_cached
				FROM requested r
				INNER JOIN embeddings_cache e
				  ON e.sample_id = r.sample_id
				WHERE e.model_hash = ?
				""",
				[model_hash],
			).fetchdf()
			n_cached = int(df.loc[0, "n_cached"]) if not df.empty else 0
			if n_cached < len(set(batch_buf)):
				return True
	return False


def compute_embeddings_from_merged_parquet(
	merged_wide_parquet: Path,
	model: torch.nn.Module,
	model_hash: str,
	cfg: EmbeddingsConfig,
) -> None:
	"""Iterate a merged wide parquet file and populate the embeddings cache."""

	pf = pq.ParquetFile(str(merged_wide_parquet))
	logger.info(
		f"embedding input parquet rows={pf.metadata.num_rows} row_groups={pf.metadata.num_row_groups}"
	)

	iterator = pf.iter_batches(batch_size=cfg.parquet_batch_rows)
	if cfg.show_progress:
		iterator = tqdm(
			iterator,
			total=None,
			desc="Parquet batches",
			unit="batch",
		)

	con = None
	if cfg.prematch and not cfg.force_recompute:
		con = init_cache(str(cfg.embeddings_db_path))

	for batch in iterator:  # type: ignore
		tbl = pa.Table.from_batches([batch])
		if tbl.num_rows == 0:
			del batch, tbl
			gc.collect()
			continue
		if "sample_id" not in tbl.column_names:
			raise ValueError(
				f"Merged parquet is missing required column 'sample_id': {merged_wide_parquet}"
			)

		# Optional pre-match: if everything in this batch is already cached, skip
		# converting full batch to pandas + calling compute_embeddings.
		if con is not None:
			id_cols = ["sample_id"]
			if "ref_id" in tbl.column_names:
				id_cols.append("ref_id")
			ids_df = tbl.select(id_cols).to_pandas()
			if ids_df.empty:
				del batch, tbl, ids_df
				gc.collect()
				continue
			ids_df["sample_id"] = ids_df["sample_id"].astype(str)
			use_ref_id = "ref_id" in ids_df.columns
			if use_ref_id:
				ids_df["ref_id"] = ids_df["ref_id"].astype(str)
			con.register("ids", ids_df)
			if use_ref_id:
				stats = con.execute(
					"""
					SELECT
						COUNT(*) AS n_total,
						SUM(CASE WHEN e.sample_id IS NULL THEN 1 ELSE 0 END) AS n_missing
					FROM ids i
					LEFT JOIN embeddings_cache e
					  ON e.model_hash = ?
					 AND e.sample_id = i.sample_id
					 AND e.ref_id IS NOT DISTINCT FROM i.ref_id
					""",
					[model_hash],
				).fetchone()
			else:
				stats = con.execute(
					"""
					SELECT
						COUNT(*) AS n_total,
						SUM(CASE WHEN e.sample_id IS NULL THEN 1 ELSE 0 END) AS n_missing
					FROM ids i
					LEFT JOIN embeddings_cache e
					  ON e.model_hash = ?
					 AND e.sample_id = i.sample_id
					""",
					[model_hash],
				).fetchone()
			n_missing = int((stats[1] if stats else 0) or 0)
			if n_missing == 0:
				del batch, tbl, ids_df
				gc.collect()
				continue
			del ids_df

		df = tbl.to_pandas()
		if df.empty:
			del batch, tbl, df
			gc.collect()
			continue
		compute_embeddings(
			df,
			model=model,
			batch_size=cfg.batch_size,
			num_workers=cfg.num_workers,
			embeddings_db_path=str(cfg.embeddings_db_path),
			force_recompute=cfg.force_recompute,
			show_progress=cfg.show_progress,
		)
		del batch, tbl, df
		gc.collect()


def update_embeddings_cache(
	*,
	input_long_dir: Path,
	wide_dir: Path,
	merged_wide_path: Path,
	parquet_glob: str,
	compare_on: CompareOn,
	process_cfg: ParquetProcessConfig,
	merge_cfg: MergeConfig,
	emb_cfg: EmbeddingsConfig,
	overwrite_wide: bool = False,
	overwrite_merged: bool = False,
	always_run: bool = False,
	dry_run: bool = False,
) -> None:
	"""Wrapper: detect changes, then (re)run long->wide, merge, and cache update."""

	raw_files = discover_parquets(input_long_dir, pattern=parquet_glob)
	if not raw_files:
		raise RuntimeError(f"No parquet files found under: {input_long_dir} (pattern={parquet_glob})")

	logger.info(f"discovered raw parquet files: n={len(raw_files)}")

	logger.info("loading presto model (needed for model_hash + embeddings)")
	model = load_presto_model_compat(emb_cfg.presto_url_or_path)
	model_hash = get_model_hash(model)
	logger.info(f"model_hash={model_hash}")

	# Ensure DB exists
	init_cache(str(emb_cfg.embeddings_db_path))

	if not always_run and not emb_cfg.force_recompute:
		has_updates = False
		if compare_on == "ref_id":
			in_ref_ids = _input_ref_ids(raw_files)
			cached_ref_ids = _cached_ref_ids(emb_cfg.embeddings_db_path, model_hash)
			missing_ref_ids = sorted(in_ref_ids - cached_ref_ids)
			logger.info(
				f"update-check(ref_id): input_ref_ids={len(in_ref_ids)} cached_ref_ids={len(cached_ref_ids)} missing_ref_ids={len(missing_ref_ids)}"
			)
			if missing_ref_ids:
				logger.info(f"new ref_id(s) not in cache: {missing_ref_ids[:10]}" + (" ..." if len(missing_ref_ids) > 10 else ""))
				has_updates = True
		elif compare_on == "sample_id":
			logger.info("update-check(sample_id): scanning parquet ids vs cache (may take a while)")
			has_updates = _any_missing_sample_ids(
				raw_files,
				emb_cfg.embeddings_db_path,
				model_hash,
				id_col="sample_id",
				parquet_batch_rows=emb_cfg.parquet_batch_rows,
			)
		else:
			raise ValueError(f"Unknown compare_on={compare_on}")

		if not has_updates:
			logger.info("no updates detected; nothing to do")
			return

	if dry_run:
		logger.info("dry-run enabled; stopping before processing")
		return

	wide_files, empty_files, errored_files = long_to_wide_parquets(
		raw_files,
		wide_dir=wide_dir,
		cfg=process_cfg,
		overwrite=overwrite_wide,
	)
	logger.info(
		f"long->wide done: wide={len(wide_files)} empty={len(empty_files)} errored={len(errored_files)}"
	)

	wide_files = [p for p in wide_files if p.exists()]
	if not wide_files:
		raise RuntimeError("No wide parquet files were produced; aborting.")

	merged_wide = merge_parquets_stream_to_one(
		files=wide_files,
		out_path=merged_wide_path,
		cfg=merge_cfg,
		overwrite=overwrite_merged,
	)

	compute_embeddings_from_merged_parquet(
		merged_wide_parquet=merged_wide,
		model=model,
		model_hash=model_hash,
		cfg=emb_cfg,
	)
	logger.info("done")


def _default_input_dir() -> Path | None:
	candidates = [
		Path("/home/vito/shahs/projects/worldcereal/data/worldcereal_all_extractions.parquet"),
		Path("/projects/worldcereal/data/worldcereal_all_extractions.parquet"),
	]
	for c in candidates:
		if c.exists():
			return c
	return None


def _default_embeddings_db() -> Path | None:
	candidates = [
		Path("/projects/worldcereal/data/cached_embeddings/embeddings_cache_LANDCOVER10_updated.duckdb"),
		Path("/projects/worldcereal/data/cached_embeddings/embeddings_cache_LANDCOVER10.duckdb"),
		Path("/home/vito/shahs/projects/worldcereal/data/cached_embeddings/embeddings_cache_LANDCOVER10.duckdb"),
	]
	for c in candidates:
		if c.exists():
			return c
	return None


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Convert long-format parquets to wide, merge them, and update embeddings cache.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	default_in = _default_input_dir()
	default_db = _default_embeddings_db()

	p.add_argument(
		"--input-long-dir",
		type=Path,
		default=default_in,
		required=default_in is None,
		help="Folder containing raw long-format parquet files (searched recursively), or a single parquet file.",
	)
	p.add_argument(
		"--parquet-glob",
		type=str,
		default="**/*.parquet",
		help="Glob pattern used under --input-long-dir.",
	)

	p.add_argument(
		"--wide-dir",
		type=Path,
		default=None,
		help="Output folder for per-file wide parquet results. Default: <input parent>/cached_wide_merged/cached_wide_parquets",
	)
	p.add_argument(
		"--merged-wide-path",
		type=Path,
		default=None,
		help="Output merged parquet path. Default: <input parent>/<input name>_wide_<freq>.parquet",
	)

	p.add_argument(
		"--embeddings-db-path",
		type=Path,
		default=default_db,
		required=default_db is None,
		help="DuckDB embeddings cache path.",
	)
	p.add_argument(
		"--presto-url-or-path",
		type=str,
		default=(
			"https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/"
			"PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
		),
		help="Presto checkpoint URL or local path.",
	)

	p.add_argument(
		"--compare-on",
		choices=["ref_id", "sample_id"],
		default="ref_id",
		help="Update-check mode before doing any work.",
	)
	p.add_argument(
		"--always-run",
		action="store_true",
		help="Ignore update-check and always run the pipeline.",
	)
	p.add_argument(
		"--dry-run",
		action="store_true",
		help="Only perform the update-check and stop.",
	)

	# process_parquet params
	p.add_argument("--freq", choices=["month", "dekad"], default="month")
	p.add_argument("--required-min-timesteps", type=int, default=None)
	p.add_argument("--use-valid-time", action=argparse.BooleanOptionalAction, default=True)
	p.add_argument("--min-edge-buffer", type=int, default=1)
	p.add_argument("--max-timesteps-trim", default="18", help="Int, 'auto', or 'None'.")
	p.add_argument("--wide-engine", choices=["fastparquet", "pyarrow"], default="fastparquet")
	p.add_argument("--wide-compression", type=str, default="snappy")
	p.add_argument("--overwrite-wide", action="store_true")

	# merge params
	p.add_argument("--merge-batch-rows", type=int, default=100_000)
	p.add_argument("--merge-row-group-size", type=int, default=100_000)
	p.add_argument("--merge-compression", type=str, default="zstd")
	p.add_argument("--overwrite-merged", action="store_true")

	# embeddings params
	p.add_argument("--batch-size", type=int, default=16_384)
	p.add_argument("--num-workers", type=int, default=8)
	p.add_argument("--parquet-batch-rows", type=int, default=200_000)
	p.add_argument("--force-recompute", action="store_true")
	p.add_argument(
		"--prematch",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Skip embedding computation for batches fully present in cache (based on sample_id/ref_id + model_hash).",
	)
	p.add_argument("--no-progress", action="store_true")

	p.add_argument("--log-level", type=str, default="INFO")
	return p


def _parse_max_timesteps_trim(v: str):
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


def main(argv: Sequence[str] | None = None) -> int:
	args = build_arg_parser().parse_args(argv)

	logger.remove()
	logger.add(sys.stderr, level=args.log_level.upper())

	input_long_dir: Path = args.input_long_dir
	if input_long_dir is None:
		raise SystemExit("--input-long-dir is required")

	# Defaults dependent on input path
	input_parent = input_long_dir.parent
	wide_dir = args.wide_dir or (input_parent / "cached_wide_merged/cached_wide_parquets")

	if args.merged_wide_path is None:
		base = input_long_dir.stem if input_long_dir.is_file() else input_long_dir.name
		merged_wide_path = input_parent / f"{base}_wide_{args.freq}.parquet"
	else:
		merged_wide_path = args.merged_wide_path

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
	emb_cfg = EmbeddingsConfig(
		presto_url_or_path=args.presto_url_or_path,
		embeddings_db_path=args.embeddings_db_path,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		force_recompute=bool(args.force_recompute),
		prematch=bool(args.prematch),
		show_progress=not bool(args.no_progress),
		parquet_batch_rows=args.parquet_batch_rows,
	)

	logger.info(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
	logger.info(f"input_long_dir={input_long_dir}")
	logger.info(f"wide_dir={wide_dir}")
	logger.info(f"merged_wide_path={merged_wide_path}")
	logger.info(f"embeddings_db_path={emb_cfg.embeddings_db_path}")

	update_embeddings_cache(
		input_long_dir=input_long_dir,
		wide_dir=wide_dir,
		merged_wide_path=merged_wide_path,
		parquet_glob=args.parquet_glob,
		compare_on=args.compare_on,
		process_cfg=process_cfg,
		merge_cfg=merge_cfg,
		emb_cfg=emb_cfg,
		overwrite_wide=bool(args.overwrite_wide),
		overwrite_merged=bool(args.overwrite_merged),
		always_run=bool(args.always_run),
		dry_run=bool(args.dry_run),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

