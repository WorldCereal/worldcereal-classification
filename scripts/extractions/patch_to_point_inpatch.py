"""patch-to-point extraction of the IN-PATCH hard-negative (non-crop) points.

These points were sampled *inside* existing S2/S1 patch extractions of other
(host) datasets, so there is nothing new to composite -- the extraction loads
the host's patches and samples them at our point locations. Three things differ
from a standard patch-to-point run, and this driver exists to handle them:

A. Job routing is by HOST ref_id, not by our own ref_id. One input file mixes
   points hosted by several datasets, and a point hosted by ref A can also fall
   inside an overlapping footprint of ref B. So the ground-truth file is split
   per host before the flow ever sees it.

B. Outputs come back keyed by the host ref_id (`ref_id` is written as a
   Categorical of the host's ref_id and `year` is derived from it -- a
   convention inherited from the openEO flow's
   `post_job_action_point_worldcereal`). The `rekey` stage regroups the rows
   under our own ref_ids using the `<our_ref_id>_<i>` sample_id prefix.

C. `h3_l3_cell` is used for *routing* inside the flow: `get_label_points`
   pre-filters the ground-truth file on the L3 cells parsed out of the host
   patch sample_ids, and `max_samples_per_job` splits jobs by the sample's own
   L3 cell (then drops cells that have no S1 patch for that host+EPSG). A point
   near a cell boundary can sit in a different L3 cell than its host patch and
   be silently dropped. The `prepare` stage therefore rewrites `h3_l3_cell` to
   the host patch's cell; `rekey` restores the point's true cell afterwards.

Both input datasets share the same three hosts and the same schema, so they are
extracted together (one set of jobs per host instead of one per host per file)
and separated again at the rekey stage.

Stages: prepare -> [extraction] -> rekey -> gate. The extraction step between
`prepare` and `rekey` is NOT part of this driver: run it locally with
`patch_to_point_local.py` (or `run_patch_to_point_local.sh` for the sharded
screen layout) on the per-host ground-truth files that `prepare` writes.

This driver used to carry an openEO `run` stage that drove the extraction as
patch-to-point batch jobs. It was removed after openEO's `aggregate_spatial`
was found to return the value of a NEIGHBOURING pixel for ~48% of sampled
points (a sub-pixel job-layout offset: the whole time series and all bands
shift together), so the campaign switched to the local route in
`patch_to_point_local.py`, validated bit-exact against the host patches. The
removed stage stays recoverable on branch `ptp-heavy-job-split`.

Campaign history (the in-patch non-crop run, ~90 hosts): host sample_ids come
in two naming flavours -- POL embeds the H3 L3 cell, BGR/DNK do not (see
`_host_h3_cell`). The openEO run stage needed `max_samples_per_job` splitting
for POL (~11.7k points; BGR/DNK stayed below any sane cap), and a single host
with a dangling STAC item (a catalogue entry whose .nc file is missing) could
fail all of its jobs -- `check_patch_stac_integrity.py` diagnoses that class
of failure.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from pandas.core.dtypes.dtypes import CategoricalDtype

from worldcereal.rdm_api.rdm_interaction import RDM_DEFAULT_COLUMNS

# --- Defaults -------------------------------------------------------------

# Merged outputs are kept apart from the regular MERGED_PARQUETS so they are
# never mistaken for a host dataset's own extraction.
MERGED_SUBDIR = "MERGED_PARQUETS_INPATCH_NONCROP"

# Suffix appended to the host ref_id for the per-job output folder, so this run
# cannot collide with (or silently re-merge) the host's own extraction that
# already lives under `<root>/<host_ref_id>/`.
RUN_SUFFIX = "INPATCH-NONCROP"

# Deliberately mirrors the pattern the extraction flow uses to recognise an
# H3 L3 cell inside a sample_id.
H3_L3_RE = re.compile(r"^83[0-9a-f]{13}$")

# The only columns `get_label_points` reads back out of the ground-truth file.
GT_COLUMNS = [c for c in RDM_DEFAULT_COLUMNS if c != "ref_id"]

PROVENANCE_COLUMNS = [
    "sample_id",
    "our_ref_id",
    "host_ref_id",
    "host_sample_id",
    "h3_l3_cell",
]


def _gt_dir(root_folder: Path, run_suffix: str) -> Path:
    return root_folder / f"_GROUND_TRUTH_{run_suffix.replace('-', '_')}"


# --- Stage 1: prepare -----------------------------------------------------


def load_input_points(input_dir: Path) -> gpd.GeoDataFrame:
    """Read every packaged in-patch geoparquet in `input_dir` into one frame.

    Adds `our_ref_id` (the file stem, which is also the sample_id prefix) and
    validates the assumptions the rest of this driver relies on.
    """
    files = sorted(input_dir.glob("*.geoparquet"))
    if not files:
        raise FileNotFoundError(f"No .geoparquet files found in {input_dir}")

    frames = []
    for file in files:
        our_ref_id = file.stem
        gdf = gpd.read_parquet(file)

        missing = [
            c
            for c in GT_COLUMNS + ["host_ref_id", "host_sample_id"]
            if c not in gdf.columns
        ]
        if missing:
            raise ValueError(f"{file.name} is missing required columns: {missing}")

        bad_prefix = ~gdf["sample_id"].str.startswith(f"{our_ref_id}_")
        if bad_prefix.any():
            raise ValueError(
                f"{file.name}: {int(bad_prefix.sum())} sample_id(s) do not start "
                f"with '{our_ref_id}_'. The rekey stage splits on that prefix, "
                "so it must hold for every row."
            )

        gdf["our_ref_id"] = our_ref_id
        frames.append(gdf)
        logger.info(f"Read {len(gdf):,} points from {file.name}")

    gdf = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs
    )

    duplicated = gdf["sample_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} duplicate sample_id(s) across input files."
        )

    logger.info(
        f"Loaded {len(gdf):,} points from {len(files)} file(s); "
        f"hosts: {gdf['host_ref_id'].value_counts().to_dict()}"
    )
    return gdf


def _host_h3_cell(host_ref_id: str, host_sample_id: str) -> Optional[str]:
    """The host patch's H3 L3 cell, as encoded in its sample_id, or None.

    Host sample_ids come in two flavours: `<ref_id>_<h3_l3_cell><index>`
    (e.g. POL) and `<ref_id>_<index>` (e.g. BGR, DNK). Only the first carries a
    cell -- and only for those does the flow's h3 pre-filter engage at all.
    """
    rest = host_sample_id[len(host_ref_id) + 1 :]
    candidate = rest[:15]
    return candidate if H3_L3_RE.match(candidate) else None


def prepare_ground_truth(
    gdf: gpd.GeoDataFrame,
    root_folder: Path,
    run_suffix: str,
    hosts: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Write one ground-truth file per host, plus a provenance sidecar.

    Returns {host_ref_id: path}.
    """
    gt_dir = _gt_dir(root_folder, run_suffix)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Keep the true point cell before overwriting it with the host's (see C in
    # the module docstring); `rekey` puts it back.
    provenance = pd.DataFrame(gdf[PROVENANCE_COLUMNS])
    provenance_path = gt_dir / "provenance.parquet"
    provenance.to_parquet(provenance_path, index=False)
    logger.info(f"Wrote provenance for {len(provenance):,} points -> {provenance_path}")

    host_ids = hosts or sorted(gdf["host_ref_id"].unique())
    gt_files: Dict[str, Path] = {}

    for host_ref_id in host_ids:
        subset = gdf[gdf["host_ref_id"] == host_ref_id].copy()
        if subset.empty:
            logger.warning(f"No points hosted by {host_ref_id}; skipping.")
            continue

        host_cells = subset.apply(
            lambda r: _host_h3_cell(host_ref_id, r["host_sample_id"]), axis=1
        )
        remapped = host_cells.notna() & (host_cells != subset["h3_l3_cell"])
        subset.loc[host_cells.notna(), "h3_l3_cell"] = host_cells[host_cells.notna()]

        if host_cells.notna().any():
            logger.info(
                f"{host_ref_id}: host sample_ids carry an H3 L3 cell; "
                f"realigned h3_l3_cell for {int(remapped.sum())} of "
                f"{len(subset):,} points to their host patch's cell."
            )
        else:
            logger.info(
                f"{host_ref_id}: host sample_ids carry no H3 L3 cell, so the "
                "flow's h3 pre-filter does not engage; leaving h3_l3_cell as is."
            )

        gt_path = gt_dir / f"{host_ref_id}.geoparquet"
        subset[GT_COLUMNS].to_parquet(gt_path, index=False)
        gt_files[host_ref_id] = gt_path
        logger.info(f"{host_ref_id}: wrote {len(subset):,} points -> {gt_path}")

    return gt_files


def subset_for_dry_run(
    gdf: gpd.GeoDataFrame, host_ref_id: str, cell: Optional[str], size: int
) -> gpd.GeoDataFrame:
    """Cut the input down to ~`size` points inside a single H3 L3 cell.

    Confining the pilot to one cell keeps it to a single job's worth of area
    (the openEO cost is driven by the spatial spread of the points, not by how
    many there are).
    """
    subset = gdf[gdf["host_ref_id"] == host_ref_id].copy()
    if subset.empty:
        raise ValueError(f"No points hosted by {host_ref_id}")

    subset["_host_cell"] = subset.apply(
        lambda r: _host_h3_cell(host_ref_id, r["host_sample_id"]), axis=1
    )
    subset["_host_cell"] = subset["_host_cell"].fillna(subset["h3_l3_cell"])

    if cell is None:
        counts = subset.groupby("_host_cell").size()
        # Smallest cell that still has enough points: least area, full sample.
        eligible = counts[counts >= size]
        cell = (eligible.idxmin() if not eligible.empty else counts.idxmax())
        logger.info(f"Dry run: auto-selected cell {cell} ({counts[cell]} points)")

    subset = subset[subset["_host_cell"] == cell]
    if subset.empty:
        raise ValueError(f"No points hosted by {host_ref_id} in cell {cell}")

    subset = subset.head(size).drop(columns=["_host_cell"])
    logger.info(
        f"Dry run subset: {len(subset)} points, host {host_ref_id}, cell {cell}, "
        f"datasets {subset['our_ref_id'].value_counts().to_dict()}"
    )
    return subset


# --- Stage 2: rekey -------------------------------------------------------


def rekey_outputs(
    root_folder: Path,
    merged_dir: Path,
    run_suffix: str,
    hosts: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Regroup the host-keyed merged parquets under our own ref_ids.

    Reads `<merged_dir>/<host>_<suffix>.geoparquet`, splits the rows by the
    `<our_ref_id>_<i>` sample_id prefix, restores each point's true
    `h3_l3_cell`, rewrites `ref_id`, and writes one parquet per our-ref_id.
    """
    provenance = pd.read_parquet(_gt_dir(root_folder, run_suffix) / "provenance.parquet")
    id_to_ref = dict(zip(provenance["sample_id"], provenance["our_ref_id"]))
    id_to_h3 = dict(zip(provenance["sample_id"], provenance["h3_l3_cell"]))

    host_files = sorted(merged_dir.glob(f"*_{run_suffix}.geoparquet"))
    if hosts is not None:
        wanted = {f"{h}_{run_suffix}.geoparquet" for h in hosts}
        host_files = [f for f in host_files if f.name in wanted]
    if not host_files:
        raise FileNotFoundError(
            f"No merged host parquets matching *_{run_suffix}.geoparquet in {merged_dir}"
        )

    frames = []
    for file in host_files:
        gdf = gpd.read_parquet(file)
        logger.info(f"Read {len(gdf):,} rows ({gdf['sample_id'].nunique():,} samples) "
                    f"from {file.name}")
        frames.append(gdf)

    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))

    # A point can sit inside overlapping footprints of the same host in two UTM
    # zones and be extracted twice; `merge_individual_parquet_files` only
    # dedupes within a host run, so do it again across hosts.
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["sample_id", "timestamp"], keep="first")
    if len(gdf) < before:
        logger.info(f"Dropped {before - len(gdf):,} duplicate (sample_id, timestamp) rows")

    unknown = ~gdf["sample_id"].isin(id_to_ref)
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} extracted rows have a sample_id absent from the "
            "provenance sidecar; the rekey mapping would be incomplete."
        )

    gdf["h3_l3_cell"] = gdf["sample_id"].map(id_to_h3).astype(str)
    our_ref = gdf["sample_id"].map(id_to_ref)

    written: Dict[str, Path] = {}
    for our_ref_id, group in gdf.groupby(our_ref):
        group = group.copy()

        # `post_job_action_point_worldcereal` derives `year` from the job's
        # ref_id, i.e. the host's. That happens to be right while our points
        # and their hosts share a year, but will not be once a year file is
        # hosted by patches of another year.
        our_year = int(our_ref_id.split("_")[0])
        wrong_year = group["year"] != our_year
        if wrong_year.any():
            logger.warning(
                f"{our_ref_id}: {int(wrong_year.sum()):,} rows carry the host's "
                f"year {sorted(group.loc[wrong_year, 'year'].unique())}; "
                f"rewriting to {our_year}."
            )
            group["year"] = our_year

        group["ref_id"] = pd.Series(
            [our_ref_id] * len(group),
            index=group.index,
            dtype=CategoricalDtype(categories=[our_ref_id], ordered=False),
        )
        out_path = merged_dir / f"{our_ref_id}.geoparquet"
        group.to_parquet(out_path, index=False)
        written[our_ref_id] = out_path
        logger.success(
            f"{our_ref_id}: {group['sample_id'].nunique():,} samples / "
            f"{len(group):,} rows -> {out_path}"
        )

    return written


# --- Stage 3: schema gate -------------------------------------------------


def schema_gate(
    path: Path, reference: Path, min_timesteps: int = 12, expected_samples: Optional[int] = None
) -> bool:
    """Compare an output parquet against a reference produced by the same flow."""
    logger.info(f"--- Schema gate: {path.name} vs {reference.name}")
    ok = True

    def _norm(arrow_type) -> str:
        # `string`/`large_string` (and the binary pair) differ only in offset
        # width and are interchangeable downstream; pandas picks either one
        # depending on version, so comparing them literally is noise.
        return (
            str(arrow_type)
            .replace("large_string", "string")
            .replace("large_binary", "binary")
        )

    ref_schema = pq.ParquetFile(reference).schema_arrow
    out_schema = pq.ParquetFile(path).schema_arrow
    ref_fields = {f.name: _norm(f.type) for f in ref_schema}
    out_fields = {f.name: _norm(f.type) for f in out_schema}

    missing = [c for c in ref_fields if c not in out_fields]
    extra = [c for c in out_fields if c not in ref_fields]
    if missing:
        logger.error(f"Missing columns: {missing}")
        ok = False
    if extra:
        logger.warning(f"Extra columns (not in reference): {extra}")

    for col in ref_fields:
        if col in out_fields and out_fields[col] != ref_fields[col]:
            # ref_id is a dictionary type whose categories differ by design.
            if col == "ref_id" and out_fields[col].startswith("dictionary"):
                continue
            logger.error(
                f"dtype mismatch for {col}: {out_fields[col]} != {ref_fields[col]}"
            )
            ok = False

    gdf = gpd.read_parquet(path)
    per_sample = gdf.groupby("sample_id")["timestamp"].nunique()
    logger.info(
        f"Samples: {len(per_sample):,} | timesteps per sample: "
        f"min={per_sample.min()} median={int(per_sample.median())} max={per_sample.max()}"
    )
    short = per_sample[per_sample < min_timesteps]
    if not short.empty:
        logger.error(
            f"{len(short):,} sample(s) have < {min_timesteps} timesteps "
            f"(min {short.min()})"
        )
        ok = False

    bands = [c for c in gdf.columns if c.startswith(("S2-", "S1-", "AGERA5-"))] + [
        "slope",
        "elevation",
    ]
    nodata = {c: round(float((gdf[c] == 65535).mean()), 4) for c in bands}
    logger.info(f"Nodata fraction per band: {nodata}")
    if nodata.get("S1-SIGMA0-VV", 0) == 1.0:
        logger.error("S1-SIGMA0-VV is entirely nodata -- S1 was not extracted.")
        ok = False

    logger.info(f"Timestamp range: {gdf['timestamp'].min()} .. {gdf['timestamp'].max()}")
    logger.info(f"valid_time values: {gdf['valid_time'].unique()[:5]}")

    if expected_samples is not None:
        kept = len(per_sample)
        loss = 1 - kept / expected_samples
        level = logger.warning if loss > 0.02 else logger.info
        level(
            f"Retention: {kept:,} of {expected_samples:,} points kept "
            f"({loss:.2%} lost to the spatial/temporal filters and the "
            "all-nodata drop)"
        )

    (logger.success if ok else logger.error)(
        f"Schema gate {'PASSED' if ok else 'FAILED'} for {path.name}"
    )
    return ok


# --- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="patch-to-point extraction of in-patch hard-negative points."
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "rekey", "gate"],
        required=True,
        help="Which stage to execute. The extraction itself runs externally "
        "via patch_to_point_local.py between prepare and rekey.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory holding the packaged in-patch .geoparquet input files. "
        "Required for --stage prepare.",
    )
    parser.add_argument(
        "--root-folder",
        type=str,
        required=True,
        help="Campaign root folder (holds the ground-truth dir and merged outputs).",
    )
    parser.add_argument(
        "--merged-subdir",
        type=str,
        default=MERGED_SUBDIR,
        help="Subfolder of --root-folder for merged outputs.",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=RUN_SUFFIX,
        help="Suffix appended to the host ref_id in per-run file/folder names. "
        "Must match the --run-suffix the extraction ran with "
        "(patch_to_point_local.py defaults to 'LOCAL').",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these host ref_ids. Default: all hosts in the inputs.",
    )
    parser.add_argument(
        "--schema-reference",
        type=str,
        default=None,
        help="Merged parquet from a regular run, used as the schema gate "
        "reference. Required for --stage gate.",
    )

    dry = parser.add_argument_group("dry run")
    dry.add_argument(
        "--dry-run",
        action="store_true",
        help="Pilot on a single H3 L3 cell of one host, into its own run folder.",
    )
    dry.add_argument("--dry-run-host", type=str, default="2025_POL_LPIS_POLY_110")
    dry.add_argument(
        "--dry-run-cell",
        type=str,
        default=None,
        help="H3 L3 cell to pilot on. Default: smallest cell holding enough points.",
    )
    dry.add_argument("--dry-run-size", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "prepare" and args.input_dir is None:
        parser.error("--input-dir is required for --stage prepare")
    if args.stage == "gate" and args.schema_reference is None:
        parser.error("--schema-reference is required for --stage gate")

    root_folder = Path(args.root_folder)
    merged_dir = root_folder / args.merged_subdir
    run_suffix = f"{args.run_suffix}-DRYRUN" if args.dry_run else args.run_suffix
    if args.dry_run:
        merged_dir = merged_dir / "_dryrun"
    merged_dir.mkdir(parents=True, exist_ok=True)

    hosts = args.hosts
    if args.dry_run:
        hosts = [args.dry_run_host]

    stages = [args.stage]
    logger.info(f"Stages: {stages} | run suffix: {run_suffix}")
    logger.info(f"Root folder: {root_folder}")
    logger.info(f"Merged outputs: {merged_dir}")

    if "prepare" in stages:
        gdf = load_input_points(Path(args.input_dir))
        if args.dry_run:
            gdf = subset_for_dry_run(
                gdf, args.dry_run_host, args.dry_run_cell, args.dry_run_size
            )
        prepare_ground_truth(gdf, root_folder, run_suffix, hosts)

    written: Dict[str, Path] = {}
    if "rekey" in stages:
        written = rekey_outputs(root_folder, merged_dir, run_suffix, hosts)

    if "gate" in stages:
        provenance = pd.read_parquet(
            _gt_dir(root_folder, run_suffix) / "provenance.parquet"
        )
        if hosts is not None:
            # Retention is only meaningful against the hosts actually run.
            provenance = provenance[provenance["host_ref_id"].isin(hosts)]
        if not written:
            written = {
                r: merged_dir / f"{r}.geoparquet"
                for r in provenance["our_ref_id"].unique()
            }
        counts = provenance["our_ref_id"].value_counts().to_dict()
        results = {
            ref: schema_gate(
                path,
                Path(args.schema_reference),
                expected_samples=counts.get(ref),
            )
            for ref, path in written.items()
            if path.exists()
        }
        if not all(results.values()):
            raise SystemExit("Schema gate failed; see errors above.")

    logger.success("All done!")


if __name__ == "__main__":
    main()
