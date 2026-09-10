"""Spark (mepsy) driver for the RDM patch-to-point campaign.

Runs the EXACT same per-ref pipeline as ptp_campaign_rdm.py / the indexed
shard driver — ptp_campaign_rdm.run_ref is reused unchanged on the executor,
with the same conventions, verification and stats — but fans refs out over
Hadoop/YARN executors instead of a single VM. Outputs are therefore
byte-identical to the linear/parallel local runs (proven property of run_ref;
one ref never spans two executors).

Task unit = one ref_id per executor task (like one S2 tile in
process_embeddings_production.py). Inside the executor the engine's own
ProcessPoolExecutor is capped via --ref-workers.

IMPORTANT prerequisites on the cluster (verify with a 2-ref test first!):
  * executors must see the NFS mounts: /data/worldcereal_data, /data/MTDA,
    /vitodata/worldcereal  (true for the HRL postprocessing jobs);
  * the global index + repo scripts are read from NFS paths passed
    explicitly (executors don't share your $HOME).

Usage (TEST, 2 small refs, 2 executors):
  python3 ptp_campaign_rdm_spark.py \
      --ref-ids 2019_TZA_CIMMYT-DM2_POINT_110 2021_MOZ_FAO-WAPOR-1_POLY_111 \
      --out-dir /vitodata/worldcereal/data/test_spark_runs/MERGED_PARQUETS \
      --global-index /vitodata/worldcereal/data/test_spark_runs/patches_index.geoparquet \
      --executors 2 --verify 20

Usage (my shard-2 queue, campaign-equivalent):
  python3 ptp_campaign_rdm_spark.py \
      --shard 2 --n-shards 3 \
      --out-dir /vitodata/worldcereal/data/test_spark_runs/MERGED_PARQUETS \
      --global-index /vitodata/worldcereal/data/test_spark_runs/patches_index.geoparquet \
      --executors 40 --verify 100
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
# In YARN cluster mode this script is re-executed from the container cache,
# where its sibling modules (ptp_engine, ptp_verify, ...) do NOT exist. The
# checkout itself is NFS-visible cluster-wide under /data/users/Private, so
# resolve the real scripts dir by probing for ptp_engine.py.
_NFS_SCRIPT_DIR = Path(
    "/data/users/Private/{user}/worldcereal-classification"
    "/scripts/extractions")
if not (SCRIPT_DIR / "ptp_engine.py").exists() \
        and (_NFS_SCRIPT_DIR / "ptp_engine.py").exists():
    SCRIPT_DIR = _NFS_SCRIPT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

REFS_FILE_DEFAULT = SCRIPT_DIR / "rdm_campaign_refs.txt"
# Workload-sorted ref table (see --tier). Lives on NFS next to the outputs
# rather than in the checkout, so the colleague's refs file stays untouched.
WORKLOAD_FILE_DEFAULT = Path(
    "/vitodata/worldcereal/data/test_spark_runs/"
    "rdm_campaign_refs_by_workload.txt")
RDM_DIR_DEFAULT = "/vitodata/worldcereal/data/RDM"


# --- Executor entry point ---------------------------------------------------
# Self-contained: takes only a plain dict (mepsy/pickle-friendly), re-imports
# everything from the NFS script dir, never touches driver state.

def run_ref_task(task: dict) -> dict:
    import sys as _sys
    import traceback
    from pathlib import Path as _Path
    from types import SimpleNamespace

    script_dir = task["script_dir"]
    if script_dir not in _sys.path:
        _sys.path.insert(0, script_dir)

    from loguru import logger as _logger

    ref_id = task["ref_id"]
    try:
        import geopandas as _gpd

        import ptp_campaign_rdm as _rdm
        import ptp_engine as _engine
        from ptp_engine import DEFAULT_CONVENTIONS as _CONV
        # S2 cloud-mask method travels in the conventions dict, so the
        # executors need it applied HERE (they re-import the engine fresh).
        _CONV = {**_CONV, "s2_mask": task.get("s2_mask", "dilated")}
        from ptp_global_index import catalog_for_ref as _cat_for_ref
        from ref_catalog import RefCatalog as _RefCatalog

        out_dir = _Path(task["out_dir"])
        _engine.AGERA5_CACHE = _Path(task["agera5_cache"])
        _engine.AGERA5_CACHE.mkdir(parents=True, exist_ok=True)

        # Same lazy index-backed loader as the indexed shard driver.
        class _Loader:
            @staticmethod
            def load(rid, source="auto", cache_dir=None):
                try:
                    sub = _gpd.read_parquet(task["global_index"],
                                            filters=[("ref_id", "==", rid)])
                    if len(sub):
                        cat = _cat_for_ref(sub, rid)
                        if any(e.get("footprint") is not None
                               for e in cat.entries.values()):
                            return cat
                except Exception as exc:
                    _logger.warning(f"{rid}: index read failed ({exc}); "
                                    "falling back to STAC")
                return _RefCatalog.load(rid, source=source,
                                        cache_dir=cache_dir)

        _rdm.RefCatalog = _Loader()

        run_args = SimpleNamespace(
            mode="extract",
            out_dir=out_dir,
            rdm_dir=_Path(task["rdm_dir"]),
            index_source="auto",
            catalog_cache=None,
            only_flagged=False,
            workers=task["ref_workers"],
            sample_limit=task["sample_limit"],
            verify=task["verify"],
            verify_pct=None,
            max_divergence_frac=task["max_divergence_frac"],
            verify_store=_Path(task["verify_store"]),
            # Hybrid centroid rule (2026-08-21). run_ref reads all five of
            # these off args, so they MUST be present or every task dies with
            # AttributeError. The placement itself happens HERE, inside the
            # executor — there is no local pre-placement pass.
            edge_fallback=task["edge_fallback"],
            legacy_centroid=task["legacy_centroid"],
            edge_margin_m=task["edge_margin_m"],
            # delta_from: skip sample_ids already in an existing store, so a
            # re-run extracts only what is genuinely new.
            delta_from=(_Path(task["delta_from"])
                        if task.get("delta_from") else None),
            dump_points=None,
            # Children sampling (her 4ed28d82). run_ref reads all four off
            # args, so they MUST exist or every task dies with AttributeError
            # — the same latent break her hybrid-centroid commit introduced.
            # Defaults = OFF, so behaviour is unchanged unless asked for.
            children=task.get("children", 0),
            children_tier=task.get("children_tier", "auto"),
            children_edge_buffer=task.get("children_edge_buffer", None),
            children_min_dist=task.get("children_min_dist", None),
            # S1 refresh mode (her _s1merge run). Not used from Spark — the
            # campaign driver always does a full extract — but run_ref reads
            # it unconditionally, so it MUST be present. None = off.
            s1_refresh_from=None,
        )
        stats = _rdm.run_ref(ref_id, run_args, _CONV)

        # Idempotent per-ref stats JSON (same scheme as the local drivers;
        # the merged CSV is regenerated by the driver afterwards).
        if stats:
            import json as _json
            stats_dir = out_dir / "_stats"
            stats_dir.mkdir(exist_ok=True)
            (stats_dir / f"{ref_id}.json").write_text(
                _json.dumps(stats, default=str, indent=1))
        return {"ref_id": ref_id, "status": "OK"}
    except Exception as exc:
        _logger.error(f"{ref_id}: FAILED — {exc}\n{traceback.format_exc()}")
        # Swallow (like the embeddings script): one bad ref must not abort
        # the whole Spark job. Leave a breadcrumb for the driver.
        try:
            fdir = _Path(task["out_dir"]) / "_failed"
            fdir.mkdir(exist_ok=True)
            (fdir / f"{ref_id}.txt").write_text(
                f"{exc}\n\n{traceback.format_exc()}")
        except Exception:
            pass
        return {"ref_id": ref_id, "status": f"FAILED: {exc}"}


# --- Driver -------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--ref-ids", nargs="+")
    sel.add_argument("--shard", type=int,
                     help="round-robin shard of --refs-file (with --n-shards)")
    sel.add_argument("--size-bin", choices=["big", "mid", "small", "all"],
                     help="DEPRECATED — bins on the '~N samples' comment in "
                          "--refs-file, which is an OLD-STORE count that "
                          "correlates only ~0.75 with the real extraction "
                          "workload and caused OOM kills. Use --tier.")
    sel.add_argument("--tier",
                     choices=["high", "mid", "low", "xl", "l", "m", "s",
                              "all"],
                     help="select refs by REAL extraction workload from "
                          "--workload-file (see WORKLOAD_FILE_DEFAULT): a "
                          "'<ref> <tier> <est_points> <rows_read>' table "
                          "derived from completed-run stats. Run one job per "
                          "tier with memory sized to it.")
    ap.add_argument("--workload-file", type=Path,
                    default=WORKLOAD_FILE_DEFAULT,
                    help="ref/tier table used by --tier")
    ap.add_argument("--big-min", type=int, default=50000,
                    help="sample-count threshold for --size-bin big")
    ap.add_argument("--small-max", type=int, default=5000,
                    help="sample-count threshold for --size-bin small")
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--refs-file", type=Path, default=REFS_FILE_DEFAULT)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--global-index", type=Path, required=True,
                    help="patches_index.geoparquet on an NFS path visible "
                         "to executors (NOT your $HOME)")
    ap.add_argument("--rdm-dir", type=Path, default=Path(RDM_DIR_DEFAULT))
    ap.add_argument("--agera5-cache", type=Path, default=None,
                    help="default: <out-dir>/_agera5_cache")
    ap.add_argument("--executors", type=int, default=20)
    ap.add_argument("--executor-memory", type=int, default=6,
                    help="GB per executor (engine ~2 GB + verify headroom; "
                         "default 6)")
    ap.add_argument("--driver-memory", type=int, default=2)
    ap.add_argument("--queue", type=str, default="default")
    ap.add_argument("--delta-from", type=Path, default=None, metavar="DIR",
                    help="targeted re-extraction: each executor skips samples "
                         "whose sample_id is already in DIR/<ref>.geoparquet, "
                         "so only genuinely new points are extracted. Point it "
                         "at an existing store to top it up in place.")
    ap.add_argument("--edge-fallback", action="store_true",
                    help="also apply the clipped fallback to centroids in the "
                         "outer --edge-margin-m band (fully production-"
                         "faithful; moves ~5-10%% of samples)")
    ap.add_argument("--edge-margin-m", type=float, default=20.0,
                    help="inward shrink of the patch footprints for the "
                         "clipped fallback (production: 20 m)")
    ap.add_argument("--s2-mask", choices=["dilated", "raw_scl"],
                    default="dilated",
                    help="S2 cloud masking. 'dilated' (default) drops obs "
                         "where the precomputed S2-L2A-SCL_DILATED_MASK == 1 "
                         "— that band has a large erosion/dilation applied, "
                         "so pixels NEAR cloud are masked too; this is what "
                         "the openEO-era store used. 'raw_scl' drops obs whose "
                         "raw S2-L2A-SCL class is in {0,1,3,8,9,10,11} with NO "
                         "erosion/dilation — less aggressive, denser "
                         "composites. Mirrors patch_to_point.py's "
                         "--optical-mask-method.")
    ap.add_argument("--children", type=int, default=0, metavar="K",
                    help="extra blue-noise points sampled INSIDE each polygon "
                         "parent (0 = off). Child ids are <parent>_child<k>; "
                         "parents keep their own row untouched.")
    ap.add_argument("--children-tier", type=str, default="auto",
                    help="child spacing preset (see CHILD_TIER_PARAMS)")
    ap.add_argument("--children-edge-buffer", type=float, default=None,
                    help="metres to keep children away from the polygon edge")
    ap.add_argument("--children-min-dist", type=float, default=None,
                    help="minimum metres between children of one parent")
    ap.add_argument("--legacy-centroid", action="store_true",
                    help="disable the clipped fallback (pre-2026-08-21 rule)")
    ap.add_argument("--executor-cores", type=int, default=None,
                    help="cores per executor (default: --ref-workers). Until "
                         "2026-08-21 this was never set, so YARN gave each "
                         "executor 1 core while --ref-workers spawned N "
                         "processes on it — permanently oversubscribed. "
                         "Patch open/decode is the bottleneck, so real cores "
                         "translate almost linearly into throughput.")
    ap.add_argument("--ref-workers", type=int, default=2,
                    help="engine worker processes inside one executor")
    ap.add_argument("--verify", type=int, default=20)
    ap.add_argument("--max-divergence-frac", type=float, default=0.3)
    ap.add_argument("--sample-limit", type=int, default=None)
    ap.add_argument("--kinit-env", type=str,
                    default="/home/{user}/Private/kinit.env")
    ap.add_argument("--environment", type=str,
                    default="hdfs:///user/{user}/environments/"
                            "ptp_env_v3.tar.gz",
                    help="HDFS conda-pack archive executors unpack for "
                         "their Python. ptp_env: py3.11 pinned to the VM "
                         "env's exact versions (numpy 2.4.3, shapely 2.1.2 "
                         "...) built with CONDA_OVERRIDE_GLIBC=2.17 so it "
                         "runs on the cluster's old glibc -> outputs are "
                         "bit-identical to VM runs")
    ap.add_argument("--spark-config", type=str, default=None,
                    help="optional mepsy spark JSON config path")
    ap.add_argument("--local", action="store_true",
                    help="run sequentially on this machine (debug)")
    args = ap.parse_args()

    if not args.global_index.exists():
        ap.error(f"global index not found: {args.global_index}")
    if str(args.global_index).startswith("/home/"):
        logger.warning("global index is under /home — executors likely "
                       "cannot read it; copy it to /vitodata first!")

    from ptp_verify import STORE_DEFAULT

    args.out_dir.mkdir(parents=True, exist_ok=True)
    agera5 = args.agera5_cache or args.out_dir / "_agera5_cache"

    if args.ref_ids:
        refs = args.ref_ids
    elif args.tier:
        if not args.workload_file.exists():
            ap.error(f"workload file not found: {args.workload_file}")
        table = []
        for ln in args.workload_file.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            table.append((parts[0], parts[1] if len(parts) > 1 else "s"))
        refs = [r for r, t in table
                if args.tier == "all" or t == args.tier]
        logger.info(f"tier '{args.tier}': {len(refs)} refs "
                    f"from {args.workload_file.name}")
    elif args.size_bin:
        # Read the refs file WITHOUT modifying it: the generator already
        # writes "<ref>  # ~N samples in old store" per line, so the size is
        # recoverable from the comment. Refs whose count is absent ("not in
        # old store") fall in 'mid' — unknown size, middling resources.
        import re as _re
        sized = []
        for ln in args.refs_file.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            m = _re.search(r"~([\d,]+)\s+samples", ln)
            sized.append((ln.split()[0],
                          int(m.group(1).replace(",", "")) if m else -1))
        if args.size_bin == "all":
            refs = [r for r, _ in sized]
        elif args.size_bin == "big":
            refs = [r for r, n in sized if n >= args.big_min]
        elif args.size_bin == "small":
            refs = [r for r, n in sized if 0 <= n < args.small_max]
        else:  # mid: the remainder, incl. unknown-size refs
            refs = [r for r, n in sized
                    if n < 0 or args.small_max <= n < args.big_min]
        logger.info(f"size-bin '{args.size_bin}': {len(refs)} refs "
                    f"(big>={args.big_min:,}, small<{args.small_max:,})")
    else:
        lines = [ln.split()[0] for ln in
                 args.refs_file.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        refs = [r for i, r in enumerate(lines)
                if i % args.n_shards == args.shard]

    # Resume-safety at the driver: skip refs whose output already exists
    # (run_ref would skip them too, but this avoids wasting executor slots).
    todo = [r for r in refs
            if not (args.out_dir / f"{r}.geoparquet").exists()]
    logger.info(f"{len(refs)} refs selected, {len(refs) - len(todo)} already "
                f"done, {len(todo)} to run")
    if not todo:
        logger.success("Nothing to do.")
        return

    tasks = [{
        "ref_id": r,
        "script_dir": str(SCRIPT_DIR),
        "out_dir": str(args.out_dir),
        "global_index": str(args.global_index),
        "rdm_dir": str(args.rdm_dir),
        "agera5_cache": str(agera5),
        "ref_workers": args.ref_workers,
        "verify": args.verify,
        "max_divergence_frac": args.max_divergence_frac,
        "sample_limit": args.sample_limit,
        "verify_store": str(STORE_DEFAULT),
        "edge_fallback": args.edge_fallback,
        "legacy_centroid": args.legacy_centroid,
        "edge_margin_m": args.edge_margin_m,
        "delta_from": str(args.delta_from) if args.delta_from else None,
        "s2_mask": args.s2_mask,
        "children": args.children,
        "children_tier": args.children_tier,
        "children_edge_buffer": args.children_edge_buffer,
        "children_min_dist": args.children_min_dist,
    } for r in todo]

    # Clear stale failure breadcrumbs for the refs we are about to run.
    fdir = args.out_dir / "_failed"
    for r in todo:
        p = fdir / f"{r}.txt"
        if p.exists():
            p.unlink()

    if args.local:
        for t in tasks:
            run_ref_task(t)
    else:
        import mepsy
        app_config = dict(
            app_name="ptp_rdm_local_extract",
            driver_memory=args.driver_memory,
            executor_memory=args.executor_memory,
            executor_cores=(args.executor_cores or args.ref_workers),
            max_executors=args.executors,
            queue=args.queue,
            kinit_env=args.kinit_env,
            environment=args.environment,
            wait_completion=True,
            local=False,
        )
        if args.spark_config:
            app_config["config_path"] = args.spark_config
        logger.info(f"Spark: {args.executors} executors x "
                    f"{args.executor_memory} GB, queue={args.queue}")
        # include_gdal_vars=True: PROJ_LIB/GDAL_DATA into ./environment.
        # LD_LIBRARY_PATH: conda-pack archives resolve some sonames via
        # baked build paths; point the loader at the unpacked env's lib dir.
        # One ref per task and refs are independent, so a task that dies (an
        # OOM kill takes the whole container down — run_ref_task's try/except
        # cannot catch it) must not abort the job and throw away every other
        # ref's work. maxFailures=1 retires the offending task immediately
        # instead of retrying it 4x on 4 different executors. Refs lost this
        # way have no output parquet, so the driver reports them as FAILED and
        # the next run (resume-skip) picks them up.
        mep = mepsy.SparkApp(
            include_gdal_vars=True,
            env_vars={"LD_LIBRARY_PATH": "./environment/lib"},
            # NB: do NOT also enable spark.excludeOnFailure — Spark validates
            # maxTaskAttemptsPerNode (default 2) < maxFailures and refuses to
            # build the context otherwise. With maxFailures=1 a task never
            # retries, so node exclusion has nothing to act on anyway.
            extra_spark_confs={
                "spark.task.maxFailures": "1",
                "spark.stage.maxConsecutiveAttempts": "1",
            },
            **app_config)
        mep.foreach(run_ref_task, tasks)

    # mepsy.foreach returns nothing — report from the filesystem. Separate the
    # three outcomes, because lumping them together made a tier that finished
    # 313/313 refs report FAILED and cost real time to re-diagnose:
    #   MISSING       - no output parquet: the task died (OOM/container kill),
    #                   this is the only true failure.
    #   VERIFY_FAILED - parquet written and usable, but verification flagged
    #                   it. Many refs fail verification in the VM campaign too
    #                   (mostly s2_unexplained); worth reviewing, not rerunning.
    #   OK            - clean.
    missing, verify_failed, ok = [], [], []
    for r in todo:
        has_out = (args.out_dir / f"{r}.geoparquet").exists()
        note = fdir / f"{r}.txt"
        if not has_out:
            missing.append(r)
        elif note.exists():
            verify_failed.append(r)
        else:
            ok.append(r)
    logger.info(f"OK: {len(ok)}  VERIFY_FAILED: {len(verify_failed)}  "
                f"MISSING: {len(missing)}")
    for r in verify_failed:
        note = fdir / f"{r}.txt"
        head = note.read_text().splitlines()[0] if note.exists() else "?"
        logger.warning(f"  verify-only (parquet is fine): {r}: {head}")
    for r in missing:
        note = fdir / f"{r}.txt"
        head = note.read_text().splitlines()[0] if note.exists() else \
            "no breadcrumb — container was killed (check executor memory)"
        logger.error(f"  MISSING: {r}: {head}")
    failed = missing

    # Regenerate the merged stats CSV from the per-ref JSONs (atomic rename).
    import json
    import os

    import pandas as pd
    stats_dir = args.out_dir / "_stats"
    if stats_dir.exists():
        rows = [json.loads(p.read_text())
                for p in sorted(stats_dir.glob("*.json"))]
        if rows:
            tmp = args.out_dir / f"_rdm_campaign_stats.csv.tmp{os.getpid()}"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.rename(args.out_dir / "_rdm_campaign_stats.csv")

    if failed:
        raise SystemExit(1)
    logger.success("All refs OK.")


if __name__ == "__main__":
    main()
