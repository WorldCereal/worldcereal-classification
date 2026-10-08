"""Spark (mepsy) driver for the RDM patch-to-point campaign.

Fans refs out over Hadoop/YARN executors instead of running them on a single
VM. ptp_campaign_rdm.run_ref is reused unchanged on the executor, with the same
conventions, verification and stats, so outputs are byte-identical to the local
runs (one ref never spans two executors).

Task unit = one ref_id per executor task. Inside the executor the engine's own
ProcessPoolExecutor is capped via --ref-workers.

Prerequisites on the cluster (test on 2 refs first):
  * executors must see the NFS mounts /data/worldcereal_data, /data/MTDA and
    /vitodata/worldcereal;
  * the global index and the repo scripts are read from NFS paths passed
    explicitly, since executors do not share your $HOME.

Usage (test, 2 small refs, 2 executors):
  python3 ptp_campaign_rdm_spark.py \
      --ref-ids 2019_TZA_CIMMYT-DM2_POINT_110 2021_MOZ_FAO-WAPOR-1_POLY_111 \
      --out-dir /vitodata/worldcereal/data/test_spark_runs/MERGED_PARQUETS \
      --global-index /vitodata/worldcereal/data/test_spark_runs/patches_index.geoparquet \
      --executors 2 --verify 20

Usage (one shard of the campaign queue):
  python3 ptp_campaign_rdm_spark.py \
      --shard 2 --n-shards 3 \
      --out-dir /vitodata/worldcereal/data/test_spark_runs/MERGED_PARQUETS \
      --global-index /vitodata/worldcereal/data/test_spark_runs/patches_index.geoparquet \
      --executors 40 --verify 100
"""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
# In YARN cluster mode this script runs from the container cache, where its
# sibling modules do not exist. The checkout is NFS-visible cluster-wide, so
# resolve the real scripts dir by probing for ptp_engine.py.
_NFS_SCRIPT_DIR = Path(
    f"/data/users/Private/{os.environ.get('USER', '')}"
    "/worldcereal-classification/scripts/extractions")
if not (SCRIPT_DIR / "ptp_engine.py").exists() \
        and (_NFS_SCRIPT_DIR / "ptp_engine.py").exists():
    SCRIPT_DIR = _NFS_SCRIPT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

REFS_FILE_DEFAULT = SCRIPT_DIR / "rdm_campaign_refs.txt"
# Workload-sorted ref table (see --tier), kept on NFS next to the outputs so
# the checked-in refs file stays untouched.
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
        _CONV = {**_CONV, "s2_mask": task.get("s2_mask", "dilated"),
                 "freq": task.get("freq", "month")}
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
            # Hybrid centroid rule. run_ref reads these off args, so they
            # must be present; placement happens inside the executor.
            edge_fallback=task["edge_fallback"],
            legacy_centroid=task["legacy_centroid"],
            edge_margin_m=task["edge_margin_m"],
            # delta_from: skip sample_ids already in an existing store, so a
            # re-run extracts only what is genuinely new.
            delta_from=(_Path(task["delta_from"])
                        if task.get("delta_from") else None),
            dump_points=None,
            # Children sampling. run_ref reads all four off args, so they
            # must exist; defaults are off.
            children=task.get("children", 0),
            children_tier=task.get("children_tier", "auto"),
            children_edge_buffer=task.get("children_edge_buffer", None),
            children_min_dist=task.get("children_min_dist", None),
            # S1 refresh: unused from Spark, but run_ref reads it
            # unconditionally, so it must be present. None = off.
            s1_refresh_from=None,
        )
        stats = _rdm.run_ref(ref_id, run_args, _CONV)

        # Per-ref stats JSON; the driver regenerates the merged CSV after.
        if stats:
            import json as _json
            stats_dir = out_dir / "_stats"
            stats_dir.mkdir(exist_ok=True)
            (stats_dir / f"{ref_id}.json").write_text(
                _json.dumps(stats, default=str, indent=1))
        return {"ref_id": ref_id, "status": "OK"}
    except Exception as exc:
        _logger.error(f"{ref_id}: FAILED — {exc}\n{traceback.format_exc()}")
        # One bad ref must not abort the whole job; leave a breadcrumb.
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
                     help="DEPRECATED: bins on the old-store sample count "
                          "in --refs-file, which correlates only ~0.75 with "
                          "the real workload and caused OOM kills. Use --tier.")
    sel.add_argument("--tier",
                     choices=["high", "mid", "low", "xl", "l", "m", "s",
                              "all"],
                     help="select refs by real extraction workload from the "
                          "'<ref> <tier> <est_points> <rows_read>' table in "
                          "--workload-file. Run one job per tier.")
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
                    help="targeted re-extraction: executors skip samples "
                         "already in DIR/<ref>.geoparquet. Point it at an "
                         "existing store to top it up in place.")
    ap.add_argument("--edge-fallback", action="store_true",
                    help="also apply the clipped fallback to centroids in the "
                         "outer --edge-margin-m band (fully production-"
                         "faithful; moves ~5-10%% of samples)")
    ap.add_argument("--edge-margin-m", type=float, default=20.0,
                    help="inward shrink of the patch footprints for the "
                         "clipped fallback (production: 20 m)")
    ap.add_argument("--freq", choices=["month", "dekad"], default="month",
                    help="compositing period (patch_to_point.py --period): "
                         "'month' (default) or 'dekad' (days 1, 11 and 21, "
                         "which also switches AGERA5 to dekadal composites)")
    ap.add_argument("--s2-mask", choices=["dilated", "raw_scl"],
                    default="dilated",
                    help="S2 cloud masking (patch_to_point.py "
                         "--optical-mask-method): 'dilated' drops "
                         "SCL_DILATED_MASK == 1 (openEO-era default), "
                         "'raw_scl' drops raw SCL classes {0,1,3,8,9,10,11}")
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
                    help="cores per executor (default: --ref-workers). Leave "
                         "it unset and YARN gives 1 core while --ref-workers "
                         "spawns N processes on it, permanently oversubscribed.")
    ap.add_argument("--ref-workers", type=int, default=2,
                    help="engine worker processes inside one executor")
    ap.add_argument("--verify", type=int, default=20)
    ap.add_argument("--max-divergence-frac", type=float, default=0.3)
    ap.add_argument("--sample-limit", type=int, default=None)
    ap.add_argument("--kinit-env", type=str,
                    default=str(Path.home() / "Private" / "kinit.env"))
    ap.add_argument("--environment", type=str,
                    default=f"hdfs:///user/{os.environ.get('USER', '')}"
                            "/environments/ptp_env_v3.tar.gz",
                    help="HDFS conda-pack archive executors unpack for their "
                         "Python. ptp_env pins the VM env's exact versions and "
                         "is built with CONDA_OVERRIDE_GLIBC=2.17")
    ap.add_argument("--task-max-failures", type=int, default=4,
                    help="spark.task.maxFailures. A task exceeding this "
                         "aborts the whole stage, so 1 is dangerous: 4 lets "
                         "Spark re-place a lost task on another executor.")
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

    # Seed AGERA5 on the driver, before any executor starts: on a cold
    # shared cache the executors issue the same GETs and can read a .tif
    # another is still writing ("TIFFReadEncodedTile failed").
    try:
        # pandas, not geopandas: a subset without the geometry column raises.
        import pandas as _pd
        from ptp_engine import seed_meteo_cache as _seed
        _ix = _pd.read_parquet(args.global_index, columns=["start_date", "end_date"])
        _seed(agera5, str(_ix.start_date.min()), str(_ix.end_date.max()),
              freq=args.freq)
        del _ix
    except Exception as exc:                    # noqa: BLE001
        logger.warning(f"AGERA5 seeding skipped ({exc}); executors will "
                       "fetch on demand")

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
        # Sizes come from the "~N samples in old store" comment the refs
        # file already carries. Refs without a count fall in 'mid'.
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

    # Resume-safety: skip refs whose output exists, so no executor slot is
    # spent on a ref run_ref would skip anyway.
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
        "freq": args.freq,
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
        # include_gdal_vars puts PROJ_LIB/GDAL_DATA into ./environment;
        # LD_LIBRARY_PATH points the loader at the unpacked env's lib dir,
        # since conda-pack resolves some sonames via baked build paths.
        # maxFailures > 1 matters: a container OOM cannot be caught in
        # run_ref_task, and Spark aborts the whole stage once a task exceeds
        # it, so one OOM-killed ref would throw away the rest of the tier.
        mep = mepsy.SparkApp(
            include_gdal_vars=True,
            env_vars={"LD_LIBRARY_PATH": "./environment/lib"},
            extra_spark_confs={
                "spark.task.maxFailures": str(args.task_max_failures),
                "spark.stage.maxConsecutiveAttempts": "4",
            },
            **app_config)
        mep.foreach(run_ref_task, tasks)

    # mepsy.foreach returns nothing, so report from the filesystem, keeping
    # the three outcomes apart:
    #   MISSING       - no output parquet, the task died. The only true failure.
    #   VERIFY_FAILED - parquet written and usable, verification flagged it.
    #                   Worth reviewing, not rerunning.
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
