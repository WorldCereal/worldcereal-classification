"""Spark (mepsy) driver for ptp_points_extract.py — arbitrary point sources.

Same relationship as ptp_campaign_rdm_spark.py has to ptp_campaign_rdm.py: the
per-unit work is reused unchanged, only the fan-out changes. Outputs are
therefore identical to a --local run.

Task unit = one (source_ref_id, host_ref_id) pair
-------------------------------------------------
The RDM driver uses one task per ref because a ref's patches all live under its
own directory. Here a single source ref can draw on dozens of hosts (a 375-point
OSM-HARDNEG ref touched 17), and a single host can serve many source refs. Using
the source ref as the task unit would serialise those hosts inside one executor
and leave the cluster idle; using the PAIR lets every (source, host) combination
run concurrently. Each task writes one part file; the driver concatenates parts
per source ref at the end. That keeps executors free of cross-task coordination
and makes the whole thing resume-safe: a part that exists is not recomputed.

The assignment (point -> patch, over the WHOLE archive) happens once on the
driver, not per task: it is a single STRtree join and shipping the result is far
cheaper than making every executor read the 90 MB index.

Usage
-----
  python ptp_points_extract_spark.py \
      --points /path/points.geoparquet \
      --index  /vitodata/worldcereal/data/test_spark_runs/patches_index_extended.geoparquet \
      --out-dir /vitodata/worldcereal/data/POINTS_RUN \
      --executors 40 --executor-memory 6

Prerequisites are the RDM driver's: executors must see the NFS mounts, and every
path passed in must be NFS-visible (not $HOME).
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
_NFS_SCRIPT_DIR = Path(
    "/data/users/Private/{user}/worldcereal-classification"
    "/scripts/extractions")
if not (SCRIPT_DIR / "ptp_engine.py").exists() \
        and (_NFS_SCRIPT_DIR / "ptp_engine.py").exists():
    SCRIPT_DIR = _NFS_SCRIPT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# --- Executor entry point ----------------------------------------------------

def run_pair_task(task: dict) -> dict:
    """Extract one (source_ref, host_ref) pair into a part file."""
    import sys as _sys
    import traceback
    from pathlib import Path as _Path

    script_dir = task["script_dir"]
    if script_dir not in _sys.path:
        _sys.path.insert(0, script_dir)

    from loguru import logger as _logger

    src_ref, host_ref = task["source_ref_id"], task["host_ref_id"]
    tag = f"{src_ref}@{host_ref}"
    part = _Path(task["part_path"])
    try:
        if part.exists():
            return {"tag": tag, "status": "SKIP"}

        import geopandas as _gpd

        import ptp_engine as _engine
        from ptp_engine import DEFAULT_CONVENTIONS as _CONV
        _CONV = {**_CONV, "s2_mask": task.get("s2_mask", "dilated"),
                 "freq": task.get("freq", "month")}
        from ptp_engine import extract_host as _extract
        from ptp_points_extract import _index_to_entries

        _engine.AGERA5_CACHE = _Path(task["agera5_cache"])
        _engine.AGERA5_CACHE.mkdir(parents=True, exist_ok=True)

        pts = _gpd.read_parquet(task["assigned_path"], filters=[
            ("ref_id", "==", src_ref), ("host_ref_id", "==", host_ref)])
        if pts.empty:
            return {"tag": tag, "status": "EMPTY"}

        idx = _gpd.read_parquet(task["index_path"],
                                filters=[("ref_id", "==", host_ref)])
        entries = _index_to_entries(idx)

        tmp = part.with_suffix(".tmp")
        _extract(host_ref, _CONV, workers=task["workers"], out_path=tmp,
                 points=pts, index_source="auto", catalog_cache=None,
                 index=entries)
        if tmp.exists():
            tmp.rename(part)          # atomic: a part is complete or absent
        return {"tag": tag, "status": "OK"}
    except Exception as exc:
        _logger.error(f"{tag}: FAILED — {exc}\n{traceback.format_exc()}")
        try:
            fdir = _Path(task["out_dir"]) / "_failed"
            fdir.mkdir(exist_ok=True)
            (fdir / f"{tag.replace('@', '__')}.txt").write_text(
                f"{exc}\n\n{traceback.format_exc()}")
        except Exception:
            pass
        return {"tag": tag, "status": f"FAILED: {exc}"}


# --- Driver ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ref-ids", nargs="+", help="only these SOURCE ref_ids")
    ap.add_argument("--refs-file", type=Path,
                    help="table of '<ref_id> <tier> <points>' lines; combine "
                         "with --tier to run one tier per job so memory can "
                         "be sized to the workload")
    ap.add_argument("--tier", type=str,
                    help="tier name to select from --refs-file")
    ap.add_argument("--executors", type=int, default=20)
    ap.add_argument("--executor-memory", type=int, default=6,
                    help="GB; mepsy gives the JVM 1 GB and the REST to "
                         "memoryOverhead, where Python runs — so N means "
                         "'Python gets ~N-1 GB'")
    ap.add_argument("--driver-memory", type=int, default=12,
                    help="GB for the YARN AM. The driver holds the whole "
                         "assignment (every point with its host patch) in "
                         "memory before writing _assigned.geoparquet, so this "
                         "scales with POINT COUNT, not executor count. 4 GB "
                         "was killed at 4.6/4.5 GB on a 1.3M-point tier "
                         "(exitCode -104), so the default is generous.")
    ap.add_argument("--queue", type=str, default="default")
    ap.add_argument("--freq", choices=["month", "dekad"], default="month",
                    help="compositing period (patch_to_point.py --period): "
                         "'month' (default) or 'dekad' (3 per month at days "
                         "1/11/21). Must match the campaign run the output "
                         "will sit alongside.")
    ap.add_argument("--s2-mask", choices=["dilated", "raw_scl"],
                    default="dilated",
                    help="S2 cloud masking: 'dilated' (precomputed "
                         "erosion/dilation band, production) or 'raw_scl' "
                         "(raw SCL classes {0,1,3,8,9,10,11}, no "
                         "erosion/dilation). Must match the campaign run the "
                         "output will sit alongside.")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="after merging each ref, classify N random points "
                         "against the openEO store and write "
                         "_verify/<ref>.json (0 = off). Same verify_ref the "
                         "RDM campaign driver uses, so the two stores are "
                         "auditable the same way.")
    ap.add_argument("--verify-store", type=Path, default=None,
                    help="openEO store to verify against "
                         "(default: ptp_verify.STORE_DEFAULT)")
    ap.add_argument("--max-divergence-frac", type=float, default=0.3,
                    help="fraction of geometry-divergent samples tolerated "
                         "before a ref is marked divergent")
    ap.add_argument("--workers", type=int, default=2,
                    help="extraction processes INSIDE one task. Each opens its "
                         "own netCDF handles (netCDF4/HDF5 is not thread-safe, "
                         "so these are processes, not threads). Keep "
                         "<= --executor-cores or they contend for one core.")
    ap.add_argument("--executor-cores", type=int, default=None,
                    help="cores per executor (default: --workers). Until "
                         "2026-08-21 this was never set, so YARN gave each "
                         "executor 1 core while --workers 2 spawned 2 "
                         "processes on it — permanently oversubscribed. "
                         "Patch-open/decode is the bottleneck for these tasks, "
                         "so real cores translate almost linearly into "
                         "throughput on the patch-heavy high tier.")
    ap.add_argument("--agera5-cache", type=Path, default=None)
    ap.add_argument("--kinit-env", type=str,
                    default="/home/{user}/Private/kinit.env")
    ap.add_argument("--environment", type=str,
                    default="hdfs:///user/{user}/environments/"
                            "ptp_env_v3.tar.gz")
    ap.add_argument("--no-temporal-check", action="store_true",
                    help="DANGEROUS: assign on space alone; see "
                         "ptp_points_extract.py --no-temporal-check")
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    if args.verify and args.verify_store is None:
        from ptp_verify import STORE_DEFAULT
        args.verify_store = STORE_DEFAULT

    for p in (args.points, args.index):
        if not p.exists():
            ap.error(f"not found: {p}")
        if str(p).startswith("/home/"):
            logger.warning(f"{p} is under /home — executors likely cannot "
                           "read it; copy it to /vitodata first!")

    from ptp_points_extract import assign_to_patches, load_points

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out_dir / "_parts"
    parts_dir.mkdir(exist_ok=True)
    agera5 = args.agera5_cache or args.out_dir / "_agera5_cache"

    # Seed AGERA5 on the DRIVER before any executor starts — a cold shared
    # cache otherwise has every executor fetch the same rasters at once and
    # read each other's half-written files. See ptp_campaign_rdm_spark.
    try:
        # pandas, not geopandas: a column subset without the geometry
        # column makes geopandas raise.
        import pandas as _pd

        from ptp_engine import seed_meteo_cache as _seed
        _ix = _pd.read_parquet(args.index, columns=["start_date", "end_date"])
        _seed(agera5, str(_ix.start_date.min()), str(_ix.end_date.max()),
              freq=args.freq)
        del _ix
    except Exception as exc:                    # noqa: BLE001
        logger.warning(f"AGERA5 seeding skipped ({exc}); executors will "
                       "fetch on demand")

    # Assign once, on the driver, and hand executors a parquet they can filter.
    import geopandas as gpd
    ref_ids = args.ref_ids
    if args.refs_file:
        if not args.refs_file.exists():
            ap.error(f"refs file not found: {args.refs_file}")
        rows = []
        for ln in args.refs_file.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
        ref_ids = [r for r, t in rows
                   if not args.tier or t == args.tier]
        logger.info(f"tier '{args.tier or 'all'}': {len(ref_ids)} source refs "
                    f"from {args.refs_file.name}")
        if not ref_ids:
            raise SystemExit(f"no refs matched tier '{args.tier}'")

    points = load_points(args.points, ref_ids)
    index = gpd.read_parquet(args.index)
    assigned, astats = assign_to_patches(
        points, index, require_temporal=not args.no_temporal_check)
    logger.info(f"assignment stats: {astats}")
    if assigned.empty:
        raise SystemExit("no point fell inside any patch footprint")

    assigned_path = args.out_dir / "_assigned.geoparquet"
    assigned.to_parquet(assigned_path)
    summary = (assigned.groupby("ref_id")
               .agg(points=("sample_id", "size"),
                    hosts=("host_ref_id", "nunique")))
    (args.out_dir / "_assignment_summary.csv").write_text(summary.to_csv())
    logger.info(f"assignment:\n{summary.to_string()}")

    pairs = (assigned.groupby(["ref_id", "host_ref_id"])
             .size().reset_index(name="n"))
    tasks = []
    for r in pairs.itertuples():
        part = parts_dir / f"{r.ref_id}__{r.host_ref_id}.geoparquet"
        if part.exists():
            continue
        tasks.append({
            "source_ref_id": r.ref_id, "host_ref_id": r.host_ref_id,
            "script_dir": str(SCRIPT_DIR),
            "assigned_path": str(assigned_path),
            "index_path": str(args.index),
            "part_path": str(part),
            "out_dir": str(args.out_dir),
            "agera5_cache": str(agera5),
            "workers": args.workers,
            "s2_mask": args.s2_mask,
            "freq": args.freq,
        })
    logger.info(f"{len(pairs)} (source,host) pairs, {len(tasks)} to run "
                f"({len(pairs) - len(tasks)} already done)")
    if not tasks:
        logger.success("Nothing to do.")
    else:
        if args.local:
            for t in tasks:
                run_pair_task(t)
        else:
            import mepsy
            _cores = args.executor_cores or args.workers
            logger.info(f"Spark: {args.executors} executors x "
                        f"{args.executor_memory} GB x {_cores} core(s), "
                        f"{args.workers} worker(s)/task, queue={args.queue}")
            mep = mepsy.SparkApp(
                app_name="ptp_points_extract",
                driver_memory=args.driver_memory,
                executor_memory=args.executor_memory,
                executor_cores=(args.executor_cores or args.workers),
                max_executors=args.executors,
                queue=args.queue,
                kinit_env=args.kinit_env,
                environment=args.environment,
                wait_completion=True,
                local=False,
                include_gdal_vars=True,
                env_vars={"LD_LIBRARY_PATH": "./environment/lib"},
                # See ptp_campaign_rdm_spark.py: one bad task must not abort
                # the job. Do NOT add spark.excludeOnFailure — Spark rejects
                # maxTaskAttemptsPerNode >= maxFailures at context creation.
                extra_spark_confs={
                    "spark.task.maxFailures": "1",
                    "spark.stage.maxConsecutiveAttempts": "1",
                    # This driver does the WHOLE point->patch assignment
                    # (load ~8M points, read the 886k-patch index, spatial
                    # join) BEFORE it ever creates the SparkContext. In YARN
                    # cluster mode the ApplicationMaster waits
                    # spark.yarn.am.waitTime for that context and then dies:
                    #   TimeoutException: Futures timed out after [600000 ms]
                    #   ApplicationMaster.runDriver -> exitCode 13
                    # The 1.3M-point tier needs ~12 min of assignment, which
                    # is longer than the 600 s default. Local runs never hit
                    # this because the driver is not under the AM there.
                    "spark.yarn.am.waitTime": "3600s",
                })
            mep.foreach(run_pair_task, tasks)

    # --- merge parts per source ref ---
    import pandas as pd
    written, missing = 0, []
    for src_ref, grp in pairs.groupby("ref_id"):
        out_path = args.out_dir / f"{src_ref}.geoparquet"
        have = [parts_dir / f"{src_ref}__{h}.geoparquet"
                for h in grp.host_ref_id]
        have = [p for p in have if p.exists()]
        if not have:
            missing.append(src_ref)
            continue
        if len(have) < len(grp):
            logger.warning(f"{src_ref}: {len(grp) - len(have)} of "
                           f"{len(grp)} host parts missing; writing partial")
        frames = [gpd.read_parquet(p) for p in have]
        out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                               crs="EPSG:4326")
        if "ref_id" in out.columns:
            out["host_ref_id"] = out["ref_id"]
        out["ref_id"] = src_ref
        out.to_parquet(out_path)
        written += 1
        logger.success(f"{src_ref}: {len(out):,} rows -> {out_path}")

        # Verification, same machinery and same _verify/ layout the RDM
        # campaign driver uses, so both stores can be audited identically.
        # Runs on the DRIVER after the merge (verify_ref needs the whole
        # per-ref file, which only exists once the parts are concatenated).
        if args.verify:
            try:
                import json as _json

                from ptp_global_index import catalog_for_ref
                from ptp_verify import verify_ref
                sub = gpd.read_parquet(args.index,
                                       filters=[("ref_id", "==", src_ref)])
                if not len(sub):
                    logger.warning(f"{src_ref}: not in the index; "
                                   "verification skipped")
                else:
                    v = verify_ref(src_ref, out_path, catalog_for_ref(sub, src_ref),
                                   store=args.verify_store,
                                   n_samples=args.verify,
                                   max_divergence_frac=args.max_divergence_frac)
                    vdir = args.out_dir / "_verify"
                    vdir.mkdir(exist_ok=True)
                    (vdir / f"{src_ref}.json").write_text(_json.dumps(v, indent=2,
                                                                     default=str))
                    (logger.success if v["status"] == "PASS" else
                     logger.warning if v["status"].startswith("SKIP") else
                     logger.error)(f"{src_ref}: verification {v['status']}")
            except Exception as exc:   # never let a verify failure lose data
                logger.warning(f"{src_ref}: verification errored ({exc})")

    # Distinguish the two very different reasons a requested ref has no file:
    #   * NOT ASSIGNABLE - not one of its points fell inside an in-year patch,
    #     so there was never any work to do. Expected, not a failure.
    #   * CRASHED - it had pairs to run but every part is missing, i.e. tasks
    #     died. That is the only case worth a non-zero exit.
    # Conflating them made healthy runs report FAILED (a tier that finished
    # 313/313 refs still exited 1), which repeatedly cost time to re-diagnose.
    requested = set(ref_ids) if ref_ids else set(assigned.ref_id.unique())
    assignable = set(pairs.ref_id.unique())
    not_assignable = sorted(requested - assignable)
    crashed = sorted(missing)

    logger.info(f"{written} source ref(s) written")
    if not_assignable:
        logger.warning(
            f"{len(not_assignable)} ref(s) had no point inside an in-year "
            "patch — nothing to extract (not a failure):")
        for r in not_assignable:
            logger.warning(f"  no assignable points: {r}")
    if crashed:
        logger.error(f"{len(crashed)} ref(s) had work but produced no parts:")
        for r in crashed:
            logger.error(f"  FAILED: {r}")

    n_failed_parts = len(list((args.out_dir / "_failed").glob("*.txt"))) \
        if (args.out_dir / "_failed").exists() else 0
    if n_failed_parts:
        logger.error(f"{n_failed_parts} (source,host) pair(s) left a failure "
                     f"breadcrumb in {args.out_dir / '_failed'}")
    if crashed or n_failed_parts:
        raise SystemExit(1)
    logger.success("All assignable refs extracted.")


if __name__ == "__main__":
    main()
