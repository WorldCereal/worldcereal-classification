"""Manage the shared AGERA5 monthly-composite cache.

The extraction engine treats the cache as FROZEN at runtime — deliberately:
refreshing values mid-campaign would give refs processed on different days
different meteo for the same month. New months are still fetched on demand
(cache miss -> S3 download -> local-daily compositing fallback); this tool
exists for everything else:

  prestage : download every month of a date range up front (idempotent)
  verify   : compare each cached file's MD5 against the S3 ETag — detects
             upstream regenerations that a frozen cache would otherwise
             serve stale, without changing anything
  refresh  : verify + re-download the mismatches (run BETWEEN campaigns,
             never during one)

CloudFerro serves these objects single-part, so the S3 ETag is the plain MD5
of the content. A month absent from S3 (e.g. composited locally from the
daily archive) is reported as 'local-only' and left alone.

Usage:
  agera5_cache.py --cache-dir DIR prestage --start 2016-08 --end 2026-03
  agera5_cache.py --cache-dir DIR verify
  agera5_cache.py --cache-dir DIR refresh
"""

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ptp_engine import AGERA5_S3, MonthlyMeteo  # noqa: E402

BANDS = ("temperature-mean", "precipitation-flux")


def _months(start: str, end: str):
    cur = dt.date(int(start[:4]), int(start[5:7]), 1)
    stop = dt.date(int(end[:4]), int(end[5:7]), 1)
    while cur <= stop:
        yield cur.year, cur.month
        cur = (cur.replace(day=28) + dt.timedelta(days=5)).replace(day=1)


def _s3_url(year: int, month: int, band: str) -> str:
    return f"{AGERA5_S3}/openEO_{year}-{month:02d}-01Z_{band}.tif"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_etag(url: str):
    try:
        r = requests.head(url, timeout=60)
        if r.status_code == 200:
            return r.headers.get("ETag", "").strip('"'), int(
                r.headers.get("Content-Length", 0))
    except requests.RequestException:
        pass
    return None, None


def cmd_prestage(cache_dir: Path, start: str, end: str) -> None:
    meteo = MonthlyMeteo(cache_dir=cache_dir)
    n = 0
    for year, month in _months(start, end):
        for band in BANDS:
            meteo._ensure(year, month, band)
            n += 1
    logger.success(f"prestaged/confirmed {n} rasters in {cache_dir}")


def cmd_verify(cache_dir: Path, do_refresh: bool) -> int:
    stale, local_only, ok = [], [], 0
    for f in sorted(cache_dir.glob("openEO_*.tif")):
        url = f"{AGERA5_S3}/{f.name}"
        etag, size = _remote_etag(url)
        if etag is None:
            local_only.append(f.name)
            continue
        if "-" in etag:  # multipart etag: fall back to size comparison
            fresh = f.stat().st_size == size
        else:
            fresh = _md5(f) == etag
        if fresh:
            ok += 1
        else:
            stale.append(f.name)
            if do_refresh:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                tmp = f.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.rename(f)
                logger.warning(f"refreshed {f.name} (upstream regenerated)")
    logger.info(f"cache check: {ok} fresh, {len(stale)} "
                f"{'refreshed' if do_refresh else 'STALE'}, "
                f"{len(local_only)} local-only (daily-derived or S3-absent)")
    if stale and not do_refresh:
        for name in stale[:10]:
            logger.warning(f"STALE vs S3: {name}")
        logger.warning("run with 'refresh' to update — but only BETWEEN "
                       "campaigns, never mid-run")
    return len(stale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prestage")
    p.add_argument("--start", default="2016-08")
    p.add_argument("--end", default="2026-03")
    sub.add_parser("verify")
    sub.add_parser("refresh")
    args = ap.parse_args()

    if args.cmd == "prestage":
        cmd_prestage(args.cache_dir, args.start, args.end)
    else:
        n_stale = cmd_verify(args.cache_dir, do_refresh=(args.cmd == "refresh"))
        if n_stale and args.cmd == "verify":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
