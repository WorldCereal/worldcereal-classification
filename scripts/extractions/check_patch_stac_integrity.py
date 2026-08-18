"""Pre-flight check: find STAC patch items whose backing file is missing.

A patch-to-point job loads *every* patch matching (ref_id, EPSG/tile, orbit) --
not just the ones holding samples -- so a single STAC item pointing at a file
that is not on disk aborts the whole job with

    load_stac/load_collection: GDAL gave an error for NETCDF:<path>:<band>
    Error message: Unable to get the metadata item. GDAL Error Code: 4

GDAL error code 4 is CPLE_OpenFailed: the file could not be opened. The band
named in the message is just the first one the reader tried, so it is not a
clue about which band is broken -- the file is simply absent.

Retrying never clears this: the item stays in the catalogue and the job fails
identically every time (and still costs credits). The fix is upstream -- the
patch has to be re-extracted or the stale STAC item removed.

Run this before launching a large campaign to learn which ref_ids are affected,
which orbit(s), and how many patches are involved.

    python check_patch_stac_integrity.py --ref-ids 2023_BEL_LPIS-Flanders_POLY_110
    python check_patch_stac_integrity.py --ref-ids-file hosts.txt --output report.csv
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

STAC_SEARCH = "https://stac.openeo.vito.be/search"
COLLECTIONS = {
    "S2": "worldcereal_sentinel_2_patch_extractions",
    "S1": "worldcereal_sentinel_1_patch_extractions",
}


def _iter_items(ref_id: str, collection: str, page_size: int = 500):
    """Yield STAC features for a ref_id, following the `next` link."""
    token = None
    while True:
        body: Dict = {
            "collections": [collection],
            "limit": page_size,
            "query": {"ref_id": {"eq": ref_id}},
            "fields": {
                "include": [
                    "properties.sample_id",
                    "properties.s2_tile",
                    "properties.sat:orbit_state",
                    "assets",
                ]
            },
        }
        if token:
            body["token"] = token
        resp = requests.post(STAC_SEARCH, json=body, timeout=300)
        resp.raise_for_status()
        payload = resp.json()
        yield from payload.get("features", [])

        token = None
        for link in payload.get("links", []):
            if link.get("rel") == "next":
                token = link.get("body", {}).get("token") or link.get(
                    "href", ""
                ).split("token=")[-1]
        if not token:
            break


def scan_ref(ref_id: str, collection: str, stat_workers: int = 16) -> Tuple[int, List[dict]]:
    """Return (item count, list of dangling item records) for one ref_id."""
    records = []
    for feature in _iter_items(ref_id, collection):
        props = feature.get("properties", {})
        for asset in feature.get("assets", {}).values():
            href = asset.get("href", "")
            path = href.replace("file://", "")
            if not path.startswith("/"):
                continue  # not a local path; nothing we can check
            records.append(
                {
                    "ref_id": ref_id,
                    "sample_id": props.get("sample_id"),
                    "s2_tile": props.get("s2_tile"),
                    "orbit_state": props.get("sat:orbit_state"),
                    "path": path,
                }
            )

    # Existence checks dominate the runtime on a network filesystem, and they
    # are independent, so fan them out.
    with ThreadPoolExecutor(max_workers=stat_workers) as pool:
        exists = list(pool.map(lambda r: os.path.exists(r["path"]), records))

    dangling = [r for r, ok in zip(records, exists) if not ok]
    return len(records), dangling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref-ids", nargs="+", help="ref_ids to check.")
    src.add_argument(
        "--ref-ids-file", type=str, help="File with one ref_id per line."
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=sorted(COLLECTIONS),
        default=sorted(COLLECTIONS),
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Write the per-item detail to CSV."
    )
    parser.add_argument(
        "--summary", type=str, default=None, help="Write the per-ref summary to CSV."
    )
    parser.add_argument("--workers", type=int, default=4, help="ref_ids in parallel.")
    args = parser.parse_args()

    if args.ref_ids_file:
        ref_ids = [
            line.strip()
            for line in Path(args.ref_ids_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        ref_ids = args.ref_ids

    logger.info(f"Checking {len(ref_ids)} ref_id(s) across {args.collections}")

    jobs = [(r, tag) for r in ref_ids for tag in args.collections]

    def _run(job):
        ref_id, tag = job
        try:
            total, dangling = scan_ref(ref_id, COLLECTIONS[tag])
            return ref_id, tag, total, dangling, None
        except Exception as exc:  # keep one bad ref from killing the sweep
            return ref_id, tag, 0, [], repr(exc)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ref_id, tag, total, dangling, err in pool.map(_run, jobs):
            if err:
                logger.error(f"{ref_id} {tag}: FAILED to scan -- {err}")
            elif dangling:
                orbits = sorted({d["orbit_state"] for d in dangling}, key=str)
                tiles = sorted({d["s2_tile"] for d in dangling}, key=str)
                logger.warning(
                    f"{ref_id} {tag}: {len(dangling)}/{total} DANGLING "
                    f"| orbits={orbits} | tiles={tiles}"
                )
            else:
                logger.info(f"{ref_id} {tag}: {total} items, clean")
            results.append(
                {
                    "ref_id": ref_id,
                    "collection": tag,
                    "items": total,
                    "dangling": len(dangling),
                    "orbits": ",".join(
                        sorted({str(d["orbit_state"]) for d in dangling})
                    ),
                    "tiles": ",".join(sorted({str(d["s2_tile"]) for d in dangling})),
                    "error": err or "",
                    "_detail": dangling,
                }
            )

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "_detail"} for r in results])
    broken = summary[summary["dangling"] > 0]

    print()
    print("=" * 78)
    print(f"{len(summary)} (ref_id, collection) pairs checked")
    print(f"{len(broken)} affected by dangling items")
    if not broken.empty:
        print()
        print(broken.sort_values("dangling", ascending=False).to_string(index=False))
        print()
        print("Affected ref_ids (these jobs cannot succeed until the patches are "
              "re-extracted or the STAC items removed):")
        for ref_id in sorted(broken["ref_id"].unique()):
            print(f"  {ref_id}")

    if args.summary:
        summary.to_csv(args.summary, index=False)
        logger.info(f"Summary written to {args.summary}")
    if args.output:
        detail = pd.DataFrame([d for r in results for d in r["_detail"]])
        detail.to_csv(args.output, index=False)
        logger.info(f"Detail written to {args.output}")


if __name__ == "__main__":
    main()
