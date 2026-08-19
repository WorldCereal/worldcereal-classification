# RDM patch-to-point reprocessing — how to run your shard

We are re-extracting all patch-based RDM datasets locally (no openEO, no
credits), because the openEO route returned a neighbouring pixel's values for
~50% of points (bug localized in `NetCDFCollection.scala`. 
The local route is validated bit-exact.

## Prerequisites

* A Terrascope VM with the standard mounts (`/data/worldcereal_data`,
  `/vitodata/worldcereal`) and the `worldcereal`-like conda env
  (set `PYBIN=/path/to/python` to the correct one).
* This repo checked out on branch `local-patch-to-point`.
* No openEO account, no credits, no authentication needed.

## Run

Agree who takes which shard number (0, 1, 2 for three people), then in a
screen session:

```bash
cd scripts/extractions
screen -S local-ptp-rdm0
bash run_ptp_campaign_rdm.sh 0 3      # person A
# person B: bash run_ptp_campaign_rdm.sh 1 3
# person C: bash run_ptp_campaign_rdm.sh 2 3
```

That's all. Each ref: index from STAC (seconds) → select samples from the
harmonized RDM file (the authoritative source) → extract from the patch
NetCDFs → **verify 20 random points against the old openEO store** → write
`<ref_id>.geoparquet` to the shared output folder.

* **Interrupted?** Relaunch the same command — completed refs are skipped.
* **A ref prints FAILED?** The shard continues with the next ref and lists
  all failures at the end. Do NOT delete outputs or retry blindly — check
  `MERGED_PARQUETS/_verify/<ref>.json` and report back.
  `FAIL(divergence_rate)` means the ref's RDM geometries changed a lot since
  the openEO era — needs a human look, not a rerun.
* **Machine etiquette**: one shard per machine; ~8 workers is polite to the
  NFS filer. Memory footprint is bounded (~2 GB); expect 2 min (small ref)
  to a few hours (LUCAS/ESP monsters) per ref.

## Outputs

```
/vitodata/worldcereal/data/PATCH_TO_POINT_LOCAL/
    MERGED_PARQUETS/<ref_id>.geoparquet     ← the deliverables
    MERGED_PARQUETS/_verify/<ref_id>.json   ← per-ref verification certificate
    MERGED_PARQUETS/_rdm_campaign_stats.csv ← selection/assignment stats
    _CATALOG/  _AGERA5_CACHE/               ← shared caches (auto-managed)
```

Schema: the canonical long format (one row per sample × month), identical to
the old store minus the anomaly columns (added downstream as before).

## What "verified" means

Each ref's certificate proves that, for the checked points, our values are
either bit-identical to the openEO-era store or differ ONLY in explained
ways: the openEO pixel-shift bug (we recompute the neighbouring pixel from
the raw patch and match it exactly), S1 orbit choice, float32 edge-month
noise (±1 DN), documented aux tolerances, or RDM geometry revisions
(reported with the pixel offset). Anything else fails the ref loudly.
