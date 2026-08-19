#!/bin/bash
set -euo pipefail

# RDM patch-to-point reprocessing campaign — LOCAL route (no openEO).
#
# Usage (each person runs one or more shards, any machine with the mounts):
#     bash run_ptp_campaign_rdm.sh <shard_idx> <n_shards> [refs_file]
# e.g. three people, one shard each:
#     person A:  bash run_ptp_campaign_rdm.sh 0 3
#     person B:  bash run_ptp_campaign_rdm.sh 1 3
#     person C:  bash run_ptp_campaign_rdm.sh 2 3
#
# Refs are assigned round-robin by line index (the list is largest-first, so
# every shard gets a fair mix). Resume-safe: refs whose output geoparquet
# already exists are skipped — just relaunch the same command after any
# interruption. One python per ref: a crash never blocks the next ref.
#
# Every ref is verified after extraction (--verify 20): 20 random points are
# checked against the openEO-era store; every difference must be explained
# (the documented openEO neighbour-pixel bug, orbit choice, float32 edge
# noise, or a geometry revision in the RDM). Unexplained differences FAIL the
# ref and stop this shard — that is intentional: report it, don't work around.

SHARD="${1:?usage: run_ptp_campaign_rdm.sh <shard_idx> <n_shards> [refs_file]}"
N_SHARDS="${2:?usage: run_ptp_campaign_rdm.sh <shard_idx> <n_shards> [refs_file]}"
REFS_FILE="${3:-$(dirname "$0")/rdm_campaign_refs.txt}"

PYBIN="${PYBIN:-/home/cbutsko/.conda/envs/worldcereal-py311/bin/python}"
CMD="$(dirname "$0")/ptp_campaign_rdm.py"

BASE="/vitodata/worldcereal/data/PATCH_TO_POINT_LOCAL"
OUT="$BASE/MERGED_PARQUETS"
CATALOG_CACHE="$BASE/_CATALOG"
AGERA5_CACHE="$BASE/_AGERA5_CACHE"
WORKERS="${WORKERS:-8}"
VERIFY_N="${VERIFY_N:-20}"

mkdir -p "$OUT" "$OUT/_verify" "$OUT/_stats" "$CATALOG_CACHE" "$AGERA5_CACHE"
# Shared campaign folder: several people write here. Group-writable with
# setgid dirs; harmless no-op for whoever doesn't own them.
chmod g+ws "$BASE" "$OUT" "$OUT/_verify" "$OUT/_stats" \
    "$CATALOG_CACHE" "$AGERA5_CACHE" 2>/dev/null || true

# Round-robin selection of this shard's refs
mapfile -t ALL < <(grep -vE '^\s*(#|$)' "$REFS_FILE" | awk '{print $1}')
MY_REFS=()
for i in "${!ALL[@]}"; do
    if [ $(( i % N_SHARDS )) -eq "$SHARD" ]; then MY_REFS+=("${ALL[$i]}"); fi
done
echo "Shard ${SHARD}/${N_SHARDS}: ${#MY_REFS[@]} refs | workers=$WORKERS verify=$VERIFY_N"
echo "Output: $OUT"

FAILED=()
for REF in "${MY_REFS[@]}"; do
    echo "================================================================"
    echo "=== $(date '+%F %T')  ${REF}"
    if ! "${PYBIN}" -u "${CMD}" --mode extract --ref-ids "${REF}" \
            --out-dir "$OUT" \
            --catalog-cache "$CATALOG_CACHE" \
            --agera5-cache "$AGERA5_CACHE" \
            --workers "$WORKERS" \
            --verify "$VERIFY_N"; then
        echo "!!! ${REF} FAILED (extraction error or verification FAIL)"
        echo "!!! see $OUT/_verify/${REF}.json if verification-related."
        echo "!!! Continuing with the next ref; report failures when done."
        FAILED+=("${REF}")
    fi
done

echo "================================================================"
echo "Shard ${SHARD} finished at $(date '+%F %T')."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED refs (${#FAILED[@]}):"
    printf '   %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All refs of this shard extracted and verified OK."
