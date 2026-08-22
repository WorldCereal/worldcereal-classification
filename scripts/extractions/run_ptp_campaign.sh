#!/bin/bash
set -euo pipefail

# LOCAL (openEO-free) patch-to-point extraction driver: one
# ptp_engine.py invocation per host, hosts read from a hosts file
# and sharded round-robin (nearest-centre pixel convention; see
# ptp_engine.py header for the validated recipe).
#
# Usage:
#     run_ptp_campaign.sh <shard_idx> <n_shards> [hosts_file]
# hosts_file defaults to inpatch_noncrop_hosts.txt next to this script
# (format: one host ref_id per line, optional point count, '#' comments).
#
# In screen, 3 shards side by side (hosts are assigned round-robin, so
# each shard gets a mix of big and small hosts):
#     screen -S p2p0;  bash run_ptp_campaign.sh 0 3
#     screen -S p2p1;  bash run_ptp_campaign.sh 1 3
#     screen -S p2p2;  bash run_ptp_campaign.sh 2 3
# Or single process (slower, ~2x wall clock):
#     bash run_ptp_campaign.sh 0 1
#
# Environment overrides (GT_DIR and MERGED_DIR are required):
#     PYBIN         python interpreter                 (default: python3)
#     GT_DIR        ground-truth dir holding <host>.geoparquet files plus
#                   provenance.parquet                 (required)
#     MERGED_DIR    output dir for <host>_<RUN_SUFFIX>.geoparquet (required)
#     RUN_SUFFIX    output filename suffix             (default: LOCAL)
#     AGERA5_CACHE  AGERA5 monthly composite cache dir (default:
#                   <MERGED_DIR>/_agera5_cache)
#     WORKERS       workers per shard                  (default: 8; 3 shards
#                   x 8 = 24 concurrent NFS readers, keep it civil)
#
# Example — the in-patch hard-negative campaign this driver was built for:
#     export PYBIN=/home/cbutsko/.conda/envs/worldcereal-py311/bin/python
#     export GT_DIR=/vitodata/worldcereal/tmp/cbutsko/EXTRACTIONS/WORLDCEREAL/PATCH_TO_POINT/_GROUND_TRUTH_INPATCH_NONCROP
#     export MERGED_DIR=/vitodata/worldcereal/tmp/cbutsko/EXTRACTIONS/WORLDCEREAL/PATCH_TO_POINT/MERGED_PARQUETS_INPATCH_NONCROP
#     export RUN_SUFFIX=INPATCH-NONCROP
#     export AGERA5_CACHE=/vitodata/worldcereal/tmp/cbutsko/EXTRACTIONS/WORLDCEREAL/AGERA5_MONTHLY_CACHE
#     bash run_ptp_campaign.sh 0 3
#
# Resume-safe: hosts whose output geoparquet already exists are skipped, so
# just relaunch the same shard after any interruption.
#
# When ALL shards are done, finish with (any single shell, from the repo
# root; the inpatch driver's openEO run stage is gone — extraction happens
# via this script, rekey/gate postprocess its outputs):
#     python scripts/extractions/ptp_campaign_inpatch.py --stage rekey \
#         --root-folder <campaign root>
#     python scripts/extractions/ptp_campaign_inpatch.py --stage gate \
#         --root-folder <campaign root> \
#         --schema-reference <merged parquet from a regular run>

SHARD="${1:-0}"
N_SHARDS="${2:-1}"
HOSTS_FILE="${3:-$(dirname "$0")/inpatch_noncrop_hosts.txt}"

PYBIN="${PYBIN:-python3}"
CMD="$(dirname "$0")/ptp_engine.py"
GT_DIR="${GT_DIR:?set GT_DIR to the ground-truth dir (<host>.geoparquet files plus provenance.parquet)}"
MERGED_DIR="${MERGED_DIR:?set MERGED_DIR to the output dir for <host>_<RUN_SUFFIX>.geoparquet}"
RUN_SUFFIX="${RUN_SUFFIX:-LOCAL}"
AGERA5_CACHE="${AGERA5_CACHE:-${MERGED_DIR}/_agera5_cache}"   # the extractor's own default
WORKERS="${WORKERS:-8}"

if [ ! -f "${HOSTS_FILE}" ]; then
    echo "Hosts file not found: ${HOSTS_FILE}" >&2
    exit 1
fi

# Hosts in file order: first whitespace-separated field of every non-comment,
# non-blank line. Round-robin sharding by line index then interleaves sizes
# when the file is sorted largest first.
HOSTS=()
while read -r HOST REST || [ -n "${HOST}" ]; do
    if [ -z "${HOST}" ] || [ "${HOST:0:1}" = "#" ]; then
        continue
    fi
    HOSTS+=("${HOST}")
done < "${HOSTS_FILE}"

# Round-robin shard selection
MY_HOSTS=()
for i in "${!HOSTS[@]}"; do
    if [ $(( i % N_SHARDS )) -eq "$SHARD" ]; then
        MY_HOSTS+=("${HOSTS[$i]}")
    fi
done

echo "Shard ${SHARD}/${N_SHARDS}: ${#MY_HOSTS[@]} of ${#HOSTS[@]} hosts, workers=${WORKERS}"
echo "Output: ${MERGED_DIR}"

# One host per invocation so a crash on host N never blocks host N+1, and the
# log clearly delimits hosts. Existing outputs are skipped inside the tool.
FAILED=()
for HOST in "${MY_HOSTS[@]}"; do
    echo "================================================================"
    echo "=== $(date '+%F %T')  ${HOST}"
    if ! "${PYBIN}" -u "${CMD}" --mode extract --hosts "${HOST}" \
            --gt-dir "${GT_DIR}" --merged-dir "${MERGED_DIR}" \
            --run-suffix "${RUN_SUFFIX}" --agera5-cache "${AGERA5_CACHE}" \
            --workers "${WORKERS}"; then
        echo "!!! ${HOST} FAILED — continuing with the next host"
        FAILED+=("${HOST}")
    fi
done

echo "================================================================"
echo "Shard ${SHARD} finished at $(date '+%F %T')."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED hosts (relaunch this shard to retry them after diagnosis):"
    printf '   %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All hosts of this shard extracted OK."
