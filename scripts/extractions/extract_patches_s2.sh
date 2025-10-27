#!/bin/bash
set -euo pipefail

# export OPENEO_AUTH_METHOD="client_credentials"
# export OPENEO_AUTH_METHOD=""

# Set the root dir
OUTDIR_ROOT="/data/worldcereal_data/EXTRACTIONS/SENTINEL_2/"

# Python and extraction command
PYTHONPATH="/home/wcextractions/.conda/envs/worldcereal/bin/python"
EXTRACT_CMD="/home/wcextractions/Private/worldcereal-classification/scripts/extractions/extract.py"

# Extraction parameters
PYTHON_MEMORY="1800m"
DRIVER_MEMORY="2G"
DRIVER_MEMORY_OVERHEAD="2G" # normally 1GB
EXECUTOR_MEMORY="1800m"
EXECUTOR_MEMORY_OVERHEAD="2G"
MAX_LOCATIONS="250"
PARALLEL_JOBS="8"
# ORGANIZATION_ID="10523"  # inseason-poc

# List of REF_IDs to process
REF_IDS=(

# ============== DONE ==============
# 2025_ZMB_CIMMYT_POLY_111
# 2019_CZE_LPIS_POLY_110
# 2025_ZWE_CIMMYT_POLY_111

# ============== TODO ==============
# 2020_CZE_LPIS_POLY_110 # Error downloading asset openEO_2020_CZE_LPIS_POLY_110_471097.nc. need to investigate
2021_CZE_LPIS_POLY_110 
2022_CZE_LPIS_POLY_110
2023_CZE_LPIS_POLY_110
2024_CZE_LPIS_POLY_110

)

for REF_ID in "${REF_IDS[@]}"; do
    # Build paths
    REFDATA_FILE="/vitodata/worldcereal/data/RDM/${REF_ID}/harmonized/${REF_ID}.geoparquet"
    OUTDIR="${OUTDIR_ROOT}${REF_ID}"

    echo
    printf '%0.s-' {1..50}
    echo
    echo "Launching extraction for REF_ID: ${REF_ID}"
    echo "  Reference file : ${REFDATA_FILE}"
    echo "  Output folder  : ${OUTDIR}"
    echo
    printf '%0.s-' {1..50}
    echo

    # Run extraction
    "${PYTHONPATH}" "${EXTRACT_CMD}" PATCH_SENTINEL2 \
    "${OUTDIR}" "${REFDATA_FILE}" \
    --ref_id "${REF_ID}" \
    --python_memory "${PYTHON_MEMORY}" \
    --driver_memory "${DRIVER_MEMORY}" \
    --driver_memory_overhead "${DRIVER_MEMORY_OVERHEAD}" \
    --executor_memory "${EXECUTOR_MEMORY}" \
    --executor_memory_overhead "${EXECUTOR_MEMORY_OVERHEAD}" \
    --parallel_jobs "${PARALLEL_JOBS}" \
    --max_locations "${MAX_LOCATIONS}" \
    --write_stac_api 1 \
    --extract_value 1 \
    --restart_failed \
    --check_existing_extractions True
    # --organization_id "${ORGANIZATION_ID}"
    # --image_name "registry.stag.waw3-1.openeo-int.v1.dataspace.copernicus.eu/dev/openeo-geotrellis-kube:20250325-2415"

    echo
    printf '%0.s-' {1..50}
    echo "Finished extraction for ${REF_ID}"
    printf '%0.s-' {1..50}
    echo
done

echo "All extractions completed!"
