#!/bin/bash
set -euo pipefail

export OPENEO_AUTH_METHOD=""

# Make sure your current work directory is the root of the worldcereal repository
# ("worldcereal-classification")
# The next line should not be touched in principle.
PROCESS_CMD="scripts/inference/cropland_croptype_mapping.py"

# Make sure you select the path to your WorldCereal Python environment
PYTHONPATH="/home/jeroendegerickx/miniconda3/envs/worldcereal-py311/bin/python"

# Parameters for spatial extent (grid file)
GRID_PATH="./bbox/test.gpkg"
# Leave empty to use AOI geometries as-is
GRID_SIZE="20"
# TARGET_EPSG="3857"

# Parameters for temporal extent
# For using OPTION 2, provide start and end date
START_DATE="2024-01-01"
END_DATE="2024-12-31"
# For using OPTION 3, provide a year
# YEAR="2024"

# Product to generate: cropland or croptype
PRODUCT="cropland"

# Parameter specifying output folder
OUTPUT_FOLDER="./outputs/maps"

# Optional parameters
PARALLEL_JOBS="2"
# S1_ORBIT_STATE="ASCENDING"

# note below we set restart_failed and randomize_jobs to True

# Run mapping
"${PYTHONPATH}" "${PROCESS_CMD}" \
--grid_path "${GRID_PATH}" \
--grid_size "${GRID_SIZE}" \
--target-epsg "${TARGET_EPSG}"} \
--start_date "${START_DATE}" \
--end_date "${END_DATE}" \
--product "${PRODUCT}" \
--output_folder "${OUTPUT_FOLDER}" \
--parallel-jobs "${PARALLEL_JOBS}" \
--restart_failed \
--randomize_jobs \
--simplify_logging