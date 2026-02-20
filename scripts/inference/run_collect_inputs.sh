#!/bin/bash
set -euo pipefail

export OPENEO_AUTH_METHOD=""

# Python and extraction command
PYTHONPATH="/.conda/envs/worldcereal-classification/bin/python"
PROCESS_CMD="/worldcereal-classification/scripts/inference/collect_inputs.py"

# Parameters
PERIOD="month"

GRID_PATH="collect_inputs_test_belgium_good.parquet"
OUTPUT_FOLDER="inference_patches"

TILE_NAME_COL="tile_name"

EXTRACTIONS_START_DATE="2024-01-01"
EXTRACTIONS_END_DATE="2024-04-30"

PARALLEL_JOBS="2"

# Run extraction
"${PYTHONPATH}" "${PROCESS_CMD}" \
--grid_path "${GRID_PATH}" \
--output_folder "${OUTPUT_FOLDER}" \
--tile_name_col "${TILE_NAME_COL}" \
--compositing_window "${PERIOD}" \
--extractions_start_date "${EXTRACTIONS_START_DATE}" \
--extractions_end_date "${EXTRACTIONS_END_DATE}" \
--parallel_jobs "${PARALLEL_JOBS}" \
--restart_failed
# --overwrite_job_df

