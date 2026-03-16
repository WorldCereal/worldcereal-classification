#!/bin/bash
set -euo pipefail

# Make sure your current work directory is the root of the worldcereal repository
# ("worldcereal-classification")
# The next line should NOT be touched.
PROCESS_CMD="scripts/inference/run_worldcereal_task_openeo.py"

# Make sure you select the path to your WorldCereal Python environment
PYTHONPATH="/PATH/TO/YOUR/WORLDCEREAL/PYTHON/ENVIRONMENT/bin/python"

# Parameters for spatial extent
# Make sure to provide a valid path to a vector file containing the grid cells
# for which you want to compute embeddings.
GRID_PATH="./bbox/test.gpkg"
GRID_SIZE="20" # km

# Parameter specifying output folder
OUTPUT_FOLDER="./embeddings"

# Parameters for temporal extent
# In this example, we provide fixed start and end dates
START_DATE="2024-01-01"
END_DATE="2024-12-31"

# Optional parameters
PARALLEL_JOBS="2"
# note below we set restart_failed to True, meaning that failed jobs
# will be restarted if you run the script again.


# Run embeddings computation workflow
"${PYTHONPATH}" "${PROCESS_CMD}" \
--task "embeddings" \
--grid_path "${GRID_PATH}" \
--grid_size "${GRID_SIZE}" \
--output_folder "${OUTPUT_FOLDER}" \
--start_date "${START_DATE}" \
--end_date "${END_DATE}" \
--parallel_jobs "${PARALLEL_JOBS}" \
--restart_failed