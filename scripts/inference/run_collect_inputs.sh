#!/bin/bash
set -euo pipefail

export OPENEO_AUTH_METHOD=""

# Make sure your current work directory is the root of the worldcereal repository
# ("worldcereal-classification")
# The next line should not be touched in principle.
PROCESS_CMD="scripts/inference/collect_inputs.py"

# Make sure you select the path to your WorldCereal Python environment
PYTHONPATH="/home/jeroendegerickx/miniconda3/envs/worldcereal-py311/bin/python"

# Parameters for spatial extent.
# Make sure to provide a valid path to a vector file containing the grid cells 
# for which you want to collect inputs.
GRID_PATH="./bbox/test.gpkg"
GRID_SIZE="20"

# Parameter specifying output folder
OUTPUT_FOLDER="./preprocessed_inputs"

# Parameters for temporal extent
# For using OPTION 2, provide start and end date
# START_DATE="2024-01-01"
# END_DATE="2024-04-30"

# For using OPTION 3, provide a year
YEAR="2024"

# Optional parameters
S1_ORBIT_STATE="ASCENDING"
PERIOD="month"
PARALLEL_JOBS="2"
# note below we set restart_failed and randomize_jobs to True


# Run extraction
"${PYTHONPATH}" "${PROCESS_CMD}" \
--grid_path "${GRID_PATH}" \
--grid_size "${GRID_SIZE}" \
--output_folder "${OUTPUT_FOLDER}" \
--compositing_window "${PERIOD}" \
--s1_orbit_state "${S1_ORBIT_STATE}" \
--year "${YEAR}" \
--parallel_jobs "${PARALLEL_JOBS}" \
--restart_failed \
--randomize_jobs