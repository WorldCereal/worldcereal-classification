#!/bin/bash
set -euo pipefail

# Make sure your current work directory is the root of the worldcereal repository
# ("worldcereal-classification")
# The next line should NOT be touched.
PROCESS_CMD="scripts/inference/run_worldcereal_task_openeo.py"

# Make sure you select the path to your WorldCereal Python environment
PYTHONPATH="/PATH/TO/YOUR/WORLDCEREAL/PYTHON/ENVIRONMENT/bin/python"

# Parameters for spatial extent.
# Make sure to provide a valid path to a vector file containing the grid cells 
# for which you want to collect inputs.
GRID_PATH="./bbox/test.gpkg"
GRID_SIZE="20" #km

# Parameter specifying output folder
OUTPUT_FOLDER="./preprocessed_inputs"

# Parameters for temporal extent
# In this example, we just provide a year.
# The system will automatically determine the appropriate start and
# end dates based on the local crop calendars.
YEAR="2024"

# Optional parameters
S1_ORBIT_STATE="ASCENDING"
COMPOSITING_WINDOW="month" # either "month" or "dekad"
PARALLEL_JOBS="2"
# note below we set restart_failed to True, meaning that failed jobs
# will be restarted if you run the script again.

# Run inputs collection
"${PYTHONPATH}" "${PROCESS_CMD}" \
--task "inputs" \
--grid_path "${GRID_PATH}" \
--grid_size "${GRID_SIZE}" \
--output_folder "${OUTPUT_FOLDER}" \
--compositing_window "${COMPOSITING_WINDOW}" \
--s1_orbit_state "${S1_ORBIT_STATE}" \
--year "${YEAR}" \
--parallel_jobs "${PARALLEL_JOBS}" \
--restart_failed