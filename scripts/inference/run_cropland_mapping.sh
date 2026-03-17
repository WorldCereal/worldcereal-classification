#!/bin/bash
set -euo pipefail

# Make sure your current work directory is the root of the worldcereal repository
# ("worldcereal-classification")
# The next line should NOT be touched.
PROCESS_CMD="scripts/inference/run_worldcereal_task_openeo.py"

# Make sure you select the path to your WorldCereal Python environment
PYTHONPATH="/PATH/TO/YOUR/WORLDCEREAL/PYTHON/ENVIRONMENT/bin/python"

# Parameters for spatial extent
# here we provide a custom bounding box and epsg
MINX=664000
MINY=5611134
MAXX=665000
MAXY=5612134
EPSG=32631
GRID_SIZE=20 # km

# minx, miny, maxx, maxy = (664000, 5611134, 684000, 5631134)  # Large test
# minx, miny, maxx, maxy = (634000, 5601134, 684000, 5651134)  # Very large test
    
# Parameter specifying output folder
OUTPUT_FOLDER="./outputs/maps/cropland_mapping"

# Parameters for temporal extent
# We just specify a year, the system will retieve the best processing window
# in function of the local crop calendar.
YEAR=2021

# Product to generate: cropland or croptype
PRODUCT="cropland"

# Postprocessing options
# (note that we activate cropland post-processing by setting enable-cropland-postprocess flag below.
POSTPROCESS_METHOD="majority_vote" # options are "majority_vote" or "smooth_probabilities"
POSTPROCESS_KERNEL_SIZE=3 # only used if method is "majority_vote"

# Optional parameters
PARALLEL_JOBS="2"

# note below we set restart_failed to True, meaning that failed jobs
# will be restarted if you run the script again.

# Run mapping
"${PYTHONPATH}" "${PROCESS_CMD}" \
--task "classification" \
--bbox "${MINX}" "${MINY}" "${MAXX}" "${MAXY}" \
--bbox_epsg "${EPSG}" \
--grid_size "${GRID_SIZE}" \
--year "${YEAR}" \
--product "${PRODUCT}" \
--output_folder "${OUTPUT_FOLDER}" \
--parallel_jobs "${PARALLEL_JOBS}" \
--restart_failed \
--enable-cropland-postprocess \
--cropland-postprocess-method "${POSTPROCESS_METHOD}" \
--cropland-postprocess-kernel-size "${POSTPROCESS_KERNEL_SIZE}"