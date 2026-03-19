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
OUTPUT_FOLDER="./outputs/maps/croptype_mapping"

# Parameters for temporal extent
# We just specify a year, the system will retieve the best processing window
# in function of the local crop calendar.
YEAR=2022

## Alternatively, you can specify the season specifications directly as a JSON string,
## and provide an explicit start and end date of the processing window encompassing both seasons.
# SEASON_SPECIFICATIONS='{"s1": ["2024-10-01", "2025-06-30"], "s2": ["2025-03-01", "2025-09-30"]}'
# START_DATE="2024-10-01"
# END_DATE="2025-09-30"
## Make sure to add the following flags to the function call below:
# --start_date "${START_DATE}" \
# --end_date "${END_DATE}" \
# --season-specifications-json "${SEASON_SPECIFICATIONS}" \

## Product to generate: cropland or croptype
PRODUCT="croptype"

## Note below we have set the following flags:
# --enable-cropland-head \  --> meaning that the model will produce a cropland map.
# --enable-croptype-head \  --> meaning that the model will produce a croptype map.
# --enforce-cropland-gate \  --> meaning that the croptype classification will be masked using the cropland product.
# --merge-classification-products \  --> this will merge the cropland and croptype classification products into a single output.
# --class-probabilities \  --> this will output class probabilities in addition to the final classification map.

##  Postprocessing options
## (note that we activate cropland post-processing by setting 
# --enable-cropland-postprocess and
# --enable-croptype-postprocess flags below.
POSTPROCESS_METHOD="majority_vote" # options are "majority_vote" or "smooth_probabilities"
POSTPROCESS_KERNEL_SIZE=3 # only used if method is "majority_vote"

##  Additionally export embeddings and NDVI time series
##  Add the following flags to the function call below:
# --export-embeddings \
# --export-ndvi \
# --driver_memory "12g"

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
--enable-croptype-postprocess \
--cropland-postprocess-method "${POSTPROCESS_METHOD}" \
--cropland-postprocess-kernel-size "${POSTPROCESS_KERNEL_SIZE}" \
--croptype-postprocess-method "${POSTPROCESS_METHOD}" \
--croptype-postprocess-kernel-size "${POSTPROCESS_KERNEL_SIZE}" \
--class-probabilities \
--enable-cropland-head \
--enable-croptype-head \
--enforce-cropland-gate \
--merge-classification-products

