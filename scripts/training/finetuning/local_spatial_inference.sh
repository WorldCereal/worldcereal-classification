#!/bin/bash
set -euo pipefail

# Python and spatial inference command
PYTHON_BIN="/home/vito/butskoc/.conda/envs/worldcereal-py311/bin/python"
PROCESS_CMD="/home/vito/butskoc/worldcereal_finetuning/worldcereal-classification/scripts/training/finetuning/spatial_inference.py"

# Required parameters
# MODEL_DIR="/home/vito/butskoc/worldcereal_finetuning/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-LC10-CT25-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202602201207"
# MODEL_DIR="/home/vito/butskoc/projects/worldcereal/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-KENYA-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202602061520"
# MODEL_DIR="/home/vito/butskoc/projects/worldcereal/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202601240103"
# MODEL_DIR="/home/vito/butskoc/worldcereal_finetuning/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-LC10-CT25-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202602251409"
MODEL_DIR="/home/vito/butskoc/projects/worldcereal/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-LC10-CT24-WesternEurope-OutlierScoreEnabled-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202603031518"
PATCHES_DIR="/home/vito/butskoc/projects/worldcereal/data/GLOBAL_TEST_PATCHES/30D/WITH_REFDATA/Western_Europe"

# Optional parameters
CONTINENTS="all"                               # e.g. all OR Africa,Europe
# OUTPUT_DIR="${MODEL_DIR}/inference_patches"    # leave as default or change
# OUTPUT_DIR="/home/vito/butskoc/worldcereal_finetuning/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202601240103/inference_patches"
# OUTPUT_DIR="/home/vito/butskoc/worldcereal_finetuning/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-KENYA-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202602061520/inference_patches"
# OUTPUT_DIR="/home/vito/butskoc/worldcereal_finetuning/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-LC10-CT25-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202602251409/inference_patches"
OUTPUT_DIR="/home/vito/butskoc/projects/worldcereal/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-LC10-CT24-WesternEurope-OutlierScoreEnabled-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202603031518/inference_patches"
HEADS="LANDCOVER,CROPTYPE_S1,CROPTYPE_S2"
LIMIT=""                                       # e.g. 10, keep empty for all
OVERWRITE="false"                               # true|false
DEVICE="cuda"                                       # e.g. cpu or cuda, keep empty for default
DEBUG_SEED=""                                   # e.g. 1234, keep empty for random
CT_MAPPING_KEY="CROPTYPE24"                      # e.g. CROPTYPE28 for base models; CROPTYPE24 for current

CMD=(
  "${PYTHON_BIN}" "${PROCESS_CMD}"
  --model_dir "${MODEL_DIR}"
  --patches_dir "${PATCHES_DIR}"
  --continents "${CONTINENTS}"
  --output_dir "${OUTPUT_DIR}"
  --heads "${HEADS}"
  --ct_mapping_key "${CT_MAPPING_KEY}"
  --debug
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  CMD+=(--overwrite)
fi

if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

if [[ -n "${DEBUG_SEED}" ]]; then
  CMD+=(--debug_seed "${DEBUG_SEED}")
fi

"${CMD[@]}"
