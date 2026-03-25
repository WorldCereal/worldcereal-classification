#!/bin/bash

#SBATCH --account=vito                      # Name of Account (default is vito)
#SBATCH --partition=batch                   # Name of Partition (default is batch)
#SBATCH --job-name=worldcereal-inference     # Name of job
#SBATCH --output=inference_output_%j.out    # Standard output (%j = job ID)
#SBATCH --error=inference_error_%j.out     # Standard error (%j = job ID)
#SBATCH --ntasks=1                          # Number of CPU processes
#SBATCH --cpus-per-task=8                   # Number of CPU threads
#SBATCH --time=64:00:00                     # Wall time (format: d-hh:mm:ss)
#SBATCH --mem=30gb                          # Amount of memory (units: gb, mg, kb)
#SBATCH --gpus=1                            # Number of GPU; either a100:1, or just 1
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=christina.butsko.ext@vito.be # Email adress to notify events.

set -euo pipefail

# Necessary for GPU usage
module load CUDA

# Do not forget to activate the required environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate worldcereal-py311

# which python

# Calling the inference script.
# MODEL_DIR="/home/vito/butskoc/projects/worldcereal/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-LC10-CT24-WesternEurope-OutlierScoreEnabled-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202603031518"
# MODEL_DIR="/home/vito/butskoc/projects/worldcereal/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-LC10-CT24-Africa-TwoSeasons-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202603241533"
MODEL_DIR="/home/vito/butskoc/projects/worldcereal/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-LC10-CT24-Africa-AnnualSeason-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202603242003"

# Heads to produce. Can differ from the model's training seasons;
# the shared croptype head generalises across season masks.
# HEADS="LANDCOVER,CROPTYPE_S1,CROPTYPE_S2"
HEADS="LANDCOVER,CROPTYPE_ANNUAL"
if [[ "${HEADS}" == *"S1"* ]]; then
    output_dir_suffix="s1s2"
else
    output_dir_suffix="annual"
fi

# Base path for patch directories — each region is a subfolder.
PATCHES_BASE="/home/vito/butskoc/projects/worldcereal/data/GLOBAL_TEST_PATCHES/30D/WITH_REFDATA"
# Regions to run inference on (space-separated).
# REGIONS="Western_Europe Northern_Europe"
REGIONS="Northern_Africa Western_Africa Eastern_Africa Southern_Africa"

for REGION in ${REGIONS}; do
    echo "=== Running inference for region: ${REGION} ==="
    if [[ ! -d "${PATCHES_BASE}/${REGION}" ]]; then
        echo "WARNING: Patches directory not found for ${REGION}, skipping."
        continue
    fi
    srun python /home/vito/butskoc/worldcereal_finetuning/worldcereal-classification/scripts/training/finetuning/spatial_inference.py \
        --model_dir "${MODEL_DIR}" \
        --patches_dir "${PATCHES_BASE}/${REGION}" \
        --continents all \
        --output_dir "${MODEL_DIR}/inference_patches_${output_dir_suffix}" \
        --heads "${HEADS}" \
        --ct_mapping_key CROPTYPE24 \
        --device cuda
done