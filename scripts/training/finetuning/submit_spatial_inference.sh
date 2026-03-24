#!/bin/bash

#SBATCH --account=vito                      # Name of Account (default is vito)
#SBATCH --partition=batch                   # Name of Partition (default is batch)
#SBATCH --job-name=worldcereal-inference     # Name of job
#SBATCH --output=outputlog.out              # Standard output written to file
#SBATCH --error=errorlog.out                # Standard error written to file
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
MODEL_DIR="/home/vito/butskoc/projects/worldcereal/models/presto-prometheo-dualtask-SeasonalMultiTaskLoss-DualBatchSampler-SouthernEurope_annual-month-augment=True-balance=True-timeexplicit=True-masking=enabled-run=202603131017"
# Heads to produce. Can differ from the model's training seasons;
# the shared croptype head generalises across season masks.
HEADS="LANDCOVER,CROPTYPE_S1,CROPTYPE_S2"
# HEADS="LANDCOVER,CROPTYPE_ANNUAL"

srun python /home/vito/butskoc/worldcereal_finetuning/worldcereal-classification/scripts/training/finetuning/spatial_inference.py \
    --model_dir "${MODEL_DIR}" \
    --patches_dir /home/vito/butskoc/projects/worldcereal/data/GLOBAL_TEST_PATCHES/30D/WITH_REFDATA/Southern_Europe \
    --continents all \
    --output_dir "${MODEL_DIR}/inference_patches_s1s2" \
    --heads "${HEADS}" \
    --ct_mapping_key CROPTYPE25 \
    --debug \
    --device cuda 