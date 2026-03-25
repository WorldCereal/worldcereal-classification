#!/bin/bash

#SBATCH --account=vito                      # Name of Account (default is vito)
#SBATCH --partition=batch                   # Name of Partition (default is batch)
#SBATCH --job-name=worldcereal-training     # Name of job
#SBATCH --output=training_output_%j.out     # Standard output (%j = job ID)
#SBATCH --error=training_error_%j.out      # Standard error (%j = job ID)
#SBATCH --ntasks=1                          # Number of CPU processes
#SBATCH --cpus-per-task=8                   # Number of CPU threads
#SBATCH --time=64:00:00                     # Wall time (format: d-hh:mm:ss)
#SBATCH --mem=40gb                          # Amount of memory (units: gb, mg, kb)
#SBATCH --gpus=a100:1                       # Number of GPU; either a100:1, or just 1
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=christina.butsko.ext@vito.be # Email adress to notify events.

set -euo pipefail

# Necessary for GPU usage
module load CUDA

# Do not forget to activate the required environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate worldcereal-py311

# which python
# # --season_ids tc-s1 tc-s2 \

# Calling the training script.
srun python /home/vito/butskoc/worldcereal_finetuning/worldcereal-classification/scripts/training/finetuning/finetune_presto.py \
    --experiment_tag dualtask-SeasonalMultiTaskLoss-LC10-CT24-Africa-AnnualSeason \
    --initial_mapping LANDCOVER10 \
    --landcover_classes_key LANDCOVER10 \
    --croptype_classes_key CROPTYPE24 \
    --val_samples_file /home/vito/butskoc/projects/worldcereal/data/balanced_splits/val_split_h3l5.csv \
    --test_samples_file /home/vito/butskoc/projects/worldcereal/data/balanced_splits/test_split_h3l5.csv \
    --ignore_samples_file /home/vito/butskoc/projects/worldcereal/data/balanced_splits/ignore_split_h3l5.csv \
    --timestep_freq month \
    --finetune_regions "Northern Africa, Middle Africa, Western Africa, Eastern Africa, Southern Africa" \
    --time_explicit \
    --use_class_balancing \
    --min_samples_per_class 200 \
    --train_min_season_coverage 0.5 \
    --season_ids tc-annual \
    --outlier_mode drop_candidate \
    --augment \
    --enable_masking \
    --max_timesteps_trim 18 \
    --head_only_training 3 \
    --post_unfreeze_warmup_epochs 2 \
    --log_tensorboard \
    --explicit_training_dataframe /home/vito/butskoc/projects/worldcereal/merged_319_wide.parquet \
    --base_output_dir /home/vito/butskoc/projects/worldcereal/models \