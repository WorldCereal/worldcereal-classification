#!/bin/bash
set -euo pipefail

# Python and extraction command
PYTHONPATH="/home/vito/butskoc/.conda/envs/worldcereal-py311/bin/python"
PROCESS_CMD="/home/vito/butskoc/worldcereal_finetuning/worldcereal-classification/scripts/training/finetuning/finetune_presto.py"

# Parameters
PERIOD="month"
OUT_DIR="/home/vito/butskoc/worldcereal_finetuning/models"
EXPERIMENT_TAG="dualtask-SeasonalMultiTaskLoss-LC10-CT24-Europe-OutlierScoreEnabled"
INITIAL_MAPPING="LANDCOVER10"
LANDCOVER_CLASSES_KEY="LANDCOVER10"
CROPTYPE_CLASSES_KEY="CROPTYPE24"
VAL_SAMPLES_FILE="/home/vito/butskoc/projects/worldcereal/data/balanced_splits/val_samples.csv"
TEST_SAMPLES_FILE="/home/vito/butskoc/projects/worldcereal/data/balanced_splits/test_samples.csv"
IGNORE_SAMPLES_FILE="/home/vito/butskoc/projects/worldcereal/data/balanced_splits/ignore_samples.csv"
PARQUET_PATH="/home/vito/butskoc/worldcereal_finetuning/merged_319_wide.parquet"

# Run extraction
"${PYTHONPATH}" "${PROCESS_CMD}" \
--base_output_dir "${OUT_DIR}" \
--experiment_tag "${EXPERIMENT_TAG}" \
--initial_mapping "${INITIAL_MAPPING}" \
--landcover_classes_key "${LANDCOVER_CLASSES_KEY}" \
--croptype_classes_key "${CROPTYPE_CLASSES_KEY}" \
--val_samples_file "${VAL_SAMPLES_FILE}" \
--test_samples_file "${TEST_SAMPLES_FILE}" \
--ignore_samples_file "${IGNORE_SAMPLES_FILE}" \
--finetune_regions "Western Europe, Southern Europe, Eastern Europe, Northern Europe" \
--outlier_mode "drop_candidate" \
--timestep_freq "${PERIOD}" \
--time_explicit \
--use_balancing \
--spatial_bin_size_deg "3.0" \
--augment \
--enable_masking \
--max_timesteps_trim "18" \
--head_only_training "3" \
--post_unfreeze_warmup_epochs "2" \
--log_tensorboard \
--explicit_training_dataframe "${PARQUET_PATH}"\



 