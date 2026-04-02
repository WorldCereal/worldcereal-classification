#!/usr/bin/env bash
# ==============================================================================
# WorldCereal Presto Finetuning — example run script
# ==============================================================================
# Run this script from any directory:
#
#   bash /path/to/finetune_presto.sh
#
# All arguments below are forwarded to:
#   scripts/training/finetuning/finetune_presto.py
#
# Convention used in this file:
#   - Arguments with a value set  → the ones you most likely need to change.
#   - Arguments that are commented out → optional features shown with their
#     defaults so you know what is possible.
# ==============================================================================

# Update the path below to point to the finetuning script on your machine.
FINETUNE_SCRIPT="/home/kristofvt/git/worldcereal-classification/scripts/training/finetuning/finetune_presto.py"

# Arguments are collected in an array so that comment lines can be freely
# placed between them without breaking bash parsing.
ARGS=(

    # --------------------------------------------------------------------------
    # EXPERIMENT IDENTITY
    # --------------------------------------------------------------------------

    # Short descriptive tag appended to the auto-generated experiment folder name.
    --experiment_tag "sudan-finetuning"

    # Root directory where the timestamped experiment folder will be created.
    --base_output_dir "."

    # --------------------------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------------------------

    # One or more parquet files with extracted time-series samples.
    # Omit to use the default global extraction list (requires VPN / cluster access).
    --parquet_files "/home/kristofvt/git/worldcereal-classification/scripts/sandbox-sudan/sudan_extractions.parquet"

    # Temporal resolution of the input data.
    #   "month"  → 12 timesteps / year  (default, most common)
    #   "dekad"  → 36 timesteps / year
    --timestep_freq "month"

    # Maximum number of timesteps to keep after trimming.
    # "auto" (default) keeps all timesteps present in the data.
    # Set an integer to hard-cap the window (e.g. 18 for a 1.5-year window;
    # also enables temporal augmentation when --augment is set).
    --max_timesteps_trim 18

    # Optional: CSV files that pin specific sample IDs to val / test / ignore.
    # Each CSV must contain a "sample_id" column.
    # When omitted, a random stratified 70/15/15 split is used.
    # --val_samples_file     "val_split.csv"
    # --test_samples_file    "test_split.csv"
    # --ignore_samples_file  "ignore_split.csv"

    # Optional: restrict training to specific world regions.
    # Comma-separated UN M49 macro-region names, e.g. "Eastern Africa,Western Africa".
    # Omit to use all available samples.
    # --finetune_regions "Eastern Africa"

    # Optional: path to the intermediate wide-format parquet (the expensive pivot/merge
    # step output). If the file already exists it is reused, skipping data preparation.
    # If it does not exist yet it is created there so future runs can reuse it.
    # On HPC the shared /projects/worldcereal directory is used automatically
    # when this is not set.
    --wide_parquet_path "./wide.parquet"

    # --------------------------------------------------------------------------
    # CLASS MAPPINGS
    # --------------------------------------------------------------------------

    # Path to a custom JSON file mapping ewoc_codes to class label strings.
    # Required structure: { "KEY": { "ewoc_code": "label", ... }, ... }
    # Omit to fetch the latest official mappings from SharePoint
    # (requires SharePoint credentials configured on your machine).
    --class_mappings_file "/home/kristofvt/git/worldcereal-classification/src/worldcereal/data/croptype_mappings/class_mappings.json"

    # Key inside the mappings JSON for the landcover head.
    # Default: LANDCOVER10
    --landcover_classes_key "LANDCOVER10"

    # Key inside the mappings JSON for the crop-type head.
    # Common values: CROPTYPE24  CROPTYPE28  CROPTYPE9
    --croptype_classes_key "CROPTYPE28"

    # Mapping key used when assigning initial ewoc_code → label during data prep.
    # Usually matches --landcover_classes_key.
    --initial_mapping "LANDCOVER10"

    # Comma-separated landcover labels treated as "cropland" for the binary
    # cropland gate inside the crop-type head. Optionally include perennial
    # crops here if your use case requires it.
    --landcover_cropland_classes "temporary_crops"

    # Classes with fewer training samples than this threshold are dropped
    # from all three splits (train / val / test).
    --min_samples_per_class 200

    # --------------------------------------------------------------------------
    # SEASON DEFINITION
    # Use EITHER --season_windows OR --season_ids — never both.
    # --------------------------------------------------------------------------

    # --season_windows  (recommended when you know the local growing season)
    #   JSON object mapping a season name to [start_date, end_date].
    #   Dates repeat annually — only the month and day matter.
    #   A window crossing a year boundary is supported (year_offset=1).
    #   Example single season:  '{"s1": ["2021-04-01", "2021-09-30"]}'
    #   Example two seasons:    '{"s1": ["2021-04-01", "2021-09-30"], "s2": ["2021-10-01", "2022-03-31"]}'
    --season_windows '{"s1": ["2021-04-01", "2021-09-30"]}'

    # --season_ids  (use official WorldCereal crop-calendar seasons instead)
    #   Space-separated season identifiers.
    #   Common values: tc-s1  tc-s2  tc-annual
    #   Omit both --season_windows and --season_ids to default to tc-s1 + tc-s2.
    # --season_ids tc-s1 tc-s2

    # Fraction of a season's timestep slots that must fall inside the selected
    # window for that season to contribute crop-type supervision during training.
    # Lower = more permissive (important when augmentation shifts the window).
    # Range: 0.0–1.0   Default: 0.5
    --train_min_season_coverage 0.5

    # Same threshold for val / test splits.
    # Default 1.0 requires all slots to be present.
    # Lower (e.g. 0.8) if your season is longer than the timestep window.
    # --eval_min_season_coverage 1.0

    # --------------------------------------------------------------------------
    # DATA QUALITY & OUTLIER FILTERING
    # --------------------------------------------------------------------------

    # How to handle samples flagged as potential outliers.
    #   "keep"           — keep everything  (default)
    #   "drop_candidate" — remove the most obvious outliers  (recommended)
    #   "drop_suspect"   — remove candidates + suspicious samples
    #   "drop_flagged"   — remove all flagged samples  (most aggressive)
    --outlier_mode "drop_candidate"

    # Low-quality sample filtering for val / test (based on combined sample weight).
    # A sample is removed when it is BOTH below the per-ref_id percentile AND
    # below the hard floor. Leave --eval_weight_floor unset to disable entirely.
    # --eval_weight_floor 0.5          # absolute quality-weight threshold
    # --eval_weight_percentile 20.0    # relative bottom-N% within each dataset
    # --eval_min_class_samples 10      # never let a class drop below N samples

    # --------------------------------------------------------------------------
    # AUGMENTATION & SENSOR MASKING
    # --------------------------------------------------------------------------

    # Randomly shift the timestep window during training. Recommended when data
    # comes from various datasets/regions/seasons. If you have a very controlled
    # local growing season it could be worth trying without augmentation.
    --augment

    # Simulate missing data by randomly masking sensors during training.
    # Improves robustness when sensor data is partially unavailable at inference.
    --enable_masking

    # Permanently disable individual sensors (sets their dropout probability to 100%).
    # Use when a sensor is unavailable in your region / time period, or to test
    # a no-SAR model for example.
    # --disable_s1      # disable Sentinel-1 (SAR)
    # --disable_s2      # disable Sentinel-2 (optical)
    # --disable_meteo   # disable AGERA5 meteorological data

    # --------------------------------------------------------------------------
    # CLASS & SPATIAL BALANCING
    # --------------------------------------------------------------------------

    # Weight the sampler so rare classes are not under-represented per mini-batch.
    --use_class_balancing

    # Balancing strategy for classes within each task head.
    # Choices: balanced (default) | log | effective | none
    # --class_balancing_method "balanced"

    # Balancing strategy across the two task heads (landcover vs crop-type).
    # --task_balancing_method "balanced"

    # Clip extreme sampler weights to prevent training instability.
    # --balancing_clip_min 0.1
    # --balancing_clip_max 10.0

    # Down-weight spatially over-represented areas to improve geographic
    # generalization. Provide either a pre-computed group column or a grid size.
    # --spatial_group_column "tile_id"    # use a pre-existing group column
    # --spatial_bin_size_deg 5.0          # or auto-bin by this grid size (degrees)
    # --spatial_balancing_method "log"

    # --------------------------------------------------------------------------
    # ENCODER FREEZING & LEARNING RATE SCHEDULE
    # --------------------------------------------------------------------------

    # Freeze the Presto encoder for N epochs and only train the new head first.
    # This protects the pre-trained features while the head warms up.
    # Set to 0 to train end-to-end from the very first epoch.
    --head_only_training 3

    # Learning rate while the encoder is frozen (head-only phase).
    # --head_learning_rate 1e-2

    # Target learning rate once the encoder is unfrozen.
    # --full_learning_rate 1e-3

    # After unfreezing the encoder, linearly ramp the LR over this many epochs
    # before reaching --full_learning_rate (avoids a sudden gradient spike).
    --post_unfreeze_warmup_epochs 2

    # Starting LR for the ramp, expressed as a fraction of --full_learning_rate.
    # --post_unfreeze_warmup_start_factor 0.1

    # Multiplicative LR decay applied every epoch after the warmup.
    # --lr_gamma 0.99

    # --------------------------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------------------------

    # Can be increased on larger machines / GPU.
    --batch_size 2048

    # Stop training after this many epochs without validation improvement.
    # On smaller datasets the model may need longer to escape a local minimum,
    # so consider raising this value.
    --patience 20

    # DataLoader worker processes. Set to 0 for single-process loading (easier
    # to debug), or match the number of available CPU cores for speed.
    --num_workers 4

    # Smooth the validation loss with an EMA before early stopping.
    # 0.0 disables smoothing; higher values give more stable (but slower) stopping.
    # --val_loss_ema_alpha 0.3

    # --------------------------------------------------------------------------
    # HEAD ARCHITECTURE
    # --------------------------------------------------------------------------

    # Projection head type used for both the landcover and crop-type heads.
    #   "linear" (default) — a single fully-connected layer. Fast, robust, and
    #             recommended when training data is limited.
    #   "mlp"    — a two-layer MLP (Linear → ReLU → Dropout → Linear). Can
    #             improve accuracy with sufficient data.
    # --head_type "linear"

    # Hidden layer width for MLP heads.
    # Only used when --head_type is "mlp"; ignored for linear heads.
    # --head_hidden_dim 256

    # Dropout rate inside the MLP head.
    # Only used when --head_type is "mlp"; ignored for linear heads.
    # --seasonal_head_dropout 0.0

    # Relative loss weight for each task head.
    # --seasonal_loss_landcover_weight 1.0
    # --seasonal_loss_croptype_weight  1.0

    # --------------------------------------------------------------------------
    # LOGGING & DIAGNOSTICS
    # --------------------------------------------------------------------------

    # Write TensorBoard event files into the experiment folder.
    # View with: tensorboard --logdir <experiment_dir>/tensorboard
    --log_tensorboard

    # Run on a small data subset to verify the full pipeline quickly.
    # --debug

)

python "$FINETUNE_SCRIPT" "${ARGS[@]}"
