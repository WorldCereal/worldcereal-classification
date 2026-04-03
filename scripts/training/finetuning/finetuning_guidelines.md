
# WorldCereal Finetuning & Inference Guide

## Overview

This guide describes the full workflow for finetuning a WorldCereal Presto model and running
inference with the resulting model artefact. Two training strategies are available:

| Option | Description | Data source | When to use |
|--------|-------------|-------------|-------------|
| **[option 1]** Train downstream classifier | Train only the classification head on top of the frozen Presto encoder | Public extraction bucket + optional user-provided Parquet folder | Small datasets, quick iteration, default user workflow |
| **[option 2]** Full end-to-end finetuning | Finetune the entire model (encoder + heads) jointly | User-specified Parquet files (bypasses public bucket; supports fully custom extractions) | Larger, region-specific datasets; need to adapt feature representations |

---

## [Option 1] Train Downstream Classifier

Use the official classification app notebook:

```
notebooks/worldcereal_classification_app.ipynb
```

This notebook guides you through spatially querying the WorldCereal public extraction bucket to
retrieve pre-computed time-series samples for your area of interest, training a Torch
classification head on frozen Presto embeddings, and evaluating the result. No GPU or HPC
required.

**Data sources for Option 1:**
- **Public extractions**: the app automatically pulls from the WorldCereal public extraction
  bucket. No additional setup needed.
- **Private extractions (optional)**: if you have additional custom Parquet files (e.g. from your
  own field campaigns), point the app to a local folder containing those files. They will be
  merged with the public extractions during training.

---

## [Option 2] Full End-to-End Finetuning

### Step 1 — Get Extractions + Labels

Use the dedicated query notebook to assemble a Parquet file of labelled extraction samples:

```
notebooks/worldcereal_query_extractions.ipynb
```

This notebook provides an interactive tool that lets you:

1. **Draw a bounding box** on a map to spatially constrain your query.
2. **Choose your data sources:**
   - *Public S3 bucket* (enabled by default) — the official WorldCereal public extraction bucket.
   - *Local path(s)* (optional) — one or more local Parquet files or directories, e.g. restricted
     institutional datasets or custom extraction outputs from your own pipeline. Use the
     **"+ Add path"** button to add as many paths as needed.
     Example: `/data/worldcereal_data/EXTRACTIONS/WORLDCEREAL/WORLDCEREAL_ALL_EXTRACTIONS/worldcereal_all_extractions.parquet`
3. **Disable** the *"Only temporary crop samples"* filter (default: off) so that all land-cover
   categories are included — required for end-to-end finetuning where the model must distinguish
   cropland from non-cropland.
4. Click **Run Query** — a summary table shows the number of samples and crop types per dataset.
   Samples that appear in both the public bucket and a local path are deduplicated automatically.
5. Set the output path and click **Save to Parquet**.

Pass the resulting file to the finetuning script via `--parquet_files`:

```bash
--parquet_files "/path/to/worldcereal_query_result.parquet"
```

When `--parquet_files` is omitted the script falls back to the default global extraction list used
by WorldCereal global training (requires VPN / cluster access).

---

### Step 2 — Run Finetuning

The finetuning entry point is:

```
scripts/training/finetuning/finetune_presto.py
```

A fully documented example invocation is provided in:

```
scripts/training/finetuning/finetune_presto.sh
```

For HPC / SLURM clusters, a ready-to-submit job script is available at:

```
scripts/training/finetuning/submit_combined_training.sh
```

#### Minimal local example

```bash
python scripts/training/finetuning/finetune_presto.py \
    --experiment_tag "my-region-finetuning" \
    --base_output_dir "." \
    --parquet_files "/path/to/my_extractions.parquet" \
    --timestep_freq "month" \
    --season_windows '{"s1": ["2021-04-01", "2021-09-30"]}' \
    --class_mappings_file "src/worldcereal/data/croptype_mappings/class_mappings.json" \
    --landcover_classes_key "LANDCOVER10" \
    --croptype_classes_key "CROPTYPE28" \
    --augment \
    --enable_masking \
    --use_class_balancing \
    --head_only_training 3 \
    --log_tensorboard
```

#### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment_tag` | `""` | Short label appended to the auto-generated output folder name |
| `--base_output_dir` | `None` | Root directory for the timestamped experiment folder |
| `--parquet_files` | global list | One or more Parquet files with extracted time-series samples |
| `--wide_parquet_path` | HPC default | Path to cache the expensive pivot/merge step; reused on subsequent runs if the file exists |
| `--timestep_freq` | `month` | Temporal resolution — `month` (12 steps/yr) or `dekad` (36 steps/yr) |
| `--max_timesteps_trim` | `auto` | Hard cap on timestep window width; also enables temporal augmentation when `--augment` is set |
| `--class_mappings_file` | SharePoint | Path to a custom `class_mappings.json`; omit to fetch from SharePoint (requires credentials) |
| `--landcover_classes_key` | `LANDCOVER10` | Key inside the mappings JSON for the land-cover head |
| `--croptype_classes_key` | `CROPTYPE24` | Key inside the mappings JSON for the crop-type head (e.g. `CROPTYPE28`, `CROPTYPE9`) |
| `--initial_mapping` | `LANDCOVER10` | Mapping key used when assigning `ewoc_code → label` during data prep |
| `--season_windows` | — | JSON mapping season names to `[start_date, end_date]`; **mutually exclusive** with `--season_ids` |
| `--season_ids` | `tc-s1 tc-s2` | Official WorldCereal crop-calendar season IDs (e.g. `tc-s1 tc-s2`, `tc-annual`); **mutually exclusive** with `--season_windows` |
| `--finetune_regions` | all | Comma-separated UN M49 macro-region names to restrict training data |
| `--augment` | off | Randomly shift the timestep window during training |
| `--enable_masking` | off | Randomly mask sensors during training to improve robustness |
| `--disable_s1/s2/meteo` | off | Permanently disable individual sensor groups |
| `--use_class_balancing` | off | Up/down-weight sampler to equalise class frequencies |
| `--min_samples_per_class` | `100` | Drop classes with fewer training samples than this threshold |
| `--outlier_mode` | `keep` | `keep` / `drop_candidate` / `drop_suspect` / `drop_flagged` |
| `--head_only_training` | `0` | Freeze encoder for N epochs to warm up the head first |
| `--head_learning_rate` | `1e-2` | LR during the head-only frozen phase |
| `--full_learning_rate` | `1e-3` | Target LR after encoder is unfrozen |
| `--post_unfreeze_warmup_epochs` | `5` | Epochs to linearly ramp up to `--full_learning_rate` |
| `--head_type` | `linear` | `linear` (single FC, recommended for small datasets) or `mlp` (two-layer) |
| `--batch_size` | `4096` | Mini-batch size; reduce if running out of memory |
| `--patience` | `20` | Early-stopping patience in epochs |
| `--num_workers` | `8` | DataLoader worker processes (set to `0` for debugging) |
| `--val_samples_file` / `--test_samples_file` / `--ignore_samples_file` | random split | CSV files pinning sample IDs to specific splits (must contain a `sample_id` column) |
| `--log_tensorboard` | off | Write TensorBoard event files to `<experiment_dir>/tensorboard/` |
| `--debug` | off | Run on a tiny subset to verify the pipeline end-to-end |

#### Class mappings file

The `class_mappings.json` file controls how raw EWoC codes (10-digit hierarchical crop/landcover
codes stored in the `ewoc_code` column of the extractions) are translated into named class labels
for each model head.

**Structure:**

```json
{
    "MAPPING_KEY": {
        "<ewoc_code>": "<class_label>",
        ...
    },
    ...
}
```

Each top-level key defines one mapping set that can be referenced by `--landcover_classes_key`,
`--croptype_classes_key`, or `--initial_mapping`. Multiple keys can coexist in the same file.

**Example — minimal custom mapping:**

```json
{
    "MY_LANDCOVER": {
        "1100000000": "temporary_crops",
        "1200000000": "permanent_crops",
        "2000000000": "non_cropland",
        "3000000000": "non_cropland"
    },
    "MY_CROPTYPE": {
        "1101010010": "wheat",
        "1101010020": "barley",
        "1101020010": "maize",
        "1100000000": "other_crop"
    }
}
```

> **Note:** EWoC codes are matched exactly as strings. Any `ewoc_code` value in the data that is
> not present in the mapping is treated as unlabelled and excluded from training for that head.
> The special label `"ignore"` (case-insensitive) can be assigned to codes that should be
> silently dropped.

---

### Step 3 — Monitor Training

**Live progress (TensorBoard)**

When `--log_tensorboard` is set, open a separate terminal and run:

```bash
tensorboard --logdir <experiment_dir>/tensorboard
```

Replace `<experiment_dir>` with the auto-generated path printed at the start of training (pattern:
`<base_output_dir>/presto-prometheo-<tag>-<freq>-augment=...-balance=...-run=<timestamp>/`).

**Intermediate evaluations**

Every time the validation loss improves, a full evaluation is automatically run against the
validation set and the results are written to:

```
<experiment_dir>/intermediate_evals/
├── CM_<experiment_name>_epoch<NNN>_val.png       # confusion matrix at that epoch
├── CM_<experiment_name>_epoch<NNN>_val_norm.png  # normalised confusion matrix
└── results_<experiment_name>_epoch<NNN>_val.csv  # per-class precision / recall / F1
```

This means you can inspect classification quality at any point during a long training run without
waiting for it to finish. The files are overwritten with a new epoch suffix each time a new best
model is found, so you always have a snapshot of the current best checkpoint.

**Log file**

Full training logs are written to:

```
<experiment_dir>/logs/<experiment_name>.log
```

---

### Step 4 — Evaluate Results

After training completes, the experiment directory contains:

```
<experiment_dir>/
├── <experiment_name>.pt              # full model checkpoint (encoder + heads)
├── <experiment_name>_encoder.pt      # encoder-only checkpoint
├── <experiment_name>.zip             # packaged artefact (use for inference)
├── train_df.parquet                  # training split used
├── val_df.parquet                    # validation split used
├── test_df.parquet                   # test split used
├── logs/
│   └── <experiment_name>.log
├── tensorboard/
└── intermediate_evals/
    ├── CM_<experiment_name>_val.png          # confusion matrix (validation)
    ├── CM_<experiment_name>_val_norm.png     # normalised confusion matrix
    ├── CM_<experiment_name>_test.png         # confusion matrix (test)
    ├── CM_<experiment_name>_test_norm.png
    ├── results_<experiment_name>_val.csv     # per-class metrics (validation)
    └── results_<experiment_name>_test.csv    # per-class metrics (test)
```

> **The artefact to use for inference is `<experiment_name>.zip`.**
> It bundles both the full checkpoint and the encoder-only checkpoint together with the
> model manifest and configuration.

---

### Step 5 — (Optional) Local Inference Patch Testing

Before running a full openEO job, you can validate the model on local `.nc` test patches:

1. Collect local patch inputs using `scripts/inference/collect_inputs_multi.py`.
2. Run `scripts/inference/run_test_patches.py` pointing to your `.zip` artefact and a directory
   of `.nc` files.

---

### Step 6 — Inference in openEO

#### 6a — Upload the model artefact

Upload `<experiment_name>.zip` to a publicly accessible HTTP location, such as:

- **Artifactory** (VITO internal): `https://artifactory.vgt.vito.be/artifactory/...`
- A **public S3 bucket** (AWS, OVH, etc.)
- Any other HTTP server that allows unauthenticated `GET` requests

#### 6b — Configure the inference run

The unified inference entry point is:

```
scripts/inference/run_worldcereal_task_openeo.py
```

Documented example shell scripts:

| Use case | Shell script |
|----------|-------------|
| Cropland mapping | `scripts/inference/run_cropland_mapping.sh` |
| Croptype mapping | `scripts/inference/run_croptype_mapping.sh` |

For the classification workflow, use `--task classification`. This runs the full end-to-end
pipeline (satellite input collection → Presto embeddings → classification map) as a single
openEO job.

#### 6c — Inject your finetuned model

Pass the URL or local path to your artefact via one or more of:

| Argument | Description |
|----------|-------------|
| `--seasonal-model-zip <url_or_path>` | Full finetuned seasonal model (encoder + heads) |
| `--landcover-head-zip <url_or_path>` | Override only the land-cover head |
| `--croptype-head-zip <url_or_path>` | Override only the crop-type head |

#### 6d — Spatial & temporal extent

Supply either a bounding box or a grid file:

```bash
# Bounding box (EPSG:32631)
--bbox 664000 5611134 684000 5631134 --bbox_epsg 32631

# Or a grid file
--grid_path /path/to/grid.gpkg
```

For the temporal extent, either provide a **year** (crop-calendar-aware automatic windowing):

```bash
--year 2022
```

or explicit dates with season specifications:

```bash
--start_date 2021-10-01 --end_date 2022-09-30 \
--season-specifications-json '{"s1": ["2022-04-01", "2022-09-30"]}'
```

#### 6e — Minimal croptype mapping example

```bash
python scripts/inference/run_worldcereal_task_openeo.py \
    --task classification \
    --bbox 664000 5611134 684000 5631134 \
    --bbox_epsg 32631 \
    --year 2022 \
    --product croptype \
    --output_folder ./outputs/maps/croptype \
    --seasonal-model-zip "https://example.com/models/my_finetuned_model.zip" \
    --enable-cropland-head \
    --enable-croptype-head \
    --enforce-cropland-gate \
    --merge-classification-products \
    --class-probabilities \
    --enable-cropland-postprocess \
    --enable-croptype-postprocess \
    --cropland-postprocess-method majority_vote \
    --croptype-postprocess-method majority_vote \
    --parallel_jobs 4 \
    --restart_failed
```

#### 6f — Additional inference options

| Argument | Description |
|----------|-------------|
| `--compositing_window month\|dekad` | Temporal compositing resolution (must match training) |
| `--s1_orbit_state ASCENDING\|DESCENDING` | Force Sentinel-1 orbit direction instead of automatic determination |
| `--grid_size <km>` | Tile size when splitting large AOIs (default: 20 km) |
| `--export-embeddings` | Also export embeddings in the output |
| `--export-ndvi` | Also export NDVI time series |
| `--target-epsg <code>` | Reproject outputs to this CRS |
| `--parallel_jobs <n>` | Max concurrent openEO jobs managed by the job manager (default: 2) |
| `--restart_failed` | Re-submit previously failed tiles |
