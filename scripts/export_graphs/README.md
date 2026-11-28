# Exported process-graph workflows

This folder contains advanced, user-managed graph-export workflows that are intentionally separate from the regular job-submission CLI.

The goal is to build the OpenEO process graph, save it to disk, and leave the final backend submission to the user. This is meant for advanced users who want to inspect, edit, or submit a graph elsewhere.

## Cropland mapping export

The current export workflow covers the cropland mapping pipeline only.

Usage:

```bash
python scripts/export_graphs/export_cropland_mapping.py \
  --bbox 664000,5611134,684000,5631134 \
  --bbox_epsg 32631 \
  --start_date 2021-01-01 \
  --end_date 2021-12-31 \
  --seasonal_model_zip /path/to/seasonal-model.zip \
  --landcover_head_zip /path/to/landcover-head.zip \
  --output_dir ./outputs/exported_graphs
```

The script writes a JSON file containing:

- a metadata section
- the resolved workflow config
- the serialized OpenEO process graph

The graph is not submitted to the backend; it is intended to be edited or submitted manually afterward.

This is intentionally a separate mode from the normal production workflow and is aimed at advanced users who need more control over the generated graph.
