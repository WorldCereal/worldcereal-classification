# CropGet · Find

A local Streamlit app with a CARMA-style landing page, precomputed training-point lookup from an uploaded WorldCereal Parquet, and location-based matching against the simplified CropRes grid lookup.

## Run locally

Open Anaconda Prompt in this folder:

```bat
conda activate cpus
python -m pip install -r requirements.txt
python -m streamlit run app_local.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`.

## 1. Retrieve other relevant data for your reference data

Upload a Parquet containing `ref_id` and `sample_id`, select one or more crops and a season (`s1`/`s2`), then choose any K from 1 to 25. The app collapses repeated time-series rows to unique `(ref_id, sample_id)` pairs, then joins those IDs to the precomputed CropGet training-point lookup. It can show neighbors in retrieval-rank or descending spectral-similarity order. It does not encode the upload, load a memory bank, or run retrieval.

Set the lookup folder with `CROPGET_TRAIN_POINTS_LOOKUP_ROOT` or edit the lookup-folder field in the app. For example:

```text
<your-data-folder>/cropget_lookup
```

Expected layout:

```text
cropget_lookup/
  top25/<crop>/season=s1.parquet
  top25/<crop>/season=s2.parquet
```

The app returns up to K rows per query and crop, including query IDs, retrieved sample ID and coordinates, rank, AEV retrieval similarity, and spectral similarity. The spectral observations and embeddings are not included. The map uses crop-specific colors, circles for retrieved points, and diamonds for uploaded query points; query points are rendered on top. If your lookup field points directly to `top25`, that folder is used directly.

Supported lookup crops: barley, maize, millet, rapeseed, rice, rye, sorghum, soybean, sunflower, and wheat.

## 2. Retrieve relevant reference data, anywhere, anytime (Experimental)

This landing-page option is a placeholder for the experimental retrieval mode.

## Location lookup (Experimental)

Mode 3 displays an interactive map. Draw a rectangle, set random query-point counts for maize, sunflower, or soybean, and find each point's nearest stored CropRes grid point and top-1 matched sample ID. It uses:

```text
<asset folder>/lightweight_top1_lookup/<crop>/top1_lookup.parquet
```

Set `CROPGET_MODE1_ASSET_ROOT` to your asset folder. It should contain the location lookup Parquets under `lightweight_top1_lookup/<crop>/top1_lookup.parquet`. The lookup query uses season and cohort year. The app reports repeated matched sample IDs and offers a CSV download.

## Dependencies

Install with `python -m pip install -r requirements.txt`.
