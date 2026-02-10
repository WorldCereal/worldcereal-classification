import json

from worldcereal.train.backbone import resolve_seasonal_encoder
from worldcereal.train.data import (
    compute_embeddings_from_splits,
    train_val_test_split,
)
from worldcereal.train.downstream import TorchTrainer


def test_seasonal_head_training_pipeline(WorldCerealExtractionsDF, tmp_path):

    train_df, val_df, test_df = train_val_test_split(
        WorldCerealExtractionsDF,
        val_size=0.2,
        test_size=0.2,
        seed=42,
        min_samples_per_class=10,
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}

    presto_checkpoint, _ = resolve_seasonal_encoder()
    embeddings_df = compute_embeddings_from_splits(
        splits["train"],
        splits["val"],
        splits["test"],
        presto_checkpoint,
        season_id="custom-s1",
        season_windows={"custom-s1": ["2025-03-01", "2025-08-31"]},
    )

    # Direct shape assert: if process_extractions_df changes, this may have to be updated
    aux_columns = [
        "ewoc_code",
        "lat",
        "quality_score_lc",
        "available_timesteps",
        "tile",
        "valid_position",
        "filename",
        "quality_score_ct",
        "year",
        "geometry",
        "extract",
        "h3_l3_cell",
        "geom_text",
        "lon",
        "ref_id",
        "start_date",
        "end_date",
        "irrigation_status",
        "valid_time",
        "label_full",
        "sampling_label",
        "sample_id",
    ]
    static_columns = ["DEM-alt-20m", "DEM-slo-20m"]
    feature_columns = [
        "OPTICAL-B02",
        "OPTICAL-B03",
        "OPTICAL-B04",
        "OPTICAL-B08",
        "OPTICAL-B05",
        "OPTICAL-B06",
        "OPTICAL-B07",
        "OPTICAL-B8A",
        "OPTICAL-B11",
        "OPTICAL-B12",
        "SAR-VV",
        "SAR-VH",
        "METEO-temperature_mean",
        "METEO-precipitation_flux",
    ]
    num_timesteps_expected = 12
    total_cols_expected = (
        len(aux_columns)
        + len(static_columns)
        + len(feature_columns) * num_timesteps_expected
    )
    # Make sure we have the expected columns and non-empty sample size
    assert train_df.shape[1] == total_cols_expected
    assert train_df.shape[0] > 0

    # We keep original ewoc_code for this test
    train_df["downstream_class"] = train_df["ewoc_code"]

    assert not embeddings_df.empty, "Expected seasonal embeddings to be generated"
    required_columns = {"split", "finetune_class", "downstream_class", "in_season"}
    missing = required_columns.difference(embeddings_df.columns)
    assert not missing, f"Missing columns in seasonal embeddings: {missing}"
    for split_name in ("train", "val", "test"):
        assert (embeddings_df["split"] == split_name).any(), (
            f"Seasonal embeddings missing '{split_name}' rows"
        )

    trainer = TorchTrainer(
        embeddings_df,
        head_task="croptype",
        head_type="linear",
        output_dir=tmp_path,
        lr=1e-2,
        weight_decay=1e-4,
        batch_size=64,
        num_workers=0,
        epochs=1,
        use_balancing=False,
        early_stopping_patience=1,
        season_id="tc-s1",
        presto_model_path=presto_checkpoint,
    )
    trainer.train()

    config_path = tmp_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    assert config["experiment"]["season_id"] == "tc-s1"
    head_config = config["heads"][0]
    replacement = head_config["replacement_contract"]
    assert replacement["input_tensor"] == "season_embeddings"
    assert replacement["expects_season_masks"] is True

    packaged = list(tmp_path.glob("PrestoDownstreamTorchHead_*tc-s1*.zip"))
    assert packaged, "Packaged seasonal head zip missing"
