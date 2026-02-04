import json
from typing import List, Tuple

import pandas as pd

from worldcereal.train.backbone import resolve_seasonal_encoder
from worldcereal.train.data import (
    compute_seasonal_embeddings_from_splits,
    train_val_test_split,
)
from worldcereal.train.downstream import TorchTrainer


def _prepare_balanced_subset(df: pd.DataFrame, max_samples: int = 80) -> pd.DataFrame:
    subset = df.reset_index(drop=True).copy()
    subset.loc[15:, ["croptype_name", "downstream_class"]] = (
        "other"  # Override for the test
    )
    if "finetune_class" not in subset.columns:
        if "downstream_class" not in subset.columns:
            raise AssertionError(
                "Fixture must provide 'finetune_class' or 'downstream_class' columns."
            )
        subset["finetune_class"] = subset["downstream_class"]
    counts = subset["finetune_class"].value_counts()
    viable = counts[counts >= 3].index.tolist()
    filtered = subset[subset["finetune_class"].isin(viable)].copy()
    assert filtered["finetune_class"].nunique() >= 2, (
        "Seasonal head training needs at least two classes"
    )
    if len(filtered) > max_samples:
        filtered = filtered.sample(n=max_samples, random_state=7).reset_index(drop=True)
    return filtered


def _prepare_splits(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = train_val_test_split(df, stratify_label="finetune_class")
    prepared: List[pd.DataFrame] = []
    for part in splits:
        copy = part.reset_index(drop=True).copy()
        if "downstream_class" not in copy.columns:
            copy["downstream_class"] = copy["finetune_class"]
        prepared.append(copy)
    return tuple(prepared)


def test_seasonal_head_training_pipeline(WorldCerealExtractionsDF, tmp_path):
    base_df = _prepare_balanced_subset(WorldCerealExtractionsDF)
    train_df, val_df, test_df = _prepare_splits(base_df)

    presto_checkpoint, _ = resolve_seasonal_encoder()
    embeddings_df = compute_seasonal_embeddings_from_splits(
        train_df,
        val_df,
        test_df,
        presto_checkpoint,
        season_id="tc-s1",
        augment=False,
        mask_on_training=False,
        repeats=1,
        batch_size=32,
        num_workers=0,
    )

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
        lr_grid=[1e-2],
        weight_decay_grid=[1e-4],
        batch_size=64,
        num_workers=0,
        epochs=1,
        cv_folds=2,
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
