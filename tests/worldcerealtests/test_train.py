import json

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from worldcereal.train.backbone import build_presto_backbone
from worldcereal.train.data import (
    collate_fn,
    compute_embeddings_from_input_df,
    dataset_to_embeddings,
    get_training_dfs_from_parquet,
)
from worldcereal.train.datasets import WorldCerealTrainingDataset
from worldcereal.train.downstream import TorchTrainer
from worldcereal.utils.refdata import get_class_mappings

LANDCOVER_KEY = "LANDCOVER10"


def test_worldcerealtraindataset(WorldCerealExtractionsDF):
    """Test creation of WorldCerealTrainingDataset and data loading"""

    df = WorldCerealExtractionsDF.reset_index()

    ds = WorldCerealTrainingDataset(df, task_type="binary", augment=True)

    # Check if data loading works
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for predictors, attrs in dl:
        assert predictors.s1.shape == (2, 1, 1, 12, 2)
        assert predictors.s2.shape == (2, 1, 1, 12, 13)
        assert predictors.meteo.shape == (2, 1, 1, 12, 2)
        assert predictors.latlon.shape == (2, 1, 1, 2)
        assert predictors.dem.shape == (2, 1, 1, 2)
        assert predictors.timestamps.shape == (2, 12, 3)
        assert isinstance(attrs, dict)
        assert "season_masks" in attrs
        assert attrs["season_masks"].shape[-1] == 12
        break


def test_dataset_to_embeddings(WorldCerealExtractionsDF):
    """Test the function that computes embeddings and targets into
    a training dataframe using a presto model
    """

    df = WorldCerealExtractionsDF.reset_index()
    ds = WorldCerealTrainingDataset(df)

    presto_model = build_presto_backbone()

    training_df = dataset_to_embeddings(ds, presto_model, batch_size=256)

    for ft in range(128):
        assert f"presto_ft_{ft}" in training_df.columns


def test_get_training_dfs_from_parquet(WorldCerealPrivateExtractionsPath):
    """Test get_training_dfs_from_parquet with real data using random split."""

    # Use the actual prepare_training_df function with real data
    train_df, val_df, test_df = get_training_dfs_from_parquet(
        WorldCerealPrivateExtractionsPath,
        finetune_classes=LANDCOVER_KEY,
        class_mappings=get_class_mappings(),
        debug=True,  # Use debug mode to limit processing time
    )

    # Verify that we got actual DataFrames back
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Check that the DataFrame has the expected columns
    expected_columns = [
        "finetune_class",
        "balancing_class",
        "lat",
        "lon",
        "ewoc_code",
    ]
    for column in expected_columns:
        assert column in train_df.columns

    # Check that finetune_class values match expected CROPLAND2 classes
    assert all(
        c in ["temporary_crops", "grasslands"]
        for c in train_df["finetune_class"].unique()
    )

    # Check that train, val, and test splits are reasonable
    total_samples = len(train_df) + len(val_df) + len(test_df)

    # Based on the implementation:
    # - train/test split with val_size=0.2 (80% train+val, 20% test)
    # - train/val split with val_size=0.2 (80% train, 20% val of the train+val set)
    # This means approximately:
    # - train: 64% (80% of 80%)
    # - val: 16% (20% of 80%)
    # - test: 20%

    # Calculate the actual ratios
    train_ratio = len(train_df) / total_samples
    val_ratio = len(val_df) / total_samples
    test_ratio = len(test_df) / total_samples

    # Allow for some deviation due to small sample sizes and rounding
    decimal = 1  # 10% tolerance for small datasets

    # Verify the ratios are within tolerance of expected values
    np.testing.assert_almost_equal(
        train_ratio,
        0.64,
        decimal=decimal,
        err_msg=f"Train ratio {train_ratio:.2f} is not close enough to expected 0.64",
    )
    np.testing.assert_almost_equal(
        val_ratio,
        0.16,
        decimal=decimal,
        err_msg=f"Validation ratio {val_ratio:.2f} is not close enough to expected 0.16",
    )
    np.testing.assert_almost_equal(
        test_ratio,
        0.20,
        decimal=decimal,
        err_msg=f"Test ratio {test_ratio:.2f} is not close enough to expected 0.20",
    )

    # Also check that the sum of all ratios equals 1
    np.testing.assert_almost_equal(
        train_ratio + val_ratio + test_ratio,
        1.0,
        decimal=2,
        err_msg="Sum of split ratios doesn't equal 1.0",
    )

    # Verify relative ratios match the split_df implementation
    # The first split should give train+val:test ≈ 80:20
    train_val_ratio = (len(train_df) + len(val_df)) / total_samples
    np.testing.assert_almost_equal(train_val_ratio, 0.8, decimal=decimal)
    np.testing.assert_almost_equal(test_ratio, 0.2, decimal=decimal)

    # The second split should give train:val ≈ 80:20 of the train+val portion
    if len(train_df) + len(val_df) > 0:  # Avoid division by zero
        train_to_trainval_ratio = len(train_df) / (len(train_df) + len(val_df))
        val_to_trainval_ratio = len(val_df) / (len(train_df) + len(val_df))
        np.testing.assert_almost_equal(train_to_trainval_ratio, 0.8, decimal=decimal)
        np.testing.assert_almost_equal(val_to_trainval_ratio, 0.2, decimal=decimal)


def test_train_downstream_torch(WorldCerealExtractionsDF, tmp_path):
    df = WorldCerealExtractionsDF.reset_index()

    embeddings_df = compute_embeddings_from_input_df(
        input_df=df, stratify_label="downstream_class"
    )

    # Train classifier
    trainer = TorchTrainer(
        embeddings_df,
        lr=1e-2,
        weight_decay=1e-5,
        output_dir=tmp_path,
    )

    trainer.train()
    config_path = tmp_path / "config.json"
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    assert "heads" in config
    assert config["heads"][0]["task"] == "croptype"
