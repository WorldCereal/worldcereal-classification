"""Test that three different embedding extraction methods produce identical results.

This verifies:
1. Standalone Presto encoder → TIME pooling → manual mean = embeddings
2. SeasonalModel → forward pass → global_embedding attribute = embeddings
3. SeasonalModel.backbone → TIME pooling → manual mean = embeddings

All three should produce identical embeddings (within numerical precision).
"""

import numpy as np
import pytest
import torch
from prometheo.models import Presto
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import Predictors

from worldcereal.openeo.inference import load_model_artifact
from worldcereal.openeo.parameters import DEFAULT_SEASONAL_MODEL_URL
from worldcereal.train.backbone import build_presto_backbone
from worldcereal.train.seasonal_head import (
    SeasonalFinetuningHead,
    WorldCerealSeasonalModel,
)


@pytest.fixture
def device():
    """Use CPU for reproducibility."""
    return torch.device("cpu")


@pytest.fixture
def dummy_predictors(device):
    """Create reproducible dummy input data."""
    batch_size = 5
    num_timesteps = 12
    height = width = 1

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # S2 optical: 13 bands (B1-B12 + B8A)
    s2_data = torch.randn(
        batch_size, height, width, num_timesteps, 13, dtype=torch.float32
    )

    # S1 SAR: 2 bands (VV, VH)
    s1_data = torch.randn(
        batch_size, height, width, num_timesteps, 2, dtype=torch.float32
    )

    # ERA5 meteo: 2 bands (temp, precip)
    meteo_data = torch.randn(
        batch_size, height, width, num_timesteps, 2, dtype=torch.float32
    )

    # DEM: 2 bands (elevation, slope)
    dem_data = torch.randn(batch_size, height, width, 2, dtype=torch.float32)

    # Lat/lon
    latlon_data = torch.randn(batch_size, height, width, 2, dtype=torch.float32)

    # Timesteps: [day, month, year]
    timestamps = torch.zeros((batch_size, num_timesteps, 3), dtype=torch.long)
    timestamps[:, :, 0] = torch.arange(1, num_timesteps + 1)  # days
    timestamps[:, :, 1] = 1  # January
    timestamps[:, :, 2] = 2024  # year

    return Predictors(
        s2=s2_data.to(device),
        s1=s1_data.to(device),
        meteo=meteo_data.to(device),
        dem=dem_data.to(device),
        latlon=latlon_data.to(device),
        timestamps=timestamps.to(device),
    )


@pytest.fixture
def season_masks(dummy_predictors):
    """Create dummy season masks."""
    batch_size = dummy_predictors.s2.shape[0]
    num_timesteps = dummy_predictors.s2.shape[3]
    return torch.ones((batch_size, 1, num_timesteps), dtype=torch.bool)


@pytest.fixture
def standalone_encoder(device):
    """Load standalone Presto encoder."""
    encoder = build_presto_backbone().to(device).eval()
    return encoder


@pytest.fixture
def seasonal_model(device):
    """Load full seasonal model."""
    artifact = load_model_artifact(DEFAULT_SEASONAL_MODEL_URL)

    backbone = Presto()
    head = SeasonalFinetuningHead(
        embedding_dim=backbone.encoder.embedding_size,
        landcover_num_outputs=10,
        crop_num_outputs=None,
        dropout=0,
    )
    model = WorldCerealSeasonalModel(backbone=backbone, head=head)

    state_dict = torch.load(artifact.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def test_standalone_encoder_embeddings(standalone_encoder, dummy_predictors):
    """Test that standalone encoder produces embeddings via TIME pooling + manual mean."""
    with torch.no_grad():
        # Get TIME embeddings
        time_emb = standalone_encoder(
            dummy_predictors, eval_pooling=PoolingMethods.TIME
        )

        # Flatten to [batch, time, embedding_dim]
        if time_emb.dim() == 5:
            b, h, w, t, d = time_emb.shape
            time_emb = time_emb.view(b * h * w, t, d)
        elif time_emb.dim() == 4:
            b, t, d, _ = time_emb.shape
            time_emb = time_emb.view(b, t, d)

        # Manual mean pooling
        embeddings = time_emb.mean(dim=1)

        # Basic sanity checks
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == 5  # batch_size
        assert embeddings.shape[1] > 0  # embedding_dim


def test_seasonal_model_forward_embeddings(
    seasonal_model, dummy_predictors, season_masks
):
    """Test that SeasonalModel.forward() produces embeddings via global_embedding."""
    attrs = {"season_masks": season_masks}

    with torch.no_grad():
        output = seasonal_model(dummy_predictors, attrs)
        embeddings = output.global_embedding

        # Basic sanity checks
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == 5  # batch_size
        assert embeddings.shape[1] > 0  # embedding_dim


def test_seasonal_backbone_embeddings(seasonal_model, dummy_predictors):
    """Test that SeasonalModel.backbone produces embeddings via TIME pooling + manual mean."""
    with torch.no_grad():
        # Get TIME embeddings from the backbone
        time_emb = seasonal_model.backbone(
            dummy_predictors, eval_pooling=PoolingMethods.TIME
        )

        # Flatten
        if time_emb.dim() == 5:
            b, h, w, t, d = time_emb.shape
            time_emb = time_emb.view(b * h * w, t, d)
        elif time_emb.dim() == 4:
            b, t, d, _ = time_emb.shape
            time_emb = time_emb.view(b, t, d)

        # Manual mean pooling
        embeddings = time_emb.mean(dim=1)

        # Basic sanity checks
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == 5  # batch_size
        assert embeddings.shape[1] > 0  # embedding_dim


def test_embedding_equivalence(
    standalone_encoder,
    seasonal_model,
    dummy_predictors,
    season_masks,
):
    """Test that all three embedding extraction methods produce identical results."""
    threshold = 1e-5

    # Method 1: Standalone encoder
    with torch.no_grad():
        time_emb1 = standalone_encoder(
            dummy_predictors, eval_pooling=PoolingMethods.TIME
        )
        if time_emb1.dim() == 5:
            b, h, w, t, d = time_emb1.shape
            time_emb1 = time_emb1.view(b * h * w, t, d)
        elif time_emb1.dim() == 4:
            b, t, d, _ = time_emb1.shape
            time_emb1 = time_emb1.view(b, t, d)
        embeddings_method1 = time_emb1.mean(dim=1)

    # Method 2: SeasonalModel forward
    attrs = {"season_masks": season_masks}
    with torch.no_grad():
        output = seasonal_model(dummy_predictors, attrs)
        embeddings_method2 = output.global_embedding

    # Method 3: SeasonalModel.backbone
    with torch.no_grad():
        time_emb3 = seasonal_model.backbone(
            dummy_predictors, eval_pooling=PoolingMethods.TIME
        )
        if time_emb3.dim() == 5:
            b, h, w, t, d = time_emb3.shape
            time_emb3 = time_emb3.view(b * h * w, t, d)
        elif time_emb3.dim() == 4:
            b, t, d, _ = time_emb3.shape
            time_emb3 = time_emb3.view(b, t, d)
        embeddings_method3 = time_emb3.mean(dim=1)

    # Compare Method 1 vs Method 2
    diff_1_2 = (embeddings_method1 - embeddings_method2).abs()
    max_diff_1_2 = diff_1_2.max().item()

    # Compare Method 1 vs Method 3
    diff_1_3 = (embeddings_method1 - embeddings_method3).abs()
    max_diff_1_3 = diff_1_3.max().item()

    # Compare Method 2 vs Method 3
    diff_2_3 = (embeddings_method2 - embeddings_method3).abs()
    max_diff_2_3 = diff_2_3.max().item()

    # Assert all methods produce identical embeddings
    max_diff = max(max_diff_1_2, max_diff_1_3, max_diff_2_3)

    assert max_diff_1_2 < threshold, (
        f"Standalone vs Seasonal.forward() embeddings differ by {max_diff_1_2:.6e} (threshold: {threshold:.0e})"
    )
    assert max_diff_1_3 < threshold, (
        f"Standalone vs Seasonal.backbone embeddings differ by {max_diff_1_3:.6e} (threshold: {threshold:.0e})"
    )
    assert max_diff_2_3 < threshold, (
        f"Seasonal.forward() vs Seasonal.backbone embeddings differ by {max_diff_2_3:.6e} (threshold: {threshold:.0e})"
    )

    # Success message when test passes
    print(
        f"\n✓ All three methods produce identical embeddings (max difference: {max_diff:.2e} < {threshold:.0e})"
    )
