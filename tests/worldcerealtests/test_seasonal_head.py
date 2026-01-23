import torch

from worldcereal.train.seasonal_head import SeasonalFinetuningHead


def test_seasonal_head_outputs_shapes():
    batch_size, timesteps, seasons, emb_dim = 3, 5, 2, 128
    embeddings = torch.randn(batch_size, timesteps, emb_dim)
    season_masks = torch.ones(batch_size, seasons, timesteps, dtype=torch.bool)

    head = SeasonalFinetuningHead(
        embedding_dim=emb_dim,
        landcover_num_outputs=4,
        crop_num_outputs=6,
        dropout=0.0,
    )
    output = head(embeddings, season_masks)

    assert output.global_logits is not None
    assert output.season_logits is not None
    assert output.global_logits.shape == (batch_size, 4)
    assert output.season_logits.shape == (batch_size, seasons, 6)
    assert output.global_embedding.shape == (batch_size, emb_dim)
    assert output.season_embeddings.shape == (batch_size, seasons, emb_dim)


def test_seasonal_head_all_zero_mask_fallback():
    batch_size, timesteps, seasons, emb_dim = 2, 4, 3, 8
    embeddings = torch.randn(batch_size, timesteps, emb_dim)
    season_masks = torch.zeros(batch_size, seasons, timesteps, dtype=torch.bool)

    head = SeasonalFinetuningHead(
        embedding_dim=emb_dim,
        landcover_num_outputs=None,
        crop_num_outputs=5,
    )

    output = head(embeddings, season_masks)

    assert output.global_logits is None
    assert output.season_logits is not None
    assert torch.isfinite(output.season_embeddings).all()
    assert torch.isfinite(output.season_logits).all()
