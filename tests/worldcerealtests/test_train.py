from torch.utils.data import DataLoader

from worldcereal.train.dataset import WorldCerealTrainingDataset


def test_worldcerealtraindataset(WorldCerealExtractionsDF):
    """Test creation of WorldCerealTrainingDataset and data loading"""

    df = WorldCerealExtractionsDF.reset_index()

    ds = WorldCerealTrainingDataset(
        df,
        task_type="cropland",
        augment=True,
        mask_ratio=0.25,
        repeats=2,
    )

    # Check if number of samples matches repeats
    assert len(ds) == 2 * len(df)

    # Check if data loading works
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    for x, y, dw, latlons, month, valid_month, variable_mask, attrs in dl:
        assert x.shape == (2, 12, 17)
        assert y.shape == (2,)
        assert dw.shape == (2, 12)
        assert dw.unique().numpy()[0] == 9
        assert latlons.shape == (2, 2)
        assert month.shape == (2,)
        assert valid_month.shape == (2,)
        assert variable_mask.shape == x.shape
        assert isinstance(attrs, dict)
        break
