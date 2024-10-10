from typing import List

import pandas as pd
import torch
from loguru import logger
from presto.dataset import WorldCerealLabelledDataset
from presto.masking import MaskParamsNoDw
from presto.presto import Presto
from presto.utils import device
from torch.utils.data import DataLoader
from tqdm import tqdm


class WorldCerealTrainingDataset(WorldCerealLabelledDataset):

    FILTER_LABELS = [0]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        task_type: str = "cropland",
        croptype_list: List = [],
        return_hierarchical_labels: bool = False,
        augment: bool = False,
        mask_ratio: float = 0.0,
        repeats: int = 1,
    ):
        dataframe = dataframe.loc[~dataframe.LANDCOVER_LABEL.isin(self.FILTER_LABELS)]

        self.task_type = task_type
        self.croptype_list = croptype_list
        self.return_hierarchical_labels = return_hierarchical_labels
        self.augment = augment
        if augment:
            logger.info(
                "Augmentation is enabled. \
    The horizontal jittering of the selected window will be performed."
            )
        self.mask_ratio = mask_ratio
        self.mask_params = MaskParamsNoDw(
            (
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ),
            mask_ratio,
        )

        super(WorldCerealLabelledDataset, self).__init__(dataframe)
        logger.info(f"Original dataset size: {len(dataframe)}")

        self.indices = []
        for _ in range(repeats):
            self.indices += [i for i in range(len(self.df))]

        logger.info(f"Dataset size after {repeats} repeats: {len(self.indices)}")

    def __iter__(self):
        for idx in self.indices:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):

        # Get the sample
        sample = super().__getitem__(idx)
        row = self.df.iloc[self.indices[idx], :]

        attrs = [
            "WORLDCOVER-LABEL-10m",
            "lat",
            "lon",
            "CROPTYPE_LABEL",
            "LANDCOVER_LABEL",
            "POTAPOV-LABEL-10m",
            "location_id",
            "ref_id",
            "sample_id",
        ]

        return sample + (row[attrs].to_dict(),)


def get_training_df(
    dataset: WorldCerealTrainingDataset,
    presto_model: Presto,
    batch_size: int = 2048,
    valid_date_as_token: bool = False,
) -> pd.DataFrame:
    """Function to extract Presto embeddings, targets and relevant
    auxiliary attributes from a dataset.

    Parameters
    ----------
    dataset : WorldCerealTrainingDataset
        dataset to extract embeddings from
    presto_model : Presto
        presto model to use for extracting embeddings
    batch_size : int, optional
        by default 2048
    valid_date_as_token : bool, optional
        use valid_date as token in Presto, by default False

    Returns
    -------
    pd.DataFrame
        training dataframe that can be used for training downstream classifier
    """

    # Create dataloader
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Initialize final dataframe
    final_df = None

    # Iterate through dataloader to consume all samples
    for x, _, dw, latlons, month, valid_month, variable_mask, attrs in tqdm(dl):
        x_f, dw_f, latlons_f, month_f, valid_month_f, variable_mask_f = [
            t.to(device) for t in (x, dw, latlons, month, valid_month, variable_mask)
        ]

        # Compute Presto embeddings; only feed valid date as token if valid_date_as_token is True
        with torch.no_grad():
            encodings = (
                presto_model.encoder(
                    x_f,
                    dynamic_world=dw_f.long(),
                    mask=variable_mask_f,
                    latlons=latlons_f,
                    month=month_f,
                    valid_month=valid_month_f if valid_date_as_token else None,
                )
                .cpu()
                .numpy()
            )

        # Convert to dataframe
        attrs = pd.DataFrame.from_dict(attrs)
        encodings = pd.DataFrame(
            encodings, columns=[f"presto_ft_{i}" for i in range(encodings.shape[1])]
        )
        result = pd.concat([encodings, attrs], axis=1)

        # Append to final dataframe
        final_df = result if final_df is None else pd.concat([final_df, result])

    return final_df
