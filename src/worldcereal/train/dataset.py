from typing import List

import pandas as pd
from loguru import logger
from presto.dataset import WorldCerealLabelledDataset
from presto.masking import MaskParamsNoDw


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
