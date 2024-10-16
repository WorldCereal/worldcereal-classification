from dataclasses import dataclass
from random import choice, randint, random, sample
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from presto.dataops import BANDS_GROUPS_IDX, NUM_TIMESTEPS, SRTM_INDEX, TIMESTEPS_IDX
from presto.dataset import WorldCerealLabelledDataset
from presto.masking import BAND_EXPANSION, MASK_STRATEGIES
from presto.presto import Presto
from presto.utils import device
from torch.utils.data import DataLoader
from tqdm import tqdm


def make_mask_no_dw(
    strategy: str,
    mask_ratio: float,
    existing_mask: np.ndarray,
    num_timesteps: int = NUM_TIMESTEPS,
) -> np.ndarray:
    """
    Make a mask for a given strategy and percentage of masked values.
    Args:
        strategy: The masking strategy to use. One of MASK_STRATEGIES
        mask_ratio: The max percentage of values to mask. Between 0 and 1.
    """
    # we assume that topography is never "naturally" masked
    mask = existing_mask.copy()

    srtm_mask = False

    if mask_ratio > 0.05:
        actual_mask_ratio = np.random.uniform(0.05, mask_ratio)
    else:
        actual_mask_ratio = mask_ratio

    num_tokens_to_mask = int(
        ((num_timesteps * (len(BANDS_GROUPS_IDX) - 1)) + 1) * actual_mask_ratio
    )
    assert num_tokens_to_mask > 0, f"num_tokens_to_mask: {num_tokens_to_mask}"

    def mask_topography(srtm_mask, num_tokens_to_mask, actual_mask_ratio):
        should_flip = random() < actual_mask_ratio
        if should_flip:
            srtm_mask = True
            num_tokens_to_mask -= 1
        return srtm_mask, num_tokens_to_mask

    def random_masking(mask, num_tokens_to_mask: int):
        if num_tokens_to_mask > 0:
            # we set SRTM to be True - this way, it won't get randomly assigned.
            # at the end of the function, it gets properly assigned
            mask[:, SRTM_INDEX] = True
            # then, we flatten the mask and dw arrays
            all_tokens_mask = mask.flatten()
            unmasked_tokens = all_tokens_mask == 0
            # unmasked_tokens = all_tokens_mask == False
            idx = np.flatnonzero(unmasked_tokens)
            np.random.shuffle(idx)
            idx = idx[:num_tokens_to_mask]
            all_tokens_mask[idx] = True
            mask = all_tokens_mask.reshape((num_timesteps, len(BANDS_GROUPS_IDX)))
        return mask

    # RANDOM BANDS
    if strategy == "random_combinations":
        srtm_mask, num_tokens_to_mask = mask_topography(
            srtm_mask, num_tokens_to_mask, actual_mask_ratio
        )
        mask = random_masking(mask, num_tokens_to_mask)

    elif strategy == "group_bands":
        srtm_mask, num_tokens_to_mask = mask_topography(
            srtm_mask, num_tokens_to_mask, actual_mask_ratio
        )
        # next, we figure out how many tokens we can mask
        num_band_groups_to_mask = int(num_tokens_to_mask / num_timesteps)
        assert (num_tokens_to_mask - num_timesteps * num_band_groups_to_mask) >= 0
        num_tokens_masked = 0
        # tuple because of mypy, which thinks lists can only hold one type
        band_groups: List[Any] = list(range(len(BANDS_GROUPS_IDX)))
        band_groups.remove(SRTM_INDEX)
        band_groups_to_mask = sample(band_groups, num_band_groups_to_mask)
        for band_group in band_groups_to_mask:
            num_tokens_masked += int(
                len(mask[:, band_group]) - sum(mask[:, band_group])
            )
            mask[:, band_group] = True
        num_tokens_to_mask -= num_tokens_masked
        mask = random_masking(mask, num_tokens_to_mask)

    # RANDOM TIMESTEPS
    elif strategy == "random_timesteps":
        srtm_mask, num_tokens_to_mask = mask_topography(
            srtm_mask, num_tokens_to_mask, actual_mask_ratio
        )
        # -1 for SRTM
        timesteps_to_mask = int(num_tokens_to_mask / (len(BANDS_GROUPS_IDX) - 1))
        max_tokens_masked = (len(BANDS_GROUPS_IDX) - 1) * timesteps_to_mask
        timesteps = sample(TIMESTEPS_IDX, k=timesteps_to_mask)
        if timesteps_to_mask > 0:
            num_tokens_to_mask -= int(max_tokens_masked - sum(sum(mask[timesteps])))
            mask[timesteps] = True
        mask = random_masking(mask, num_tokens_to_mask)
    elif strategy == "chunk_timesteps":
        srtm_mask, num_tokens_to_mask = mask_topography(
            srtm_mask, num_tokens_to_mask, actual_mask_ratio
        )
        # -1 for SRTM
        timesteps_to_mask = int(num_tokens_to_mask / (len(BANDS_GROUPS_IDX) - 1))
        if timesteps_to_mask > 0:
            max_tokens_masked = (len(BANDS_GROUPS_IDX) - 1) * timesteps_to_mask
            start_idx = randint(0, num_timesteps - timesteps_to_mask)
            num_tokens_to_mask -= int(
                max_tokens_masked
                - sum(sum(mask[start_idx : start_idx + timesteps_to_mask]))
            )
            mask[start_idx : start_idx + timesteps_to_mask] = True  # noqa
        mask = random_masking(mask, num_tokens_to_mask)
    else:
        raise ValueError(f"Unknown strategy {strategy} not in {MASK_STRATEGIES}")

    mask[:, SRTM_INDEX] = srtm_mask
    return np.repeat(mask, BAND_EXPANSION, axis=1)


@dataclass
class MaskParamsNoDw:
    strategies: Tuple[str, ...] = ("NDVI",)
    ratio: float = 0.5
    num_timesteps: int = NUM_TIMESTEPS

    def __post_init__(self):
        for strategy in self.strategies:
            assert strategy in [
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ]

    def mask_data(
        self, eo_data: np.ndarray, mask: np.ndarray, num_timesteps: int = NUM_TIMESTEPS
    ):
        strategy = choice(self.strategies)

        mask = make_mask_no_dw(
            strategy=strategy,
            mask_ratio=self.ratio,
            existing_mask=mask,
            num_timesteps=num_timesteps,
        )

        x = eo_data * ~mask
        y = np.zeros(eo_data.shape).astype(np.float32)
        y[mask] = eo_data[mask]

        return mask, x, y, strategy


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
                # "random_combinations",
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
            "downstream_class",
        ]

        attrs = [attr for attr in attrs if attr in row.index]

        return sample + (row[attrs].to_dict(),)


def get_training_df(
    dataset: WorldCerealTrainingDataset,
    presto_model: Presto,
    batch_size: int = 2048,
    valid_date_as_token: bool = False,
    num_workers: int = 0,
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
    num_workers : int, optional
        number of workers to use in DataLoader, by default 0

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
        num_workers=num_workers,
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
