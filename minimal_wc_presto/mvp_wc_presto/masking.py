from collections import namedtuple
from dataclasses import dataclass
from random import choice, randint, random, sample
from typing import Any, List, Tuple

import numpy as np

from .dataops import (
    BAND_EXPANSION,
    BANDS_GROUPS_IDX,
    NUM_TIMESTEPS,
    SRTM_INDEX,
    TIMESTEPS_IDX,
)

MASK_STRATEGIES = (
    "group_bands",
    "random_timesteps",
    "chunk_timesteps",
    "random_combinations",
)

MaskedExample = namedtuple(
    "MaskedExample",
    [
        "mask_eo",
        "mask_dw",
        "x_eo",
        "y_eo",
        "x_dw",
        "y_dw",
        "start_month",
        "latlon",
        "strategy",
        "real_mask",
    ],
)


def make_mask_no_dw(strategy: str, mask_ratio: float, existing_mask: np.ndarray) -> np.ndarray:
    """
    Make a mask for a given strategy and percentage of masked values.
    Args:
        strategy: The masking strategy to use. One of MASK_STRATEGIES
        mask_ratio: The percentage of values to mask. Between 0 and 1.
    """
    # we assume that topography is never "naturally" masked
    mask = existing_mask.copy()
    srtm_mask = False
    num_tokens_to_mask = int(
        ((NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) - 1)) + 1) * mask_ratio - sum(sum(mask))
    )
    assert num_tokens_to_mask > 0

    def mask_topography(srtm_mask, num_tokens_to_mask, mask_ratio):
        should_flip = random() < mask_ratio
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
            unmasked_tokens = all_tokens_mask == False
            idx = np.flatnonzero(unmasked_tokens)
            np.random.shuffle(idx)
            idx = idx[:num_tokens_to_mask]
            all_tokens_mask[idx] = True
            mask = all_tokens_mask.reshape((NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)))
        return mask

    # RANDOM BANDS
    if strategy == "random_combinations":
        srtm_mask, num_tokens_to_mask = mask_topography(srtm_mask, num_tokens_to_mask, mask_ratio)
        mask = random_masking(mask, num_tokens_to_mask)

    elif strategy == "group_bands":
        srtm_mask, num_tokens_to_mask = mask_topography(srtm_mask, num_tokens_to_mask, mask_ratio)
        # next, we figure out how many tokens we can mask
        num_band_groups_to_mask = int(num_tokens_to_mask / NUM_TIMESTEPS)
        assert (num_tokens_to_mask - NUM_TIMESTEPS * num_band_groups_to_mask) >= 0
        num_tokens_masked = 0
        # tuple because of mypy, which thinks lists can only hold one type
        band_groups: List[Any] = list(range(len(BANDS_GROUPS_IDX)))
        band_groups.remove(SRTM_INDEX)
        band_groups_to_mask = sample(band_groups, num_band_groups_to_mask)
        for band_group in band_groups_to_mask:
            num_tokens_masked += int(len(mask[:, band_group]) - sum(mask[:, band_group]))
            mask[:, band_group] = True
        num_tokens_to_mask -= num_tokens_masked
        mask = random_masking(mask, num_tokens_to_mask)

    # RANDOM TIMESTEPS
    elif strategy == "random_timesteps":
        srtm_mask, num_tokens_to_mask = mask_topography(srtm_mask, num_tokens_to_mask, mask_ratio)
        # -1 for SRTM
        timesteps_to_mask = int(num_tokens_to_mask / (len(BANDS_GROUPS_IDX) - 1))
        max_tokens_masked = (len(BANDS_GROUPS_IDX) - 1) * timesteps_to_mask
        timesteps = sample(TIMESTEPS_IDX, k=timesteps_to_mask)
        if timesteps_to_mask > 0:
            num_tokens_to_mask -= int(max_tokens_masked - sum(sum(mask[timesteps])))
            mask[timesteps] = True
        mask = random_masking(mask, num_tokens_to_mask)
    elif strategy == "chunk_timesteps":
        srtm_mask, num_tokens_to_mask = mask_topography(srtm_mask, num_tokens_to_mask, mask_ratio)
        # -1 for SRTM
        timesteps_to_mask = int(num_tokens_to_mask / (len(BANDS_GROUPS_IDX) - 1))
        if timesteps_to_mask > 0:
            max_tokens_masked = (len(BANDS_GROUPS_IDX) - 1) * timesteps_to_mask
            start_idx = randint(0, NUM_TIMESTEPS - timesteps_to_mask)
            num_tokens_to_mask -= int(
                max_tokens_masked - sum(sum(mask[start_idx : start_idx + timesteps_to_mask]))
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

    def __post_init__(self):
        for strategy in self.strategies:
            assert strategy in [
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ]

    def mask_data(self, eo_data: np.ndarray, mask: np.ndarray):
        strategy = choice(self.strategies)
        mask = make_mask_no_dw(strategy=strategy, mask_ratio=self.ratio, existing_mask=mask)
        x = eo_data * ~mask
        y = np.zeros(eo_data.shape).astype(np.float32)
        y[mask] = eo_data[mask]

        return mask, x, y, strategy
