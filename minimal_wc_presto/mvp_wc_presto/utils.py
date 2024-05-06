import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
import torch
import xarray as xr

from .dataops import (
    BANDS,
    ERA5_BANDS,
    NORMED_BANDS,
    REMOVED_BANDS,
    S1_BANDS,
    S1_S2_ERA5_SRTM,
    S2_BANDS,
    SRTM_BANDS,
    DynamicWorld2020_2021,
)

logger = logging.getLogger("__main__")

data_dir = Path(__file__).parent.parent / "data"
config_dir = Path(__file__).parent.parent / "config"
default_model_path = data_dir / "default_model.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED: int = 42


# From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int = DEFAULT_SEED):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def initialize_logging(output_dir: Union[str, Path], to_file=True, logger_name="__main__"):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.setLevel(logging.INFO)

    if to_file:
        path = os.path.join(output_dir, "console-output.log")
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Initialized logging to %s" % path)
    return logger


def timestamp_dirname(suffix: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    return f"{ts}_{suffix}" if suffix is not None else ts


def construct_single_presto_input(
    s1: Optional[torch.Tensor] = None,
    s1_bands: Optional[List[str]] = None,
    s2: Optional[torch.Tensor] = None,
    s2_bands: Optional[List[str]] = None,
    era5: Optional[torch.Tensor] = None,
    era5_bands: Optional[List[str]] = None,
    srtm: Optional[torch.Tensor] = None,
    srtm_bands: Optional[List[str]] = None,
    dynamic_world: Optional[torch.Tensor] = None,
    normalize: bool = True,
):
    """
    Inputs are paired into a tensor input <X> and a list <X>_bands, which describes <X>.

    <X> should have shape (num_timesteps, len(<X>_bands)), with the following bands possible for
    each input:

    s1: ["VV", "VH"]
    s2: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    era5: ["temperature_2m", "total_precipitation"]
        "temperature_2m": Temperature of air at 2m above the surface of land,
            sea or in-land waters in Kelvin (K)
        "total_precipitation": Accumulated liquid and frozen water, including rain and snow,
            that falls to the Earth's surface. Measured in metres (m)
    srtm: ["elevation", "slope"]

    dynamic_world is a 1d input of shape (num_timesteps,) representing the dynamic world classes
        of each timestep for that pixel
    """
    num_timesteps_list = [x.shape[0] for x in [s1, s2, era5, srtm] if x is not None]
    if dynamic_world is not None:
        num_timesteps_list.append(len(dynamic_world))

    assert len(num_timesteps_list) > 0
    assert all(num_timesteps_list[0] == timestep for timestep in num_timesteps_list)
    num_timesteps = num_timesteps_list[0]
    mask, x = torch.ones(num_timesteps, len(BANDS)), torch.zeros(num_timesteps, len(BANDS))

    for band_group in [
        (s1, s1_bands, S1_BANDS),
        (s2, s2_bands, S2_BANDS),
        (era5, era5_bands, ERA5_BANDS),
        (srtm, srtm_bands, SRTM_BANDS),
    ]:
        data, input_bands, output_bands = band_group
        if data is not None:
            assert input_bands is not None
        else:
            continue

        kept_output_bands = [x for x in output_bands if x not in REMOVED_BANDS]
        # construct a mapping from the input bands to the expected bands
        kept_input_band_idxs = [i for i, val in enumerate(input_bands) if val in kept_output_bands]
        kept_input_band_names = [val for val in input_bands if val in kept_output_bands]

        input_to_output_mapping = [BANDS.index(val) for val in kept_input_band_names]

        x[:, input_to_output_mapping] = data[:, kept_input_band_idxs]
        mask[:, input_to_output_mapping] = 0

    if dynamic_world is None:
        dynamic_world = torch.ones(num_timesteps) * (DynamicWorld2020_2021.class_amount)

    keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
    mask = mask[:, keep_indices]

    if normalize:
        # normalize includes x = x[:, keep_indices]
        x = S1_S2_ERA5_SRTM.normalize(x)
        if s2_bands is not None:
            if ("B8" in s2_bands) and ("B4" in s2_bands):
                mask[:, NORMED_BANDS.index("NDVI")] = 0
    else:
        x = x[:, keep_indices]
    return x, mask, dynamic_world


def load_world_df() -> pd.DataFrame:
    # this could be memoized, but it should only be called 2 or 3 times in a run
    filename = "world-administrative-boundaries/world-administrative-boundaries.shp"
    world_df = gpd.read_file(data_dir / filename)
    world_df = world_df.drop(columns=["iso3", "status", "color_code", "iso_3166_1_"])
    return world_df
