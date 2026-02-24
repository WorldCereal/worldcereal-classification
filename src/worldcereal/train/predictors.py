"""Predictor builders shared between training and inference flows."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr
from einops import rearrange, repeat
from prometheo.infer import extract_features_from_model
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    Predictors,
)
from pyproj import Transformer
from torch import nn


def _predictor_from_xarray(arr: xr.DataArray, epsg: int) -> Predictors:
    def _get_timestamps() -> np.ndarray:
        timestamps = arr.t.values
        years = timestamps.astype("datetime64[Y]").astype(int) + 1970
        months = timestamps.astype("datetime64[M]").astype(int) % 12 + 1
        days = timestamps.astype("datetime64[D]").astype("datetime64[M]")
        days = (timestamps - days).astype(int) + 1

        components = np.stack([days, months, years], axis=1)
        return components[None, ...]  # Add batch dimension

    def _initialize_eo_inputs() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        num_timesteps = arr.t.size
        h, w = len(arr.y), len(arr.x)
        s1 = np.full(
            (1, h, w, num_timesteps, len(S1_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        s2 = np.full(
            (1, h, w, num_timesteps, len(S2_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        meteo = np.full(
            (1, h, w, num_timesteps, len(METEO_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        dem = np.full(
            (1, h, w, len(DEM_BANDS)), fill_value=NODATAVALUE, dtype=np.float32
        )
        return s1, s2, meteo, dem

    # Temporary renames for legacy inputs
    arr["bands"] = arr.bands.where(arr.bands != "temperature_2m", "temperature")
    arr["bands"] = arr.bands.where(arr.bands != "total_precipitation", "precipitation")

    s1, s2, meteo, dem = _initialize_eo_inputs()

    for band in S2_BANDS + S1_BANDS + METEO_BANDS + DEM_BANDS:
        if band not in arr.bands.values:
            continue
        values = arr.sel(bands=band).values.astype(np.float32)
        idx_valid = values != NODATAVALUE
        if band in S2_BANDS:
            s2[..., S2_BANDS.index(band)] = rearrange(values, "t x y -> 1 y x t")
        elif band in S1_BANDS:
            idx_valid = idx_valid & (values > 0)
            values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            s1[..., S1_BANDS.index(band)] = rearrange(values, "t x y -> 1 y x t")
        elif band == "precipitation":
            values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            meteo[..., METEO_BANDS.index("precipitation")] = rearrange(
                values, "t x y -> 1 y x t"
            )
        elif band == "temperature":
            values[idx_valid] = values[idx_valid] / 100
            meteo[..., METEO_BANDS.index("temperature")] = rearrange(
                values, "t x y -> 1 y x t"
            )
        elif band in DEM_BANDS:
            dem[..., DEM_BANDS.index(band)] = rearrange(values[0], "x y -> 1 y x")

    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = np.meshgrid(arr.x, arr.y)
    lon, lat = transformer.transform(x, y)
    latlon = rearrange(np.stack([lat, lon]), "c x y -> y x c")

    predictors_dict: dict[str, Any] = {
        "s1": rearrange(s1, "1 h w t c -> (h w) 1 1 t c"),
        "s2": rearrange(s2, "1 h w t c -> (h w) 1 1 t c"),
        "meteo": rearrange(meteo, "1 h w t c -> (h w) 1 1 t c"),
        "latlon": rearrange(latlon, "h w c -> (h w) 1 1 c"),
        "dem": rearrange(dem, "1 h w c -> (h w) 1 1 c"),
        "timestamps": repeat(_get_timestamps(), "1 t d -> b t d", b=x.size),
    }

    return Predictors(**predictors_dict)


def generate_predictor(x: Union[pd.DataFrame, xr.DataArray], epsg: int) -> Predictors:
    if isinstance(x, xr.DataArray):
        return _predictor_from_xarray(x, epsg)
    raise NotImplementedError("DataFrame inputs are not supported yet")


def run_model_inference(
    inarr: Union[pd.DataFrame, xr.DataArray],
    model: nn.Module,
    epsg: int = 4326,
    batch_size: int = 8192,
) -> Union[np.ndarray, xr.DataArray]:
    predictor = generate_predictor(inarr, epsg)
    features = (
        extract_features_from_model(model, predictor, batch_size, PoolingMethods.GLOBAL)
        .cpu()
        .numpy()
    )

    if isinstance(inarr, pd.DataFrame):
        return features

    features = rearrange(
        features, "(y x) 1 1 1 c -> x y c", x=len(inarr.x), y=len(inarr.y)
    )
    return xr.DataArray(
        features, dims=["x", "y", "bands"], coords={"x": inarr.x, "y": inarr.y}
    )
