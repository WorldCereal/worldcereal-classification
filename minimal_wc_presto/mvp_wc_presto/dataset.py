import logging
from datetime import datetime
from math import modf
from pathlib import Path
from random import sample
from typing import Callable, Dict, List, Optional, Tuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from einops import rearrange, repeat
from pyproj import Transformer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from .dataops import (
    BANDS,
    BANDS_GROUPS_IDX,
    NORMED_BANDS,
    S1_S2_ERA5_SRTM,
    DynamicWorld2020_2021,
)
from .masking import BAND_EXPANSION, MaskedExample, MaskParamsNoDw
from .utils import DEFAULT_SEED, data_dir, load_world_df

logger = logging.getLogger("__main__")

IDX_TO_BAND_GROUPS = {}
for band_group_idx, (key, val) in enumerate(BANDS_GROUPS_IDX.items()):
    for idx in val:
        IDX_TO_BAND_GROUPS[NORMED_BANDS[idx]] = band_group_idx


class WorldCerealBase(Dataset):
    _NODATAVALUE = 65535
    NUM_TIMESTEPS = 12
    BAND_MAPPING = {
        "OPTICAL-B02-ts{}-10m": "B2",
        "OPTICAL-B03-ts{}-10m": "B3",
        "OPTICAL-B04-ts{}-10m": "B4",
        "OPTICAL-B05-ts{}-20m": "B5",
        "OPTICAL-B06-ts{}-20m": "B6",
        "OPTICAL-B07-ts{}-20m": "B7",
        "OPTICAL-B08-ts{}-10m": "B8",
        "OPTICAL-B8A-ts{}-20m": "B8A",
        "OPTICAL-B11-ts{}-20m": "B11",
        "OPTICAL-B12-ts{}-20m": "B12",
        "SAR-VH-ts{}-20m": "VH",
        "SAR-VV-ts{}-20m": "VV",
        "METEO-precipitation_flux-ts{}-100m": "total_precipitation",
        "METEO-temperature_mean-ts{}-100m": "temperature_2m",
    }
    STATIC_BAND_MAPPING = {"DEM-alt-20m": "elevation", "DEM-slo-20m": "slope"}

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    @staticmethod
    def target_crop(row_d: Dict) -> int:
        # by default, we predict crop vs non crop
        return int(row_d["LANDCOVER_LABEL"] == 11)

    @classmethod
    def row_to_arrays(
        cls, row: pd.Series, target_function: Callable[[Dict], int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        # https://stackoverflow.com/questions/45783891/is-there-a-way-to-speed-up-the-pandas-getitem-getitem-axis-and-get-label
        # This is faster than indexing the series every time!
        row_d = pd.Series.to_dict(row)

        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)
        month = datetime.strptime(row_d["start_date"], "%Y-%m-%d").month - 1

        eo_data = np.zeros((cls.NUM_TIMESTEPS, len(BANDS)))
        # an assumption we make here is that all timesteps for a token
        # have the same masking
        mask = np.zeros((cls.NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)))
        for df_val, presto_val in cls.BAND_MAPPING.items():
            values = np.array([float(row_d[df_val.format(t)]) for t in range(cls.NUM_TIMESTEPS)])
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(values, nan=cls._NODATAVALUE)
            idx_valid = values != cls._NODATAVALUE
            if presto_val in ["VV", "VH"]:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            elif presto_val == "total_precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            elif presto_val == "temperature_2m":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid
            eo_data[:, BANDS.index(presto_val)] = values
        for df_val, presto_val in cls.STATIC_BAND_MAPPING.items():
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(row_d[df_val], nan=cls._NODATAVALUE)
            idx_valid = values != cls._NODATAVALUE
            eo_data[:, BANDS.index(presto_val)] = values
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid

        return (
            cls.check(eo_data),
            mask.astype(bool),
            latlon,
            month,
            target_function(row_d),
        )

    def __getitem__(self, idx):
        raise NotImplementedError

    @classmethod
    def normalize_and_mask(cls, eo: np.ndarray):
        # TODO: this can be removed
        keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
        normed_eo = S1_S2_ERA5_SRTM.normalize(eo)
        # TODO: fix this. For now, we replicate the previous behaviour
        normed_eo = np.where(eo[:, keep_indices] != cls._NODATAVALUE, normed_eo, 0)
        return normed_eo

    @staticmethod
    def check(array: np.ndarray) -> np.ndarray:
        assert not np.isnan(array).any()
        return array

    @staticmethod
    def split_df(
        df: pd.DataFrame, val_sample_ids: Optional[List[str]] = None, val_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if val_sample_ids is None:
            logger.warning(f"No val_ids; randomly splitting {val_size} to the val set instead")
            val, train = np.split(
                df.sample(frac=1, random_state=DEFAULT_SEED), [int(val_size * len(df))]
            )
        else:
            is_val = df.sample_id.isin(val_sample_ids)
            logger.info(f"Using {len(is_val) - sum(is_val)} train and {sum(is_val)} val samples")
            train = df[~is_val]
            val = df[is_val]
        return train, val


class WorldCerealMaskedDataset(WorldCerealBase):
    def __init__(self, dataframe: pd.DataFrame, mask_params: MaskParamsNoDw):
        super().__init__(dataframe)
        self.mask_params = mask_params

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, real_mask_per_token, latlon, month, _ = self.row_to_arrays(row, self.target_crop)
        mask_eo, x_eo, y_eo, strat = self.mask_params.mask_data(
            self.normalize_and_mask(eo), real_mask_per_token
        )
        real_mask_per_variable = np.repeat(real_mask_per_token, BAND_EXPANSION, axis=1)

        dynamic_world = np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount)
        mask_dw = np.full(self.NUM_TIMESTEPS, True)
        y_dw = dynamic_world.copy()
        return MaskedExample(
            mask_eo,
            mask_dw,
            x_eo,
            y_eo,
            dynamic_world,
            y_dw,
            month,
            latlon,
            strat,
            real_mask_per_variable,
        )


def filter_remove_noncrops(df: pd.DataFrame) -> pd.DataFrame:
    crop_labels = [10, 11, 12, 13]
    df = df.loc[df.LANDCOVER_LABEL.isin(crop_labels)]
    return df


def target_maize(row_d) -> int:
    # 1200 is maize
    return int(row_d["CROPTYPE_LABEL"] == 1200)


class WorldCerealLabelledDataset(WorldCerealBase):
    # 0: no information, 10: could be both annual or perennial
    FILTER_LABELS = [0, 10]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        countries_to_remove: Optional[List[str]] = None,
        years_to_remove: Optional[List[int]] = None,
        target_function: Optional[Callable[[Dict], int]] = None,
        balance: bool = False,
    ):
        dataframe = dataframe.loc[~dataframe.LANDCOVER_LABEL.isin(self.FILTER_LABELS)]

        if countries_to_remove is not None:
            dataframe = self.join_with_world_df(dataframe)
            for country in countries_to_remove:
                assert dataframe.name.str.contains(
                    country
                ).any(), f"Tried removing {country} but it is not in the dataframe"
            dataframe = dataframe[(~dataframe.name.isin(countries_to_remove))]
        if years_to_remove is not None:
            dataframe["end_date"] = pd.to_datetime(dataframe.end_date)
            dataframe = dataframe[(~dataframe.end_date.dt.year.isin(years_to_remove))]
        self.target_function = target_function if target_function is not None else self.target_crop
        self._class_weights: Optional[np.ndarray] = None

        super().__init__(dataframe)
        if balance:
            neg_indices, pos_indices = [], []
            for loc_idx, (_, row) in enumerate(self.df.iterrows()):
                target = self.target_function(row.to_dict())
                if target == 0:
                    neg_indices.append(loc_idx)
                else:
                    pos_indices.append(loc_idx)
            if len(pos_indices) > len(neg_indices):
                self.indices = pos_indices + (len(pos_indices) // len(neg_indices)) * neg_indices
            elif len(neg_indices) > len(pos_indices):
                self.indices = neg_indices + (len(neg_indices) // len(pos_indices)) * pos_indices
            else:
                self.indices = neg_indices + pos_indices
        else:
            self.indices = [i for i in range(len(self.df))]

    @staticmethod
    def multiply_list_length_by_float(input_list: List, multiplier: float) -> List:
        decimal_part, integer_part = modf(multiplier)
        sublist = sample(input_list, k=int(len(input_list) * decimal_part))
        return input_list * int(integer_part) + sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the sample
        df_index = self.indices[idx]
        row = self.df.iloc[df_index, :]
        eo, mask_per_token, latlon, month, target = self.row_to_arrays(row, self.target_function)
        mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)
        return (
            self.normalize_and_mask(eo),
            target,
            np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount),
            latlon,
            month,
            mask_per_variable,
        )

    @staticmethod
    def join_with_world_df(dataframe: pd.DataFrame) -> pd.DataFrame:
        world_df = load_world_df()
        dataframe = gpd.GeoDataFrame(
            data=dataframe,
            geometry=gpd.GeoSeries.from_xy(x=dataframe.lon, y=dataframe.lat),
            crs="EPSG:4326",
        )
        # project to non geographic CRS, otherwise geopandas gives a warning
        joined = gpd.sjoin_nearest(
            dataframe.to_crs("EPSG:3857"), world_df.to_crs("EPSG:3857"), how="left"
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        if joined.isna().any(axis=1).any():
            logger.warning("Some coordinates couldn't be matched to a country")
        return joined.to_crs("EPSG:4326")

    @property
    def class_weights(self) -> np.ndarray:
        if self._class_weights is None:
            ys = []
            for _, row in self.df.iterrows():
                ys.append(self.target_function(row.to_dict()))
            self._class_weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(ys), y=ys
            )
        return self._class_weights


class WorldCerealInferenceDataset(Dataset):
    _NODATAVALUE = 65535
    Y = "worldcereal_cropland"
    BAND_MAPPING = {
        "B02": "B2",
        "B03": "B3",
        "B04": "B4",
        "B05": "B5",
        "B06": "B6",
        "B07": "B7",
        "B08": "B8",
        # B8A is missing
        "B11": "B11",
        "B12": "B12",
        "VH": "VH",
        "VV": "VV",
        "precipitation-flux": "total_precipitation",
        "temperature-mean": "temperature_2m",
    }

    def __init__(self, path_to_files: Path = data_dir / "inference_areas"):
        self.path_to_files = path_to_files
        self.all_files = list(self.path_to_files.glob("*.nc"))

    def __len__(self):
        return len(self.all_files)

    @classmethod
    def nc_to_arrays(
        cls, filepath: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ds = cast(xr.Dataset, rioxarray.open_rasterio(filepath, decode_times=False))
        epsg_coords = ds.rio.crs.to_epsg()

        num_instances = len(ds.x) * len(ds.y)
        num_timesteps = len(ds.t)
        eo_data = np.zeros((num_instances, num_timesteps, len(BANDS)))
        mask = np.zeros((num_instances, num_timesteps, len(BANDS_GROUPS_IDX)))
        # for now, B8A is missing
        mask[:, :, IDX_TO_BAND_GROUPS["B8A"]] = 1

        for org_band, presto_val in cls.BAND_MAPPING.items():
            # flatten the values
            values = np.swapaxes(ds[org_band].values.reshape((num_timesteps, -1)), 0, 1)
            idx_valid = values != cls._NODATAVALUE

            if presto_val in ["VV", "VH"]:
                # convert to dB
                values = 20 * np.log10(values) - 83
            elif presto_val == "total_precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values = values / (100 * 1000.0)
            elif presto_val == "temperature_2m":
                # remove scaling
                values = values / 100

            eo_data[:, :, BANDS.index(presto_val)] = values
            mask[:, :, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid

        y = rearrange(ds[cls.Y].values, "t x y -> (x y) t")
        # -1 because we index from 0
        start_month = (ds.t.values[0].astype("datetime64[M]").astype(int) % 12 + 1) - 1
        months = np.ones((num_instances)) * start_month

        transformer = Transformer.from_crs(f"EPSG:{epsg_coords}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(ds.x, ds.y)

        latlons = np.stack(
            [np.repeat(lat, repeats=len(lon)), repeat(lon, "c -> (h c)", h=len(lat))],
            axis=-1,
        )

        return eo_data, np.repeat(mask, BAND_EXPANSION, axis=-1), latlons, months, y

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        eo, mask, latlons, months, y = self.nc_to_arrays(filepath)

        dynamic_world = np.ones((eo.shape[0], eo.shape[1])) * (DynamicWorld2020_2021.class_amount)

        return S1_S2_ERA5_SRTM.normalize(eo), dynamic_world, mask, latlons, months, y

    @staticmethod
    def combine_predictions(
        latlons: np.ndarray, all_preds: np.ndarray, gt: np.ndarray, ndvi: np.ndarray
    ) -> pd.DataFrame:
        flat_lat, flat_lon = latlons[:, 0], latlons[:, 1]
        if len(all_preds.shape) == 1:
            all_preds = np.expand_dims(all_preds, axis=-1)

        data_dict: Dict[str, np.ndarray] = {"lat": flat_lat, "lon": flat_lon}
        for i in range(all_preds.shape[1]):
            prediction_label = f"prediction_{i}"
            data_dict[prediction_label] = all_preds[:, i]
        data_dict["ground_truth"] = gt[:, 0]
        data_dict["ndvi"] = ndvi
        return pd.DataFrame(data=data_dict).set_index(["lat", "lon"])
