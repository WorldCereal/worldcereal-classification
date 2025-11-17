from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from einops import rearrange, repeat
from loguru import logger
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
from torch.utils.data import Dataset, WeightedRandomSampler

# minimum distance from valid_position to the edges when augmenting
# we need to define it globally so that it can be used in process_parquet as well
MIN_EDGE_BUFFER = 2


def get_class_weights(
    labels: np.ndarray[Any, Any],
    method: str = "balanced",  # 'balanced', 'log', or 'none'
    clip_range: Optional[tuple] = None,  # e.g. (0.2, 10.0)
    normalize: bool = True,
) -> Dict[int, float]:
    """
    Compute class weights for classification tasks.

    Args:
        labels: list of integer class labels.
        method: 'balanced' (scikit-learn style), or 'log' (log-scaled), or 'none'.
        clip_range: tuple (min, max) to clip weights.
        normalize: whether to rescale weights to mean = 1.

    Returns:
        class_weights_dict: dict mapping class index → weight
    """
    counts = Counter(labels)
    classes = sorted(counts.keys())
    total_samples = sum(counts.values())
    num_classes = len(classes)
    freq = np.array([counts[c] for c in classes], dtype=np.float32)

    if method == "balanced":
        weights = total_samples / (num_classes * freq)
    elif method == "log":
        inv_freq = 1.0 / freq
        weights = np.log1p(inv_freq / np.mean(inv_freq))
    elif method == "none":
        weights = np.ones_like(freq)
    else:
        raise ValueError(f"Unknown method: {method}")

    if clip_range:
        logger.info(f"Clipping weights to range {clip_range}")
        weights = np.clip(weights, clip_range[0], clip_range[1])

    if normalize:
        logger.info("Renormalizing weights to mean = 1")
        weights = weights / weights.mean()

    return dict(zip(classes, weights))


@dataclass
class SensorMaskingConfig:
    """Configuration for simulating real-world missing data scenarios.

    Probabilities are applied independently per sample. Values are in [0,1].
    Set config to None or enabled=False to disable masking.

    Attributes
    ----------
    enable: bool
        Master switch.
    s1_full_dropout_prob: float
        Probability that all S1 timesteps (VV & VH) are missing (e.g. prolonged platform outage).
    s1_timestep_dropout_prob: float
        Probability applied per timestep to drop S1 values (sporadic acquisition gaps).
    s2_cloud_timestep_prob: float
        Probability applied per timestep to cloud-mask S2 (all optical bands) individually.
    s2_cloud_block_prob: float
        Probability to create a contiguous cloud block of S2 masked timesteps.
    s2_cloud_block_min: int
        Minimum length of the contiguous S2 cloud block.
    s2_cloud_block_max: int
        Maximum length of the contiguous S2 cloud block.
    meteo_timestep_dropout_prob: float
        Probability applied per timestep to mask meteorological data.
    dem_dropout_prob: float
        Probability to mask DEM (rare but possible missing elevation ancillary data).
    seed: Optional[int]
        Optional random seed for reproducibility at dataset construction time.
    """

    enable: bool = False
    s1_full_dropout_prob: float = 0.0
    s1_timestep_dropout_prob: float = 0.0
    s2_cloud_timestep_prob: float = 0.0
    s2_cloud_block_prob: float = 0.0
    s2_cloud_block_min: int = 2
    s2_cloud_block_max: int = 5
    meteo_timestep_dropout_prob: float = 0.0
    dem_dropout_prob: float = 0.0
    seed: Optional[int] = None

    def validate(self, num_timesteps: int):
        if self.s2_cloud_block_min > self.s2_cloud_block_max:
            raise ValueError(
                "s2_cloud_block_min cannot be greater than s2_cloud_block_max"
            )
        if self.s2_cloud_block_max > num_timesteps:
            raise ValueError("s2_cloud_block_max cannot exceed num_timesteps")
        for name in [
            "s1_full_dropout_prob",
            "s1_timestep_dropout_prob",
            "s2_cloud_timestep_prob",
            "s2_cloud_block_prob",
            "meteo_timestep_dropout_prob",
            "dem_dropout_prob",
        ]:
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {v}")


class WorldCerealDataset(Dataset):
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
        "METEO-precipitation_flux-ts{}-100m": "precipitation",
        "METEO-temperature_mean-ts{}-100m": "temperature",
        "DEM-alt-20m": "elevation",
        "DEM-slo-20m": "slope",
    }

    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_timesteps: int = 12,
        timestep_freq: Literal["month", "dekad"] = "month",
        task_type: Literal["ssl", "binary", "multiclass"] = "ssl",
        num_outputs: Optional[int] = None,
        augment: bool = False,
        masking_config: Optional[SensorMaskingConfig] = None,
    ):
        """WorldCereal base dataset. This dataset is typically used for
        self-supervised learning.

        Parameters
        ----------
        dataframe : pd.DataFrame
            input dataframe containing the data
        num_timesteps : int, optional
            number of timesteps for a sample, by default 12
        timestep_freq : str, optional. Should be one of ['month', 'dekad']
            frequency of the timesteps, by default "month"
        task_type : str, optional. One of ['ssl', 'binary', 'multiclass', 'regression']
            type of the task, by default self-supervised learning "ssl"
        num_outputs : int, optional
            number of outputs for the task, by default None. If task_type is 'ssl',
            the value of this parameter is ignored.
        augment : bool, optional
            whether to augment the data, by default False
        masking_config : Optional[SensorMaskingConfig], optional
            configuration for sensor masking during training, by default None.
        """
        self.dataframe = dataframe.replace({np.nan: NODATAVALUE})
        self.num_timesteps = num_timesteps

        if timestep_freq not in ["month", "dekad"]:
            raise NotImplementedError(
                f"timestep_freq should be one of ['month', 'dekad']. Got `{timestep_freq}`"
            )
        self.timestep_freq = timestep_freq
        self.task_type = task_type
        self.num_outputs = num_outputs
        self.is_ssl = task_type == "ssl"
        self.augment = augment
        self.masking_config = masking_config
        if self.masking_config:
            if self.masking_config.seed is not None:
                # set a per-dataset RNG seed (numpy global for simplicity)
                np.random.seed(self.masking_config.seed)
            self.masking_config.validate(self.num_timesteps)
            if self.masking_config.enable:
                logger.info(
                    "Sensor masking enabled for this dataset with config: {}".format(
                        self.masking_config
                    )
                )
            else:
                logger.info(
                    "Sensor masking config provided but enable=False; masking disabled."
                )

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        row = pd.Series.to_dict(self.dataframe.iloc[idx, :])
        timestep_positions, _ = self.get_timestep_positions(row)
        return Predictors(**self.get_inputs(row, timestep_positions))

    def get_timestep_positions(
        self,
        row_d: Dict,
        min_edge_buffer: int = MIN_EDGE_BUFFER,
    ) -> Tuple[List[int], int]:
        available_timesteps = int(row_d["available_timesteps"])
        valid_position = int(row_d["valid_position"])

        # Get the center point to use for extracting a sequence of timesteps
        center_point = self._get_center_point(
            available_timesteps, valid_position, self.augment, min_edge_buffer
        )

        # Determine the timestep positions to extract
        last_timestep = min(available_timesteps, center_point + self.num_timesteps // 2)
        first_timestep = max(0, last_timestep - self.num_timesteps)
        timestep_positions = list(range(first_timestep, last_timestep))

        # Sanity check to make sure we will extract the correct number of timesteps
        if len(timestep_positions) != self.num_timesteps:
            raise ValueError(
                (
                    "Acquired timestep positions do not have correct length: "
                    f"required {self.num_timesteps}, got {len(timestep_positions)}"
                )
            )

        # Sanity check to make sure valid_position is still within the extracted timesteps
        assert (
            valid_position in timestep_positions
        ), f"Valid position {valid_position} not in timestep positions {timestep_positions}"

        return timestep_positions, valid_position

    def _get_center_point(
        self, available_timesteps, valid_position, augment, min_edge_buffer
    ):
        """Helper method to decide on the center point based on which to
        extract the timesteps."""

        if not augment or available_timesteps == self.num_timesteps:
            #  check if the valid position is too close to the start_date and force shifting it
            if valid_position < self.num_timesteps // 2:
                center_point = self.num_timesteps // 2
            #  or too close to the end_date
            elif valid_position > (available_timesteps - self.num_timesteps // 2):
                center_point = available_timesteps - self.num_timesteps // 2
            else:
                # Center the timesteps around the valid position
                center_point = valid_position
        else:
            if self.is_ssl:
                # Take a random center point enabling horizontal jittering
                center_point = int(
                    np.random.choice(
                        range(
                            self.num_timesteps // 2,
                            (available_timesteps - self.num_timesteps // 2),
                        ),
                        1,
                    )
                )
            else:
                # Randomly shift the center point but make sure the resulting range
                # well includes the valid position

                min_center_point = max(
                    self.num_timesteps // 2,
                    valid_position + min_edge_buffer - self.num_timesteps // 2,
                )
                max_center_point = min(
                    available_timesteps - self.num_timesteps // 2,
                    valid_position - min_edge_buffer + self.num_timesteps // 2,
                )

                center_point = np.random.randint(
                    min_center_point, max_center_point + 1
                )  # max_center_point included

        return center_point

    def _get_timestamps(self, row: Dict, timestep_positions: List[int]) -> np.ndarray:
        """
        Generate an array of dates based on the specified compositing window.
        """
        # adjust start date depending on the compositing window
        start_date = np.datetime64(row["start_date"], "D")
        end_date = np.datetime64(row["end_date"], "D")

        # Generate date vector depending on the compositing window
        if self.timestep_freq == "dekad":
            days, months, years = get_dekad_timestamp_components(start_date, end_date)
        elif self.timestep_freq == "month":
            days, months, years = get_monthly_timestamp_components(start_date, end_date)
        else:
            raise ValueError(f"Unknown compositing window: {self.timestep_freq}")

        return np.stack(
            [
                days[timestep_positions],
                months[timestep_positions],
                years[timestep_positions],
            ],
            axis=1,
        )

    def get_inputs(self, row_d: Dict, timestep_positions: List[int]) -> dict:
        # Get latlons which need to have spatial dims
        latlon = np.reshape(
            np.array([row_d["lat"], row_d["lon"]], dtype=np.float32), (1, 1, 2)
        )

        # Get timestamps belonging to each timestep
        timestamps = self._get_timestamps(row_d, timestep_positions)

        # Initialize inputs
        s1, s2, meteo, dem = self.initialize_inputs()

        # Fill inputs
        for src_attr, dst_atr in self.BAND_MAPPING.items():
            keys = [src_attr.format(t) for t in timestep_positions]
            values = np.array([float(row_d[key]) for key in keys], dtype=np.float32)
            idx_valid = values != NODATAVALUE
            if dst_atr in S2_BANDS:
                s2[..., S2_BANDS.index(dst_atr)] = values
            elif dst_atr in S1_BANDS:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
                s1[..., S1_BANDS.index(dst_atr)] = values
            elif dst_atr == "precipitation":
                # scaling, and AgERA5 is in mm, prometheo convention expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
                meteo[..., METEO_BANDS.index(dst_atr)] = values
            elif dst_atr == "temperature":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
                meteo[..., METEO_BANDS.index(dst_atr)] = values
            elif dst_atr in DEM_BANDS:
                values = values[0]  # dem is not temporal
                dem[..., DEM_BANDS.index(dst_atr)] = values
            else:
                raise ValueError(f"Unknown band {dst_atr}")

        # Apply masking if configured
        if self.masking_config and self.masking_config.enable:
            s1, s2, meteo, dem = self._apply_masking(s1, s2, meteo, dem)
        return dict(
            s1=s1, s2=s2, meteo=meteo, dem=dem, latlon=latlon, timestamps=timestamps
        )

    def initialize_inputs(self):
        s1 = np.full(
            (1, 1, self.num_timesteps, len(S1_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(S1_BANDS)]
        s2 = np.full(
            (1, 1, self.num_timesteps, len(S2_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(S2_BANDS)]
        meteo = np.full(
            (1, 1, self.num_timesteps, len(METEO_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(METEO_BANDS)]
        dem = np.full(
            (1, 1, len(DEM_BANDS)), fill_value=NODATAVALUE, dtype=np.float32
        )  # [H, W, len(DEM_BANDS)]

        return s1, s2, meteo, dem

    def _apply_masking(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        meteo: np.ndarray,
        dem: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply sensor/timestep masking according to the masking_config.

        Rules applied in order:
        1. Full S1 dropout (overrides timestep dropout).
        2. Per-timestep S1 dropout.
        3. S2 contiguous cloud block.
        4. Per-timestep S2 cloud dropout.
        5. Per-timestep meteo dropout.
        6. DEM dropout.
        """
        # Guard: if masking_config is None (should not happen when enable checked)
        if self.masking_config is None:
            return s1, s2, meteo, dem
        cfg: SensorMaskingConfig = self.masking_config  # type narrowing for mypy
        T = self.num_timesteps
        # 1. Full S1 dropout
        if np.random.rand() < cfg.s1_full_dropout_prob:
            s1[:] = NODATAVALUE
            # logger.debug("Applied full S1 dropout")
        else:
            # 2. Per-timestep S1 dropout
            if cfg.s1_timestep_dropout_prob > 0:
                s1_mask = np.random.rand(T) < cfg.s1_timestep_dropout_prob
                if s1_mask.any():
                    s1[..., s1_mask, :] = NODATAVALUE
                    # logger.debug(
                    #     f"Applied S1 timestep dropout on {s1_mask.sum()} of {T} timesteps"
                    # )

        # 3. S2 contiguous cloud block
        if cfg.s2_cloud_block_prob > 0 and np.random.rand() < cfg.s2_cloud_block_prob:
            block_len = np.random.randint(
                cfg.s2_cloud_block_min, cfg.s2_cloud_block_max + 1
            )
            if block_len >= T:
                start = 0
                end = T
            else:
                start = np.random.randint(0, T - block_len + 1)
                end = start + block_len
            s2[..., start:end, :] = NODATAVALUE
            # logger.debug(
            #     f"Applied S2 cloud block dropout from timestep {start} to {end - 1} (len={block_len})"
            # )

        # 4. Per-timestep S2 cloud dropout (skip already-masked timesteps)
        if cfg.s2_cloud_timestep_prob > 0:
            s2_mask = np.random.rand(T) < cfg.s2_cloud_timestep_prob
            # Avoid double logging of block; still mask independent timesteps not in block
            newly_masked = s2_mask & (s2[0, 0, :, 0] != NODATAVALUE)
            if newly_masked.any():
                s2[..., newly_masked, :] = NODATAVALUE
                # logger.debug(
                #     f"Applied S2 per-timestep cloud masking on {newly_masked.sum()} timesteps"
                # )

        # 5. Meteo per-timestep dropout
        if cfg.meteo_timestep_dropout_prob > 0:
            meteo_mask = np.random.rand(T) < cfg.meteo_timestep_dropout_prob
            if meteo_mask.any():
                meteo[..., meteo_mask, :] = NODATAVALUE
                # logger.debug(
                #     f"Applied meteo timestep dropout on {meteo_mask.sum()} timesteps"
                # )

        # 6. DEM dropout
        if cfg.dem_dropout_prob > 0 and np.random.rand() < cfg.dem_dropout_prob:
            dem[:] = NODATAVALUE
            # logger.debug("Applied DEM dropout")

        return s1, s2, meteo, dem


class WorldCerealLabelledDataset(WorldCerealDataset):
    def __init__(
        self,
        dataframe,
        task_type: Literal["binary", "multiclass"] = "binary",
        num_outputs: int = 1,
        classes_list: Union[np.ndarray, List[str]] = [],
        time_explicit: bool = False,
        label_jitter: int = 0,  # ± timesteps to jitter true label pos, for time_explicit only
        label_window: int = 0,  # ± timesteps to expand around label pos (true or moved), for time_explicit only
        return_sample_id: bool = False,
        **kwargs,
    ):
        """Labelled version of WorldCerealDataset for supervised training.
        Additional arguments are explained below.

        Parameters
        ----------
        num_outputs : int, optional
            number of outputs to supervise training on, by default 1
        classes_list : List, optional
            list of column names in the dataframe containing class labels for multiclass tasks,
            used to extract labels from each row of the dataframe, by default []
        time_explicit : bool, optional
            if True, labels respect the full temporal dimension
            to have temporally explicit outputs, by default False
        label_jitter : int, optional
            ± timesteps to jitter true label pos, for time_explicit only, by default 0.
            Only used if `time_explicit` is True.
        label_window : int, optional
            ± timesteps to expand around label pos (true or moved), for time_explicit only, by default 0.
            Only used if `time_explicit` is True.
        return_sample_id : bool, optional
            whether to return the sample_id in the output, by default False.
            If True, the sample_id will be included in the output as a separate element.
        """
        assert task_type in [
            "binary",
            "multiclass",
        ], f"Invalid task type `{task_type}` for labelled dataset"

        super().__init__(
            dataframe,
            task_type=task_type,
            num_outputs=num_outputs,
            **kwargs,
        )
        self.classes_list = classes_list
        self.time_explicit = time_explicit
        self.label_jitter = label_jitter
        self.label_window = label_window
        self.return_sample_id = return_sample_id

        if self.return_sample_id and "sample_id" not in self.dataframe.columns:
            raise ValueError(
                "`return_sample_id` is True, but 'sample_id' column not found in dataframe."
            )

    def __getitem__(self, idx):
        row = pd.Series.to_dict(self.dataframe.iloc[idx, :])
        timestep_positions, valid_position = self.get_timestep_positions(row)
        inputs = self.get_inputs(row, timestep_positions)
        label = self.get_label(
            row,
            task_type=self.task_type,
            classes_list=self.classes_list,
            valid_position=valid_position - timestep_positions[0],
        )

        predictors = Predictors(**inputs, label=label)

        if self.return_sample_id:
            sample_id = row["sample_id"]
            return predictors, sample_id
        else:
            return predictors

    def initialize_label(self):
        tsteps = self.num_timesteps if self.time_explicit else 1
        label = np.full(
            (1, 1, tsteps, 1),
            fill_value=NODATAVALUE,
            dtype=np.int32,
        )  # [H, W, T or 1, 1]

        return label

    def get_label(
        self,
        row_d: Dict,
        task_type: Literal["binary", "multiclass"] = "binary",
        classes_list: Optional[List] = None,
        valid_position: Optional[
            Union[int, Sequence[int]]
        ] = None,  # TO DO: this can also be a list of positions
    ) -> np.ndarray:
        """Get the label for the given row. Label is a 2D array based on
        the number of timesteps and number of outputs. If time_explicit is False,
        the number of timesteps will be set to 1.

        Parameters
        ----------
        row_d : Dict
            input row as a dictionary
        task_type : str, optional
            task type to infer labels from, by default "binary"
        classes_list : Optional[List], optional
            list of column names in the dataframe containing class labels for multiclass tasks,
            must be provided if task_type is "multiclass", by default None
        valid_position : int, optional
            the ‘true’ timestep index where the label lives, by default None.
            If provided and `time_explicit` is True,
            only the label at the corresponding timestep will be
            set while other timesteps will be set to NODATAVALUE.
            We’ll optionally jitter it and/or expand it into a small time‐window.

        Returns
        -------
        np.ndarray
            label array
        """

        label = self.initialize_label()
        T = self.num_timesteps

        # 1) determine base position (single int) or all-positions if not time_explicit
        base_idxs: List[int]
        if not self.time_explicit:
            base_idxs = [0]
        else:
            if valid_position is None:
                # putting label at every timestep
                base_idxs = list(range(T))
            elif isinstance(valid_position, (list, tuple, np.ndarray)):
                # bring into a flat Python list of ints
                if isinstance(valid_position, np.ndarray):
                    seq: List[int] = valid_position.astype(int).tolist()
                else:
                    seq = [int(x) for x in valid_position]
                # Apply either jittering or label_window, but not both
                if self.label_jitter > 0 and self.label_window > 0:
                    apply_jitter = np.random.choice([True, False])
                else:
                    apply_jitter = self.label_jitter > 0

                if apply_jitter:
                    # one global jitter shift
                    shift = np.random.randint(-self.label_jitter, self.label_jitter + 1)
                    seq = [int(np.clip(p + shift, 0, T - 1)) for p in seq]
                elif self.label_window > 0:
                    # one contiguous window around the min→max of seq
                    mn = min(seq)
                    mx = max(seq)
                    start = max(0, mn - self.label_window)
                    end = min(T - 1, mx + self.label_window)
                    base_idxs = list(range(start, end + 1))
                else:
                    base_idxs = seq
            else:
                # apply jitter
                # scalar valid_position must be an int here
                assert isinstance(
                    valid_position, int
                ), f"Expected single int valid_position, got {type(valid_position)}"
                p = valid_position
                if self.label_jitter > 0:
                    shift = np.random.randint(-self.label_jitter, self.label_jitter + 1)
                    p = int(np.clip(p + shift, 0, T - 1))
                # apply window expansion
                if self.label_window > 0:
                    start = max(0, p - self.label_window)
                    end = min(T - 1, p + self.label_window)
                    base_idxs = list(range(start, end + 1))
                else:
                    base_idxs = [p]

        valid_idx = np.array(base_idxs, dtype=int)

        # 2) set the labels at those indices
        if task_type == "binary":
            label[0, 0, valid_idx, 0] = int(
                not row_d["finetune_class"].startswith("not_")
            )
        elif task_type == "multiclass":
            if not classes_list:
                raise ValueError("classes_list should be provided for multiclass task")
            label[0, 0, valid_idx, 0] = classes_list.index(row_d["finetune_class"])

        return label

    def get_balanced_sampler(
        self,
        method: str = "balanced",
        clip_range: Optional[tuple] = None,  # e.g. (0.2, 10.0)
        normalize: bool = True,
        generator: Optional[Any] = None,
        sampling_class: str = "finetune_class",
    ) -> "WeightedRandomSampler":
        """
        Build a WeightedRandomSampler so that rare classes (from `balancing_class`)
        are upsampled and common classes downsampled.
        max_upsample:
            maximum upsampling factor for the rarest class (e.g. 10 means
            no class will be sampled >10× more than its frequency).
        sampling_class:
            column name in the dataframe to use for balancing.
            Default is `finetune_class`, which is the class label
            used in the training. `balancing_class` can be used as well.
        """
        # extract the sampling class (strings or ints)
        bc_vals = self.dataframe[sampling_class].values

        logger.info("Computing class weights ...")
        class_weights = get_class_weights(
            bc_vals, method, clip_range=clip_range, normalize=normalize
        )
        logger.info(f"Class weights: {class_weights}")

        # per‐sample weight
        sample_weights = np.ones_like(bc_vals).astype(np.float32)
        for k, v in class_weights.items():
            sample_weights[bc_vals == k] = v

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        return sampler


class WorldCerealTrainingDataset(WorldCerealDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_timesteps: int = 12,
        timestep_freq: Literal["month", "dekad"] = "month",
        task_type: Literal["ssl", "binary", "multiclass"] = "ssl",
        num_outputs: Optional[int] = None,
        augment: bool = False,
        masking_config: Optional[SensorMaskingConfig] = None,
        repeats: int = 1,
    ):
        super().__init__(
            dataframe=dataframe,
            num_timesteps=num_timesteps,
            timestep_freq=timestep_freq,
            task_type=task_type,
            num_outputs=num_outputs,
            augment=augment,
            masking_config=masking_config,
        )

        repeats = _check_augmentation_settings(augment, masking_config, repeats)

        base_indices = list(range(len(self.dataframe)))
        self.indices = base_indices * repeats
        self._repeats = repeats

    def __len__(self):
        # Return total repeated length, not the base dataframe length
        return len(self.indices)

    def __iter__(self):
        for idx in self.indices:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        # Map incoming idx to the original dataframe index
        real_idx = self.indices[idx]

        # Get the sample
        sample = super().__getitem__(real_idx)
        row = self.dataframe.iloc[real_idx, :]
        timestep_positions, valid_position = self.get_timestep_positions(row)
        valid_position = valid_position - timestep_positions[0]
        attrs = [
            "lat",
            "lon",
            "ref_id",
            "sample_id",
            "downstream_class",
            "valid_time",
        ]
        attrs = [attr for attr in attrs if attr in row.index]
        attrs = row[attrs].to_dict()
        attrs["valid_position"] = valid_position

        return sample, attrs


def _check_augmentation_settings(
    augment: bool, masking_config: Optional[SensorMaskingConfig], repeats: int
) -> int:
    """
    Check augmentation/masking settings. If no augmentation or masking is
    enabled but repeats > 1, set repeats to 1 and log a warning.
    If augmentation or masking is enabled but repeats = 1, log a warning
    suggesting to increase repeats for more variability.
    """
    some_augmentation = augment or (masking_config and masking_config.enable)
    if repeats == 1 and some_augmentation:
        logger.warning(
            "Dataset augmentation or masking is enabled but repeats=1. "
            "Consider setting repeats > 1 to increase training variability."
        )
    elif repeats > 1 and not some_augmentation:
        logger.warning(
            "Dataset is repeated but not augmented which is useless; "
            "consider setting `augment=True` or `masking_config` for training. "
            "Setting repeats=1 instead."
        )
        repeats = 1
    elif repeats > 1:
        logger.info(
            f"Dataset will be repeated {repeats} times for training with augmentation/masking."
        )

    return repeats


def _predictor_from_xarray(arr: xr.DataArray, epsg: int) -> Predictors:
    def _get_timestamps() -> np.ndarray:
        timestamps = arr.t.values
        years = timestamps.astype("datetime64[Y]").astype(int) + 1970
        months = timestamps.astype("datetime64[M]").astype(int) % 12 + 1
        days = timestamps.astype("datetime64[D]").astype("datetime64[M]")
        days = (timestamps - days).astype(int) + 1

        components = np.stack(
            [
                days,
                months,
                years,
            ],
            axis=1,
        )

        return components[None, ...]  # Add batch dimension

    def _initialize_eo_inputs():
        num_timesteps = arr.t.size
        h, w = arr.y.size, arr.x.size
        s1 = np.full(
            (1, h, w, num_timesteps, len(S1_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [B, H, W, T, len(S1_BANDS)]
        s2 = np.full(
            (1, h, w, num_timesteps, len(S2_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [B, H, W, T, len(S2_BANDS)]
        meteo = np.full(
            (1, h, w, num_timesteps, len(METEO_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [B, H, W, T, len(METEO_BANDS)]
        dem = np.full(
            (1, h, w, len(DEM_BANDS)), fill_value=NODATAVALUE, dtype=np.float32
        )  # [B, H, W, len(DEM_BANDS)]

        return s1, s2, meteo, dem

    # TODO: remove temporary band renaming due to old data file
    arr["bands"] = arr.bands.where(arr.bands != "temperature_2m", "temperature")
    arr["bands"] = arr.bands.where(arr.bands != "total_precipitation", "precipitation")

    # Initialize EO inputs
    s1, s2, meteo, dem = _initialize_eo_inputs()

    # Fill EO inputs
    for band in S2_BANDS + S1_BANDS + METEO_BANDS + DEM_BANDS:
        if band not in arr.bands.values:
            print(f"Band {band} not found in the input data, skipping.")
            continue  # skip bands that are not present in the data
        values = arr.sel(bands=band).values.astype(np.float32)
        idx_valid = values != NODATAVALUE
        if band in S2_BANDS:
            s2[..., S2_BANDS.index(band)] = rearrange(
                values, "t x y -> 1 y x t"
            )  # TODO check if this is correct
        elif band in S1_BANDS:
            # convert to dB
            idx_valid = idx_valid & (values > 0)
            values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            s1[..., S1_BANDS.index(band)] = rearrange(values, "t x y -> 1 y x t")
        elif band == "precipitation":
            # scaling, and AgERA5 is in mm, prometheo convention expects m
            values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            meteo[..., METEO_BANDS.index(band)] = rearrange(values, "t x y -> 1 y x t")
        elif band == "temperature":
            # remove scaling
            values[idx_valid] = values[idx_valid] / 100
            meteo[..., METEO_BANDS.index(band)] = rearrange(values, "t x y -> 1 y x t")
        elif band in DEM_BANDS:
            values = values[0]  # dem is not temporal
            dem[..., DEM_BANDS.index(band)] = rearrange(values, "x y -> 1 y x")
        else:
            raise ValueError(f"Unknown band {band}")

    # Extract the latlons
    # EPSG:4326 is the supported crs for presto
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = np.meshgrid(arr.x, arr.y)
    lon, lat = transformer.transform(x, y)
    latlon = rearrange(np.stack([lat, lon]), "c x y ->  y x c")

    predictors_dict = {
        "s1": rearrange(s1, "1 h w t c -> (h w) 1 1 t c"),
        "s2": rearrange(s2, "1 h w t c -> (h w) 1 1 t c"),
        "meteo": rearrange(meteo, "1 h w t c -> (h w) 1 1 t c"),
        "latlon": rearrange(latlon, "h w c ->  (h w) 1 1 c"),
        "dem": rearrange(dem, "1 h w c -> (h w) 1 1 c"),
        "timestamps": repeat(_get_timestamps(), "1 t d -> b t d", b=x.size),
    }

    return Predictors(**predictors_dict)


def generate_predictor(x: Union[pd.DataFrame, xr.DataArray], epsg: int) -> Predictors:
    if isinstance(x, xr.DataArray):
        return _predictor_from_xarray(x, epsg)
    raise NotImplementedError


def run_model_inference(
    inarr: Union[pd.DataFrame, xr.DataArray],
    model: nn.Module,  # Wrapper
    epsg: int = 4326,
    batch_size: int = 8192,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Runs a forward pass of the model on the input data

    Args:
        inarr (xr.DataArray or pd.DataFrame): Input data as xarray DataArray or pandas DataFrame.
        model (nn.Module): A Prometheo compatible (wrapper) model.
        epsg (int) : EPSG code describing the coordinates.
        batch_size (int): Batch size to be used for Presto inference.

    Returns:
        xr.DataArray or np.ndarray: Model output as xarray DataArray or numpy ndarray.
    """

    predictor = generate_predictor(inarr, epsg)
    # fixing the pooling method to keep the function signature the same
    # as in presto-worldcereal but this could be an input argument too
    features = (
        extract_features_from_model(model, predictor, batch_size, PoolingMethods.GLOBAL)
        .cpu()
        .numpy()
    )

    # todo - return the output tensors to the right shape, either xarray or df
    if isinstance(inarr, pd.DataFrame):
        return features
    else:
        features = rearrange(
            features, "(y x) 1 1 1 c -> x y c", x=len(inarr.x), y=len(inarr.y)
        )
        features_da = xr.DataArray(
            features,
            dims=["x", "y", "bands"],
            coords={"x": inarr.x, "y": inarr.y},
        )

        return features_da


def align_to_composite_window(
    dt_in: np.datetime64, timestep_freq: Literal["month", "dekad"]
) -> np.datetime64:
    """
    Determine the composite window start date based on the input date and compositing window.

    Parameters
    ----------
    dt_in : np.datetime64
        Input date string in a format compatible with numpy.datetime64 (e.g., 'YYYY-MM-DD').
    timestep_freq : Literal["month", "dekad"]
        The compositing window to use for determining the correct date.
        - "month": Returns the first day of the month.
        - "dekad": Returns the first day of the dekad (1st, 11th, or 21st of the month).

    Returns
    -------
    np.datetime64
        The corrected date as a numpy.datetime64 object, corresponding to the start of the specified compositing window.

    Raises
    ------
    ValueError
        If an unknown compositing window is provided.

    """
    # Extract year, month, and day
    year = dt_in.astype("object").year
    month = dt_in.astype("object").month
    day = dt_in.astype("object").day

    if timestep_freq == "dekad":
        if day <= 10:
            correct_date = np.datetime64(f"{year}-{month:02d}-01")
        elif 11 <= day <= 20:
            correct_date = np.datetime64(f"{year}-{month:02d}-11")
        else:
            correct_date = np.datetime64(f"{year}-{month:02d}-21")
    elif timestep_freq == "month":
        correct_date = np.datetime64(f"{year}-{month:02d}-01")
    else:
        raise ValueError(f"Unknown compositing window: {timestep_freq}")

    return correct_date


def get_dekad_timestamp_components(
    start_date: np.datetime64, end_date: np.datetime64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dekad (10-day period) timestamp components (day, month, year) between a start and end date.

    Parameters
    ----------
    start_date : np.datetime64
        The starting date from which to generate dekad timestamps.
    end_date : np.datetime64
        The ending date up to which dekad timestamps are generated (inclusive).

    Returns
    -------
    days : np.ndarray
        Array of day components for each dekad timestamp.
    months : np.ndarray
        Array of month components for each dekad timestamp.
    years : np.ndarray
        Array of year components for each dekad timestamp.
    """

    # Align start and end dates to the dekad window
    start_date = align_to_composite_window(start_date, "dekad")
    end_date = align_to_composite_window(end_date, "dekad")

    # Extract year, month, and day
    year = start_date.astype("object").year
    month = start_date.astype("object").month
    day = start_date.astype("object").day

    year_end = end_date.astype("object").year
    month_end = end_date.astype("object").month
    day_end = end_date.astype("object").day

    days, months, years = [day], [month], [year]
    while f"{year}-{month}-{day}" != f"{year_end}-{month_end}-{day_end}":
        if day < 21:
            day += 10
        else:
            month = month + 1 if month < 12 else 1
            year = year + 1 if month == 1 else year
            day = 1
        days.append(day)
        months.append(month)
        years.append(year)
    return np.array(days), np.array(months), np.array(years)


def get_monthly_timestamp_components(
    start_date: np.datetime64, end_date: np.datetime64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate monthly timestamp components (day, month, year) between a start and end date.

    Parameters
    ----------
    start_date : np.datetime64
        The starting date from which to generate month timestamps.
    end_date : np.datetime64
        The ending date up to which to generate month timestamps.

    Returns
    -------
    days : np.ndarray
        Array of day components for each month timestamp.
    months : np.ndarray
        Array of month components for each month timestamp.
    years : np.ndarray
        Array of year components for each month timestamp.
    """

    # Align start and end dates to the first day of the month
    start_date = align_to_composite_window(start_date, "month")
    end_date = align_to_composite_window(end_date, "month")

    # Truncate to month precision (year and month only, day is dropped)
    start_month = np.datetime64(start_date, "M")
    end_month = np.datetime64(end_date, "M")
    num_timesteps = (end_month - start_month).astype(int) + 1

    # generate date vector based on the number of timesteps
    date_vector = start_month + np.arange(num_timesteps, dtype="timedelta64[M]")

    # generate day, month and year vectors with numpy operations
    days = np.ones(len(date_vector), dtype=int)
    months = (date_vector.astype("datetime64[M]").astype(int) % 12) + 1
    years = (date_vector.astype("datetime64[Y]").astype(int)) + 1970
    return days, months, years
