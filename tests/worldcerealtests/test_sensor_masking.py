import numpy as np
import pandas as pd

from worldcereal.train.datasets import (
    NODATAVALUE,
    SensorMaskingConfig,
    WorldCerealDataset,
)


def _make_dummy_df(num_timesteps: int, nrows: int = 3):
    rows = []
    for i in range(nrows):
        d = {
            "lat": 50.0 + i * 0.01,
            "lon": 4.0 + i * 0.01,
            "available_timesteps": num_timesteps,
            "valid_position": num_timesteps // 2,
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
        }
        # fill bands
        for t in range(num_timesteps):
            d[f"OPTICAL-B02-ts{t}-10m"] = 100 + t
            d[f"OPTICAL-B03-ts{t}-10m"] = 101 + t
            d[f"OPTICAL-B04-ts{t}-10m"] = 102 + t
            d[f"OPTICAL-B05-ts{t}-20m"] = 103 + t
            d[f"OPTICAL-B06-ts{t}-20m"] = 104 + t
            d[f"OPTICAL-B07-ts{t}-20m"] = 105 + t
            d[f"OPTICAL-B08-ts{t}-10m"] = 106 + t
            d[f"OPTICAL-B8A-ts{t}-20m"] = 107 + t
            d[f"OPTICAL-B11-ts{t}-20m"] = 108 + t
            d[f"OPTICAL-B12-ts{t}-20m"] = 109 + t
            d[f"SAR-VH-ts{t}-20m"] = 0.2 + 0.01 * t  # positive for dB conv
            d[f"SAR-VV-ts{t}-20m"] = 0.3 + 0.01 * t
            d[f"METEO-precipitation_flux-ts{t}-100m"] = 5 + t
            d[f"METEO-temperature_mean-ts{t}-100m"] = 1500 + 10 * t
        # dem
        d["DEM-alt-20m"] = 250
        d["DEM-slo-20m"] = 3
        rows.append(d)
    return pd.DataFrame(rows)


def test_s1_full_dropout():
    num_timesteps = 8
    df = _make_dummy_df(num_timesteps, nrows=1)
    cfg = SensorMaskingConfig(enable=True, s1_full_dropout_prob=1.0)
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    sample = ds[0]
    # After full dropout all s1 timesteps should be NODATAVALUE
    assert np.all(sample.s1 == NODATAVALUE), "S1 full dropout failed"


def test_s2_cloud_block_and_timestep():
    num_timesteps = 10
    df = _make_dummy_df(num_timesteps, nrows=1)
    cfg = SensorMaskingConfig(
        enable=True,
        s2_cloud_block_prob=1.0,  # always apply block
        s2_cloud_block_min=3,
        s2_cloud_block_max=3,
        s2_cloud_timestep_prob=0.5,  # may add extra masked timesteps outside block
        seed=42,
    )
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    sample = ds[0]
    s2 = sample.s2[0, 0]  # shape (T, bands)
    masked_tsteps = np.where(s2[:, 0] == NODATAVALUE)[0]
    # We expect at least the block length masked
    assert masked_tsteps.size >= 3, "S2 cloud block not applied"

    # Ensure that block is contiguous of length >= 3
    # Find longest contiguous sequence
    if masked_tsteps.size:
        diffs = np.diff(masked_tsteps)
        breaks = np.where(diffs != 1)[0]
        start_idx = 0
        longest = 1
        for b in list(breaks) + [len(diffs)]:
            seg_len = b - start_idx + 1
            longest = max(longest, seg_len)
            start_idx = b + 1
        assert longest >= 3, "Contiguous S2 cloud block length < expected"


def test_per_timestep_meteo_dropout():
    num_timesteps = 6
    df = _make_dummy_df(num_timesteps, nrows=1)
    cfg = SensorMaskingConfig(enable=True, meteo_timestep_dropout_prob=0.9, seed=7)
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    sample = ds[0]
    meteo = sample.meteo[0, 0]  # shape (T, bands)
    masked = np.sum(meteo[:, 0] == NODATAVALUE)
    assert masked > 0, "Expected some meteo timesteps masked"


def test_dem_dropout():
    num_timesteps = 6
    df = _make_dummy_df(num_timesteps, nrows=1)
    cfg = SensorMaskingConfig(enable=True, dem_dropout_prob=1.0)
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    sample = ds[0]
    assert np.all(sample.dem == NODATAVALUE), "DEM dropout failed"
