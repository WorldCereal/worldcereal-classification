import numpy as np
import pandas as pd
import pytest

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


def _valid_timesteps(arr):
    # arr shape [H, W, T, bands] -> boolean [T], True where any band has data
    return (arr[0, 0] != NODATAVALUE).any(axis=-1)


def test_joint_guard_restores_s2_when_s1_disabled():
    # S1 intentionally fully dropped; a full-length S2 cloud block would wipe
    # S2 too, so the guard must restore exactly one S2 timestep.
    num_timesteps = 8
    df = _make_dummy_df(num_timesteps, nrows=1)
    cfg = SensorMaskingConfig(
        enable=True,
        s1_full_dropout_prob=1.0,
        s2_cloud_block_prob=1.0,
        s2_cloud_block_min=num_timesteps,
        s2_cloud_block_max=num_timesteps,
        seed=0,
    )
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    sample = ds[0]
    assert np.all(sample.s1 == NODATAVALUE), "S1 full dropout should stay intact"
    assert _valid_timesteps(sample.s2).sum() == 1, (
        "Guard should restore exactly one S2 timestep"
    )


def test_joint_guard_restores_s1_when_s2_disabled():
    # S2 intentionally fully masked; per-timestep S1 dropout at 1.0 would wipe
    # S1 too, so the guard must restore exactly one S1 timestep.
    num_timesteps = 8
    df = _make_dummy_df(num_timesteps, nrows=1)
    cfg = SensorMaskingConfig(
        enable=True,
        s1_timestep_dropout_prob=1.0,
        s2_cloud_timestep_prob=1.0,
        seed=0,
    )
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    sample = ds[0]
    assert np.all(sample.s2 == NODATAVALUE), "S2 full dropout should stay intact"
    assert _valid_timesteps(sample.s1).sum() == 1, (
        "Guard should restore exactly one S1 timestep"
    )


def test_joint_guard_with_real_missing_s1():
    # S1 entirely missing in the input data; masking must never wipe S2 fully.
    num_timesteps = 8
    df = _make_dummy_df(num_timesteps, nrows=1)
    for t in range(num_timesteps):
        df[f"SAR-VV-ts{t}-20m"] = NODATAVALUE
        df[f"SAR-VH-ts{t}-20m"] = NODATAVALUE
    cfg = SensorMaskingConfig(
        enable=True,
        s2_cloud_block_prob=1.0,
        s2_cloud_block_min=num_timesteps,
        s2_cloud_block_max=num_timesteps,
        seed=0,
    )
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    sample = ds[0]
    assert np.all(sample.s1 == NODATAVALUE)
    assert _valid_timesteps(sample.s2).sum() == 1, (
        "Guard should restore exactly one S2 timestep for S1-missing samples"
    )


def test_joint_guard_never_wipes_both_statistically():
    # Hammer the aggressive config: no draw may leave both S1 and S2 empty.
    num_timesteps = 12
    df = _make_dummy_df(num_timesteps, nrows=1)
    cfg = SensorMaskingConfig(
        enable=True,
        s1_full_dropout_prob=0.5,
        s1_timestep_dropout_prob=0.9,
        s2_cloud_timestep_prob=0.9,
        s2_cloud_block_prob=0.5,
        s2_cloud_block_min=1,
        s2_cloud_block_max=12,
        seed=123,
    )
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    for _ in range(500):
        sample = ds[0]
        s1_ok = _valid_timesteps(sample.s1).any()
        s2_ok = _valid_timesteps(sample.s2).any()
        assert s1_ok or s2_ok, "S1 and S2 both fully masked"


def _wipe_s1_s2(df, row_idx, timesteps):
    for t in timesteps:
        for col in [
            f"SAR-VV-ts{t}-20m",
            f"SAR-VH-ts{t}-20m",
            f"OPTICAL-B02-ts{t}-10m",
            f"OPTICAL-B03-ts{t}-10m",
            f"OPTICAL-B04-ts{t}-10m",
            f"OPTICAL-B05-ts{t}-20m",
            f"OPTICAL-B06-ts{t}-20m",
            f"OPTICAL-B07-ts{t}-20m",
            f"OPTICAL-B08-ts{t}-10m",
            f"OPTICAL-B8A-ts{t}-20m",
            f"OPTICAL-B11-ts{t}-20m",
            f"OPTICAL-B12-ts{t}-20m",
        ]:
            df.loc[row_idx, col] = NODATAVALUE


def test_remove_samples_without_s1_s2():
    num_timesteps = 8
    df = _make_dummy_df(num_timesteps, nrows=4)
    _wipe_s1_s2(df, 1, range(num_timesteps))
    _wipe_s1_s2(df, 3, range(num_timesteps))

    ds = WorldCerealDataset(
        df, num_timesteps=num_timesteps, remove_samples_without_s1_s2=True
    )
    assert len(ds) == 2, "Both S1+S2-empty samples should be removed"
    # remaining samples must all have some S1 or S2 data
    for i in range(len(ds)):
        sample = ds[i]
        assert (sample.s1 != NODATAVALUE).any() or (sample.s2 != NODATAVALUE).any()


def test_remove_samples_without_s1_s2_default_keeps_rows():
    num_timesteps = 8
    df = _make_dummy_df(num_timesteps, nrows=3)
    _wipe_s1_s2(df, 1, range(num_timesteps))

    ds = WorldCerealDataset(df, num_timesteps=num_timesteps)
    assert len(ds) == 3, "Without the flag no samples should be removed"


def test_window_repositioned_when_selected_window_empty():
    # Row has 18 timesteps but S1/S2 data only at ts 16-17; the default window
    # around valid_position=9 ([3, 15)) is empty. With the flag, the window
    # must shift to an admissible one containing data (still covering ts 9).
    df = _make_dummy_df(18, nrows=1)
    df["end_date"] = "2023-06-30"  # 18 monthly timesteps from start_date
    _wipe_s1_s2(df, 0, range(16))

    ds = WorldCerealDataset(df, num_timesteps=12, remove_samples_without_s1_s2=True)
    assert len(ds) == 1, "Row has reachable data and must not be removed"
    row = ds.dataframe.iloc[0].to_dict()
    positions, valid_position = ds.get_timestep_positions(row)
    assert valid_position in positions
    assert 16 in positions, f"Window {positions} should be shifted onto the data"
    sample = ds[0]
    assert (sample.s2 != NODATAVALUE).any(), "Sample should contain S2 data"

    # Without the flag the default (empty) window is kept unchanged.
    ds_default = WorldCerealDataset(df, num_timesteps=12)
    positions_default, _ = ds_default.get_timestep_positions(row)
    assert positions_default == list(range(3, 15))


def test_row_removed_when_no_admissible_window_has_data():
    # Row 1 has data only at ts 16-17, but valid_position=2 keeps every
    # admissible window within [0, 14) — the data is unreachable.
    df = _make_dummy_df(18, nrows=2)
    _wipe_s1_s2(df, 1, range(16))
    df.loc[1, "valid_position"] = 2

    ds = WorldCerealDataset(df, num_timesteps=12, remove_samples_without_s1_s2=True)
    assert len(ds) == 1, "Row with unreachable S1/S2 data should be removed"


def test_validate_rejects_double_disable():
    cfg = SensorMaskingConfig(
        enable=True,
        s1_full_dropout_prob=1.0,
        s2_cloud_timestep_prob=1.0,
    )
    with pytest.raises(ValueError, match="cannot both be 1.0"):
        cfg.validate(num_timesteps=12)


def test_s2_cloud_timestep_full_dropout():
    # Regression: s2_cloud_timestep_prob=1.0 must mask every S2 timestep,
    # independent of the cloud-block path. A previous bug guarded this path
    # with a check against the (never-populated) B1 band, making it a no-op.
    num_timesteps = 8
    df = _make_dummy_df(num_timesteps, nrows=1)
    cfg = SensorMaskingConfig(
        enable=True,
        s2_cloud_block_prob=0.0,
        s2_cloud_timestep_prob=1.0,
        seed=42,
    )
    ds = WorldCerealDataset(df, num_timesteps=num_timesteps, masking_config=cfg)
    sample = ds[0]
    assert np.all(sample.s2 == NODATAVALUE), "S2 per-timestep full dropout failed"
