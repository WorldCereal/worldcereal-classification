from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from worldcereal.openeo import inference


class DummyCube:
    """Minimal stand-in for openEO's XarrayDataCube"""

    def __init__(self, array: xr.DataArray):
        self._array = array

    def get_array(self) -> xr.DataArray:
        return self._array


def _make_udf_data(array: xr.DataArray, context: dict | None = None, epsg: int = 32631):
    cube = DummyCube(array)
    return SimpleNamespace(
        datacube_list=[cube], user_context=context or {}, proj={"EPSG": epsg}
    )


def _create_sample_array() -> xr.DataArray:
    bands = ["B2", "COP-DEM"]
    data = np.arange(16, dtype=np.float32).reshape(len(bands), 2, 2, 2)
    return xr.DataArray(
        data,
        dims=("bands", "t", "y", "x"),
        coords={"bands": bands, "t": [0, 1], "y": [0, 1], "x": [0, 1]},
    )


def _build_probability_engine(
    keep_probs: bool = True,
) -> inference.SeasonalInferenceEngine:
    engine = inference.SeasonalInferenceEngine.__new__(
        inference.SeasonalInferenceEngine
    )
    engine._keep_class_probabilities = keep_probs
    engine._croptype_enabled = True
    engine.bundle = SimpleNamespace(
        landcover_spec=inference.HeadSpec(
            task="landcover",
            class_names=["nocrop", "crop"],
            cropland_classes=["crop"],
            gating_enabled=False,
        ),
        croptype_spec=inference.HeadSpec(
            task="croptype",
            class_names=["wheat", "maize", "other"],
            cropland_classes=[],
            gating_enabled=False,
        ),
        cropland_gate_classes=["crop"],
    )
    return engine


def _dummy_probability_arr() -> xr.DataArray:
    return xr.DataArray(
        np.zeros((1, 1, 2, 2), dtype=np.float32),
        dims=("bands", "t", "x", "y"),
        coords={"bands": ["B2"], "t": [0], "x": [0, 1], "y": [0, 1]},
    )


def _probability_dataset() -> xr.Dataset:
    engine = _build_probability_engine()
    arr = _dummy_probability_arr()
    landcover_logits = torch.tensor(
        [[6.0, 1.0], [1.0, 6.0], [1.0, 6.0], [1.0, 6.0]],
        dtype=torch.float32,
    )
    croptype_logits = torch.tensor(
        [
            [[3.0, 1.0, 0.5], [2.0, 1.0, 0.5]],
            [[1.0, 3.0, 0.5], [1.0, 3.0, 0.5]],
            [[1.0, 3.0, 0.5], [1.0, 3.0, 0.5]],
            [[1.0, 3.0, 0.5], [1.0, 3.0, 0.5]],
        ],
        dtype=torch.float32,
    )
    return engine._format_outputs(
        arr=arr,
        outputs=(landcover_logits, croptype_logits),
        season_ids=["tc-s1", "tc-s2"],
        enforce_cropland_gate=True,
    )


def test_extract_udf_configuration_applies_context_overrides(tmp_path):
    context = {
        "workflow_config": {
            "model": {
                "seasonal_model_zip": "user.zip",
                "landcover_head_zip": "lc.pt",
                "croptype_head_zip": "ct.pt",
                "enable_croptype_head": True,
            },
            "runtime": {
                "cache_root": tmp_path.as_posix(),
                "batch_size": "128",
                "device": "cuda:0",
            },
            "season": {
                "season_ids": ["S-user"],
                "season_windows": {"S-user": ("2021-01-01", "2021-06-01")},
                "season_masks": np.ones((2, 3), dtype=bool).tolist(),
                "composite_frequency": "dekad",
                "enforce_cropland_gate": False,
            },
        },
    }

    config = inference._extract_udf_configuration(context)

    assert config["seasonal_model_zip"] == "user.zip"
    assert config["landcover_head_zip"] == "lc.pt"
    assert config["croptype_head_zip"] == "ct.pt"
    assert config["batch_size"] == 128
    assert str(config["device"]) == "cuda:0"
    assert config["season_ids"] == ["S-user"]
    assert config["season_windows"] == {"S-user": ("2021-01-01", "2021-06-01")}
    assert np.array_equal(config["season_masks"], np.ones((2, 3), dtype=bool))
    assert config["enforce_cropland_gate"] is False
    assert config["season_composite_frequency"] == "dekad"
    assert config["cache_root"] == Path(tmp_path)
    assert config["enable_croptype_head"] is True


def test_extract_udf_configuration_requires_model_zip(monkeypatch):
    def fake_presets():
        return "default", {"default": {"model": {}, "runtime": {}, "season": {}}}

    monkeypatch.setattr(inference, "_seasonal_workflow_presets", fake_presets)

    with pytest.raises(ValueError, match="seasonal_model_zip"):
        inference._extract_udf_configuration({})


def test_extract_udf_configuration_disables_croptype_head(monkeypatch):
    def fake_presets():
        return "default", {
            "default": {
                "model": {"seasonal_model_zip": "preset.zip"},
                "runtime": {},
                "season": {},
            }
        }

    monkeypatch.setattr(inference, "_seasonal_workflow_presets", fake_presets)

    context = {"seasonal_model_zip": "preset.zip", "disable_croptype_head": True}
    config = inference._extract_udf_configuration(context)

    assert config["enable_croptype_head"] is False


def test_build_masks_from_windows_handles_multiple_seasons():
    timestamps = np.array(
        ["2022-01-01", "2022-01-20", "2022-01-31"], dtype="datetime64[D]"
    )
    windows = {
        "S1": [("2022-01-01", "2022-01-05")],
        "S2": [("2022-01-15", "2022-01-31")],
    }
    normalized = inference._normalize_season_windows_input(windows)
    masks = inference._build_masks_from_windows(
        timestamps, ["S1", "S2"], normalized, batch_size=2, composite_frequency=None
    )

    assert masks.shape == (2, 2, 3)
    assert masks[0, 0, 0]
    assert not masks[0, 0, 1]
    assert not masks[0, 0, 2]
    assert masks[0, 1, 1]
    assert masks[0, 1, 2]
    assert np.array_equal(masks[0], masks[1])


def test_normalize_provided_masks_broadcasts_single_batch():
    masks = np.ones((1, 2, 4), dtype=bool)
    normalized = inference._normalize_provided_masks(
        masks, batch_size=3, num_timesteps=4
    )
    assert normalized.shape == (3, 2, 4)
    assert normalized.dtype == bool

    with pytest.raises(ValueError):
        inference._normalize_provided_masks(
            np.ones((2, 2), dtype=bool), batch_size=3, num_timesteps=4
        )


def test_apply_udf_data_wraps_engine_output(monkeypatch):
    arr = _create_sample_array()
    udf = _make_udf_data(arr, {"seasonal_model_zip": "fake.zip"})

    monkeypatch.setattr(inference, "_require_openeo_runtime", lambda: None)
    monkeypatch.setattr(
        inference, "_extract_udf_configuration", lambda context: {"config": "value"}
    )
    monkeypatch.setattr(inference, "_infer_udf_epsg", lambda udf_data: 32631)
    monkeypatch.setattr(inference, "XarrayDataCube", DummyCube)

    captured = {}

    def fake_run(arr, epsg, **config):
        captured["dims"] = arr.dims
        captured["config"] = config
        assert epsg == 32631
        lc = xr.DataArray(np.zeros((2, 2), dtype=np.uint16), dims=("y", "x"))
        ct = xr.DataArray(
            np.zeros((1, 2, 2), dtype=np.uint16),
            dims=("season", "y", "x"),
            coords={"season": ["S1"]},
        )
        return xr.Dataset(
            {"cropland_classification": lc, "croptype_classification": ct}
        )

    monkeypatch.setattr(inference, "run_seasonal_workflow", fake_run)

    result = inference.apply_udf_data(udf)

    assert captured["dims"] == ("bands", "t", "y", "x")
    assert captured["config"] == {"config": "value"}
    out_cube = result.datacube_list[0]
    stacked = out_cube.get_array()
    assert stacked.dims == ("bands", "y", "x")
    assert list(stacked.bands.values) == [
        "cropland_classification",
        "croptype_classification:S1",
    ]


def test_apply_metadata_omits_croptype_labels_when_disabled(monkeypatch):
    monkeypatch.setattr(inference, "_require_openeo_runtime", lambda: None)

    def fake_extract(context):
        return {
            "seasonal_model_zip": "fake.zip",
            "cache_root": None,
            "keep_class_probabilities": False,
            "enable_croptype_head": False,
        }

    monkeypatch.setattr(inference, "_extract_udf_configuration", fake_extract)

    class DummyMetadata:
        def __init__(self):
            self.calls = []

        def rename_labels(self, dimension, target):
            self.calls.append((dimension, target))
            return target

    metadata = DummyMetadata()
    context = {
        "seasonal_model_zip": "fake.zip",
        "season_ids": ["S1"],
        "disable_croptype_head": True,
    }

    result = inference.apply_metadata(metadata, context)

    assert result == [
        "cropland_classification",
        "cropland_probability",
    ]
    assert metadata.calls[0][0] == "bands"
    assert metadata.calls[0][1] == [
        "cropland_classification",
        "cropland_probability",
    ]


def test_flatten_spatial_dataset_creates_2d_variables():
    da = xr.DataArray(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        dims=("season", "y", "x"),
        coords={"season": ["tc-s1", "tc-s2"], "y": [0, 1], "x": [0, 1]},
    )
    ds = xr.Dataset({"croptype_confidence": da})
    flat = inference._flatten_spatial_dataset(ds)

    assert "croptype_confidence:tc-s1" in flat
    assert "croptype_confidence:tc-s2" in flat
    assert flat["croptype_confidence:tc-s1"].dims == ("y", "x")


def test_probabilities_are_flattened_and_gated():
    ds = _probability_dataset()

    assert "croptype_probability:tc-s1:wheat" in ds
    assert ds["croptype_probability:tc-s1:wheat"].dims == ("y", "x")
    assert np.isclose(
        ds["croptype_probability:tc-s1:wheat"].values[0, 0],
        0.0,
        atol=1e-5,
    )
    assert ds["croptype_classification:tc-s1"].values[0, 0] == inference.NOCROP_VALUE
    assert ds["landcover_probabilities:crop"].dtype == np.uint8
    assert ds["landcover_probabilities:other"].dtype == np.uint8
    season_two_wheat = ds["croptype_probability:tc-s2:wheat"]
    assert season_two_wheat.dtype == np.uint8
    assert season_two_wheat.values.min() >= 0
    assert season_two_wheat.values.max() <= 100
    assert ds["cropland_classification"].dtype == np.uint8
    assert set(np.unique(ds["cropland_classification"].values)) <= {0, 1}
    assert ds["cropland_probability"].dtype == np.uint8
    assert ds["croptype_confidence:tc-s1"].dtype == np.uint8
    assert ds["croptype_classification:tc-s1"].dtype == np.uint8


def test_dataset_to_multiband_array_handles_flattened_dataset():
    ds = _probability_dataset()
    cube = inference._dataset_to_multiband_array(ds)
    bands = cube.coords["bands"].values.tolist()

    assert "croptype_probability:tc-s1:wheat" in bands
    assert "croptype_confidence:tc-s2" in bands
    assert "landcover_probabilities:other" in bands
    assert cube.dtype == np.uint8
