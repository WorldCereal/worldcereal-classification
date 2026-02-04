from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
import xarray as xr

from worldcereal.openeo import inference, mapping


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
    engine._export_class_probabilities = keep_probs
    engine._croptype_enabled = True
    engine._cropland_enabled = True
    engine._cropland_postprocess = inference.PostprocessOptions()
    engine._croptype_postprocess = inference.PostprocessOptions()
    engine.bundle = cast(
        inference.SeasonalModelBundle,
        SimpleNamespace(
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
        ),
    )
    return engine


def _dummy_probability_arr() -> xr.DataArray:
    return xr.DataArray(
        np.zeros((1, 1, 2, 2), dtype=np.float32),
        dims=("bands", "t", "x", "y"),
        coords={"bands": ["B2"], "t": [0], "x": [0, 1], "y": [0, 1]},
    )


def _probability_dataset_with_engine(
    keep_probs: bool = True,
) -> tuple[inference.SeasonalInferenceEngine, xr.Dataset]:
    engine = _build_probability_engine(keep_probs=keep_probs)
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
    dataset = engine._format_outputs(
        arr=arr,
        outputs=(landcover_logits, croptype_logits),
        season_ids=["tc-s1", "tc-s2"],
        enforce_cropland_gate=True,
    )

    return engine, dataset


def _probability_dataset(keep_probs: bool = True) -> xr.Dataset:
    _, dataset = _probability_dataset_with_engine(keep_probs=keep_probs)
    return dataset


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
            "postprocess": {
                "cropland": {"enabled": True, "kernel_size": 7},
                "croptype": {
                    "enabled": True,
                    "method": "smooth_probabilities",
                },
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
    assert config["enable_cropland_head"] is True
    assert config["cropland_postprocess"] == {"enabled": True, "kernel_size": 7}
    assert config["croptype_postprocess"]["method"] == "smooth_probabilities"


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


def test_extract_udf_configuration_allows_croptype_only(monkeypatch):
    def fake_presets():
        return "default", {
            "default": {
                "model": {"seasonal_model_zip": "preset.zip"},
                "runtime": {},
                "season": {},
            }
        }

    monkeypatch.setattr(inference, "_seasonal_workflow_presets", fake_presets)

    context = {"seasonal_model_zip": "preset.zip", "croptype_only": True}
    config = inference._extract_udf_configuration(context)

    assert config["enable_croptype_head"] is True
    assert config["enable_cropland_head"] is False
    assert config["enforce_cropland_gate"] is False


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
        ct = xr.DataArray(np.zeros((2, 2), dtype=np.uint16), dims=("y", "x"))
        dataset = xr.Dataset(
            {
                "cropland_classification": lc,
                "croptype_classification:S1": ct,
            }
        )
        return inference._dataset_to_multiband_array(dataset)

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
            "export_class_probabilities": False,
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
        "probability_cropland",
    ]
    assert metadata.calls[0][0] == "bands"
    assert metadata.calls[0][1] == [
        "cropland_classification",
        "probability_cropland",
    ]


def test_apply_metadata_handles_cropland_disabled(monkeypatch):
    monkeypatch.setattr(inference, "_require_openeo_runtime", lambda: None)

    def fake_extract(context):
        return {
            "seasonal_model_zip": "fake.zip",
            "cache_root": None,
            "export_class_probabilities": False,
            "enable_croptype_head": True,
            "enable_cropland_head": False,
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
        "croptype_only": True,
    }

    result = inference.apply_metadata(metadata, context)

    assert result == [
        "croptype_classification:S1",
        "croptype_probability:S1",
    ]
    assert metadata.calls[0][0] == "bands"
    assert metadata.calls[0][1] == [
        "croptype_classification:S1",
        "croptype_probability:S1",
    ]


def test_probabilities_are_flattened_and_gated():
    ds = _probability_dataset()

    assert "croptype_probability:tc-s1" in ds
    assert ds["croptype_probability:tc-s1"].dtype == np.uint8
    assert "croptype_probability:tc-s1:wheat" in ds
    assert ds["croptype_probability:tc-s1:wheat"].dims == ("y", "x")
    assert ds["croptype_probability:tc-s1:wheat"].dtype == np.uint8
    wheat_probs = ds["croptype_probability:tc-s1:wheat"].values
    valid_mask = wheat_probs != inference.NOCROP_VALUE
    assert wheat_probs[valid_mask].min() >= 0
    assert wheat_probs[valid_mask].max() <= 100
    assert ds["croptype_classification:tc-s1"].values[0, 0] == inference.NOCROP_VALUE
    assert ds["probability_crop"].dtype == np.uint8
    assert ds["probability_other"].dtype == np.uint8
    season_two_wheat = ds["croptype_probability:tc-s2:wheat"]
    assert season_two_wheat.dtype == np.uint8
    s2_vals = season_two_wheat.values
    s2_valid = s2_vals != inference.NOCROP_VALUE
    assert s2_vals[s2_valid].min() >= 0
    assert s2_vals[s2_valid].max() <= 100
    assert ds["cropland_classification"].dtype == np.uint8
    assert set(np.unique(ds["cropland_classification"].values)) <= {0, 1}
    assert ds["probability_cropland"].dtype == np.uint8
    assert ds["croptype_classification:tc-s1"].dtype == np.uint8
    cropland_probs = ds["probability_cropland"].values
    cropland_labels = ds["cropland_classification"].values.astype(bool)
    assert np.any(cropland_labels)
    assert np.any(~cropland_labels)
    assert np.all(cropland_probs[cropland_labels] > 50)
    assert np.all(cropland_probs[~cropland_labels] < 50)


def test_dataset_to_multiband_array_preserves_band_order():
    ds = _probability_dataset()
    cube = inference._dataset_to_multiband_array(ds)
    bands = cube.coords["bands"].values.tolist()

    expected_prefix = [
        "cropland_classification",
        "probability_cropland",
        "probability_crop",
        "probability_other",
        "croptype_classification:tc-s1",
        "croptype_classification:tc-s2",
        "croptype_probability:tc-s1",
        "croptype_probability:tc-s2",
    ]
    assert bands[: len(expected_prefix)] == expected_prefix
    assert bands[len(expected_prefix)] == "croptype_probability:tc-s1:wheat"
    assert bands[len(expected_prefix) + 3] == "croptype_probability:tc-s2:wheat"
    assert cube.dtype == np.uint8


def test_expected_udf_labels_match_probability_dataset_order_with_class_probs():
    engine, dataset = _probability_dataset_with_engine(keep_probs=True)

    expected_labels = inference._expected_udf_band_labels(
        ["tc-s1", "tc-s2"],
        export_class_probabilities=True,
        cropland_gate_classes=engine.bundle.cropland_gate_classes,
        croptype_classes=engine.bundle.croptype_spec.class_names,
        croptype_enabled=engine._croptype_enabled,
        cropland_enabled=engine._cropland_enabled,
    )

    assert list(dataset.data_vars) == expected_labels


def test_expected_udf_labels_match_probability_dataset_order_without_class_probs():
    engine, dataset = _probability_dataset_with_engine(keep_probs=False)

    expected_labels = inference._expected_udf_band_labels(
        ["tc-s1", "tc-s2"],
        export_class_probabilities=False,
        cropland_gate_classes=engine.bundle.cropland_gate_classes,
        croptype_classes=engine.bundle.croptype_spec.class_names,
        croptype_enabled=engine._croptype_enabled,
        cropland_enabled=engine._cropland_enabled,
    )

    assert list(dataset.data_vars) == expected_labels


def test_croptype_probabilities_use_nocrop_value_when_gated():
    ds = _probability_dataset()
    classification = ds["croptype_classification:tc-s1"].values
    wheat_probs = ds["croptype_probability:tc-s1:wheat"].values
    nocrop_mask = classification == inference.NOCROP_VALUE

    assert np.any(nocrop_mask)
    assert np.all(wheat_probs[nocrop_mask] == inference.NOCROP_VALUE)


def test_sorted_croptype_band_names_prioritizes_classification():
    unordered = [
        "croptype_probability:tc-s1:wheat",
        "croptype_probability:tc-s1",
        "croptype_classification:tc-s1",
        "croptype_raw_probability:tc-s1",
        "croptype_raw_classification:tc-s1",
    ]

    ordered = mapping._sorted_croptype_band_names(unordered)

    assert ordered[0] == "croptype_classification:tc-s1"
    assert ordered[1] == "croptype_probability:tc-s1"
    assert ordered[-1] == "croptype_raw_probability:tc-s1"
