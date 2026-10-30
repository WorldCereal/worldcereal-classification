import importlib
import json
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from worldcereal.train import backbone
from worldcereal.utils import models


@pytest.fixture
def seasonal_zip(tmp_path, monkeypatch):
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"custom encoder weights")
    fingerprint = backbone.checkpoint_fingerprint(encoder)
    monkeypatch.setattr(models, "DEFAULT_CACHE_ROOT", tmp_path / "cache")
    backbone.resolve_seasonal_encoder.cache_clear()

    def create(*, encoder_entry="encoder.pt", declared_fingerprint=fingerprint):
        manifest = {
            "artifacts": {"checkpoints": {"full": "full.pt"}},
            "backbone": {},
        }
        if encoder_entry is not None:
            manifest["artifacts"]["checkpoints"]["encoder_only"] = encoder_entry
        if declared_fingerprint is not None:
            manifest["backbone"]["fingerprint"] = declared_fingerprint
        path = tmp_path / "seasonal.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("config.json", json.dumps(manifest))
            archive.writestr("full.pt", b"full model weights")
            archive.write(encoder, "encoder.pt")
        return path, fingerprint

    yield create
    backbone.resolve_seasonal_encoder.cache_clear()


@pytest.mark.parametrize("remote", [False, True])
def test_resolve_encoder_from_seasonal_zip(seasonal_zip, monkeypatch, remote):
    path, expected_fingerprint = seasonal_zip()
    source = str(path)
    if remote:
        source = "https://example.test/seasonal.zip"
        download = Mock(return_value=path.read_bytes())
        monkeypatch.setattr(models, "_get_url", download)

    checkpoint, fingerprint = backbone.resolve_seasonal_encoder(source)

    assert Path(checkpoint).read_bytes() == b"custom encoder weights"
    assert fingerprint == expected_fingerprint
    assert backbone.resolve_seasonal_encoder(source) == (checkpoint, fingerprint)
    if remote:
        download.assert_called_once_with(source)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"encoder_entry": None}, ValueError, "encoder_only"),
        ({"encoder_entry": "missing.pt"}, FileNotFoundError, "missing.pt"),
        ({"declared_fingerprint": "incorrect"}, ValueError, "fingerprint mismatch"),
    ],
)
def test_invalid_seasonal_encoder_raises(seasonal_zip, options, error, message):
    path, _ = seasonal_zip(**options)
    with pytest.raises(error, match=message):
        backbone.resolve_seasonal_encoder(str(path))


def test_encoder_without_declared_fingerprint_is_hashed(seasonal_zip):
    path, expected_fingerprint = seasonal_zip(declared_fingerprint=None)
    _, fingerprint = backbone.resolve_seasonal_encoder(str(path))
    assert fingerprint == expected_fingerprint


@pytest.mark.parametrize("source", ["", " ", "\n"])
def test_empty_seasonal_source_never_loads_default(monkeypatch, source):
    load = Mock()
    monkeypatch.setattr(backbone, "load_model_artifact", load)
    with pytest.raises(ValueError, match="must not be empty"):
        backbone.resolve_seasonal_encoder(source)
    load.assert_not_called()


def test_unspecified_seasonal_source_still_uses_default(seasonal_zip, monkeypatch):
    path, expected_fingerprint = seasonal_zip()
    load = Mock(wraps=backbone.load_model_artifact)
    monkeypatch.setattr(backbone, "load_model_artifact", load)
    monkeypatch.setattr(backbone, "DEFAULT_SEASONAL_MODEL_URL", str(path))
    _, fingerprint = backbone.resolve_seasonal_encoder()
    load.assert_called_once_with(str(path))
    assert fingerprint == expected_fingerprint


@pytest.fixture
def app_module(monkeypatch):
    pytest.importorskip("ipywidgets")
    notebooks = Path(__file__).resolve().parents[2] / "notebooks"
    monkeypatch.syspath_prepend(str(notebooks))
    module = importlib.import_module("notebook_utils.classification_app")
    app_class = module.WorldCerealClassificationApp
    for name in (
        "_build_tab0_workflow",
        "_build_tab1_retrieve_reference_data",
        "_build_tab2_select_training_data",
        "_build_tab3_season_alignment",
        "_build_tab4_crop_selection",
        "_build_tab5_compute_embeddings",
        "_build_tab6_train_model",
        "_build_tab7_deploy_model",
        "_build_tab8_generate_map",
        "_build_tab9_visualize_results",
    ):
        monkeypatch.setattr(app_class, name, lambda self: module.widgets.VBox())
    monkeypatch.setattr(app_class, "_update_nav_buttons", lambda self: None)
    monkeypatch.setattr(module, "display", Mock())
    return module


@pytest.mark.parametrize("entrypoint", ["constructor", "run"])
def test_app_accepts_single_zip(app_module, seasonal_zip, entrypoint):
    path, expected_fingerprint = seasonal_zip()
    app_class = app_module.WorldCerealClassificationApp
    create = app_class.run if entrypoint == "run" else app_class

    app = create(seasonal_model_path=path)

    assert app.seasonal_model_path == str(path)
    assert Path(app.presto_model_path).read_bytes() == b"custom encoder weights"
    assert app.presto_model_fingerprint == expected_fingerprint


def test_app_accepts_single_remote_zip(app_module, seasonal_zip, monkeypatch):
    path, fingerprint = seasonal_zip()
    source = "https://example.test/seasonal.zip"
    download = Mock(return_value=path.read_bytes())
    monkeypatch.setattr(models, "_get_url", download)

    app = app_module.WorldCerealClassificationApp.run(seasonal_model_path=source)

    assert app.seasonal_model_path == source
    assert app.presto_model_fingerprint == fingerprint
    assert Path(app.presto_model_path).is_file()
    download.assert_called_once_with(source)


@pytest.mark.parametrize("corrupt_zip", [False, True])
def test_remote_zip_failure_never_downloads_default(
    app_module, seasonal_zip, monkeypatch, corrupt_zip
):
    seasonal_zip()
    source = "https://example.test/unavailable.zip"
    if corrupt_zip:
        download = Mock(return_value=b"not a zip archive")
        error = zipfile.BadZipFile
    else:
        download = Mock(side_effect=OSError("Download failed"))
        error = OSError
    monkeypatch.setattr(models, "_get_url", download)

    with pytest.raises(error):
        app_module.WorldCerealClassificationApp(seasonal_model_path=source)

    download.assert_called_once_with(source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seasonal_model_path": ""},
        {"seasonal_model_path": " "},
        {"seasonal_model_path": 123},
        {"seasonal_model_path": {}},
        {"seasonal_model_path": {"seasonal_model_path": "seasonal.zip"}},
    ],
)
def test_invalid_custom_input_never_resolves_default(app_module, monkeypatch, kwargs):
    resolve = Mock()
    monkeypatch.setattr(app_module, "resolve_seasonal_encoder", resolve)
    with pytest.raises(ValueError):
        app_module.WorldCerealClassificationApp(**kwargs)
    resolve.assert_not_called()


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"encoder_entry": None}, ValueError),
        ({"encoder_entry": "missing.pt"}, FileNotFoundError),
        ({"declared_fingerprint": "incorrect"}, ValueError),
    ],
)
def test_app_rejects_invalid_zip(app_module, seasonal_zip, options, error):
    path, _ = seasonal_zip(**options)
    with pytest.raises(error):
        app_module.WorldCerealClassificationApp(seasonal_model_path=path)


def test_app_rejects_missing_zip(app_module, tmp_path):
    with pytest.raises(FileNotFoundError):
        app_module.WorldCerealClassificationApp(
            seasonal_model_path=tmp_path / "nonexistent.zip"
        )


@pytest.mark.parametrize("entrypoint", ["constructor", "run"])
def test_app_no_longer_accepts_legacy_dictionary(app_module, monkeypatch, entrypoint):
    resolve = Mock()
    monkeypatch.setattr(app_module, "resolve_seasonal_encoder", resolve)
    app_class = app_module.WorldCerealClassificationApp
    create = app_class.run if entrypoint == "run" else app_class

    with pytest.raises(TypeError, match="presto_model_package"):
        create(presto_model_package={"seasonal_model_path": "seasonal.zip"})
    with pytest.raises(TypeError):
        create({"seasonal_model_path": "seasonal.zip"})
    resolve.assert_not_called()


def test_default_app_does_not_resolve_custom_encoder(app_module, monkeypatch):
    resolve = Mock()
    monkeypatch.setattr(app_module, "resolve_seasonal_encoder", resolve)
    app = app_module.WorldCerealClassificationApp()
    assert app.seasonal_model_path is None
    assert app.presto_model_path is None
    assert app.presto_model_fingerprint is None
    resolve.assert_not_called()


def test_training_callbacks_use_resolved_custom_encoder(
    app_module, seasonal_zip, monkeypatch, tmp_path
):
    path, fingerprint = seasonal_zip()
    app = app_module.WorldCerealClassificationApp(seasonal_model_path=path)
    checkpoint = app.presto_model_path
    df = pd.DataFrame({"downstream_class": [1], "presto_ft_0": [0.5]})
    app.season_id = "custom-s1"
    app.season_window = app_module.TemporalContext("2025-03-01", "2025-08-31")
    app.tab4_df = df
    app.tab5_widgets["embeddings_output"] = app_module.widgets.Output()
    app.tab6_widgets["progress_output"] = app_module.widgets.Output()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "_get_tab4_working_df", lambda: df)
    monkeypatch.setattr(app, "_get_aoi_filename_part", lambda: "test")
    monkeypatch.setattr(app, "_get_season_filename_part", lambda: "custom-s1")
    monkeypatch.setattr(app, "_update_tab6_state", Mock())
    compute = Mock(return_value=df)
    monkeypatch.setattr(app_module, "compute_seasonal_presto_embeddings", compute)
    train = Mock()
    monkeypatch.setattr(app_module, "train_seasonal_torch_head", train)

    app._on_tab5_compute_embeddings()
    app._on_train_click(None)

    assert app.embeddings_df_path.is_file()
    assert compute.call_args.kwargs["custom_presto_url"] == checkpoint
    assert train.call_args.kwargs["presto_model_path"] == checkpoint
    assert train.call_args.kwargs["presto_model_fingerprint"] == fingerprint
    assert app.seasonal_model_path == str(path)


def test_run_metadata_does_not_replace_training_encoder(
    app_module, seasonal_zip, tmp_path
):
    path, _ = seasonal_zip()
    app = app_module.WorldCerealClassificationApp(seasonal_model_path=path)
    checkpoint = app.presto_model_path
    fingerprint = app.presto_model_fingerprint
    (tmp_path / "model_metadata.json").write_text(
        json.dumps(
            {
                "presto_model": "different-encoder.pt",
                "seasonal_model": "https://example.test/previous-run.zip",
                "product_type": "croptype",
            }
        )
    )

    assert app._load_model_metadata(tmp_path)
    assert app.tab8_seasonal_model_url == "https://example.test/previous-run.zip"
    assert app.seasonal_model_path == str(path)
    assert app.presto_model_path == checkpoint
    assert app.presto_model_fingerprint == fingerprint
