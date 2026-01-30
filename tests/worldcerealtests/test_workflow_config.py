from worldcereal.openeo.workflow_config import WorldCerealWorkflowConfigBuilder


def test_workflow_config_builder_serializes_sections():
    cfg = (
        WorldCerealWorkflowConfigBuilder()
        .export_class_probabilities(True)
        .disable_croptype_head()
        .disable_cropland_head()
        .batch_size(512)
        .device("cuda:0")
        .season_ids(["tc-s1", "tc-s2"])
        .composite_frequency("dekad")
        .build()
    )

    cfg_dict = cfg.to_dict()

    assert cfg_dict["season"]["export_class_probabilities"] is True
    assert cfg_dict["season"]["composite_frequency"] == "dekad"
    assert cfg_dict["model"]["enable_croptype_head"] is False
    assert cfg_dict["model"]["enable_cropland_head"] is False
    assert cfg_dict["runtime"]["batch_size"] == 512
    assert cfg_dict["runtime"]["device"] == "cuda:0"
    assert cfg_dict["season"]["season_ids"] == ["tc-s1", "tc-s2"]


def test_workflow_config_builder_supports_postprocess_sections():
    cfg = (
        WorldCerealWorkflowConfigBuilder()
        .cropland_postprocess(enabled=True, method="majority_vote", kernel_size=7)
        .croptype_postprocess(
            enabled=True,
            method="smooth_probabilities",
        )
        .build()
    )

    cfg_dict = cfg.to_dict()

    assert cfg_dict["postprocess"]["cropland"]["enabled"] is True
    assert cfg_dict["postprocess"]["cropland"]["kernel_size"] == 7
    assert cfg_dict["postprocess"]["croptype"]["method"] == "smooth_probabilities"
