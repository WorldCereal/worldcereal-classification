from worldcereal.openeo.workflow_config import WorldCerealWorkflowConfigBuilder


def test_workflow_config_builder_serializes_sections():
    cfg = (
        WorldCerealWorkflowConfigBuilder()
        .keep_class_probabilities(True)
        .disable_croptype_head()
        .batch_size(512)
        .device("cuda:0")
        .season_ids(["tc-s1", "tc-s2"])
        .composite_frequency("dekad")
        .build()
    )

    cfg_dict = cfg.to_dict()

    assert cfg_dict["season"]["keep_class_probabilities"] is True
    assert cfg_dict["season"]["composite_frequency"] == "dekad"
    assert cfg_dict["model"]["enable_croptype_head"] is False
    assert cfg_dict["runtime"]["batch_size"] == 512
    assert cfg_dict["runtime"]["device"] == "cuda:0"
    assert cfg_dict["season"]["season_ids"] == ["tc-s1", "tc-s2"]
