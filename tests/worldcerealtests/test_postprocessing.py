import numpy as np
import pytest

from worldcereal.openeo.inference import (
    PostprocessOptions,
    _build_postprocess_options,
    _run_postprocess,
)

CLASS_VALUE_TO_INDEX = {0: 0, 1: 1}


def test_build_postprocess_options_from_mapping_handles_strings():
    spec = {
        "enabled": "true",
        "method": "smooth_probabilities",
        "kernel_size": "7",
    }
    options = _build_postprocess_options(spec)
    assert isinstance(options, PostprocessOptions)
    assert options.enabled is True
    assert options.method == "smooth_probabilities"
    assert options.kernel_size == 7


def test_build_postprocess_options_rejects_non_integer_kernel():
    with pytest.raises(ValueError):
        _build_postprocess_options({"enabled": True, "kernel_size": "abc"})


def test_run_postprocess_no_method_returns_original_labels_and_probs():
    labels = np.array([[0, 1], [1, 0]], dtype=np.uint16)
    probabilities = np.array(
        [
            [[0.8, 0.3], [0.7, 0.2]],
            [[0.2, 0.7], [0.3, 0.8]],
        ],
        dtype=np.float32,
    )
    options = PostprocessOptions(enabled=False, method="majority_vote", kernel_size=5)

    updated_labels, updated_probabilities, updated_cube = _run_postprocess(
        labels,
        probabilities,
        class_value_to_index=CLASS_VALUE_TO_INDEX,
        options=options,
        excluded_values=(),
    )

    np.testing.assert_array_equal(updated_labels, labels)
    expected_probs = np.where(labels == 0, probabilities[0], probabilities[1])
    np.testing.assert_allclose(updated_probabilities, expected_probs)
    np.testing.assert_allclose(updated_cube, probabilities)


def test_run_postprocess_majority_vote_respects_exclusions():
    labels = np.array(
        [
            [255, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint16,
    )
    probabilities = np.array(
        [
            [[0.1, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
            [[0.9, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]],
        ],
        dtype=np.float32,
    )
    options = PostprocessOptions(enabled=True, method="majority_vote", kernel_size=3)

    updated_labels, updated_probabilities, updated_cube = _run_postprocess(
        labels,
        probabilities,
        class_value_to_index=CLASS_VALUE_TO_INDEX,
        options=options,
        excluded_values=(255,),
    )

    assert updated_labels[1, 1] == 1  # center pixel adopts neighborhood majority
    assert updated_labels[0, 0] == 255  # excluded value is preserved
    assert updated_probabilities[1, 1] == pytest.approx(0.8)
    np.testing.assert_allclose(updated_cube, probabilities)


def test_run_postprocess_smooth_probabilities_updates_labels_and_probs():
    labels = np.zeros((3, 3), dtype=np.uint16)
    class0 = np.array(
        [
            [0.2, 0.2, 0.2],
            [0.2, 0.9, 0.2],
            [0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    class1 = np.array(
        [
            [0.8, 0.8, 0.8],
            [0.8, 0.1, 0.8],
            [0.8, 0.8, 0.8],
        ],
        dtype=np.float32,
    )
    probabilities = np.stack([class0, class1])
    options = PostprocessOptions(
        enabled=True, method="smooth_probabilities", kernel_size=5
    )

    updated_labels, updated_probabilities, updated_cube = _run_postprocess(
        labels,
        probabilities,
        class_value_to_index=CLASS_VALUE_TO_INDEX,
        options=options,
        excluded_values=(),
    )

    assert updated_labels[1, 1] == 1
    expected_center_prob = ((0.1 * 3) + (0.8 * 12)) / 15
    assert updated_probabilities[1, 1] == pytest.approx(expected_center_prob, rel=1e-3)
    assert updated_cube.shape == probabilities.shape


def test_run_postprocess_unknown_method_raises():
    labels = np.zeros((2, 2), dtype=np.uint16)
    probabilities = np.ones((2, 2, 2), dtype=np.float32)
    options = PostprocessOptions(enabled=True, method="does_not_exist", kernel_size=3)

    with pytest.raises(ValueError):
        _run_postprocess(
            labels,
            probabilities,
            class_value_to_index=CLASS_VALUE_TO_INDEX,
            options=options,
            excluded_values=(),
        )
