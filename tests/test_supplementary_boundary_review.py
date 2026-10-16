"""Categorical supplementary boundaries and accepted integer-state compatibility."""

from fractions import Fraction
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "supplementary_review_" + name, ROOT / "tools" / ("supplementary_" + name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.uint64, np.int64])
def test_endpoint_counts_accept_all_valid_state_integer_widths(dtype):
    m = load("counts")
    y = np.array([[1, 0, 0, 0, 0, 3], [0, 1, 0, 0, 0, 0], [-1, 0, 1, 0, 1, -1]])
    expected = m.endpoint_counts(
        {2: np.array([0, 1, 0])}, y, np.array([1, 1, 2]), np.array([10, 11, 12])
    )
    actual = m.endpoint_counts(
        {2: np.array([0, 1, 0], dtype=dtype)}, y, np.array([1, 1, 2]), np.array([10, 11, 12])
    )
    for key in expected[0]:
        np.testing.assert_array_equal(actual[0][key], expected[0][key])
    if dtype == np.uint64:
        frozen = json.loads(
            (
                ROOT / "data/synthetic/fixtures/release/supplementary-integer-states-v1.json"
            ).read_text()
        )
        scope = {}
        exec(compile(frozen["source_module"], "original-project-tool", "exec"), scope)
        with pytest.raises(TypeError):
            scope["endpoint_counts"](
                {2: np.array([0, 1, 0], dtype=dtype)},
                y,
                np.array([1, 1, 2]),
                np.array([10, 11, 12]),
            )


@pytest.mark.parametrize("blocks", [(), ((0, 0),), ((0,),), ((0,), (1,), ()), ((0, 2),)])
def test_endpoint_envelope_rejects_bad_partition(blocks):
    with pytest.raises(ValueError):
        load("intervals").uniform_endpoint_envelope(np.zeros((2, 2)), np.ones((2, 2)), blocks)


def test_non_dyadic_endpoint_average_is_outward_and_inputs_unchanged():
    m = load("intervals")
    p = np.array([[0.125, 0.875], [0.25, 0.75], [0.5, 0.5]])
    before = p.copy()
    result = m.uniform_endpoint_envelope(p, p, ((0, 1, 2),))
    for j in range(2):
        exact = sum(Fraction(float(v)) for v in p[:, j]) / 3
        assert (
            Fraction(float(result["abstract_lower"][0, j]))
            <= exact
            <= Fraction(float(result["abstract_upper"][0, j]))
        )
    np.testing.assert_array_equal(p, before)


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_role_sampling_rejects_invalid_cap(maximum):
    with pytest.raises(ValueError):
        load("outcomes").fixed_role_sample(np.array([2]), maximum)


def test_categorical_merge_rejects_empty_and_mixed_class_blocks():
    m = load("metrics")
    with pytest.raises(ValueError, match="At least one"):
        m.merge_categorical_blocks([])
    a = m.categorical_block(np.array([1]), np.array([0]), np.array([[0.5, 0.5]]))
    b = m.categorical_block(np.array([2]), np.array([0]), np.array([[0.25] * 4]))
    with pytest.raises(ValueError, match="class counts"):
        m.merge_categorical_blocks([a, b])
