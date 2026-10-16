import importlib.util
from pathlib import Path
import itertools
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "supplementary_intervals", Path(__file__).parents[1] / "tools/supplementary_intervals.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_uniform_box_envelope_contains_every_enumerated_feasible_binary_row():
    lower = np.array([[0.0, 0.5], [0.25, 0.25]])
    upper = np.array([[0.5, 1.0], [0.75, 0.75]])
    result = module.uniform_endpoint_envelope(lower, upper, ((0, 1),))
    for a, b in itertools.product([0.0, 0.25, 0.5], [0.25, 0.5, 0.75]):
        parent = np.array([[a, 1 - a], [b, 1 - b]])
        quotient = parent.mean(axis=0)
        assert np.all(quotient >= result["abstract_lower"][0]) and np.all(
            quotient <= result["abstract_upper"][0]
        )
        assert np.max(np.abs(parent - quotient)) <= result["uniform_delta"]
        assert np.all(parent >= result["original_lower"]) and np.all(
            parent <= result["original_upper"]
        )


def test_point_identity_has_zero_error_and_vacuous_family_remains_vacuous():
    p = np.array([[0.25, 0.75], [0.25, 0.75]])
    exact = module.uniform_endpoint_envelope(p, p, ((0, 1),))
    assert exact["uniform_delta"] == 0 and np.array_equal(exact["original_lower"], p)
    vacuous = module.uniform_endpoint_envelope(np.zeros((2, 4)), np.ones((2, 4)), ((0, 1),))
    assert vacuous["uniform_delta"] == 1
    assert np.all(vacuous["original_lower"] == 0) and np.all(vacuous["original_upper"] == 1)


@pytest.mark.parametrize("change", ["simplex", "order", "duplicate", "missing"])
def test_invalid_interval_or_partition_rejected(change):
    lo = np.zeros((2, 2))
    hi = np.ones((2, 2))
    blocks = ((0, 1),)
    if change == "simplex":
        lo[:] = 0.75
    elif change == "order":
        lo[0, 0] = 1
        hi[0, 0] = 0
    elif change == "duplicate":
        blocks = ((0, 0),)
    else:
        blocks = ((0,),)
    with pytest.raises(ValueError):
        module.uniform_endpoint_envelope(lo, hi, blocks)
