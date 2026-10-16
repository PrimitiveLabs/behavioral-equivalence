"""Inference alignment and probability-domain boundaries without human data."""

from fractions import Fraction

import numpy as np
import pytest

from behavioral_equivalence.case_inference import (
    available_history_lengths,
    conditional_response_forecasts,
    forecast_indices,
    normalized_positive_emissions,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ids", [4, 4]),
        ("ids", [-1, 5]),
        ("users", [-1, 2]),
        ("roles", [4, 3]),
        ("user_roles", [-1, 3]),
        ("context", [5, 4]),
        ("users", [1]),
        ("ids", [4.0, 5.0]),
    ],
)
def test_forecast_alignment_rejects_invalid_inputs(field, value):
    a = dict(ids=[4, 5], users=[1, 2], roles=[1, 3], user_roles=[1, 3], context=[4, 5])
    a[field] = value
    with pytest.raises(ValueError, match="Aligned immutable"):
        forecast_indices(*[np.asarray(v) for v in a.values()], "known_users", "test")


@pytest.mark.parametrize("protocol", ["known_users", "unseen_users"])
@pytest.mark.parametrize("split", ["validation", "test"])
def test_empty_forecast_cohort_and_large_case_ids(protocol, split):
    a = np.array([], dtype=np.int64)
    assert forecast_indices(a, a, a, a, a, protocol, split).size == 0
    ids = np.array([2**63 + 1, 2**63 + 2], dtype=np.uint64)
    roles = np.array([1, 3])
    expected = [0] if split == "validation" else [1]
    np.testing.assert_array_equal(
        forecast_indices(ids, ids, roles, roles, ids, protocol, split), expected
    )


@pytest.mark.parametrize("tokens", [[], [[0]], [[17, 17]], [[17, 1]], [[-1, 1]], [[1.0]]])
def test_invalid_available_history_rows(tokens):
    with pytest.raises(ValueError):
        available_history_lengths(tokens)


@pytest.mark.parametrize("rows", [[], [[1, 0, 0]], [[-0.1, 1.1]], [[0.2, 0.2]], [[0, np.inf]]])
def test_invalid_normalized_emissions(rows):
    with pytest.raises(ValueError):
        normalized_positive_emissions(rows)


def test_near_stochastic_emissions_use_exact_serialized_ratio():
    rows = np.array([[0.7, np.nextafter(0.3, 1)], [1, 0], [0, 1]])
    expected = [
        float(Fraction(float(y)) / (Fraction(float(x)) + Fraction(float(y)))) for x, y in rows
    ]
    np.testing.assert_array_equal(normalized_positive_emissions(rows), expected)


@pytest.mark.parametrize("field", ["coarse", "mapping", "env", "probability"])
def test_response_forecast_rejects_out_of_domain_lookup(field):
    states, env, coarse, mapping = [np.array([0]) for _ in range(4)]
    p = np.full((1, 75), 0.5)
    if field == "coarse":
        coarse[0] = 1
    elif field == "mapping":
        mapping[0] = 1
    elif field == "env":
        env[0] = -1
    else:
        p[0, 0] = np.nan
    with pytest.raises(ValueError, match="Aligned valid"):
        conditional_response_forecasts(states, env, coarse, mapping, p)
