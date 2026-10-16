"""Boundary checks for outcome-labeled decision model lifting."""

import numpy as np
import pytest
from behavioral_equivalence.decision_model import (
    decision_chain,
    decision_horizon,
    decision_partition,
)


@pytest.mark.parametrize("decisions", [-1, True, False, 1.5])
def test_decision_horizon_rejects_invalid_counts(decisions):
    with pytest.raises(ValueError, match="nonnegative integer"):
        decision_horizon(decisions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_latent", np.array([])),
        ("emissions", np.ones((2, 3))),
        ("transitions", np.ones((2, 2, 3))),
        ("initial_latent", np.array([np.nan, 0])),
        ("initial_latent", np.array([1, 1])),
        ("emissions", np.array([[np.nan, 0], [0.5, 0.5]])),
        ("emissions", np.array([[0.5, 0.6], [0.5, 0.5]])),
        ("transitions", np.array([[[0.5, 0.5], [0.5, 0.5]], [[-0.5, 1.5], [0.5, 0.5]]])),
    ],
)
def test_decision_lift_rejects_shape_and_invalid_stochastic_inputs(field, value):
    args = dict(
        transitions=np.full((2, 2, 2), 0.5),
        emissions=np.full((2, 2), 0.5),
        initial_latent=np.array([1.0, 0]),
    )
    args[field] = value
    with pytest.raises(ValueError):
        decision_chain(**args)


def test_decision_lift_preserves_stage_labels_initial_law_and_outcome_conditioned_transitions():
    transitions = np.array([[[1, 0], [0, 1]], [[0.25, 0.75], [0.5, 0.5]]])
    emissions = np.array([[0.75, 0.25], [0.5, 0.5]])
    initial = np.array([0.25, 0.75])
    model = decision_chain(transitions, emissions, initial)
    assert model.labels == ("pending", "negative", "positive") * 2
    np.testing.assert_array_equal(model.initial, [0.25, 0, 0, 0.75, 0, 0])
    for state in range(2):
        np.testing.assert_array_equal(
            model.transitions[3 * state, 3 * state + 1 : 3 * state + 3], emissions[state]
        )
        for outcome in range(2):
            np.testing.assert_array_equal(
                model.transitions[3 * state + 1 + outcome, ::3], transitions[state, outcome]
            )
    np.testing.assert_allclose(model.transitions.sum(axis=1), 1)
    assert decision_partition(((0, 1),)) == ((0, 3), (1, 4), (2, 5))
    assert [decision_horizon(h) for h in [0, 1, 2, 50]] == [0, 1, 3, 99]
