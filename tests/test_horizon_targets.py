import numpy as np
import pytest
from behavioral_equivalence.horizon_targets import (
    horizon_event_targets,
    origin_iid_event_probabilities,
)
from behavioral_equivalence.decision_model import decision_chain
from behavioral_equivalence.nested_design import human_event_suite
from behavioral_equivalence.property_bounds import event_predictions


def test_missing_horizons_and_event_order_match_model_properties():
    y, events = horizon_event_targets(
        [1, 5],
        np.array([[True, True], [True, False]]),
        np.array([[1, 3], [0, -1]]),
        np.array([[1, 0], [0, -1]]),
    )
    assert events == human_event_suite((1, 5))
    assert y[0].tolist() == [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    assert y[1].tolist() == [0, -1, 0, -1, 1, -1, 0, -1, 0, -1]


def test_origin_only_iid_reference_matches_full_single_state_decision_dynamics():
    h = (1, 5, 10)
    for p in [0.0, 0.37, 1.0]:
        chain = decision_chain(np.ones((1, 2, 1)), np.array([[1 - p, p]]), np.ones(1))
        expected = event_predictions(chain, human_event_suite(h))[:, 0]
        assert origin_iid_event_probabilities([p], h)[0] == pytest.approx(expected, abs=1e-14)


def test_censored_values_and_inconsistent_counts_are_rejected():
    with pytest.raises(ValueError):
        horizon_event_targets([1], np.array([[False]]), np.array([[0]]), np.array([[-1]]))
    with pytest.raises(ValueError):
        horizon_event_targets([1], np.array([[True]]), np.array([[0]]), np.array([[1]]))
    with pytest.raises(ValueError):
        origin_iid_event_probabilities([1.1], [1])
