import numpy as np
import pytest
from behavioral_equivalence.case_inference import (
    forecast_indices,
    available_history_lengths,
    normalized_positive_emissions,
)
from behavioral_equivalence.decision_model import decision_chain
from behavioral_equivalence.learned_abstraction import next_positive_probabilities


def test_split_selection_keeps_case_order_and_user_separation():
    ids = np.array([10, 20, 30, 40])
    users = np.array([1, 2, 3, 4])
    roles = np.array([1, 3, 3, -1])
    ur = np.array([0, 3, 2, 3])
    assert np.array_equal(
        forecast_indices(ids, users, roles, ur, ids, "known_users", "test"), [1, 2]
    )
    assert np.array_equal(forecast_indices(ids, users, roles, ur, ids, "unseen_users", "test"), [1])
    assert np.array_equal(
        forecast_indices(ids, users, roles, ur, ids, "known_users", "validation"), [0]
    )
    with pytest.raises(ValueError):
        forecast_indices(ids, users, roles, ur, ids[::-1], "known_users", "test")
    with pytest.raises(ValueError):
        forecast_indices(ids, users, roles, ur, ids, "known_users", "calibration")


def test_empty_history_marker_is_not_a_feedback_bundle():
    tokens = np.array([[0, 0, 0, 17], [0, 0, 0, 1], [0, 0, 4, 16], [1, 2, 3, 4]], dtype=np.uint8)
    assert np.array_equal(available_history_lengths(tokens), [0, 1, 2, 4])
    assert len(available_history_lengths(np.empty((0, 32), dtype=np.uint8))) == 0
    for bad in [[[0, 17, 0, 1]], [[0, 0, 1, 17]], [[0, 0, 0, 0]], [[0, 1, 0, 2]], [[0, 0, 0, 18]]]:
        with pytest.raises(ValueError):
            available_history_lengths(np.array(bad, dtype=np.int64))


def test_parent_emissions_match_saved_integer_normalized_identity_forecasts():
    emissions = np.array([[0.9, 0.1], [0.7, 0.3], [1.0, 0.0], [0.0, 1.0]])
    transitions = np.tile(np.eye(4)[:, None, :], (1, 2, 1))
    parent = decision_chain(transitions, emissions, np.full(4, 0.25))
    expected = next_positive_probabilities(parent, ((0,), (1,), (2,), (3,)))
    assert np.array_equal(normalized_positive_emissions(emissions), expected)
    with pytest.raises(ValueError):
        normalized_positive_emissions([[0, 0]])
    with pytest.raises(ValueError):
        normalized_positive_emissions([[float("nan"), 1]])


def test_response_forecasts_apply_each_nonidentity_map_once():
    from behavioral_equivalence.case_inference import conditional_response_forecasts

    emissions = np.tile([[0.2], [0.8]], (1, 75))
    fine = np.array([0, 1, 2, 3])
    env = np.array([0, 1, 2, 3])
    coarse = np.array([2, 0, 1, 0])
    mapping = np.array([1, 0, 1])
    assert np.array_equal(
        conditional_response_forecasts(fine, env, coarse, mapping, emissions), [0.8, 0.8, 0.2, 0.8]
    )
    assert len(conditional_response_forecasts(fine[:0], env[:0], coarse, mapping, emissions)) == 0
    with pytest.raises(ValueError):
        conditional_response_forecasts(np.array([4]), env[:1], coarse, mapping, emissions)
    with pytest.raises(ValueError):
        conditional_response_forecasts(fine, np.array([75, 0, 0, 0]), coarse, mapping, emissions)
