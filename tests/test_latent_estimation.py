import numpy as np
import pytest
from behavioral_equivalence.latent_estimation import (
    adjacent_transition_counts,
    point_transitions,
    equal_user_initial_mass,
)


def test_actual_adjacent_outcome_conditioned_counts_and_nominal_unsupported_rows():
    states = np.array([0, 1, 0, 1])
    targets = np.array([1, 0, 1, 0])
    users = np.array([1, 1, 1, 2])
    bundles = np.array([3, 4, 5, 6])
    counts = adjacent_transition_counts(
        states, targets, users, bundles, np.array([0, 1]), np.array([1, 2]), 2
    )
    assert counts[0, 1, 1] == 1 and counts[1, 0, 0] == 1 and counts.sum() == 2
    p = point_transitions(counts)
    np.testing.assert_allclose(p[0, 1], [0.25, 0.75])
    np.testing.assert_allclose(p[0, 0], [0.5, 0.5])
    np.testing.assert_allclose(p.sum(axis=2), 1)
    with pytest.raises(ValueError, match="consecutive"):
        adjacent_transition_counts(states, targets, users, bundles, np.array([0]), np.array([2]), 2)
    with pytest.raises(ValueError, match="same user"):
        adjacent_transition_counts(states, targets, users, bundles, np.array([2]), np.array([3]), 2)


def test_initial_mixture_weights_users_equally_despite_unequal_origin_counts():
    mass, n = equal_user_initial_mass(np.array([0, 1, 1, 1]), np.array([1, 2, 2, 2]), 2)
    np.testing.assert_allclose(mass / n, [0.5, 0.5])
    assert n == 2
    mass, n = equal_user_initial_mass(np.array([], dtype=int), np.array([], dtype=int), 2)
    np.testing.assert_array_equal(mass, [0.0, 0.0])
    assert n == 0
