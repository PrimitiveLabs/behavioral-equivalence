import numpy as np
import pytest
from behavioral_equivalence.adjacent_calibration import representative_indices, adjacent_origins


def test_one_uniform_representative_per_bundle_ignores_outcomes_and_preserves_ties():
    groups = np.array([0, 0, 1, 2, 2, 2])
    chosen = representative_indices(groups, 7)
    np.testing.assert_array_equal(groups[chosen], np.arange(3))
    np.testing.assert_array_equal(representative_indices(groups, 7), chosen)
    draws = np.array([representative_indices(groups, seed)[2] for seed in range(300)])
    assert set(draws) == {3, 4, 5}
    assert all(60 < np.sum(draws == i) < 140 for i in [3, 4, 5])
    with pytest.raises(ValueError):
        representative_indices(np.array([0, 2]), 7)


def test_adjacent_pairs_do_not_jump_over_exclusions_or_user_boundaries():
    users = np.array([1, 1, 1, 1, 1, 2, 2, 2])
    times = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    roles = np.array([2, 2, -1, 2, 2, 2, 3, 3])
    np.testing.assert_array_equal(adjacent_origins(users, times, roles), [0, 3])
    np.testing.assert_array_equal(adjacent_origins(users, times, np.zeros(8, dtype=int)), [])
    with pytest.raises(ValueError):
        adjacent_origins(users, np.zeros(8, dtype=int), roles)


def test_user_cap_and_random_choice_are_stable_when_other_users_are_added():
    users = np.repeat([1, 2], 10)
    times = np.tile(np.arange(10), 2)
    roles = np.full(20, 2)
    selected = adjacent_origins(users, times, roles, 3)
    assert len(selected) == 6
    np.testing.assert_array_equal(
        selected[:3], adjacent_origins(users[:10], times[:10], roles[:10], 3)
    )
    assert all(users[i] == users[i + 1] and times[i + 1] == times[i] + 1 for i in selected)


def test_selected_queries_keep_unsampled_intermediate_history_and_exclude_current_response():
    from behavioral_equivalence.adjacent_calibration import query_history_tokens

    users = np.ones(5, dtype=int)
    times = np.arange(5) * 10
    available = times + 1
    clicks = np.array([0, 1, 0, 1, 0])
    exposures = np.ones(5, dtype=int)
    query = np.array([0, 3])
    tokens = query_history_tokens(users, times, available, clicks, exposures, query, 0)
    assert tokens[0, -1] == 17
    np.testing.assert_array_equal(tokens[1, -3:], [1, 16, 1])
    changed = clicks.copy()
    changed[3:] = 1 - changed[3:]
    np.testing.assert_array_equal(
        query_history_tokens(users, times, available, changed, exposures, query, 0), tokens
    )
    embargoed = query_history_tokens(users, times, available, clicks, exposures, query, 100)
    assert np.all(embargoed[:, -1] == 17)
