import numpy as np
import pytest
from behavioral_equivalence.multistep_cohort import (
    select_test_origins,
    observed_horizon_cohorts,
    recorded_horizon_targets,
)


def test_origins_keep_future_censoring_and_never_cross_users_or_role_gaps():
    users = np.array([1, 1, 1, 1, 2, 2])
    times = np.array([10, 20, 30, 40, 10, 20])
    roles = np.array([3, -1, 3, 3, 3, 3])
    origins = select_test_origins(users, times, roles)
    assert origins.tolist() == [0, 2, 3, 4, 5]
    cohorts = observed_horizon_cohorts(users, times, roles, origins, [1, 2, 3])
    assert cohorts[0]["observed_origins"] == 5
    assert cohorts[1]["observed_mask"].tolist() == [False, True, False, True, False]
    assert cohorts[2]["observed_origins"] == 0
    y = np.array([1, 0, 1, 0, 0, 1])
    target = recorded_horizon_targets(y, cohorts[1])
    assert target["positive_count"].tolist() == [-1, 1, -1, 1, -1]
    assert target["last_positive"].tolist() == [-1, 0, -1, 1, -1]
    assert cohorts[1]["selected_origins"] == 5
    assert cohorts[1]["censored_origins"] == 3


def test_per_user_sampling_reproducible_independent_of_other_users():
    users = np.repeat([3, 4], 20)
    times = np.tile(np.arange(20), 2)
    roles = np.full(40, 3)
    a = select_test_origins(users, times, roles, per_user_cap=4)
    b = select_test_origins(users[:20], times[:20], roles[:20], per_user_cap=4)
    assert a[:4].tolist() == b.tolist()
    assert len(a) == 8
    assert np.array_equal(a, select_test_origins(users, times, roles, per_user_cap=4))


def test_empty_cohort_and_boundary_validation():
    u = np.array([], dtype=int)
    assert select_test_origins(u, u, u).size == 0
    c = observed_horizon_cohorts(u, u, u, u, [1])[0]
    assert recorded_horizon_targets(u, c)["positive_count"].size == 0
    for h in ([], [0], [True], [1, 1]):
        with pytest.raises(ValueError, match="horizons"):
            observed_horizon_cohorts(u, u, u, u, h)
    with pytest.raises(ValueError, match="cap"):
        select_test_origins(u, u, u, per_user_cap=True)
    with pytest.raises(ValueError, match="timestamp"):
        select_test_origins(np.array([1, 1]), np.array([1, 1]), np.array([3, 3]))
    with pytest.raises(ValueError, match="Sorted aligned"):
        select_test_origins(np.array([2, 1]), np.array([1, 2]), np.array([3, 3]))
    with pytest.raises(ValueError, match="origin"):
        observed_horizon_cohorts(np.array([1]), np.array([1]), np.array([3]), np.array([1]), [1])
    with pytest.raises(ValueError, match="test role"):
        observed_horizon_cohorts(np.array([1]), np.array([1]), np.array([2]), np.array([0]), [1])
    with pytest.raises(ValueError, match="Binary"):
        recorded_horizon_targets(np.array([2]), c)
