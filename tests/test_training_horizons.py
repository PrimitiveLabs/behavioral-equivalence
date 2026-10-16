import numpy as np
import pytest
from behavioral_equivalence.training_horizons import training_role_cohort
from behavioral_equivalence.multistep_cohort import recorded_horizon_targets


def test_role_origins_never_skip_exclusions_or_cross_users_and_roles():
    users = np.array([1] * 6 + [2] * 3)
    times = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3])
    roles = np.array([0, 0, -1, 0, 1, 3, 0, 0, 1])
    origins, cohorts = training_role_cohort(users, times, roles, 0, [1, 2, 3])
    assert origins.tolist() == [0, 1, 3, 6, 7]
    assert cohorts[1]["observed_mask"].tolist() == [True, False, False, True, False]
    assert not cohorts[2]["observed_mask"].any()
    validation, _ = training_role_cohort(users, times, roles, 1, [1])
    assert validation.tolist() == [4, 8]


def test_fitting_targets_ignore_heldout_outcomes_and_sampling_is_outcome_blind():
    users = np.ones(6, dtype=int)
    times = np.arange(6)
    roles = np.array([0, 0, 0, 1, 1, 3])
    origins, cohorts = training_role_cohort(users, times, roles, 0, [1, 2], per_user_cap=2, seed=11)
    y = np.array([1, 0, 1, 0, 0, 0])
    before = [recorded_horizon_targets(y, c) for c in cohorts]
    y[roles != 0] = 1
    after = [recorded_horizon_targets(y, c) for c in cohorts]
    assert np.array_equal(origins, training_role_cohort(users, times, roles, 0, [1, 2], 2, 11)[0])
    for a, b in zip(before, after, strict=True):
        assert np.array_equal(a["positive_count"], b["positive_count"])
        assert np.array_equal(a["last_positive"], b["last_positive"])


def test_test_and_calibration_roles_cannot_become_fit_origins():
    for role in [2, 3, True]:
        with pytest.raises(ValueError):
            training_role_cohort(np.array([1]), np.array([1]), np.array([role]), role, [1])
