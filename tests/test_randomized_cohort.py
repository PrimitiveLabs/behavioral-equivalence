import numpy as np
import pytest
from behavioral_equivalence.randomized_cohort import randomized_role_indices, exposure_environment


def test_all_random_concurrent_exposures_retained_and_unseen_roles_respected():
    args = [
        np.array([1, 1, 1, 2, 2, 2]),
        np.array([1, 1, 2, 1, 2, 3]),
        np.arange(6),
        np.array([1, 1, 0, 1, 1, 1]),
        np.array([0, 1, 3, 1, 2, 3]),
        np.array([0, 0, 0, 3, 3, 3]),
    ]
    known = randomized_role_indices(*args, "known_users")
    assert known["train"].tolist() == [0]
    assert known["validation"].tolist() == [1, 3]
    assert known["calibration"].tolist() == [4]
    assert known["test"].tolist() == [5]
    unseen = randomized_role_indices(*args, "unseen_users")
    assert unseen["validation"].size == 0
    assert unseen["test"].tolist() == [5]


def test_duration_boundaries_and_unavailable_context_are_explicit():
    duration = np.array([7000, 7001, 18000, 18001, 60000, 60001, 0, 8000], dtype=float)
    upload = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    times = np.full(8, 86400000)
    result = exposure_environment(np.arange(8), duration, upload, times)
    assert result["duration_group"].tolist() == [0, 1, 1, 2, 2, 3, 4, 4]
    assert result["environment"].tolist() == [0, 6, 11, 17, 22, 28, 34, 39]
    with pytest.raises(ValueError, match="scenario"):
        exposure_environment(np.full(8, 15), duration, upload, times)


def test_exposure_identity_and_role_validation():
    base = [
        np.array([1, 1]),
        np.array([1, 2]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([2, 3]),
        np.array([3, 3]),
    ]
    for index, values in [(2, [1, 1]), (3, [1, 2]), (4, [2, 4]), (0, [2, 1])]:
        changed = base.copy()
        changed[index] = np.array(values)
        with pytest.raises(ValueError, match="canonical"):
            randomized_role_indices(*changed, "known_users")
    changed = base.copy()
    changed[5] = np.array([2, 3])
    with pytest.raises(ValueError, match="fixed role"):
        randomized_role_indices(*changed, "known_users")
    with pytest.raises(ValueError, match="canonical"):
        randomized_role_indices(*base, "invalid")
