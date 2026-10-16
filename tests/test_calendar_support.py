import numpy as np
import pytest
from behavioral_equivalence.calendar_support import available_history_counts, support_by_role
from behavioral_equivalence.calendar_splits import EvaluationCalendar


def test_history_support_matches_independent_user_and_timestamp_enumeration():
    users = np.array([1, 1, 1, 1, 2, 2, 2])
    times = np.array([1, 4, 5, 10, 1, 4, 8], dtype=np.int64)
    available = np.array([5, 4, 30, 10, 2, 5, 9], dtype=np.int64)
    for embargo in [0, 1, 20]:
        expected = [
            sum(u == user and a + embargo < t for u, a in zip(users, available))
            for user, t in zip(users, times)
        ]
        assert available_history_counts(users, times, available, embargo).tolist() == expected


def test_role_support_counts_cases_users_and_feedback_cutoff_without_targets():
    users = np.array([1, 1, 1, 1, 2, 2])
    times = np.array([10, 90, 150, 350, 20, 360], dtype=np.int64)
    availability = np.array([10, 99, 190, 400, 20, 360], dtype=np.int64)
    exposures = np.array([2, 3, 4, 5, 6, 7])
    user_roles = np.array(["train"] * 4 + ["test"] * 2)
    result = support_by_role(
        EvaluationCalendar(100, 200, 300),
        users,
        times,
        availability,
        exposures,
        user_roles,
        "known_users",
        2,
    )
    assert [r["cases"] for r in result["roles"]] == [8, 4, 0, 12]
    assert result["excluded_cases"] == 3
    assert result["roles"][3]["users"] == 2
    assert result["roles"][0]["cases_by_available_history_0_1to7_8to31_32plus"] == [8, 0, 0, 0]
    assert result["roles"][1]["latest_label_available_plus_embargo_ms"] == 192
    unseen = support_by_role(
        EvaluationCalendar(100, 200, 300),
        users,
        times,
        availability,
        exposures,
        user_roles,
        "unseen_users",
        2,
    )
    assert [r["cases"] for r in unseen["roles"]] == [2, 0, 0, 7]
    assert sum(r["cases"] for r in unseen["roles"]) + unseen["excluded_cases"] == exposures.sum()


def test_invalid_history_order_and_empty_support():
    empty = np.array([], dtype=np.int64)
    assert available_history_counts(empty, empty, empty, 0).tolist() == []
    with pytest.raises(ValueError, match="sorted"):
        available_history_counts(np.array([1, 1]), np.array([2, 1]), np.array([2, 2]), 0)
    with pytest.raises(ValueError, match="int64"):
        available_history_counts(
            np.array([1]), np.array([0]), np.array([np.iinfo(np.int64).max]), 1
        )
    with pytest.raises(ValueError, match="Positive"):
        support_by_role(
            EvaluationCalendar(1, 2, 3),
            empty,
            empty,
            empty,
            np.array([0]),
            np.array([]),
            "known_users",
            0,
        )
