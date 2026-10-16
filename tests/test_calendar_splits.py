import pytest
import numpy as np
from behavioral_equivalence.calendar_splits import (
    EvaluationCalendar,
    CONFIRMATORY_CANDIDATE,
    utc_ms,
    calendar_roles,
    ROLE_NAMES,
)


def test_future_training_records_from_other_users_are_not_training_cases():
    c = CONFIRMATORY_CANDIDATE
    assert c.role(utc_ms("2022-04-20"), "known_users", "test", utc_ms("2022-04-21"), 0) == "train"
    assert c.role(utc_ms("2022-05-06"), "known_users", "train", utc_ms("2022-05-07"), 0) == "test"
    assert c.role(c.train_end, "known_users", "train", c.train_end, 0) == "validation"
    assert c.role(c.validation_end, "known_users", "train", c.validation_end, 0) == "calibration"
    assert c.role(c.calibration_end, "known_users", "train", c.calibration_end, 0) == "test"


def test_unseen_user_roles_require_both_identity_and_prospective_time_isolation():
    c = CONFIRMATORY_CANDIDATE
    assert c.role(utc_ms("2022-04-20"), "unseen_users", "test", utc_ms("2022-04-20"), 0) is None
    assert c.role(utc_ms("2022-05-06"), "unseen_users", "train", utc_ms("2022-05-06"), 0) is None
    assert c.role(utc_ms("2022-05-06"), "unseen_users", "test", utc_ms("2022-05-07"), 0) == "test"


def test_training_selection_excludes_feedback_not_available_at_fit_cutoff():
    c = CONFIRMATORY_CANDIDATE
    assert c.role(c.train_end - 10, "known_users", "train", c.train_end - 1, 0) == "train"
    assert c.role(c.train_end - 10, "known_users", "train", c.train_end + 1, 0) is None
    assert c.role(c.train_end - 10, "known_users", "train", c.train_end - 1, 2) is None
    with pytest.raises(ValueError, match="availability"):
        c.role(10, "known_users", "train", 9, 0)
    with pytest.raises(ValueError, match="ordered"):
        EvaluationCalendar(3, 2, 1)


def test_batch_roles_match_scalar_at_boundaries_and_randomized_times():
    c = EvaluationCalendar(100, 200, 300)
    rng = np.random.default_rng(44)
    times = np.r_[np.array([99, 100, 199, 200, 299, 300]), rng.integers(0, 400, 1000)]
    availability = times + rng.integers(0, 80, len(times))
    users = rng.choice(ROLE_NAMES, len(times))
    for protocol in ["known_users", "unseen_users"]:
        for embargo in [0, 10, 100]:
            expected = [
                c.role(int(t), protocol, str(u), int(a), embargo)
                for t, a, u in zip(times, availability, users)
            ]
            assert calendar_roles(c, times, availability, users, protocol, embargo).tolist() == [
                -1 if r is None else ROLE_NAMES.index(r) for r in expected
            ]


def test_batch_empty_and_invalid_inputs():
    c = EvaluationCalendar(100, 200, 300)
    empty = np.array([], dtype=np.int64)
    assert len(calendar_roles(c, empty, empty, np.array([], dtype=str), "known_users", 0)) == 0
    with pytest.raises(ValueError, match="Aligned"):
        calendar_roles(c, np.array([3]), np.array([2]), np.array(["train"]), "known_users", 0)
    with pytest.raises(ValueError, match="Aligned"):
        calendar_roles(c, np.array([3.0]), np.array([3]), np.array(["train"]), "known_users", 0)
    for protocol, embargo in [("unknown", 0), ("known_users", -1), ("known_users", True)]:
        with pytest.raises(ValueError, match="Known protocol"):
            calendar_roles(c, empty, empty, np.array([], dtype=str), protocol, embargo)


def test_batch_embargo_does_not_overflow_when_availability_is_int64_max():
    maximum = np.iinfo(np.int64).max
    c = EvaluationCalendar(maximum - 20, maximum - 10, maximum - 1)
    result = calendar_roles(
        c,
        np.array([maximum - 21, maximum], dtype=np.int64),
        np.array([maximum, maximum], dtype=np.int64),
        np.array(["train", "test"]),
        "known_users",
        100,
    )
    assert result.tolist() == [-1, 3]
