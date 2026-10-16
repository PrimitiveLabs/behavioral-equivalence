"""Causal-history and support-diagnostic boundary regressions."""

import numpy as np
import pytest
from behavioral_equivalence.calendar_splits import EvaluationCalendar
from behavioral_equivalence.calendar_support import available_history_counts, support_by_role
from behavioral_equivalence.features import bundle_history_tokens


@pytest.mark.parametrize(
    "change",
    [
        dict(users=np.array([[1]])),
        dict(times=np.array([1, 2])),
        dict(availability=np.array([1, 2])),
        dict(times=np.array([1.0])),
        dict(availability=np.array([1.0])),
        dict(availability=np.array([0])),
        dict(embargo_ms=-1),
        dict(embargo_ms=True),
        dict(embargo_ms=0.5),
    ],
)
def test_history_support_rejects_alignment_types_timing_and_embargo(change):
    args = dict(
        users=np.array([1]),
        times=np.array([1], dtype=np.int64),
        availability=np.array([1], dtype=np.int64),
        embargo_ms=0,
    )
    args.update(change)
    with pytest.raises(ValueError, match="Aligned"):
        available_history_counts(**args)


@pytest.mark.parametrize("users,times", [([2, 1], [1, 2]), ([1, 1], [1, 1])])
def test_history_support_rejects_user_order_and_duplicate_timestamps(users, times):
    with pytest.raises(ValueError, match="sorted"):
        available_history_counts(np.array(users), np.array(times), np.array(times), 0)


@pytest.mark.parametrize(
    "exposures", [np.array([1, 1]), np.array([1.0]), np.array([0]), np.array([-1])]
)
def test_role_support_rejects_exposure_alignment_type_and_nonpositive_values(exposures):
    with pytest.raises(ValueError, match="Positive integer exposure"):
        support_by_role(
            EvaluationCalendar(100, 200, 300),
            np.array([1]),
            np.array([1]),
            np.array([1]),
            exposures,
            np.array(["train"]),
            "known_users",
            0,
        )


def test_empty_role_support_preserves_zero_counts_and_undefined_times():
    empty = np.array([], dtype=np.int64)
    result = support_by_role(
        EvaluationCalendar(100, 200, 300),
        empty,
        empty,
        empty,
        empty,
        np.array([], dtype=str),
        "known_users",
        0,
    )
    assert result["excluded_bundles"] == result["excluded_cases"] == 0
    for role in result["roles"]:
        assert role["bundles"] == role["cases"] == role["users"] == 0
        assert role["first_timestamp_ms"] is role["last_timestamp_ms"] is None
        assert role["latest_label_available_plus_embargo_ms"] is None
        assert role["cases_by_available_history_0_1to7_8to31_32plus"] == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "change",
    [
        dict(available=np.array([1])),
        dict(clicks=np.array([0])),
        dict(exposures=np.array([1])),
        dict(history=0),
        dict(extra_embargo_ms=-1),
        dict(timestamps=np.array([2, 1])),
        dict(available=np.array([0, 2])),
        dict(exposures=np.array([0, 1])),
        dict(clicks=np.array([-1, 0])),
        dict(clicks=np.array([2, 0])),
    ],
)
def test_history_feature_explicit_validation_gates(change):
    args = dict(
        timestamps=np.array([1, 2]),
        available=np.array([1, 2]),
        clicks=np.array([0, 1]),
        exposures=np.array([1, 1]),
    )
    args.update(change)
    with pytest.raises(ValueError):
        bundle_history_tokens(**args)


def test_empty_and_truncated_feature_history_remains_causal_without_input_mutation():
    empty = np.array([], dtype=np.int64)
    assert bundle_history_tokens(empty, empty, empty, empty, history=2).shape == (0, 2)
    times = np.arange(1, 6, dtype=np.int64)
    available = times.copy()
    clicks = np.array([0, 1, 0, 1, 0])
    exposure = np.ones(5, dtype=int)
    result = bundle_history_tokens(times, available, clicks, exposure, history=2)
    assert result.tolist() == [[0, 17], [0, 1], [1, 16], [16, 1], [1, 16]]
    np.testing.assert_array_equal(available, times)
    np.testing.assert_array_equal(clicks, [0, 1, 0, 1, 0])
