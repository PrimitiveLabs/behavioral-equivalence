import numpy as np
import pytest
from behavioral_equivalence.features import bundle_history_tokens


def test_delayed_unordered_feedback_and_current_target_independence():
    timestamps = np.array([100, 200, 300, 400])
    available = np.array([350, 250, 310, 500])
    clicks = np.array([1, 0, 2, 1])
    exposures = np.array([2, 2, 2, 1])
    x = bundle_history_tokens(timestamps, available, clicks, exposures, history=3)
    assert x.tolist() == [[0, 0, 17], [0, 0, 17], [0, 0, 1], [9, 1, 16]]
    altered = clicks.copy()
    altered[2] = 0
    y = bundle_history_tokens(timestamps, available, altered, exposures, history=3)
    np.testing.assert_array_equal(x[:3], y[:3])
    assert x[3, -1] != y[3, -1]


def test_embargo_and_strict_boundary():
    times = np.array([100, 200, 300])
    available = np.array([200, 250, 350])
    clicks = np.ones(3)
    exposures = np.ones(3)
    x = bundle_history_tokens(times, available, clicks, exposures, extra_embargo_ms=100)
    assert x[:, -1].tolist() == [17, 17, 17]
    with pytest.raises(ValueError):
        bundle_history_tokens(times[::-1], available, clicks, exposures)
