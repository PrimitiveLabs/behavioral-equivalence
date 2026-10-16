import numpy as np
import pytest

from behavioral_equivalence.cluster_ratio import pooled_decision_intervals
from behavioral_equivalence.pooled_forecasts import pooled_emission_forecasts


def test_nominal_and_interval_share_pooled_emission_estimand():
    counts = np.array([[[1, 0], [3, 0]], [[0, 0], [0, 0]]])
    model, _ = pooled_decision_intervals(counts, users=100, maximum_per_user=1)
    p, lo, hi = pooled_emission_forecasts(counts, model.lower, model.upper, np.array([0, 1, 0]))
    np.testing.assert_array_equal(p, [0.75, 0.5, 0.75])
    assert lo[1] == 0 and hi[1] == 1
    np.testing.assert_array_equal(lo[[0, 2]], model.lower[[0, 0], [2, 2]])
    np.testing.assert_array_equal(hi[[0, 2]], model.upper[[0, 0], [2, 2]])


def test_invalid_state_or_bounds_rejected_and_empty_cohort_supported():
    counts = np.ones((1, 2, 1), dtype=int)
    model, _ = pooled_decision_intervals(counts, users=2, maximum_per_user=1)
    assert (
        len(pooled_emission_forecasts(counts, model.lower, model.upper, np.array([], dtype=int))[0])
        == 0
    )
    for states in [np.array([1]), np.array([-1]), np.array([0.0])]:
        with pytest.raises(ValueError, match="origin states"):
            pooled_emission_forecasts(counts, model.lower, model.upper, states)
    with pytest.raises(ValueError):
        pooled_emission_forecasts(counts, -model.upper, model.upper, np.array([0]))
