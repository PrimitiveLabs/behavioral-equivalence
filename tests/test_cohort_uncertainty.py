import numpy as np
import pytest
from behavioral_equivalence.cohort_uncertainty import cohort_interval_diagnostic


def diagnostic(y, lo, hi, users):
    return cohort_interval_diagnostic(
        y, lo, hi, users, minimum_users=2, minimum_cases=2, replications=137
    )


def test_interval_diagnostic_uses_cohort_probability_not_binary_containment():
    # No binary observation is inside [0.2,0.8], but every user's mean is0.5.
    r = diagnostic([0, 1, 0, 1], [0.2] * 4, [0.8] * 4, [1, 1, 2, 2])
    assert r["observed_equal_user_rate"] == 0.5
    assert r["finding"] == "mean_compatibility_supported_under_bootstrap_assumptions"
    assert r["confidence_interval_coverage"] is None
    assert r["mean_interval_width"] == pytest.approx(0.6)


def test_bad_cohort_mean_is_detected_and_zero_width_identity_is_unresolved():
    r = diagnostic([1, 1], [0.1, 0.1], [0.3, 0.3], [1, 2])
    assert r["finding"] == "mean_incompatibility_supported_under_bootstrap_assumptions"
    assert r["upper_minus_observed"]["nominal_ci975"] == pytest.approx([-0.7, -0.7])
    r = diagnostic([0, 1, 0, 1], [0.5] * 4, [0.5] * 4, [1, 1, 2, 2])
    assert r["finding"] == "unresolved"


def test_users_are_equally_weighted_despite_different_case_counts():
    r = diagnostic([0, 0, 0, 1], [0] * 4, [1] * 4, [1, 1, 1, 2])
    assert r["observed_equal_user_rate"] == 0.5
    assert r["mean_interval_width"] == 1


def test_unsupported_cohort_retains_counts_without_invented_inference():
    r = cohort_interval_diagnostic([1, 0], [0, 0], [1, 1], [1, 2])
    assert r["status"] == "unsupported"
    assert r["users"] == r["cases"] == 2
    assert "finding" not in r
    empty = diagnostic(np.array([]), np.array([]), np.array([]), np.array([], dtype=int))
    assert empty["status"] == "unsupported"


@pytest.mark.parametrize("lo,hi", [([-0.1, 0], [1, 1]), ([0, 0], [1, 1.1]), ([0.5, 0], [0.4, 1])])
def test_invalid_bounds_rejected(lo, hi):
    with pytest.raises(ValueError):
        diagnostic([0, 1], lo, hi, [1, 2])
