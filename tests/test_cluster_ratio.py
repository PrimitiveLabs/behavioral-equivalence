from fractions import Fraction

import numpy as np
import pytest

from behavioral_equivalence.cluster_ratio import cluster_ratio_bounds


def test_perfectly_dependent_within_user_counts_are_not_iid_transitions():
    counts = np.tile([[[32, 32], [0, 0]]], (1000, 1, 1))
    lo, hi, evidence = cluster_ratio_bounds(counts, maximum_per_user=64)
    assert lo[0, 0] < 0.5 < hi[0, 0]
    assert np.array_equal(lo[1], [0, 0])
    assert np.array_equal(hi[1], [1, 1])
    assert evidence["users"] == 1000
    assert evidence["coordinates"] == 6
    assert evidence["tail_inequality_verified"]
    assert evidence["human_multistep_coverage_established"] is False
    scaled = cluster_ratio_bounds(counts // 32, maximum_per_user=2)
    np.testing.assert_array_equal(lo, scaled[0])
    np.testing.assert_array_equal(hi, scaled[1])


def test_small_cohort_vacuous_and_zero_users_preserved():
    counts = np.array([[[1, 0]], [[0, 0]]])
    lo, hi, evidence = cluster_ratio_bounds(counts, maximum_per_user=1)
    np.testing.assert_array_equal(lo, [[0, 0]])
    np.testing.assert_array_equal(hi, [[1, 1]])
    assert evidence["users"] == 2


@pytest.mark.parametrize(
    "counts,cap", [([[[2]]], 1), ([[[-1]]], 1), ([[[1.0]]], 1), ([], 1), ([[[1]]], 0)]
)
def test_invalid_counts_and_caps(counts, cap):
    with pytest.raises(ValueError, match="user tables"):
        cluster_ratio_bounds(counts, maximum_per_user=cap)


def test_invalid_confidence_parameters():
    for alpha, bits in [(0.05, 32), (Fraction(1), 32), (Fraction(1, 20), 2)]:
        with pytest.raises(ValueError, match="Rational"):
            cluster_ratio_bounds([[[1]]], maximum_per_user=1, alpha=alpha, bits=bits)


def test_aggregate_interface_matches_full_tables_without_dense_user_tensor():
    from behavioral_equivalence.cluster_ratio import cluster_ratio_from_totals

    users = np.tile([[[32, 32], [0, 0]]], (1000, 1, 1))
    full = cluster_ratio_bounds(users, maximum_per_user=64)
    aggregate = cluster_ratio_from_totals(users.sum(axis=0), users=1000, maximum_per_user=64)
    np.testing.assert_array_equal(full[0], aggregate[0])
    np.testing.assert_array_equal(full[1], aggregate[1])
    assert full[2] == aggregate[2]
    for totals, count, cap in [([[1.2]], 2, 1), ([[3]], 2, 1), ([[1]], 0, 1), ([[1]], 1, 0)]:
        with pytest.raises(ValueError, match="aggregate"):
            cluster_ratio_from_totals(totals, users=count, maximum_per_user=cap)


def test_pooled_lift_contains_nominal_and_propagates_interval_uncertainty():
    from behavioral_equivalence.cluster_ratio import pooled_decision_intervals
    from behavioral_equivalence.decision_model import decision_chain
    from behavioral_equivalence.uncertainty import robust_reachability
    from behavioral_equivalence.core import reachability

    counts = np.full((2, 2, 2), 1000, dtype=np.int64)
    model, cert = pooled_decision_intervals(counts, users=8000, maximum_per_user=1)
    point = decision_chain(np.full((2, 2, 2), 0.5), np.full((2, 2), 0.5), np.array([1.0, 0.0]))
    assert np.all(model.lower <= point.transitions)
    assert np.all(model.upper >= point.transitions)
    for horizon in [1, 9, 49, 99]:
        lower, upper = robust_reachability(model, "positive", horizon)
        exact = reachability(point, "positive", horizon)
        assert np.all(lower <= exact + 1e-12)
        assert np.all(upper >= exact - 1e-12)
    assert cert["emissions"]["alpha"] == [1, 40]
    assert cert["transitions"]["alpha"] == [1, 40]
    assert not cert["teacher_emissions_used"]
    with pytest.raises(ValueError, match="counts"):
        pooled_decision_intervals(np.ones((2, 3, 2), dtype=int), users=8, maximum_per_user=1)
