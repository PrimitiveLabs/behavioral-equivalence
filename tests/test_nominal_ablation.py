from fractions import Fraction
import copy
import numpy as np
import pytest
from behavioral_equivalence.nominal_ablation import (
    empirical_point_intervals,
    verify_empirical_point,
)
from behavioral_equivalence.core import Chain
from behavioral_equivalence.robust_benchmark import assess_intervals


def test_exact_empirical_rows_and_uniform_missing_rows_are_enclosed():
    counts = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]])
    labels = ("active", "complete", "complete")
    for closure in (True, False):
        model, certificate = empirical_point_intervals(counts, labels, closure)
        verify_empirical_point(counts, labels, certificate)
        assert Fraction.from_float(model.lower[0, 0]) <= Fraction(1, 3)
        assert Fraction.from_float(model.upper[0, 0]) >= Fraction(1, 3)
        assert model.lower[0, 0] < model.upper[0, 0]
        assert certificate["zero_sample_rows"] == [1, 2]
        assert certificate["statistical_coverage_claim"] == "none"
        if closure:
            assert model.lower[1].tolist() == [0, 0.5, 0.5]
        else:
            assert model.lower[1, 0] > 0
        changed = copy.deepcopy(certificate)
        changed["lower"][0][0] = 0
        with pytest.raises(ValueError, match="differs"):
            verify_empirical_point(counts, labels, changed)


def test_control_can_fail_generating_probability_coverage():
    chain = Chain(
        np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1]]),
        ("active", "complete", "disengage"),
        np.array([1.0, 0, 0]),
    )
    counts = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    model, _ = empirical_point_intervals(counts, chain.labels, True)
    report, _ = assess_intervals(chain, model, ((0,), (1,), (2,)), [1])
    assert not report["kernel_membership"]
    assert any(x["proven_noncoverage_count"] > 0 for x in report["properties"])


def test_invalid_counts_and_closure_contradiction():
    for counts, labels in (
        (np.ones((2, 2)), ("a", "b")),
        (np.array([[-1]]), ("a",)),
        (np.empty((0, 0), dtype=int), ()),
    ):
        with pytest.raises(ValueError, match="counts"):
            empirical_point_intervals(counts, labels, False)
    with pytest.raises(ValueError, match="closure"):
        empirical_point_intervals(np.array([[0, 1], [0, 0]]), ("complete", "active"), True)
