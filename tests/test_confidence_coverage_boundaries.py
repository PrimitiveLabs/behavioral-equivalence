"""Keep confidence-set membership, event containment and Monte Carlo units distinct."""

from fractions import Fraction
import numpy as np
import pytest
from behavioral_equivalence.core import Chain
from behavioral_equivalence.confidence import certified_hoeffding, verify_hoeffding, _endpoint
from behavioral_equivalence.coverage_analysis import (
    coverage_interval,
    summarize_coverage_uncertainty,
)
from behavioral_equivalence.uncertainty import IntervalChain
from behavioral_equivalence.robust_benchmark import assess_intervals


@pytest.mark.parametrize(
    "counts,labels",
    [
        (np.array([], dtype=int), ()),
        (np.ones((1, 1), dtype=int), ("a", "b")),
        (np.array([[-1, 1], [0, 1]]), ("a", "b")),
        (np.eye(2, dtype=bool), ("a", "b")),
    ],
)
def test_certified_confidence_count_boundaries(counts, labels):
    with pytest.raises(ValueError, match="nonnegative integer"):
        certified_hoeffding(counts, labels)


@pytest.mark.parametrize(
    "alpha,bits",
    [
        (Fraction(0), 32),
        (Fraction(1), 32),
        (Fraction(-1), 32),
        (Fraction(1, 20), 7),
        (Fraction(1, 20), 49),
    ],
)
def test_certified_confidence_alpha_and_precision_limits(alpha, bits):
    with pytest.raises(ValueError, match="Rational alpha"):
        certified_hoeffding(np.eye(2, dtype=int), ("a", "b"), alpha, bits)


@pytest.mark.parametrize("bits", [8, 48])
def test_confidence_precision_endpoints_and_repeated_sample_count_cache(bits):
    counts = np.array([[100, 0], [0, 100]])
    model, certificate = certified_hoeffding(counts, ("a", "b"), bits=bits)
    assert certificate["radius_denominator"] == 1 << bits
    assert certificate["rows"][0] == certificate["rows"][1]
    assert model.upper[0, 0] == model.upper[1, 1] == 1
    assert model.lower[0, 1] == model.lower[1, 0] == 0
    verify_hoeffding(counts, model.labels, certificate)


@pytest.mark.parametrize("denominator", [-1, 0, 3])
def test_confidence_replay_rejects_non_power_of_two_denominator(denominator):
    counts = np.eye(2, dtype=int)
    _, certificate = certified_hoeffding(counts, ("a", "b"))
    certificate["radius_denominator"] = denominator
    with pytest.raises(ValueError, match="power of two"):
        verify_hoeffding(counts, ("a", "b"), certificate)


@pytest.mark.parametrize("value", [Fraction(1, 10), Fraction(1, 3), Fraction(1, 2)])
def test_confidence_endpoint_rounding_is_outward_in_both_directions(value):
    assert Fraction.from_float(_endpoint(value, False)) <= value
    assert Fraction.from_float(_endpoint(value, True)) >= value


@pytest.mark.parametrize(
    "successes,datasets,comparisons",
    [
        (-1, 10, 1),
        (11, 10, 1),
        (0, 0, 1),
        (1.5, 10, 1),
        (True, 10, 1),
        (1, 10.5, 1),
        (1, True, 1),
        (1, 10, 0),
        (1, 10, 1.5),
        (1, 10, True),
    ],
)
def test_monte_carlo_interval_requires_independent_dataset_counts(successes, datasets, comparisons):
    with pytest.raises(ValueError, match="integer counts"):
        coverage_interval(successes, datasets, 0.05, comparisons)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.inf, np.nan])
def test_monte_carlo_family_error_probability(alpha):
    with pytest.raises(ValueError, match="Alpha"):
        coverage_interval(5, 10, alpha)


def summary():
    return dict(
        status="complete",
        campaign_id="fixture",
        protocol_digest="frozen",
        coverage=[
            dict(
                method="a",
                size=32,
                samples_per_row=100,
                sampled_datasets=10,
                kernel_inclusion_count=7,
                all_probability_enclosures_contained_count=9,
            )
        ],
    )


def test_coverage_summary_rejects_unfinished_empty_and_duplicate_cells():
    for case in [dict(summary(), status="running"), dict(summary(), coverage=[])]:
        with pytest.raises(ValueError, match="Complete coverage"):
            summarize_coverage_uncertainty(case)
    duplicate = summary()
    duplicate["coverage"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_coverage_uncertainty(duplicate)


def test_coverage_summary_reports_methods_separately_with_different_membership_and_event_rates():
    result = summarize_coverage_uncertainty(summary())
    assert result["reported_intervals"] == len(result["intervals"]) == 4
    assert {r["independent_datasets"] for r in result["intervals"]} == {10}
    assert {r["successes"] for r in result["intervals"]} == {7, 9}
    assert result["family_confidence"] == 0.95
    lo0, hi0 = coverage_interval(0, 100, 0.05)
    lo1, hi1 = coverage_interval(100, 100, 0.05)
    assert lo0 == 0 and hi1 == 1 and lo1 == pytest.approx(1 - hi0)
    lo_more, hi_more = coverage_interval(50, 100, 0.05, comparisons=10)
    lo, hi = coverage_interval(50, 100, 0.05)
    assert lo_more < lo < hi < hi_more


def models():
    p = np.array([[0.75, 0, 0.25], [0, 0.25, 0.75], [0, 0, 1.0]])
    q = np.array([[0.25, 0, 0.75], [0, 0.75, 0.25], [0, 0, 1.0]])
    chain = Chain(p, ("active", "active", "complete"), np.array([1, 0, 0]))
    return chain, IntervalChain(q, q, chain.labels)


def test_nonmember_kernel_can_still_have_all_its_event_enclosures_contained():
    chain, uncertainty = models()
    result, _ = assess_intervals(chain, uncertainty, ((0, 1), (2,)), [1, 5])
    assert not result["kernel_membership"]
    assert all(r["enclosure_containment_count"] == 3 for r in result["properties"])
    assert all(
        r["proven_noncoverage_count"] == r["unresolved_count"] == 0 for r in result["properties"]
    )
    assert result["conditional_certificate_contradictions"] == 0


def test_nonmember_point_kernel_can_prove_noncoverage_without_claiming_certificate_contradiction():
    chain, uncertainty = models()
    result, _ = assess_intervals(chain, uncertainty, ((0,), (1,), (2,)), [1])
    assert not result["kernel_membership"]
    complete = next(r for r in result["properties"] if r["property"] == "complete")
    assert (
        complete["proven_noncoverage_count"] == 2 and complete["enclosure_containment_count"] == 1
    )
    assert result["conditional_certificate_contradictions"] == 0
    mismatched = IntervalChain(uncertainty.lower, uncertainty.upper, ("other",) * 3)
    with pytest.raises(ValueError, match="labels must match"):
        assess_intervals(chain, mismatched, ((0,), (1,), (2,)), [1])
