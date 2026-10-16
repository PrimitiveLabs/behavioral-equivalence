from fractions import Fraction
from math import comb
import pytest
from behavioral_equivalence.coverage_analysis import (
    coverage_interval,
    summarize_coverage_uncertainty,
)


def test_interval_has_conservative_binomial_coverage_and_counts_datasets():
    failure = Fraction()
    for k in range(101):
        lo, hi = coverage_interval(k, 100, 0.05)
        if not lo <= 0.5 <= hi:
            failure += Fraction(comb(100, k), 2**100)
    assert failure <= Fraction(1, 20)
    summary = {
        "status": "complete",
        "campaign_id": "test",
        "protocol_digest": "frozen",
        "coverage": [
            {
                "method": method,
                "size": 32,
                "samples_per_row": m,
                "sampled_datasets": 100,
                "kernel_inclusion_count": 100,
                "all_probability_enclosures_contained_count": 100,
            }
            for method in ["a", "b"]
            for m in [100, 1000, 10000]
        ],
    }
    result = summarize_coverage_uncertainty(summary)
    assert result["reported_intervals"] == 16
    aggregate = [r for r in result["intervals"] if r["scope"].startswith("all")]
    assert len(aggregate) == 4
    assert all(r["independent_datasets"] == 300 for r in aggregate)
    assert all(r["simultaneous_ci"][1] == 1 for r in aggregate)
    with pytest.raises(ValueError, match="counts"):
        coverage_interval(101, 100, 0.05)
