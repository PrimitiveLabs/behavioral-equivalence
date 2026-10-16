import numpy as np
from behavioral_equivalence.core import Chain
from behavioral_equivalence.uncertainty import IntervalChain
from behavioral_equivalence.robust_benchmark import assess_intervals


def test_uniform_interval_benchmark_checks_correct_probability_estimands():
    p = np.array([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5], [0, 0, 1.0]])
    chain = Chain(p, ("active", "active", "complete"), np.array([1.0, 0, 0]))
    lower = np.array([[0, 0, 0.25], [0, 0, 0.25], [0, 0, 1.0]])
    upper = np.array([[0.75, 0.75, 0.75], [0.75, 0.75, 0.75], [0, 0, 1.0]])
    result, evidence = assess_intervals(
        chain, IntervalChain(lower, upper, chain.labels), ((0, 1), (2,)), [1, 5]
    )
    assert result["kernel_membership"]
    assert len(result["properties"]) == 6
    assert all(r["enclosure_containment_count"] == chain.size for r in result["properties"])
    assert all(r["proven_noncoverage_count"] == 0 for r in result["properties"])
    assert "uniform_abstraction" in evidence
