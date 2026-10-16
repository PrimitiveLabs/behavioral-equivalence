from copy import deepcopy
from fractions import Fraction
import numpy as np
import pytest
from behavioral_equivalence.core import Chain, quotient
from behavioral_equivalence.integer_bounds import (
    certify_abstraction,
    state_interval,
    verify_abstraction,
    robust_enclosures,
)
from behavioral_equivalence.rational import normalized_kernel, exact_values
from behavioral_equivalence.uncertainty import IntervalChain


def test_exact_integer_abstraction_bounds_contain_independent_rational_probabilities():
    chain = Chain(
        np.array([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5], [0, 0, 1.0]]),
        ("active", "active", "complete"),
        np.array([1.0, 0, 0]),
    )
    blocks = ((0, 1), (2,))
    abstract, _ = quotient(chain, blocks)
    certificate = certify_abstraction(chain, abstract, blocks, [1, 5, 10], bits=32)
    verify_abstraction(chain, abstract, certificate)
    for h in [1, 5, 10]:
        truth = exact_values(normalized_kernel(chain), chain.labels, "complete", h)
        for s in range(chain.size):
            lower, upper = state_interval(certificate, s, "complete", h)
            assert lower <= truth[s] <= upper
    assert state_interval(certificate, 0, "complete", 1) == (Fraction(1, 4), Fraction(1, 2))
    damaged = deepcopy(certificate)
    damaged["epsilon"]["numerator_hex"] = "0"
    with pytest.raises(ValueError, match="recomputation"):
        verify_abstraction(chain, abstract, damaged)


def test_conditional_integer_bounds_and_undefined_conditioning():
    chain = Chain(
        np.array([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5], [0, 0, 1.0]]),
        ("active", "active", "complete"),
        np.array([1.0, 0, 0]),
    )
    abstract, _ = quotient(chain, ((0, 1), (2,)))
    cert = certify_abstraction(chain, abstract, ((0, 1), (2,)), [5], bits=32)
    for s in [0, 1]:
        row = normalized_kernel(chain)[s]
        future = exact_values(normalized_kernel(chain), chain.labels, "complete", 4)
        truth = sum(row[j] * future[j] for j in [0, 1]) / sum(row[j] for j in [0, 1])
        lower, upper = state_interval(cert, s, "complete_given_active_next", 5)
        assert lower <= truth <= upper
    with pytest.raises(ValueError, match="undefined"):
        state_interval(cert, 2, "complete_given_active_next", 5)


def test_integer_robust_endpoints_and_arithmetic_width():
    model = IntervalChain(
        np.array([[0.25, 0.25], [0, 1.0]]), np.array([[0.75, 0.75], [0, 1.0]]), ("active", "goal")
    )
    report = robust_enclosures(model, "goal", [5], bits=32)
    x = report["values"]["5"]
    d = 1 << 32
    minimum = 1 - Fraction(3, 4) ** 5
    maximum = 1 - Fraction(1, 4) ** 5
    assert Fraction(x["min_lower"][0], d) <= minimum <= Fraction(x["min_upper"][0], d)
    assert Fraction(x["max_lower"][0], d) <= maximum <= Fraction(x["max_upper"][0], d)
    assert x["min_upper"][0] - x["min_lower"][0] <= 5
    assert x["max_upper"][0] - x["max_lower"][0] <= 5


def test_exact_interval_feasibility_is_not_floating_row_tolerance():
    p = np.array([[0.2, 0.8], [0, 1.0]])
    model = IntervalChain(p, p, ("active", "goal"))
    with pytest.raises(ValueError, match="exactly infeasible"):
        robust_enclosures(model, "goal", [5])
