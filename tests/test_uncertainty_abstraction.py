from copy import deepcopy
from fractions import Fraction
import numpy as np
import pytest
from behavioral_equivalence.uncertainty import IntervalChain
from behavioral_equivalence.uncertainty_abstraction import abstract_envelope, verify_envelope
from behavioral_equivalence.integer_bounds import robust_enclosures
from behavioral_equivalence.rational import exact_values


def test_uniform_envelope_covers_different_concrete_origins_and_all_kernels():
    lower = np.array([[0, 0, 0.25], [0, 0, 0.5], [0, 0, 1.0]])
    upper = np.array([[0.75, 0.75, 0.5], [0.5, 0.5, 0.75], [0, 0, 1.0]])
    source = IntervalChain(lower, upper, ("active", "active", "goal"))
    abstract, cert = abstract_envelope(source, ((0, 1), (2,)))
    verify_envelope(source, cert)
    assert np.array_equal(abstract.lower, [[0.25, 0.25], [0, 1.0]])
    assert np.array_equal(abstract.upper, [[0.75, 0.75], [0, 1.0]])
    report = robust_enclosures(abstract, "goal", [1, 5, 10], bits=40)
    rng = np.random.default_rng(128)
    for _ in range(20):
        # Rational interior kernels with different successor mixtures within the merge.
        first = Fraction(int(rng.integers(16, 33)), 64)
        second = Fraction(int(rng.integers(32, 49)), 64)
        kernel = (
            (1 - first, Fraction(), first),
            (Fraction(), 1 - second, second),
            (Fraction(), Fraction(), Fraction(1)),
        )
        for h in [1, 5, 10]:
            truth = exact_values(kernel, source.labels, "goal", h)
            values = report["values"][str(h)]
            for s in [0, 1]:
                assert Fraction(values["min_lower"][0], 1 << 40) <= truth[s]
                assert truth[s] <= Fraction(values["max_upper"][0], 1 << 40)
    tampered = deepcopy(cert)
    tampered["abstract_lower"][0][1] = 0.5
    with pytest.raises(ValueError, match="recomputation"):
        verify_envelope(source, tampered)


def test_projection_uses_simplex_constraints_and_outward_rational_rounding():
    source = IntervalChain(
        np.array([[0.1, 0.1, 0.1], [0, 1.0, 0], [0, 0, 1.0]]),
        np.array([[0.125, 0.125, 1.0], [0, 1.0, 0], [0, 0, 1.0]]),
        ("active", "active", "goal"),
    )
    abstract, cert = abstract_envelope(source, ((0, 1), (2,)))
    # State 1 never reaches goal, so the hull minimum is zero. State 0's exact
    # maximum is 1-2*float(.1), whose nearest float rounds inward.
    exact_upper = Fraction(1) - 2 * Fraction.from_float(0.1)
    assert Fraction.from_float(abstract.upper[0, 1]) >= exact_upper
    assert cert["exact_projected_hulls"][0]["upper"][1] == [
        exact_upper.numerator,
        exact_upper.denominator,
    ]
    unmerged, _ = abstract_envelope(source, ((0,), (1,), (2,)))
    assert unmerged.lower[0, 2] == 0.75
    with pytest.raises(ValueError, match="label"):
        abstract_envelope(source, ((0, 2), (1,)))


def test_envelope_rejects_exactly_infeasible_input_even_if_float_sum_passes():
    p = np.array([[0.2, 0.8], [0, 1.0]])
    with pytest.raises(ValueError, match="exactly infeasible"):
        abstract_envelope(IntervalChain(p, p, ("active", "goal")), ((0,), (1,)))
