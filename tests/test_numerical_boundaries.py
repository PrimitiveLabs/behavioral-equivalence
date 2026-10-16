"""Boundary assertions for the released numerical certificate interfaces."""

from fractions import Fraction
import numpy as np
import pytest
from behavioral_equivalence.core import Chain
from behavioral_equivalence.integer_bounds import (
    integer_kernel,
    reachability_enclosures,
    robust_enclosures,
)
from behavioral_equivalence.rational import certify, verify
from behavioral_equivalence.uncertainty import (
    IntervalChain,
    clopper_pearson_intervals,
    multinomial_intervals,
)


@pytest.fixture
def chain():
    return Chain(np.array([[0.5, 0.5], [0, 1.0]]), ("active", "goal"), np.array([1.0, 0]))


@pytest.mark.parametrize("horizon", [-1, True, False, 1.5])
def test_rational_certificate_rejects_invalid_horizon(chain, horizon):
    with pytest.raises(ValueError, match="nonnegative integer"):
        certify(chain, ((0,), (1,)), "goal", horizon)


def test_rational_horizon_zero_has_exact_label_values_and_no_error(chain):
    certificate = certify(chain, ((0,), (1,)), "goal", 0)
    assert certificate["abstract_values"] == ["0", "1"]
    assert certificate["intervals"] == [["0", "0"], ["1", "1"]]
    assert Fraction(certificate["delta"]) == Fraction(certificate["observed_max_error"]) == 0
    verify(chain, certificate)


@pytest.mark.parametrize("horizons", [[], [-1], [True], [False], [1.5]])
@pytest.mark.parametrize("robust", [False, True])
def test_integer_enclosures_reject_invalid_horizons(chain, horizons, robust):
    with pytest.raises(ValueError, match="horizons"):
        if robust:
            robust_enclosures(
                IntervalChain(chain.transitions, chain.transitions, chain.labels), "goal", horizons
            )
        else:
            reachability_enclosures(
                integer_kernel(chain), chain.labels, frozenset({"goal"}), horizons
            )


@pytest.mark.parametrize("bits", [-1, 0, 63, 1.5])
@pytest.mark.parametrize("robust", [False, True])
def test_integer_enclosures_reject_invalid_precision(chain, bits, robust):
    with pytest.raises(ValueError, match="precision"):
        if robust:
            robust_enclosures(
                IntervalChain(chain.transitions, chain.transitions, chain.labels), "goal", [0], bits
            )
        else:
            reachability_enclosures(
                integer_kernel(chain), chain.labels, frozenset({"goal"}), [0], bits
            )


@pytest.mark.parametrize("bits", [1, 62])
def test_precision_endpoints_and_zero_horizon_vectors(chain, bits):
    denominator = 1 << bits
    expected = (0, denominator)
    point = reachability_enclosures(
        integer_kernel(chain), chain.labels, frozenset({"goal"}), [0], bits
    )
    assert point == {0: (expected, expected)}
    result = robust_enclosures(
        IntervalChain(chain.transitions, chain.transitions, chain.labels), "goal", [0], bits
    )
    assert result["values"] == {
        "0": {key: list(expected) for key in ["min_lower", "min_upper", "max_lower", "max_upper"]}
    }


@pytest.mark.parametrize(
    "lower,upper,labels,message",
    [
        ([], [], (), "dimensions"),
        ([[0]], [[1]], ("a", "b"), "dimensions"),
        ([[0]], [[1, 1]], ("a",), "dimensions"),
        ([[np.nan]], [[1]], ("a",), "finite"),
        ([[0]], [[np.inf]], ("a",), "finite"),
        ([[-0.1]], [[1]], ("a",), "Invalid probability"),
        ([[0]], [[1.1]], ("a",), "Invalid probability"),
        ([[1]], [[0.5]], ("a",), "Invalid probability"),
        ([[0.6, 0.6], [0, 1]], [[1, 1], [0, 1]], ("a", "b"), "stochastic member"),
        ([[0, 0], [0, 1]], [[0.4, 0.4], [0, 1]], ("a", "b"), "stochastic member"),
    ],
)
def test_interval_model_rejects_invalid_boundaries(lower, upper, labels, message):
    with pytest.raises(ValueError, match=message):
        IntervalChain(np.array(lower), np.array(upper), labels)


def test_interval_model_copies_inputs_and_protects_probability_storage():
    lower = np.zeros((2, 2))
    upper = np.ones((2, 2))
    model = IntervalChain(lower, upper, ("a", "b"))
    lower[:] = 1
    upper[:] = 0
    assert np.array_equal(model.lower, np.zeros((2, 2)))
    assert np.array_equal(model.upper, np.ones((2, 2)))
    for values in [model.lower, model.upper]:
        with pytest.raises(ValueError, match="read-only"):
            values[0, 0] = 0.5


@pytest.mark.parametrize("method", [multinomial_intervals, clopper_pearson_intervals])
@pytest.mark.parametrize(
    "counts",
    [
        np.array([1, 2]),
        np.ones((1, 1)),
        np.array([[-1, 2], [0, 1]]),
        np.array([[np.nan, 1], [0, 1]]),
        np.array([[np.inf, 1], [0, 1]]),
        np.array([[1.5, 1], [0, 1]]),
    ],
)
def test_statistical_intervals_reject_invalid_counts(method, counts):
    with pytest.raises(ValueError):
        method(counts, ("a", "b"))


@pytest.mark.parametrize("method", [multinomial_intervals, clopper_pearson_intervals])
@pytest.mark.parametrize("alpha", [-0.1, 0, 1, 1.1, np.nan, np.inf])
def test_statistical_intervals_reject_invalid_family_error_probability(method, alpha):
    with pytest.raises(ValueError, match="alpha"):
        method(np.eye(2), ("a", "b"), alpha)


@pytest.mark.parametrize("method", [multinomial_intervals, clopper_pearson_intervals])
def test_statistical_intervals_unsampled_rows_and_all_positive_zero_count_edges(method):
    counts = np.array([[100, 0], [0, 0]])
    result = method(counts, ("a", "b"))
    assert np.array_equal(result.lower[1], [0, 0])
    assert np.array_equal(result.upper[1], [1, 1])
    assert result.upper[0, 0] == 1 and result.lower[0, 0] > 0
    assert result.lower[0, 1] == 0 and result.upper[0, 1] < 1
    assert np.array_equal(counts, [[100, 0], [0, 0]])
