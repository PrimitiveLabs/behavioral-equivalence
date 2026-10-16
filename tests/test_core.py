from fractions import Fraction
import numpy as np
import pytest
from behavioral_equivalence.core import (
    Chain,
    approximate_partition,
    coupling_bound,
    exact_partition,
    quotient,
    rational_reachability,
    reachability,
)
from behavioral_equivalence.synthetic import FAMILIES, generate, intervene
from behavioral_equivalence.uncertainty import (
    IntervalChain,
    multinomial_intervals,
    robust_reachability,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_known_exact_quotient_and_independent_oracle(family):
    w = generate(family, 32, 19)
    q, epsilon = quotient(w.chain, w.partition)
    assert epsilon == 0
    blocks = exact_partition([w.chain])
    assert len(blocks) <= len(w.partition)
    for horizon in (0, 1, 5, 10):
        concrete = reachability(w.chain, "complete", horizon)
        abstract = reachability(q, "complete", horizon)
        for i, block in enumerate(w.partition):
            np.testing.assert_allclose(concrete[list(block)], abstract[i], atol=1e-14)
    oracle = rational_reachability(q, "complete", 5)
    np.testing.assert_allclose([float(x) for x in oracle], reachability(q, "complete", 5))
    assert w.chain.digest() == generate(family, 32, 19).chain.digest()


def test_exact_rational_fixture():
    c = Chain(
        np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]]),
        ("active", "complete", "disengage"),
        np.array([1.0, 0, 0]),
    )
    assert rational_reachability(c, "complete", 5)[0] == Fraction(31, 64)


def test_approximate_certificate_and_environment_refinement():
    w = generate("trust", 32, 3, 0.01)
    blocks = approximate_partition([w.chain], 0.02)
    q, epsilon = quotient(w.chain, blocks)
    assert epsilon <= 0.02
    for h in (1, 5, 25, 50):
        actual = reachability(w.chain, "complete", h)
        abstract = reachability(q, "complete", h)
        error = max(abs(actual[s] - abstract[i]) for i, b in enumerate(blocks) for s in b)
        assert error <= coupling_bound(epsilon, h) + 1e-12
    exact = generate("trust", 32, 3)
    changed = intervene(exact, 0.3)
    assert quotient(changed, exact.partition)[1] > 0.1
    refined = approximate_partition([exact.chain, changed], 0.02)
    assert quotient(changed, refined)[1] <= 0.02


def test_nontransitivity_cannot_chain_merge():
    p = np.array(
        [
            [0, 0, 0, 0.4, 0.6],
            [0, 0, 0, 0.5, 0.5],
            [0, 0, 0, 0.6, 0.4],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    c = Chain(p, ("a", "a", "a", "complete", "disengage"), np.array([1.0, 0, 0, 0, 0]))
    assert quotient(c, ((0, 1, 2), (3,), (4,)))[1] > 0.06
    blocks = approximate_partition([c], 0.06)
    assert quotient(c, blocks)[1] <= 0.06
    assert not any(set((0, 1, 2)).issubset(b) for b in blocks)


@pytest.mark.parametrize(
    "epsilon,eta,h,expected",
    [(0, 0, 50, 0), (1, 0, 0, 0), (1, 0, 1, 1), (0, 0.2, 0, 0.2), (0.5, 0, 2, 0.75)],
)
def test_coupling_edges(epsilon, eta, h, expected):
    assert coupling_bound(epsilon, h, eta) == expected


def test_robust_known_extrema():
    model = IntervalChain(
        np.array([[0.4, 0.6], [0, 1]]), np.array([[0.8, 0.6], [0, 1]]), ("active", "complete")
    )
    # Stochastic feasibility forces first row to exactly [.4,.6].
    lower, upper = robust_reachability(model, "complete", 2)
    assert lower[0] == pytest.approx(0.84)
    assert upper[0] == pytest.approx(0.84)
    uncertain = IntervalChain(
        np.array([[0, 0.2], [0, 1]]), np.array([[0.8, 1], [0, 1]]), ("active", "complete")
    )
    lower, upper = robust_reachability(uncertain, "complete", 2)
    assert lower[0] == pytest.approx(0.36)
    assert upper[0] == 1


def test_uncertainty_nominal_inclusion_and_unobserved_rows():
    x = np.array([[90.0, 10.0], [0.0, 0.0]])
    model = multinomial_intervals(x, ("active", "complete"))
    assert model.lower[1].sum() == 0
    assert model.upper[1].sum() == 2
    c = Chain(np.array([[0.9, 0.1], [0, 1]]), model.labels, np.array([1.0, 0]))
    lo, hi = robust_reachability(model, "complete", 5)
    nominal = reachability(c, "complete", 5)
    assert np.all(lo <= nominal) and np.all(nominal <= hi)


def test_reject_invalid_inputs():
    with pytest.raises(ValueError):
        Chain(np.array([[0.2]]), ("active",), np.array([1.0]))
    with pytest.raises(ValueError):
        IntervalChain(np.array([[0.1, 0.1], [0, 0]]), np.array([[0.2, 0.2], [0, 1]]), ("a", "b"))
    with pytest.raises(ValueError):
        coupling_bound(-0.1, 5)
    with pytest.raises(ValueError):
        multinomial_intervals(np.array([[1.5]]), ("a",))
    w = generate("trust", 16, 0)
    with pytest.raises(ValueError):
        quotient(w.chain, ((0,),))
    with pytest.raises(ValueError):
        reachability(w.chain, "complete", -1)


def test_robust_row_optimizer_against_independent_linear_program():
    from scipy.optimize import linprog
    from behavioral_equivalence.uncertainty import row_extreme

    rng = np.random.default_rng(42)
    for _ in range(20):
        nominal = rng.dirichlet(np.ones(6))
        lo = nominal * rng.uniform(0, 1, 6)
        hi = nominal + (1 - nominal) * rng.uniform(0, 1, 6)
        values = rng.uniform(0, 1, 6)
        for maximize in (False, True):
            sign = -1 if maximize else 1
            solved = linprog(
                sign * values,
                A_eq=np.ones((1, 6)),
                b_eq=[1],
                bounds=list(zip(lo, hi)),
                method="highs",
            )
            assert solved.success
            assert row_extreme(lo, hi, values, maximize) == pytest.approx(sign * solved.fun)


@pytest.mark.parametrize("family", FAMILIES)
def test_controlled_perturbation_preserves_same_base_kernel(family):
    exact = generate(family, 32, 42)
    changed = generate(family, 32, 42, 0.01)
    deviations = np.abs(exact.chain.transitions - changed.chain.transitions).sum(axis=1) / 2
    assert deviations.max() <= 0.01 + 1e-14


def test_large_generator_endpoint():
    w = generate("memory", 512, 7, 0.03)
    assert w.chain.size == 512
    np.testing.assert_allclose(w.chain.transitions.sum(axis=1), 1, atol=1e-14)


def test_exact_binomial_intervals_handle_zero_and_full_counts():
    from behavioral_equivalence.uncertainty import clopper_pearson_intervals

    counts = np.array([[0.0, 100.0], [0.0, 0.0]])
    model = clopper_pearson_intervals(counts, ("a", "b"), alpha=0.05)
    tail = 0.05 / (2 * 4)
    assert model.lower[0, 0] == 0
    assert model.upper[0, 0] == pytest.approx(1 - tail**0.01)
    assert model.lower[0, 1] == pytest.approx(tail**0.01)
    assert model.upper[0, 1] == 1
    np.testing.assert_array_equal(model.lower[1], [0, 0])
    np.testing.assert_array_equal(model.upper[1], [1, 1])


def test_exact_partition_does_not_merge_differences_hidden_by_float_aggregation():
    from behavioral_equivalence.rational import certify

    p = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, np.nextafter(0.25, 0)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    chain = Chain(p, ("active", "active", "goal", "goal"), np.array([1.0, 0, 0, 0]))
    # A floating signature loses this distinction, but the rational models differ.
    assert p[0, 2:].sum() == p[1, 2:].sum()
    blocks = exact_partition([chain])
    assert len(blocks) == 3
    certificate = certify(chain, blocks, "goal", 5)
    assert certificate["epsilon"] == "0"
    falsely_merged = certify(chain, ((0, 1), (2, 3)), "goal", 1)
    assert Fraction(falsely_merged["epsilon"]) > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_seed_changes_behavioral_kernel_not_only_redundant_wiring(family):
    quotients = []
    for seed in range(5):
        w = generate(family, 32, seed)
        q, error = quotient(w.chain, w.partition)
        assert error == 0
        quotients.append(q.transitions.tobytes())
    assert len(set(quotients)) > 1


def test_bayesian_family_quantizes_actual_posterior_updates():
    from fractions import Fraction

    w = generate("belief", 32, 9)
    q, _ = quotient(w.chain, w.partition)
    params = dict(w.parameters)
    active = q.size - 2
    s = 2
    prior = Fraction(s, active - 1)
    reliability = Fraction(params["evidence_reliability_tenths"], 10)
    cue = reliability * prior + (1 - reliability) * (1 - prior)
    positive = int(reliability * prior / cue * (active - 1) + Fraction(1, 2))
    negative = int((1 - reliability) * prior / (1 - cue) * (active - 1) + Fraction(1, 2))
    destinations = set(np.flatnonzero(q.transitions[s, :active]))
    assert destinations == {positive, negative}


def test_verification_guided_refinement_preserves_initial_partition():
    world = generate("trust", 32, 4, 0.01)
    initial = tuple((s,) for s in range(32))
    assert approximate_partition([world.chain], 0.03, initial) == initial
    with pytest.raises(ValueError):
        approximate_partition([world.chain], 0.03, ((0, 31), tuple(range(1, 31))))
    exact = generate("trust", 16, 2)
    blocks = approximate_partition([exact.chain], 0)
    assert blocks == exact_partition([exact.chain])
