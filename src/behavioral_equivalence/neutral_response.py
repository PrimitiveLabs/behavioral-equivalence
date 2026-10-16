"""Certified neutral one-step quotients and exact fitted-count counterexamples."""

from fractions import Fraction
import numpy as np
from .response_chain import binary_response_chain
from .core import quotient
from .integer_bounds import certify_abstraction, verify_abstraction
from .response_refinement import validate_latent_partition


def neutral_training_probabilities(prior, counts):
    p, n = np.asarray(prior, dtype=float), np.asarray(counts)
    if (
        p.ndim != 2
        or p.shape[1] != 75
        or n.shape != p.shape
        or n.dtype.kind not in "iu"
        or not len(p)
        or np.any(n < 0)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or n.sum() == 0
    ):
        raise ValueError("Valid teacher means and nonempty standard-training counts required")
    totals = (p * n).sum(axis=1)
    mass = n.sum(axis=1)
    fallback = float(totals.sum() / mass.sum())
    neutral = np.full(len(p), fallback)
    np.divide(totals, mass, out=neutral, where=mass > 0)
    return neutral


def certified_neutral_partition(probabilities, epsilon=0.01):
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("Probability error budget in[0,1]required")
    binary_response_chain(probabilities)
    screening = max(0, float(epsilon) - 2**-40)
    # One-step scalar grouping can merge nearby rows that generic split-only
    # refinement would separate early. No minimum-cardinality claim is made.
    groups = []
    values = np.asarray(probabilities, dtype=float)
    for state in np.argsort(values, kind="stable"):
        trial = groups[-1] + (int(state),) if groups else (int(state),)
        if groups and np.max(np.abs(values[list(trial)] - values[list(trial)].mean())) <= screening:
            groups[-1] = trial
        else:
            groups.append((int(state),))
    return repair_neutral_partition(probabilities, tuple(tuple(sorted(b)) for b in groups), epsilon)


def _neutral_certificate(probabilities, latent):
    parent = binary_response_chain(probabilities)
    k = len(probabilities)
    validate_latent_partition(latent, k)
    blocks = latent + ((k,), (k + 1,))
    abstract, _ = quotient(parent, blocks)
    certificate = certify_abstraction(
        parent,
        abstract,
        blocks,
        [1],
        bits=48,
        properties={"positive": ("positive",), "negative": ("negative",)},
    )
    verify_abstraction(parent, abstract, certificate)
    value = certificate["epsilon"]
    radius = Fraction(int(value["numerator_hex"], 16), int(value["denominator_hex"], 16))
    return certificate, radius


def repair_neutral_partition(probabilities, latent, epsilon=0.01):
    """Splitting approximate classes can move means; reverify and repair that shift."""
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("Probability budget in[0,1]required")
    budget = Fraction.from_float(float(epsilon))
    while True:
        certificate, radius = _neutral_certificate(probabilities, latent)
        if radius <= budget:
            return latent, certificate
        state = certificate["epsilon_witness"]
        index = next(i for i, block in enumerate(latent) if state in block)
        block = latent[index]
        if len(block) == 1:
            raise ArithmeticError("Singleton quotient must preserve its normalized row")
        latent = (
            latent[:index] + (tuple(s for s in block if s != state), (state,)) + latent[index + 1 :]
        )


def random_neutral_capacity(probabilities, latent, target_count, seed, epsilon=0.01):
    """Random valid ordered cuts, with exact acceptance at the neutral precision."""
    probabilities = np.asarray(probabilities, dtype=float)
    validate_latent_partition(latent, len(probabilities))
    if (
        not isinstance(target_count, int)
        or not len(latent) <= target_count <= len(probabilities)
        or not np.isfinite(epsilon)
        or not 0 <= epsilon <= 1
    ):
        raise ValueError("Feasible target capacity and probability budget required")
    budget = Fraction.from_float(float(epsilon))
    certificate, radius = _neutral_certificate(probabilities, latent)
    if radius > budget:
        raise ValueError("Neutral-certified starting partition required")
    rng = np.random.default_rng(seed)
    while len(latent) < target_count:
        candidates = []
        for i, block in enumerate(latent):
            ordered = sorted(block, key=lambda state: (probabilities[state], state))
            for cut in range(1, len(ordered)):
                split = (tuple(sorted(ordered[:cut])), tuple(sorted(ordered[cut:])))
                candidates.append(latent[:i] + split + latent[i + 1 :])
        accepted = None
        for index in rng.permutation(len(candidates)):
            candidate = candidates[int(index)]
            next_certificate, next_radius = _neutral_certificate(probabilities, candidate)
            if next_radius <= budget:
                accepted = (candidate, next_certificate)
                break
        if accepted is None:
            raise ArithmeticError(
                "No neutral-valid one-state capacity increase at this numeric precision"
            )
        latent, certificate = accepted
    return latent, certificate


def exact_response_gap(counts, positives, states, environment):
    n, y = np.asarray(counts), np.asarray(positives)
    if (
        n.ndim != 2
        or n.shape[1] != 75
        or y.shape != n.shape
        or n.dtype.kind not in "iu"
        or len(states) != 2
        or states[0] == states[1]
        or any(not isinstance(s, (int, np.integer)) or not 0 <= s < len(n) for s in states)
        or not isinstance(environment, (int, np.integer))
        or not 0 <= environment < 75
    ):
        raise ValueError("Two valid parent states and context in aligned count tables required")
    a, b = map(int, states)
    e = int(environment)
    trials = [int(n[a, e]), int(n[b, e])]
    positive = [y[a, e], y[b, e]]
    if any(t < 1 for t in trials) or any(
        not np.isfinite(v) or v != int(v) or not 0 <= int(v) <= t for v, t in zip(positive, trials)
    ):
        raise ValueError("Positive trial counts and binary-success counts required")
    return abs(Fraction(int(positive[0]), trials[0]) - Fraction(int(positive[1]), trials[1]))


def response_gap_witness(counts, positives, states, environment, epsilon=0.01):
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("Probability budget in[0,1]required")
    gap = exact_response_gap(counts, positives, states, environment)
    lower = gap / 2
    budget = Fraction.from_float(float(epsilon))
    return dict(
        states=list(map(int, states)),
        environment=int(environment),
        gap=dict(numerator_hex=hex(gap.numerator), denominator_hex=hex(gap.denominator)),
        minimum_shared_row_radius=dict(
            numerator_hex=hex(lower.numerator), denominator_hex=hex(lower.denominator)
        ),
        exceeds_budget=lower > budget,
        scope="Exact fitted empirical count kernels: triangle inequality excludes a shared row at the stated radius. Not a human probability or significance guarantee.",
    )
