"""Endpoint-specific one-observation abstractions, separate from click trajectories."""

from fractions import Fraction
import numpy as np
from behavioral_equivalence.core import Chain, check_partition, quotient
from behavioral_equivalence.integer_bounds import integer_kernel, exact_radius, certify_abstraction


def endpoint_chain(emissions):
    e = np.asarray(emissions, dtype=float)
    if (
        e.ndim != 2
        or not len(e)
        or e.shape[1] not in [2, 4]
        or not np.isfinite(e).all()
        or np.any(e < 0)
        or not np.allclose(e.sum(axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Finite normalized binary or four-class emissions required")
    k, c = e.shape
    p = np.zeros((k + c, k + c))
    p[:k, k:] = e
    p[k:, k:] = np.eye(c)
    return Chain(
        p,
        ("pending",) * k + tuple(f"outcome_{i}" for i in range(c)),
        np.r_[np.full(k, 1 / k), np.zeros(c)],
    )


def certify_endpoint(emissions, initial_blocks, epsilon):
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("Finite epsilon in [0,1] required")
    parent = endpoint_chain(emissions)
    k, c = np.asarray(emissions).shape
    blocks = tuple(tuple(b) for b in initial_blocks)
    if set(s for b in blocks for s in b) != set(range(k)):
        raise ValueError("Latent partition must cover exactly the endpoint parent states")
    kernel = integer_kernel(parent)
    budget = Fraction.from_float(float(epsilon))
    trace = []
    while True:
        lifted = blocks + tuple((k + i,) for i in range(c))
        check_partition(parent, lifted)
        abstract, _ = quotient(parent, lifted)
        radius, witness = exact_radius(kernel, integer_kernel(abstract), lifted)
        trace.append(dict(latent_states=len(blocks), radius=float(radius)))
        if radius <= budget:
            break
        index = next(i for i, b in enumerate(blocks) if witness in b)
        if len(blocks[index]) == 1:
            raise AssertionError("Singleton endpoint row must preserve its exact kernel")
        remainder = tuple(s for s in blocks[index] if s != witness)
        blocks = blocks[:index] + (remainder, (witness,)) + blocks[index + 1 :]
    certificate = certify_abstraction(
        parent,
        abstract,
        lifted,
        [1],
        properties={f"outcome_{i}": (f"outcome_{i}",) for i in range(c)},
    )
    rows = integer_kernel(abstract)
    probabilities = np.zeros((k, c))
    for i, b in enumerate(blocks):
        row = rows[i]
        terms = dict(row.terms)
        probability = [float(Fraction(terms.get(len(blocks) + j, 0), row.total)) for j in range(c)]
        probabilities[list(b)] = probability
    return dict(
        parent=parent,
        abstract=abstract,
        blocks=blocks,
        certificate=certificate,
        probabilities=probabilities,
        trace=trace,
        scope="One recorded outcome from a frozen predictive state. Absorbing terminal labels do not model subsequent decisions or joint dependence across different endpoints.",
    )
