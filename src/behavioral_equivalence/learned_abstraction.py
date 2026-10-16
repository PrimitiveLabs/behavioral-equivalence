"""Nested, emission-aware latent compression with exact numeric certificates."""

from fractions import Fraction
import numpy as np
from .core import Chain, Partition, check_partition, quotient
from .decision_model import decision_partition, decision_horizon
from .integer_bounds import integer_kernel, exact_radius, certify_abstraction


def certified_latent_refinement(
    parent: Chain,
    initial_blocks: Partition,
    epsilon: float,
    decision_horizons: tuple[int, ...] = (1, 5, 10, 25, 50),
) -> tuple[Chain, Partition, dict]:
    """Split a proposed latent merge until the actual lifted radius is valid.

    All three observation stages use the same latent partition. Every returned
    state is a union of parent states; this differs from refitting a separate
    K-state clustering. This algorithm does not guarantee optimal compression.
    """
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("Finite epsilon in [0,1] required")
    if parent.size % 3 or parent.labels != tuple(
        label for _ in range(parent.size // 3) for label in ("pending", "negative", "positive")
    ):
        raise ValueError("Expected an emission-aware 3K decision chain")
    blocks = initial_blocks
    check_partition(parent, decision_partition(blocks))
    kernel = integer_kernel(parent)
    budget = Fraction.from_float(float(epsilon))
    trace = []
    while True:
        lifted = decision_partition(blocks)
        abstract, _ = quotient(parent, lifted)
        radius, witness = exact_radius(kernel, integer_kernel(abstract), lifted)
        trace.append(
            {
                "latent_states": len(blocks),
                "epsilon_for_display": float(radius),
                "witness_latent": witness // 3,
            }
        )
        if radius <= budget:
            break
        state = witness // 3
        selected = next(i for i, block in enumerate(blocks) if state in block)
        if len(blocks[selected]) == 1:
            # Rounding of aggregated successor weights can expose a singleton
            # origin. Refine a nontrivial successor partition in that case.
            selected = max(range(len(blocks)), key=lambda i: len(blocks[i]))
            state = blocks[selected][0]
        if len(blocks[selected]) == 1:
            raise AssertionError("Singleton partition must have zero exact radius")
        remainder = tuple(s for s in blocks[selected] if s != state)
        blocks = blocks[:selected] + (remainder, (state,)) + blocks[selected + 1 :]
    horizons = [decision_horizon(h) for h in decision_horizons]
    certificate = certify_abstraction(
        parent,
        abstract,
        decision_partition(blocks),
        horizons,
        properties={"positive": ("positive",), "negative": ("negative",)},
    )
    return (
        abstract,
        blocks,
        {
            "certificate": certificate,
            "refinement_trace": trace,
            "parent_latent_states": parent.size // 3,
            "reduced_latent_states": len(blocks),
            "latent_compression_ratio": (parent.size // 3) / len(blocks),
            "decision_horizons": list(decision_horizons),
            "prediction_scope": "next-emission and at-least-one labeled outcome in the specified finite model; not a human Markov sufficiency theorem",
        },
    )


def next_positive_probabilities(abstract: Chain, latent_blocks: Partition) -> np.ndarray:
    """Return one forecast per parent latent state under the certified kernel."""
    if abstract.labels != tuple(
        label for _ in latent_blocks for label in ("pending", "negative", "positive")
    ):
        raise ValueError("Abstract labels and latent partition disagree")
    n = sum(len(block) for block in latent_blocks)
    if set(s for block in latent_blocks for s in block) != set(range(n)):
        raise ValueError("Latent blocks must partition parent indices")
    result = np.empty(n)
    rows = integer_kernel(abstract)
    for b, block in enumerate(latent_blocks):
        row = rows[3 * b]
        probability = Fraction(
            sum(w for j, w in row.terms if abstract.labels[j] == "positive"), row.total
        )
        result[list(block)] = float(probability)
    return result
