"""Partition refinement controls for fixed observed-response discovery tables."""

import numpy as np
from .core import Partition


def validate_latent_partition(blocks: Partition, state_count: int) -> None:
    flat = [s for block in blocks for s in block]
    if not blocks or any(not b for b in blocks) or sorted(flat) != list(range(state_count)):
        raise ValueError("Complete disjoint nonempty latent partition required")


def split_response_witness(
    blocks: Partition, probabilities: np.ndarray, witness: dict
) -> Partition:
    """Split one block at a supported witness midpoint, preserving all parent states.

    Unobserved other states remain in the lower-probability subgroup. This is a
    discovery heuristic, not a human significance test or a verified quotient.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 75:
        raise ValueError("Parent-state by scenario/item-group probabilities required")
    validate_latent_partition(blocks, len(p))
    a, b = witness["states"]
    env = witness["environment"]
    if not isinstance(env, int) or not 0 <= env < 75 or env % 5 == 4:
        raise ValueError("Known-duration environment required")
    if a == b or not 0 <= a < len(p) or not 0 <= b < len(p):
        raise ValueError("Two distinct valid parent states required")
    selected = next((i for i, block in enumerate(blocks) if a in block and b in block), None)
    if selected is None:
        raise ValueError("Witness states must still share one block")
    pa, pb = p[a, env], p[b, env]
    if not np.isfinite([pa, pb]).all() or not 0 <= pa <= 1 or not 0 <= pb <= 1 or pa == pb:
        raise ValueError("Finite distinct Bernoulli witness probabilities required")
    values = p[list(blocks[selected]), env]
    if (
        np.isinf(values).any()
        or np.any(values[np.isfinite(values)] < 0)
        or np.any(values[np.isfinite(values)] > 1)
    ):
        raise ValueError("Observed block probabilities must be Bernoulli means")
    lower, upper = min(pa, pb), max(pa, pb)
    midpoint = min(lower + (upper - lower) / 2, np.nextafter(upper, lower))
    high = tuple(s for s in blocks[selected] if np.isfinite(p[s, env]) and p[s, env] > midpoint)
    low = tuple(s for s in blocks[selected] if s not in high)
    if not high or not low:
        raise ArithmeticError("Distinct witness probabilities must separate")
    return blocks[:selected] + (tuple(sorted(low)), tuple(sorted(high))) + blocks[selected + 1 :]


def random_capacity_refinement(
    blocks: Partition, state_count: int, target_count: int, seed: int
) -> Partition:
    """Refine the same initial partition to the targeted control's exact capacity."""
    validate_latent_partition(blocks, state_count)
    if not isinstance(target_count, int) or not len(blocks) <= target_count <= state_count:
        raise ValueError("Feasible target count at least initial partition capacity required")
    rng = np.random.default_rng(seed)
    result = blocks
    while len(result) < target_count:
        choices = [i for i, block in enumerate(result) if len(block) > 1]
        chosen = int(rng.choice(choices))
        members = rng.permutation(result[chosen])
        cut = int(rng.integers(1, len(members)))
        split = tuple(
            tuple(sorted(int(s) for s in group)) for group in (members[:cut], members[cut:])
        )
        result = result[:chosen] + split + result[chosen + 1 :]
    return result


def group_response_model(counts: np.ndarray, positives: np.ndarray, blocks: Partition) -> dict:
    """Fit smoothed point emissions on estimation data after discovery is frozen.

    Zero-support parent cells use the environment-wide Jeffreys mean. A quotient
    uses a convex mixture with weights count+1. Numerical discrepancy here is
    descriptive; this function does not issue arithmetic or sampling certificates.
    """
    n, y = np.asarray(counts), np.asarray(positives, dtype=float)
    if (
        n.ndim != 2
        or n.shape[1] != 75
        or n.dtype.kind not in "iu"
        or y.shape != n.shape
        or np.any(n < 0)
        or not np.isfinite(y).all()
        or np.any(y < 0)
        or np.any(y > n)
        or np.any(y != np.floor(y))
    ):
        raise ValueError("Integer cell counts and aligned binary positive counts required")
    validate_latent_partition(blocks, len(n))
    fallback = (y.sum(axis=0) + 0.5) / (n.sum(axis=0) + 1)
    parent = np.where(n > 0, (y + 0.5) / (n + 1), fallback)
    abstract = np.stack(
        [np.average(parent[list(block)], axis=0, weights=n[list(block)] + 1) for block in blocks]
    )
    mapping = np.empty(len(n), dtype=np.int64)
    for index, block in enumerate(blocks):
        mapping[list(block)] = index
    discrepancy = np.abs(parent - abstract[mapping])
    return dict(
        parent=parent,
        abstract=abstract,
        mapping=mapping,
        absolute_discrepancy=discrepancy,
        maximum_point_discrepancy=float(discrepancy.max()),
        scope="Fitted one-step point-model emissions only; no human uncertainty or arithmetic certificate",
    )
