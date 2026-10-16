"""Fixed-design row imbalance and known-terminal support controls."""

import numpy as np
from .core import Chain
from .uncertainty import IntervalChain


def row_sample_schedule(chain: Chain, regime: str, per_active: int = 1000) -> np.ndarray:
    if (
        regime not in {"uniform_active", "imbalanced_active"}
        or not isinstance(per_active, int)
        or per_active < 1
    ):
        raise ValueError("Known fixed row regime and positive per-active budget required")
    active = np.flatnonzero(np.array(chain.labels) == "active")
    if len(active) < 4:
        raise ValueError("At least four active rows required for the imbalance control")
    result = np.zeros(chain.size, dtype=np.int64)
    if regime == "uniform_active":
        result[active] = per_active
        return result
    distribution = chain.initial.copy()
    visits = np.zeros(chain.size)
    for _ in range(50):
        visits += distribution
        distribution = np.einsum("i,ij->j", distribution, chain.transitions)
    weights = visits[active] + 1e-12
    # Allocation uses the specified world before any calibration samples. It is
    # not an adaptive stopping rule and is not inferred from human trajectories.
    order = np.argsort(weights, kind="stable")
    weights[order[: max(1, len(active) // 4)]] = 0
    expected = weights / weights.sum() * (len(active) * per_active)
    allocation = np.floor(expected).astype(np.int64)
    remainder = len(active) * per_active - int(allocation.sum())
    extra = np.argsort(-(expected - allocation), kind="stable")[:remainder]
    allocation[extra] += 1
    result[active] = allocation
    return result


def terminal_label_support(uncertain: IntervalChain, concrete: Chain) -> IntervalChain:
    """Intersect a row set with explicitly known closed terminal label classes."""
    if uncertain.labels != concrete.labels:
        raise ValueError("Matching labels required")
    lower, upper = uncertain.lower.copy(), uncertain.upper.copy()
    labels = np.array(concrete.labels)
    for s, label in enumerate(concrete.labels):
        if label not in {"complete", "disengage"}:
            continue
        outside = labels != label
        if np.any(concrete.transitions[s, outside] != 0):
            raise ValueError("Specified terminal label class is not closed")
        lower[s, outside] = upper[s, outside] = 0
    return IntervalChain(lower, upper, concrete.labels)
