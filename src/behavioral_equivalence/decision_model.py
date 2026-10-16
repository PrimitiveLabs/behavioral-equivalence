"""Emission-aware finite decision models with explicit observable outcome stages."""

import numpy as np
from .core import Chain, Partition


def decision_chain(
    transitions: np.ndarray, emissions: np.ndarray, initial_latent: np.ndarray
) -> Chain:
    """Lift K predictive states to 3K labeled states.

    Pending(s) emits negative/positive with E_s. The observed outcome then moves
    to the next pending predictive state via T[s,outcome,:]. One decision uses
    two graph transitions, except that its outcome is visible after the first.
    These are specified-model semantics, not an assertion of human Markovity.
    """
    t = np.asarray(transitions, dtype=float)
    e = np.asarray(emissions, dtype=float)
    initial = np.asarray(initial_latent, dtype=float)
    k = len(initial)
    if k == 0 or t.shape != (k, 2, k) or e.shape != (k, 2):
        raise ValueError("Expected transitions[K,2,K], emissions[K,2], initial[K]")
    p = np.zeros((3 * k, 3 * k))
    mu = np.zeros(3 * k)
    for s in range(k):
        p[3 * s, 3 * s + 1 : 3 * s + 3] = e[s]
        for y in range(2):
            p[3 * s + 1 + y, ::3] = t[s, y]
        mu[3 * s] = initial[s]
    return Chain(
        p, tuple(label for _ in range(k) for label in ("pending", "negative", "positive")), mu
    )


def decision_partition(latent_blocks: Partition) -> Partition:
    """Lift each latent merge into separate, label-preserving stage merges."""
    return tuple(
        tuple(3 * s + offset for s in block) for block in latent_blocks for offset in range(3)
    )


def decision_horizon(decisions: int) -> int:
    if not isinstance(decisions, int) or isinstance(decisions, bool) or decisions < 0:
        raise ValueError("Decision count must be a nonnegative integer")
    return max(0, 2 * decisions - 1)
