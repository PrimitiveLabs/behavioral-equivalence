"""Outcome-free support checks for training-defined nested history resolutions."""

import numpy as np
from .history_state import initial_predictive_blocks


def resolution_map(training_means: np.ndarray, count: int) -> np.ndarray:
    """Coarsen parent states by standard-training teacher rank, without new fitting."""
    blocks = initial_predictive_blocks(training_means, count)
    mapping = np.empty(len(training_means), dtype=np.int64)
    for index, block in enumerate(blocks):
        mapping[list(block)] = index
    return mapping


def cell_support(states, environments, users, count: int) -> dict:
    s, e, u = map(np.asarray, (states, environments, users))
    if (
        not isinstance(count, int)
        or count < 1
        or s.ndim != 1
        or e.shape != s.shape
        or u.shape != s.shape
        or any(a.dtype.kind not in "iu" for a in (s, e, u))
        or np.any(s < 0)
        or np.any(s >= count)
        or np.any(e < 0)
        or np.any(e >= 75)
        or np.any(u < 0)
    ):
        raise ValueError("Aligned valid state, environment and user identifiers required")
    cells = s.astype(np.int64) * 75 + e.astype(np.int64)
    n = np.bincount(cells, minlength=count * 75).reshape(count, 75)
    pairs = np.unique(np.rec.fromarrays([cells, u], names=("cell", "user")))
    distinct = np.bincount(pairs["cell"], minlength=count * 75).reshape(count, 75)
    return dict(counts=n, users=distinct)


def supported_pair_counts(
    counts: np.ndarray, users: np.ndarray, *, min_cases=2000, min_users=50
) -> dict:
    n, u = np.asarray(counts), np.asarray(users)
    if (
        n.ndim != 2
        or n.shape[1] != 75
        or u.shape != n.shape
        or len(n) % 4
        or not len(n)
        or any(a.dtype.kind not in "iu" for a in (n, u))
        or np.any(n < 0)
        or np.any(u < 0)
        or np.any(u > n)
        or min_cases < 1
        or min_users < 1
    ):
        raise ValueError("Valid support tables and groups of four rank-ordered states required")
    valid = (n >= min_cases) & (u >= min_users)
    valid[:, 4::5] = False
    per_block = valid.reshape(len(n) // 4, 4, 75).sum(axis=1)
    return dict(
        supported_state_environment_cells=int(valid.sum()),
        supported_within_initial_block_pairs=int((per_block * (per_block - 1) // 2).sum()),
    )
