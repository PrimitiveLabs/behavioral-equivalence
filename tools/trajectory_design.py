"""Outcome-blind trajectory sampling and decision-lift extraction."""

import hashlib
import numpy as np


def sampled_case_ids(case_ids, limit, seed):
    ids = np.asarray(case_ids)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Distinct nonnegative integer case IDs required")
    if type(limit) is not int or limit < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive sample limit and nonnegative integer seed required")
    ranks = [hashlib.sha256(f"trajectory-v1:{seed}:{int(i)}".encode()).digest() for i in ids]
    chosen = sorted(range(len(ids)), key=lambda j: (ranks[j], int(ids[j])))[:limit]
    return np.sort(ids[chosen])


def decision_arrays(transition_matrix):
    p = np.asarray(transition_matrix, dtype=float)
    n = len(p)
    if (
        p.shape != (n, n)
        or n == 0
        or n % 3
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Stochastic three-stage decision lift required")
    k = n // 3
    allowed = np.zeros_like(p, dtype=bool)
    for s in range(k):
        allowed[3 * s, 3 * s + 1 : 3 * s + 3] = True
        allowed[3 * s + 1 : 3 * s + 3, ::3] = True
    if np.any(p[~allowed] != 0):
        raise ValueError("Decision lift has unsupported cross-stage edges")
    e = np.column_stack([p[::3, 1::3].sum(axis=1), p[::3, 2::3].sum(axis=1)])
    t = np.stack([p[1::3, ::3], p[2::3, ::3]], axis=1)
    return t, e


def user_sequence_losses(users, losses):
    u, values = np.asarray(users), np.asarray(losses, dtype=float)
    if (
        u.ndim != 1
        or values.shape != u.shape
        or u.dtype.kind not in "iu"
        or np.any(u < 0)
        or np.isnan(values).any()
        or np.any(values < 0)
    ):
        raise ValueError("Original user identities and nonnegative sequence losses required")
    unique, inverse, counts = np.unique(u, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=values, minlength=len(unique)) / counts
    return unique, counts, means
