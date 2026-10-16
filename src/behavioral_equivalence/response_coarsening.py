"""Bind coarse observed-response tables to independently counted user support."""

import numpy as np


def coarsen_response_table(fine_counts, fine_positives, mapping, coarse_counts, coarse_users):
    n, y, m, expected, users = map(
        np.asarray, (fine_counts, fine_positives, mapping, coarse_counts, coarse_users)
    )
    if (
        n.ndim != 2
        or n.shape[1] != 75
        or y.shape != n.shape
        or m.shape != (len(n),)
        or expected.ndim != 2
        or expected.shape[1] != 75
        or users.shape != expected.shape
        or any(x.dtype.kind not in "iu" for x in (n, m, expected, users))
        or np.any(n < 0)
        or not np.isfinite(y).all()
        or np.any(y < 0)
        or np.any(y > n)
        or np.any(y != np.floor(y))
        or np.any(m < 0)
        or np.any(m >= len(expected))
        or set(m.tolist()) != set(range(len(expected)))
        or np.any(users < 0)
        or np.any(users > expected)
    ):
        raise ValueError(
            "Aligned complete state map, binary counts and independently counted coarse user support required"
        )
    counts = np.zeros_like(expected)
    positives = np.zeros(expected.shape, dtype=float)
    np.add.at(counts, m, n)
    np.add.at(positives, m, y)
    if not np.array_equal(counts, expected):
        raise ValueError("Response and outcome-free support cohorts differ")
    probabilities = np.full(expected.shape, np.nan)
    np.divide(positives, counts, out=probabilities, where=counts > 0)
    return dict(counts=counts, positives=positives, users=users.copy(), probabilities=probabilities)
