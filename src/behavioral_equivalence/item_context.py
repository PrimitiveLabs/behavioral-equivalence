"""Training-only vocabulary for current exposed-item identity."""

import numpy as np


def fit_item_vocabulary(items: np.ndarray, maximum: int = 65535) -> np.ndarray:
    items = np.asarray(items)
    if items.ndim != 1 or items.dtype.kind not in "iu" or np.any(items < 0) or not len(items):
        raise ValueError("Nonempty nonnegative integer training item IDs required")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ValueError("Positive vocabulary size required")
    unique, counts = np.unique(items, return_counts=True)
    order = np.lexsort((unique, -counts))[:maximum]
    return np.sort(unique[order])


def encode_items(vocabulary: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Known training items receive indices 1..V; all others share index zero."""
    v, x = np.asarray(vocabulary), np.asarray(items)
    if (
        v.ndim != 1
        or x.ndim != 1
        or v.dtype.kind not in "iu"
        or x.dtype.kind not in "iu"
        or np.any(v < 0)
        or np.any(x < 0)
        or np.any(v[1:] <= v[:-1])
    ):
        raise ValueError("Sorted unique vocabulary and nonnegative integer item IDs required")
    if not len(v):
        return np.zeros(len(x), dtype=np.int64)
    v, x = v.astype(np.uint64, copy=False), x.astype(np.uint64, copy=False)
    positions = np.searchsorted(v, x)
    known = positions < len(v)
    known[known] &= v[positions[known]] == x[known]
    return np.where(known, positions + 1, 0).astype(np.int64)
