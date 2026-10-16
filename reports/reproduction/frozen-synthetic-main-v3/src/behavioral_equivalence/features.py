"""History features that preserve unordered exposure bundles and delayed feedback."""

from bisect import insort
import numpy as np
from numpy.typing import NDArray


def bundle_history_tokens(
    timestamps: NDArray,
    available: NDArray,
    clicks: NDArray,
    exposures: NDArray,
    history: int = 32,
    extra_embargo_ms: int = 0,
) -> NDArray[np.uint8]:
    """Encode one user's completed bundles, strictly before each query timestamp.

    Tokens 1..16 encode preceding bundle outcome rate; 17 is a start marker,
    zero is padding. Current-bundle outcomes never enter its own feature row.
    This operates on observations under a declared availability assumption.
    """
    n = len(timestamps)
    if any(len(x) != n for x in (available, clicks, exposures)) or history < 1:
        raise ValueError("Bundle arrays must align and history must be positive")
    if extra_embargo_ms < 0 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Strictly ordered timestamps and nonnegative embargo required")
    if np.any(available < timestamps) or np.any(exposures <= 0):
        raise ValueError("Invalid observation timing or exposure counts")
    if np.any(clicks < 0) or np.any(clicks > exposures):
        raise ValueError("Invalid bundle outcome count")
    encoded = 1 + np.minimum(15, np.floor(16 * clicks / exposures).astype(np.uint8))
    result = np.zeros((n, history), dtype=np.uint8)
    order = np.argsort(available, kind="stable")
    completed: list[int] = []
    cursor = 0
    for query, timestamp in enumerate(timestamps):
        while cursor < n and int(available[order[cursor]]) + extra_embargo_ms < int(timestamp):
            index = int(order[cursor])
            # available >= timestamp means the current/future bundles cannot enter.
            insort(completed, index)
            cursor += 1
        chosen = completed[-history:]
        if chosen:
            result[query, -len(chosen) :] = encoded[chosen]
        else:
            result[query, -1] = 17
    return result
