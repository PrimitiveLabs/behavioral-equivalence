"""Fixed held-out-condition replicas with original case identity preserved."""

import numpy as np
from .heldout_response import response_group_folds


def response_replicas(groups, forecasts):
    """Take the three held-out predictions for every known-duration case."""
    g = np.asarray(groups)
    folds = response_group_folds()
    names = ["q" + "".join(map(str, f["query"])) for f in folds]
    if (
        g.ndim != 1
        or g.dtype.kind not in "iu"
        or np.any(g < 0)
        or np.any(g > 3)
        or set(forecasts) != set(names)
    ):
        raise ValueError("All six fixed folds and known-duration case groups required")
    arrays = {name: np.asarray(forecasts[name]) for name in names}
    if any(
        p.shape != g.shape or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
        for p in arrays.values()
    ):
        raise ValueError("Aligned finite fold probabilities required")
    result = np.empty((len(g), 3))
    written = np.zeros(len(g), dtype=int)
    for name, fold in zip(names, folds, strict=True):
        ix = np.flatnonzero(np.isin(g, fold["heldout"]))
        result[ix, written[ix]] = arrays[name][ix]
        written[ix] += 1
    if not np.all(written == 3):
        raise AssertionError("Every known group must be withheld by exactly three folds")
    return result
