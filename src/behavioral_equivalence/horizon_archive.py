"""Lossless factorized storage for reproducible marginal horizon forecasts."""

import numpy as np
from .horizon_targets import origin_iid_event_probabilities


def restore_horizon_forecasts(arrays, finite_tables, horizons):
    """Rebuild all forecasts from direct arrays, iid origins and finite states.

    Arrays include private case/user/target data, which callers keep on cloud.
    Factorization changes storage only, not predictive precision or scoring.
    """
    targets = np.asarray(arrays["targets"])
    state = np.asarray(arrays["state_k256"])
    columns = 5 * len(horizons)
    if (
        targets.ndim != 2
        or targets.shape[1] != columns
        or state.shape != (len(targets),)
        or state.dtype.kind not in "iu"
        or np.any(state < 0)
    ):
        raise ValueError("Aligned horizon target rows and finite parent states required")
    result = {}
    for key, value in arrays.items():
        if key.startswith("direct."):
            name = key.removeprefix("direct.") + ".direct"
            result[name] = np.asarray(value)
        elif key.startswith("iid_origin."):
            name = key.removeprefix("iid_origin.")
            result[name] = origin_iid_event_probabilities(value, horizons)
    for name, table in finite_tables.items():
        table = np.asarray(table)
        if table.ndim != 2 or table.shape[1] != columns or np.any(state >= len(table)):
            raise ValueError("Finite table must cover every stored parent state")
        if name in result:
            raise ValueError("Duplicate forecast name across storage classes")
        result[name] = table[state]
    if not result or any(
        p.shape != targets.shape or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
        for p in result.values()
    ):
        raise ValueError("Finite aligned reconstructed probability arrays required")
    return result
