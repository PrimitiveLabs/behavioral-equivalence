"""Shared finite-event target order and origin-only iid reference forecasts."""

import numpy as np
from scipy.stats import binom
from .nested_design import human_event_suite


def horizon_event_targets(horizons, observed, positive_counts, last_positive):
    """Use the same event order as the certified pending-state event suite.

    Missing observed horizons remain sentinel-1 for every event. In particular,
    an unavailable future cannot become a negative training or evaluation label.
    """
    events = human_event_suite(tuple(horizons))
    mask, count, last = map(np.asarray, (observed, positive_counts, last_positive))
    if (
        mask.ndim != 2
        or mask.shape[1] != len(horizons)
        or mask.dtype.kind != "b"
        or count.shape != mask.shape
        or last.shape != mask.shape
        or count.dtype.kind not in "iu"
        or last.dtype.kind not in "iu"
        or np.any(count[~mask] != -1)
        or np.any(last[~mask] != -1)
        or not np.isin(last[mask], [0, 1]).all()
    ):
        raise ValueError(
            "Aligned observed-horizon masks, integer counts and sentinel outcomes required"
        )
    for j, h in enumerate(horizons):
        take = mask[:, j]
        c, y = count[take, j], last[take, j]
        if (
            np.any(c < 0)
            or np.any(c > h)
            or np.any((c == 0) & (y != 0))
            or np.any((c == h) & (y != 1))
        ):
            raise ValueError(
                "Recorded count and terminal response contradict their decision horizon"
            )
    columns = []
    for mode in ["occupancy", "positive", "negative", "two", "four"]:
        for j, h in enumerate(horizons):
            value = (
                last[:, j]
                if mode == "occupancy"
                else count[:, j] >= 1
                if mode == "positive"
                else count[:, j] < h
                if mode == "negative"
                else count[:, j] >= (2 if mode == "two" else 4)
            )
            columns.append(np.where(mask[:, j], value, -1))
    return np.column_stack(columns).astype(np.int8), events


def origin_iid_event_probabilities(probabilities, horizons):
    """Weak reference: hold the origin's probability fixed for H iid responses.

    This uses no future context and is not a learned sequence rollout. Direct
    horizon predictors and finite-state dynamics are separate comparators.
    """
    events = human_event_suite(tuple(horizons))
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Finite origin response probabilities required")
    columns = []
    for event in events:
        h = (event.horizon + 1) // 2
        if event.mode == "occupancy":
            value = p
        else:
            value = binom.sf(event.threshold - 1, h, p if event.targets == ("positive",) else 1 - p)
        columns.append(value)
    return np.column_stack(columns)
