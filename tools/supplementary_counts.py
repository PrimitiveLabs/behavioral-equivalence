"""Capped-user sufficient statistics for supplementary endpoint emissions."""

import numpy as np

CLASSES = (2, 2, 2, 2, 2, 4)


def endpoint_counts(states, targets, users, case_ids, maximum_per_user=64):
    y, u, ids = map(np.asarray, (targets, users, case_ids))
    if (
        y.ndim != 2
        or y.shape[1] != 6
        or y.dtype.kind not in "iu"
        or u.shape != (len(y),)
        or ids.shape != u.shape
        or u.dtype.kind not in "iu"
        or ids.dtype.kind not in "iu"
        or np.any(u < 0)
        or np.any(ids < 0)
        or len(np.unique(ids)) != len(ids)
        or type(maximum_per_user) is not int
        or maximum_per_user < 1
    ):
        raise ValueError(
            "Distinct aligned capped-user origin IDs and six integer endpoints required"
        )
    if any(not np.isin(y[:, j], range(-1, c)).all() for j, c in enumerate(CLASSES)):
        raise ValueError("Endpoint labels must be missing or valid class indices")
    unique, frequency = np.unique(u, return_counts=True)
    if np.any(frequency > maximum_per_user):
        raise ValueError("Outcome-blind per-user origin cap exceeded")
    totals = {}
    for k, z in states.items():
        z = np.asarray(z)
        if (
            type(k) is not int
            or k < 1
            or z.shape != u.shape
            or z.dtype.kind not in "iu"
            or np.any(z < 0)
            or np.any(z >= k)
        ):
            raise ValueError("Valid aligned predictive state assignments required")
        for j, c in enumerate(CLASSES):
            observed = y[:, j] >= 0
            totals[(k, j)] = np.bincount(
                z[observed].astype(np.int64) * c + y[observed, j].astype(np.int64),
                minlength=k * c,
            ).reshape(k, c)
    return totals, unique, int(frequency.max(initial=0))


def smoothed_emissions(counts, prior_mass=1.0):
    a = np.asarray(counts)
    if (
        a.ndim != 2
        or not len(a)
        or a.shape[1] not in [2, 4]
        or a.dtype.kind not in "iu"
        or np.any(a < 0)
        or not np.isfinite(prior_mass)
        or prior_mass <= 0
    ):
        raise ValueError("Integer endpoint counts and positive finite prior mass required")
    p = a.astype(float) + prior_mass / a.shape[1]
    return p / p.sum(axis=1, keepdims=True)
