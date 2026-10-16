"""Response-group holdouts with training-only priors and disjoint calibration."""

from itertools import combinations
import numpy as np
from .response_refinement import validate_latent_partition


def response_group_folds():
    return tuple(
        dict(query=pair, heldout=tuple(g for g in range(4) if g not in pair))
        for pair in combinations(range(4), 2)
    )


def teacher_response_prior(states, environments, teacher, state_count):
    s, e, p = map(np.asarray, (states, environments, teacher))
    if (
        s.ndim != 1
        or not len(s)
        or e.shape != s.shape
        or p.shape != s.shape
        or any(a.dtype.kind not in "iu" for a in (s, e))
        or not isinstance(state_count, int)
        or state_count < 1
        or np.any(s < 0)
        or np.any(s >= state_count)
        or np.any(e < 0)
        or np.any(e >= 75)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
    ):
        raise ValueError(
            "Aligned standard-training state/context and teacher probabilities required"
        )
    n = np.bincount(s.astype(np.int64) * 75 + e, minlength=state_count * 75).reshape(
        state_count, 75
    )
    total = np.bincount(s.astype(np.int64) * 75 + e, weights=p, minlength=state_count * 75).reshape(
        state_count, 75
    )
    en = n.sum(axis=0)
    ep = total.sum(axis=0)
    fallback = np.full(75, float(p.mean()))
    np.divide(ep, en, out=fallback, where=en > 0)
    prior = np.broadcast_to(fallback, n.shape).copy()
    np.divide(total, n, out=prior, where=n > 0)
    return dict(probabilities=prior, counts=n, teacher_sum=total, environment_fallback=fallback)


def query_only_table(table, query_groups):
    if (
        len(query_groups) != 2
        or len(set(query_groups)) != 2
        or not set(query_groups) <= set(range(4))
    ):
        raise ValueError("Two distinct known-duration query groups required")
    counts = np.asarray(table["counts"])
    users = np.asarray(table["users"])
    positives = np.asarray(table["positives"])
    if (
        counts.ndim != 2
        or counts.shape[1] != 75
        or users.shape != counts.shape
        or positives.shape != counts.shape
        or any(a.dtype.kind not in "iu" for a in (counts, users))
        or np.any(counts < 0)
        or np.any(users < 0)
        or np.any(users > counts)
        or not np.isfinite(positives).all()
        or np.any(positives < 0)
        or np.any(positives > counts)
        or np.any(positives != np.floor(positives))
    ):
        raise ValueError("Valid aligned response counts and unique-user counts required")
    visible = np.isin(np.arange(75) % 5, query_groups)
    result = {k: np.where(visible, table[k], 0) for k in ["counts", "users", "positives"]}
    result["probabilities"] = np.full(counts.shape, np.nan)
    np.divide(
        result["positives"],
        result["counts"],
        out=result["probabilities"],
        where=result["counts"] > 0,
    )
    return result


def heldout_point_model(prior, estimation, blocks, query_groups, prior_mass=1.0):
    prior = np.asarray(prior, dtype=float)
    visible = query_only_table(estimation, query_groups)
    if (
        prior.shape != visible["counts"].shape
        or not np.isfinite(prior).all()
        or np.any(prior < 0)
        or np.any(prior > 1)
        or not np.isfinite(prior_mass)
        or prior_mass <= 0
    ):
        raise ValueError("Aligned valid training-only priors and positive prior mass required")
    validate_latent_partition(blocks, len(prior))
    weights = visible["counts"] + prior_mass
    parent = (visible["positives"] + prior_mass * prior) / weights
    abstract = np.stack(
        [np.average(parent[list(b)], axis=0, weights=weights[list(b)]) for b in blocks]
    )
    mapping = np.empty(len(prior), dtype=np.int64)
    for i, b in enumerate(blocks):
        mapping[list(b)] = i
    return dict(
        parent=parent,
        abstract=abstract,
        mapping=mapping,
        maximum_point_discrepancy=float(np.abs(parent - abstract[mapping]).max()),
    )
