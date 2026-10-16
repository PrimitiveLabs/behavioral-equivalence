"""Outcome-independent representatives and genuinely adjacent calibration pairs."""

import numpy as np


def representative_indices(bundle_index: np.ndarray, seed: int) -> np.ndarray:
    index = np.asarray(bundle_index)
    if (
        index.ndim != 1
        or not len(index)
        or index.dtype.kind not in "iu"
        or index[0] != 0
        or np.any(index[1:] < index[:-1])
    ):
        raise ValueError("Sorted nonempty contiguous bundle indices required")
    starts = np.r_[0, np.flatnonzero(index[1:] != index[:-1]) + 1]
    if not np.array_equal(index[starts], np.arange(len(starts))):
        raise ValueError("Every bundle must have at least one exposure")
    counts = np.diff(np.r_[starts, len(index)])
    return starts + np.random.default_rng(seed).integers(0, counts)


def adjacent_origins(
    users: np.ndarray,
    timestamps: np.ndarray,
    roles: np.ndarray,
    per_user_cap: int = 64,
    seed: int = 20261020,
) -> np.ndarray:
    users, times, roles = map(np.asarray, (users, timestamps, roles))
    if (
        users.ndim != 1
        or times.shape != users.shape
        or roles.shape != users.shape
        or users.dtype.kind not in "iu"
        or times.dtype.kind not in "iu"
        or (users < 0).any()
        or (users[1:] < users[:-1]).any()
        or not np.isin(roles, [-1, 0, 1, 2, 3]).all()
        or not isinstance(per_user_cap, int)
        or isinstance(per_user_cap, bool)
        or per_user_cap < 1
    ):
        raise ValueError("Sorted user bundles, aligned roles and positive integer cap required")
    same = users[1:] == users[:-1]
    if np.any(times[1:][same] <= times[:-1][same]):
        raise ValueError("Timestamp bundles must be strictly ordered within each user")
    eligible = np.flatnonzero(same & (roles[:-1] == 2) & (roles[1:] == 2))
    if not len(eligible):
        return np.empty(0, dtype=np.int64)
    selected = []
    candidate_users = users[eligible]
    bounds = np.r_[
        0, np.flatnonzero(candidate_users[1:] != candidate_users[:-1]) + 1, len(eligible)
    ]
    for left, right in zip(bounds[:-1], bounds[1:]):
        candidates = eligible[left:right]
        rng = np.random.default_rng([seed, int(candidate_users[left])])
        chosen = rng.choice(candidates, min(per_user_cap, len(candidates)), replace=False)
        selected.extend(sorted(map(int, chosen)))
    return np.asarray(selected, dtype=np.int64)


def query_history_tokens(
    users, timestamps, available, clicks, exposures, query_indices, embargo_ms
):
    """Build selected queries from complete user histories, never selected-case gaps."""
    from .features import bundle_history_tokens

    users, times, availability, positive, total, query = map(
        np.asarray, (users, timestamps, available, clicks, exposures, query_indices)
    )
    if (
        users.ndim != 1
        or any(x.shape != users.shape for x in (times, availability, positive, total))
        or query.ndim != 1
        or query.dtype.kind not in "iu"
        or (users[1:] < users[:-1]).any()
        or (
            len(query)
            and (query[0] < 0 or query[-1] >= len(users) or np.any(query[1:] <= query[:-1]))
        )
    ):
        raise ValueError("Aligned full histories and sorted distinct valid query indices required")
    result = np.empty((len(query), 32), dtype=np.uint8)
    bounds = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        first, last = np.searchsorted(query, [left, right])
        if first == last:
            continue
        history = bundle_history_tokens(
            times[left:right],
            availability[left:right],
            positive[left:right],
            total[left:right],
            extra_embargo_ms=embargo_ms,
        )
        result[first:last] = history[query[first:last] - left]
    return result
