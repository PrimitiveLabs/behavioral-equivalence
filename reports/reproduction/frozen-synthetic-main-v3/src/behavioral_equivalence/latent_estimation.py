"""Point-model estimation from observed adjacent decision-state transitions."""

import numpy as np


def adjacent_transition_counts(
    states, targets, users, bundle_indices, current, following, state_count
):
    state, target, user, bundle, a, b = map(
        np.asarray, (states, targets, users, bundle_indices, current, following)
    )
    if (
        state.ndim != 1
        or any(v.shape != state.shape for v in (target, user, bundle))
        or a.ndim != 1
        or b.shape != a.shape
        or a.dtype.kind not in "iu"
        or b.dtype.kind not in "iu"
        or state.dtype.kind not in "iu"
        or not np.isin(target, [0, 1]).all()
        or not isinstance(state_count, int)
        or state_count < 1
        or (state < 0).any()
        or (state >= state_count).any()
        or (
            len(a)
            and (
                (a < 0).any() or (b < 0).any() or (a >= len(state)).any() or (b >= len(state)).any()
            )
        )
    ):
        raise ValueError("Aligned discrete states, binary targets and valid pair indices required")
    if np.any(user[a] != user[b]) or np.any(bundle[b] != bundle[a] + 1):
        raise ValueError("Transitions require consecutive source bundles from the same user")
    counts = np.zeros((state_count, 2, state_count), dtype=np.int64)
    np.add.at(counts, (state[a], target[a].astype(int), state[b]), 1)
    return counts


def point_transitions(counts: np.ndarray, prior_mass: float = 1.0) -> np.ndarray:
    c = np.asarray(counts)
    if (
        c.ndim != 3
        or c.shape[0] < 1
        or c.shape[1:] != (2, c.shape[0])
        or c.dtype.kind not in "iu"
        or (c < 0).any()
        or not np.isfinite(prior_mass)
        or prior_mass <= 0
    ):
        raise ValueError("Nonnegative K-by-2-by-K counts and a positive nominal prior required")
    weights = c.astype(np.float64) + prior_mass / c.shape[0]
    return weights / weights.sum(axis=2, keepdims=True)


def equal_user_initial_mass(states: np.ndarray, users: np.ndarray, state_count: int):
    s, u = np.asarray(states), np.asarray(users)
    if (
        s.ndim != 1
        or u.shape != s.shape
        or s.dtype.kind not in "iu"
        or (s < 0).any()
        or (s >= state_count).any()
    ):
        raise ValueError("Aligned supported initial states and user IDs required")
    unique, inverse, counts = np.unique(u, return_inverse=True, return_counts=True)
    return np.bincount(s, weights=1 / counts[inverse], minlength=state_count), len(unique)
