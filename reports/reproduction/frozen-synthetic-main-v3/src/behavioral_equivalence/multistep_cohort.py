"""Outcome-blind origins and explicit observed-horizon cohorts for human tests."""

import numpy as np


def _bundles(users, timestamps, roles):
    u, t, r = map(np.asarray, (users, timestamps, roles))
    if (
        u.ndim != 1
        or t.shape != u.shape
        or r.shape != u.shape
        or u.dtype.kind not in "iu"
        or t.dtype.kind not in "iu"
        or (u < 0).any()
        or (u[1:] < u[:-1]).any()
        or not np.isin(r, [-1, 0, 1, 2, 3]).all()
    ):
        raise ValueError("Sorted aligned user bundles, integer timestamps and known roles required")
    same = u[1:] == u[:-1]
    if np.any(t[1:][same] <= t[:-1][same]):
        raise ValueError("Strictly increasing timestamp bundles required within each user")
    return u, t, r


def select_test_origins(users, timestamps, roles, per_user_cap=64, seed=20261021):
    """Select origins before inspecting any future support or outcomes."""
    u, _, r = _bundles(users, timestamps, roles)
    if not isinstance(per_user_cap, int) or isinstance(per_user_cap, bool) or per_user_cap < 1:
        raise ValueError("Positive integer per-user cap required")
    candidates = np.flatnonzero(r == 3)
    if not len(candidates):
        return np.empty(0, dtype=np.int64)
    groups = u[candidates]
    boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    selected = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = candidates[left:right]
        rng = np.random.default_rng([seed, int(groups[left])])
        selected.extend(
            sorted(map(int, rng.choice(indices, min(len(indices), per_user_cap), replace=False)))
        )
    return np.array(selected, dtype=np.int64)


def observed_horizon_cohorts(users, timestamps, roles, origins, horizons):
    """Keep censored origins in the support denominator; never cross role/user gaps."""
    u, _, r = _bundles(users, timestamps, roles)
    origin = np.asarray(origins)
    if (
        origin.ndim != 1
        or origin.dtype.kind not in "iu"
        or (
            len(origin)
            and (
                (origin < 0).any() or (origin >= len(u)).any() or np.any(origin[1:] <= origin[:-1])
            )
        )
    ):
        raise ValueError("Sorted distinct valid origin indices required")
    if (
        any(not isinstance(h, int) or isinstance(h, bool) or h < 1 for h in horizons)
        or len(set(horizons)) != len(horizons)
        or not len(horizons)
    ):
        raise ValueError("Distinct positive integer decision horizons required")
    if len(origin) and np.any(r[origin] != 3):
        raise ValueError("Origins must belong to the test role")
    # Prefix sums identify intervening excluded bundles even when endpoints match.
    excluded = np.r_[0, np.cumsum(r != 3)]
    result = []
    for h in horizons:
        end = origin + h - 1
        inside = end < len(u)
        same_user = np.zeros(len(origin), dtype=bool)
        same_user[inside] = u[end[inside]] == u[origin[inside]]
        observed = inside & same_user
        positions = np.flatnonzero(observed)
        observed[positions] &= (excluded[end[positions] + 1] - excluded[origin[positions]]) == 0
        result.append(
            dict(
                horizon=h,
                origin_indices=origin.copy(),
                observed_mask=observed,
                end_indices=np.where(observed, end, -1),
                selected_origins=len(origin),
                observed_origins=int(observed.sum()),
                censored_origins=int((~observed).sum()),
                scope="conditional on H observed consecutive test bundles; no unconditional survival or abandonment claim",
            )
        )
    return result


def recorded_horizon_targets(outcomes, cohort):
    """Score frozen one-exposure-per-bundle representatives, including the origin."""
    y = np.asarray(outcomes)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Binary representative outcome vector required")
    origin = cohort["origin_indices"]
    end = cohort["end_indices"]
    observed = cohort["observed_mask"]
    if len(origin) != len(end) or len(origin) != len(observed) or np.any(end[observed] >= len(y)):
        raise ValueError("Aligned cohort and complete outcome vector required")
    prefix = np.r_[0, np.cumsum(y, dtype=np.int64)]
    counts = np.full(len(origin), -1, dtype=np.int64)
    counts[observed] = prefix[end[observed] + 1] - prefix[origin[observed]]
    last = np.full(len(origin), -1, dtype=np.int8)
    last[observed] = y[end[observed]]
    return dict(
        positive_count=counts, last_positive=last, observed_mask=observed.copy(), missing_value=-1
    )
