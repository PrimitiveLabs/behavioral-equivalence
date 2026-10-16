"""Response-free selection and explicit exposure groups for observed random logs."""

import numpy as np
from .intrinsic_context import duration_context


def randomized_role_indices(users, timestamps, case_ids, is_random, roles, user_roles, protocol):
    """Keep all random exposures; role restrictions are supplied by the frozen calendar.

    The random flag defines the observed-response cohort, not a known impression
    propensity. No response values, future support, or simulator predictions enter.
    """
    u, t, ids, flag, role, ur = map(
        np.asarray, (users, timestamps, case_ids, is_random, roles, user_roles)
    )
    if (
        u.ndim != 1
        or any(x.shape != u.shape for x in (t, ids, flag, role, ur))
        or any(x.dtype.kind not in "iu" for x in (u, t, ids, role, ur))
        or np.any(u < 0)
        or np.any(u[1:] < u[:-1])
        or len(np.unique(ids)) != len(ids)
        or not np.isin(flag, [0, 1]).all()
        or not np.isin(role, [-1, 0, 1, 2, 3]).all()
        or not np.isin(ur, [0, 1, 2, 3]).all()
        or protocol not in {"known_users", "unseen_users"}
    ):
        raise ValueError(
            "Aligned canonical exposure identities, flags, calendar and user roles required"
        )
    same = u[1:] == u[:-1]
    if np.any(t[1:][same] < t[:-1][same]) or np.any(ur[1:][same] != ur[:-1][same]):
        raise ValueError("Chronological exposures and one fixed role per user required")
    eligible = flag == 1
    if protocol == "unseen_users":
        eligible &= ur == role
    return {
        name: np.flatnonzero(eligible & (role == code))
        for code, name in enumerate(("train", "validation", "calibration", "test"))
    }


def exposure_environment(tab, duration_ms, upload_day, exposure_ms):
    """Five duration groups crossed with fifteen recorded scenarios, including unknown."""
    tab = np.asarray(tab)
    _, known = duration_context(duration_ms, upload_day, exposure_ms)
    duration = np.asarray(duration_ms)
    if (
        tab.shape != duration.shape
        or tab.dtype.kind not in "iu"
        or np.any(tab < 0)
        or np.any(tab >= 15)
    ):
        raise ValueError("Aligned scenario indices from zero through fourteen required")
    group = np.full(len(duration), 4, dtype=np.int8)
    group[known] = np.searchsorted([7000, 18000, 60000], duration[known], side="left")
    return dict(
        duration_group=group,
        environment=(tab.astype(np.int16) * 5 + group),
        intrinsic_duration_known=known,
    )
