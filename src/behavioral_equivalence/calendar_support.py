"""Outcome-free support diagnostics for prospective human evaluation roles."""

import numpy as np
from .calendar_splits import EvaluationCalendar, ROLE_NAMES, calendar_roles


def available_history_counts(
    users: np.ndarray, times: np.ndarray, availability: np.ndarray, embargo_ms: int
) -> np.ndarray:
    """Count completed earlier bundles per query, keeping same-time ties excluded."""
    if (
        users.ndim != 1
        or times.shape != users.shape
        or availability.shape != users.shape
        or times.dtype != np.dtype("int64")
        or availability.dtype != np.dtype("int64")
        or np.any(availability < times)
        or not isinstance(embargo_ms, int)
        or isinstance(embargo_ms, bool)
        or embargo_ms < 0
    ):
        raise ValueError("Aligned integer bundle timestamps and nonnegative embargo required")
    if len(users) == 0:
        return np.array([], dtype=np.int64)
    if np.any(users[1:] < users[:-1]) or np.any(
        (users[1:] == users[:-1]) & (times[1:] <= times[:-1])
    ):
        raise ValueError("Bundles must be sorted by user and strictly increasing time")
    if int(availability.max()) + embargo_ms > np.iinfo(np.int64).max:
        raise ValueError("Embargo exceeds int64 timestamp range")
    boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    result = np.empty(len(users), dtype=np.int64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        completed_times = np.sort(availability[left:right] + embargo_ms)
        result[left:right] = np.searchsorted(completed_times, times[left:right], side="left")
    return result


def support_by_role(
    calendar: EvaluationCalendar,
    users: np.ndarray,
    times: np.ndarray,
    availability: np.ndarray,
    exposures: np.ndarray,
    user_roles: np.ndarray,
    protocol: str,
    embargo_ms: int,
) -> dict:
    """Aggregate availability and case support without using behavioral targets."""
    if exposures.shape != times.shape or exposures.dtype.kind not in "iu" or np.any(exposures <= 0):
        raise ValueError("Positive integer exposure counts aligned with bundles required")
    roles = calendar_roles(calendar, times, availability, user_roles, protocol, embargo_ms)
    history = available_history_counts(users, times, availability, embargo_ms)
    records = []
    for r, name in enumerate(ROLE_NAMES):
        selected = roles == r
        history_counts = [
            int(exposures[selected & mask].sum())
            for mask in [
                history == 0,
                (history >= 1) & (history < 8),
                (history >= 8) & (history < 32),
                history >= 32,
            ]
        ]
        records.append(
            {
                "role": name,
                "bundles": int(selected.sum()),
                "cases": int(exposures[selected].sum()),
                "users": int(len(np.unique(users[selected]))),
                "first_timestamp_ms": int(times[selected].min()) if selected.any() else None,
                "last_timestamp_ms": int(times[selected].max()) if selected.any() else None,
                "latest_label_available_plus_embargo_ms": int(availability[selected].max())
                + embargo_ms
                if selected.any()
                else None,
                "cases_by_available_history_0_1to7_8to31_32plus": history_counts,
            }
        )
    return {
        "protocol": protocol,
        "embargo_ms": embargo_ms,
        "roles": records,
        "excluded_bundles": int((roles < 0).sum()),
        "excluded_cases": int(exposures[roles < 0].sum()),
    }
