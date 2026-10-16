"""Prospective shared-calendar roles with explicit feedback-availability guards."""

from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np


def utc_ms(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp() * 1000)


@dataclass(frozen=True)
class EvaluationCalendar:
    train_end: int
    validation_end: int
    calibration_end: int

    def __post_init__(self):
        if (
            any(
                not isinstance(t, int) or isinstance(t, bool)
                for t in (self.train_end, self.validation_end, self.calibration_end)
            )
            or not self.train_end < self.validation_end < self.calibration_end
        ):
            raise ValueError("Strictly ordered integer calendar cutoffs required")

    def role(
        self, timestamp: int, protocol: str, user_role: str, available_ms: int, embargo_ms: int
    ) -> str | None:
        if protocol not in {"known_users", "unseen_users"} or user_role not in {
            "train",
            "validation",
            "calibration",
            "test",
        }:
            raise ValueError("Known evaluation protocol and user role required")
        if (
            not isinstance(timestamp, int)
            or not isinstance(available_ms, int)
            or available_ms < timestamp
            or not isinstance(embargo_ms, int)
            or embargo_ms < 0
        ):
            raise ValueError(
                "Integer timestamps, causal availability proxy and nonnegative embargo required"
            )
        role = (
            "train"
            if timestamp < self.train_end
            else "validation"
            if timestamp < self.validation_end
            else "calibration"
            if timestamp < self.calibration_end
            else "test"
        )
        if protocol == "unseen_users" and user_role != role:
            return None
        boundary = {
            "train": self.train_end,
            "validation": self.validation_end,
            "calibration": self.calibration_end,
        }.get(role)
        if boundary is not None and available_ms + embargo_ms > boundary:
            return None
        return role


CONFIRMATORY_CANDIDATE = EvaluationCalendar(
    utc_ms("2022-04-29"), utc_ms("2022-05-02"), utc_ms("2022-05-05")
)

ROLE_NAMES = ("train", "validation", "calibration", "test")


def calendar_roles(
    calendar: EvaluationCalendar,
    timestamps: np.ndarray,
    availability: np.ndarray,
    user_roles: np.ndarray,
    protocol: str,
    embargo_ms: int,
) -> np.ndarray:
    """Return -1 for excluded cases, otherwise the index in ROLE_NAMES.

    Availability and timestamps are int64 milliseconds. This batch operation
    implements the scalar calendar rule without consulting behavioral targets.
    """
    t, a, u = np.asarray(timestamps), np.asarray(availability), np.asarray(user_roles)
    if (
        t.ndim != 1
        or a.shape != t.shape
        or u.shape != t.shape
        or t.dtype != np.dtype("int64")
        or a.dtype != np.dtype("int64")
        or np.any(a < t)
        or not np.isin(u, ROLE_NAMES).all()
    ):
        raise ValueError("Aligned int64 times, causal availability and known user roles required")
    if (
        protocol not in {"known_users", "unseen_users"}
        or not isinstance(embargo_ms, int)
        or isinstance(embargo_ms, bool)
        or not 0 <= embargo_ms <= np.iinfo(np.int64).max
    ):
        raise ValueError("Known protocol and nonnegative int64 embargo required")
    cutoffs = np.array([calendar.train_end, calendar.validation_end, calendar.calibration_end])
    roles = np.searchsorted(cutoffs, t, side="right").astype(np.int8)
    # Subtract the embargo from a boundary rather than adding it to a timestamp.
    # Python integers keep the comparison defined even at int64 boundaries.
    for r, cutoff in enumerate(cutoffs):
        roles[(roles == r) & (a > int(cutoff) - embargo_ms)] = -1
    if protocol == "unseen_users":
        for r, name in enumerate(ROLE_NAMES):
            roles[(roles == r) & (u != name)] = -1
    return roles
