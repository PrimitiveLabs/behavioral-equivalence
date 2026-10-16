"""Leakage-aware prediction cases and reproducible evaluation splits."""

from dataclasses import dataclass
import hashlib
from typing import Literal, Sequence

Split = Literal["train", "validation", "calibration", "test"]
SPLITS: tuple[Split, ...] = ("train", "validation", "calibration", "test")


@dataclass(frozen=True)
class Interaction:
    case_id: str
    user_id: str
    timestamp_ms: int
    click: int
    context_available_ms: int

    def __post_init__(self) -> None:
        if self.click not in (0, 1) or not self.case_id or not self.user_id:
            raise ValueError("Interaction needs valid identity and binary click")
        if self.context_available_ms > self.timestamp_ms:
            raise ValueError("Context contains information from after prediction time")


@dataclass(frozen=True)
class PredictionCase:
    event: Interaction
    history: tuple[Interaction, ...]
    split: Split

    def __post_init__(self) -> None:
        if any(
            x.user_id != self.event.user_id or x.timestamp_ms >= self.event.timestamp_ms
            for x in self.history
        ):
            raise ValueError("History must precede target and belong to the same user")
        if any(a.timestamp_ms >= b.timestamp_ms for a, b in zip(self.history, self.history[1:])):
            raise ValueError("History must be strictly chronological")
        if self.split not in SPLITS:
            raise ValueError("Unknown split")


def _group(events: Sequence[Interaction]) -> dict[str, list[Interaction]]:
    if len({e.case_id for e in events}) != len(events):
        raise ValueError("Duplicate prediction case ID")
    users: dict[str, list[Interaction]] = {}
    for event in events:
        users.setdefault(event.user_id, []).append(event)
    for sequence in users.values():
        sequence.sort(key=lambda event: event.timestamp_ms)
        if any(a.timestamp_ms == b.timestamp_ms for a, b in zip(sequence, sequence[1:])):
            raise ValueError("Ambiguous timestamp ties require an upstream ordering policy")
    return users


def chronological_cases(
    events: Sequence[Interaction], max_history: int = 50, horizon: int = 1
) -> list[PredictionCase]:
    """70/10/10/10 within-user split, with horizon-crossing origins embargoed.

    Target is the event itself. horizon>1 reserves future targets through that
    many events; the present case retains only the next-click target. Sessions
    too short for four supported splits are excluded explicitly.
    """
    if max_history < 1 or horizon < 1:
        raise ValueError("Positive history and horizon required")
    result = []
    for sequence in _group(events).values():
        n = len(sequence)
        if n < 20:
            continue
        boundaries = [0, int(0.7 * n), int(0.8 * n), int(0.9 * n), n]
        for split, start, end in zip(SPLITS, boundaries, boundaries[1:]):
            for i in range(max(1, start), end - horizon + 1):
                result.append(
                    PredictionCase(sequence[i], tuple(sequence[max(0, i - max_history) : i]), split)
                )
    return result


def disjoint_user_cases(
    events: Sequence[Interaction], seed: int = 0, max_history: int = 50
) -> list[PredictionCase]:
    """User-disjoint splits. Prior test-user observations become available online.

    The sequence model never trains on those observations. Cold-start cases
    with empty history remain in evaluation and are reported separately.
    """
    if max_history < 1:
        raise ValueError("Positive history length required")
    users = _group(events)
    if len(users) < 10:
        raise ValueError("At least ten users required for supported disjoint splits")
    order = sorted(users, key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).digest())
    n = len(order)
    boundaries = [0, int(0.7 * n), int(0.8 * n), int(0.9 * n), n]
    result = []
    for split, start, end in zip(SPLITS, boundaries, boundaries[1:]):
        for user in order[start:end]:
            sequence = users[user]
            for i, event in enumerate(sequence):
                result.append(
                    PredictionCase(event, tuple(sequence[max(0, i - max_history) : i]), split)
                )
    return result


def audit_cases(
    cases: Sequence[PredictionCase], protocol: Literal["chronological", "users"]
) -> dict:
    if protocol not in ("chronological", "users"):
        raise ValueError("Unknown split protocol")
    if len({c.event.case_id for c in cases}) != len(cases):
        raise ValueError("Prediction target appears in multiple cases")
    by_user: dict[str, list[PredictionCase]] = {}
    counts = {split: 0 for split in SPLITS}
    for case in cases:
        # Revalidate externally constructed objects at the audit boundary.
        PredictionCase(case.event, case.history, case.split)
        by_user.setdefault(case.event.user_id, []).append(case)
        counts[case.split] += 1
    for user_cases in by_user.values():
        if protocol == "users" and len({c.split for c in user_cases}) > 1:
            raise ValueError("User appears in disjoint groups")
        if protocol == "chronological":
            ordered = sorted(user_cases, key=lambda c: c.event.timestamp_ms)
            ranks = [SPLITS.index(c.split) for c in ordered]
            if ranks != sorted(ranks):
                raise ValueError("Split boundary leaks future targets into training")
    if any(count == 0 for count in counts.values()):
        raise ValueError("Every split must contain prediction targets")
    return {"protocol": protocol, "cases": len(cases), "users": len(by_user), "counts": counts}


@dataclass(frozen=True)
class TimestampBundle:
    user_id: str
    timestamp_ms: int
    available_ms: int
    outcomes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.user_id or self.available_ms < self.timestamp_ms or not self.outcomes:
            raise ValueError("Bundle needs identity, outcomes and valid availability")
        if any(y not in (0, 1) for y in self.outcomes):
            raise ValueError("Bundle outcomes must be binary")


def available_bundle_history(
    bundles: Sequence[TimestampBundle], user_id: str, prediction_ms: int, max_history: int = 50
) -> tuple[TimestampBundle, ...]:
    """Keep the timestamp group unordered and embargo outcomes not yet observed."""
    if max_history < 1:
        raise ValueError("Positive history length required")
    selected = [
        b
        for b in bundles
        if b.user_id == user_id
        and b.timestamp_ms < prediction_ms
        and b.available_ms < prediction_ms
    ]
    selected.sort(key=lambda b: b.timestamp_ms)
    if len({b.timestamp_ms for b in selected}) != len(selected):
        raise ValueError("Combine same-timestamp observations into one unordered bundle")
    return tuple(selected[-max_history:])
