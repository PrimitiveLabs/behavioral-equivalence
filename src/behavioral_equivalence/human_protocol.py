"""Outcome-free identity exclusion and deterministic confirmatory user splits."""

import hashlib
import numpy as np

SOURCE_COLUMNS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
)
BINARY_COLUMNS = frozenset(
    {
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "is_rand",
    }
)


def identity_fingerprints(users: np.ndarray, times: np.ndarray, tabs: np.ndarray) -> list[dict]:
    arrays = [np.asarray(x) for x in (users, times, tabs)]
    if (
        any(x.ndim != 1 or x.dtype.kind not in "iu" for x in arrays)
        or len({len(x) for x in arrays}) != 1
    ):
        raise ValueError("Equal-length integer identity columns required")
    if any(x.dtype.kind == "u" and np.any(x > np.iinfo(np.int64).max) for x in arrays[1:]):
        raise ValueError("Timestamp and tab identities must fit the signed int64 source schema")
    if not len(users):
        return []
    order = np.lexsort((tabs, times, users))
    u, t, a = (x[order] for x in arrays)
    boundaries = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1, len(u)]
    result = []
    for left, right in zip(boundaries, boundaries[1:]):
        payload = (
            np.column_stack([t[left:right].astype("<i8"), a[left:right].astype("<i8")])
            .astype("<i8")
            .tobytes(order="C")
        )
        result.append(
            {
                "user_id": int(u[left]),
                "rows": int(right - left),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return result


def confirmatory_cohort(pilot: list[dict], final: list[dict], seed: int) -> dict:
    if not pilot or not final:
        raise ValueError("Nonempty pilot and candidate identity records required")
    p = {r["sha256"]: r for r in pilot}
    f = {r["sha256"]: r for r in final}
    if len(p) != len(pilot) or len(f) != len(final):
        raise ValueError("Identity fingerprints must be unique in each release")
    if len({r["user_id"] for r in final}) != len(final):
        raise ValueError("Candidate user IDs must be unique")
    missing = sorted(p.keys() - f.keys())
    if missing:
        raise ValueError(f"Pilot identities missing from final release: {len(missing)}")
    if any(p[h]["rows"] != f[h]["rows"] for h in p):
        raise ValueError("Matched identity row counts differ")
    excluded = sorted(f[h]["user_id"] for h in p)
    excluded_set = set(excluded)
    eligible = [r["user_id"] for r in final if r["user_id"] not in excluded_set]
    ranked = sorted(eligible, key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).digest())
    n = len(ranked)
    assignment = {
        str(u): "train"
        if i < int(0.7 * n)
        else "validation"
        if i < int(0.8 * n)
        else "calibration"
        if i < int(0.9 * n)
        else "test"
        for i, u in enumerate(ranked)
    }
    return {
        "schema_version": "human-cohort-v1",
        "seed": seed,
        "source_users": len(final),
        "pilot_users_excluded": len(excluded),
        "eligible_users": n,
        "excluded_user_ids": excluded,
        "user_assignment": assignment,
        "identity_rule": "exact timestamp/tab multiset SHA256 match and row count; no outcomes or item IDs",
    }


def validate_27k_logs(entries: list[dict]) -> tuple[str, ...]:
    """Require every observed release shard and all 19 source columns."""
    from pathlib import PurePosixPath

    expected = {
        "log_standard_4_22_to_5_08_27k_part1.csv",
        "log_standard_4_22_to_5_08_27k_part2.csv",
        "log_standard_4_08_to_4_21_27k_part1.csv",
        "log_standard_4_08_to_4_21_27k_part2.csv",
        "log_random_4_22_to_5_08_27k.csv",
    }
    logs = [
        entry
        for entry in entries
        if PurePosixPath(entry["name"]).name.startswith("log_") and entry["name"].endswith(".csv")
    ]
    if len(logs) != 5 or {PurePosixPath(e["name"]).name for e in logs} != expected:
        raise ValueError("Expected all five inspected 27K log shards")
    if any(
        len(e["columns"]) != len(SOURCE_COLUMNS) or set(e["columns"]) != set(SOURCE_COLUMNS)
        for e in logs
    ):
        raise ValueError("Full source-column preservation needs schema review")
    return tuple(sorted(e["name"] for e in logs))
