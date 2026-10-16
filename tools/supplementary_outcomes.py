"""Frozen candidate target construction for supplementary human outcomes.

Kept outside the locked primary library until the active primary campaign ends.
No source rows are read or written by this module.
"""

import numpy as np

FEEDBACK_FIELDS = ("is_like", "is_follow", "is_comment", "is_forward", "is_hate")
WATCH_CUTS_MS = (1000, 5000, 30000)


def supplementary_targets(columns):
    """Return binary indicators, preserving missingness as -1.

    Watch categories are [0,1000), [1000,5000), [5000,30000), [30000,infinity)
        milliseconds over recorded exposures, including recorded zero play time.
        Feedback values other than 0/1 are unknown, never silently treated as zero.
    """
    names = (*FEEDBACK_FIELDS, "play_time_ms")
    if set(names) - set(columns):
        raise ValueError("All supplementary outcome fields are required")
    arrays = {name: np.asarray(columns[name], dtype=float) for name in names}
    if any(a.ndim != 1 for a in arrays.values()):
        raise ValueError("One-dimensional outcome columns required")
    lengths = {len(a) for a in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Aligned outcome columns required")
    results = {}
    for name in FEEDBACK_FIELDS:
        a = arrays[name]
        valid = np.isfinite(a) & np.isin(a, [0, 1])
        results[name] = np.where(valid, a, -1).astype(np.int8)
    watch = arrays["play_time_ms"]
    valid = np.isfinite(watch) & (watch >= 0)
    categories = np.searchsorted(WATCH_CUTS_MS, watch, side="right")
    for category in range(4):
        results[f"watch_category_{category}"] = np.where(valid, categories == category, -1).astype(
            np.int8
        )
    return results


def source_bound_targets(source, cases, bucket):
    """Attach outcomes by audited physical-row identity, never by event-key join.

    The caller must first verify the complete prepared parquet SHA against its
        audited manifest. Reordering or filtering that parquet invalidates case IDs.
    """
    if not isinstance(bucket, int) or isinstance(bucket, bool) or not 0 <= bucket < 32:
        raise ValueError("Prepared bucket must be an integer in [0,32)")
    ids = np.asarray(cases["case_id"])
    if ids.ndim != 1 or ids.dtype.kind != "u" or ids.dtype.itemsize != 8:
        raise ValueError("Unsigned 64-bit physical case IDs required")
    if len(np.unique(ids)) != len(ids) or np.any(ids >> np.uint64(32) != bucket):
        raise ValueError("Distinct case IDs in the requested bucket required")
    names = ("user_id", "time_ms", "video_id", "tab", "is_click", *FEEDBACK_FIELDS, "play_time_ms")
    arrays = {name: np.asarray(source[name]) for name in names}
    if any(a.ndim != 1 for a in arrays.values()) or len({len(a) for a in arrays.values()}) != 1:
        raise ValueError("Aligned one-dimensional source columns required")
    rows = (ids & np.uint64(2**32 - 1)).astype(np.int64)
    if np.any(rows >= len(arrays["user_id"])):
        raise ValueError("Case ordinal outside the authenticated prepared source")
    for target, original in [
        ("user_id", "user_id"),
        ("time_ms", "time_ms"),
        ("video_id", "video_id"),
        ("tab", "tab"),
        ("target", "is_click"),
    ]:
        observed = np.asarray(cases[target])
        if observed.shape != ids.shape or not np.array_equal(observed, arrays[original][rows]):
            raise ValueError("Canonical case/source identity mismatch: " + target)
    return supplementary_targets(
        {name: arrays[name][rows] for name in (*FEEDBACK_FIELDS, "play_time_ms")}
    )


def categorical_targets(indicators):
    """Combine the four watch indicators into one categorical endpoint."""
    binary = [np.asarray(indicators[name]) for name in FEEDBACK_FIELDS]
    watch = np.column_stack([indicators[f"watch_category_{k}"] for k in range(4)])
    if (
        watch.ndim != 2
        or any(a.shape != (len(watch),) for a in binary)
        or any(not np.isin(a, [-1, 0, 1]).all() for a in binary)
        or not np.isin(watch, [-1, 0, 1]).all()
    ):
        raise ValueError("Aligned valid endpoint indicators required")
    missing = np.all(watch == -1, axis=1)
    if np.any(~missing & ((watch < 0).any(axis=1) | (watch.sum(axis=1) != 1))):
        raise ValueError("Watch indicators must be one-hot or wholly missing")
    category = np.where(missing, -1, np.argmax(watch, axis=1))
    return np.column_stack([*binary, category]).astype(np.int8)


def fixed_role_sample(role_counts, maximum=100000, seed=20261023):
    """Outcome-blind global ordinal sample, common across architectures/seeds."""
    counts = np.asarray(role_counts)
    if (
        counts.ndim != 1
        or counts.dtype.kind not in "iu"
        or (counts < 0).any()
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum < 1
    ):
        raise ValueError("Nonnegative bucket counts and positive integer sample cap required")
    total = sum(map(int, counts))
    chosen = np.sort(np.random.default_rng(seed).choice(total, min(maximum, total), replace=False))
    offset = 0
    result = []
    for count in counts:
        end = offset + int(count)
        result.append(
            chosen[np.searchsorted(chosen, offset) : np.searchsorted(chosen, end)] - offset
        )
        offset = end
    return result
