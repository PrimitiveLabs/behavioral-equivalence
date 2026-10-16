"""Independent array/heap audit of sampled cases and their available histories.

The only shared sampling primitive is the pinned Polars hash specified by the
protocol. Calendar eligibility, selection and histories are rebuilt separately.
"""

import heapq
import numpy as np


def audit_cases(
    events: dict, bundles: dict, cases: dict, bucket: int, config: dict, hours: int
) -> dict:
    import polars as pl

    eu, et = events["user_id"], events["time_ms"]
    bu, bt = bundles["user_id"], bundles["time_ms"]
    if np.any(eu[1:] < eu[:-1]) or np.any((eu[1:] == eu[:-1]) & (et[1:] < et[:-1])):
        raise ValueError("Source events are not in canonical user/time order")
    if np.any(bu[1:] < bu[:-1]) or np.any((bu[1:] == bu[:-1]) & (bt[1:] <= bt[:-1])):
        raise ValueError("Source bundles are not in canonical user/time order")
    ids = cases["case_id"]
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids >> 32 != bucket):
        raise ValueError("Unique source-bound case namespace required")
    rows = (ids & np.uint64(2**32 - 1)).astype(np.int64)
    if np.any(rows >= len(eu)):
        raise ValueError("Case row ordinal is outside the source")
    for target, source in {
        "user_id": "user_id",
        "time_ms": "time_ms",
        "tab": "tab",
        "video_id": "video_id",
        "is_random": "is_rand",
        "target": "is_click",
    }.items():
        if not np.array_equal(cases[target], events[source][rows]):
            raise ValueError("Case differs from physical source records: " + target)
    embargo = hours * 3600 * 1000
    available = bundles["available_sum_ms"]
    if np.any(available < bt) or (
        len(available) and int(available.max()) + embargo > np.iinfo(np.int64).max
    ):
        raise ValueError("Invalid source availability timestamps")
    cutoffs = [
        config["calendar"][key] for key in ["train_end", "validation_end", "calibration_end"]
    ]
    role = np.full(len(bt), 3, dtype=np.int8)
    role[bt < cutoffs[2]] = 2
    role[bt < cutoffs[1]] = 1
    role[bt < cutoffs[0]] = 0
    for r, boundary in enumerate(cutoffs):
        role[(role == r) & (available + embargo > boundary)] = -1
    event_boundaries = np.r_[0, np.flatnonzero(eu[1:] != eu[:-1]) + 1, len(eu)]
    bundle_boundaries = np.r_[0, np.flatnonzero(bu[1:] != bu[:-1]) + 1, len(bu)]
    bundle_ranges = {
        int(bu[left]): (left, right)
        for left, right in zip(bundle_boundaries[:-1], bundle_boundaries[1:])
    }
    source_bundle = np.empty(len(eu), dtype=np.int64)
    for left, right in zip(event_boundaries[:-1], event_boundaries[1:]):
        lo, hi = bundle_ranges[int(eu[left])]
        index = np.searchsorted(bt[lo:hi], et[left:right]) + lo
        if np.any(index >= hi) or not np.array_equal(bt[index], et[left:right]):
            raise ValueError("Source event has no matching bundle")
        source_bundle[left:right] = index
    selected_bundle = source_bundle[rows]
    if not np.array_equal(cases["bundle_index"], selected_bundle):
        raise ValueError("Case bundle indices differ from independent source mapping")
    if not np.array_equal(cases["available_sum_ms"], available[selected_bundle]):
        raise ValueError("Case availability differs from its source bundle")
    if np.any(role[selected_bundle] < 0) or not np.array_equal(
        cases["role_code"], role[selected_bundle]
    ):
        raise ValueError("Calendar role or fitting-cutoff availability guard violated")
    role_names = ("train", "validation", "calibration", "test")
    user_code = np.array([role_names.index(str(r)) for r in bundles["user_split"]], dtype=np.int8)
    if not np.array_equal(cases["user_role_code"], user_code[selected_bundle]):
        raise ValueError("Outcome-independent user assignment changed")
    # Use the contract's hash, then independent NumPy sorting/group selection.
    priorities = (
        pl.Series(np.arange(len(eu), dtype=np.uint64) + np.uint64(bucket << 32))
        .hash(seed=config["sampling_seed"])
        .to_numpy()
    )
    event_roles = role[source_bundle]
    expected_rows = []
    for left, right in zip(event_boundaries[:-1], event_boundaries[1:]):
        for r, name in enumerate(role_names):
            eligible = np.flatnonzero(event_roles[left:right] == r) + left
            order = np.lexsort((eligible, priorities[eligible]))
            expected_rows.append(eligible[order[: config["per_user_role_caps"][name]]])
    expected_rows = np.sort(np.concatenate(expected_rows))
    if not np.array_equal(np.sort(rows), expected_rows):
        raise ValueError("Case sample differs from the independent priority/cap reconstruction")
    query, inverse = np.unique(selected_bundle, return_inverse=True)
    expected_tokens = np.zeros((len(query), 32), dtype=np.uint8)
    encoded = 1 + np.minimum(15, 16 * bundles["clicks"].astype(np.int64) // bundles["exposures"])
    # A bounded min-heap stores the 32 latest exposure indices among all
    # completed bundles. Older discarded indices can never become recent again.
    for left, right in zip(bundle_boundaries[:-1], bundle_boundaries[1:]):
        first, last = np.searchsorted(query, [left, right])
        if first == last:
            continue
        arrivals = np.argsort(available[left:right], kind="stable") + left
        cursor, recent = 0, []
        for q in range(first, last):
            timestamp = bt[query[q]]
            while cursor < len(arrivals) and available[arrivals[cursor]] + embargo < timestamp:
                heapq.heappush(recent, int(arrivals[cursor]))
                if len(recent) > 32:
                    heapq.heappop(recent)
                cursor += 1
            if recent:
                expected_tokens[q, -len(recent) :] = encoded[sorted(recent)]
            else:
                expected_tokens[q, -1] = 17
    if not np.array_equal(cases["tokens"], expected_tokens[inverse]):
        raise ValueError("Case history differs from independent availability/heap reconstruction")
    return {
        "status": "complete",
        "bucket": bucket,
        "embargo_hours": hours,
        "cases_audited": len(ids),
        "unique_bundle_queries_audited": len(query),
        "checks": {
            "source_binding": True,
            "calendar_and_fit_availability": True,
            "user_assignment": True,
            "exact_priority_sample": True,
            "all_history_tokens_reconstructed": True,
        },
        "independence": "PyArrow physical source rows and NumPy calendar/selection with bounded-heap histories; shared pinned Polars hash primitive only",
        "limits": "Timing proxies remain assumptions; no human predictive or Markov guarantee.",
    }
