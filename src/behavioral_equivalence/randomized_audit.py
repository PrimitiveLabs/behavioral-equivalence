"""Independent physical-row audit of all randomized exposures and their histories."""

import heapq
import numpy as np


def audit_randomized_cases(
    events: dict, bundles: dict, cases: dict, bucket: int, config: dict
) -> dict:
    eu, et = events["user_id"], events["time_ms"]
    bu, bt = bundles["user_id"], bundles["time_ms"]
    if (
        np.any(eu[1:] < eu[:-1])
        or np.any((eu[1:] == eu[:-1]) & (et[1:] < et[:-1]))
        or np.any(bu[1:] < bu[:-1])
        or np.any((bu[1:] == bu[:-1]) & (bt[1:] <= bt[:-1]))
    ):
        raise ValueError("Canonical source user/time order required")
    b_bound = np.r_[0, np.flatnonzero(bu[1:] != bu[:-1]) + 1, len(bu)]
    e_bound = np.r_[0, np.flatnonzero(eu[1:] != eu[:-1]) + 1, len(eu)]
    ranges = {int(bu[left]): (left, right) for left, right in zip(b_bound[:-1], b_bound[1:])}
    source_bundle = np.empty(len(eu), dtype=np.int64)
    for left, right in zip(e_bound[:-1], e_bound[1:]):
        lo, hi = ranges[int(eu[left])]
        selected = np.searchsorted(bt[lo:hi], et[left:right]) + lo
        if np.any(selected >= hi) or not np.array_equal(bt[selected], et[left:right]):
            raise ValueError("Physical source row lacks its timestamp bundle")
        source_bundle[left:right] = selected
    available = bundles["available_sum_ms"]
    embargo = config["embargo_hours"] * 3600000
    if np.any(available < bt):
        raise ValueError("Source feedback availability precedes exposure")
    cutoffs = np.array(
        [config["calendar"][k] for k in ["train_end", "validation_end", "calibration_end"]]
    )
    roles = np.searchsorted(cutoffs, bt, side="right").astype(np.int8)
    for role, boundary in enumerate(cutoffs):
        roles[(roles == role) & (available + embargo > boundary)] = -1
    rows = np.flatnonzero(events["is_rand"] == 1)
    event_bundles = source_bundle[rows]
    queries = np.unique(event_bundles)
    if not np.array_equal(cases["bundle_index"], event_bundles):
        raise ValueError("All random physical exposure rows must be retained")
    expected_ids = rows.astype(np.uint64) + np.uint64(bucket << 32)
    if not np.array_equal(cases["case_id"], expected_ids):
        raise ValueError("Randomized identity differs from the physical source row")
    for target, source in {
        "user_id": "user_id",
        "time_ms": "time_ms",
        "tab": "tab",
        "video_id": "video_id",
        "is_random": "is_rand",
        "target": "is_click",
    }.items():
        if not np.array_equal(cases[target], events[source][rows]):
            raise ValueError("Query differs from physical source: " + target)
    names = ("train", "validation", "calibration", "test")
    user_role = np.array([names.index(str(s)) for s in bundles["user_split"]], dtype=np.int8)
    expected = dict(
        role_code=roles[event_bundles],
        user_role_code=user_role[event_bundles],
    )
    for name, values in expected.items():
        if not np.array_equal(cases[name], values):
            raise ValueError("Randomized exposure role binding differs: " + name)
    encoded = 1 + np.minimum(15, 16 * bundles["clicks"].astype(np.int64) // bundles["exposures"])
    histories = np.zeros((len(queries), 32), dtype=np.uint8)
    for left, right in zip(b_bound[:-1], b_bound[1:]):
        first, last = np.searchsorted(queries, [left, right])
        if first == last:
            continue
        arrivals = np.argsort(available[left:right], kind="stable") + left
        cursor, recent = 0, []
        for q in range(first, last):
            timestamp = bt[queries[q]]
            while cursor < len(arrivals) and available[arrivals[cursor]] + embargo < timestamp:
                heapq.heappush(recent, int(arrivals[cursor]))
                if len(recent) > 32:
                    heapq.heappop(recent)
                cursor += 1
            if recent:
                histories[q, -len(recent) :] = encoded[sorted(recent)]
            else:
                histories[q, -1] = 17
    if not np.array_equal(cases["tokens"], histories[np.searchsorted(queries, event_bundles)]):
        raise ValueError("Randomized history differs from independent bounded-heap reconstruction")
    support = []
    for protocol in ["known_users", "unseen_users"]:
        for role, role_name in enumerate(names):
            selected = roles[event_bundles] == role
            if protocol == "unseen_users":
                selected &= user_role[event_bundles] == role
            support.append(
                dict(
                    protocol=protocol,
                    role=role_name,
                    exposures=int(selected.sum()),
                    users=int(len(np.unique(eu[rows][selected]))),
                )
            )
    return dict(
        status="complete",
        bucket=bucket,
        random_exposures_audited=len(rows),
        query_bundles_audited=len(queries),
        support=support,
        checks=dict(
            all_random_physical_rows=True,
            concurrent_exposures_retained=True,
            calendar_and_user_roles=True,
            all_available_history_tokens=True,
            response_binding=True,
        ),
        independence="Separate full physical-row selection, per-user bundle mapping and bounded-heap history reconstruction. No production selection or history helper called.",
        limitations="Conditional observed random responses only; no assignment propensity, human confidence coverage or causal-effect claim.",
    )
