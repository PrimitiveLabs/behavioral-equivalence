"""Independent physical-row, outcome-blind origin and observed-horizon audit."""

import heapq
import numpy as np


def audit_multistep_cases(
    events: dict, bundles: dict, cases: dict, bucket: int, config: dict, *, target_role: int = 3
) -> dict:
    if (
        not isinstance(target_role, int)
        or isinstance(target_role, bool)
        or target_role not in {0, 1, 2, 3}
    ):
        raise ValueError("Declared calendar role required for independent audit")
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
    counts = np.bincount(source_bundle, minlength=len(bu))
    starts = np.cumsum(counts) - counts
    if (counts == 0).any() or not np.array_equal(source_bundle[starts], np.arange(len(bu))):
        raise ValueError("Contiguous complete source bundles required")
    rng = np.random.default_rng(config["representative_seed_base"] + bucket)
    representatives = starts + rng.integers(0, counts)
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
    origins = []
    for left, right in zip(b_bound[:-1], b_bound[1:]):
        candidates = np.arange(left, right)[roles[left:right] == target_role]
        rng = np.random.default_rng([config["origin_seed"], int(bu[left])])
        chosen = rng.choice(
            candidates, min(config["origin_cap_per_user"], len(candidates)), replace=False
        )
        origins.extend(sorted(map(int, chosen)))
    origins = np.asarray(origins, dtype=np.int64)
    queries = origins
    if not np.array_equal(cases["bundle_index"], queries):
        raise ValueError("Declared-role origins differ from independent outcome-blind selection")
    rows = representatives[queries]
    expected_ids = rows.astype(np.uint64) + np.uint64(bucket << 32)
    if not np.array_equal(cases["case_id"], expected_ids):
        raise ValueError("Representative differs from the outcome-independent physical-row draw")
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
        role_code=roles[queries],
        user_role_code=user_role[queries],
    )
    for name, values in expected.items():
        if not np.array_equal(cases[name], values):
            raise ValueError("Declared origin role binding differs: " + name)
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
    if not np.array_equal(cases["tokens"], histories):
        raise ValueError("Origin history differs from independent bounded-heap reconstruction")
    horizons = config["decision_horizons"]
    if not np.array_equal(cases["decision_horizons"], horizons):
        raise ValueError("Decision horizons differ")
    observed = np.zeros((len(origins), len(horizons)), dtype=bool)
    ends = np.full(observed.shape, -1, dtype=np.int64)
    positive = np.full(observed.shape, -1, dtype=np.int16)
    last_positive = np.full(observed.shape, -1, dtype=np.int8)
    feedback = events["is_click"][representatives]
    for q, origin in enumerate(origins):
        _, user_end = ranges[int(bu[origin])]
        for j, horizon in enumerate(horizons):
            stop = int(origin) + horizon
            if stop <= user_end and np.all(roles[origin:stop] == target_role):
                observed[q, j] = True
                ends[q, j] = stop - 1
                positive[q, j] = sum(map(int, feedback[origin:stop]))
                last_positive[q, j] = feedback[stop - 1]
    for name, values in dict(
        horizon_observed=observed,
        horizon_end_bundle=ends,
        horizon_positive_count=positive,
        horizon_last_positive=last_positive,
    ).items():
        if not np.array_equal(cases[name], values):
            raise ValueError("Observed-horizon target or censoring differs: " + name)
    unseen = user_role[origins] == target_role
    return dict(
        status="complete",
        bucket=bucket,
        queries_audited=len(queries),
        known_users=len(np.unique(bu[origins])),
        unseen_origins=int(unseen.sum()),
        unseen_users=len(np.unique(bu[origins][unseen])),
        horizons=[
            dict(
                horizon=h,
                known_observed=int(observed[:, j].sum()),
                unseen_observed=int(observed[unseen, j].sum()),
            )
            for j, h in enumerate(horizons)
        ],
        checks=dict(
            physical_source_binding=True,
            representative_draw=True,
            outcome_blind_origin_draw=True,
            calendar_and_user_roles=True,
            all_available_history_tokens=True,
            target_and_censoring_reconstruction=True,
        ),
        independence="Separate physical-row NumPy mapping, per-user origin sampling, bounded-heap histories and direct future slices. Shared specified NumPy PRNG only; no production cohort/history helper called.",
        limitations="Availability proxies and released-cohort assumptions remain. No predictive scores, unconditional survival or human confidence guarantee.",
    )
