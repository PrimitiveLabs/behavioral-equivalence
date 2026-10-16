from copy import deepcopy
import numpy as np
import pytest
from test_adjacent_audit import fixture as source_fixture
from behavioral_equivalence.adjacent_calibration import query_history_tokens
from behavioral_equivalence.calendar_splits import EvaluationCalendar, calendar_roles
from behavioral_equivalence.randomized_audit import audit_randomized_cases


def fixture():
    events, bundles, _, config = source_fixture()
    events["is_rand"] = (np.arange(len(events["user_id"])) % 4 == 0).astype(np.int8)
    events["is_rand"][1] = 1
    bundles["user_split"][50:] = "test"
    rows = np.flatnonzero(events["is_rand"] == 1)
    event_bundles = rows // 3
    queries = np.unique(event_bundles)
    roles = calendar_roles(
        EvaluationCalendar(**config["calendar"]),
        bundles["time_ms"],
        bundles["available_sum_ms"],
        bundles["user_split"],
        "known_users",
        0,
    )
    history = query_history_tokens(
        bundles["user_id"],
        bundles["time_ms"],
        bundles["available_sum_ms"],
        bundles["clicks"],
        bundles["exposures"],
        queries,
        0,
    )
    cases = {
        k: events[v][rows]
        for k, v in dict(
            user_id="user_id",
            time_ms="time_ms",
            tab="tab",
            video_id="video_id",
            is_random="is_rand",
            target="is_click",
        ).items()
    }
    cases.update(
        case_id=rows.astype(np.uint64),
        bundle_index=event_bundles,
        role_code=roles[event_bundles],
        user_role_code=np.repeat([0, 3], 50)[event_bundles],
        tokens=history[np.searchsorted(queries, event_bundles)],
    )
    return events, bundles, cases, config


def test_all_concurrent_rows_and_delayed_feedback_are_independently_reconstructed():
    events, bundles, cases, config = fixture()
    report = audit_randomized_cases(events, bundles, cases, 0, config)
    assert report["random_exposures_audited"] == 76
    assert report["query_bundles_audited"] == 75
    assert all(report["checks"].values())
    assert sum(r["exposures"] for r in report["support"] if r["protocol"] == "unseen_users") < 76


@pytest.mark.parametrize(
    "field", ["tokens", "case_id", "bundle_index", "target", "role_code", "user_role_code"]
)
def test_corrupted_randomized_identity_history_and_outcomes_are_rejected(field):
    events, bundles, cases, config = fixture()
    bad = deepcopy(cases)
    bad[field].flat[0] += 1
    with pytest.raises(ValueError):
        audit_randomized_cases(events, bundles, bad, 0, config)
