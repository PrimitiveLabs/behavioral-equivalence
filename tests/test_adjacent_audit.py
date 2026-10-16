from copy import deepcopy
import numpy as np
import pytest
from behavioral_equivalence.adjacent_audit import audit_adjacent_cases
from behavioral_equivalence.adjacent_calibration import (
    representative_indices,
    adjacent_origins,
    query_history_tokens,
)
from behavioral_equivalence.calendar_splits import EvaluationCalendar, calendar_roles


def fixture():
    config = dict(
        representative_seed_base=19,
        origin_seed=71,
        origin_cap_per_user=8,
        embargo_hours=0,
        calendar=dict(train_end=100, validation_end=200, calibration_end=400),
    )
    bu = np.repeat([1, 2], 50)
    bt = np.tile(np.arange(10, 510, 10), 2)
    availability = bt + np.where(np.arange(100) % 5 == 0, 110, 0)
    clicks = np.arange(100) % 3
    bundles = dict(
        user_id=bu,
        time_ms=bt,
        available_sum_ms=availability,
        clicks=clicks,
        exposures=np.full(100, 3),
        user_split=np.repeat(["train", "calibration"], 50),
    )
    index = np.repeat(np.arange(100), 3)
    events = dict(
        user_id=bu[index],
        time_ms=bt[index],
        is_click=np.tile([0, 1, 0], 100),
        video_id=np.arange(300),
        tab=np.tile([0, 1, 2], 100),
        is_rand=np.zeros(300, dtype=int),
    )
    rep = representative_indices(index, 19)
    roles = calendar_roles(
        EvaluationCalendar(**config["calendar"]),
        bt,
        availability,
        bundles["user_split"],
        "known_users",
        0,
    )
    origins = adjacent_origins(bu, bt, roles, 8, 71)
    queries = np.unique(np.r_[origins, origins + 1])
    rows = rep[queries]
    user_code = np.repeat([0, 2], 50)
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
        bundle_index=queries,
        role_code=roles[queries],
        user_role_code=user_code[queries],
        pair_current=np.searchsorted(queries, origins),
        pair_next=np.searchsorted(queries, origins + 1),
        pair_user_id=bu[origins],
        pair_user_role_code=user_code[origins],
        tokens=query_history_tokens(bu, bt, availability, clicks, np.full(100, 3), queries, 0),
    )
    return events, bundles, cases, config


def test_independent_audit_reconstructs_delayed_feedback_and_all_adjacent_endpoints():
    events, bundles, cases, config = fixture()
    r = audit_adjacent_cases(events, bundles, cases, 0, config)
    assert r["transitions_audited"] == 16 and r["known_users"] == 2
    assert r["unseen_transitions"] == 8 and r["unseen_users"] == 1
    assert all(r["checks"].values())


@pytest.mark.parametrize(
    "field,message",
    [
        ("tokens", "history"),
        ("target", "physical source"),
        ("pair_next", "pair binding"),
        ("case_id", "Representative"),
    ],
)
def test_audit_rejects_source_pair_representative_and_history_corruption(field, message):
    events, bundles, cases, config = fixture()
    corrupt = deepcopy(cases)
    corrupt[field].flat[0] += 1
    with pytest.raises(ValueError, match=message):
        audit_adjacent_cases(events, bundles, corrupt, 0, config)
