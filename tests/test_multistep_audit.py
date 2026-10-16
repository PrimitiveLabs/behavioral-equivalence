from copy import deepcopy
import numpy as np
import pytest
from test_adjacent_audit import fixture as source_fixture
from behavioral_equivalence.multistep_audit import audit_multistep_cases
from behavioral_equivalence.multistep_cohort import (
    select_test_origins,
    observed_horizon_cohorts,
    recorded_horizon_targets,
)
from behavioral_equivalence.adjacent_calibration import representative_indices, query_history_tokens
from behavioral_equivalence.calendar_splits import EvaluationCalendar, calendar_roles


def fixture(role=3):
    events, bundles, _, config = source_fixture()
    config["decision_horizons"] = [1, 3, 7, 12]
    bundles["user_split"][50:] = ("train", "validation", "calibration", "test")[role]
    users, times = bundles["user_id"], bundles["time_ms"]
    index = np.repeat(np.arange(len(users)), 3)
    representatives = representative_indices(index, config["representative_seed_base"])
    roles = calendar_roles(
        EvaluationCalendar(**config["calendar"]),
        times,
        bundles["available_sum_ms"],
        bundles["user_split"],
        "known_users",
        0,
    )
    if role == 3:
        origins = select_test_origins(
            users, times, roles, config["origin_cap_per_user"], config["origin_seed"]
        )
        cohorts = observed_horizon_cohorts(
            users, times, roles, origins, config["decision_horizons"]
        )
    else:
        from behavioral_equivalence.training_horizons import training_role_cohort

        origins, cohorts = training_role_cohort(
            users,
            times,
            roles,
            role,
            config["decision_horizons"],
            config["origin_cap_per_user"],
            config["origin_seed"],
        )
    targets = [recorded_horizon_targets(events["is_click"][representatives], c) for c in cohorts]
    rows = representatives[origins]
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
        bundle_index=origins,
        role_code=roles[origins],
        user_role_code=np.repeat([0, role], 50)[origins],
        tokens=query_history_tokens(
            users,
            times,
            bundles["available_sum_ms"],
            bundles["clicks"],
            bundles["exposures"],
            origins,
            0,
        ),
        decision_horizons=np.array(config["decision_horizons"]),
        horizon_observed=np.column_stack([c["observed_mask"] for c in cohorts]),
        horizon_end_bundle=np.column_stack([c["end_indices"] for c in cohorts]),
        horizon_positive_count=np.column_stack([t["positive_count"] for t in targets]),
        horizon_last_positive=np.column_stack([t["last_positive"] for t in targets]),
    )
    return events, bundles, cases, config


def test_separate_slice_and_heap_reconstruction():
    events, bundles, cases, config = fixture()
    result = audit_multistep_cases(events, bundles, cases, 0, config)
    assert result["queries_audited"] == 16
    assert result["unseen_origins"] == 8
    assert all(result["checks"].values())
    assert result["horizons"][-1]["known_observed"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "tokens",
        "case_id",
        "bundle_index",
        "target",
        "role_code",
        "horizon_observed",
        "horizon_end_bundle",
        "horizon_positive_count",
        "horizon_last_positive",
        "decision_horizons",
    ],
)
def test_independent_audit_rejects_query_target_and_censoring_corruption(field):
    events, bundles, cases, config = fixture()
    corrupt = deepcopy(cases)
    if corrupt[field].dtype.kind == "b":
        corrupt[field].flat[0] = not corrupt[field].flat[0]
    else:
        corrupt[field].flat[0] += 1
    with pytest.raises(ValueError):
        audit_multistep_cases(events, bundles, corrupt, 0, config)


@pytest.mark.parametrize("role", [0, 1])
def test_fit_roles_are_reconstructed_from_physical_rows(role):
    events, bundles, cases, config = fixture(role)
    report = audit_multistep_cases(events, bundles, cases, 0, config, target_role=role)
    assert report["queries_audited"] == len(cases["case_id"])
    assert np.all(cases["role_code"] == role)
    assert all(report["checks"].values())
    with pytest.raises(ValueError):
        audit_multistep_cases(events, bundles, cases, 0, config, target_role=3)
