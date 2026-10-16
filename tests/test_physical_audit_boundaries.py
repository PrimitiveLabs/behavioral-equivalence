"""Independent audit rejection paths on small fabricated physical records."""

from copy import deepcopy

import numpy as np
import pytest

from behavioral_equivalence.adjacent_audit import audit_adjacent_cases
from behavioral_equivalence.adjacent_calibration import adjacent_origins, representative_indices
from behavioral_equivalence.multistep_audit import audit_multistep_cases
from behavioral_equivalence.randomized_audit import audit_randomized_cases
from behavioral_equivalence.response_frontier import select_partition
from test_adjacent_audit import fixture as adjacent_fixture
from test_multistep_audit import fixture as multistep_fixture
from test_randomized_audit import fixture as randomized_fixture
from test_response_frontier import tables


@pytest.mark.parametrize("kind", ["adjacent", "multistep", "randomized"])
@pytest.mark.parametrize(
    "damage", ["event_order", "bundle_order", "availability", "user_role", "video", "tab"]
)
def test_audits_reject_physical_order_and_bindings(kind, damage):
    factory, audit = {
        "adjacent": (adjacent_fixture, audit_adjacent_cases),
        "multistep": (multistep_fixture, audit_multistep_cases),
        "randomized": (randomized_fixture, audit_randomized_cases),
    }[kind]
    events, bundles, cases, config = factory()
    events, bundles, cases = map(deepcopy, (events, bundles, cases))
    if damage == "event_order":
        events["time_ms"][0] = 10000
    elif damage == "bundle_order":
        bundles["time_ms"][0] = bundles["time_ms"][1]
    elif damage == "availability":
        bundles["available_sum_ms"][0] = -1
    elif damage == "user_role":
        cases["user_role_code"][0] = 1 - cases["user_role_code"][0]
    elif damage == "video":
        cases["video_id"][0] += 1
    else:
        cases["tab"][0] += 1
    with pytest.raises(ValueError):
        audit(events, bundles, cases, 0, config)


@pytest.mark.parametrize("role", [-1, 4, True, 1.5])
def test_multistep_audit_rejects_undeclared_role(role):
    events, bundles, cases, config = multistep_fixture()
    with pytest.raises(ValueError, match="Declared calendar role"):
        audit_multistep_cases(events, bundles, cases, 0, config, target_role=role)


@pytest.mark.parametrize("index", [[], [1], [0, 2], [0, 1, 0], [0.0]])
def test_representatives_require_contiguous_integer_bundles(index):
    with pytest.raises(ValueError):
        representative_indices(np.array(index), 0)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_adjacent_origins_rejects_invalid_cap(cap):
    with pytest.raises(ValueError):
        adjacent_origins(np.array([0, 0]), np.array([0, 1]), np.array([2, 2]), cap)


@pytest.mark.parametrize(
    "policy", ["targeted", "random_environment", "fixed_grid", "uncertainty", "random_witness"]
)
def test_zero_split_budget_preserves_partition_for_every_policy(policy):
    discovery, estimation = tables()
    initial = ((0, 1, 2, 3), (4, 5, 6, 7))
    assert select_partition(discovery, estimation, initial, 0, policy, 17) == (initial, [])


def test_count_gate_requires_success_counts_and_supported_candidates():
    discovery, estimation = tables()
    with pytest.raises(ValueError, match="success counts"):
        select_partition(
            discovery,
            estimation,
            (tuple(range(8)),),
            1,
            "targeted",
            17,
            min_cases=1,
            min_users=1,
            exact_count_gate=True,
        )
