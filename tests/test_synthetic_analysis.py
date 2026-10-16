from copy import deepcopy
import pytest
from behavioral_equivalence.synthetic_analysis import world_method_rows


def fixture():
    method = dict(
        method="example",
        parameters={},
        states=2,
        compression_ratio=2,
        epsilon_measured=0.01,
        partition_seconds=2,
        property_seconds=3,
        independent_verification={"seconds": 4},
        process_cumulative_peak_rss_kib=100,
        transition_edges=4,
        properties=[
            dict(target=t, horizon=5, max_error=e, bound=b, violated=False)
            for t, e, b in [
                ("complete", 0.02, 0.05),
                ("disengage", 0.03, 0.05),
                ("survival", 0.01, 0.05),
                ("complete_given_active_next", 0.4, 0.8),
            ]
        ],
    )
    report = dict(status="complete", spec={"family": "example"}, world_digest="a", methods=[method])
    audit = dict(
        status="complete",
        spec=report["spec"],
        world_digest="a",
        methods=[
            dict(
                index=0,
                method="example",
                states=2,
                verified=True,
                epsilon_approximation_for_display=0.01,
                exact_budget_excess_for_display=None,
                budget_strictly_satisfied=None,
                epsilon_is_exactly_zero=False,
            )
        ],
    )
    return report, audit


def test_conditional_error_keeps_its_distinct_bound_and_does_not_enter_unconditional_maximum():
    report, audit = fixture()
    methods, properties = world_method_rows(report, audit, [5])
    assert methods[0]["compression_ratio"] == 2
    assert properties[0]["maximum_unconditional_floating_error"] == 0.03
    assert properties[0]["floating_coupling_bound"] == 0.05
    assert properties[0]["conditional_floating_error"] == 0.4
    assert properties[0]["conditional_bound"] == 0.8


def test_missing_duplicate_and_mismatched_audit_evidence_is_rejected():
    report, audit = fixture()
    bad = deepcopy(report)
    bad["methods"][0]["properties"].pop()
    with pytest.raises(ValueError, match="matrix"):
        world_method_rows(bad, audit, [5])
    bad = deepcopy(report)
    bad["methods"][0]["properties"].append(bad["methods"][0]["properties"][0])
    with pytest.raises(ValueError, match="matrix"):
        world_method_rows(bad, audit, [5])
    bad = deepcopy(audit)
    bad["methods"][0]["states"] = 3
    with pytest.raises(ValueError, match="Method"):
        world_method_rows(report, bad, [5])
