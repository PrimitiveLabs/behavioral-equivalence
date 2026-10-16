"""Analysis summaries reject mismatched evidence and preserve negative results."""

from copy import deepcopy

import numpy as np
import pytest

from behavioral_equivalence.intervention_analysis import summarize_interventions
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.synthetic_analysis import world_method_rows
from behavioral_equivalence.uncertainty_repetition import (
    row_sample_schedule,
    terminal_label_support,
)
from behavioral_equivalence.uncertainty import IntervalChain
from test_intervention_analysis import fixture as intervention_fixture
from test_synthetic_analysis import fixture as synthetic_fixture


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "audit_status",
        "digest",
        "spec",
        "method_count",
        "index",
        "method",
        "verified",
        "bounds",
    ],
)
def test_synthetic_summary_requires_matching_complete_audit(damage):
    report, audit = synthetic_fixture()
    if damage == "status":
        report["status"] = "partial"
    elif damage == "audit_status":
        audit["status"] = "partial"
    elif damage == "digest":
        audit["world_digest"] = "changed"
    elif damage == "spec":
        audit["spec"] = {"family": "other"}
    elif damage == "method_count":
        audit["methods"].clear()
    elif damage == "index":
        audit["methods"][0]["index"] = 1
    elif damage == "method":
        audit["methods"][0]["method"] = "other"
    elif damage == "verified":
        audit["methods"][0]["verified"] = 1
    else:
        report["methods"][0]["properties"][0]["bound"] = 0.1
    with pytest.raises(ValueError):
        world_method_rows(report, audit, [5])


def test_synthetic_summary_keeps_failed_bounds_and_conditional_failures_separate():
    report, audit = synthetic_fixture()
    report["methods"][0]["properties"][0]["violated"] = True
    report["methods"][0]["properties"][-1]["violated"] = True
    _, rows = world_method_rows(report, audit, [5])
    assert rows[0]["unconditional_violations"] == 1
    assert rows[0]["conditional_violation"] is True


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_world",
        "duplicate_mean",
        "nan",
        "search_duplicate",
        "winner",
        "identity",
        "total_certificates",
        "protocol",
    ],
)
def test_intervention_analysis_rejects_invalid_design_evidence(damage):
    manifest, protocol = intervention_fixture()
    report = manifest["results"][0]
    if damage == "duplicate_world":
        manifest["results"][1] = deepcopy(report)
    elif damage == "duplicate_mean":
        report["means"].append(deepcopy(report["means"][0]))
    elif damage == "nan":
        report["means"][0]["mean_abstraction_max_error"] = np.nan
    elif damage == "search_duplicate":
        report["searches"][0]["indices"][0] = 1
    elif damage == "winner":
        report["searches"][0]["winner"] = 100
    elif damage == "identity":
        next(r for r in report["means"] if r["control"] == "identity")["states"] = 15
    elif damage == "total_certificates":
        manifest["verified_estimated_model_certificates"] -= 1
    else:
        manifest["protocol_digest"] = "other"
    with pytest.raises(ValueError):
        summarize_interventions(manifest, protocol)


def test_intervention_analysis_preserves_candidate_worse_than_baseline():
    manifest, protocol = intervention_fixture()
    for report in manifest["results"]:
        if report["spec"]["surface"] == "negative_control":
            continue
        for row in report["means"]:
            if row["control"] == "targeted_witness":
                row["mean_total_prediction_max_error"] = 0.9
    summary = summarize_interventions(manifest, protocol)
    assert all(r["paired_mean_difference"] > 0 for r in summary["comparisons"])
    assert all(r["seed_cluster_bootstrap_interval"][0] > 0 for r in summary["comparisons"])


@pytest.mark.parametrize(
    "regime,budget",
    [("adaptive", 10), ("uniform_active", 0), ("imbalanced_active", -1), ("uniform_active", 1.5)],
)
def test_invalid_fixed_row_schedule(regime, budget):
    with pytest.raises(ValueError):
        row_sample_schedule(generate("memory", 32, 0, 0).chain, regime, budget)


def test_terminal_support_does_not_mutate_input_and_requires_matching_labels():
    chain = generate("memory", 32, 0, 0).chain
    unknown = IntervalChain(np.zeros((32, 32)), np.ones((32, 32)), chain.labels)
    before = unknown.upper.copy()
    terminal_label_support(unknown, chain)
    np.testing.assert_array_equal(unknown.upper, before)
    wrong = IntervalChain(unknown.lower, unknown.upper, ("other",) * 32)
    with pytest.raises(ValueError, match="Matching labels"):
        terminal_label_support(wrong, chain)
