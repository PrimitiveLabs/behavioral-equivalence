from copy import deepcopy
from itertools import product
import pytest
from behavioral_equivalence.diversity_analysis import (
    diversity_world_means,
    summarize_diversity,
    SPANS,
    CONTROLS,
    METRICS,
)
from behavioral_equivalence.environment_diversity import run_diversity_world


@pytest.fixture(scope="module")
def invariant_world():
    return run_diversity_world("trust", 32, 40000, "negative_control")[0]


def test_actual_invariant_world_means_preserve_all_controls(invariant_world):
    means = diversity_world_means(invariant_world)
    assert len(means) == 15
    assert max(r["mean_total_prediction_max_error"] for r in means) < 1e-12


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "capacity", "negative", "query"])
def test_world_summary_rejects_incomplete_or_invalid_evidence(invariant_world, mutation):
    report = deepcopy(invariant_world)
    if mutation == "missing":
        report["records"].pop()
    elif mutation == "duplicate":
        report["records"].append(report["records"][0])
    elif mutation == "capacity":
        for row in report["records"]:
            if row["span"] == "narrow" and row["control"] == "random_capacity":
                row["states"] = 32
    elif mutation == "negative":
        report["records"][0]["total_prediction_max_error"] = 0.01
    else:
        report["searches"][0]["query_count"] = 7
    with pytest.raises(ValueError):
        diversity_world_means(report)


def fixture_summaries():
    protocol = dict(
        families=["trust"],
        sizes=[32],
        seeds=[1, 2],
        surfaces=["linear", "negative_control"],
        expected_worlds=4,
    )
    rows = []
    for seed, surface in product(protocol["seeds"], protocol["surfaces"]):
        means = []
        for span, control in product(SPANS, CONTROLS):
            value = (
                0
                if surface == "negative_control"
                else 0.2 + 0.05 * seed - {"narrow": 0, "medium": 0.02, "broad": 0.05}[span]
            )
            states = 32 if control == "identity" else 8
            means.append(
                dict(
                    span=span,
                    control=control,
                    states=states,
                    **{"mean_" + m: value for m in METRICS},
                )
            )
        rows.append(
            dict(
                status="complete",
                spec=dict(family="trust", size=32, seed=seed, surface=surface),
                source_digest="a" * 64,
                certificate_replay_count=48,
                means=means,
            )
        )
    return rows, protocol


def test_paired_seed_clustering_and_sign_use_complete_design():
    rows, protocol = fixture_summaries()
    result = summarize_diversity(rows, protocol, "a" * 64)
    assert result["worlds"] == 4 and result["response_changing_worlds"] == 2
    assert result["negative_control_worlds"] == 2 and result["seed_clusters"] == 2
    comparison = result["comparisons"][0]
    assert comparison["paired_mean_difference"] == pytest.approx(-0.05)
    assert comparison["descriptive_seed_cluster_ci95"] == pytest.approx([-0.05, -0.05])
    assert result["certificates_replayed"] == 192


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "source", "replay", "status", "nonfinite", "negative"]
)
def test_aggregate_rejects_broken_matrix_or_replay_bindings(mutation):
    rows, protocol = fixture_summaries()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "source":
        rows[0]["source_digest"] = "b" * 64
    elif mutation == "replay":
        rows[0]["certificate_replay_count"] = 47
    elif mutation == "status":
        rows[0]["status"] = "failed"
    elif mutation == "nonfinite":
        rows[0]["means"][0]["mean_initial_completion_error"] = float("nan")
    else:
        rows[1]["means"][0]["mean_total_prediction_max_error"] = 0.01
    with pytest.raises(ValueError):
        summarize_diversity(rows, protocol, "a" * 64)


def test_aggregate_rejects_capacity_mismatch():
    rows, protocol = fixture_summaries()
    next(
        r for r in rows[0]["means"] if r["span"] == "narrow" and r["control"] == "random_capacity"
    )["states"] = 9
    with pytest.raises(ValueError):
        summarize_diversity(rows, protocol, "a" * 64)


def test_persisted_json_and_npz_certificate_replay_rejects_tampering():
    import io
    import json
    import numpy as np
    from behavioral_equivalence.diversity_analysis import replay_diversity_certificates

    report, arrays = run_diversity_world("trust", 32, 40001, "localized")
    persisted = json.loads(json.dumps(report))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        assert replay_diversity_certificates(persisted, saved) == 48
        persisted["certificates"][0]["certificate"]["bits"] += 1
        with pytest.raises(ValueError):
            replay_diversity_certificates(persisted, saved)


def test_persisted_replay_requires_all_context_bindings(invariant_world):
    from behavioral_equivalence.diversity_analysis import replay_diversity_certificates

    report = deepcopy(invariant_world)
    report["certificates"].pop()
    with pytest.raises(ValueError):
        replay_diversity_certificates(report, {})
