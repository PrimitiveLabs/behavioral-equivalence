from copy import deepcopy
from itertools import product
import pytest
from behavioral_equivalence.intervention_analysis import summarize_interventions, CONTROLS
from behavioral_equivalence.catalogue_search import SEARCH_METHODS


def fixture():
    protocol = dict(
        source_digest="s",
        protocol_digest="p",
        campaign_id="c",
        families=["trust"],
        sizes=[16],
        seeds=[0, 1],
        surfaces=["linear", "localized", "oscillatory", "negative_control"],
        horizons=[10],
        primary_horizon=10,
        expected_worlds=8,
        environment_budget=8,
    )
    reports = []
    for seed, surface in product(protocol["seeds"], protocol["surfaces"]):
        means = []
        states = dict(
            unchanged=4,
            targeted_witness=8,
            random_capacity=8,
            random_witness_same_data=6,
            all_observed=10,
            identity=16,
        )
        for method, control in product(SEARCH_METHODS, CONTROLS):
            error = (
                0
                if surface == "negative_control"
                else (0.1 + 0.01 * seed if control == "targeted_witness" else 0.2 + 0.03 * seed)
            )
            means.append(
                dict(
                    method=method,
                    control=control,
                    horizon=10,
                    states=states[control],
                    heldout_environments=16,
                    mean_total_prediction_max_error=error,
                    mean_abstraction_max_error=error / 2,
                    mean_environment_model_max_error=error / 2,
                    mean_initial_completion_error=error / 3,
                )
            )
        searches = [
            dict(method=m, indices=list(range(8)), winner=1, environment_evaluations=8)
            for m in SEARCH_METHODS
        ]
        reports.append(
            dict(
                source_digest="s",
                protocol_digest="p",
                spec=dict(family="trust", size=16, seed=seed, surface=surface),
                means=means,
                searches=searches,
                original_states=4,
                verified_estimated_model_certificates=64,
            )
        )
    return dict(
        status="complete",
        phase="frozen multidimensional synthetic intervention",
        failures=[],
        source_digest="s",
        protocol_digest="p",
        results=reports,
        worlds=8,
        expected_worlds=8,
        verified_estimated_model_certificates=512,
    ), protocol


def test_seed_clusters_not_environment_or_property_rows_determine_uncertainty():
    manifest, protocol = fixture()
    result = summarize_interventions(manifest, protocol)
    assert result["response_worlds"] == 8
    assert result["primary_response_changing_worlds"] == 6
    assert result["seed_clusters"] == 2
    assert result["negative_control_max_mean_prediction_error"] == 0
    assert len(result["comparisons"]) == 5
    for comparison in result["comparisons"]:
        assert comparison["paired_mean_difference"] == pytest.approx(-0.11)
        assert comparison["independent_seed_clusters"] == 2
        assert comparison["seed_cluster_bootstrap_interval"] == pytest.approx([-0.12, -0.1])
        assert [r["world_cells"] for r in comparison["seed_effects"]] == [3, 3]
    assert result["comparisons"][0]["interval_level"] == 0.95
    assert result["comparisons"][1]["interval_level"] == 0.9875


@pytest.mark.parametrize(
    "damage", ["world", "capacity", "budget", "negative", "certificate", "mean", "source", "smoke"]
)
def test_incomplete_controls_and_invalid_evidence_cannot_produce_summary(damage):
    manifest, protocol = fixture()
    manifest = deepcopy(manifest)
    report = manifest["results"][0]
    if damage == "world":
        manifest["results"].pop()
    elif damage == "capacity":
        next(r for r in report["means"] if r["control"] == "random_capacity")["states"] = 9
    elif damage == "budget":
        report["searches"][0]["indices"].pop()
    elif damage == "negative":
        manifest["results"][3]["means"][0]["mean_total_prediction_max_error"] = 0.1
    elif damage == "certificate":
        report["verified_estimated_model_certificates"] = 63
    elif damage == "mean":
        report["means"].pop()
    elif damage == "source":
        report["source_digest"] = "changed"
    else:
        manifest["phase"] = "smoke"
    with pytest.raises(ValueError):
        summarize_interventions(manifest, protocol)
