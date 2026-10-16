"""Generator endpoints and complete exploratory-diversity evidence gates."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from behavioral_equivalence.synthetic import FAMILIES, generate, intervene
from behavioral_equivalence.core import check_partition, quotient
from behavioral_equivalence.environment_diversity import diversity_catalogues, run_diversity_world
from behavioral_equivalence.diversity_analysis import diversity_world_means, summarize_diversity


@pytest.mark.parametrize(
    "family,size,perturbation",
    [
        ("bad", 16, 0),
        ("trust", 12, 0),
        ("trust", 18, 0),
        ("trust", 16, -0.1),
        ("trust", 16, 1.1),
        ("trust", 16, np.nan),
    ],
)
def test_generator_rejects_unsupported_configuration(family, size, perturbation):
    with pytest.raises(ValueError):
        generate(family, size, 7, perturbation)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("size", [16, 20])
def test_smallest_and_odd_base_mechanisms_have_exact_ground_truth_and_valid_extreme_perturbation(
    family, size
):
    world = generate(family, size, 7)
    check_partition(world.chain, world.partition)
    _, radius = quotient(world.chain, world.partition)
    assert radius == 0
    changed = generate(family, size, 7, 1)
    assert changed.parameters == world.parameters
    assert world.chain.labels == changed.chain.labels
    np.testing.assert_allclose(changed.chain.transitions.sum(axis=1), 1)
    active = np.array(world.chain.labels) == "active"
    assert np.all(np.count_nonzero(changed.chain.transitions[active], axis=1) == 1)
    np.testing.assert_array_equal(
        changed.chain.transitions[~active], world.chain.transitions[~active]
    )
    assert world.chain.digest() == generate(family, size, 7).chain.digest()


@pytest.mark.parametrize("strength", [-0.1, 1.1, np.nan, np.inf])
def test_scalar_intervention_rejects_invalid_strength(strength):
    with pytest.raises(ValueError, match="strength"):
        intervene(generate("trust", 16, 7), strength)


def test_intervention_endpoints_preserve_original_and_expose_duplicate_identity():
    world = generate("trust", 16, 7)
    before = world.chain.transitions.copy()
    np.testing.assert_array_equal(intervene(world, 0).transitions, before)
    changed = intervene(world, 1)
    complete = world.chain.labels.index("complete")
    disengage = world.chain.labels.index("disengage")
    for state, label in enumerate(world.chain.labels):
        if label == "active":
            target = complete if state % 4 < 2 else disengage
            assert changed.transitions[state, target] == 1
            assert np.count_nonzero(changed.transitions[state]) == 1
        else:
            np.testing.assert_array_equal(changed.transitions[state], before[state])
    np.testing.assert_array_equal(world.chain.transitions, before)


def test_frozen_diversity_geometry_guard_rejects_changed_catalogue(monkeypatch):
    monkeypatch.setattr(
        "behavioral_equivalence.environment_diversity.environment_catalogues",
        lambda: (np.zeros((64, 2)), np.zeros((16, 2))),
    )
    with pytest.raises(ValueError, match="geometry changed"):
        diversity_catalogues()


@pytest.fixture(scope="module")
def world_report():
    return run_diversity_world("trust", 16, 7, "negative_control")[0]


@pytest.mark.parametrize(
    "mutation",
    [
        "status",
        "horizon",
        "state_type",
        "state_zero",
        "nonfinite",
        "certificate",
        "search_missing",
        "duplicate_query",
        "outside_query",
        "partition_changes",
        "identity",
    ],
)
def test_diversity_world_rejects_invalid_evidence_constraints(world_report, mutation):
    report = deepcopy(world_report)
    if mutation == "status":
        report["status"] = "running"
    elif mutation == "horizon":
        report["records"][0]["horizon"] = 5
    elif mutation == "state_type":
        report["records"][0]["states"] = 4.5
    elif mutation == "state_zero":
        report["records"][0]["states"] = 0
    elif mutation == "nonfinite":
        report["records"][0]["abstraction_max_error"] = np.nan
    elif mutation == "certificate":
        report["certificates"].pop()
    elif mutation == "search_missing":
        report["searches"].pop()
    elif mutation == "duplicate_query":
        report["searches"][0]["selected_points"][1] = report["searches"][0]["selected_points"][0]
    elif mutation == "outside_query":
        report["searches"][0]["selected_points"][0] = [9, 9]
    elif mutation == "partition_changes":
        next(r for r in report["records"] if r["span"] == "narrow" and r["control"] == "unchanged")[
            "states"
        ] += 1
    else:
        for row in report["records"]:
            if row["control"] == "identity":
                row["states"] = 1
    with pytest.raises(ValueError):
        diversity_world_means(report)


def aggregate_fixture():
    source = Path(__file__).with_name("test_diversity_analysis.py")
    spec = importlib.util.spec_from_file_location("diversity_boundary_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture_summaries()


@pytest.mark.parametrize(
    "mutation",
    ["world_count", "missing_means", "bad_state", "identity", "single_seed", "no_changing_worlds"],
)
def test_diversity_aggregate_rejects_missing_design_capacity_and_seed_support(mutation):
    rows, protocol = aggregate_fixture()
    if mutation == "world_count":
        protocol["expected_worlds"] = 5
    elif mutation == "missing_means":
        rows[0]["means"].pop()
    elif mutation == "bad_state":
        rows[0]["means"][0]["states"] = 0
    elif mutation == "identity":
        for row in rows[0]["means"]:
            if row["control"] == "identity":
                row["states"] = 8
    elif mutation == "single_seed":
        protocol["seeds"] = [1]
        protocol["expected_worlds"] = 2
        rows = [r for r in rows if r["spec"]["seed"] == 1]
    else:
        protocol["surfaces"] = ["negative_control"]
        protocol["expected_worlds"] = 2
        rows = [r for r in rows if r["spec"]["surface"] == "negative_control"]
    with pytest.raises(ValueError):
        summarize_diversity(rows, protocol, "a" * 64)
