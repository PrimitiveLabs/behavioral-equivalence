"""Search feasibility and separation of environment versus abstraction error."""

import numpy as np
import pytest
from behavioral_equivalence.core import Chain, exact_partition
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.refinement import (
    js_divergence,
    outcome_distribution,
    search_witness,
    budgeted_witness,
)
from behavioral_equivalence.intervention_space import environment_chain, interpolate_environment
from behavioral_equivalence.catalogue_search import (
    maximum_within_block_js,
    search_catalogue,
    random_refinement_to_count,
)
from behavioral_equivalence.intervention_benchmark import run_intervention_world


@pytest.mark.parametrize(
    "left,right",
    [([1, 0], [1]), ([-0.1, 1.1], [0.5, 0.5]), ([1, 0], [0.5, 0.1]), ([np.nan, 0], [1, 0])],
)
def test_js_rejects_invalid_distributions(left, right):
    with pytest.raises(ValueError):
        js_divergence(np.array(left), np.array(right))


def test_outcome_catalogue_rejects_nonabsorbing_terminal_labels():
    chain = Chain(np.array([[0, 1], [0, 1.0]]), ("complete", "active"), np.array([1, 0]))
    with pytest.raises(ValueError, match="absorbing terminal"):
        outcome_distribution(chain, 1)


def test_witness_requires_feasible_query_and_a_merged_pair():
    world = generate("trust", 16, 7)
    blocks = exact_partition([world.chain])
    with pytest.raises(ValueError, match="nonempty"):
        search_witness(world, blocks, [], 1)
    with pytest.raises(ValueError, match="No merged states"):
        search_witness(world, tuple((i,) for i in range(16)), [0.1], 1)


@pytest.mark.parametrize(
    "method,budget,maximum",
    [
        ("bad", 2, 0.3),
        ("random", 1, 0.3),
        ("grid", 2, 0),
        ("adaptive", 2, 1.1),
        ("random", 2, np.nan),
    ],
)
def test_scalar_budgeted_search_feasibility(method, budget, maximum):
    world = generate("trust", 16, 7)
    with pytest.raises(ValueError, match="Supported search"):
        budgeted_witness(world, world.partition, method, budget, 1, maximum_strength=maximum)


@pytest.mark.parametrize(
    "point,surface",
    [((0,), "linear"), ((0, 0), "bad"), ((0, np.inf), "linear"), ((-1.1, 0), "localized")],
)
def test_environment_point_and_surface_boundaries(point, surface):
    with pytest.raises(ValueError, match="documented surface"):
        environment_chain(generate("trust", 16, 7), point, surface)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "point_shape",
        "points_nan",
        "query_inf",
        "zero_bandwidth",
        "nan_bandwidth",
        "labels",
        "initial",
    ],
)
def test_interpolation_requires_compatible_finite_observed_models(case):
    world = generate("trust", 16, 7)
    points = np.array([[0.0, 0.0]])
    observed, query, bandwidth = [world.chain], (0.2, 0.2), 0.5
    if case == "empty":
        points, observed = np.empty((0, 2)), []
    elif case == "point_shape":
        points = np.zeros((1, 1))
    elif case == "points_nan":
        points[0, 0] = np.nan
    elif case == "query_inf":
        query = (np.inf, 0)
    elif case == "zero_bandwidth":
        bandwidth = 0
    elif case == "nan_bandwidth":
        bandwidth = np.nan
    else:
        points = np.array([[0.0, 0.0], [0.1, 0.1]])
        labels = (
            tuple("changed" for _ in world.chain.labels) if case == "labels" else world.chain.labels
        )
        initial = np.roll(world.chain.initial, 4) if case == "initial" else world.chain.initial
        observed.append(Chain(world.chain.transitions, labels, initial))
    with pytest.raises(ValueError, match="Aligned observed kernels"):
        interpolate_environment(points, observed, query, bandwidth)


@pytest.mark.parametrize(
    "probabilities,blocks",
    [
        (np.ones(2), ((0, 1),)),
        (np.array([[np.nan, 0], [1, 0]]), ((0, 1),)),
        (np.array([[-1, 2], [1, 0]]), ((0, 1),)),
        (np.array([[0.1, 0.2], [1, 0]]), ((0, 1),)),
        (np.eye(2), ((0,), (1,))),
        (np.eye(2), ((0, 0), (1,))),
    ],
)
def test_vectorized_disagreement_requires_distributions_and_merged_partition(probabilities, blocks):
    with pytest.raises(ValueError):
        maximum_within_block_js(probabilities, blocks)


@pytest.mark.parametrize(
    "change",
    [
        dict(method="bad"),
        dict(budget=0),
        dict(budget=-1),
        dict(budget=True),
        dict(budget=1.5),
        dict(budget=3),
        dict(points=np.zeros(2)),
        dict(points=np.zeros((2, 3))),
        dict(points=np.zeros((2, 2))),
        dict(points=np.array([[0, 0], [np.nan, 1]])),
    ],
)
def test_catalogue_validation_precedes_any_evaluator_call(change):
    args = dict(points=np.array([[0.0, 0.0], [1.0, 1.0]]), method="random", budget=2, seed=1)
    args.update(change)

    def forbidden(_):
        pytest.fail("Invalid catalogue must not query an environment")

    with pytest.raises(ValueError, match="Distinct finite catalogue"):
        search_catalogue(evaluate=forbidden, **args)


@pytest.mark.parametrize("score", [-0.1, 1.0, np.inf, np.nan])
def test_catalogue_rejects_invalid_evaluator_score_without_extra_queries(score):
    called = []

    def evaluate(index):
        called.append(index)
        return score

    with pytest.raises(ValueError, match="finite Jensen-Shannon"):
        search_catalogue(np.array([[0.0, 0.0], [1, 1]]), evaluate, "random", 2, 1)
    assert len(called) == 1


def test_search_propagates_environment_error_and_capacity_endpoints_preserve_partition():
    def unavailable(_):
        raise RuntimeError("environment unavailable")

    with pytest.raises(RuntimeError, match="environment unavailable"):
        search_catalogue(np.array([[0.0, 0.0]]), unavailable, "random", 1, 1)
    blocks = ((0, 1), (2, 3))
    assert random_refinement_to_count(blocks, 2, 1) == blocks
    assert all(len(b) == 1 for b in random_refinement_to_count(blocks, 4, 1))
    with pytest.raises(ValueError, match="Target"):
        random_refinement_to_count(blocks, 1, 1)


@pytest.mark.parametrize("surface", ["linear", "negative_control"])
def test_small_complete_intervention_benchmark_preserves_data_and_error_scopes(surface):
    report, artifacts = run_intervention_world("trust", 16, 7, surface, budget=2, horizons=(1,))
    assert report["status"] == "complete" and report["environment_budget"] == 2
    assert len(report["records"]) == 4 * 16 * 6
    assert len(artifacts["certificates"]) == report["verified_estimated_model_certificates"] == 64
    for search in report["searches"]:
        assert len(set(search["indices"])) == search["environment_evaluations"] == 2
        assert search["winner"] in search["indices"]
        maps = search["partition_maps"]
        assert len(maps["random_capacity"]) == len(maps["targeted_witness"])
        source = next(
            r
            for r in artifacts["retained_models"]
            if r["method"] == search["method"] and "observed_points" in r
        )
        np.testing.assert_array_equal(
            source["observed_points"][1:], np.array(report["search_catalogue"])[search["indices"]]
        )
        assert source["observed_points"][0] == [0, 0]
    for row in report["records"]:
        assert (
            row["total_prediction_max_error"]
            <= row["environment_model_max_error"] + row["abstraction_max_error"] + 1e-12
        )
        if row["control"] == "identity":
            assert row["abstraction_max_error"] < 1e-12
            assert row["total_prediction_max_error"] == pytest.approx(
                row["environment_model_max_error"]
            )
        if surface == "negative_control":
            assert row["total_prediction_max_error"] < 1e-12
