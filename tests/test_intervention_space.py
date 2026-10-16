import numpy as np
import pytest
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.core import quotient, reachability
from behavioral_equivalence.intervention_space import (
    environment_chain,
    environment_catalogues,
    interpolate_environment,
    SURFACES,
)
from behavioral_equivalence.refinement import outcome_distribution, js_divergence
from behavioral_equivalence.catalogue_search import (
    maximum_within_block_js,
    search_catalogue,
    random_refinement_to_count,
    SEARCH_METHODS,
)


def test_environment_validity_neutral_origin_and_structural_negative_control():
    for family in ["preference", "belief", "trust", "patience", "memory"]:
        world = generate(family, 16, 33)
        reference = outcome_distribution(world.chain, 10)
        for surface in SURFACES:
            neutral = environment_chain(world, (0.0, 0.0), surface)
            np.testing.assert_array_equal(neutral.transitions, world.chain.transitions)
            changed = environment_chain(world, (0.7, -0.6), surface)
            np.testing.assert_allclose(changed.transitions.sum(axis=1), 1, atol=1e-14)
            if surface == "negative_control":
                assert not np.array_equal(changed.transitions, world.chain.transitions)
                np.testing.assert_allclose(outcome_distribution(changed, 10), reference, atol=1e-14)
                assert quotient(changed, world.partition)[1] < 1e-14
            else:
                assert (
                    maximum_within_block_js(outcome_distribution(changed, 10), world.partition)[0]
                    > 0
                )
    with pytest.raises(ValueError, match="lumpable"):
        environment_chain(generate("trust", 16, 7, 0.01), (0.2, 0.4), "negative_control")
    for point in [(2.0, 0.0), (float("nan"), 0.0)]:
        with pytest.raises(ValueError):
            environment_chain(world, point, "linear")


def test_interpolation_uses_only_queried_kernels_and_reproduces_observed_points():
    world = generate("trust", 16, 7)
    points = np.array([[0.0, 0.0], [0.5, 0.5]])
    observations = [world.chain, environment_chain(world, (0.5, 0.5), "localized")]
    assert interpolate_environment(points, observations, (0.0, 0.0)) is observations[0]
    estimate = interpolate_environment(points, observations, (0.25, 0.25))
    np.testing.assert_allclose(
        estimate.transitions, (observations[0].transitions + observations[1].transitions) / 2
    )
    assert np.all(np.isfinite(reachability(estimate, "complete", 10)))
    with pytest.raises(ValueError):
        interpolate_environment(points[:1], observations, (0.0, 0.0))


def test_search_budget_and_disjoint_catalogues_without_oracle_scores():
    points, heldout = environment_catalogues()
    assert len(points) == 64 and len(heldout) == 16
    assert not any(np.array_equal(a, b) for a in points for b in heldout)
    for method in SEARCH_METHODS:
        called = []

        def evaluate(i):
            called.append(i)
            return float(0.5 * np.exp(-np.sum((points[i] - [0.3, -0.4]) ** 2)))

        result = search_catalogue(points, evaluate, method, 8, 13)
        assert len(called) == len(set(called)) == result["environment_evaluations"] == 8
        assert result["indices"] == called
        assert result["winner"] in called
        again = search_catalogue(points, evaluate, method, 8, 13)
        assert again == result
    with pytest.raises(ValueError):
        search_catalogue(points, lambda i: float("nan"), "random", 8, 1)
    with pytest.raises(ValueError):
        search_catalogue(points, lambda i: 0.0, "random", 65, 1)


def test_vectorized_js_agrees_with_individual_witness_and_random_capacity_control():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
    blocks = ((0, 1), (2, 3))
    score, pair = maximum_within_block_js(p, blocks)
    assert pair == (0, 1)
    assert score == pytest.approx(js_divergence(p[0], p[1]))
    refined = random_refinement_to_count(blocks, 3, 11)
    assert len(refined) == 3
    assert sorted(s for block in refined for s in block) == list(range(4))
    assert all(any(set(block) <= set(parent) for parent in blocks) for block in refined)
    with pytest.raises(ValueError):
        random_refinement_to_count(blocks, 5, 1)
