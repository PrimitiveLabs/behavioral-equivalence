import numpy as np
import pytest
from behavioral_equivalence.core import exact_partition, quotient
from behavioral_equivalence.synthetic import generate, intervene
from behavioral_equivalence.refinement import (
    search_witness,
    refine,
    js_divergence,
    outcome_distribution,
)


def test_environment_witness_and_independent_refinement_certificate():
    world = generate("trust", 32, 23)
    partition = exact_partition([world.chain])
    zero = search_witness(world, partition, [0.0], 5)
    assert zero.js_nats == pytest.approx(0, abs=1e-14)
    witness = search_witness(world, partition, [0.05, 0.15, 0.3], 5)
    assert witness.js_nats > 0 and witness.completion_gap > 0.1
    split = refine(world, witness, 0.01)
    assert quotient(world.chain, split)[1] <= 0.01
    assert quotient(intervene(world, witness.strength), split)[1] <= 0.01
    assert len(split) > len(partition)
    np.testing.assert_allclose(outcome_distribution(world.chain, 5).sum(axis=1), 1)


def test_js_known_limits_and_invalid_inputs():
    assert js_divergence(np.array([1.0, 0]), np.array([0.0, 1])) == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        js_divergence(np.array([0.1]), np.array([0.2]))


def test_search_methods_respect_actual_environment_budget(monkeypatch):
    import behavioral_equivalence.refinement as module

    world = generate("memory", 16, 9)
    blocks = exact_partition([world.chain])
    actual = module.search_witness
    visited = []

    def counted(world, partition, strengths, horizon):
        visited.extend(strengths)
        return actual(world, partition, strengths, horizon)

    monkeypatch.setattr(module, "search_witness", counted)
    for method in ["random", "grid", "adaptive"]:
        visited.clear()
        witness = module.budgeted_witness(world, blocks, method, 8, 5, seed=3)
        assert len(visited) == len(set(visited)) == 8
        assert witness.environment_evaluations == 8
