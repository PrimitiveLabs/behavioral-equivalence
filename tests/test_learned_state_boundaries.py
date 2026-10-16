"""Sparse point estimation and exact acceptance of learned-state reductions."""

import numpy as np
import pytest
from behavioral_equivalence.core import Chain
from behavioral_equivalence.decision_model import decision_chain
from behavioral_equivalence.latent_estimation import (
    adjacent_transition_counts,
    point_transitions,
    equal_user_initial_mass,
)
from behavioral_equivalence.quantization import (
    fit_representation_scaling,
    assign_centers,
    emission_probabilities,
)
from behavioral_equivalence.learned_abstraction import (
    certified_latent_refinement,
    next_positive_probabilities,
)
from behavioral_equivalence.property_refinement import property_guided_latent_refinement
from behavioral_equivalence.property_bounds import LabelEvent, verify_properties


@pytest.mark.parametrize(
    "change",
    [
        dict(states=[[0, 1]]),
        dict(targets=[1]),
        dict(users=[1]),
        dict(bundle_indices=[1]),
        dict(current=[[0]]),
        dict(following=[]),
        dict(current=[0.0]),
        dict(following=[1.0]),
        dict(states=[0.0, 1.0]),
        dict(targets=[1, 2]),
        dict(state_count=0),
        dict(states=[-1, 1]),
        dict(states=[0, 2]),
        dict(current=[-1]),
        dict(following=[2]),
    ],
)
def test_transition_counts_require_supported_aligned_states_and_pair_indices(change):
    args = dict(
        states=[0, 1],
        targets=[1, 0],
        users=[1, 1],
        bundle_indices=[1, 2],
        current=[0],
        following=[1],
        state_count=2,
    )
    args.update(change)
    with pytest.raises(ValueError, match="Aligned discrete states"):
        adjacent_transition_counts(**args)


def test_empty_transition_pair_selection_adds_no_invented_observations():
    counts = adjacent_transition_counts(
        [0, 1], [1, 0], [1, 1], [1, 2], np.array([], dtype=int), np.array([], dtype=int), 2
    )
    assert counts.shape == (2, 2, 2) and counts.sum() == 0
    for prior in [0.1, 2.0]:
        np.testing.assert_allclose(point_transitions(counts, prior), 0.5)
    counts[0, 1, 0] = 2
    np.testing.assert_allclose(point_transitions(counts, 2)[0, 1], [0.75, 0.25])
    assert counts[0, 1, 0] == 2


@pytest.mark.parametrize(
    "counts,prior",
    [
        (np.zeros((2, 2), dtype=int), 1),
        (np.zeros((0, 2, 0), dtype=int), 1),
        (np.zeros((2, 3, 2), dtype=int), 1),
        (np.zeros((2, 2, 2)), 1),
        (-np.ones((2, 2, 2), dtype=int), 1),
        (np.zeros((2, 2, 2), dtype=int), 0),
        (np.zeros((2, 2, 2), dtype=int), np.nan),
    ],
)
def test_nominal_point_model_count_and_prior_boundaries(counts, prior):
    with pytest.raises(ValueError, match="positive nominal prior"):
        point_transitions(counts, prior)


@pytest.mark.parametrize(
    "states,users",
    [
        ([[0, 1]], [[1, 2]]),
        ([0, 1], [1]),
        ([0.0, 1.0], [1, 2]),
        ([-1, 1], [1, 2]),
        ([0, 2], [1, 2]),
    ],
)
def test_equal_user_initial_mass_requires_supported_aligned_states(states, users):
    with pytest.raises(ValueError, match="supported initial"):
        equal_user_initial_mass(np.asarray(states), np.asarray(users), 2)


@pytest.mark.parametrize(
    "training", [np.ones(2), np.empty((0, 2)), np.empty((2, 0)), np.array([[np.inf]])]
)
def test_representation_scaling_requires_nonempty_finite_training(training):
    with pytest.raises(ValueError, match="Finite nonempty training"):
        fit_representation_scaling(training)


@pytest.mark.parametrize(
    "change",
    [
        dict(representations=np.ones(2)),
        dict(centers=np.ones(2)),
        dict(centers=np.empty((0, 2))),
        dict(centers=np.ones((2, 3))),
        dict(mean=np.zeros(1)),
        dict(scale=np.ones(1)),
        dict(representations=np.array([[np.nan, 0]])),
        dict(mean=np.array([np.inf, 0])),
        dict(centers=np.array([[0, np.nan]])),
        dict(scale=np.array([-1, 1])),
    ],
)
def test_center_assignment_alignment_finiteness_and_positive_scale(change):
    args = dict(
        representations=np.ones((1, 2)),
        mean=np.zeros(2),
        scale=np.ones(2),
        centers=np.zeros((1, 2)),
    )
    args.update(change)
    with pytest.raises(ValueError, match="Aligned finite representations"):
        assign_centers(**args)


@pytest.mark.parametrize(
    "counts,sums,global_p",
    [
        ([[1]], [[0.5]], 0.5),
        ([1], [0.5, 0.5], 0.5),
        ([1.0], [0.5], 0.5),
        ([-1], [0], 0.5),
        ([1], [np.nan], 0.5),
        ([1], [-0.1], 0.5),
        ([1], [0.5], np.nan),
        ([1], [0.5], -0.1),
        ([1], [0.5], 1.1),
    ],
)
def test_teacher_emission_count_sum_and_fallback_boundaries(counts, sums, global_p):
    with pytest.raises(ValueError, match="Valid training counts"):
        emission_probabilities(np.array(counts), np.array(sums), global_p)


@pytest.mark.parametrize("global_p", [0, 1])
def test_unsupported_teacher_emissions_keep_declared_fallback_endpoints(global_p):
    np.testing.assert_array_equal(
        emission_probabilities(np.zeros(2, dtype=int), np.zeros(2), global_p), [global_p, global_p]
    )


@pytest.fixture
def parent():
    return decision_chain(
        np.full((2, 2, 2), 0.5), np.array([[0.75, 0.25], [0.25, 0.75]]), np.array([0.5, 0.5])
    )


@pytest.mark.parametrize("budget", [-0.1, 1.1, np.inf, np.nan])
@pytest.mark.parametrize("method", ["strict", "property"])
def test_latent_refinement_rejects_invalid_probability_budget(parent, budget, method):
    with pytest.raises(ValueError):
        if method == "strict":
            certified_latent_refinement(parent, ((0, 1),), budget, (1,))
        else:
            property_guided_latent_refinement(
                parent, ((0, 1),), (LabelEvent("occupancy", ("positive",), 1),), budget
            )


def test_latent_refinements_require_decision_layout_and_valid_decision_horizons(parent):
    invalid = Chain(np.eye(2), ("a", "b"), np.array([1, 0]))
    with pytest.raises(ValueError, match="decision chain"):
        certified_latent_refinement(invalid, ((0,), (1,)), 0)
    with pytest.raises(ValueError, match="decision chain"):
        property_guided_latent_refinement(
            invalid, ((0,), (1,)), (LabelEvent("occupancy", ("positive",), 1),), 0
        )
    with pytest.raises(ValueError, match="Decision count"):
        certified_latent_refinement(parent, ((0,), (1,)), 0, (-1,))
    with pytest.raises(ValueError, match="partition parent indices"):
        next_positive_probabilities(parent, ((0,), (2,)))


@pytest.mark.parametrize("horizon,first_acceptance", [(3, "refine"), (5, "not attempted")])
def test_float_screening_never_overrides_exact_acceptance_and_singleton_witness_fallback(
    horizon, first_acceptance
):
    transitions = np.array(
        [[[0.25, 0.25, 0.5]] * 2, [[0.25, 0.25, 0.5]] * 2, [[0.1, 0.2, 0.7]] * 2]
    )
    parent = decision_chain(
        transitions, np.array([[0.75, 0.25], [0.75, 0.25], [0.5, 0.5]]), np.array([0.25, 0.25, 0.5])
    )
    q, blocks, result = property_guided_latent_refinement(
        parent, ((0, 1), (2,)), (LabelEvent("occupancy", ("positive",), horizon),), 0
    )
    trace = result["refinement_trace"]
    assert len(blocks) == 3 and len(trace) == 2
    assert trace[0]["numeric_acceptance"] == first_acceptance
    if horizon == 3:
        assert trace[0]["screening_error"] == 0
        assert trace[0]["verified_max_error_for_display"] > 0
    else:
        assert trace[0]["witness_state"] == 6
    assert trace[-1]["numeric_acceptance"] == "passed"
    assert trace[-1]["verified_max_error_for_display"] == 0
    verify_properties(parent, q, result["certificate"])


def test_dtype_fix_preserves_frozen_python_float_callers_and_frozen_source_identity():
    import hashlib
    import json
    from pathlib import Path

    fixture = json.loads(
        Path("data/synthetic/fixtures/release/quantization-dtype-v1.json").read_text()
    )
    assert hashlib.sha256(fixture["source_module"].encode()).hexdigest() == fixture["source_sha256"]
    assert (
        fixture["source_sha256"]
        == "79be6782bfe31d0ae16b6998a7b9b5bf1c757b33612504da0143f98af71c95ec"
    )
    scope = {}
    exec(compile(fixture["source_module"], "frozen-quantization-module", "exec"), scope)
    old_emission = scope["emission_probabilities"]
    rng = np.random.default_rng(20261016)
    for probability in [0.0, 0.4, 1.0]:
        for _ in range(20):
            counts = rng.integers(0, 100, 16)
            counts[0] = 0
            sums = counts * rng.uniform(0, 1, 16)
            before = old_emission(counts, sums, probability)
            after = emission_probabilities(counts, sums, probability)
            assert before.dtype == after.dtype == np.dtype("float64")
            np.testing.assert_array_equal(before, after)
    for probability in [0, 1]:
        with pytest.raises(TypeError):
            old_emission(np.zeros(2, dtype=int), np.zeros(2), probability)
        fixed = emission_probabilities(np.zeros(2, dtype=int), np.zeros(2), probability)
        assert fixed.dtype == np.dtype("float64")
        np.testing.assert_array_equal(fixed, [probability, probability])
