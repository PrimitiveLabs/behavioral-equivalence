import numpy as np
import pytest
from behavioral_equivalence.decision_model import decision_chain
from behavioral_equivalence.learned_abstraction import (
    certified_latent_refinement,
    next_positive_probabilities,
)
from behavioral_equivalence.integer_bounds import verify_abstraction


def test_latent_merge_is_nested_label_preserving_and_exactly_certified():
    transitions = np.full((2, 2, 2), 0.5)
    parent = decision_chain(
        transitions, np.array([[0.75, 0.25], [0.75, 0.25]]), np.array([0.5, 0.5])
    )
    abstract, blocks, evidence = certified_latent_refinement(parent, ((0, 1),), 0)
    assert blocks == ((0, 1),)
    assert abstract.size == 3 and evidence["latent_compression_ratio"] == 2
    verify_abstraction(parent, abstract, evidence["certificate"])
    assert np.array_equal(next_positive_probabilities(abstract, blocks), [0.25, 0.25])
    assert evidence["certificate"]["horizons"] == [1, 9, 19, 49, 99]


def test_distinct_emission_states_split_even_when_latent_embeddings_are_identical():
    parent = decision_chain(
        np.full((2, 2, 2), 0.5), np.array([[0.75, 0.25], [0.25, 0.75]]), np.array([0.5, 0.5])
    )
    abstract, blocks, evidence = certified_latent_refinement(parent, ((0, 1),), 0.01)
    assert len(blocks) == 2
    assert evidence["reduced_latent_states"] == 2
    assert len(evidence["refinement_trace"]) == 2
    assert np.array_equal(next_positive_probabilities(abstract, blocks), [0.25, 0.75])
    verify_abstraction(parent, abstract, evidence["certificate"])
    with pytest.raises(ValueError, match="epsilon"):
        certified_latent_refinement(parent, ((0, 1),), float("nan"))


def test_zero_budget_refines_rounded_successor_aggregation_from_singleton_origin():
    transitions = np.array(
        [[[0.25, 0.25, 0.5]] * 2, [[0.25, 0.25, 0.5]] * 2, [[0.1, 0.2, 0.7]] * 2]
    )
    parent = decision_chain(
        transitions, np.array([[0.75, 0.25], [0.75, 0.25], [0.5, 0.5]]), np.array([0.25, 0.25, 0.5])
    )
    abstract, blocks, evidence = certified_latent_refinement(parent, ((0, 1), (2,)), 0)
    assert evidence["refinement_trace"][0]["witness_latent"] == 2
    assert evidence["refinement_trace"][0]["epsilon_for_display"] > 0
    assert len(blocks) == 3
    verify_abstraction(parent, abstract, evidence["certificate"])
    with pytest.raises(ValueError, match="labels"):
        next_positive_probabilities(abstract, ((0, 1, 2),))
