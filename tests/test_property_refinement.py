import numpy as np
from behavioral_equivalence.decision_model import decision_chain
from behavioral_equivalence.property_bounds import LabelEvent, verify_properties
from behavioral_equivalence.property_refinement import property_guided_latent_refinement


def test_property_refinement_scopes_prediction_origins_and_splits_future_counterexample():
    transitions = np.zeros((3, 2, 3))
    transitions[0, :, 0] = 1
    transitions[1, :, 2] = 1
    transitions[2, :, 2] = 1
    parent = decision_chain(
        transitions, np.array([[0.75, 0.25], [0.75, 0.25], [0.25, 0.75]]), np.array([0.5, 0.5, 0])
    )
    first = (LabelEvent("occupancy", ("positive",), 1),)
    q, blocks, evidence = property_guided_latent_refinement(parent, ((0, 1), (2,)), first, 0)
    assert len(blocks) == 2
    assert evidence["certificate"]["concrete_origins"] == [0, 3, 6]
    verify_properties(parent, q, evidence["certificate"])
    future = first + (
        LabelEvent("occupancy", ("positive",), 3),
        LabelEvent("count_at_least", ("positive",), 3, 2),
    )
    q, blocks, evidence = property_guided_latent_refinement(parent, ((0, 1), (2,)), future, 0.01)
    assert len(blocks) == 3
    assert len(evidence["refinement_trace"]) == 2
    verify_properties(parent, q, evidence["certificate"])
