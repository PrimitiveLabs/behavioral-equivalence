import numpy as np
import pytest
from behavioral_equivalence.decision_model import (
    decision_chain,
    decision_partition,
    decision_horizon,
)
from behavioral_equivalence.core import quotient, reachability, coupling_bound
from behavioral_equivalence.rational import certify, verify


def test_emission_error_is_part_of_certificate_and_horizon_counts_decisions():
    transitions = np.full((2, 2, 2), 0.5)
    model = decision_chain(transitions, np.array([[0.75, 0.25], [0.5, 0.5]]), np.array([1.0, 0]))
    blocks = decision_partition(((0, 1),))
    reduced, epsilon = quotient(model, blocks)
    assert model.size == 6 and reduced.size == 3
    assert epsilon == 0.125
    for decisions in [1, 2, 5]:
        h = decision_horizon(decisions)
        full = reachability(model, "positive", h)
        abstract = reachability(reduced, "positive", h)
        for i, block in enumerate(blocks):
            assert (
                max(abs(full[s] - abstract[i]) for s in block) <= coupling_bound(epsilon, h) + 1e-14
            )
    certificate = certify(model, blocks, "positive", decision_horizon(3))
    verify(model, certificate)


def test_exact_emissions_and_stage_labels():
    transitions = np.array([[[1.0, 0], [0, 1]], [[0.5, 0.5], [0.75, 0.25]]])
    model = decision_chain(
        transitions, np.array([[0.75, 0.25], [0.75, 0.25]]), np.array([0.5, 0.5])
    )
    reduced, error = quotient(model, decision_partition(((0, 1),)))
    assert error == 0
    assert reachability(reduced, "positive", 1)[0] == 0.25
    assert decision_horizon(0) == 0
    with pytest.raises(ValueError):
        decision_chain(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2) / 2)
