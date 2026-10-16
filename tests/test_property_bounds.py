from copy import deepcopy
from fractions import Fraction
import itertools
import numpy as np
import pytest
from behavioral_equivalence.core import Chain, quotient
from behavioral_equivalence.decision_model import decision_chain, decision_partition
from behavioral_equivalence.property_bounds import (
    LabelEvent,
    event_enclosures,
    event_predictions,
    certify_properties,
    verify_properties,
)
from behavioral_equivalence.rational import normalized_kernel


def exact_event(chain, start, event):
    p = normalized_kernel(chain)
    total = Fraction()
    for successors in itertools.product(range(chain.size), repeat=event.horizon):
        path = (start,) + successors
        probability = Fraction(1)
        for s, t in zip(path, path[1:]):
            probability *= p[s][t]
        visits = sum(chain.labels[s] in event.targets for s in path)
        holds = (
            chain.labels[path[-1]] in event.targets
            if event.mode == "occupancy"
            else visits >= event.threshold
        )
        if holds:
            total += probability
    return total


def test_event_enclosures_match_independent_path_enumeration():
    c = Chain(
        np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.5, 0.25, 0.25]]),
        ("active", "positive", "negative"),
        np.array([1.0, 0, 0]),
    )
    events = tuple(
        LabelEvent(mode, ("positive",), h, threshold)
        for mode, threshold in [
            ("occupancy", 1),
            ("reachability", 1),
            ("count_at_least", 2),
            ("count_at_least", 4),
        ]
        for h in [0, 1, 3, 5]
    )
    enclosed = event_enclosures(c, events, bits=32)
    floating = event_predictions(c, events)
    for i, event in enumerate(events):
        for s in range(c.size):
            truth = exact_event(c, s, event)
            assert (
                Fraction(enclosed[i]["lower"][s], 1 << 32)
                <= truth
                <= Fraction(enclosed[i]["upper"][s], 1 << 32)
            )
            assert floating[i, s] == float(truth)
    with pytest.raises(ValueError, match="threshold"):
        LabelEvent("occupancy", ("positive",), 1, 2)


def test_certificate_covers_actual_property_error_and_rejects_tampering():
    c = Chain(
        np.array(
            [
                [0.5, 0.25, 0.125, 0.125],
                [0.25, 0.5, 0.0625, 0.1875],
                [0.5, 0.25, 0.125, 0.125],
                [0.25, 0.5, 0.125, 0.125],
            ]
        ),
        ("active", "active", "positive", "negative"),
        np.array([1.0, 0, 0, 0]),
    )
    blocks = ((0, 1), (2,), (3,))
    q, _ = quotient(c, blocks)
    events = tuple(
        LabelEvent(mode, ("positive",), h, r)
        for mode, r in [("occupancy", 1), ("reachability", 1), ("count_at_least", 2)]
        for h in [1, 3]
    )
    cert = certify_properties(c, q, blocks, events, bits=32)
    verify_properties(c, q, cert)
    for record, event in zip(cert["properties"], events):
        value = record["error_bound"]
        bound = Fraction(int(value["numerator_hex"], 16), int(value["denominator_hex"], 16))
        for b, block in enumerate(blocks):
            for s in block:
                assert abs(exact_event(c, s, event) - exact_event(q, b, event)) <= bound
    damaged = deepcopy(cert)
    damaged["properties"][0]["error_bound"]["numerator_hex"] = "0"
    with pytest.raises(ValueError, match="recomputation"):
        verify_properties(c, q, damaged)


def test_first_step_certificate_does_not_imply_unlisted_future_equivalence():
    transitions = np.zeros((3, 2, 3))
    transitions[0, :, 0] = 1
    transitions[1, :, 2] = 1
    transitions[2, :, 2] = 1
    c = decision_chain(
        transitions, np.array([[0.75, 0.25], [0.75, 0.25], [0.25, 0.75]]), np.array([0.5, 0.5, 0])
    )
    blocks = decision_partition(((0, 1), (2,)))
    q, _ = quotient(c, blocks)
    events = (LabelEvent("occupancy", ("positive",), 1), LabelEvent("occupancy", ("positive",), 3))
    cert = certify_properties(c, q, blocks, events)
    assert cert["properties"][0]["direct_max_error_units"] == 0
    assert Fraction(cert["properties"][1]["direct_max_error_units"], 1 << 48) >= Fraction(1, 4)
