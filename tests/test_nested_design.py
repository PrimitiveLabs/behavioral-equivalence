import numpy as np
import pytest
from behavioral_equivalence.nested_design import (
    latent_partition,
    human_event_suite,
    nested_run_identity,
)


def test_parent_state_controls_have_exact_capacity_and_deterministic_ties():
    x = np.zeros((8, 3))
    for random in (False, True):
        blocks = latent_partition(x, 4, 71, random)
        assert len(blocks) == 4 and all(blocks)
        assert sorted(s for b in blocks for s in b) == list(range(8))
        assert blocks == latent_partition(x, 4, 71, random)
    assert len(latent_partition(x, 8, 71)) == 8
    for count in (0, 9, True, 1.2):
        with pytest.raises(ValueError, match="count"):
            latent_partition(x, count, 71)
    with pytest.raises(ValueError, match="feature"):
        latent_partition(np.array([[np.nan]]), 1, 71)


def test_property_suite_keeps_decision_horizon_and_event_claims_explicit():
    events = human_event_suite((1, 5, 10, 25, 50))
    assert len(events) == 25 and len(set(events)) == 25
    assert [e.horizon for e in events[:5]] == [1, 9, 19, 49, 99]
    assert events[15].mode == "count_at_least" and events[15].threshold == 2
    assert events[20].threshold == 4
    for horizons in ((), (0,), (1, 1)):
        with pytest.raises(ValueError, match="horizons"):
            human_event_suite(horizons)
    with pytest.raises(ValueError, match="Decision"):
        human_event_suite((True,))


def test_cache_binding_changes_for_every_upstream_selector():
    baseline = nested_run_identity("source", "parent-a", "sha-a")
    assert baseline == nested_run_identity("source", "parent-a", "sha-a")
    assert baseline != nested_run_identity("source-2", "parent-a", "sha-a")
    assert baseline != nested_run_identity("source", "parent-b", "sha-a")
    assert baseline != nested_run_identity("source", "parent-a", "sha-b")
