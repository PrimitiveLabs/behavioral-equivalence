"""Explicit domain and scope checks for abstraction/property certificates."""

from fractions import Fraction
import json
import numpy as np
import pytest
from behavioral_equivalence.core import Chain, check_partition
from behavioral_equivalence.clustering import (
    feature_partition,
    random_partition,
    predictive_features,
)
from behavioral_equivalence.property_bounds import (
    LabelEvent,
    event_enclosures,
    event_predictions,
    certify_properties,
    verify_properties,
)
from behavioral_equivalence.uncertainty_abstraction import _outward


@pytest.fixture
def chain():
    return Chain(np.array([[0.5, 0.5], [0, 1.0]]), ("active", "goal"), np.array([1.0, 0]))


@pytest.mark.parametrize("k", [0, 1, 3])
@pytest.mark.parametrize("method", [random_partition, feature_partition])
def test_clustering_state_count_covers_labels_without_exceeding_source(chain, k, method):
    with pytest.raises(ValueError, match="State count"):
        if method is feature_partition:
            method(chain, np.ones((2, 1)), k)
        else:
            method(chain, k)


@pytest.mark.parametrize(
    "features",
    [
        np.ones(2),
        np.empty((2, 0)),
        np.ones((1, 2)),
        np.array([[np.nan], [1]]),
        np.array([[1], [np.inf]]),
    ],
)
def test_feature_clustering_requires_finite_nonempty_aligned_vectors(chain, features):
    with pytest.raises(ValueError, match="finite nonempty feature"):
        feature_partition(chain, features, 2)


def test_clustering_iterations_predictive_catalog_and_determinism(chain):
    with pytest.raises(ValueError, match="Positive iteration"):
        feature_partition(chain, np.ones((2, 1)), 2, iterations=0)
    with pytest.raises(ValueError, match="predictive horizon"):
        predictive_features(chain, ())
    features = predictive_features(chain, (0, 1))
    assert features.shape == (2, 4)
    a = feature_partition(chain, features, 2, seed=17)
    assert a == feature_partition(chain, features, 2, seed=17)
    check_partition(chain, a)


@pytest.mark.parametrize(
    "change",
    [
        dict(mode="unsupported"),
        dict(targets=()),
        dict(targets=("goal", "goal")),
        dict(horizon=-1),
        dict(horizon=True),
        dict(horizon=1.5),
        dict(threshold=0),
        dict(threshold=True),
        dict(threshold=1.5),
        dict(mode="occupancy", threshold=2),
    ],
)
def test_label_event_constructor_rejects_invalid_domain(change):
    args = dict(mode="reachability", targets=("goal",), horizon=1, threshold=1)
    args.update(change)
    with pytest.raises(ValueError):
        LabelEvent(**args)


@pytest.mark.parametrize("function", [event_enclosures, event_predictions])
@pytest.mark.parametrize("duplicate", [False, True])
def test_property_catalog_requires_distinct_nonempty_events(chain, function, duplicate):
    event = LabelEvent("reachability", ("goal",), 1)
    with pytest.raises(ValueError, match="Distinct nonempty event"):
        function(chain, (event, event) if duplicate else ())


@pytest.mark.parametrize("bits", [7, 63, True, 8.5])
def test_event_enclosures_reject_invalid_precision(chain, bits):
    with pytest.raises(ValueError, match="precision"):
        event_enclosures(chain, (LabelEvent("occupancy", ("goal",), 1),), bits)


@pytest.mark.parametrize("origins", [(), (0, 0), (-1,), (2,), (True,), (0.5,)])
def test_property_certification_requires_distinct_supported_integer_origins(chain, origins):
    with pytest.raises(ValueError, match="prediction origins"):
        certify_properties(
            chain, chain, ((0,), (1,)), (LabelEvent("reachability", ("goal",), 1),), origins=origins
        )


def test_abstract_labels_must_match_relation(chain):
    mismatch = Chain(chain.transitions, ("goal", "active"), chain.initial)
    with pytest.raises(ValueError, match="Abstract labels"):
        certify_properties(
            chain, mismatch, ((0,), (1,)), (LabelEvent("reachability", ("goal",), 1),)
        )


@pytest.mark.parametrize("bits", [8, 62])
def test_json_roundtrip_and_subset_origin_scope_are_explicit(chain, bits):
    events = (LabelEvent("occupancy", ("goal",), 0), LabelEvent("count_at_least", ("goal",), 1, 3))
    cert = certify_properties(chain, chain, ((0,), (1,)), events, bits=bits, origins=(1,))
    assert cert["concrete_origins"] == [1]
    assert all(record["witness_state"] == 1 for record in cert["properties"])
    assert all(record["error_bound"]["numerator_hex"] == "0" for record in cert["properties"])
    verify_properties(chain, chain, json.loads(json.dumps(cert)))


@pytest.mark.parametrize(
    "value", [Fraction(1, 10), Fraction(1, 3), Fraction(1, 2), Fraction(0), Fraction(1)]
)
def test_outward_rational_projection_rounding_in_both_directions(value):
    lower, upper = _outward(value, False), _outward(value, True)
    assert Fraction.from_float(lower) <= value <= Fraction.from_float(upper)
    assert lower <= upper
    if Fraction.from_float(float(value)) == value:
        assert lower == upper == float(value)
    else:
        assert np.nextafter(lower, np.inf) == upper


@pytest.mark.parametrize("horizons", [[], [0], [-1], [1.5]])
def test_benchmark_requires_positive_integer_horizons(chain, horizons):
    from behavioral_equivalence.benchmark import probability_tables

    with pytest.raises(ValueError, match="positive integers"):
        probability_tables(chain, horizons)


def test_benchmark_rejects_undefined_conditional_origins():
    from behavioral_equivalence.benchmark import probability_tables, partition_properties

    chain = Chain(np.eye(2), ("complete", "disengage"), np.array([1.0, 0]))
    truth = probability_tables(chain, [1])
    with pytest.raises(ValueError, match="No supported conditional"):
        partition_properties(chain, ((0,), (1,)), [1], truth, 1e-10)


def test_benchmark_external_callback_receives_original_and_every_real_variant():
    from behavioral_equivalence.benchmark import evaluate_world

    calls = []

    def checker(chain, index, name):
        calls.append((chain.size, index, name))
        return {"status": "fixture_boundary_observed"}

    protocol = dict(
        horizons=[1],
        epsilon_grid=[0.03],
        state_divisors=[4],
        predictive_feature_horizons=[1],
        numerical_tolerance=1e-10,
    )
    result = evaluate_world(
        dict(family="memory", size=16, seed=90, perturbation=0.01), protocol, verify=checker
    )
    assert calls[0] == (16, -1, "original")
    assert len(calls) == 7 and [c[1] for c in calls[1:]] == list(range(6))
    for call, method in zip(calls[1:], result["methods"], strict=True):
        assert call[0] == method["states"] and call[2] == method["method"]
        assert method["independent_verification"]["status"] == "fixture_boundary_observed"
    assert result["original_verification"]["status"] == "fixture_boundary_observed"
    assert result["bound_violations"] == 0


def test_benchmark_preserves_external_verifier_failure():
    from behavioral_equivalence.benchmark import evaluate_world

    def unavailable(*args):
        raise RuntimeError("fixture verifier unavailable")

    with pytest.raises(RuntimeError, match="verifier unavailable"):
        evaluate_world(
            dict(family="memory", size=16, seed=90, perturbation=0),
            dict(horizons=[1], numerical_tolerance=1e-10),
            verify=unavailable,
        )
