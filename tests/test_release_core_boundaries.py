"""Release regressions for model validity and immutable probability inputs."""

from copy import deepcopy
import numpy as np
import pytest
from pydantic import ValidationError
from behavioral_equivalence.core import (
    Chain,
    check_partition,
    exact_partition,
    approximate_partition,
    reachability,
    rational_reachability,
    coupling_bound,
)
from behavioral_equivalence.contracts import ModelCertificate, RunManifest, ArtifactRef


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "transition_shape",
        "initial_shape",
        "transition_nan",
        "initial_inf",
        "negative_transition",
        "negative_initial",
        "initial_sum",
    ],
)
def test_chain_rejects_invalid_probability_objects(change):
    p = np.eye(2)
    mu = np.array([1.0, 0.0])
    labels = ("active", "complete")
    if change == "empty":
        p = np.empty((0, 0))
        mu = np.empty(0)
        labels = ()
    if change == "transition_shape":
        p = np.eye(3)
    if change == "initial_shape":
        mu = np.array([1.0])
    if change == "transition_nan":
        p[0, 0] = np.nan
    if change == "initial_inf":
        mu[0] = np.inf
    if change == "negative_transition":
        p[0] = [-0.1, 1.1]
    if change == "negative_initial":
        mu[:] = [-0.1, 1.1]
    if change == "initial_sum":
        mu[:] = [0.2, 0.2]
    with pytest.raises(ValueError):
        Chain(p, labels, mu)


def test_chain_copies_probability_inputs_and_rejects_in_place_mutation():
    p = np.eye(2)
    mu = np.array([1.0, 0.0])
    chain = Chain(p, ("a", "b"), mu)
    digest = chain.digest()
    p[0] = [0, 1]
    mu[:] = [0, 1]
    assert chain.digest() == digest and np.array_equal(chain.initial, [1, 0])
    with pytest.raises(ValueError):
        chain.transitions[0, 0] = 0
    with pytest.raises(ValueError):
        chain.initial[0] = 0


@pytest.mark.parametrize(
    "blocks", [(), ((0,), (1,), ()), ((0, 0), (1,)), ((0,),), ((0, 2), (1,)), ((0, 1),)]
)
def test_invalid_partitions_rejected_before_quotient_use(blocks):
    chain = Chain(np.eye(2), ("a", "b"), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        check_partition(chain, blocks)


def test_empty_and_mismatched_environment_families_rejected():
    a = Chain(np.eye(2), ("a", "b"), np.array([1.0, 0.0]))
    b = Chain(np.eye(2), ("b", "a"), np.array([1.0, 0.0]))
    for family in [[], [a, b]]:
        with pytest.raises(ValueError):
            exact_partition(family)
        with pytest.raises(ValueError):
            approximate_partition(family, 0.1)


@pytest.mark.parametrize("h", [-1, True, 1.5])
def test_public_horizon_guards_reject_invalid_counts(h):
    chain = Chain(np.eye(1), ("a",), np.ones(1))
    with pytest.raises(ValueError):
        reachability(chain, "a", h)
    with pytest.raises(ValueError):
        coupling_bound(0.1, h)
    if h == -1:
        with pytest.raises(ValueError):
            rational_reachability(chain, "a", h)


@pytest.mark.parametrize("epsilon,eta", [(np.nan, 0), (0.1, np.inf), (1.1, 0), (0.1, -0.1)])
def test_invalid_coupling_probabilities_rejected(epsilon, eta):
    with pytest.raises(ValueError):
        coupling_bound(epsilon, 2, eta)


def certificate():
    return dict(
        concrete_digest="a" * 64,
        abstract_digest="b" * 64,
        partition=((0,), (1,)),
        target_label="goal",
        horizon=5,
        epsilon_upper=0.1,
        initial_mismatch_upper=0,
        uncertainty_semantics="none",
        nominal_lower=0.2,
        nominal_upper=0.8,
        numeric_status="outward_rounded",
        numeric_allowance=1e-12,
        proof_revision="fixture",
    )


def test_generic_certificate_valid_instance_is_narrowly_scoped():
    c = ModelCertificate(**certificate())
    assert c.scope == "specified_model_only" and c.partition == ((0,), (1,))


@pytest.mark.parametrize(
    "change",
    [
        {"nominal_lower": 0.9},
        {"partition": ()},
        {"partition": ((0,), ())},
        {"partition": ((0, 0), (1,))},
        {"partition": ((0,), (2,))},
        {"epsilon_upper": np.nan},
        {"initial_mismatch_upper": np.inf},
        {"numeric_allowance": -0.1},
        {"scope": "human_guarantee"},
    ],
)
def test_generic_certificate_rejects_invalid_intervals_partitions_and_claims(change):
    value = deepcopy(certificate())
    value.update(change)
    with pytest.raises(ValidationError):
        ModelCertificate(**value)


@pytest.mark.parametrize("state", ["submitted", "running", "complete", "failed", "blocked"])
def test_manifest_terminal_evidence_and_live_handle_contracts(state):
    base = dict(
        run_id="r",
        task_id="E2",
        phase="confirmatory",
        source_digest="a" * 64,
        config_digest="b" * 64,
        scientific_outcome="unsupported",
    )
    with pytest.raises(ValidationError):
        RunManifest(**base, state=state)
    if state in ["submitted", "running"]:
        extra = dict(cloud_job_id="fc-fixture")
    elif state == "complete":
        extra = dict(
            outputs=(
                ArtifactRef(
                    sha256="c" * 64, uri="report.json", media_type="application/json", bytes=2
                ),
            )
        )
    else:
        extra = dict(failure_reason="retained observation or execution evidence")
    assert RunManifest(**base, state=state, **extra).scientific_outcome == "unsupported"
