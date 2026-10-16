"""Direct input-boundary checks for split and user-weighted evaluation APIs."""

from dataclasses import replace
import numpy as np
import pytest
from behavioral_equivalence.calendar_splits import EvaluationCalendar, calendar_roles
from behavioral_equivalence.data import (
    Interaction,
    PredictionCase,
    TimestampBundle,
    available_bundle_history,
    audit_cases,
    chronological_cases,
    disjoint_user_cases,
)
from behavioral_equivalence.statistics import prediction_scores, paired_user_nll
from behavioral_equivalence.streaming_scores import metric_block, shared_user_bootstrap


@pytest.mark.parametrize(
    "change", [{"click": 2}, {"case_id": ""}, {"user_id": ""}, {"context_available_ms": 11}]
)
def test_interaction_identity_outcome_and_causal_context(change):
    with pytest.raises(ValueError):
        replace(Interaction("c", "u", 10, 1, 10), **change)


def test_history_order_unknown_split_and_duplicate_target_audit():
    a = Interaction("a", "u", 1, 0, 1)
    b = Interaction("b", "u", 2, 1, 2)
    c = Interaction("c", "u", 3, 0, 3)
    with pytest.raises(ValueError, match="strictly chronological"):
        PredictionCase(c, (b, a), "test")
    with pytest.raises(ValueError, match="Unknown split"):
        PredictionCase(c, (a, b), "other")
    case = PredictionCase(c, (a, b), "test")
    with pytest.raises(ValueError, match="multiple cases"):
        audit_cases([case, case], "users")
    with pytest.raises(ValueError, match="Unknown split protocol"):
        audit_cases([case], "other")
    with pytest.raises(ValueError, match="Every split"):
        audit_cases([case], "chronological")


@pytest.mark.parametrize(
    "method,kwargs",
    [
        (chronological_cases, {"max_history": 0}),
        (chronological_cases, {"horizon": 0}),
        (disjoint_user_cases, {"max_history": 0}),
    ],
)
def test_split_helpers_reject_unsupported_history_or_horizon(method, kwargs):
    with pytest.raises(ValueError, match="Positive"):
        method([], **kwargs)


def test_short_sequences_and_unsupported_user_split():
    events = [Interaction(str(t), "u", t, 0, t) for t in range(19)]
    assert chronological_cases(events) == []
    with pytest.raises(ValueError, match="ten users"):
        disjoint_user_cases(events)


@pytest.mark.parametrize(
    "user,timestamp,available,outcomes",
    [("", 1, 1, (0,)), ("u", 2, 1, (0,)), ("u", 1, 1, ()), ("u", 1, 1, (2,))],
)
def test_timestamp_bundle_boundaries(user, timestamp, available, outcomes):
    with pytest.raises(ValueError):
        TimestampBundle(user, timestamp, available, outcomes)


def test_bundle_history_availability_equality_and_history_cap():
    bundles = [TimestampBundle("u", t, t + 1, (t % 2,)) for t in range(1, 6)]
    assert available_bundle_history(bundles, "u", 6, 2) == tuple(bundles[2:4])
    with pytest.raises(ValueError, match="Positive"):
        available_bundle_history(bundles, "u", 6, 0)


@pytest.mark.parametrize("cutoffs", [(True, 2, 3), (1, 2.5, 3), (1, 1, 3), (3, 2, 1)])
def test_calendar_constructor_rejects_invalid_cutoffs(cutoffs):
    with pytest.raises(ValueError, match="ordered"):
        EvaluationCalendar(*cutoffs)


@pytest.mark.parametrize(
    "args",
    [
        (1, "bad", "train", 1, 0),
        (1, "known_users", "bad", 1, 0),
        (1.5, "known_users", "train", 2, 0),
        (1, "known_users", "train", 1.5, 0),
        (1, "known_users", "train", 1, -1),
        (1, "known_users", "train", 1, 0.5),
    ],
)
def test_scalar_calendar_rejects_invalid_protocol_and_time_arguments(args):
    with pytest.raises(ValueError):
        EvaluationCalendar(100, 200, 300).role(*args)


@pytest.mark.parametrize(
    "timestamps,availability,users",
    [
        (np.array([[1]]), np.array([[1]]), np.array([["train"]])),
        (np.array([1]), np.array([1, 2]), np.array(["train"])),
        (np.array([1]), np.array([1]), np.array(["bad"])),
        (np.array([1]), np.array([1.0]), np.array(["train"])),
    ],
)
def test_batch_calendar_rejects_alignment_types_and_roles(timestamps, availability, users):
    with pytest.raises(ValueError, match="Aligned"):
        calendar_roles(
            EvaluationCalendar(100, 200, 300), timestamps, availability, users, "known_users", 0
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(targets=[]),
        dict(targets=[[0, 1]]),
        dict(targets=[0, 2]),
        dict(probabilities=[0.5]),
        dict(users=[1]),
        dict(probabilities=[-0.1, 0.5]),
        dict(probabilities=[np.inf, 0.5]),
    ],
)
def test_prediction_score_boundaries(change):
    args = dict(targets=[0, 1], probabilities=[0.2, 0.8], users=[1, 2])
    args.update(change)
    with pytest.raises(ValueError, match="Aligned"):
        prediction_scores(**{k: np.array(v) for k, v in args.items()})


@pytest.mark.parametrize(
    "change",
    [
        dict(targets=[]),
        dict(targets=[[0, 1]]),
        dict(candidate=[0.5]),
        dict(users=[1]),
        dict(targets=[0, 2]),
        dict(reference=[np.nan, 0.5]),
        dict(candidate=[np.inf, 0.5]),
        dict(reference=[-0.1, 0.5]),
        dict(candidate=[0.5, 1.1]),
        dict(replications=99),
        dict(margin=0),
        dict(margin=np.inf),
        dict(users=[1, 1]),
    ],
)
def test_paired_user_comparison_boundaries(change):
    args = dict(
        targets=np.array([0, 1]),
        reference=np.array([0.2, 0.8]),
        candidate=np.array([0.3, 0.7]),
        users=np.array([1, 2]),
        replications=100,
    )
    args.update(change)
    with pytest.raises(ValueError):
        paired_user_nll(**args)


@pytest.mark.parametrize(
    "change",
    [
        dict(case_ids=[[1, 2]]),
        dict(users=[1]),
        dict(targets=[0]),
        dict(probabilities=[[[]], [[]]]),
        dict(probabilities=[0.5]),
        dict(case_ids=[1.0, 2.0]),
        dict(case_ids=[-1, 2]),
        dict(users=[1.0, 2.0]),
        dict(users=[-1, 2]),
        dict(targets=[0, 2]),
        dict(probabilities=[np.nan, 0.5]),
        dict(probabilities=[-0.1, 0.5]),
    ],
)
def test_streaming_metric_boundary_rejections(change):
    args = dict(case_ids=[1, 2], users=[1, 2], targets=[0, 1], probabilities=[0.2, 0.8])
    args.update(change)
    with pytest.raises(ValueError):
        metric_block(**args)


@pytest.mark.parametrize(
    "change",
    [
        dict(user_ids=[[1, 2]]),
        dict(case_counts=[1]),
        dict(user_ids=[1]),
        dict(user_ids=[1.0, 2.0]),
        dict(user_ids=[-1, 2]),
        dict(case_counts=[1.0, 1.0]),
        dict(case_counts=[0, 1]),
        dict(model_losses={}),
        dict(comparisons={}),
        dict(comparisons={"test": ("missing", "a")}),
        dict(replications=99),
        dict(replications=100.5),
        dict(batch_size=0),
        dict(batch_size=1.5),
        dict(margin=0),
        dict(margin=np.nan),
        dict(model_losses={"a": [np.nan, 0.2]}),
        dict(model_losses={"a": [[0.1, 0.2]]}),
    ],
)
def test_shared_bootstrap_boundary_rejections(change):
    args = dict(
        user_ids=[1, 2],
        case_counts=[1, 1],
        model_losses={"a": [0.1, 0.2]},
        comparisons={"identity": ("a", "a")},
        replications=100,
    )
    args.update(change)
    with pytest.raises(ValueError):
        shared_user_bootstrap(**args)
