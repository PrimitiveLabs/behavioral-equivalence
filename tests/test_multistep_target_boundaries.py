"""Censoring, finite-event ordering and within-role source-window boundaries."""

from itertools import product
import numpy as np
import pytest
from behavioral_equivalence.horizon_targets import (
    horizon_event_targets,
    origin_iid_event_probabilities,
)
from behavioral_equivalence.multistep_cohort import (
    select_test_origins,
    observed_horizon_cohorts,
    recorded_horizon_targets,
)
from behavioral_equivalence.training_horizons import training_role_cohort
from behavioral_equivalence.nested_design import latent_partition, human_event_suite


@pytest.mark.parametrize(
    "change",
    [
        dict(users=[[1, 1]]),
        dict(timestamps=[1]),
        dict(roles=[3]),
        dict(users=[1.0, 1.0]),
        dict(timestamps=[1.0, 2.0]),
        dict(users=[-1, 1]),
        dict(roles=[3, 4]),
    ],
)
def test_origin_source_bundle_boundaries(change):
    args = dict(users=[1, 1], timestamps=[1, 2], roles=[3, 3])
    args.update(change)
    with pytest.raises(ValueError, match="Sorted aligned"):
        select_test_origins(**args)


@pytest.mark.parametrize("cap", [0, -1, 1.5])
def test_origin_sampling_cap_boundaries(cap):
    with pytest.raises(ValueError, match="per-user cap"):
        select_test_origins([1], [1], [3], per_user_cap=cap)


@pytest.mark.parametrize(
    "origins",
    [np.array([[0]]), np.array([0.0]), np.array([-1]), np.array([0, 0]), np.array([1, 0])],
)
def test_observed_cohort_origin_alignment_order_and_support(origins):
    with pytest.raises(ValueError, match="origin indices"):
        observed_horizon_cohorts([1, 1], [1, 2], [3, 3], origins, [1])


@pytest.mark.parametrize("horizons", [[-1], [1.5], [False]])
def test_observed_cohort_horizon_domain(horizons):
    with pytest.raises(ValueError, match="decision horizons"):
        observed_horizon_cohorts([1], [1], [3], np.array([0]), horizons)


def test_no_test_origins_remain_empty_without_inventing_support():
    origins = select_test_origins([1, 1], [1, 2], [0, 1])
    assert origins.size == 0
    cohorts = observed_horizon_cohorts([1, 1], [1, 2], [0, 1], origins, [1, 5])
    assert all(
        c["selected_origins"] == c["observed_origins"] == c["censored_origins"] == 0
        for c in cohorts
    )


def test_recorded_targets_require_complete_aligned_outcome_vector():
    cohort = observed_horizon_cohorts([1, 1], [1, 2], [3, 3], np.array([0]), [2])[0]
    with pytest.raises(ValueError, match="complete outcome"):
        recorded_horizon_targets(np.array([1]), cohort)
    altered = dict(cohort, end_indices=np.array([], dtype=int))
    with pytest.raises(ValueError, match="Aligned cohort"):
        recorded_horizon_targets(np.array([0, 1]), altered)
    with pytest.raises(ValueError, match="Binary representative"):
        recorded_horizon_targets(np.array([[0, 1]]), cohort)


@pytest.mark.parametrize(
    "mask,count,last",
    [
        (np.array([True]), np.array([[0]]), np.array([[0]])),
        (np.array([[1]]), np.array([[0]]), np.array([[0]])),
        (np.array([[True]]), np.array([[0.0]]), np.array([[0]])),
        (np.array([[True]]), np.array([[0]]), np.array([[0.0]])),
        (np.array([[False]]), np.array([[-1]]), np.array([[0]])),
        (np.array([[True]]), np.array([[0]]), np.array([[-1]])),
        (np.array([[True]]), np.array([[-1]]), np.array([[0]])),
        (np.array([[True]]), np.array([[3]]), np.array([[1]])),
        (np.array([[True]]), np.array([[2]]), np.array([[0]])),
    ],
)
def test_horizon_event_masks_counts_and_last_outcome_consistency(mask, count, last):
    with pytest.raises(ValueError):
        horizon_event_targets([2], mask, count, last)


def test_every_short_binary_trajectory_matches_explicit_event_semantics_and_censored_sentinels():
    horizons = [1, 2, 3, 4]
    for trajectory in product([0, 1], repeat=4):
        counts = np.array([[sum(trajectory[:h]) for h in horizons], [-1] * 4])
        last = np.array([[trajectory[h - 1] for h in horizons], [-1] * 4])
        labels, events = horizon_event_targets(
            horizons, np.array([[True] * 4, [False] * 4]), counts, last
        )
        expected = [trajectory[h - 1] for h in horizons]
        expected += [int(any(trajectory[:h])) for h in horizons]
        expected += [int(any(y == 0 for y in trajectory[:h])) for h in horizons]
        expected += [int(sum(trajectory[:h]) >= 2) for h in horizons]
        expected += [int(sum(trajectory[:h]) >= 4) for h in horizons]
        assert labels[0].tolist() == expected
        assert labels[1].tolist() == [-1] * 20
        assert events == human_event_suite(tuple(horizons))


@pytest.mark.parametrize("probabilities", [0.5, [[0.5]], [-0.1], [np.nan], [np.inf]])
def test_origin_iid_reference_requires_finite_probability_vector(probabilities):
    with pytest.raises(ValueError, match="origin response"):
        origin_iid_event_probabilities(probabilities, [1, 5])


def test_empty_iid_reference_has_no_prediction_cases():
    assert origin_iid_event_probabilities([], [1, 5]).shape == (0, 10)


@pytest.mark.parametrize(
    "features", [np.ones(2), np.empty((0, 2)), np.empty((2, 0)), np.array([[np.inf]])]
)
def test_parent_capacity_control_requires_supported_finite_features(features):
    with pytest.raises(ValueError, match="parent feature"):
        latent_partition(features, 1, 7)


@pytest.mark.parametrize(
    "role,roles",
    [
        (0.0, np.array([0])),
        (-1, np.array([0])),
        (0, np.array([[0]])),
        (0, np.array([0.0])),
        (0, np.array([5])),
    ],
)
def test_direct_horizon_fitting_requires_declared_integer_roles(role, roles):
    with pytest.raises(ValueError, match="training or validation"):
        training_role_cohort(np.array([1]), np.array([1]), roles, role, [1])
