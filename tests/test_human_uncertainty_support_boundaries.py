"""Human uncertainty assumptions, unsupported cells and exact integer keys."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from behavioral_equivalence.cohort_uncertainty import cohort_interval_diagnostic
from behavioral_equivalence.cluster_ratio import (
    cluster_ratio_from_totals,
    pooled_decision_intervals,
)
from behavioral_equivalence.resolution_support import cell_support, supported_pair_counts
from behavioral_equivalence.precision_planning import (
    normal_precision_plan,
    forecast_identity,
    improvement_precision_plan,
)


@pytest.mark.parametrize(
    "change",
    [
        dict(targets=[[0, 1]]),
        dict(lower=[0]),
        dict(targets=[0, 2]),
        dict(lower=[np.nan, 0]),
        dict(users=[1.0, 2.0]),
        dict(users=[-1, 2]),
        dict(replications=99),
        dict(minimum_users=1),
        dict(minimum_cases=1),
    ],
)
def test_cohort_interval_diagnostic_boundaries(change):
    args = dict(
        targets=[0, 1],
        lower=[0, 0],
        upper=[1, 1],
        users=[1, 2],
        replications=100,
        minimum_users=2,
        minimum_cases=2,
    )
    args.update(change)
    with pytest.raises(ValueError, match="Aligned binary"):
        cohort_interval_diagnostic(**args)


@pytest.mark.parametrize(
    "totals,users,cap",
    [
        ([], 1, 1),
        ([[[-1]]], 1, 1),
        ([[-1]], 1, 1),
        ([[1]], True, 1),
        ([[1]], 1, True),
        ([[1]], 1, -1),
    ],
)
def test_aggregate_ratio_requires_integer_tables_and_declared_user_capacity(totals, users, cap):
    with pytest.raises(ValueError, match="aggregate counts"):
        cluster_ratio_from_totals(totals, users=users, maximum_per_user=cap)


@pytest.mark.parametrize(
    "alpha,bits", [(Fraction(0), 32), (Fraction(-1), 32), (Fraction(1, 20), 49)]
)
def test_ratio_confidence_parameter_boundaries(alpha, bits):
    with pytest.raises(ValueError, match="Rational alpha"):
        cluster_ratio_from_totals([[1]], users=1, maximum_per_user=1, alpha=alpha, bits=bits)


@pytest.mark.parametrize("alpha", [0.05, Fraction(0), Fraction(1)])
def test_pooled_intervals_require_rational_family_error(alpha):
    with pytest.raises(ValueError, match="Rational alpha"):
        pooled_decision_intervals(
            np.ones((1, 2, 1), dtype=int), users=2, maximum_per_user=1, alpha=alpha
        )


def test_ratio_sufficient_statistics_do_not_overflow_machine_integer_sum():
    maximum = int(np.iinfo(np.int64).max)
    lo, hi, evidence = cluster_ratio_from_totals(
        np.array([[maximum, maximum]], dtype=np.int64), users=2, maximum_per_user=maximum
    )
    assert np.all(lo <= 0.5) and np.all(hi >= 0.5)
    assert evidence["human_multistep_coverage_established"] is False
    assert evidence["tail_inequality_verified"]


@pytest.mark.parametrize("field", ["environment", "user"])
def test_accepted_unsigned_64bit_ids_never_promote_exact_cell_keys_to_float(field):
    state = np.array([0, 0, 0], dtype=np.int64)
    environment = np.array([0, 0, 0], dtype=np.int64)
    users = np.array([1, 2, 1], dtype=np.int64)
    if field == "environment":
        environment = environment.astype(np.uint64)
    else:
        users = np.array([2**64 - 1, 2**64 - 2, 2**64 - 1], dtype=np.uint64)
    table = cell_support(state, environment, users, 4)
    assert table["counts"][0, 0] == 3 and table["users"][0, 0] == 2
    assert table["counts"].sum() == 3 and table["users"].sum() == 2


def test_signed_pipeline_compatibility_and_frozen_unsigned_failure_are_retained():
    path = Path("data/synthetic/fixtures/release/support-integer-identities-v1.json")
    fixture = json.loads(path.read_text())
    assert hashlib.sha256(fixture["source_module"].encode()).hexdigest() == fixture["source_sha256"]
    scope = {"__package__": "behavioral_equivalence"}
    exec(compile(fixture["source_module"], "frozen-resolution-support", "exec"), scope)
    old = scope["cell_support"]
    rng = np.random.default_rng(20261016)
    for size in [0, 1, 100, 1000]:
        states = rng.integers(0, 8, size, dtype=np.int64)
        environments = rng.integers(0, 75, size, dtype=np.int16)
        users = rng.integers(0, 30, size, dtype=np.int64)
        before, after = (
            old(states, environments, users, 8),
            cell_support(states, environments, users, 8),
        )
        for key in ["counts", "users"]:
            np.testing.assert_array_equal(before[key], after[key])
        for state in range(8):
            for environment in range(75):
                chosen = (states == state) & (environments == environment)
                assert after["users"][state, environment] == len(set(map(int, users[chosen])))
    with pytest.raises(TypeError):
        old(np.array([0]), np.array([0], dtype=np.uint64), np.array([1]), 4)
    with pytest.raises(TypeError):
        old(np.array([0]), np.array([0]), np.array([1], dtype=np.uint64), 4)


@pytest.mark.parametrize(
    "counts,users,minimum",
    [
        (np.zeros((4, 74), dtype=int), np.zeros((4, 74), dtype=int), 1),
        (np.zeros((3, 75), dtype=int), np.zeros((3, 75), dtype=int), 1),
        (np.zeros((4, 75), dtype=int), np.ones((4, 75), dtype=int), 1),
        (np.zeros((4, 75), dtype=int), np.zeros((4, 75), dtype=int), 0),
    ],
)
def test_pair_support_requires_groups_counts_and_thresholds(counts, users, minimum):
    with pytest.raises(ValueError, match="support tables"):
        supported_pair_counts(counts, users, min_cases=minimum)


@pytest.mark.parametrize(
    "alpha,power", [(0, 0.8), (0.5, 0.8), (np.nan, 0.8), (0.025, 0.5), (0.025, 1)]
)
def test_normal_planning_probability_boundaries(alpha, power):
    with pytest.raises(ValueError, match="planning probabilities"):
        normal_precision_plan([0.0, 0.1], alpha=alpha, power=power)


@pytest.mark.parametrize(
    "candidate,reference,environments",
    [
        ([[0.5]], [[0.5, 0.5]], [0]),
        ([], [], [0]),
        ([[np.nan]], [[0.5]], [0]),
        ([[0.5]], [[0.5]], []),
        ([[0.5]], [[0.5]], [0.0]),
        ([[0.5]], [[0.5]], [-1]),
    ],
)
def test_structural_identity_requires_declared_valid_contexts(candidate, reference, environments):
    with pytest.raises(ValueError, match="Aligned probability"):
        forecast_identity(candidate, reference, environments)


def test_structural_identity_flag_cannot_be_an_unvalidated_truthy_value():
    with pytest.raises(ValueError, match="Structural identity"):
        improvement_precision_plan([0.0, 0.0], [0.01], structurally_identical=1)
