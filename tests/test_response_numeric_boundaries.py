"""Exact empirical counts and held-out-condition response safeguards."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from behavioral_equivalence.heldout_response import teacher_response_prior, query_only_table
from behavioral_equivalence.neutral_response import exact_response_gap, response_gap_witness
from behavioral_equivalence.response_chain import response_chain_pair
from behavioral_equivalence.response_coarsening import coarsen_response_table
from behavioral_equivalence.response_replicas import response_replicas
from behavioral_equivalence.response_refinement import (
    split_response_witness,
    validate_latent_partition,
)


def original(fixture, function):
    r = json.loads(Path("data/synthetic/fixtures/release", fixture).read_text())
    assert hashlib.sha256(r["source_module"].encode()).hexdigest() == r["source_sha256"]
    scope = {"__package__": "behavioral_equivalence"}
    exec(compile(r["source_module"], "frozen-response-module", "exec"), scope)
    return scope[function]


def test_teacher_prior_accepts_exact_unsigned_environment_keys():
    states = np.array([0, 0, 1], dtype=np.int64)
    env = np.array([0, 0, 1], dtype=np.uint64)
    teacher = np.array([0.2, 0.4, 0.8])
    new = teacher_response_prior(states, env, teacher, 3)
    old = original("heldout-integer-environments-v1.json", "teacher_response_prior")
    with pytest.raises(TypeError):
        old(states, env, teacher, 3)
    expected = old(states, env.astype(np.int16), teacher, 3)
    for key in new:
        np.testing.assert_array_equal(new[key], expected[key])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_exact_gap_preserves_integer_success_difference_above_float_exactness(dtype):
    trials = 2**53 + 1
    n = np.full((2, 75), trials, dtype=dtype)
    y = np.zeros((2, 75), dtype=dtype)
    y[:, 0] = [trials - 1, trials]
    old = original("neutral-exact-counts-v1.json", "exact_response_gap")
    assert old(n, y, (0, 1), 0) == 0
    assert exact_response_gap(n, y, (0, 1), 0) == Fraction(1, trials)
    witness = response_gap_witness(n, y, (0, 1), 0, epsilon=0)
    assert witness["exceeds_budget"]
    value = witness["minimum_shared_row_radius"]
    assert Fraction(int(value["numerator_hex"], 16), int(value["denominator_hex"], 16)) == Fraction(
        1, 2 * trials
    )


def test_exact_gap_rejects_success_count_above_trial_integer_boundary():
    n = np.full((2, 75), 2**53 + 3, dtype=np.int64)
    y = np.zeros_like(n, dtype=np.float64)
    y[1, 0] = float(2**53 + 4)
    with pytest.raises(ValueError, match="binary-success counts"):
        exact_response_gap(n, y, (0, 1), 0)


def test_original_count_domain_and_teacher_prior_are_unchanged():
    old_gap = original("neutral-exact-counts-v1.json", "exact_response_gap")
    old_prior = original("heldout-integer-environments-v1.json", "teacher_response_prior")
    rng = np.random.default_rng(20261016)
    for _ in range(20):
        n = rng.integers(1, 10000, (4, 75), dtype=np.int64)
        y = np.floor(n * rng.uniform(0, 1, n.shape))
        assert exact_response_gap(n, y, (0, 1), 0) == old_gap(n, y, (0, 1), 0)
    state = rng.integers(0, 4, 200, dtype=np.int64)
    env = rng.integers(0, 75, 200, dtype=np.int16)
    p = rng.uniform(0, 1, 200)
    before, after = old_prior(state, env, p, 4), teacher_response_prior(state, env, p, 4)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("blocks,n", [((), 2), (((0,), ()), 2), (((0, 1), (1,)), 2)])
def test_latent_response_partition_requires_nonempty_disjoint_cover(blocks, n):
    with pytest.raises(ValueError, match="Complete disjoint"):
        validate_latent_partition(blocks, n)


@pytest.mark.parametrize("parent,abstract", [(0.5, [0.5]), ([0.5, 0.5], [0.5, 0.5])])
def test_response_chain_pair_requires_one_abstract_value_per_block(parent, abstract):
    with pytest.raises(ValueError, match="Aligned parent"):
        response_chain_pair(parent, abstract, ((0, 1),))


@pytest.mark.parametrize("value", [np.inf, -0.1, 1.1])
def test_response_split_rejects_invalid_observed_other_block_probabilities(value):
    p = np.full((3, 75), np.nan)
    p[:, 0] = [0.1, 0.9, value]
    with pytest.raises(ValueError, match="block probabilities"):
        split_response_witness(((0, 1, 2),), p, dict(states=(0, 1), environment=0))


@pytest.mark.parametrize("change", ["map_missing", "users_exceed", "fractional", "shape"])
def test_coarsening_requires_complete_map_and_independent_user_support(change):
    n = np.ones((2, 75), dtype=np.int64)
    y = np.zeros_like(n, dtype=float)
    mapping = np.array([0, 1])
    expected = n.copy()
    users = n.copy()
    if change == "map_missing":
        mapping[:] = 0
    elif change == "users_exceed":
        users[0, 0] = 2
    elif change == "fractional":
        y[0, 0] = 0.5
    else:
        y = y[:1]
    with pytest.raises(ValueError, match="Aligned complete"):
        coarsen_response_table(n, y, mapping, expected, users)


def test_coarsening_copies_user_support_and_does_not_change_input_tables():
    n = np.ones((2, 75), dtype=np.int64)
    y = np.zeros_like(n)
    users = n.copy()
    result = coarsen_response_table(n, y, np.array([0, 1]), n, users)
    users[0, 0] = 0
    assert result["users"][0, 0] == 1
    np.testing.assert_array_equal(n, np.ones((2, 75), dtype=np.int64))


@pytest.mark.parametrize("groups", [np.array([[0]]), np.array([0.0]), np.array([-1])])
def test_heldout_replicas_require_known_integer_groups(groups):
    forecasts = {name: np.array([0.5]) for name in ["q01", "q02", "q03", "q12", "q13", "q23"]}
    with pytest.raises(ValueError):
        response_replicas(groups, forecasts)


def test_query_table_hides_all_heldout_counts_and_probabilities_without_mutation():
    n = np.full((2, 75), 10, dtype=np.int64)
    y = np.full((2, 75), 4)
    u = np.full((2, 75), 5)
    table = dict(counts=n, positives=y, users=u)
    result = query_only_table(table, (0, 2))
    hidden = ~np.isin(np.arange(75) % 5, [0, 2])
    assert not result["counts"][:, hidden].any() and not result["positives"][:, hidden].any()
    assert (
        not result["users"][:, hidden].any() and np.isnan(result["probabilities"][:, hidden]).all()
    )
    np.testing.assert_array_equal(n, np.full((2, 75), 10))
