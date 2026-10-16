from fractions import Fraction
import numpy as np
import pytest
from behavioral_equivalence.neutral_response import (
    neutral_training_probabilities,
    certified_neutral_partition,
    exact_response_gap,
    response_gap_witness,
)


def test_neutral_mean_uses_training_context_mass_and_empty_state_fallback():
    p = np.full((3, 75), 0.8)
    p[0, 0] = 0.2
    n = np.zeros((3, 75), dtype=np.int64)
    n[0, :2] = [3, 1]
    n[1, 0] = 4
    means = neutral_training_probabilities(p, n)
    assert np.allclose(means, [0.35, 0.8, 0.575])


def test_neutral_partition_has_replayed_exact_budget_certificate():
    blocks, cert = certified_neutral_partition([0.1, 0.105, 0.7, 0.705], 0.01)
    assert len(blocks) == 2 and sorted(s for b in blocks for s in b) == [0, 1, 2, 3]
    eps = cert["epsilon"]
    assert Fraction(
        int(eps["numerator_hex"], 16), int(eps["denominator_hex"], 16)
    ) <= Fraction.from_float(0.01)
    assert len(certified_neutral_partition([0.1, 0.2, 0.3], 0)[0]) == 3
    assert len(certified_neutral_partition([0.5, 0.5, 0.5], 0.01)[0]) == 1


def test_count_gap_is_an_exact_shared_row_lower_bound():
    n = np.full((2, 75), 10000, dtype=np.int64)
    y = np.zeros_like(n)
    y[:, 0] = [4000, 4300]
    assert exact_response_gap(n, y, (0, 1), 0) == Fraction(3, 100)
    witness = response_gap_witness(n, y, (0, 1), 0, 0.01)
    assert witness["exceeds_budget"]
    assert Fraction(
        int(witness["minimum_shared_row_radius"]["numerator_hex"], 16),
        int(witness["minimum_shared_row_radius"]["denominator_hex"], 16),
    ) == Fraction(3, 200)
    y[1, 0] = 4200
    assert not response_gap_witness(n, y, (0, 1), 0, 0.01)["exceeds_budget"]


def test_invalid_probability_counts_or_budgets_rejected():
    with pytest.raises(ValueError):
        certified_neutral_partition([0.5], -0.1)
    with pytest.raises(ValueError):
        neutral_training_probabilities(np.full((2, 75), 0.5), np.zeros((2, 75), dtype=np.int64))
    n = np.ones((2, 75), dtype=np.int64)
    y = np.zeros_like(n)
    with pytest.raises(ValueError):
        exact_response_gap(n, y, (0, 0), 0)
    with pytest.raises(ValueError):
        exact_response_gap(n, y, (0, 1), 75)


def test_neutral_repair_and_random_capacity_both_preserve_exact_budget():
    from behavioral_equivalence.neutral_response import (
        repair_neutral_partition,
        random_neutral_capacity,
    )

    p = [0.0, 0.001, 0.019, 0.02]
    # Removing one middle state can shift a subset mean outside the old budget.
    repaired, certificate = repair_neutral_partition(p, ((0, 1, 3), (2,)), 0.01)
    assert len(repaired) >= 3
    value = certificate["epsilon"]
    assert Fraction(
        int(value["numerator_hex"], 16), int(value["denominator_hex"], 16)
    ) <= Fraction.from_float(0.01)
    initial, _ = certified_neutral_partition(p, 0.01)
    final, certificate = random_neutral_capacity(p, initial, 4, 17, 0.01)
    assert len(final) == 4
    assert final == random_neutral_capacity(p, initial, 4, 17, 0.01)[0]
    with pytest.raises(ValueError):
        random_neutral_capacity(p, initial, 5, 17, 0.01)
