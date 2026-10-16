import numpy as np
import pytest
from behavioral_equivalence.response_refinement import (
    split_response_witness,
    random_capacity_refinement,
    group_response_model,
)


def test_targeted_split_separates_witness_and_retains_unknown_states():
    p = np.full((5, 75), np.nan)
    p[:3, 0] = [0.1, 0.9, 0.7]
    blocks = ((0, 1, 2, 3), (4,))
    result = split_response_witness(blocks, p, dict(states=(0, 1), environment=0))
    assert result == ((0, 3), (1, 2), (4,))
    with pytest.raises(ValueError):
        split_response_witness(result, p, dict(states=(0, 1), environment=0))
    with pytest.raises(ValueError):
        split_response_witness(blocks, p, dict(states=(0, 1), environment=4))
    with pytest.raises(ValueError):
        split_response_witness(blocks, p, dict(states=(0, 0), environment=0))
    p[1, 0] = 0.1
    with pytest.raises(ValueError):
        split_response_witness(blocks, p, dict(states=(0, 1), environment=0))


def test_random_control_matches_capacity_without_crossing_initial_blocks():
    blocks = ((0, 1, 2, 3), (4, 5, 6, 7))
    result = random_capacity_refinement(blocks, 8, 6, 17)
    assert result == random_capacity_refinement(blocks, 8, 6, 17)
    assert len(result) == 6 and sorted(s for b in result for s in b) == list(range(8))
    assert all(any(set(b) <= set(parent) for parent in blocks) for b in result)
    assert len(random_capacity_refinement(blocks, 8, 8, 17)) == 8
    assert random_capacity_refinement(blocks, 8, 2, 17) == blocks
    with pytest.raises(ValueError):
        random_capacity_refinement(blocks, 8, 9, 17)
    with pytest.raises(ValueError):
        random_capacity_refinement(((0, 1), (1, 2)), 3, 3, 17)


def test_group_point_model_uses_fresh_counts_convex_weights_and_zero_support_fallback():
    n = np.zeros((3, 75), dtype=np.int64)
    y = np.zeros_like(n)
    n[:2, 0] = [10, 2]
    y[:2, 0] = [0, 2]
    result = group_response_model(n, y, ((0, 1), (2,)))
    assert result["parent"][2, 0] == pytest.approx(2.5 / 13)
    assert result["abstract"][0, 0] == pytest.approx(3 / 14)
    assert np.array_equal(result["mapping"], [0, 0, 1])
    assert np.all(result["parent"][:, 1:] == 0.5)
    assert result["maximum_point_discrepancy"] > 0
    identity = group_response_model(n, y, ((0,), (1,), (2,)))
    assert identity["maximum_point_discrepancy"] == 0
    y[2, 1] = 1
    with pytest.raises(ValueError):
        group_response_model(n, y, ((0, 1, 2),))


def test_adjacent_float_witness_still_separates():
    p = np.full((2, 75), 0.5)
    p[:, 0] = [np.nextafter(1.0, 0.0), 1.0]
    assert split_response_witness(((0, 1),), p, dict(states=(0, 1), environment=0)) == ((0,), (1,))
