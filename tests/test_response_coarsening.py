import numpy as np
import pytest
from behavioral_equivalence.response_coarsening import coarsen_response_table


def test_coarse_counts_match_support_without_summing_duplicate_users():
    n = np.zeros((4, 75), dtype=np.int64)
    n[:, 0] = [2, 3, 4, 1]
    y = np.zeros_like(n)
    y[:, 0] = [1, 2, 0, 1]
    mapping = np.array([0, 0, 1, 1])
    expected = np.zeros((2, 75), dtype=np.int64)
    expected[:, 0] = 5
    users = np.zeros_like(expected)
    users[:, 0] = [2, 4]
    result = coarsen_response_table(n, y, mapping, expected, users)
    assert np.array_equal(result["users"][:, 0], [2, 4])
    assert np.array_equal(result["positives"][:, 0], [3, 1])
    assert np.allclose(result["probabilities"][:, 0], [0.6, 0.2])
    assert np.isnan(result["probabilities"][:, 1:]).all()
    expected[0, 0] = 6
    with pytest.raises(ValueError, match="cohorts differ"):
        coarsen_response_table(n, y, mapping, expected, users)


def test_invalid_count_or_map_does_not_silently_coarsen():
    n = np.ones((2, 75), dtype=np.int64)
    y = n.copy()
    mapping = np.array([0, 1])
    users = n.copy()
    with pytest.raises(ValueError):
        coarsen_response_table(n, y, np.array([0, 2]), n, users)
    y[0, 0] = 2
    with pytest.raises(ValueError):
        coarsen_response_table(n, y, mapping, n, users)
