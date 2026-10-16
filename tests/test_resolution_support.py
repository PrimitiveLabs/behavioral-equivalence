import numpy as np
import pytest
from behavioral_equivalence.resolution_support import (
    resolution_map,
    cell_support,
    supported_pair_counts,
)


def test_training_rank_maps_nest_at_power_of_two_resolutions():
    p = np.arange(256)[::-1] / 256
    fine = resolution_map(p, 32)
    coarse = resolution_map(p, 16)
    assert np.array_equal(fine // 2, coarse)
    assert np.array_equal(resolution_map(np.full(16, 0.5), 4), np.repeat(np.arange(4), 4))


def test_coarse_user_counts_deduplicate_across_original_states():
    mapping = resolution_map(np.array([0.1, 0.2, 0.8, 0.9]), 2)
    s = np.array([0, 1, 2, 3, 3])
    e = np.array([0, 0, 0, 0, 4])
    u = np.array([1, 1, 2, 3, 3])
    table = cell_support(mapping[s], e, u, 2)
    assert table["counts"][0, 0] == 2 and table["users"][0, 0] == 1
    assert table["users"][1, 0] == 2
    assert cell_support(s[:0], e[:0], u[:0], 4)["counts"].sum() == 0


def test_support_pairs_require_same_initial_block_and_known_environment():
    n = np.zeros((8, 75), dtype=np.int64)
    u = np.zeros_like(n)
    n[[0, 1, 4], 0] = 2000
    u[[0, 1, 4], 0] = 50
    n[:, 4] = 3000
    u[:, 4] = 60
    assert supported_pair_counts(n, u) == dict(
        supported_state_environment_cells=3, supported_within_initial_block_pairs=1
    )
    u[1, 0] = 49
    assert supported_pair_counts(n, u)["supported_within_initial_block_pairs"] == 0


@pytest.mark.parametrize(
    "s,e,u,k",
    [([-1], [0], [0], 4), ([0], [75], [0], 4), ([0], [0], [-1], 4), ([0, 1], [0], [0], 4)],
)
def test_bad_support_inputs(s, e, u, k):
    with pytest.raises(ValueError):
        cell_support(np.array(s), np.array(e), np.array(u), k)
