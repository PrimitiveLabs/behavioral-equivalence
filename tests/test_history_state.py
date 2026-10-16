import numpy as np
import pytest
from behavioral_equivalence.history_state import standard_training_sample, initial_predictive_blocks


def test_standard_sample_excludes_all_random_exposures_without_responses():
    cases = np.array([10, 20, 30, 40, 50])
    random = np.array([20, 40, 99])
    assert np.array_equal(standard_training_sample(cases, random, 100, 17), [0, 2, 4])
    chosen = standard_training_sample(cases, random, 2, 17)
    assert len(chosen) == 2 and np.all(np.isin(chosen, [0, 2, 4]))
    assert np.array_equal(chosen, standard_training_sample(cases, random, 2, 17))


@pytest.mark.parametrize(
    "cases,random,size",
    [([1, 1], [], 1), ([1], [1], 1), ([-1], [], 1), ([1], [2, 2], 1), ([1], [], 0)],
)
def test_invalid_training_sample(cases, random, size):
    with pytest.raises(ValueError):
        standard_training_sample(
            np.array(cases, dtype=np.int64), np.array(random, dtype=np.int64), size, 0
        )


def test_initial_blocks_are_complete_stable_training_predictive_groups():
    assert initial_predictive_blocks([0.7, 0.1, 0.7, 0.2, 0.9], 2) == ((0, 1, 3), (2, 4))
    assert initial_predictive_blocks([0.5, 0.5, 0.5, 0.5], 2) == ((0, 1), (2, 3))
    with pytest.raises(ValueError):
        initial_predictive_blocks([np.nan], 1)
    with pytest.raises(ValueError):
        initial_predictive_blocks([0.5], 2)
