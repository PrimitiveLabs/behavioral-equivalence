import numpy as np
import pytest
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.clustering import (
    random_partition,
    feature_partition,
    predictive_features,
)
from behavioral_equivalence.core import check_partition


def test_baselines_match_state_count_and_observable_labels_even_identical_features():
    world = generate("trust", 32, 12)
    for k in (3, 8, 16, 32):
        for partition in [
            random_partition(world.chain, k, 2),
            feature_partition(world.chain, np.zeros((32, 2)), k, 2),
            feature_partition(world.chain, predictive_features(world.chain), k, 2),
        ]:
            assert len(partition) == k
            check_partition(world.chain, partition)
    assert random_partition(world.chain, 8, 2) == random_partition(world.chain, 8, 2)


def test_invalid_baseline_dimensions_and_state_counts():
    world = generate("memory", 16, 2)
    with pytest.raises(ValueError):
        random_partition(world.chain, 2)
    with pytest.raises(ValueError):
        feature_partition(world.chain, np.zeros((15, 3)), 8)
