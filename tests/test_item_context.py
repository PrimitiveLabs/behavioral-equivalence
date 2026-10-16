import numpy as np
import pytest
from behavioral_equivalence.item_context import fit_item_vocabulary, encode_items


def test_vocabulary_uses_only_training_frequency_and_fixed_id_tie_break():
    training = np.array([90, 2, 90, 3, 3, 4, 4, 5])
    vocabulary = fit_item_vocabulary(training, 2)
    assert vocabulary.tolist() == [3, 4]
    assert encode_items(vocabulary, np.array([3, 90, 4, 999])).tolist() == [1, 0, 2, 0]
    np.testing.assert_array_equal(training, np.array([90, 2, 90, 3, 3, 4, 4, 5]))
    assert encode_items(np.array([], dtype=np.int64), np.array([3])).tolist() == [0]


def test_item_context_rejects_target_like_float_ids_and_invalid_vocabulary():
    with pytest.raises(ValueError, match="integer"):
        fit_item_vocabulary(np.array([0.5, 0.8]))
    with pytest.raises(ValueError, match="unique"):
        encode_items(np.array([2, 2]), np.array([2]))
    with pytest.raises(ValueError, match="Positive"):
        fit_item_vocabulary(np.array([1]), 0)
