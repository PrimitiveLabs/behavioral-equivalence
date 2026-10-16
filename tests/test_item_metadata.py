import numpy as np
import pytest
from behavioral_equivalence.item_metadata import duration_groups


def test_duration_group_boundaries_and_missing_context_are_explicit():
    values = np.array([-1, 0, np.nan, np.inf, 1, 7000, 7001, 18000, 18001, 60000, 60001])
    assert duration_groups(values).tolist() == [-1, -1, -1, -1, 0, 0, 1, 1, 2, 2, 3]
    with pytest.raises(ValueError, match="dimensional"):
        duration_groups(np.array([[1, 2]]))
