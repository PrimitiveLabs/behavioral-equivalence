import numpy as np
import pytest
from behavioral_equivalence.content_scaling import fit_duration_scaling, scale_duration


def test_scaling_uses_known_training_values_and_keeps_missing_rows_distinct():
    training = np.array([[2, 0], [4, 0], [0, 1]], dtype=np.float32)
    scaling = fit_duration_scaling(training)
    assert scaling == {"mean": 3.0, "scale": 1.0}
    np.testing.assert_array_equal(scale_duration(training, scaling), [[-1, 0], [1, 0], [0, 1]])
    np.testing.assert_array_equal(
        scale_duration(np.array([[100, 0], [0, 1]]), scaling), [[97, 0], [0, 1]]
    )
    np.testing.assert_array_equal(training, [[2, 0], [4, 0], [0, 1]])
    assert fit_duration_scaling(np.array([[0, 1]])) == {"mean": 0.0, "scale": 1.0}
    with pytest.raises(ValueError, match="Finite two"):
        fit_duration_scaling(np.array([[2, 2]]))
