import numpy as np
import pytest
from behavioral_equivalence.quantization import (
    fit_representation_scaling,
    assign_centers,
    emission_probabilities,
)


def test_normalization_is_fitted_only_on_supplied_training_and_ties_are_stable():
    train = np.array([[1.0, 2.0], [3.0, 2.0]])
    mean, scale = fit_representation_scaling(train)
    np.testing.assert_array_equal(mean, [2.0, 2.0])
    np.testing.assert_array_equal(scale, [1.0, 1e-6])
    centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(
        assign_centers(np.array([[1.0, 2.0], [2.0, 2.0], [100.0, 2.0]]), mean, scale, centers),
        [0, 0, 1],
    )
    np.testing.assert_array_equal(mean, [2.0, 2.0])
    np.testing.assert_array_equal(assign_centers(np.empty((0, 2)), mean, scale, centers), [])
    with pytest.raises(ValueError):
        assign_centers(train, mean, np.zeros(2), centers)
    with pytest.raises(ValueError):
        fit_representation_scaling(np.array([[np.nan]]))


def test_distillation_uses_training_teacher_sums_and_explicit_empty_state_fallback():
    np.testing.assert_allclose(
        emission_probabilities(np.array([2, 0, 1]), np.array([1.5, 0.0, 0.2]), 0.4),
        [0.75, 0.4, 0.2],
    )
    with pytest.raises(ValueError):
        emission_probabilities(np.array([1]), np.array([1.1]), 0.4)
