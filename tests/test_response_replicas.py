import numpy as np
import pytest

from behavioral_equivalence.response_replicas import response_replicas


def test_fixed_three_forecasts_per_case_without_averaging_or_changing_case_order():
    groups = np.array([3, 0, 2, 1, 3])
    names = ["q01", "q02", "q03", "q12", "q13", "q23"]
    forecasts = {name: np.full(5, i / 10) for i, name in enumerate(names)}
    actual = response_replicas(groups, forecasts)
    np.testing.assert_array_equal(
        actual, [[0, 0.1, 0.3], [0.3, 0.4, 0.5], [0, 0.2, 0.4], [0.1, 0.2, 0.5], [0, 0.1, 0.3]]
    )
    assert response_replicas(
        np.array([], dtype=int), {name: np.empty(0) for name in names}
    ).shape == (0, 3)


def test_missing_fold_unknown_group_and_misaligned_forecast_rejected():
    names = ["q01", "q02", "q03", "q12", "q13", "q23"]
    forecasts = {name: np.array([0.5]) for name in names}
    for groups, values in [
        (np.array([4]), forecasts),
        (np.array([0]), {}),
        (np.array([0]), dict(forecasts, q01=np.array([2]))),
        (np.array([0]), dict(forecasts, q01=np.empty(0))),
    ]:
        with pytest.raises(ValueError):
            response_replicas(groups, values)
