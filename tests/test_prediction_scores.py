import numpy as np
import pytest
from behavioral_equivalence.statistics import prediction_scores


def test_equal_user_metrics_and_bin_weighting_do_not_overweight_active_users():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.8, 0.8, 0.8, 0.2])
    users = np.array([1, 1, 1, 2])
    result = prediction_scores(y, p, users)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    assert result["nll_equal_user_mean"] == pytest.approx((loss[:3].mean() + loss[3]) / 2)
    assert result["nll_case_mean"] == pytest.approx(loss.mean())
    bins = result["calibration_bins"]
    assert bins[16]["observed"] == pytest.approx(2 / 3)
    assert bins[16]["equal_user_weight"] == pytest.approx(0.5)
    assert sum(b["equal_user_weight"] for b in bins) == pytest.approx(1)


def test_endpoint_probabilities_are_finite_and_invalid_inputs_fail():
    result = prediction_scores(np.array([0, 1]), np.array([0.0, 1.0]), np.array([1, 2]))
    assert np.isfinite(result["nll_equal_user_mean"])
    assert result["brier_equal_user_mean"] == 0
    for probability in [np.array([np.nan, 0.5]), np.array([1.1, 0.5])]:
        with pytest.raises(ValueError, match="Aligned"):
            prediction_scores(np.array([0, 1]), probability, np.array([1, 2]))
