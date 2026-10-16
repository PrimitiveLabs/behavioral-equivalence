import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "supplementary_heads", Path(__file__).parents[1] / "tools/supplementary_heads.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_categorical_prediction_is_simplex_and_stable_at_extreme_logits():
    p = dict(
        mean=np.zeros(1),
        scale=np.ones(1),
        coefficients=np.array([[0.0], [1.0], [-1.0], [2.0]]),
        intercept=np.zeros(4),
    )
    actual = module.predict_head(np.array([[10000.0], [-10000.0], [0.0]]), p)
    assert np.isfinite(actual).all()
    assert np.allclose(actual.sum(axis=1), 1)
    assert actual[0, 3] == 1 and actual[1, 2] == 1
    assert np.allclose(actual[2], 0.25)


def test_prediction_rejects_bad_scaling():
    p = dict(
        mean=np.zeros(1), scale=np.zeros(1), coefficients=np.zeros((2, 1)), intercept=np.zeros(2)
    )
    with pytest.raises(ValueError):
        module.predict_head([[1]], p)


@pytest.mark.parametrize("classes", [2, 4])
def test_fit_preserves_unobserved_categories_and_missing_label_exclusion(classes):
    pytest.importorskip("sklearn")
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    parameters, report = module.fit_head(x, y, np.arange(10), classes)
    actual = module.predict_head(x, parameters)
    assert actual.shape == (10, classes)
    assert np.all(actual > 0) and np.allclose(actual.sum(axis=1), 1)
    assert np.all(actual[:, 0] > actual[:, 1])
    assert report["observed_cases"] == 9 and report["observed_users"] == 9
    assert np.allclose(parameters["mean"], x[:9].mean(axis=0))


@pytest.mark.parametrize("classes", [0, 3, True, 2.5, None])
def test_invalid_class_contract_is_rejected_before_optional_backend(classes):
    with pytest.raises(ValueError, match="classes"):
        module.fit_head(np.zeros((2, 1)), np.array([0, 1]), np.array([1, 2]), classes)


def test_missing_all_training_targets_rejected_before_optional_backend():
    with pytest.raises(ValueError, match="No observed"):
        module.fit_head(np.zeros((2, 1)), np.array([-1, -1]), np.array([1, 2]), 2)


def test_categorical_scores_equal_users_and_exclude_missing_targets():
    p = np.array([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    result = module.categorical_scores(p, np.array([0, 0, 0, -1]), np.array([1, 1, 2, 3]))
    assert result["cases"] == 3 and result["users"] == 2
    assert result["nll"] == pytest.approx((-np.log(0.9) - np.log(0.2)) / 2)
    assert result["brier"] == pytest.approx((0.02 + 1.28) / 2)
    assert module.categorical_scores(p, np.full(4, -1), np.arange(4))["nll"] is None


def test_categorical_score_rejects_non_simplex_and_regularizes_zero_probability():
    assert np.isfinite(
        module.categorical_scores(np.array([[1.0, 0.0]]), np.array([1]), np.array([1]))["nll"]
    )
    with pytest.raises(ValueError):
        module.categorical_scores(np.array([[0.8, 0.8]]), np.array([1]), np.array([1]))
