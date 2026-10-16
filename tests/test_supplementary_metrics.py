import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "supplementary_metrics", Path(__file__).parents[1] / "tools/supplementary_metrics.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_categorical_user_means_missing_cases_and_class_calibration():
    block = module.categorical_block(
        np.array([1, 1, 2, 3]),
        np.array([0, 0, 0, -1]),
        np.array([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]),
    )
    result = module.merge_categorical_blocks([block])
    m = result["metrics"]
    assert m["cases"] == 3 and m["users"] == 2 and m["missing_cases"] == 1
    assert m["nll_equal_user_mean"] == pytest.approx((-np.log(0.9) - np.log(0.2)) / 2)
    assert m["brier_equal_user_mean"] == pytest.approx((0.02 + 1.28) / 2)
    assert len(m["class_calibration_bins"]) == 2
    for rows in m["class_calibration_bins"].values():
        assert sum(r["equal_user_weight"] for r in rows) == pytest.approx(1)


def test_user_cannot_cross_disjoint_bucket_boundary():
    block = module.categorical_block(
        np.array([1]), np.array([0]), np.array([[0.25, 0.25, 0.25, 0.25]])
    )
    with pytest.raises(ValueError, match="multiple"):
        module.merge_categorical_blocks([block, block])
    empty = module.categorical_block(
        np.array([2]), np.array([-1]), np.array([[0.25, 0.25, 0.25, 0.25]])
    )
    assert module.merge_categorical_blocks([empty])["metrics"]["users"] == 0


def test_four_class_log_loss_uses_category_not_binary_complement():
    block = module.categorical_block(
        np.array([1, 2]), np.array([3, 0]), np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]])
    )
    result = module.merge_categorical_blocks([block])
    assert result["metrics"]["nll_equal_user_mean"] == pytest.approx(
        (-np.log(0.4) - np.log(0.7)) / 2
    )
