import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

folder = Path(__file__).parents[1] / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, folder / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


module = load("supplementary_result_audit")
production = load("supplementary_metrics")
sys.path.insert(0, str(folder))
try:
    reference = load("final_result_audit")
finally:
    sys.path.pop(0)


def test_independent_user_sort_losses_and_full_calibration_agree():
    u = np.array([2, 1, 1, 3])
    y = np.array([0, 1, 0, -1])
    p = np.array([[0.2, 0.8], [0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
    audit = module.categorical_audit_block(u, y, p)
    actual = reference.combine([audit])
    expected = production.merge_categorical_blocks([production.categorical_block(u, y, p)])
    assert np.allclose(audit["nll"], [-np.log(0.9), -np.log(0.2)])
    assert max(reference.check_metrics(actual["metrics"], expected["metrics"]).values()) < 1e-14
    assert module.verify_calibration(expected["metrics"], [audit], 2) < 1e-14
    expected["metrics"]["class_calibration_bins"]["0"][0]["cases"] += 1
    with pytest.raises(ValueError, match="support"):
        module.verify_calibration(expected["metrics"], [audit], 2)


def test_four_categories_extreme_probabilities_and_bin_edges():
    u = np.array([1, 2])
    y = np.array([3, 0])
    p = np.array([[0.0, 0.05, 0.25, 0.7], [1.0, 0.0, 0.0, 0.0]])
    actual = module.categorical_audit_block(u, y, p)
    expected = production.merge_categorical_blocks([production.categorical_block(u, y, p)])
    assert (
        max(
            reference.check_metrics(
                reference.combine([actual])["metrics"], expected["metrics"]
            ).values()
        )
        < 1e-10
    )
    assert module.verify_calibration(expected["metrics"], [actual], 2) < 1e-10


def test_bootstrap_matches_production_paired_resampling_with_different_batching():
    from behavioral_equivalence.streaming_scores import shared_user_bootstrap

    x = np.array([[0.1, 0.3], [0.5, 0.7], [0.2, 0.1], [0.8, 0.6], [0.4, 0.9]])
    production_result = shared_user_bootstrap(
        np.arange(5),
        np.ones(5, dtype=int),
        {"a": x[:, 0], "b": x[:, 1], "base": np.zeros(5)},
        {"a": ("a", "base"), "b": ("b", "base")},
        replications=200,
        seed=42,
    )
    actual = module.bootstrap_many(x, replications=200, seed=42)
    assert np.allclose(
        actual,
        np.column_stack([production_result[n]["ci95"] for n in ["a", "b"]]),
        rtol=0,
        atol=1e-14,
    )
