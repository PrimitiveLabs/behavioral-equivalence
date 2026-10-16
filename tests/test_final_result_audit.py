import importlib.util
from pathlib import Path
import numpy as np
import pytest
from behavioral_equivalence.streaming_scores import (
    metric_block,
    merge_metric_blocks,
    shared_user_bootstrap,
)

spec = importlib.util.spec_from_file_location(
    "final_audit", Path(__file__).parents[1] / "tools/final_result_audit.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_independent_sorted_user_replay_handles_replicas_float32_and_empty_rows():
    u = np.array([3, 1, 3, 1, 2])
    y = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    for p in [np.linspace(0.1, 0.9, 5).astype(np.float32), np.arange(15).reshape(5, 3) / 15]:
        actual = module.combine([module.loss_block(u, y, p)])
        expected = merge_metric_blocks([metric_block(np.arange(5), u, y, p)])
        assert max(module.check_metrics(actual["metrics"], expected["metrics"]).values()) < 1e-10
        np.testing.assert_allclose(actual["nll"], expected["private_user_statistics"]["user_nll"])
        altered = dict(expected["metrics"], nll_equal_user_mean=1.2)
        with pytest.raises(ValueError):
            module.check_metrics(actual["metrics"], altered)
    block = module.loss_block(np.array([], dtype=int), np.array([]), np.empty((0, 3)))
    assert module.combine([block])["metrics"]["cases"] == 0
    with pytest.raises(ValueError):
        module.combine([module.loss_block(u, y, p)] * 2)


def test_explicit_cluster_bootstrap_matches_shared_multiplicity_bootstrap():
    x = np.array([-0.1, 0.05, -0.03, 0.01])
    actual = module.bootstrap_primary(x, replications=257, seed=22)
    expected = shared_user_bootstrap(
        np.arange(4),
        np.ones(4, dtype=int),
        {"a": x, "b": np.zeros(4)},
        {"contrast": ("a", "b")},
        replications=257,
        seed=22,
    )["contrast"]
    np.testing.assert_allclose(actual["ci95"], expected["ci95"], atol=1e-14, rtol=0)
    assert actual["mean"] == expected["candidate_minus_reference_nats"]
