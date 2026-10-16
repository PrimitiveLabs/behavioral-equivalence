import numpy as np
import pytest
from behavioral_equivalence.horizon_heads import fit_horizon_heads, predict_horizon_heads
from behavioral_equivalence.horizon_targets import horizon_event_targets


def test_prediction_uses_only_origin_features_and_frozen_parameters():
    p = dict(
        mean=np.zeros(2),
        scale=np.ones(2),
        coefficients=np.array([[1.0, 0.0], [0.0, 0.0]]),
        intercepts=np.zeros(2),
        is_constant=np.array([False, True]),
        constants=np.array([0.0, 0.0]),
    )
    result = predict_horizon_heads(np.array([[0.0, 2.0], [2.0, -2.0]]), p)
    assert result[:, 0] == pytest.approx([0.5, 1 / (1 + np.exp(-2))])
    assert np.array_equal(result[:, 1], [0, 0])
    with pytest.raises(ValueError):
        predict_horizon_heads(np.ones((2, 3)), p)


def test_fitted_heads_learn_origin_signal_and_preserve_h1_identities():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(31)
    x = rng.normal(size=(200, 3))
    first = (x[:, 0] > 0).astype(int)
    future = rng.integers(0, 2, (200, 4))
    counts = np.column_stack([first, first + future.sum(axis=1)])
    last = np.column_stack([first, future[:, -1]])
    y, _ = horizon_event_targets([1, 5], np.ones((200, 2), dtype=bool), counts, last)
    model, records = fit_horizon_heads(x, y, np.repeat(np.arange(20), 10), [1, 5])
    p = predict_horizon_heads(x, model)
    assert np.mean((p[:, 0] > 0.5) == first) > 0.95
    assert np.array_equal(p[:, 0], p[:, 2])
    assert np.allclose(p[:, 4], 1 - p[:, 0], atol=1e-15)
    assert not p[:, 6].any() and not p[:, 8].any()
    assert records[6]["kind"] == "structural_zero"
    assert len(records) == 10


def test_single_class_fallback_and_missing_targets_remain_distinct():
    pytest.importorskip("sklearn")
    x = np.arange(12, dtype=float).reshape(6, 2)
    y, _ = horizon_event_targets(
        [1], np.ones((6, 1), dtype=bool), np.zeros((6, 1), dtype=int), np.zeros((6, 1), dtype=int)
    )
    parameters, records = fit_horizon_heads(x, y, np.array([1, 1, 2, 2, 3, 3]), [1])
    p = predict_horizon_heads(x, parameters)
    assert p[:, 0] == pytest.approx(np.full(6, 0.5 / 7))
    assert records[0]["kind"] == "jeffreys_constant"
    assert not p[:, 3].any()
    y[:, 0] = -1
    with pytest.raises(ValueError, match="No observed"):
        fit_horizon_heads(x, y, np.array([1, 1, 2, 2, 3, 3]), [1])
