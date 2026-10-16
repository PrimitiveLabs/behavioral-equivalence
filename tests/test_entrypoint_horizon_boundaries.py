"""Planner dependency gates and frozen horizon-head parameter boundaries."""

from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from behavioral_equivalence.__main__ import plan
from behavioral_equivalence.horizon_heads import predict_horizon_heads, fit_horizon_heads
from behavioral_equivalence.horizon_targets import horizon_event_targets


def configuration():
    return dict(
        tasks=[
            dict(id="a", status="complete", depends_on=[]),
            dict(id="b", status="pending", depends_on=["a"]),
            dict(id="c", status="running", depends_on=["a"]),
        ],
        deliverables={"D1": ["a", "b"]},
        budget_usd=None,
    )


def test_ready_tasks_require_completed_dependencies_and_exclude_running():
    c = configuration()
    assert plan(c)["ready"] == ["b"]
    c["tasks"][0]["status"] = "pending"
    assert plan(c)["ready"] == ["a"]


@pytest.mark.parametrize(
    "damage", ["duplicate", "missing", "self_cycle", "empty_deliverable", "bad_deliverable"]
)
def test_invalid_plan_structure(damage):
    c = configuration()
    if damage == "duplicate":
        c["tasks"].append(deepcopy(c["tasks"][0]))
    elif damage == "missing":
        c["tasks"][0]["depends_on"] = ["missing"]
    elif damage == "self_cycle":
        c["tasks"][0]["depends_on"] = ["a"]
    elif damage == "empty_deliverable":
        c["deliverables"]["D1"] = []
    else:
        c["deliverables"]["D1"] = ["missing"]
    with pytest.raises(ValueError):
        plan(c)


def test_cli_reads_specified_configuration_without_submitting(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(configuration()))
    result = subprocess.run(
        [sys.executable, "-m", "behavioral_equivalence", "plan", "--config", str(p)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == plan(configuration())


def parameters():
    return dict(
        mean=np.zeros(2),
        scale=np.ones(2),
        coefficients=np.ones((2, 2)),
        intercepts=np.zeros(2),
        is_constant=np.array([False, True]),
        constants=np.array([0.0, 1.0]),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", np.zeros((1, 2))),
        ("scale", np.array([0.0, 1.0])),
        ("scale", np.array([np.nan, 1.0])),
        ("coefficients", np.ones((2, 3))),
        ("coefficients", np.empty((0, 2))),
        ("intercepts", np.array([np.inf, 0.0])),
        ("is_constant", np.array([0, 1])),
        ("constants", np.array([0.0, 1.1])),
    ],
)
def test_predictor_rejects_invalid_frozen_parameters(field, value):
    p = parameters()
    p[field] = value
    with pytest.raises(ValueError, match="Finite aligned"):
        predict_horizon_heads(np.zeros((1, 2)), p)


def test_empty_prediction_and_constant_head_are_stable():
    p = parameters()
    assert predict_horizon_heads(np.empty((0, 2)), p).shape == (0, 2)
    result = predict_horizon_heads(np.array([[10000.0, 10000.0], [-10000.0, -10000.0]]), p)
    np.testing.assert_array_equal(result, [[1.0, 1.0], [0.0, 1.0]])


@pytest.mark.parametrize("damage", ["missing", "impossible", "identity", "features", "users"])
def test_horizon_fit_rejects_inconsistent_training_inputs(damage):
    pytest.importorskip("sklearn")
    x = np.arange(12, dtype=float).reshape(6, 2)
    y, _ = horizon_event_targets(
        [1], np.ones((6, 1), dtype=bool), np.zeros((6, 1), dtype=int), np.zeros((6, 1), dtype=int)
    )
    u = np.arange(6)
    if damage == "missing":
        y[:, 0] = -1
    elif damage == "impossible":
        y[:, 3] = 1
    elif damage == "identity":
        y[:, 1] = 1
    elif damage == "features":
        x[0, 0] = np.nan
    else:
        u[0] = -1
    with pytest.raises(ValueError):
        fit_horizon_heads(x, y, u, [1])
