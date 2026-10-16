"""Trajectory domains, zero-support accounting and frozen-input immutability."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def load(name, folder="tools"):
    spec = importlib.util.spec_from_file_location(
        "trajectory_boundary_" + name, ROOT / folder / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


likelihood = load("trajectory_likelihood")
backward = load("trajectory_backward_audit")
design = load("trajectory_design")
targets = load("trajectory_targets")
freeze = load("freeze_trajectory")


@pytest.mark.parametrize(
    "ids,limit,seed",
    [
        ([[1]], 1, 1),
        ([1.0], 1, 1),
        ([-1], 1, 1),
        ([1], 0, 1),
        ([1], True, 1),
        ([1], 1.5, 1),
        ([1], 1, -1),
        ([1], 1, True),
        ([1], 1, 1.5),
    ],
)
def test_outcome_blind_sampling_rejects_invalid_ids_and_settings(ids, limit, seed):
    with pytest.raises(ValueError):
        design.sampled_case_ids(ids, limit, seed)


def test_empty_case_sampling_does_not_invent_origins():
    assert design.sampled_case_ids(np.array([], dtype=int), 10, 1).size == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(emissions=np.empty((0, 2))),
        dict(transitions=np.ones((1, 2))),
        dict(initial_states=np.array([[0]])),
        dict(initial_states=np.array([0.0])),
        dict(initial_states=np.array([-1])),
        dict(outcomes=np.array([0])),
        dict(outcomes=np.zeros((2, 1))),
        dict(transitions=np.array([[[np.nan], [1]]])),
        dict(transitions=np.array([[[-1.0], [1]]])),
        dict(emissions=np.array([[0.1, 0.1]])),
    ],
)
def test_forward_requires_stochastic_model_and_aligned_origin_sequences(change):
    args = dict(
        transitions=np.ones((1, 2, 1)),
        emissions=np.array([[0.5, 0.5]]),
        initial_states=np.array([0]),
        outcomes=np.array([[0]]),
    )
    args.update(change)
    with pytest.raises(ValueError, match="Stochastic binary"):
        likelihood.sequence_nll(**args)


def test_empty_batches_and_prefix_losses_preserve_inputs_and_impossible_support():
    t = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=float)
    e = np.array([[1, 0], [0.5, 0.5]], dtype=float)
    saved_t, saved_e = t.copy(), e.copy()
    states = np.array([0, 1])
    y = np.array([[0, 1, 0], [1, 0, 1]])
    prefix = likelihood.sequence_nll(t, e, states, y, return_prefix=True)
    assert prefix[0, 0] == 0 and np.isinf(prefix[0, 1:]).all()
    assert np.isinf(backward.backward_nll(t, e, states, y)[0])
    np.testing.assert_array_equal(t, saved_t)
    np.testing.assert_array_equal(e, saved_e)
    empty_states = np.array([], dtype=int)
    empty_y = np.empty((0, 3), dtype=int)
    assert likelihood.sequence_nll(t, e, empty_states, empty_y).shape == (0,)
    assert backward.backward_nll(t, e, empty_states, empty_y).shape == (0,)


@pytest.mark.parametrize(
    "matrix",
    [
        np.empty((0, 0)),
        np.ones((2, 2)) / 2,
        np.ones((3, 2)),
        np.full((3, 3), np.nan),
        np.array([[-1, 1, 1], [1, 0, 0], [1, 0, 0]]),
        np.eye(3) * 0.5,
    ],
)
def test_decision_extraction_rejects_invalid_stochastic_lifts(matrix):
    with pytest.raises(ValueError, match="Stochastic three-stage"):
        design.decision_arrays(matrix)


@pytest.mark.parametrize(
    "users,losses",
    [([[1]], [[1]]), ([1, 2], [1]), ([1.0], [1]), ([-1], [1]), ([1], [-1]), ([1], [-np.inf])],
)
def test_user_loss_aggregation_rejects_invalid_original_clusters_or_losses(users, losses):
    with pytest.raises(ValueError, match="Original user"):
        design.user_sequence_losses(users, losses)


def test_empty_user_losses_retain_empty_support():
    ids, counts, means = design.user_sequence_losses(np.array([], dtype=int), np.array([]))
    assert ids.size == counts.size == means.size == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(users=[[1]]),
        dict(roles=[]),
        dict(outcomes=[]),
        dict(origins=[[0]]),
        dict(origins=[0.0]),
        dict(origins=[-1]),
        dict(origins=[1]),
        dict(horizons=[]),
        dict(horizons=[1, 1]),
        dict(horizons=[True]),
        dict(observed=np.array([[1]])),
        dict(counts=np.array([1])),
        dict(last=np.array([1])),
        dict(outcomes=[2]),
    ],
)
def test_reconstruction_requires_aligned_audited_arrays(change):
    args = dict(
        users=[1],
        roles=[3],
        outcomes=[1],
        origins=np.array([0]),
        observed=np.array([[True]]),
        counts=np.array([[1]]),
        last=np.array([[1]]),
        horizons=[1],
    )
    args.update(change)
    with pytest.raises(ValueError, match="Aligned binary"):
        targets.reconstruct_sequences(**args)


@pytest.fixture
def frozen_inputs(tmp_path):
    helper = load("test_freeze_trajectory", "tests")
    helper.fixture(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "mutation",
    [
        "sequence_status",
        "sequence_source",
        "sequence_duplicate",
        "forecast_status",
        "forecast_source",
        "forecast_protocol",
        "parent_missing",
    ],
)
def test_trajectory_freeze_rejects_source_status_and_matrix_changes(frozen_inputs, mutation):
    root = frozen_inputs
    if mutation.startswith("sequence"):
        path = root / "reports/preflight/trajectory-sequences-f0ea31ab14ee9c5e.json"
        row = json.loads(path.read_text())
        if mutation == "sequence_status":
            row["status"] = "running"
        elif mutation == "sequence_source":
            row["source_digest"] = "changed"
        else:
            row["buckets"][-1] = row["buckets"][0]
    elif mutation.startswith("forecast"):
        path = root / "reports/final-human/direct_horizon-known_users-20261015-24.json"
        row = json.loads(path.read_text())
        row[
            {
                "forecast_status": "status",
                "forecast_source": "source_digest",
                "forecast_protocol": "protocol",
            }[mutation]
        ] = "changed"
    else:
        path = root / "configs/human-final-horizon-v1.json"
        row = json.loads(path.read_text())
        row["probes"].pop()
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        freeze.build_config(root)


def test_actual_trajectory_freeze_cli_is_idempotent_and_refuses_replacement(frozen_inputs):
    root = frozen_inputs
    script = root / "tools/freeze_trajectory.py"
    script.parent.mkdir()
    script.write_bytes((ROOT / "tools/freeze_trajectory.py").read_bytes())
    subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
    output = root / "configs/human-trajectory-final-v1.json"
    original = output.read_bytes()
    subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
    assert output.read_bytes() == original
    output.write_text('{"changed":true}')
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode != 0 and "cannot change" in result.stderr
    assert output.read_text() == '{"changed":true}'
