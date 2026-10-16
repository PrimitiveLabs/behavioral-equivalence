import importlib.util
import itertools
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "trajectory_likelihood", Path(__file__).parents[1] / "tools/trajectory_likelihood.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_forward_matches_exhaustive_hidden_paths_and_distribution_normalizes():
    t = np.array([[[0.75, 0.25], [0.25, 0.75]], [[0.5, 0.5], [1, 0]]])
    e = np.array([[0.25, 0.75], [0.75, 0.25]])
    outcomes = np.array(list(itertools.product([0, 1], repeat=4)))
    for initial in [0, 1]:
        expected = []
        for observed in outcomes:
            total = 0
            for future in itertools.product([0, 1], repeat=3):
                path = (initial,) + future
                p = e[path[0], observed[0]]
                for j in range(1, 4):
                    p *= t[path[j - 1], observed[j - 1], path[j]] * e[path[j], observed[j]]
                total += p
            expected.append(total)
        actual = module.sequence_nll(t, e, np.full(16, initial), outcomes)
        assert np.exp(-actual) == pytest.approx(expected)
        assert np.exp(-actual).sum() == pytest.approx(1)


def test_empty_horizon_impossible_path_and_long_sequence_scaling():
    t = np.ones((1, 2, 1))
    assert module.sequence_nll(t, [[0, 1]], np.array([0]), np.zeros((1, 0))).item() == 0
    assert np.isinf(module.sequence_nll(t, [[0, 1]], np.array([0]), [[0, 1]])).all()
    value = module.sequence_nll(t, [[0.5, 0.5]], np.array([0]), np.ones((1, 2000)))
    assert value.item() == pytest.approx(2000 * np.log(2))


def test_outcome_conditioned_transition_precedes_only_the_next_emission():
    t = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    e = np.array([[0.5, 0.5], [1, 0]])
    assert module.sequence_nll(t, e, np.array([0, 0]), [[0, 0], [1, 1]]) == pytest.approx(
        [-np.log(0.5), -np.log(0.25)]
    )


@pytest.mark.parametrize("change", ["negative", "sum", "state", "outcome", "nan"])
def test_rejects_invalid_model_or_sequences(change):
    t = np.ones((1, 2, 1))
    e = np.array([[0.5, 0.5]])
    state = np.array([0])
    y = np.array([[1]])
    if change == "negative":
        e[0] = [-0.5, 1.5]
    if change == "sum":
        t[0, 0, 0] = 0.5
    if change == "state":
        state[0] = 1
    if change == "outcome":
        y[0, 0] = -1
    if change == "nan":
        e[0, 0] = np.nan
    with pytest.raises(ValueError):
        module.sequence_nll(t, e, state, y)


def test_prefix_losses_match_independent_shorter_sequences():
    t = np.array([[[0.75, 0.25], [0.25, 0.75]], [[0.5, 0.5], [1, 0]]])
    e = np.array([[0.25, 0.75], [0.75, 0.25]])
    states = np.array([0, 1])
    y = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])
    result = module.sequence_nll(t, e, states, y, return_prefix=True)
    for h in range(1, 5):
        assert result[:, h - 1] == pytest.approx(module.sequence_nll(t, e, states, y[:, :h]))
    assert module.sequence_nll(t, e, states, y[:, :0], return_prefix=True).shape == (2, 0)
    with pytest.raises(ValueError):
        module.sequence_nll(t, e, states, y, return_prefix=1)
