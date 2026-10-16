import importlib.util
from pathlib import Path
import itertools
import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "tools" / (name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


forward = load("trajectory_likelihood")
audit = load("trajectory_backward_audit")


def test_independent_backward_matches_all_binary_sequences_and_latent_origins():
    rng = np.random.default_rng(341)
    t = rng.dirichlet(np.ones(3), size=(3, 2))
    e = rng.dirichlet(np.ones(2), size=3)
    y = np.array(list(itertools.product([0, 1], repeat=6)))
    for state in range(3):
        states = np.full(len(y), state)
        assert audit.backward_nll(t, e, states, y) == pytest.approx(
            forward.sequence_nll(t, e, states, y), abs=1e-12
        )


def test_empty_impossible_and_long_sequences():
    t = np.ones((1, 2, 1))
    states = np.array([0])
    for h in [0, 1, 2000]:
        y = np.ones((1, h), dtype=int)
        assert audit.backward_nll(t, np.array([[0.5, 0.5]]), states, y).item() == pytest.approx(
            h * np.log(2)
        )
    assert np.isinf(
        audit.backward_nll(t, np.array([[1.0, 0.0]]), states, np.ones((1, 3), dtype=int))
    ).all()
