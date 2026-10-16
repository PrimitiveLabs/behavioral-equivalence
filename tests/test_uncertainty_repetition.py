import numpy as np
import pytest
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.core import Chain
from behavioral_equivalence.uncertainty import IntervalChain
from behavioral_equivalence.uncertainty_repetition import (
    row_sample_schedule,
    terminal_label_support,
)


def test_fixed_schedules_preserve_budget_and_leave_terminal_rows_unobserved():
    world = generate("memory", 32, 30000, 0.01)
    active = np.array(world.chain.labels) == "active"
    uniform = row_sample_schedule(world.chain, "uniform_active", 100)
    sparse = row_sample_schedule(world.chain, "imbalanced_active", 100)
    assert uniform.sum() == sparse.sum() == active.sum() * 100
    assert not uniform[~active].any() and not sparse[~active].any()
    assert (sparse[active] == 0).sum() >= active.sum() // 4
    np.testing.assert_array_equal(
        sparse, row_sample_schedule(world.chain, "imbalanced_active", 100)
    )


def test_known_terminal_constraint_preserves_clone_transitions_and_rejects_false_assumption():
    chain = Chain(
        np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0, 1, 0]]),
        ("active", "complete", "complete"),
        np.array([1.0, 0, 0]),
    )
    unknown = IntervalChain(np.zeros((3, 3)), np.ones((3, 3)), chain.labels)
    bounded = terminal_label_support(unknown, chain)
    assert not bounded.upper[1:, 0].any()
    assert (bounded.upper[1:, 1:] == 1).all()
    broken = Chain(np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]]), chain.labels, chain.initial)
    with pytest.raises(ValueError, match="not closed"):
        terminal_label_support(unknown, broken)
