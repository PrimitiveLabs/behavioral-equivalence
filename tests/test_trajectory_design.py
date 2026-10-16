import importlib.util
from pathlib import Path
import numpy as np
import pytest
from behavioral_equivalence.decision_model import decision_chain

spec = importlib.util.spec_from_file_location(
    "trajectory_design", Path(__file__).parents[1] / "tools/trajectory_design.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sampling_is_order_independent_and_never_replicates_small_cohorts():
    ids = np.arange(200, dtype=np.uint64)
    assert np.array_equal(
        module.sampled_case_ids(ids, 20, 7), module.sampled_case_ids(ids[::-1], 20, 7)
    )
    assert np.array_equal(module.sampled_case_ids(ids[:3], 20, 7), ids[:3])
    assert not np.array_equal(
        module.sampled_case_ids(ids, 20, 7), module.sampled_case_ids(ids, 20, 8)
    )
    with pytest.raises(ValueError):
        module.sampled_case_ids([1, 1], 1, 0)


def test_extracts_only_the_documented_decision_lift():
    t = np.array([[[0.75, 0.25], [0.25, 0.75]], [[0.5, 0.5], [1, 0]]])
    e = np.array([[0.25, 0.75], [0.75, 0.25]])
    chain = decision_chain(t, e, np.array([0.5, 0.5]))
    actual_t, actual_e = module.decision_arrays(chain.transitions)
    assert np.array_equal(actual_t, t) and np.array_equal(actual_e, e)
    p = chain.transitions.copy()
    p[0, 1] -= 0.1
    p[0, 4] += 0.1
    with pytest.raises(ValueError):
        module.decision_arrays(p)


def test_equal_user_losses_preserve_zero_support_failures():
    ids, counts, means = module.user_sequence_losses([2, 1, 2, 3], [2, 4, 6, np.inf])
    assert np.array_equal(ids, [1, 2, 3]) and np.array_equal(counts, [1, 2, 1])
    assert np.array_equal(means, [4, 4, np.inf])
    with pytest.raises(ValueError):
        module.user_sequence_losses([1], [np.nan])
