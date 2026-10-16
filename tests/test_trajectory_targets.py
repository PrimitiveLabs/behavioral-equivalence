import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "trajectory_targets", Path(__file__).parents[1] / "tools/trajectory_targets.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def arguments():
    return [
        np.array([1, 1, 1, 2, 2]),
        np.array([3, 3, 3, 3, 2]),
        np.array([1, 0, 1, 0, 1]),
        np.array([0, 2, 3]),
        np.array([[True, True], [True, False], [True, False]]),
        np.array([[1, 1], [1, -1], [0, -1]]),
        np.array([[1, 0], [1, -1], [0, -1]]),
        [1, 2],
    ]


def test_exact_order_user_boundary_role_boundary_and_missing_sentinels():
    assert np.array_equal(module.reconstruct_sequences(*arguments()), [[1, 0], [1, -1], [0, -1]])


@pytest.mark.parametrize("change", ["count", "last", "support", "sentinel", "origin"])
def test_reconstruction_rejects_misaligned_or_corrupted_audited_targets(change):
    a = arguments()
    if change == "count":
        a[5][0, 1] = 2
    if change == "last":
        a[6][0, 1] = 1
    if change == "support":
        a[4][1, 1] = True
    if change == "sentinel":
        a[5][2, 1] = 0
    if change == "origin":
        a[3][1] = 0
    with pytest.raises(ValueError):
        module.reconstruct_sequences(*a)
