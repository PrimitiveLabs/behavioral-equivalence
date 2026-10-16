import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "supplementary_counts", Path(__file__).parents[1] / "tools/supplementary_counts.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    y = np.array([[1, 0, 0, 0, 0, 3], [0, 1, 0, 0, 0, 0], [-1, 0, 1, 0, 1, -1]])
    return {2: np.array([0, 1, 0])}, y, np.array([1, 1, 2]), np.array([10, 11, 12])


def test_counts_keep_zero_visit_users_and_endpoint_specific_missingness():
    counts, users, cap = module.endpoint_counts(*fixture())
    assert users.tolist() == [1, 2] and cap == 2
    assert counts[(2, 0)].tolist() == [[0, 1], [1, 0]]
    assert counts[(2, 5)].tolist() == [[0, 0, 0, 1], [1, 0, 0, 0]]
    assert counts[(2, 2)].sum() == 3
    p = module.smoothed_emissions(np.array([[0, 0, 0, 0], [0, 0, 0, 4]]))
    assert np.allclose(p[0], 0.25) and np.allclose(p[1], [0.05, 0.05, 0.05, 0.85])


@pytest.mark.parametrize("change", ["duplicate", "cap", "label", "state", "length"])
def test_count_boundary_rejections(change):
    states, y, u, ids = fixture()
    cap = 64
    if change == "duplicate":
        ids[1] = ids[0]
    elif change == "cap":
        cap = 1
    elif change == "label":
        y[0, 0] = 3
    elif change == "state":
        states[2][0] = 2
    else:
        u = u[:1]
    with pytest.raises(ValueError):
        module.endpoint_counts(states, y, u, ids, cap)
