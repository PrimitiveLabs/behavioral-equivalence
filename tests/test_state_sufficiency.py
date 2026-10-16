import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "state_sufficiency", Path(__file__).parents[1] / "tools/state_sufficiency.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_equal_users_within_each_state_category_and_disjoint_bucket_merge():
    a = m.cell_totals([1, 1, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0], [0.5] * 4, 2)
    b = m.cell_totals([3, 4], [0, 0], [1, 1], [1, 1], [0.5, 0.5], 2)
    r = m.combine_cells([a, b], 2, 2)
    assert r["cells"][0]["observed_equal_user_rate"] == 0.5
    assert r["cells"][0]["cases"] == 4 and r["cells"][0]["users"] == 2
    assert r["cells"][1]["residual"] == 0.5
    assert r["within_state_spans"][0]["observed_rate_span"] == 0.5


def test_empty_and_unsupported_cells_do_not_publish_rates_or_spans():
    assert (
        m.cell_totals(
            np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), [], [], 2
        )
        == []
    )
    r = m.combine_cells([m.cell_totals([1], [0], [0], [1], [0.5], 2)])
    assert r["cells"][0]["observed_equal_user_rate"] is None and not r["within_state_spans"]


@pytest.mark.parametrize("which", [0, 1, 2, 3, 4])
def test_invalid_state_category_target_probability_or_user_rejected(which):
    a = [np.array([1]), np.array([0]), np.array([0]), np.array([1]), np.array([0.5])]
    a[which][0] = -1
    with pytest.raises(ValueError):
        m.cell_totals(*a, 2)
