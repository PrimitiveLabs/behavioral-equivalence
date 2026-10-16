import importlib.util
from pathlib import Path
from fractions import Fraction
import numpy as np
import pytest
from scipy.optimize import linprog
from behavioral_equivalence.uncertainty import IntervalChain

spec = importlib.util.spec_from_file_location(
    "interval_vertices", Path(__file__).parents[1] / "tools/interval_vertices.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_simplex_vertices_and_independent_linear_program_extrema():
    vertices = module.row_vertices([0, 0, 0], [1, 1, 1])
    assert len(vertices) == 3 and all(
        sum(row) == 1 and sum(v == 1 for v in row) == 1 for row in vertices
    )
    lo = np.array([0, 0.25, 0])
    hi = np.array([0.5, 0.75, 0.75])
    vertices = module.row_vertices(lo, hi)
    for objective in [np.array([2.0, -1.0, 3.0]), np.array([-0.2, 0.9, 0.1])]:
        actual = min(sum(float(p) * c for p, c in zip(row, objective)) for row in vertices)
        expected = linprog(
            objective, A_eq=np.ones((1, 3)), b_eq=[1.0], bounds=list(zip(lo, hi)), method="highs"
        )
        assert expected.success and actual == pytest.approx(expected.fun, abs=1e-12)


def test_zero_lower_rows_export_as_supported_nondeterministic_point_actions(tmp_path):
    model = IntervalChain(
        np.array([[0.0, 0.25], [0.0, 1.0]]),
        np.array([[0.75, 1.0], [0.0, 1.0]]),
        ("active", "complete"),
    )
    assert module.vertex_reachability(model, "complete", 2, False)[0] == Fraction(7, 16)
    assert module.vertex_reachability(model, "complete", 2, True)[0] == 1
    path = tmp_path / "model.prism"
    counts = module.write_vertex_mdp(model, path, 0)
    assert counts == [2, 1] and path.read_text().startswith("mdp\n")
    assert "[row0action0]" in path.read_text() and 'label "complete" = s=1;' in path.read_text()


@pytest.mark.parametrize(
    "lo,hi",
    [
        ([0] * 9, [1] * 9),
        ([0.8, 0.8], [1.0, 1.0]),
        ([0, 0], [0.2, 0.2]),
        ([0, 0.5], [1, 0.4]),
        ([np.nan], [1]),
    ],
)
def test_invalid_or_oversized_rows_rejected(lo, hi):
    with pytest.raises(ValueError):
        module.row_vertices(lo, hi)
