import numpy as np
import pytest
from behavioral_equivalence.response_frontier import response_frontier, select_partition


def tables():
    n = np.full((8, 75), 10, dtype=np.int64)
    users = np.full_like(n, 5)
    p = np.tile(np.linspace(0.1, 0.9, 8)[:, None], (1, 75))
    p[:, 4::5] = np.nan
    # Unknown contexts have counts but are never candidate witnesses.
    p[:, 4::5] = 0.5
    return dict(counts=n, users=users, probabilities=p), dict(counts=n.copy(), users=users.copy())


def test_all_controls_match_target_capacity_and_preserve_initial_classes():
    d, e = tables()
    initial = ((0, 1, 2, 3), (4, 5, 6, 7))
    rows = response_frontier(d, e, initial, [1, 3], 17, min_cases=2, min_users=2)
    for budget in [1, 3]:
        chosen = [r for r in rows if r["split_budget"] == budget]
        assert len(chosen) == 6 and len({len(r["blocks"]) for r in chosen}) == 1
        for row in chosen:
            assert sorted(s for b in row["blocks"] for s in b) == list(range(8))
            assert all(any(set(b) <= set(parent) for parent in initial) for b in row["blocks"])
            assert all(t["environment"] % 5 != 4 for t in row["trace"])


def test_estimation_outcomes_are_not_selector_inputs_and_support_is_required():
    d, e = tables()
    initial = ((0, 1, 2, 3), (4, 5, 6, 7))
    first = select_partition(d, e, initial, 2, "targeted", 17, min_cases=2, min_users=2)
    e["positives"] = np.zeros_like(e["counts"])
    assert select_partition(d, e, initial, 2, "targeted", 17, min_cases=2, min_users=2) == first
    e["counts"][:] = 0
    e["users"][:] = 0
    assert select_partition(d, e, initial, 2, "targeted", 17, min_cases=2, min_users=2) == (
        initial,
        [],
    )
    with pytest.raises(ValueError):
        response_frontier(d, e, initial, [1, 1], 17)
    with pytest.raises(ValueError):
        select_partition(d, e, initial, 1, "missing", 17)


def test_exact_count_gate_rejects_small_or_boundary_gaps_for_every_selector():
    n = np.full((4, 75), 10000, dtype=np.int64)
    y = np.repeat(np.array([4000, 4300, 8000, 8100])[:, None], 75, axis=1)
    users = np.full_like(n, 1000)
    discovery = dict(counts=n, positives=y, users=users, probabilities=y / n)
    estimation = dict(counts=n.copy(), users=users.copy())
    initial = ((0, 1), (2, 3))
    rows = response_frontier(
        discovery, estimation, initial, [2], 17, minimum_probability_gap=0.02, exact_count_gate=True
    )
    for row in rows:
        assert all(t["states"] == (0, 1) for t in row["trace"])
    assert next(r for r in rows if r["name"] == "targeted-b2")["blocks"] == ((0,), (1,), (2, 3))
    y[1, :] = 4200
    discovery["probabilities"] = y / n
    blocks, trace = select_partition(
        discovery,
        estimation,
        initial,
        2,
        "targeted",
        17,
        minimum_probability_gap=0.02,
        exact_count_gate=True,
    )
    assert blocks == initial and not trace
