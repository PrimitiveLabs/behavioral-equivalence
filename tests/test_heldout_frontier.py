from fractions import Fraction
import numpy as np
from behavioral_equivalence.heldout_frontier import heldout_refinement_frontier


def table():
    n = np.full((4, 75), 10000, dtype=np.int64)
    y = np.repeat(np.array([2000, 2100, 8000, 8100])[:, None], 75, axis=1)
    users = np.full_like(n, 1000)
    return dict(counts=n, positives=y, users=users, probabilities=y / n)


def test_all_controls_match_capacity_and_retain_neutral_certificate():
    d = table()
    e = table()
    rows = heldout_refinement_frontier(
        [0.49, 0.495, 0.505, 0.51], d, e, (0, 1), [1, 2], 17, epsilon=0.02
    )
    assert len(rows) == 14
    for budget in [1, 2]:
        group = [r for r in rows if r["split_budget"] == budget]
        assert len(group) == 6 and len({len(r["blocks"]) for r in group}) == 1
    for row in rows:
        value = row["neutral_certificate"]["epsilon"]
        radius = Fraction(int(value["numerator_hex"], 16), int(value["denominator_hex"], 16))
        assert radius <= Fraction.from_float(0.02)
        assert all(w["environment"] % 5 in {0, 1} for w in row["trace"])
        assert all(c["exceeds_budget"] for c in row.get("witness_certificates", []))


def test_hidden_labels_cannot_select_partitions_or_witnesses():
    d = table()
    e = table()
    neutral = [0.49, 0.495, 0.505, 0.51]
    a = heldout_refinement_frontier(neutral, d, e, (0, 1), [2], 17, epsilon=0.02)
    hidden = ~np.isin(np.arange(75) % 5, [0, 1])
    d["positives"][:, hidden] = 0
    e["positives"][:, hidden] = 10000
    d["probabilities"] = d["positives"] / d["counts"]
    e["probabilities"] = e["positives"] / e["counts"]
    b = heldout_refinement_frontier(neutral, d, e, (0, 1), [2], 17, epsilon=0.02)
    assert [r["blocks"] for r in a] == [r["blocks"] for r in b]
    assert [r["trace"] for r in a] == [r["trace"] for r in b]


def test_no_supported_query_witnesses_produces_unchanged_matched_controls():
    d = table()
    e = table()
    d["users"][:] = 1
    e["users"][:] = 1
    rows = heldout_refinement_frontier(
        [0.49, 0.495, 0.505, 0.51], d, e, (2, 3), [8], 17, epsilon=0.02
    )
    initial = rows[0]["blocks"]
    assert all(r["blocks"] == initial for r in rows if r["split_budget"] == 8)
    assert all(not r["trace"] for r in rows if r["split_budget"] == 8)


def test_precision_contract_names_match_constructed_frontier():
    import json
    from pathlib import Path

    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/human-heldout-response-precision-v1.json"
        ).read_text()
    )
    rows = heldout_refinement_frontier(
        [0.49, 0.495, 0.505, 0.51], table(), table(), (0, 1), [8], 17, epsilon=0.02
    )
    names = {row["name"] for row in rows}
    assert config["primary_candidate"] in names
    assert config["primary_reference"] in names
