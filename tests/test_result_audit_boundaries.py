"""Exact identifiers and nonfinite calibration rejection in independent audits."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from test_supplementary_result_audit import module as audit, production

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "audit_review_" + name, ROOT / "tools" / (name + ".py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("dimension", ["users", "states", "categories"])
def test_state_cells_preserve_accepted_integer_width_and_large_identities(dimension):
    m = load("state_sufficiency")
    users = np.array([1, 2], dtype=np.int64)
    states = np.array([0, 0], dtype=np.int64)
    categories = np.array([0, 0], dtype=np.int64)
    if dimension == "users":
        users = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    elif dimension == "states":
        states = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    else:
        categories = categories.astype(np.uint64)
    rows = m.cell_totals(users, states, categories, [0, 1], [0.5, 0.5], 2)
    assert sum(r["users"] for r in rows) == 2
    assert sum(r["cases"] for r in rows) == 2
    assert sum(r["user_observed_sum"] for r in rows) == 1
    if dimension == "states":
        assert [r["state"] for r in rows] == [2**63, 2**63 + 1]


@pytest.mark.parametrize("group", ["top_label_calibration_bins", "class_calibration_bins"])
@pytest.mark.parametrize("field", ["predicted", "observed", "equal_user_weight"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_reported_calibration_is_rejected(group, field, value):
    u = np.array([1, 2])
    y = np.array([0, 1])
    p = np.array([[0.8, 0.2], [0.3, 0.7]])
    block = audit.categorical_audit_block(u, y, p)
    metrics = production.merge_categorical_blocks([production.categorical_block(u, y, p)])[
        "metrics"
    ]
    rows = metrics[group] if group == "top_label_calibration_bins" else metrics[group]["0"]
    row = next(r for r in rows if r["cases"])
    row[field] = value
    with pytest.raises(ValueError, match="Finite"):
        audit.verify_calibration(metrics, [block], 2)


def test_state_cell_structured_keys_preserve_recorded_signed_results():
    import json

    m = load("state_sufficiency")
    fixture = json.loads(
        (
            ROOT / "data/synthetic/fixtures/release/state-sufficiency-integer-identities-v1.json"
        ).read_text()
    )
    scope = {}
    exec(compile(fixture["source_module"], "frozen-project-tool", "exec"), scope)
    rng = np.random.default_rng(22)
    args = [
        rng.integers(0, 10, 100),
        rng.integers(0, 8, 100),
        rng.integers(0, 4, 100),
        rng.integers(0, 2, 100),
        rng.uniform(size=100),
        4,
    ]
    assert m.cell_totals(*args) == scope["cell_totals"](*args)


def test_calibration_none_mismatch_and_missing_support_are_rejected():
    u = np.array([1])
    y = np.array([0])
    p = np.array([[0.8, 0.2]])
    b = audit.categorical_audit_block(u, y, p)
    metrics = production.merge_categorical_blocks([production.categorical_block(u, y, p)])[
        "metrics"
    ]
    wrong = deepcopy(metrics)
    wrong["missing_cases"] = 1
    with pytest.raises(ValueError, match="Missing endpoint"):
        audit.verify_calibration(wrong, [b], 1)
    wrong = deepcopy(metrics)
    next(r for r in wrong["top_label_calibration_bins"] if r["cases"])["predicted"] = None
    with pytest.raises(ValueError, match="Empty calibration"):
        audit.verify_calibration(wrong, [b], 1)
