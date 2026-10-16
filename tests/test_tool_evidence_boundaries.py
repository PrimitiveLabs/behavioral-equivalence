"""Read-only supplementary report validation and small vertex-oracle boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def load(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("review_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(module):
    return dict(
        status="complete",
        source_digest=module.DIGEST,
        protocol="known_users",
        architecture="gru",
        seed=17,
        records=[
            dict(
                endpoint=n,
                status="complete",
                selected_C=0.1,
                candidates=[dict(C=c, validation={"nll": 0.5}) for c in [0.1, 1.0, 10.0]],
            )
            for n in [
                "is_like",
                "is_follow",
                "is_comment",
                "is_forward",
                "is_hate",
                "watch_category",
            ]
        ],
    )


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "source",
        "protocol",
        "architecture",
        "seed",
        "endpoint",
        "grid",
        "selected",
        "row_status",
    ],
)
def test_supplementary_fit_report_rejects_wrong_identity_or_selection(monkeypatch, damage):
    m = load("run_supplementary_fits", monkeypatch)
    r = report(m)
    if damage in {"status", "protocol", "architecture", "seed"}:
        r[damage] = "other"
    elif damage == "source":
        r["source_digest"] = "other"
    elif damage == "endpoint":
        r["records"].pop()
    elif damage == "grid":
        r["records"][0]["candidates"].pop()
    elif damage == "selected":
        r["records"][0]["selected_C"] = 10.0
    else:
        r["records"][0]["status"] = "pending"
    with pytest.raises(ValueError):
        m.validate_report(r, "known_users", "gru", 17)


def test_supplementary_fit_rule_ties_missing_metrics_and_unavailable_endpoint(monkeypatch):
    m = load("run_supplementary_fits", monkeypatch)
    r = report(m)
    m.validate_report(r, "known_users", "gru", 17)
    r["records"][0]["candidates"][0]["validation"]["nll"] = None
    r["records"][0]["selected_C"] = 1.0
    r["records"][1] = dict(endpoint="is_follow", status="unavailable_no_observed_training_labels")
    m.validate_report(r, "known_users", "gru", 17)


@pytest.mark.parametrize("horizon", [-1, True, 1.5])
def test_vertex_oracle_rejects_invalid_horizon(monkeypatch, horizon):
    m = load("interval_vertices", monkeypatch)
    with pytest.raises(ValueError, match="horizon"):
        m.vertex_reachability(None, "complete", horizon, True)


@pytest.mark.parametrize(
    "lo,hi", [([], []), ([[0]], [[1]]), ([0], [1, 1]), ([-0.1], [1]), ([0], [np.inf])]
)
def test_vertex_oracle_rejects_malformed_coordinate_domains(monkeypatch, lo, hi):
    m = load("interval_vertices", monkeypatch)
    with pytest.raises(ValueError):
        m.row_vertices(lo, hi)
