"""Preserve fixed-study thresholds and exercise the actual summary entrypoint."""

import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from behavioral_equivalence.human_comparison import HumanPrimaryResult, decide_primary
from behavioral_equivalence.human_protocol import (
    identity_fingerprints,
    confirmatory_cohort,
    validate_27k_logs,
    SOURCE_COLUMNS,
)


def summary_module():
    spec = importlib.util.spec_from_file_location(
        "human_summary_boundary", Path(__file__).parents[1] / "tools/summarize_human_primary.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(protocol, states=128):
    return HumanPrimaryResult(
        protocol=protocol,
        seed=1,
        phase="confirmatory",
        parent_latent_states=256,
        reduced_latent_states=states,
        candidate_minus_reference_nats=0,
        ci95=(-0.001, 0.009999),
        users=10,
        cases=100,
        report_sha256="a" * 64,
    )


def test_adjacent_compression_boundary_and_empty_fixed_matrix():
    good = (result("known_users"), result("unseen_users"))
    assert decide_primary(good, (1,))["scientific_outcome"] == "supported"
    bad = (good[0], result("unseen_users", 129))
    rejected = decide_primary(bad, (1,))
    assert rejected["scientific_outcome"] == "unsupported"
    assert rejected["observed_compression_failures"] == 1
    assert rejected["observed_noninferiority_failures"] == 0
    empty = decide_primary((), (1,))
    assert empty["status"] == "incomplete" and empty["scientific_outcome"] == "not_evaluated"
    assert len(empty["missing"]) == 2


@pytest.mark.parametrize(
    "seeds,margin", [((1.5,), 0.01), ((1,), np.nan), ((1,), np.inf), ((1,), -0.1)]
)
def test_primary_declared_seed_and_margin_boundaries(seeds, margin):
    with pytest.raises(ValueError, match="Distinct integer seeds"):
        decide_primary((), seeds, margin)


@pytest.mark.parametrize(
    "users,times,tabs",
    [
        (np.array([[1]]), np.array([1]), np.array([1])),
        (np.array([1]), np.array([1, 2]), np.array([1])),
        (np.array([1]), np.array([1]), np.array([1.0])),
    ],
)
def test_fingerprints_require_aligned_integer_identity_columns(users, times, tabs):
    with pytest.raises(ValueError, match="Equal-length integer"):
        identity_fingerprints(users, times, tabs)


@pytest.mark.parametrize(
    "mutation", ["empty_pilot", "empty_final", "duplicate_pilot", "duplicate_final_user"]
)
def test_cohort_identity_boundaries(mutation):
    pilot = [dict(user_id=100, rows=3, sha256="identity-a")]
    final = [
        dict(user_id=1, rows=3, sha256="identity-a"),
        dict(user_id=2, rows=4, sha256="identity-b"),
    ]
    if mutation == "empty_pilot":
        pilot = []
    elif mutation == "empty_final":
        final = []
    elif mutation == "duplicate_pilot":
        pilot *= 2
    else:
        final[1]["user_id"] = 1
    with pytest.raises(ValueError):
        confirmatory_cohort(pilot, final, 1)


def archive_entries():
    path = Path(__file__).parents[1] / "reports/preflight/kuairand-27k-inspection.json"
    return json.loads(path.read_text())["entries"]


def test_archive_rejects_duplicate_shards_and_columns_but_not_column_order():
    entries = archive_entries()
    names = validate_27k_logs(entries)
    logs = [e for e in entries if e["name"] in names]
    with pytest.raises(ValueError, match="five"):
        validate_27k_logs([logs[0], logs[0], *logs[2:]])
    for columns in [
        list(SOURCE_COLUMNS) + [SOURCE_COLUMNS[0]],
        [SOURCE_COLUMNS[0]] * len(SOURCE_COLUMNS),
    ]:
        changed = [dict(e, columns=columns) if e["name"] == names[0] else e for e in entries]
        with pytest.raises(ValueError, match="source-column"):
            validate_27k_logs(changed)
    reordered = [
        dict(e, columns=list(reversed(e["columns"]))) if e["name"] in names else e for e in entries
    ]
    assert validate_27k_logs(reordered) == names


@pytest.fixture
def study(tmp_path, monkeypatch):
    module = summary_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    reports = tmp_path / "reports"
    final = reports / "final-human"
    preflight = reports / "preflight"
    final.mkdir(parents=True)
    preflight.mkdir()
    frozen = dict(
        controllers={
            p: dict(source_digest="source-" + p)
            for p in ["one_step", "response_all_groups", "response_heldout_groups"]
        }
    )
    (reports / "final-analysis-freeze.json").write_text(json.dumps(frozen))
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    for seed in range(20261015, 20261020):
        for protocol in ["known_users", "unseen_users"]:
            nested = preflight / f"human-nested-{protocol}-{seed}.json"
            nested.write_text(
                json.dumps(dict(records=[dict(name="strict", reduced_latent_states=256)]))
            )
            row = dict(
                status="complete",
                analysis_freeze_digest=digest,
                source_digest="source-one_step",
                feedback_embargo_hours=24,
                primary="nested.strict",
                protocol=protocol,
                seed=seed,
                nested_report_sha256=hashlib.sha256(nested.read_bytes()).hexdigest(),
                comparisons={
                    "all": {
                        "nested.strict-vs-parent": dict(
                            candidate_minus_reference_nats=0, ci95=[0, 0]
                        )
                    }
                },
                metrics={"all/parent_k256": dict(users=10, cases=100)},
            )
            (final / f"primary-{protocol}-{seed}.json").write_text(json.dumps(row))
        for phase in ["response_all_groups", "response_heldout_groups"]:
            interval = (
                [-0.01, -0.001] if phase == "response_all_groups" and seed < 20261019 else [0, 0]
            )
            row = dict(
                status="complete",
                analysis_freeze_digest=digest,
                source_digest="source-" + phase,
                phase=phase,
                seed=seed,
                primary_candidate="targeted",
                users=10,
                cases=100,
                comparisons={
                    "aggregate": {
                        "targeted-vs-reference": dict(
                            candidate_minus_reference_nats=0, ci95=interval
                        )
                    }
                },
            )
            (final / f"{phase}-{seed}.json").write_text(json.dumps(row))
    return module, reports


def test_complete_summarizer_retains_negative_compression_and_response_decisions(study):
    module, reports = study
    module.main()
    result = json.loads((reports / "final-human-primary-decisions.json").read_text())
    assert result["one_step"]["status"] == "complete"
    assert result["one_step"]["scientific_outcome"] == "unsupported"
    assert result["one_step"]["observed_compression_failures"] == 10
    assert result["one_step"]["observed_noninferiority_failures"] == 0
    assert all(
        r["status"] == "complete" and r["scientific_outcome"] == "unsupported"
        for r in result["responses"].values()
    )
    assert all(len(r["rows"]) == 5 for r in result["responses"].values())


@pytest.mark.parametrize("mutation", ["freeze", "one_step_source", "response_source", "partition"])
def test_summarizer_rejects_identity_changes_before_publishing_decision(study, mutation):
    module, reports = study
    primary = next((reports / "final-human").glob("primary-*.json"))
    target = (
        next((reports / "final-human").glob("response_all_groups-*.json"))
        if mutation == "response_source"
        else primary
    )
    row = json.loads(target.read_text())
    if mutation == "partition":
        path = reports / "preflight" / f"human-nested-{row['protocol']}-{row['seed']}.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        row["analysis_freeze_digest" if mutation == "freeze" else "source_digest"] = "changed"
        target.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        module.main()
    assert not (reports / "final-human-primary-decisions.json").exists()


def test_summarizer_missing_fit_is_incomplete_not_positive(study):
    module, reports = study
    next((reports / "final-human").glob("primary-*.json")).unlink()
    module.main()
    result = json.loads((reports / "final-human-primary-decisions.json").read_text())
    assert result["one_step"]["status"] == "incomplete"
    assert result["one_step"]["scientific_outcome"] == "not_evaluated"
