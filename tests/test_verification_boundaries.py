"""Real subprocess and filesystem checks for verification boundaries."""

import gzip
import hashlib
import json
import subprocess
import sys
import numpy as np
import pytest
from behavioral_equivalence.core import Chain, quotient
from behavioral_equivalence.prism import check, write_model
from behavioral_equivalence.certificate_audit import audit_world
from behavioral_equivalence.evidence_registry import (
    EvidenceEntry,
    EvidenceRegistry,
    verify_registry,
)


@pytest.mark.parametrize("extremum,operator", [(None, "P"), ("min", "Pmin"), ("max", "Pmax")])
def test_adapter_real_process_receives_literal_property_and_retains_both_streams(
    tmp_path, extremum, operator
):
    script, log = tmp_path / "solver stand in.py", tmp_path / "solver.log"
    script.write_text(
        'import sys,json\nprint(json.dumps(sys.argv[1:]))\nprint("Result: 2.5e-1")\nprint("diagnostic",file=sys.stderr)\n'
    )
    assert check(script, "complete", 10, log, executable=sys.executable, extremum=extremum) == 0.25
    text = log.read_text()
    argv = json.loads(text.splitlines()[0])
    assert argv == ["-pf", f'{operator}=? [ F<=10 "complete" ]', "-explicit"]
    assert "diagnostic" in text


@pytest.mark.parametrize(
    "exit_code,output,message",
    [
        (3, "Result: 0.5", "failed with code 3"),
        (0, "no result", "no scalar result"),
        (0, "Result: NaN", "no scalar result"),
    ],
)
def test_adapter_failed_or_missing_result_retains_logs(tmp_path, exit_code, output, message):
    script, log = tmp_path / "solver.py", tmp_path / "solver.log"
    script.write_text(
        f'import sys\nprint({output!r})\nprint("failure detail",file=sys.stderr)\nsys.exit({exit_code})\n'
    )
    with pytest.raises(RuntimeError, match=message):
        check(script, "complete", 1, log, executable=sys.executable)
    assert output in log.read_text() and "failure detail" in log.read_text()


@pytest.mark.parametrize(
    "target,horizon,extremum",
    [
        ('bad"target', 1, None),
        ("bad target", 1, None),
        ("complete", -1, None),
        ("complete", 1, "unsupported"),
    ],
)
def test_invalid_property_rejected_before_launch(tmp_path, target, horizon, extremum):
    with pytest.raises(ValueError):
        check(
            tmp_path / "missing",
            target,
            horizon,
            tmp_path / "log",
            executable="/nonexistent-prism-fixture",
            extremum=extremum,
        )
    assert not (tmp_path / "log").exists()


def test_timeout_propagates_with_original_command_and_no_fake_result(tmp_path, monkeypatch):
    def expired(command, **kwargs):
        assert kwargs == dict(capture_output=True, text=True, timeout=120)
        raise subprocess.TimeoutExpired(command, 120)

    monkeypatch.setattr("behavioral_equivalence.prism.subprocess.run", expired)
    with pytest.raises(subprocess.TimeoutExpired) as error:
        check(tmp_path / "model", "complete", 1, tmp_path / "log")
    assert error.value.timeout == 120
    assert not (tmp_path / "log").exists()


def test_missing_solver_is_not_a_verified_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        check(
            tmp_path / "model",
            "complete",
            1,
            tmp_path / "log",
            executable=str(tmp_path / "missing executable"),
        )


def test_export_explicit_initial_state_and_no_output_for_bad_label(tmp_path):
    chain = Chain(np.eye(2), ("active", "complete"), np.array([1, 0]))
    path = tmp_path / "model.prism"
    write_model(chain, path, 1)
    assert "init 1;" in path.read_text() and "0:(s'=" not in path.read_text()
    with pytest.raises(ValueError, match="out of bounds"):
        write_model(chain, tmp_path / "negative.prism", -1)
    invalid = Chain(np.eye(2), ("active", "bad\nlabel"), chain.initial)
    with pytest.raises(ValueError, match="simple identifiers"):
        write_model(invalid, tmp_path / "injected.prism")
    assert not (tmp_path / "injected.prism").exists()


def saved_world(source, budget):
    source.mkdir()
    concrete = Chain(
        np.array([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5], [0, 0, 1.0]]),
        ("active", "active", "complete"),
        np.array([1.0, 0, 0]),
    )
    blocks = ((0, 1), (2,))
    abstract, _ = quotient(concrete, blocks)
    parameters = {} if budget is None else {"epsilon_budget": budget}
    report = dict(
        status="complete",
        world_digest=concrete.digest(),
        protocol_digest="frozen-protocol-fixture",
        source_digest="frozen-source-fixture",
        spec={"size": 3},
        methods=[dict(method="test", parameters=parameters, partition=blocks)],
    )
    (source / "report.json").write_text(json.dumps(report))
    np.savez_compressed(
        source / "world.npz",
        p=concrete.transitions,
        labels=concrete.labels,
        initial=concrete.initial,
    )
    write_model(abstract, source / "model-0.prism")
    return report


@pytest.mark.parametrize(
    "budget,strict,excess", [(None, None, None), (0, False, 0.125), (0.125, True, 0)]
)
def test_audit_budget_reporting_is_separate_from_certificate_validity(
    tmp_path, budget, strict, excess
):
    source, out = tmp_path / "world", tmp_path / "audit"
    report = saved_world(source, budget)
    result = audit_world(source, out, [0, 1])
    row = result["methods"][0]
    assert result["status"] == "complete" and row["verified"]
    assert row["budget_strictly_satisfied"] is strict
    assert row["exact_budget_excess_for_display"] == excess
    packed = (out / row["certificate"]).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == row["certificate_sha256"]
    assert json.loads(gzip.decompress(packed))["concrete_digest"] == report["world_digest"]
    assert (
        result["source_report_sha256"]
        == hashlib.sha256((source / "report.json").read_bytes()).hexdigest()
    )


def test_audit_rejects_unfinished_world_before_artifact_access(tmp_path):
    source = tmp_path / "world"
    source.mkdir()
    (source / "report.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="completed synthetic worlds"):
        audit_world(source, tmp_path / "audit", [1])
    assert not (tmp_path / "audit").exists()


def evidence(identity, parents=()):
    return EvidenceEntry(
        evidence_id=identity,
        stage="release_test_reproduction",
        domain="release",
        artifact=dict(
            uri="report.json",
            sha256=hashlib.sha256(b"{}").hexdigest(),
            bytes=2,
            media_type="application/json",
        ),
        scope="Identity only",
        parents=parents,
    )


def test_registry_shared_ancestry_and_empty_inventory_never_assert_science(tmp_path):
    (tmp_path / "report.json").write_bytes(b"{}")
    entries = (
        evidence("root"),
        evidence("left", ("root",)),
        evidence("right", ("root",)),
        evidence("child", ("left", "right")),
    )
    result = verify_registry(EvidenceRegistry(entries=entries), tmp_path)
    assert result["artifacts"] == 4 and result["scientific_completion"] == "not_assessed"
    empty = verify_registry(EvidenceRegistry(entries=()), tmp_path)
    assert empty["artifacts"] == 0 and empty["scientific_completion"] == "not_assessed"
    with pytest.raises(ValueError, match="Cyclic"):
        EvidenceRegistry(entries=(evidence("a", ("b",)), evidence("b", ("a",))))
