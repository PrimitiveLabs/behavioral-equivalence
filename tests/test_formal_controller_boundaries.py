"""Controller guard logic with simulated compiler results, never a proof check."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_cloud_summary_boundaries import body

CONTROLLERS = ["check_formal_bounds.py", "check_finite_coupling.py", "check_weighted_events.py"]


def configure(
    tmp_path,
    monkeypatch,
    controller,
    *,
    proof="example : True := by trivial",
    commit="pinned",
    positive=0,
    negative=1,
    output="",
):
    import subprocess

    calls = []
    saved = {
        "proof.lean": proof,
        "config.json": json.dumps(
            dict(
                mathlib_commit="pinned",
                theorems=["fixture"],
                scope="Simulated compiler guard test, not Lean execution",
            )
        ),
    }

    def path(value):
        text = str(value)
        return (
            tmp_path / text.lstrip("/") if text.startswith(("/artifacts", "/tmp/")) else Path(value)
        )

    (tmp_path / "artifacts").mkdir(exist_ok=True)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kwargs: commit if command[0] == "git" else "fixture-version",
    )

    def run(command, **kwargs):
        is_negative = "invalid-coupling-control" in command[-1]
        calls.append("negative" if is_negative else "positive")
        return SimpleNamespace(
            returncode=negative if is_negative else positive,
            stdout="" if is_negative else output,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    volume = SimpleNamespace(commit=lambda: calls.append("commit"))
    return body(
        controller, "check", Path=path, snapshot=lambda: (saved, "a" * 64), volume=volume
    ), calls


@pytest.mark.parametrize("controller", CONTROLLERS)
@pytest.mark.parametrize("declaration", ["sorry", "admit", "axiom"])
def test_proof_controller_rejects_admitted_declarations_before_compiler(
    tmp_path, monkeypatch, controller, declaration
):
    run, calls = configure(tmp_path, monkeypatch, controller, proof=declaration)
    with pytest.raises(ValueError, match="forbidden"):
        run()
    assert not calls


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_proof_controller_requires_exact_mathlib_revision(tmp_path, monkeypatch, controller):
    run, calls = configure(tmp_path, monkeypatch, controller, commit="other")
    with pytest.raises(ValueError, match="checkout changed"):
        run()
    assert not calls


@pytest.mark.parametrize("controller", CONTROLLERS)
@pytest.mark.parametrize("failure", ["compiler", "sorry_ax"])
def test_failed_or_axiom_using_compiler_output_cannot_complete(
    tmp_path, monkeypatch, controller, failure
):
    run, calls = configure(
        tmp_path,
        monkeypatch,
        controller,
        positive=1 if failure == "compiler" else 0,
        output="sorryAx" if failure == "sorry_ax" else "",
    )
    r = run()
    assert r["status"] == "failed"
    assert calls[-1] == "commit"
    assert json.loads((Path(r["root"]) / "report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("controller", CONTROLLERS[1:])
def test_negative_control_must_be_rejected_by_compiler(tmp_path, monkeypatch, controller):
    run, calls = configure(tmp_path, monkeypatch, controller, negative=0)
    assert run()["status"] == "failed"
    assert calls == ["positive", "negative", "commit"]
