"""Supplementary controller artifact gates and zero-skip validation statuses."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_cloud_summary_boundaries import body


@pytest.mark.parametrize(
    "code,stdout,expected",
    [(0, "8 passed", "complete"), (0, "7 passed, 1 skipped", "failed"), (1, "1 failed", "failed")],
)
def test_supplementary_validator_requires_success_without_skips(
    tmp_path, monkeypatch, code, stdout, expected
):
    import subprocess

    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=code, stdout=stdout, stderr=""),
    )
    (tmp_path / "artifacts").mkdir()
    run = body(
        "validate_supplementary.py",
        "validate",
        Path=lambda value: tmp_path / "artifacts" if str(value) == "/artifacts" else Path(value),
        source_identity=lambda: ({"fixture": "source"}, "a" * 64),
        volume=SimpleNamespace(commit=lambda: calls.append("commit")),
    )
    r = run()
    assert r["status"] == expected and r["returncode"] == code
    assert calls == ["commit"]
    assert (
        json.loads(
            (
                tmp_path / "artifacts" / ("supplementary-validation-" + "a" * 16) / "report.json"
            ).read_text()
        )["status"]
        == expected
    )


@pytest.mark.parametrize(
    "controller", ["attach_supplementary_outcomes.py", "attach_supplementary_calibration.py"]
)
@pytest.mark.parametrize("changed", ["prepared", "cases"])
def test_target_attachment_rejects_wrong_manifest_before_reading_arrays(
    tmp_path, monkeypatch, controller, changed
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    calls = []

    def checksum(path):
        return (
            ("changed" if changed == "prepared" else "prepared")
            if str(path).endswith("prepared/report.json")
            else ("changed" if changed == "cases" else "cases")
        )

    run = body(
        controller,
        "attach",
        PREPARED="prepared",
        CASE_ROOT="cases",
        PREPARED_REPORT_SHA="prepared",
        CASE_REPORT_SHA="cases",
        checksum=checksum,
        source_identity=lambda: ({}, "a" * 64),
        volume=SimpleNamespace(
            reload=lambda: calls.append("reload"), commit=lambda: calls.append("commit")
        ),
    )
    with pytest.raises(ValueError, match="manifest changed"):
        run()
    assert calls == ["reload"]


def test_validation_timeout_does_not_persist_success_or_commit(tmp_path, monkeypatch):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("fixture", 240)

    monkeypatch.setattr(subprocess, "run", timeout)
    calls = []
    run = body(
        "validate_supplementary.py",
        "validate",
        source_identity=lambda: ({}, "a" * 64),
        volume=SimpleNamespace(commit=lambda: calls.append("commit")),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run()
    assert not calls
