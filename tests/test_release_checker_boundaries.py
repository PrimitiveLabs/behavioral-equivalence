"""Release-checker failure gates with real temporary archives, no provider calls."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from test_cloud_summary_boundaries import body


def archive(member="input.txt", *, kind=tarfile.REGTYPE, wrong_manifest=False):
    raw = b"synthetic input"
    manifest = {
        "files": {"input.txt": "0" * 64 if wrong_manifest else hashlib.sha256(raw).hexdigest()}
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as target:
        for name, content in [
            (member, raw),
            ("reports/source-manifest.json", json.dumps(manifest).encode()),
        ]:
            info = tarfile.TarInfo(name)
            info.type = kind if name == member else tarfile.REGTYPE
            info.size = len(content) if info.isfile() else 0
            target.addfile(info, io.BytesIO(content) if info.isfile() else None)
    return stream.getvalue()


def checker(tmp_path):
    calls = []

    def mapped_path(value):
        return tmp_path / "artifacts" if str(value) == "/artifacts" else Path(value)

    (tmp_path / "artifacts").mkdir()
    function = body(
        "reproduce_source_package.py",
        "reproduce",
        Path=mapped_path,
        volume=SimpleNamespace(commit=lambda: calls.append("commit")),
    )
    return function, calls


def test_archive_identity_rejected_before_any_subprocess(tmp_path, monkeypatch):
    function, calls = checker(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected subprocess"))
    with pytest.raises(ValueError, match="Source archive identity"):
        function(b"not an archive", "0" * 64)
    assert calls == []


@pytest.mark.parametrize(
    "member,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/escape", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("directory", tarfile.DIRTYPE),
    ],
)
def test_unsafe_archive_members_rejected(tmp_path, member, kind):
    function, calls = checker(tmp_path)
    raw = archive(member, kind=kind)
    with pytest.raises(ValueError, match="Unsafe non-file"):
        function(raw, hashlib.sha256(raw).hexdigest())
    assert calls == []


def test_extracted_file_hash_mismatch_rejected(tmp_path):
    function, calls = checker(tmp_path)
    raw = archive(wrong_manifest=True)
    with pytest.raises(ValueError, match="Extracted source hash"):
        function(raw, hashlib.sha256(raw).hexdigest())
    assert calls == []


@pytest.mark.parametrize(
    "failure,stage",
    [
        ("build", "wheel_build"),
        ("no_wheel", "wheel_build"),
        ("two_wheels", "wheel_build"),
        ("venv", "installed_wheel_tests"),
        ("install", "installed_wheel_tests"),
        ("import", "installed_wheel_tests"),
        ("tests", "installed_wheel_tests"),
        ("skips", "installed_wheel_tests"),
        ("scientific", "scientific_reproduction"),
        (None, "installed_wheel_tests"),
    ],
)
def test_release_gates_and_retained_report(tmp_path, monkeypatch, failure, stage):
    import importlib.metadata

    function, commits = checker(tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["PYTHONPATH"] == ""
        output, code = "", 0
        if command[1:4] == ["-m", "pip", "wheel"]:
            if failure == "build":
                code = 1
            elif failure != "no_wheel":
                (Path(command[-1]) / "example.whl").write_bytes(b"synthetic wheel")
                if failure == "two_wheels":
                    (Path(command[-1]) / "other.whl").write_bytes(b"other synthetic wheel")
        elif command[1:3] == ["-m", "venv"]:
            code = int(failure == "venv")
        elif command[1:4] == ["-m", "pip", "install"]:
            code = int(failure == "install")
        elif command[1] == "-c":
            output = (
                "/source/src/behavioral_equivalence/__init__.py"
                if failure == "import"
                else str(Path(command[0]).parents[1] / "lib/behavioral_equivalence/__init__.py")
            )
        elif command[1:3] == ["-m", "pytest"]:
            assert "-W" in command and "error" in command and "pythonpath=" in command
            code = int(failure == "tests")
            output = "3 passed, 1 skipped" if failure == "skips" else "3 passed"
        elif command[1] == "tools/scientific_reproduction.py":
            code = int(failure == "scientific")
            if not code:
                folder = Path(command[-1])
                folder.mkdir()
                (folder / "report.json").write_text(
                    json.dumps({"status": "complete", "worlds": 10})
                )
        else:
            pytest.fail("unexpected subprocess shape")
        return SimpleNamespace(returncode=code, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-version")
    raw = archive()
    digest = hashlib.sha256(raw).hexdigest()
    result = function(raw, digest)
    assert result["status"] == ("complete" if failure is None else "failed")
    assert result["stage"] == stage
    assert result["source_hashes_verified"] is True
    assert result["source_bundle_sha256"] == digest
    assert commits == ["commit"]
    saved = tmp_path / "artifacts" / ("source-package-reproduction-" + digest[:16]) / "report.json"
    assert json.loads(saved.read_text()) == result
    scientific_called = any(c[1] == "tools/scientific_reproduction.py" for c in commands)
    assert scientific_called == (failure in {None, "scientific"})
    if failure == "skips":
        assert result["tests_skipped"] == 1
    if failure is None:
        assert result["tests_skipped"] == 0
        assert result["scientific_reproduction"]["worlds"] == 10
