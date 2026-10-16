"""Archive-pointer, explicit-fixture and local guard summary regressions."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import pytest

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        "release_boundary_" + name, ROOT / "tools" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packager_actual_cli_binds_pointer_to_deterministic_archive(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/example.py").write_text("value = 1\n")
    (tmp_path / "README.md").write_text("Project source fixture\n")
    command = [sys.executable, str(ROOT / "tools/package_source.py"), "--root", str(tmp_path)]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    pointer = json.loads(first.stdout)
    assert pointer == json.loads((tmp_path / "reports/current-source-package.json").read_text())
    archive = tmp_path / pointer["path"]
    raw = archive.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pointer["sha256"]
    assert pointer["bytes"] == len(raw) and pointer["source_files"] == 2
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        manifest = json.loads(tar.extractfile("reports/source-manifest.json").read())
        assert len(manifest["files"]) == 2
        assert len(tar.getmembers()) == 3
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest
    again = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(again.stdout) == pointer
    assert archive.read_bytes() == raw
    assert len(list((tmp_path / "reports/source-snapshots").glob("*.tar.gz"))) == 1


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "LICENSE",
        "reports/preflight/kuairand-27k-inspection.json",
        "reports/source-snapshots/estimate-calendar-parent-c0c628bb9aaf.py",
    ],
)
def test_explicit_top_files_and_test_fixtures_cannot_be_symlinks(tmp_path, name):
    outside = tmp_path / "private-content"
    outside.write_text("must not enter archive")
    link = tmp_path / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        load("package_source").package(tmp_path)


def test_cache_and_generated_package_files_never_enter_source_archive(tmp_path):
    for name in [
        "src/__pycache__/cached.pyc",
        "src/module.pyc",
        "src/module.pyo",
        "src/example.egg-info/PKG-INFO",
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"generated cache")
    raw, manifest = load("package_source").package(tmp_path)
    assert manifest["files"] == {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        assert tar.getnames() == ["reports/source-manifest.json"]


@pytest.mark.parametrize("status", [None, 3, {"private": "payload"}])
def test_guard_nonscalar_status_never_persists_provider_payload(tmp_path, status):
    (tmp_path / "cloud").mkdir()
    (tmp_path / "cloud/frozen.py").write_text("# source fixture\n")
    (tmp_path / ".campaign").mkdir()
    (tmp_path / ".campaign/original.json").write_text('{"function_call_id":"fc-original"}')
    module = load("guarded_modal")
    report = module.run_guarded(
        tmp_path,
        "cloud/frozen.py",
        ".campaign/original.json",
        observer=lambda identity, wait: {
            "status": status,
            "private_result": "not-a-release-artifact",
        },
    )
    assert report["function_call_id"] == "fc-original"
    assert report["provider_result_status"] == "available"
    assert report["scientific_acceptance"] == "not_assessed"
    assert not report["launched_cli"]
    assert "private" not in json.dumps(report)
