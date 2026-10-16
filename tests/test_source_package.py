import importlib.util
from pathlib import Path
import io
import json
import tarfile
import pytest

spec = importlib.util.spec_from_file_location(
    "source_packager", Path(__file__).parents[1] / "tools/package_source.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_archive_is_deterministic_includes_formal_and_excludes_human_rows_and_weights(tmp_path):
    for name, contents in [
        ("src/example.py", "print(1)"),
        ("formal/Bounds.lean", "example : True := by trivial"),
        ("data/synthetic/fixtures/a.json", "{}"),
        ("data/private/rows.parquet", "private"),
        ("reports/private/model.pt", "weights"),
        ("src/example.egg-info/PKG-INFO", "generated metadata"),
        (".env", "secret"),
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)
    first, manifest = module.package(tmp_path)
    second, _ = module.package(tmp_path)
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as tar:
        names = tar.getnames()
        assert names == sorted(names)
        assert set(names) == {
            "src/example.py",
            "formal/Bounds.lean",
            "data/synthetic/fixtures/a.json",
            "reports/source-manifest.json",
        }
        assert json.load(tar.extractfile("reports/source-manifest.json")) == manifest
        assert all(m.mtime == 0 for m in tar.getmembers())


def test_symlink_and_unexpected_binary_in_source_directory_rejected(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "secret").write_text("secret")
    p = tmp_path / "src/leak.py"
    p.symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="symlink"):
        module.collect_sources(tmp_path)
    p.unlink()
    (tmp_path / "src/weights.npz").write_bytes(b"private")
    with pytest.raises(ValueError, match="non-source"):
        module.collect_sources(tmp_path)


@pytest.mark.parametrize("name", ["src", "src/nested", "data/rights", "data/synthetic/fixtures"])
def test_directory_symlink_is_rejected(tmp_path, name):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("private")
    link = tmp_path / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.package(tmp_path)


def test_dangling_source_symlink_is_rejected(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/dangling.py").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlink"):
        module.package(tmp_path)


def test_manifest_matches_exact_archived_content(tmp_path, monkeypatch):
    import hashlib

    (tmp_path / "src").mkdir()
    source = tmp_path / "src/example.py"
    source.write_text("original")
    original_read = Path.read_bytes
    reads = []

    def changing_read(path):
        raw = original_read(path)
        if path == source:
            reads.append(raw)
            source.write_text("changed")
        return raw

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    raw, manifest = module.package(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        content = archive.extractfile("src/example.py").read()
    assert reads == [b"original"]
    assert manifest["files"]["src/example.py"] == hashlib.sha256(content).hexdigest()
