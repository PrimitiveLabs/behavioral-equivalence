"""Deterministic source-only archive with explicit static-data inclusion."""

from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("cloud", "configs", "docs", "formal", "spec", "src", "tests", "tools")
TOP_FILES = (
    ".gitignore",
    ".gitleaks.toml",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "NOTICE",
)
EXPLICIT_FIXTURES = (
    "reports/source-snapshots/estimate-calendar-parent-c0c628bb9aaf.py",
    "reports/preflight/kuairand-27k-inspection.json",
)
ALLOWED_SUFFIXES = {".py", ".lean", ".json", ".md", ".txt", ".toml", ".lock", ".yaml", ".yml"}


def collect_sources(root):
    root = Path(root).resolve()
    selected = set()

    def reject_symlink(path):
        for candidate in (path, *path.parents):
            if candidate == root:
                break
            if candidate.is_symlink():
                raise ValueError("Source archives cannot follow symlinks")

    for directory in SOURCE_ROOTS:
        reject_symlink(root / directory)
        for path in (root / directory).rglob("*"):
            reject_symlink(path)
            if (
                (
                    "__pycache__" in path.parts
                    or any(part.endswith(".egg-info") for part in path.parts)
                )
                or path.suffix in {".pyc", ".pyo"}
                or not path.is_file()
            ):
                continue
            if path.is_symlink():
                raise ValueError("Source archives cannot follow symlinks")
            if path.suffix not in ALLOWED_SUFFIXES:
                raise ValueError(
                    "Unexpected non-source file in source directory: " + str(path.relative_to(root))
                )
            selected.add(path.relative_to(root).as_posix())
    for pattern in [
        "data/synthetic/fixtures/**/*.json",
        "data/rights/*.json",
        "data/rights/*.txt",
        "data/license-register.json",
    ]:
        reject_symlink(root / pattern.split("*")[0].rstrip("/"))
        for path in root.glob(pattern):
            reject_symlink(path)
            if path.is_symlink():
                raise ValueError("Static fixture symlink forbidden")
            if path.is_file():
                selected.add(path.relative_to(root).as_posix())
    for name in (*TOP_FILES, *EXPLICIT_FIXTURES):
        path = root / name
        reject_symlink(path)
        if path.is_symlink():
            raise ValueError("Explicit source symlink forbidden")
        if path.is_file():
            selected.add(name)
    return sorted(selected)


def package(root):
    root = Path(root)
    files = collect_sources(root)
    members = {name: (root / name).read_bytes() for name in files}
    manifest = dict(
        schema_version="source-only-package-v1",
        scope="Source, tests, formal proofs, frozen protocols, static rights records and two explicit schema/source test fixtures. No human interactions, individual forecasts, model weights, secrets, caches or runtime ledgers. Scientific results and execution environments are separate artifacts.",
        files={name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()},
    )
    members["reports/source-manifest.json"] = json.dumps(manifest, indent=2).encode() + b"\n"
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, raw in sorted(members.items()):
                entry = tarfile.TarInfo(name)
                entry.size = len(raw)
                entry.mode = 0o644
                entry.mtime = 0
                tar.addfile(entry, io.BytesIO(raw))
    return output.getvalue(), manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    raw, manifest = package(args.root)
    digest = hashlib.sha256(raw).hexdigest()
    destination = args.root / "reports/source-snapshots" / ("source-" + digest + ".tar.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    pointer = args.root / "reports/current-source-package.json"
    pointer.write_text(
        json.dumps(
            dict(
                sha256=digest,
                path=str(destination.relative_to(args.root)),
                source_files=len(manifest["files"]),
                bytes=len(raw),
            ),
            indent=2,
        )
        + "\n"
    )
    print(pointer.read_text())


if __name__ == "__main__":
    main()
