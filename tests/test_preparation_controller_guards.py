"""Preparation source freezes, bounded bucket namespaces and cached output hashes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_cloud_summary_boundaries import body


@pytest.mark.parametrize(
    "controller,function",
    [
        ("prepare_27k.py", "fingerprint_bucket"),
        ("prepare_27k.py", "prepare_bucket"),
        ("calendar_support_27k.py", "inspect_bucket"),
        ("confirmatory_cases_27k.py", "prepare_bucket"),
    ],
)
@pytest.mark.parametrize("bucket", [-1, 32])
def test_preparation_bucket_namespace_bounds(controller, function, bucket):
    with pytest.raises(ValueError):
        body(controller, function, BUCKETS=32)(bucket)


@pytest.mark.parametrize("changed", [False, True])
def test_full_preparation_refuses_different_frozen_source(tmp_path, changed):
    saved = {"sha256": "original", "files": {"fixture.py": "value = 1\n"}}
    (tmp_path / "source.json").write_text(json.dumps(saved))
    actual = dict(saved, sha256="changed") if changed else saved
    run = body(
        "prepare_27k.py",
        "frozen_source",
        paths=lambda: (tmp_path, tmp_path),
        source_snapshot=lambda: actual,
    )
    if changed:
        with pytest.raises(ValueError, match="source changed"):
            run()
    else:
        assert run() == saved


@pytest.mark.parametrize("damage", ["none", "source", "output"])
def test_case_preparation_cache_requires_current_source_and_output_bytes(tmp_path, damage):
    data = tmp_path / "data/bucket-00"
    data.mkdir(parents=True)
    for name in ["interactions.parquet", "timestamp-bundles.parquet"]:
        (data / name).write_bytes(b"fixture")

    def checksum(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    r = dict(
        interactions_sha256=checksum(data / "interactions.parquet"),
        bundles_sha256=checksum(data / "timestamp-bundles.parquet"),
    )
    (data / "report.json").write_text(json.dumps(r))
    digest = "a" * 64
    out = tmp_path / ("confirmatory-cases-" + digest[:16]) / "bucket-00"
    out.mkdir(parents=True)
    (out / "fixture.npz").write_bytes(b"fixture")
    cached = dict(
        source_digest=digest,
        input_sha256={
            n: checksum(data / n) for n in ["interactions.parquet", "timestamp-bundles.parquet"]
        },
        output_sha256={"fixture.npz": checksum(out / "fixture.npz")},
    )
    if damage == "source":
        cached["source_digest"] = "other"
    if damage == "output":
        (out / "fixture.npz").write_bytes(b"changed")
    (out / "report.json").write_text(json.dumps(cached))
    config = tmp_path / "case-config.json"
    config.write_text("{}")

    def path(value):
        return config if str(value) == "/research/case-config.json" else Path(value)

    calls = []
    run = body(
        "confirmatory_cases_27k.py",
        "prepare_bucket",
        DATA=tmp_path,
        Path=path,
        snapshot=lambda: ({}, digest),
        sha256=checksum,
        volume=SimpleNamespace(commit=lambda: calls.append("commit")),
    )
    if damage == "none":
        assert run(0) == cached
    else:
        with pytest.raises(ValueError):
            run(0)
    assert not calls
