"""Case audit input manifests reject changed bytes, sources and incomplete buckets."""

import hashlib
import json
from pathlib import Path

import pytest

from test_cloud_summary_boundaries import body


@pytest.mark.parametrize(
    "controller",
    ["audit_adjacent_calibration.py", "audit_multistep_test.py", "audit_multistep_fit.py"],
)
@pytest.mark.parametrize("damage", ["none", "bytes", "source", "status", "buckets"])
def test_audit_input_bindings(tmp_path, controller, damage):
    original = {"config.json": json.dumps({"fixture": "configuration"})}
    source_digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    report = dict(
        status="complete", source_digest=source_digest, buckets=[{"bucket": i} for i in range(32)]
    )
    if damage == "source":
        report["source_digest"] = "changed"
    elif damage == "status":
        report["status"] = "partial"
    elif damage == "buckets":
        report["buckets"].pop()
    (tmp_path / "source.json").write_text(json.dumps(original))
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    if damage == "bytes":
        p.write_text(json.dumps(dict(report, extra="changed")))

    def checksum(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    if controller == "audit_adjacent_calibration.py":
        run = body(
            controller, "inputs", CALIBRATION=tmp_path, REPORT_SHA=expected, checksum=checksum
        )
        args = ()
    else:
        run = body(controller, "inputs", TEST_COHORT=tmp_path, checksum=checksum)
        args = (expected,)
    if damage == "none":
        actual, config = run(*args)
        assert actual == report and config == {"fixture": "configuration"}
    else:
        with pytest.raises(ValueError):
            run(*args)


@pytest.mark.parametrize(
    "controller",
    [
        "audit_confirmatory_cases.py",
        "audit_adjacent_calibration.py",
        "audit_multistep_test.py",
        "audit_multistep_fit.py",
    ],
)
@pytest.mark.parametrize("bucket", [-1, 32])
def test_bucket_limits_reject_before_source_load(controller, bucket):
    args = (bucket, "a" * 64) if "multistep" in controller else (bucket,)
    with pytest.raises(ValueError):
        body(controller, "audit_bucket")(*args)
