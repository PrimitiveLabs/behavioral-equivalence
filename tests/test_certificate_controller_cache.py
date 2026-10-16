"""Certificate and exploratory-world cache identities without provider execution."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_cloud_summary_boundaries import body


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "damage", ["none", "source_report", "certificate", "source_digest", "protocol_digest"]
)
def test_cached_integer_audit_is_bound_to_source_and_certificate_bytes(tmp_path, damage):
    key = "a" * 64
    source = tmp_path / "worlds" / key
    out = tmp_path / "audit/worlds" / key
    source.mkdir(parents=True)
    out.mkdir(parents=True)
    protocol = dict(source_digest="source", protocol_digest="protocol", horizons=[5])
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    original = dict(source_digest="source", protocol_digest="protocol")
    if damage in ["source_digest", "protocol_digest"]:
        original[damage] = "changed"
    (source / "report.json").write_text(json.dumps(original))
    (out / "cert.json").write_text("retained certificate")
    audit = dict(
        source_report_sha256=checksum(source / "report.json"),
        verified_certificates=1,
        methods=[
            dict(
                certificate="cert.json",
                certificate_sha256=checksum(out / "cert.json"),
                budget_strictly_satisfied=False,
                method="exact_rational",
                epsilon_is_exactly_zero=False,
            )
        ],
    )
    (out / "report.json").write_text(json.dumps(audit))
    if damage == "source_report":
        (source / "report.json").write_text(json.dumps(dict(original, extra="changed")))
    if damage == "certificate":
        (out / "cert.json").write_text("changed")
    calls = []
    run = body(
        "certify_campaign_parallel.py",
        "audit_one",
        volume=SimpleNamespace(
            reload=lambda: calls.append("reload"), commit=lambda: calls.append("commit")
        ),
        checked_root=lambda campaign: tmp_path,
        checked_audit_root=lambda root: tmp_path / "audit",
        sha256=checksum,
    )
    if damage == "none":
        r = run("fixture", key)
        assert r["verified_certificates"] == 1
        assert r["strict_requested_epsilon_exceedances"] == 1
        assert r["serialized_exact_quotients_with_nonzero_radius"] == 1
        assert r["audit_report_sha256"] == checksum(out / "report.json")
    else:
        with pytest.raises(ValueError):
            run("fixture", key)
    assert calls == ["reload"]


@pytest.mark.parametrize(
    "controller,function,args",
    [
        ("certify_campaign.py", "audit_ready", ("../escape", 1)),
        ("certify_campaign.py", "audit_ready", ("fixture", 0)),
        ("certify_campaign.py", "audit_ready", ("fixture", 26)),
        ("certify_campaign_parallel.py", "checked_root", ("../escape",)),
        ("certify_campaign_parallel.py", "audit_one", ("fixture", "A" * 64)),
    ],
)
def test_certificate_controller_identity_and_wave_limits(controller, function, args):
    with pytest.raises(ValueError):
        body(controller, function)(*args)


@pytest.mark.parametrize("damage", ["none", "spec", "source", "artifact"])
def test_diversity_cache_checks_spec_source_and_artifact_hash(tmp_path, damage):
    spec = dict(family="trust", size=32, seed=7, surface="linear")
    digest = "a" * 64
    out = tmp_path / ("environment-diversity-" + digest[:16]) / "trust-32-7-linear"
    out.mkdir(parents=True)
    (out / "report.json.gz").write_bytes(b"original")
    saved = dict(
        source_digest=digest,
        spec=spec,
        output_sha256={"report.json.gz": checksum(out / "report.json.gz")},
    )
    if damage == "spec":
        saved["spec"] = dict(spec, seed=8)
    elif damage == "source":
        saved["source_digest"] = "other"
    elif damage == "artifact":
        (out / "report.json.gz").write_bytes(b"changed")
    (out / "completion.json").write_text(json.dumps(saved))
    calls = []
    run = body(
        "environment_diversity_campaign.py",
        "world",
        Path=lambda value: tmp_path if str(value) == "/artifacts" else Path(value),
        snapshot=lambda: ({}, digest),
        checksum=checksum,
        volume=SimpleNamespace(
            reload=lambda: calls.append("reload"), commit=lambda: calls.append("commit")
        ),
    )
    if damage == "none":
        assert run(spec) == saved
    else:
        with pytest.raises(ValueError):
            run(spec)
    assert calls == ["reload"]
