"""Frozen campaign protocol and cached numerical evidence guards."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_cloud_summary_boundaries import body


@pytest.mark.parametrize(
    "controller,prefix,snapshot",
    [
        ("intervention_campaign.py", "intervention-main-v1-", "snapshot"),
        ("uncertainty_repeated_campaign.py", "uncertainty-repeated-v1-", "source_snapshot"),
    ],
)
@pytest.mark.parametrize("damage", ["none", "source", "digest", "campaign"])
def test_campaign_protocol_requires_bound_source_and_semantics(
    controller, prefix, snapshot, damage
):
    protocol = {"source_digest": "source", "design": "fixture"}
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol.update(protocol_digest=digest, campaign_id=prefix + digest[:12])
    if damage == "source":
        protocol["source_digest"] = "other"
    elif damage == "digest":
        protocol["protocol_digest"] = "other"
    elif damage == "campaign":
        protocol["campaign_id"] = "other"
    run = body(controller, "validate", **{snapshot: lambda: ({}, "source")})
    if damage == "none":
        run(protocol)
    else:
        with pytest.raises(ValueError):
            run(protocol)


@pytest.mark.parametrize(
    "controller,function,file,subfolder,field",
    [
        ("intervention_campaign.py", "world", "summary.json", "worlds", "artifact_sha256"),
        ("uncertainty_repeated_campaign.py", "worker", "report.json", "chunks", "artifact_sha256"),
    ],
)
@pytest.mark.parametrize("damage", ["none", "source", "spec", "artifact"])
def test_stochastic_campaign_cache_requires_exact_evidence(
    tmp_path, controller, function, file, subfolder, field, damage
):
    spec = {"family": "trust", "size": 32, "seed": 7}
    protocol = {"campaign_id": "fixture", "source_digest": "source", "protocol_digest": "protocol"}
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    out = tmp_path / "fixture" / subfolder / identity
    out.mkdir(parents=True)
    artifact = out / "evidence.gz"
    artifact.write_bytes(b"original")
    saved = dict(
        source_digest="source",
        protocol_digest="protocol",
        spec=spec,
        **{field: {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}},
    )
    if damage == "source":
        saved["source_digest"] = "other"
    elif damage == "spec":
        saved["spec"] = {"family": "other"}
    elif damage == "artifact":
        artifact.write_bytes(b"changed")
    (out / file).write_text(json.dumps(saved))
    calls = []
    run = body(
        controller,
        function,
        validate=lambda protocol: None,
        Path=lambda value: tmp_path if str(value) == "/artifacts" else Path(value),
        volume=SimpleNamespace(
            reload=lambda: calls.append("reload"), commit=lambda: calls.append("commit")
        ),
    )
    if damage == "none":
        assert run(spec, protocol) == saved
    else:
        with pytest.raises(ValueError):
            run(spec, protocol)
    assert calls == ["reload"]
