"""Source/protocol bindings for retained summary replay, without provider imports."""

import hashlib
import json

import pytest

from test_cloud_summary_boundaries import body


@pytest.mark.parametrize(
    "controller,function",
    [("summarize_repeated.py", "load_protocol"), ("summarize_interventions.py", "load")],
)
@pytest.mark.parametrize(
    "damage", ["none", "protocol_digest", "campaign_id", "source_digest", "source_code"]
)
def test_summary_loaders_require_immutable_protocol_and_matching_code(
    tmp_path, controller, function, damage
):
    campaign = "fixture-campaign"
    root = tmp_path / "artifacts" / campaign
    code = tmp_path / "research/src/behavioral_equivalence"
    root.mkdir(parents=True)
    code.mkdir(parents=True)
    original = {"fixture.py": "value = 1\n"}
    protocol = dict(
        campaign_id=campaign,
        source_digest=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
    )
    semantic = {k: v for k, v in protocol.items() if k not in {"campaign_id", "protocol_digest"}}
    protocol["protocol_digest"] = hashlib.sha256(
        json.dumps(semantic, sort_keys=True).encode()
    ).hexdigest()
    (code / "fixture.py").write_text(original["fixture.py"])
    if damage == "protocol_digest":
        protocol["protocol_digest"] = "changed"
    elif damage == "campaign_id":
        protocol["campaign_id"] = "other"
    elif damage == "source_digest":
        original["fixture.py"] = "changed"
    elif damage == "source_code":
        (code / "fixture.py").write_text("different\n")
    (root / "source.json").write_text(json.dumps(original))
    (root / "protocol.json").write_text(json.dumps(protocol))

    def local_path(value):
        return tmp_path / str(value).lstrip("/")

    run = body(controller, function, Path=local_path)
    if damage == "none":
        path, actual = run(campaign)
        assert path == root and actual == protocol
    else:
        with pytest.raises(ValueError):
            run(campaign)


@pytest.mark.parametrize(
    "controller,function,key",
    [
        ("summarize_repeated.py", "audit_one", "chunk"),
        ("summarize_interventions.py", "audit_world", "identity"),
    ],
)
@pytest.mark.parametrize("identity", ["", "a" * 63, "A" * 64, "../escape"])
def test_world_and_chunk_audits_reject_invalid_keys_before_provider_io(
    controller, function, key, identity
):
    run = body(controller, function)
    with pytest.raises(ValueError):
        run("fixture", identity)
