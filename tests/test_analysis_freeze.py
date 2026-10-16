from copy import deepcopy
import hashlib
import json

import pytest

from behavioral_equivalence.analysis_freeze import REQUIRED_COMPONENTS, verify_analysis_freeze


def fixture():
    fits = [
        dict(protocol=p, seed=s)
        for p in ["known_users", "unseen_users"]
        for s in range(20261015, 20261020)
    ]
    payload = b"bound input"
    config = json.dumps(dict(split="test", noninferiority_margin_nats=0.01, probes=fits)).encode()
    manifest = dict(
        schema_version="human-final-analysis-freeze-v1",
        status="frozen",
        test_results_inspected=False,
        components=sorted(REQUIRED_COMPONENTS),
        source_digest="a" * 64,
        config_sha256=hashlib.sha256(config).hexdigest(),
        fits=fits,
        artifacts=[
            dict(path=role + "/report.json", role=role, sha256=hashlib.sha256(payload).hexdigest())
            for role in sorted(REQUIRED_COMPONENTS)
        ],
    )
    manifest.pop("source_digest", None)
    manifest.pop("config_sha256", None)
    manifest["controllers"] = {
        phase: dict(source_digest="a" * 64, config_sha256=hashlib.sha256(config).hexdigest())
        for phase in [
            "one_step",
            "direct_horizon",
            "response_all_groups",
            "response_heldout_groups",
        ]
    }
    return manifest, config, payload


def test_complete_input_freeze_and_corrupted_artifact():
    manifest, config, payload = fixture()
    assert (
        len(
            verify_analysis_freeze(
                manifest,
                source_digest="a" * 64,
                config_bytes=config,
                read_artifact=lambda p: payload,
            )
        )
        == 64
    )
    with pytest.raises(ValueError, match="artifact changed"):
        verify_analysis_freeze(
            manifest,
            source_digest="a" * 64,
            config_bytes=config,
            read_artifact=lambda p: b"changed",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(status="candidate"),
        lambda m: m.update(test_results_inspected=True),
        lambda m: m["controllers"]["one_step"].update(source_digest="b" * 64),
        lambda m: m.update(components=[]),
        lambda m: m["fits"].pop(),
        lambda m: m["artifacts"].pop(),
        lambda m: m["artifacts"].append(deepcopy(m["artifacts"][0])),
        lambda m: m["artifacts"][0].update(path="../secret"),
    ],
)
def test_incomplete_or_changed_freeze_rejected(mutation):
    manifest, config, payload = fixture()
    mutation(manifest)
    with pytest.raises(ValueError):
        verify_analysis_freeze(
            manifest, source_digest="a" * 64, config_bytes=config, read_artifact=lambda p: payload
        )


@pytest.mark.parametrize(
    "config_change",
    [
        dict(split="validation"),
        dict(noninferiority_margin_nats=0.02),
        dict(probes=[]),
    ],
)
def test_wrong_analysis_config_rejected_even_when_its_hash_matches(config_change):
    manifest, config, payload = fixture()
    value = json.loads(config)
    value.update(config_change)
    config = json.dumps(value).encode()
    manifest["controllers"]["one_step"]["config_sha256"] = hashlib.sha256(config).hexdigest()
    with pytest.raises(ValueError):
        verify_analysis_freeze(
            manifest, source_digest="a" * 64, config_bytes=config, read_artifact=lambda p: payload
        )


@pytest.mark.parametrize(
    "phase", ["direct_horizon", "response_all_groups", "response_heldout_groups"]
)
def test_joint_freeze_supports_each_controller_with_its_own_fit_matrix(phase):
    manifest, config, payload = fixture()
    if phase.startswith("response_"):
        cfg = json.loads(config)
        cfg["probes"] = [p for p in cfg["probes"] if p["protocol"] == "known_users"]
        config = json.dumps(cfg).encode()
        manifest["controllers"][phase]["config_sha256"] = hashlib.sha256(config).hexdigest()
    assert (
        len(
            verify_analysis_freeze(
                manifest,
                source_digest="a" * 64,
                config_bytes=config,
                read_artifact=lambda p: payload,
                phase=phase,
            )
        )
        == 64
    )


def test_joint_freeze_rejects_missing_or_malformed_controller_bindings():
    manifest, config, payload = fixture()
    manifest["controllers"].pop("direct_horizon")
    with pytest.raises(ValueError, match="four"):
        verify_analysis_freeze(
            manifest, source_digest="a" * 64, config_bytes=config, read_artifact=lambda p: payload
        )
    manifest, config, payload = fixture()
    manifest["controllers"]["direct_horizon"]["config_sha256"] = "bad"
    with pytest.raises(ValueError, match="hashes"):
        verify_analysis_freeze(
            manifest, source_digest="a" * 64, config_bytes=config, read_artifact=lambda p: payload
        )
