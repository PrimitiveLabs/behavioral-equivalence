import gzip
import hashlib
import json
import numpy as np
import pytest
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.nominal_chunk import evaluate_nominal_chunk
from behavioral_equivalence.nominal_replay import replay_nominal_chunk


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    calibration = tmp_path / "calibration"
    ablation = tmp_path / "ablation"
    calibration.mkdir()
    ablation.mkdir()
    protocol = dict(
        source_digest="a" * 64,
        protocol_digest="b" * 64,
        families=["preference"],
        sizes=[32],
        seeds=[7],
        regimes=["uniform_active"],
        repetitions=1,
        perturbation=0.01,
        horizons=[1],
        bits=24,
    )
    world = generate("preference", 32, 7, 0.01)
    rng = np.random.default_rng(7)
    counts = np.array(
        [
            rng.multinomial(10 if label == "active" else 0, row)
            for label, row in zip(world.chain.labels, world.chain.transitions, strict=True)
        ]
    )
    packed = gzip.compress(json.dumps(dict(counts=counts.tolist())).encode())
    name = "repetition-000.json.gz"
    (calibration / name).write_bytes(packed)
    spec = dict(
        family="preference",
        size=32,
        seed=7,
        regime="uniform_active",
        first_repetition=0,
        last_repetition=1,
    )
    report = dict(
        status="complete",
        spec=spec,
        chunk_digest=sha(json.dumps(spec, sort_keys=True).encode()),
        source_digest=protocol["source_digest"],
        protocol_digest=protocol["protocol_digest"],
        world_digest=world.chain.digest(),
        artifact_sha256={name: sha(packed)},
        row_sample_schedule=counts.sum(axis=1).tolist(),
    )
    raw = json.dumps(report).encode()
    (calibration / "report.json").write_bytes(raw)
    original_sha = sha(raw)
    evidence = evaluate_nominal_chunk(calibration, protocol, original_sha)
    packed = gzip.compress(json.dumps(evidence, sort_keys=True).encode())
    (ablation / "evidence.json.gz").write_bytes(packed)
    report = {k: v for k, v in evidence.items() if k != "certificates"}
    report["evidence_sha256"] = sha(packed)
    raw = json.dumps(report).encode()
    (ablation / "report.json").write_bytes(raw)
    return calibration, ablation, protocol, original_sha, sha(raw)


def test_replays_both_saved_support_certificates_without_resampling(inputs):
    r = replay_nominal_chunk(*inputs)
    assert r["verified_datasets"] == 1
    assert r["verified_point_certificates"] == 2


def test_arithmetic_tampering_is_rejected_even_with_updated_hashes(inputs):
    calibration, ablation, protocol, original_sha, _ = inputs
    stored = json.loads(gzip.decompress((ablation / "evidence.json.gz").read_bytes()))
    stored["certificates"][0]["point_numeric"]["lower"][0][0] += 0.01
    packed = gzip.compress(json.dumps(stored).encode())
    (ablation / "evidence.json.gz").write_bytes(packed)
    report = json.loads((ablation / "report.json").read_text())
    report["evidence_sha256"] = sha(packed)
    raw = json.dumps(report).encode()
    (ablation / "report.json").write_bytes(raw)
    with pytest.raises(ValueError, match="arithmetic replay"):
        replay_nominal_chunk(calibration, ablation, protocol, original_sha, sha(raw))


def test_report_and_compressed_evidence_hashes_are_checked(inputs):
    calibration, ablation, protocol, original_sha, report_sha = inputs
    with pytest.raises(ValueError, match="report digest"):
        replay_nominal_chunk(calibration, ablation, protocol, original_sha, "0" * 64)
    (ablation / "evidence.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source-bound"):
        replay_nominal_chunk(calibration, ablation, protocol, original_sha, report_sha)
