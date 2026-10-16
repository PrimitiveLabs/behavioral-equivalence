import gzip
import hashlib
import json
import numpy as np
import pytest
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.nominal_chunk import evaluate_nominal_chunk


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def test_paired_count_reuse_and_tamper_rejection(tmp_path):
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
            for label, row in zip(world.chain.labels, world.chain.transitions)
        ]
    )
    packed = gzip.compress(json.dumps({"counts": counts.tolist()}).encode())
    name = "repetition-000.json.gz"
    (tmp_path / name).write_bytes(packed)
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
        chunk_digest=digest(json.dumps(spec, sort_keys=True).encode()),
        source_digest=protocol["source_digest"],
        protocol_digest=protocol["protocol_digest"],
        world_digest=world.chain.digest(),
        artifact_sha256={name: digest(packed)},
        row_sample_schedule=counts.sum(axis=1).tolist(),
    )

    def save():
        raw = json.dumps(report).encode()
        (tmp_path / "report.json").write_bytes(raw)
        return digest(raw)

    original = save()
    result = evaluate_nominal_chunk(tmp_path, protocol, original)
    assert len(result["records"]) == len(result["certificates"]) == 2
    assert result["statistical_coverage_claim"] == "none"
    assert {x["support"] for x in result["records"]} == {
        "unknown_terminal_rows",
        "known_closed_terminal_labels",
    }
    assert all(x["completion_widths"][0]["horizon"] == 1 for x in result["records"])
    with pytest.raises(ValueError, match="report digest"):
        evaluate_nominal_chunk(tmp_path, protocol, "0" * 64)
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="count file digest"):
        evaluate_nominal_chunk(tmp_path, protocol, original)
    (tmp_path / name).write_bytes(packed)
    report["row_sample_schedule"][0] += 1
    with pytest.raises(ValueError, match="totals"):
        evaluate_nominal_chunk(tmp_path, protocol, save())
    report["row_sample_schedule"][0] -= 1
    report["world_digest"] = "0" * 64
    with pytest.raises(ValueError, match="world"):
        evaluate_nominal_chunk(tmp_path, protocol, save())
    report["world_digest"] = world.chain.digest()
    report["artifact_sha256"] = {}
    with pytest.raises(ValueError, match="Missing"):
        evaluate_nominal_chunk(tmp_path, protocol, save())
    report["status"] = "failed"
    with pytest.raises(ValueError, match="source-bound"):
        evaluate_nominal_chunk(tmp_path, protocol, save())
