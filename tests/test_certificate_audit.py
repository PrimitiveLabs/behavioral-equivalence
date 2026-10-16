import gzip
import json
import numpy as np
import pytest
from behavioral_equivalence.certificate_audit import audit_world
from behavioral_equivalence.core import Chain, quotient
from behavioral_equivalence.integer_bounds import verify_abstraction
from behavioral_equivalence.prism import write_model


def test_audit_binds_certificate_to_saved_world_and_actual_prism_input(tmp_path):
    source, out = tmp_path / "world", tmp_path / "audit"
    source.mkdir()
    concrete = Chain(
        np.array([[0.5, 0.25, 0.25], [0.25, 0.25, 0.5], [0, 0, 1.0]]),
        ("active", "active", "complete"),
        np.array([1.0, 0, 0]),
    )
    blocks = ((0, 1), (2,))
    abstract, _ = quotient(concrete, blocks)
    report = {
        "status": "complete",
        "world_digest": concrete.digest(),
        "protocol_digest": "frozen-protocol",
        "source_digest": "frozen-source",
        "spec": {"size": 3},
        "methods": [
            {"method": "test", "parameters": {"epsilon_budget": 0.125}, "partition": blocks}
        ],
    }
    (source / "report.json").write_text(json.dumps(report))
    np.savez_compressed(
        source / "world.npz",
        p=concrete.transitions,
        labels=concrete.labels,
        initial=concrete.initial,
    )
    write_model(abstract, source / "model-0.prism")
    result = audit_world(source, out, [1, 5])
    assert result["verified_certificates"] == 1
    assert result["methods"][0]["budget_strictly_satisfied"]
    cert = json.loads(gzip.decompress((out / "certificate-0.json.gz").read_bytes()))
    verify_abstraction(concrete, abstract, cert)
    (source / "model-0.prism").write_text("different actual solver input")
    with pytest.raises(ValueError, match="PRISM input"):
        audit_world(source, out, [1, 5])
    report["world_digest"] = "modified"
    (source / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="recorded digest"):
        audit_world(source, out, [1, 5])
