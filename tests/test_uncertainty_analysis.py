import gzip
import hashlib
import json
import pytest
from behavioral_equivalence.uncertainty_analysis import summarize_world


def test_world_summary_weights_active_states_and_rejects_incomplete_or_changed_evidence(tmp_path):
    protocol = {"protocol_digest": "frozen", "samples_per_row": [10], "horizons": [1]}
    table = {"bits": 2, "values": {"1": {"min_lower": [1, 4], "max_upper": [3, 4]}}}
    evidence = {
        "evidence": {
            "uniform_abstraction": {"partition": [[0, 1], [2]], "labels": ["active", "complete"]},
            "robust_values": {target: table for target in ["complete", "disengage", "terminal"]},
        }
    }
    packed = gzip.compress(json.dumps(evidence).encode())
    (tmp_path / "cert.gz").write_bytes(packed)
    records = []
    for method in ["certified_hoeffding", "clopper_pearson_float_quantiles"]:
        records.append(
            {
                "method": method,
                "samples_per_row": 10,
                "certificate": "cert.gz",
                "certificate_sha256": hashlib.sha256(packed).hexdigest(),
                "kernel_membership": True,
                "properties": [
                    {
                        "property": target,
                        "horizon": 1,
                        "states": 3,
                        "enclosure_containment_count": 3,
                        "proven_noncoverage_count": 0,
                        "unresolved_count": 0,
                    }
                    for target in ["complete", "disengage", "survival"]
                ],
            }
        )
    report = {
        "status": "complete",
        "protocol_digest": "frozen",
        "spec": {"family": "test", "size": 3, "seed": 1},
        "records": records,
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    result = summarize_world(tmp_path, protocol)
    assert len(result["rows"]) == 6
    assert all(row["width"] == 0.5 for row in result["rows"])
    assert all(row["all_state_property_enclosures_contained"] for row in result["coverage_records"])
    (tmp_path / "cert.gz").write_bytes(packed + b"changed")
    with pytest.raises(ValueError, match="Certificate artifact changed"):
        summarize_world(tmp_path, protocol)
    (tmp_path / "cert.gz").write_bytes(packed)
    report["records"][0]["kernel_membership"] = False
    report["records"][0]["properties"][0]["enclosure_containment_count"] = 2
    report["records"][0]["properties"][0]["unresolved_count"] = 1
    (tmp_path / "report.json").write_text(json.dumps(report))
    failed = summarize_world(tmp_path, protocol)["coverage_records"][0]
    assert failed["kernel_included"] is False
    assert failed["all_state_property_enclosures_contained"] is False
    assert failed["unresolved_count"] == 1
    report["records"][0]["properties"].pop()
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="property comparisons"):
        summarize_world(tmp_path, protocol)
