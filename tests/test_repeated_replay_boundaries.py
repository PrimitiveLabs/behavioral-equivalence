"""Retained synthetic replay integrity and paired summary boundaries."""

from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from behavioral_equivalence.nominal_ablation import empirical_point_intervals
from behavioral_equivalence.repeated_analysis import summarize_repetitions
from behavioral_equivalence.repeated_evidence import audit_repetition_chunk
from test_repeated_analysis import fixture
from test_repeated_evidence import retained, write_artifacts  # noqa: F401


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "failures",
        "protocol",
        "chunk_protocol",
        "outside_repetition",
        "outside_method",
        "duplicate_property",
        "missing_width",
        "out_of_range_width",
        "method_count",
    ],
)
def test_repetition_summary_rejects_design_mismatch(damage):
    manifest, protocol = fixture()
    chunk = manifest["chunks"][0]
    row = chunk["records"][0]
    if damage == "status":
        manifest["status"] = "partial"
    elif damage == "failures":
        manifest["failures"] = ["failed"]
    elif damage == "protocol":
        manifest["protocol_digest"] = "other"
    elif damage == "chunk_protocol":
        chunk["protocol_digest"] = "other"
    elif damage == "outside_repetition":
        row["repetition"] = 2
    elif damage == "outside_method":
        row["method"] = "other"
    elif damage == "duplicate_property":
        row["properties"][-1] = row["properties"][0].copy()
    elif damage == "missing_width":
        row["completion_widths"].pop()
    elif damage == "out_of_range_width":
        row["completion_widths"][0]["mean_active_width"] = 1.1
    else:
        manifest["method_dataset_evaluations"] += 1
    with pytest.raises(ValueError):
        summarize_repetitions(manifest, protocol)


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "source",
        "protocol",
        "identity",
        "family",
        "world",
        "artifact_names",
        "record_duplicate",
        "certificate_duplicate",
        "confidence",
    ],
)
def test_replay_rejects_bound_identity_and_certificate_damage(retained, damage):  # noqa: F811
    path, protocol, report, evidence = retained
    if damage == "status":
        report["status"] = "partial"
    elif damage == "source":
        report["source_digest"] = "other"
    elif damage == "protocol":
        report["protocol_digest"] = "other"
    elif damage == "identity":
        report["chunk_digest"] = "other"
    elif damage == "family":
        report["spec"]["family"] = "other"
        report["chunk_digest"] = hashlib.sha256(
            json.dumps(report["spec"], sort_keys=True).encode()
        ).hexdigest()
    elif damage == "world":
        report["world_digest"] = "other"
    elif damage == "artifact_names":
        report["artifact_sha256"] = {}
        (path / "report.json").write_text(json.dumps(report))
        with pytest.raises(ValueError, match="artifacts"):
            audit_repetition_chunk(path, protocol)
        return
    elif damage == "record_duplicate":
        report["records"][-1] = report["records"][0].copy()
    elif damage == "certificate_duplicate":
        evidence["methods"][-1] = evidence["methods"][0].copy()
    else:
        evidence["methods"][0]["confidence"]["changed"] = True
    write_artifacts(path, report, evidence)
    with pytest.raises(ValueError):
        audit_repetition_chunk(path, protocol)


@pytest.mark.parametrize("closure", [False, True])
def test_empirical_numeric_envelope_retains_wide_integer_counts(closure):
    counts = np.array([[2**63 + 1, 2**63 + 3], [0, 0]], dtype=np.uint64)
    before = counts.copy()
    model, cert = empirical_point_intervals(counts, ("active", "complete"), closure)
    total = 2**64 + 4
    for j in range(2):
        exact = Fraction(int(counts[0, j]), total)
        assert Fraction(float(model.lower[0, j])) <= exact <= Fraction(float(model.upper[0, j]))
    assert cert["statistical_coverage_claim"] == "none"
    assert cert["zero_sample_rows"] == [1]
    np.testing.assert_array_equal(counts, before)
